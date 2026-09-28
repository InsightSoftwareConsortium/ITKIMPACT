/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkImpactFineRegistration_hxx
#define itkImpactFineRegistration_hxx

// LibTorch-backed implementation. This file is included by the public header only when
// ITK_MANUAL_INSTANTIATION is undefined, so castxml (which defines it) never sees torch.

#include "itkImpactFineRegistration.h"
#include "itkImpactTorchRegistrationHelpers.h"
#include "itkImpactOnlineInference.h" // PatchTensorShape, PatchPlaneRotation
#include "ImpactLoss.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkIdentityTransform.h>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace itk
{
namespace Impact
{

/** Where model `config`'s patch around each of `points` ({N, Dim} fixed-grid voxel indices, ITK order) samples the
 * fixed grid, in its voxels (ITK order), {N, P, Dim} double: patch voxel k -- model axis 0 running fastest, as
 * PatchTensorShape lays the patch out -- at (k_d - P_d / 2) * step_d along patch axis d, so the voxel the centre
 * readback reads (size / 2) is the point itself. A model of the image's dimension spans the image axes, step_d its
 * voxel size along axis d (the image's spacing when unset). A 2D model in a volume spans the plane drawn for each point
 * (PatchPlaneRotation's first two columns, from `seed` and `modelIndex`, as the metric's PatchPlane), its steps its
 * voxel sizes (the finest spacing when unset). `voxels`, when given, keeps those patch voxels only (P of them). */
template <unsigned int Dim>
torch::Tensor
SampledPatchPositions(const ImpactModelConfiguration & config,
                      const torch::Tensor &            points,
                      const std::vector<double> &      spacing,
                      unsigned int                     seed,
                      size_t                           modelIndex,
                      const std::vector<int64_t> &     voxels = {})
{
  const std::vector<int64_t> & patchSize = config.GetPatchSize();
  const std::vector<float> &   voxelSize = config.GetVoxelSize();
  const unsigned int           modelDimension = config.GetDimension();
  const double                 finest = *std::min_element(spacing.begin(), spacing.end());
  std::vector<int64_t>         kept = voxels;
  if (kept.empty())
  {
    int64_t patchVoxels = 1;
    for (unsigned int d = 0; d < modelDimension; ++d)
    {
      patchVoxels *= patchSize[d];
    }
    kept.resize(static_cast<size_t>(patchVoxels));
    std::iota(kept.begin(), kept.end(), int64_t{ 0 });
  }
  // Each patch voxel's offset along each patch axis, in millimetres.
  torch::Tensor offsets =
    torch::empty({ static_cast<int64_t>(kept.size()), static_cast<int64_t>(modelDimension) }, torch::kFloat64);
  auto offset = offsets.accessor<double, 2>();
  for (int64_t k = 0; k < offsets.size(0); ++k)
  {
    int64_t rem = kept[k];
    for (unsigned int d = 0; d < modelDimension; ++d)
    {
      const int64_t index = rem % patchSize[d];
      rem /= patchSize[d];
      const double step =
        d < voxelSize.size() && voxelSize[d] > 0.0f ? voxelSize[d] : (modelDimension == Dim ? spacing[d] : finest);
      offset[k][d] = static_cast<double>(index - patchSize[d] / 2) * step;
    }
  }
  const torch::Tensor centres = points.to(torch::kFloat64).unsqueeze(1); // {N, 1, Dim}
  const torch::Tensor perVoxel = 1.0 / torch::tensor(spacing, torch::kFloat64);
  if (modelDimension == Dim)
  {
    return centres + (offsets * perVoxel).unsqueeze(0);
  }
  if (modelDimension != 2 || Dim != 3)
  {
    itkGenericExceptionMacro("IMPACT: a " << modelDimension << "D model's patch cannot be cut in a " << Dim
                                          << "D image; only a 2D model in a volume is swept.");
  }
  // The two patch axes of each point's plane, in the image's frame: {N, Dim, 2}.
  const int64_t count = points.size(0);
  torch::Tensor axes = torch::empty({ count, static_cast<int64_t>(Dim), 2 }, torch::kFloat64);
  auto          axis = axes.accessor<double, 3>();
  const auto    point = points.accessor<int64_t, 2>();
  for (int64_t n = 0; n < count; ++n)
  {
    const std::array<int64_t, 3> index{ point[n][0], point[n][1], point[n][Dim - 1] };
    const Matrix<double, 3, 3>   plane = PatchPlaneRotation(seed, modelIndex, index);
    for (unsigned int j = 0; j < Dim; ++j)
    {
      axis[n][j][0] = plane[j][0];
      axis[n][j][1] = plane[j][1];
    }
  }
  return centres + torch::einsum("njd,pd->npj", { axes, offsets }) * perVoxel;
}

/** The corners of model `config`'s patch, as patch voxel numbers (model axis 0 running fastest): a patch is the image
 * of a box, so its extreme positions are at these. */
inline std::vector<int64_t>
PatchCorners(const ImpactModelConfiguration & config)
{
  std::vector<int64_t> corners{ 0 };
  int64_t              stride = 1;
  for (unsigned int d = 0; d < config.GetDimension(); ++d)
  {
    const int64_t last = (config.GetPatchSize()[d] - 1) * stride;
    const size_t  count = corners.size();
    for (size_t c = 0; c < count; ++c)
    {
      corners.push_back(corners[c] + last);
    }
    stride *= config.GetPatchSize()[d];
  }
  return corners;
}

} // namespace Impact

template <typename TFixedImage, typename TMovingImage>
ImpactFineRegistration<TFixedImage, TMovingImage>::ImpactFineRegistration() = default;

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::SetFixedImage(const FixedImageType * image)
{
  if (m_FixedImage.GetPointer() != image)
  {
    m_FixedImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::SetMovingImage(const MovingImageType * image)
{
  if (m_MovingImage.GetPointer() != image)
  {
    m_MovingImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactFineRegistration<TFixedImage, TMovingImage>::GetDisplacementField() -> DisplacementFieldType *
{
  return this->GetOutput();
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactFineRegistration<TFixedImage, TMovingImage>::GetDisplacementFieldTransform()
  -> DisplacementFieldTransformType *
{
  return m_DisplacementFieldTransform.GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactFineRegistration<TFixedImage, TMovingImage>::GetWarpedMovingImage() -> WarpedImageType *
{
  return m_WarpedMovingImage.GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  if (m_FixedImage.IsNull())
  {
    return;
  }
  DisplacementFieldType * output = this->GetOutput();
  output->SetLargestPossibleRegion(m_FixedImage->GetLargestPossibleRegion());
  output->SetSpacing(m_FixedImage->GetSpacing());
  output->SetOrigin(m_FixedImage->GetOrigin());
  output->SetDirection(m_FixedImage->GetDirection());
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::GenerateData()
{
  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }
  // Feature (IMPACT) mode if any model is configured; otherwise raw-intensity MSE.
  const bool featureMode = !m_FixedModelsConfiguration.empty();

  const torch::Device device(m_Device);
  torch::manual_seed(m_Seed);

  // Release the Python GIL for the whole torch section: the Adam loop runs loss.backward(),
  // and libtorch's Python autograd engine (active whenever we run under Python) requires the
  // GIL released. No-op in a pure C++ process. None of the code below re-enters Python.
  const Impact::PythonGilReleaseGuard gilRelease;

  // ---- 1. Fixed and moving onto the same (fixed) voxel grid, as {1,1, z,y,x} tensors;
  // resample only if geometries differ. ----
  typename MovingImageType::ConstPointer movingOnFixed = m_MovingImage;
  {
    const auto & fSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
    const auto & mSize = m_MovingImage->GetLargestPossibleRegion().GetSize();
    bool         sameGrid = (fSize == mSize) && (m_FixedImage->GetSpacing() == m_MovingImage->GetSpacing()) &&
                    (m_FixedImage->GetOrigin() == m_MovingImage->GetOrigin()) &&
                    (m_FixedImage->GetDirection() == m_MovingImage->GetDirection());
    if (!sameGrid)
    {
      using ResampleType = ResampleImageFilter<MovingImageType, MovingImageType, double>;
      using IdentityType = IdentityTransform<double, ImageDimension>;
      using InterpType = LinearInterpolateImageFunction<MovingImageType, double>;
      auto resample = ResampleType::New();
      resample->SetInput(m_MovingImage);
      resample->SetTransform(IdentityType::New());
      resample->SetInterpolator(InterpType::New());
      resample->SetUseReferenceImage(true);
      resample->SetReferenceImage(m_FixedImage);
      resample->Update();
      movingOnFixed = resample->GetOutput();
    }
  }

  torch::Tensor fixedT = Impact::ImageToBatchTensor(m_FixedImage.GetPointer()).to(device);
  torch::Tensor movingT = Impact::ImageToBatchTensor(movingOnFixed.GetPointer()).to(device);
  // The masks on the fixed grid, bool; undefined when absent, and then nothing below costs anything.
  const torch::Tensor fixedMask = Impact::MaskOnGrid(m_FixedMask.GetPointer(), m_FixedImage.GetPointer(), device);
  const torch::Tensor movingMask = Impact::MaskOnGrid(m_MovingMask.GetPointer(), m_FixedImage.GetPointer(), device);
  const bool          masked = fixedMask.defined() || movingMask.defined();

  // Spatial sizes in torch (z, y, x) order.
  const auto &         fixedSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
  std::vector<int64_t> spatial(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    spatial[d] = static_cast<int64_t>(fixedSize[ImageDimension - 1 - d]);
  }

  // ---- 2. Optimizable control grid {1, N, coarse z,y,x}; component order (z,y,x), in units of the
  // fixed image's finest voxel side s_min, so that an Adam step of LearningRate moves LearningRate * s_min
  // mm along every axis. Along axis a the grid is shrink_a = max(1, round(GridShrinkFactor * s_min / s_a))
  // times coarser than the image, (nearly) isotropic in millimetres, and is upsampled to full resolution
  // each iteration (ConvexAdam-style). An isotropic image keeps GridShrinkFactor on every axis and its
  // voxels as the unit. Warm-started from an initial field (e.g. the coarse stage) if provided, else zero. ----
  const std::vector<double>  voxelSide = Impact::TensorVoxelSides<ImageDimension>(m_FixedImage->GetSpacing());
  const double               finest = *std::min_element(voxelSide.begin(), voxelSide.end());
  const std::vector<int64_t> shrink =
    Impact::IsotropicVoxelCounts(voxelSide, static_cast<double>(std::max(1u, m_GridShrinkFactor)));
  std::vector<int64_t> coarseSpatial(ImageDimension);
  std::vector<float>   voxelsPerUnit(ImageDimension); // s_min / s_a: a field unit in voxels of axis a
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    coarseSpatial[d] = std::max<int64_t>(1, spatial[d] / shrink[d]);
    voxelsPerUnit[d] = static_cast<float>(finest / voxelSide[d]);
  }
  // One factor per displacement component, {1, N, 1, ...}.
  std::vector<int64_t> componentShape(ImageDimension + 2, 1);
  componentShape[1] = static_cast<int64_t>(ImageDimension);
  const torch::Tensor  unitToVoxels = torch::tensor(voxelsPerUnit, torch::kFloat32).view(componentShape).to(device);
  std::vector<int64_t> fieldShape;
  fieldShape.push_back(1);
  fieldShape.push_back(static_cast<int64_t>(ImageDimension));
  for (auto s : coarseSpatial)
  {
    fieldShape.push_back(s);
  }
  torch::Tensor theta;
  if (m_InitialDisplacementField.IsNotNull())
  {
    if (m_InitialDisplacementField->GetLargestPossibleRegion().GetSize() != fixedSize)
    {
      itkExceptionMacro("InitialDisplacementField must be defined on the fixed-image grid.");
    }
    torch::Tensor initField =
      Impact::DisplacementToVoxelField<ImageDimension>(
        m_InitialDisplacementField, m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), device) /
      unitToVoxels; // {1,N,z,y,x}, in units of s_min
    if (coarseSpatial != spatial)
    {
      if constexpr (ImageDimension == 3)
        initField = torch::nn::functional::interpolate(
          initField,
          torch::nn::functional::InterpolateFuncOptions().size(coarseSpatial).mode(torch::kTrilinear).align_corners(true));
      else
        initField = torch::nn::functional::interpolate(
          initField,
          torch::nn::functional::InterpolateFuncOptions().size(coarseSpatial).mode(torch::kBilinear).align_corners(true));
    }
    theta = initField.set_requires_grad(true);
  }
  else
  {
    theta =
      torch::zeros(fieldShape, torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(true));
  }

  // ---- 3. Base identity sampling grid (normalized [-1,1], last-dim order x,y,z), align_corners=true. ----
  torch::Tensor idAffine =
    torch::eye(ImageDimension, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  idAffine = torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(ImageDimension), 1 }, idAffine.options()) }, 1)
               .unsqueeze(0); // {1, N, N+1}
  std::vector<int64_t> gridSize;
  gridSize.push_back(1);
  gridSize.push_back(1);
  for (auto s : spatial)
  {
    gridSize.push_back(s);
  }
  torch::Tensor grid0 =
    torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/true); // {1, z,y,x, N}

  // Per-component normalization in (z, y, x) order: the field units (s_min) per grid_sample unit, (size-1)/2 voxels
  // (exact for align_corners=true) of s_a / s_min units each.
  std::vector<float> scaleValues(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    scaleValues[d] = (spatial[d] > 1) ? static_cast<float>((spatial[d] - 1) / 2.0 * (voxelSide[d] / finest)) : 1.0f;
  }
  torch::Tensor scale =
    torch::from_blob(scaleValues.data(), { static_cast<int64_t>(ImageDimension) }, torch::kFloat32).clone().to(device);

  // Permutation moving the channel dim (1) to last: {0, 2, 3, ..., N+1, 1}.
  std::vector<int64_t> toChannelLast;
  toChannelLast.push_back(0);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    toChannelLast.push_back(2 + static_cast<int64_t>(d));
  }
  toChannelLast.push_back(1);

  namespace F = torch::nn::functional;
  const auto sampleOpts =
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);

  // Optional 3x3x3 average-pool smoothing passes on the control field (identity if disabled).
  auto smoothControl = [&](const torch::Tensor & control) -> torch::Tensor {
    torch::Tensor s = control;
    for (unsigned int k = 0; k < m_ControlGridSmoothingIterations; ++k)
    {
      if constexpr (ImageDimension == 3)
        s = F::avg_pool3d(s, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
      else
        s = F::avg_pool2d(s, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    }
    return s;
  };

  // Resize a displacement field {1, N, ...} to a target spatial size (identity if already equal).
  // Values stay in full-res voxel units regardless of the sampling resolution, so no unit rescaling.
  auto resizeField = [&](const torch::Tensor & field, const std::vector<int64_t> & target) -> torch::Tensor {
    bool needs = false;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      if (field.size(2 + static_cast<int64_t>(d)) != target[d])
      {
        needs = true;
      }
    }
    if (!needs)
    {
      return field;
    }
    if constexpr (ImageDimension == 3)
      return F::interpolate(field, F::InterpolateFuncOptions().size(target).mode(torch::kTrilinear).align_corners(true));
    else
      return F::interpolate(field, F::InterpolateFuncOptions().size(target).mode(torch::kBilinear).align_corners(true));
  };

  // Map the (optionally low-resolution) control grid to a full-resolution displacement field:
  // smoothing passes then trilinear/bilinear upsampling to the full image grid. With
  // GridShrinkFactor==1 and no smoothing this is the identity.
  auto controlGridToFullField = [&](const torch::Tensor & control) -> torch::Tensor {
    return resizeField(smoothControl(control), spatial);
  };

  // Build the normalized grid_sample sampling grid from a control field: upsample to full
  // resolution, channel-last, normalize and reorder (z,y,x) -> (x,y,z).
  auto gridFromControl = [&](const torch::Tensor & control) -> torch::Tensor {
    torch::Tensor dd = controlGridToFullField(control).permute(toChannelLast);
    return grid0 + (dd / scale).flip(-1);
  };

  // Build the normalized grid_sample grid for a feature layer DIRECTLY at that layer's resolution:
  // resize the (smoothed) control field to the layer size, then form the grid on the layer's own
  // identity base grid. Avoids the full-res detour (upsample field, then downsample grid) for
  // downsampled layers. Values stay in full-res voxel units, so the divisor is the full-res `scale`.
  // `layerBaseGrid` / `layerSpatials` are filled once after feature extraction, below.
  std::vector<torch::Tensor>        layerBaseGrid; // per-layer identity base grid [1, layer..., Dim]
  std::vector<std::vector<int64_t>> layerSpatials; // per-layer spatial size (z,y,x)
  auto gridForLayerControl = [&](const torch::Tensor & smoothedControl, size_t l) -> torch::Tensor {
    torch::Tensor dd = resizeField(smoothedControl, layerSpatials[l]).permute(toChannelLast);
    return layerBaseGrid[l] + (dd / scale).flip(-1);
  };

  // Evaluate the per-iteration feature similarity at the Adam resolution (coarseSpatial ==
  // spatial / GridShrinkFactor, the grid the intensity path uses and ConvexAdam's grid_sp_adam):
  // average-pool each feature layer DOWN to that grid when finer, never upsample an already-coarser
  // one. Lets GridShrinkFactor reduce the feature-loss cost generically instead of always comparing
  // at each layer's native resolution.
  auto poolLayerToLoss = [&](const torch::Tensor & layer) -> torch::Tensor {
    std::vector<int64_t> target(ImageDimension);
    bool                 needs = false;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      const int64_t nat = layer.size(2 + static_cast<int64_t>(d));
      target[d] = std::min(nat, coarseSpatial[d]);
      if (target[d] != nat)
      {
        needs = true;
      }
    }
    if (!needs)
    {
      return layer;
    }
    if constexpr (ImageDimension == 3)
      return F::adaptive_avg_pool3d(layer, F::AdaptiveAvgPool3dFuncOptions({ target[0], target[1], target[2] }));
    else
      return F::adaptive_avg_pool2d(layer, F::AdaptiveAvgPool2dFuncOptions({ target[0], target[1] }));
  };

  // "Jacobian" (online) mode with a swept model: pool every axis but the swept one (tensor spatial index `keep`) to
  // the loss resolution and keep that one native, so a chunk of re-extracted moving slices lines up with the same
  // slice range of the fixed features.
  auto poolExcept = [&](const torch::Tensor & layer, int64_t keep) -> torch::Tensor {
    std::vector<int64_t> target(ImageDimension);
    bool                 needs = false;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      const int64_t nat = layer.size(2 + static_cast<int64_t>(d));
      target[d] = static_cast<int64_t>(d) == keep ? nat : std::min(nat, coarseSpatial[d]);
      needs = needs || target[d] != nat;
    }
    if (!needs)
    {
      return layer;
    }
    if constexpr (ImageDimension == 3)
      return F::adaptive_avg_pool3d(layer, F::AdaptiveAvgPool3dFuncOptions({ target[0], target[1], target[2] }));
    else
      return F::adaptive_avg_pool2d(layer, F::AdaptiveAvgPool2dFuncOptions({ target[0], target[1] }));
  };
  const bool jacobianMode = featureMode && (m_Mode == "Jacobian");
  if (m_LNCCKernel < 1 || m_LNCCKernel % 2 == 0)
  {
    // An even window has no centre voxel: the correlation would sit half a voxel off the point it is read at.
    itkExceptionMacro("ImpactFineRegistration: LNCCKernel must be odd, got " << m_LNCCKernel << ".");
  }
  if (!(m_SamplingPercentage > 0.0 && m_SamplingPercentage <= 1.0))
  {
    itkExceptionMacro("ImpactFineRegistration: SamplingPercentage must lie in (0, 1], got " << m_SamplingPercentage
                                                                                            << ".");
  }
  const bool sampled = m_SamplingPercentage < 1.0;
  if (sampled && !featureMode)
  {
    // Refused rather than ignored: the intensity path compares whole images.
    itkExceptionMacro("ImpactFineRegistration: SamplingPercentage samples the feature similarity; set a model, or "
                      "leave it at 1.");
  }
  // Jacobian mode on a share of the voxels: elastix's scheme, each model run on the patch around every drawn point.
  const bool sampledJacobian = jacobianMode && sampled;
  // A "sliced" model has lower dimension than the image (a 2D backbone run slice by slice): its feature map keeps the
  // swept axis 1:1 with the image, so the online loss can be taken in chunks of slices that line up by index
  // (memory-bounded). Each iteration sweeps ONE image axis drawn at random (seeded): a dense engine draws no points,
  // so it cannot give each point a random plane of its own as the metric does; turning the volume by a random rotation
  // spent 22 to 57 % of the network's work on corners brought in from outside the image (measured in FireANTs),
  // while one axis of the image costs one sweep and shows the network the anatomy in every orientation over the
  // iterations. A full-dimension model may downsample every axis, so its features are compared whole and aligned by
  // resampling, like the itkv4 metric.
  const bool jacSliced = jacobianMode && !sampled && !m_FixedModelsConfiguration.empty() &&
                         m_FixedModelsConfiguration[0].GetDimension() < ImageDimension;
  // The moving mask as floats at full resolution, which the modes warping it there need.
  const torch::Tensor movingMaskShare =
    movingMask.defined() && (jacSliced || sampledJacobian) ? movingMask.to(torch::kFloat32) : torch::Tensor();

  // Feature-mode setup: extract the fixed/moving feature layers (constants, not differentiated
  // through), optionally PCA-reduce them (fit on fixed), and build one loss per kept layer.
  // Intensity mode skips this and compares raw voxels.
  std::vector<torch::Tensor>                    fixedLayers;
  std::vector<torch::Tensor>                    movingLayers;
  std::vector<torch::Tensor>                    fixedLayersOnline; // "Jacobian" mode: fixed features, pooled
  std::vector<torch::Tensor>                    pcaBasis;          // per kept layer; undefined entry = no PCA
  std::vector<std::unique_ptr<Impact::Loss>>    losses;
  std::vector<float>                            layerWeights;
  // SubsetFeatures: per kept layer, that many of its channels drawn at random at every iteration (0 = all).
  std::vector<torch::Tensor> subsets; // per layer; an undefined entry keeps every channel
  auto drawSubsets = [&](const std::vector<torch::Tensor> & layers) {
    subsets.assign(layers.size(), torch::Tensor());
    for (size_t l = 0; l < layers.size(); ++l)
    {
      const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
      if (kept > 0 && kept < layers[l].size(1))
      {
        subsets[l] =
          torch::randperm(layers[l].size(1), torch::TensorOptions().dtype(torch::kLong)).narrow(0, 0, kept).to(device);
      }
    }
  };
  auto pick = [&](size_t l, const torch::Tensor & layer) -> torch::Tensor {
    return (l < subsets.size() && subsets[l].defined()) ? layer.index_select(1, subsets[l]) : layer;
  };
  // Layer l's loss between two feature maps on one grid ({1, C, spatial...}): a spatial distance (LNCC) reads the
  // maps whole (its window LNCCKernel voxels along the map's finest axis in millimetres, the same length along the
  // others), the others compare their feature vectors voxel by voxel; over the voxels `mask` keeps, when given.
  auto layerLoss = [&](size_t                l,
                       const torch::Tensor & fixedMap,
                       const torch::Tensor & movingMap,
                       const torch::Tensor & mask = {}) -> torch::Tensor {
    return Impact::MapLoss<ImageDimension>(
      *losses[l],
      fixedMap,
      movingMap,
      mask,
      losses[l]->IsSpatial() ? Impact::IsotropicWindow(voxelSide, spatial, fixedMap.sizes(), m_LNCCKernel)
                             : std::vector<int64_t>{});
  };
  // The masks on a grid of `size` voxels: the fixed one as bool (its share of each voxel >= 0.5), the moving one as its
  // share, to be warped; undefined when absent.
  auto fixedMaskOn = [&](const std::vector<int64_t> & size) {
    return fixedMask.defined() ? Impact::MaskShare<ImageDimension>(fixedMask, size) >= 0.5 : torch::Tensor();
  };
  auto movingShareOn = [&](const std::vector<int64_t> & size) {
    return movingMask.defined() ? Impact::MaskShare<ImageDimension>(movingMask, size) : torch::Tensor();
  };
  // The voxels that count: `fixedOn`, and where the moving mask's share `movingShare` warped by `grid` (grid_sample
  // coordinates, a function returning them, only called with a moving mask) holds.
  auto countedVoxels = [&](const torch::Tensor & fixedOn, const torch::Tensor & movingShare, const auto & grid) {
    if (!movingShare.defined())
    {
      return fixedOn;
    }
    torch::NoGradGuard  noGrad;
    const torch::Tensor warped = F::grid_sample(movingShare, grid().detach(), sampleOpts) >= 0.5;
    return fixedOn.defined() ? fixedOn.logical_and(warped) : warped;
  };
  // SamplingPercentage < 1: layer l's loss on that share of its voxels, drawn anew at every iteration. Only those
  // points are warped: the fixed features are read at the drawn voxels, the moving ones at the same voxels displaced
  // by the residual field, interpolated there as gridForLayerControl's resizeField would (align_corners both).
  // With masks, the points are drawn in the fixed mask (that share of its voxels), and those the moving mask, warped by
  // the total field, does not hold are dropped.
  std::vector<torch::Tensor> fixedLayerMask, movingLayerShare, fixedLayerCandidates; // Static mode, per layer
  auto                       sampledLayerLoss =
    [&](size_t l, const torch::Tensor & smoothedResidual, const torch::Tensor & smoothedTotal) -> torch::Tensor {
    const int64_t voxels = layerBaseGrid[l].numel() / ImageDimension;
    int64_t       count = std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(voxels)));
    torch::Tensor drawn;
    if (fixedMask.defined())
    {
      const int64_t candidates = fixedLayerCandidates[l].numel();
      if (candidates == 0)
      {
        return torch::zeros({}, theta.options());
      }
      count = std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(candidates)));
      drawn = fixedLayerCandidates[l].index_select(
        0, torch::randint(candidates, { count }, torch::TensorOptions().dtype(torch::kLong).device(device)));
    }
    else
    {
      drawn = torch::randint(voxels, { count }, torch::TensorOptions().dtype(torch::kLong).device(device));
    }
    const torch::Tensor  points = layerBaseGrid[l].reshape({ voxels, ImageDimension }).index_select(0, drawn);
    std::vector<int64_t> pointShape(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, n, N}
    pointShape[ImageDimension] = count;
    pointShape[ImageDimension + 1] = ImageDimension;
    const torch::Tensor displacement = F::grid_sample(smoothedResidual, points.reshape(pointShape), sampleOpts)
                                         .reshape({ static_cast<int64_t>(ImageDimension), count })
                                         .t(); // {n, N}, (z, y, x) in full-res voxels
    const torch::Tensor warpedPoints = (points + (displacement / scale).flip(-1)).reshape(pointShape);
    const torch::Tensor moving = F::grid_sample(pick(l, movingLayers[l]), warpedPoints, sampleOpts);
    const int64_t       channels = moving.size(1);
    const torch::Tensor fixed = pick(l, fixedLayers[l]).reshape({ channels, voxels }).index_select(1, drawn);
    if (!movingMask.defined())
    {
      return losses[l]->forwardValue(fixed.t(), moving.reshape({ channels, count }).t());
    }
    torch::Tensor rows;
    {
      torch::NoGradGuard noGrad;
      torch::Tensor      atTotal = warpedPoints.detach();
      if (smoothedTotal.defined())
      {
        const torch::Tensor total = F::grid_sample(smoothedTotal, points.reshape(pointShape), sampleOpts)
                                      .reshape({ static_cast<int64_t>(ImageDimension), count })
                                      .t();
        atTotal = (points + (total / scale).flip(-1)).reshape(pointShape);
      }
      rows = (F::grid_sample(movingLayerShare[l], atTotal, sampleOpts) >= 0.5).flatten().nonzero().flatten();
    }
    if (rows.numel() == 0)
    {
      return (moving * 0.0).sum();
    }
    return losses[l]->forwardValue(fixed.index_select(1, rows).t(),
                                   moving.reshape({ channels, count }).index_select(1, rows).t());
  };
  const std::vector<ImpactModelConfiguration> & movingConfigs =
    m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
  // Every tensor here is on the fixed grid: each model sees it resampled from that spacing to its voxel size.
  std::vector<double> fixedSpacing(ImageDimension);
  bool                resamples = false;
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    fixedSpacing[d] = m_FixedImage->GetSpacing()[d];
  }
  for (const auto & config : m_FixedModelsConfiguration)
  {
    const std::vector<float> & voxel = config.GetVoxelSize();
    for (unsigned int d = 0; d < ImageDimension && voxel.size() == ImageDimension; ++d)
    {
      resamples = resamples || (voxel[d] > 0.0f && std::abs(voxel[d] - fixedSpacing[d]) > 1e-6 * fixedSpacing[d]);
    }
  }
  // Outside the dense Jacobian mode, a 2D model sweeps the axis closest to head-feet; every tensor here is on the
  // fixed grid.
  const unsigned int         headFeetAxis = Impact::HeadFeetAxis(m_FixedImage->GetDirection(), ImageDimension);
  const std::vector<int64_t> pcaSweep =
    Impact::PcaSweepDimensions<ImageDimension>(m_FixedModelsConfiguration, headFeetAxis);
  // Layer l's number of principal components, 0 for none (unset, or not fewer than its channels).
  auto pcaComponents = [&](size_t l, int64_t channels) -> int64_t {
    const int64_t components = l < m_PCA.size() ? static_cast<int64_t>(m_PCA[l]) : 0;
    return components > 0 && components < channels ? components : 0;
  };
  // Each image's metadata, for the metadata-aware models (nArgs >= 4, e.g. SAM): their own normalisation stats, not
  // the undefined default. Fixed and moving configurations added through AddModelConfiguration share one impl, so the
  // image a model runs on is set right before it runs; a forward already run keeps the tensors it was handed.
  const ImpactImageMetadata fixedMetadata =
    featureMode ? ComputeImageMetadata<FixedImageType>(m_FixedImage) : ImpactImageMetadata{};
  const ImpactImageMetadata movingMetadata =
    featureMode ? ComputeImageMetadata<MovingImageType>(movingOnFixed) : ImpactImageMetadata{};
  auto useMetadata = [](const std::vector<ImpactModelConfiguration> & configs, const ImpactImageMetadata & metadata) {
    for (const auto & config : configs)
    {
      config.GetImpl()->imageMetadata = metadata;
    }
  };
  // Extract the moving feature layers from an image tensor and project them onto the stored PCA
  // bases (used for the initial extraction and for the FeatureMapUpdateInterval re-extraction).
  auto extractMovingFeatures = [&](const torch::Tensor & imageTensor) -> std::vector<torch::Tensor> {
    auto layers = Impact::ExtractFeatureLayers<ImageDimension>(
      movingConfigs, imageTensor, device, {}, false, fixedSpacing, static_cast<int>(headFeetAxis));
    for (size_t l = 0; l < layers.size() && l < pcaBasis.size(); ++l)
    {
      if (pcaBasis[l].defined())
      {
        layers[l] = Impact::PcaReduce(layers[l].squeeze(0), pcaBasis[l], 0, pcaSweep[l]).unsqueeze(0).contiguous();
      }
    }
    return layers;
  };
  // Dense Jacobian mode with a swept model: per ITK axis, the fixed layers swept along it -- PCA-reduced on their own
  // slices along it, pooled to the loss grid but along it -- and each layer's PCA basis. Built the first time the axis
  // is drawn: at most one set per axis.
  struct SweptFixedLayers
  {
    std::vector<torch::Tensor> layers;
    std::vector<torch::Tensor> basis;
    std::vector<torch::Tensor> masks; // the fixed mask on each layer's grid (with one)
  };
  std::vector<SweptFixedLayers> sweptFixed(ImageDimension);
  auto                          fixedSweptAlong = [&](unsigned int axis) -> const SweptFixedLayers & {
    SweptFixedLayers & entry = sweptFixed[axis];
    if (entry.layers.empty())
    {
      useMetadata(m_FixedModelsConfiguration, fixedMetadata);
      entry.layers = Impact::ExtractFeatureLayers<ImageDimension>(
        m_FixedModelsConfiguration, fixedT, device, {}, false, fixedSpacing, static_cast<int>(axis));
      useMetadata(movingConfigs, movingMetadata);
      const std::vector<int64_t> sweep = Impact::PcaSweepDimensions<ImageDimension>(m_FixedModelsConfiguration, axis);
      const auto                 keep = static_cast<int64_t>(ImageDimension - 1 - axis); // its tensor spatial index
      entry.basis.resize(entry.layers.size());
      for (size_t l = 0; l < entry.layers.size(); ++l)
      {
        torch::Tensor layer = entry.layers[l];
        if (const int64_t components = pcaComponents(l, layer.size(1)); components > 0)
        {
          layer = Impact::PcaReduce(layer.squeeze(0), entry.basis[l], components, sweep[l]).unsqueeze(0);
        }
        entry.layers[l] = poolExcept(layer, keep).contiguous();
        if (fixedMask.defined())
        {
          entry.masks.push_back(
            fixedMaskOn(std::vector<int64_t>(entry.layers[l].sizes().begin() + 2, entry.layers[l].sizes().end())));
        }
      }
    }
    return entry;
  };
  std::mt19937                                axisGenerator(m_Seed);
  std::uniform_int_distribution<unsigned int> axisDistribution(0, ImageDimension - 1);
  if (featureMode)
  {
    size_t layerCount = 0;
    for (const auto & config : m_FixedModelsConfiguration)
    {
      layerCount += NumberOfKeptLayers(config);
    }
    for (size_t l = 0; l < layerCount; ++l)
    {
      const std::string name =
        m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)];
      losses.push_back(Impact::LossFactory::Instance().Create(name));
      layerWeights.push_back(l < m_LayersWeight.size() ? m_LayersWeight[l] : 1.0f);
      if (sampled && losses.back()->IsSpatial())
      {
        itkExceptionMacro("ImpactFineRegistration: distance '"
                          << name
                          << "' correlates windows of whole maps, "
                             "which a random share of their voxels does not have; leave SamplingPercentage at 1.");
      }
    }
    if (!jacSliced && !sampledJacobian)
    {
      // The moving features are extracted here for Static mode only: the Jacobian mode re-extracts them from the
      // warped image at every iteration.
      useMetadata(m_FixedModelsConfiguration, fixedMetadata);
      fixedLayers = Impact::ExtractFeatureLayers<ImageDimension>(
        m_FixedModelsConfiguration, fixedT, device, {}, false, fixedSpacing, static_cast<int>(headFeetAxis));
      if (!jacobianMode)
      {
        useMetadata(movingConfigs, movingMetadata);
        movingLayers = Impact::ExtractFeatureLayers<ImageDimension>(
          movingConfigs, movingT, device, {}, false, fixedSpacing, static_cast<int>(headFeetAxis));
      }
      if (layerCount == 0 || fixedLayers.size() != layerCount || (!jacobianMode && movingLayers.size() != layerCount))
      {
        itkExceptionMacro("ImpactFineRegistration: fixed and moving produced "
                          << fixedLayers.size() << " and " << movingLayers.size() << " feature layers where "
                          << layerCount << " are kept; they must match and be non-empty.");
      }
      // Optional per-layer PCA: fit the basis on the fixed features and project BOTH onto it so
      // they live in a consistent reduced space (PCA[l] components; 0 or >= channels = no-op).
      pcaBasis.resize(fixedLayers.size());
      for (size_t l = 0; l < fixedLayers.size(); ++l)
      {
        const int64_t components = pcaComponents(l, fixedLayers[l].size(1));
        if (components == 0)
        {
          continue;
        }
        fixedLayers[l] =
          Impact::PcaReduce(fixedLayers[l].squeeze(0), pcaBasis[l], components, pcaSweep[l]).unsqueeze(0).contiguous();
        if (!jacobianMode)
        {
          movingLayers[l] = Impact::PcaReduce(movingLayers[l].squeeze(0), pcaBasis[l], components, pcaSweep[l])
                              .unsqueeze(0)
                              .contiguous();
        }
      }
    }
    // Every later extraction runs on the moving image.
    useMetadata(movingConfigs, movingMetadata);
    if (jacobianMode)
    {
      if (jacSliced)
      {
        if (resamples)
        {
          // The moving slices are compared index by index with the fixed features of the same slices: a model
          // resampling the image to its voxel size would hand back slices of another grid.
          itkExceptionMacro("ImpactFineRegistration Jacobian mode: a model of lower dimension than the image is "
                            "compared slice by slice, which a model resampling the image to its voxel size breaks; "
                            "give it the image's own spacing (or no voxel size).");
        }
        // Chunked evaluation (a sliced model split to bound memory) recombines per-chunk means by their point-count
        // weights, which equals the whole-volume loss ONLY for a per-point-mean distance. Reject a global distance
        // (e.g. NCC) when chunking can be active along the axis drawn, rather than silently returning a wrong
        // value/gradient; FeatureChunkSize=0 (whole volume) lifts the restriction.
        const int64_t longest = *std::max_element(spatial.begin(), spatial.end());
        if (m_FeatureChunkSize != 0 && static_cast<int64_t>(m_FeatureChunkSize) < longest)
        {
          for (size_t l = 0; l < losses.size(); ++l)
          {
            if (!losses[l]->IsPerPointMean())
            {
              itkExceptionMacro("ImpactFineRegistration Jacobian mode: distance '"
                                << (l < m_Distance.size() ? m_Distance[l] : std::string("L2"))
                                << "' does not decompose over the chunks of slices used to bound memory (a global "
                                   "statistic, or a window crossing the chunks). Set FeatureChunkSize=0 (whole "
                                   "volume) to use it.");
            }
          }
        }
      }
      else if (sampledJacobian)
      {
        // Each model runs on the patch of its receptive field around every point: a strictly positive PatchSize on
        // each of its axes, which a patch in a volume can be cut along. How many patches one forward takes is
        // measured on the device, as for the metric.
        const std::vector<ImpactModelConfiguration> * const sides[] = { &m_FixedModelsConfiguration, &movingConfigs };
        for (const auto * configs : sides)
        {
          for (const ImpactModelConfiguration & config : *configs)
          {
            const std::vector<int64_t> & patchSize = config.GetPatchSize();
            if (patchSize.size() < config.GetDimension() || std::any_of(patchSize.begin(),
                                                                        patchSize.begin() + config.GetDimension(),
                                                                        [](int64_t extent) { return extent <= 0; }))
            {
              itkExceptionMacro("ImpactFineRegistration: SamplingPercentage in Jacobian mode runs each model on the "
                                "patch of its receptive field around every drawn point; give "
                                << config.GetModelPath() << " a strictly positive PatchSize on each of its "
                                << config.GetDimension() << " axes.");
            }
            if (config.GetDimension() != ImageDimension && (config.GetDimension() != 2 || ImageDimension != 3))
            {
              itkExceptionMacro("ImpactFineRegistration: a " << config.GetDimension() << "D model's patch cannot be "
                                                             << "cut in a " << ImageDimension << "D image.");
            }
            ModelTo(config, device);
          }
        }
        Impact::ConfigureBatchSize(m_FixedModelsConfiguration, device, static_cast<int64_t>(m_BatchSize));
        if (!m_MovingModelsConfiguration.empty())
        {
          Impact::ConfigureBatchSize(m_MovingModelsConfiguration, device, static_cast<int64_t>(m_BatchSize));
        }
      }
      else
      {
        // Online mode: precompute the fixed features on the comparison grid. The moving features are
        // re-extracted from the warped image every iteration in the Adam loop and brought to this same
        // grid by resampling, so movingLayers / layerBaseGrid (the frozen-warp machinery) are unused here.
        fixedLayersOnline.resize(fixedLayers.size());
        for (size_t l = 0; l < fixedLayers.size(); ++l)
        {
          fixedLayersOnline[l] = poolLayerToLoss(fixedLayers[l]).contiguous();
          if (masked)
          {
            const std::vector<int64_t> ls(fixedLayersOnline[l].sizes().begin() + 2, fixedLayersOnline[l].sizes().end());
            fixedLayerMask.push_back(fixedMaskOn(ls));
            movingLayerShare.push_back(movingShareOn(ls));
            std::vector<int64_t> gs{ 1, 1 };
            gs.insert(gs.end(), ls.begin(), ls.end());
            layerBaseGrid.push_back(torch::affine_grid_generator(idAffine, gs, /*align_corners=*/true));
            layerSpatials.push_back(ls);
          }
        }
      }
    }
    else
    {
      // Static mode: downsample the fixed/moving feature layers to the Adam similarity resolution
      // (never upsampling), then precompute the identity base grid at each layer's resulting
      // resolution so the loss grid is built directly there (reuses grid0 for full-resolution layers).
      layerBaseGrid.resize(movingLayers.size());
      layerSpatials.resize(movingLayers.size());
      for (size_t l = 0; l < movingLayers.size(); ++l)
      {
        fixedLayers[l] = poolLayerToLoss(fixedLayers[l]).contiguous();
        movingLayers[l] = poolLayerToLoss(movingLayers[l]).contiguous();
        std::vector<int64_t> ls(movingLayers[l].sizes().begin() + 2, movingLayers[l].sizes().end());
        layerSpatials[l] = ls;
        if (ls == spatial)
        {
          layerBaseGrid[l] = grid0;
        }
        else
        {
          std::vector<int64_t> gs{ 1, 1 };
          for (auto s : ls)
          {
            gs.push_back(s);
          }
          layerBaseGrid[l] = torch::affine_grid_generator(idAffine, gs, /*align_corners=*/true);
        }
        if (masked)
        {
          fixedLayerMask.push_back(fixedMaskOn(ls));
          movingLayerShare.push_back(movingShareOn(ls));
          if (sampled && fixedMask.defined())
          {
            fixedLayerCandidates.push_back(fixedLayerMask.back().flatten().nonzero().flatten());
          }
        }
      }
    }
  }

  // Reference control field at which `movingLayers` were extracted (0 = un-warped). With
  // FeatureMapUpdateInterval > 0 the moving features are periodically re-extracted from the
  // moving image warped by the current total field, and the loop warps them by the residual
  // (theta - thetaRef); thetaRef stays 0 (residual == theta) when disabled.
  torch::Tensor thetaRef = torch::zeros_like(theta);
  bool          refreshed = false; // whether thetaRef has moved off 0

  // ConvexAdam-style intensity refinement: evaluate the similarity at the control-grid resolution
  // (coarseSpatial), so GridShrinkFactor plays the role of the reference's grid_sp_adam. Moving/fixed
  // are pooled once to that grid and the field is warped and compared there, instead of upsampling
  // the field and warping at full resolution every iteration. With GridShrinkFactor==1
  // coarseSpatial==spatial and this is bit-identical to the full-res path.
  torch::Tensor movingCoarse, fixedCoarse, gridBaseCoarse, fixedCoarseMask, movingCoarseShare;
  if (!featureMode)
  {
    auto poolToCoarse = [&](const torch::Tensor & t) -> torch::Tensor {
      const std::vector<int64_t> ts(t.sizes().begin() + 2, t.sizes().end());
      if (ts == coarseSpatial)
      {
        return t;
      }
      if constexpr (ImageDimension == 3)
        return F::adaptive_avg_pool3d(
          t, F::AdaptiveAvgPool3dFuncOptions({ coarseSpatial[0], coarseSpatial[1], coarseSpatial[2] }));
      else
        return F::adaptive_avg_pool2d(t, F::AdaptiveAvgPool2dFuncOptions({ coarseSpatial[0], coarseSpatial[1] }));
    };
    movingCoarse = poolToCoarse(movingT);
    fixedCoarse = poolToCoarse(fixedT);
    fixedCoarseMask = fixedMaskOn(coarseSpatial);
    movingCoarseShare = movingShareOn(coarseSpatial);
    std::vector<int64_t> gs{ 1, 1 };
    for (auto s : coarseSpatial)
    {
      gs.push_back(s);
    }
    gridBaseCoarse = torch::affine_grid_generator(idAffine, gs, /*align_corners=*/true);
  }

  // ---- 4. Adam loop (entirely on device; no host copies). ----
  torch::optim::Adam optimizer({ theta },
                               torch::optim::AdamOptions(m_LearningRate)
                                 .betas(std::make_tuple(m_Beta1, m_Beta2))
                                 .eps(m_Epsilon));

  m_MetricValuesPerIteration.clear();
  m_MetricValuesPerIteration.reserve(m_NumberOfIterations);

  // Every layer's loss divided by its value at the stage's first iteration, so each starts at 1 and
  // LayersWeight weighs comparable quantities (see Impact::LossNormalization).
  Impact::LossNormalization normalization;
  auto normalized = [&](size_t l, const torch::Tensor & value) -> torch::Tensor {
    if (!m_NormalizeLosses)
    {
      return value;
    }
    if (!normalization.IsLatched(l))
    {
      normalization.Latch(l, value.detach().template item<double>());
    }
    return value * normalization.Factor(l);
  };

  torch::Tensor warped;
  for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
  {
    optimizer.zero_grad();
    // Dense Jacobian mode with a swept model: the image axis this iteration sweeps, and its fixed layers.
    const unsigned int       sweptAxis = jacSliced ? axisDistribution(axisGenerator) : 0;
    const SweptFixedLayers * fixedSwept = jacSliced ? &fixedSweptAlong(sweptAxis) : nullptr;
    if (featureMode && !sampledJacobian) // the sampled Jacobian mode draws them once it knows the channels
    {
      drawSubsets(jacSliced ? fixedSwept->layers : jacobianMode ? fixedLayersOnline : fixedLayers);
    }

    // Diffusion regularizer: sum over spatial axes of mean( forward-difference(field)^2 ).
    // As in ConvexAdam, the penalty is measured on the SMOOTHED control field (the same field that
    // is warped with), so the smoothing passes shape the regularizer too (identity when smoothing is
    // disabled). Each axis term is normalized by the control-point spacing (full / coarse size)
    // squared, so the penalty is a physical gradient (independent of GridShrinkFactor and matching
    // ConvexAdam's grid_sp_adam-unit field). A size-1 axis has no forward difference and is skipped.
    const torch::Tensor regField = smoothControl(theta);
    torch::Tensor       reg = torch::zeros({}, theta.options());
    for (unsigned int ax = 0; ax < ImageDimension; ++ax)
    {
      const int64_t tdim = 2 + static_cast<int64_t>(ax);
      const int64_t len = regField.size(tdim);
      if (len < 2)
      {
        continue;
      }
      // The gradient in mm per mm: the field (s_min units) differentiated per control cell of controlSpacing voxels,
      // each s_a mm -- exactly the voxel-unit gradient on an isotropic image.
      const double controlSpacing = static_cast<double>(spatial[ax]) / static_cast<double>(len);
      const double unitRatio = finest / voxelSide[ax];
      reg = reg + (regField.narrow(tdim, 1, len - 1) - regField.narrow(tdim, 0, len - 1)).pow(2).mean() /
                    (controlSpacing * controlSpacing) * (unitRatio * unitRatio);
    }
    reg = reg * m_RegularizationWeight;

    double lossValue;
    if (jacobianMode)
    {
      // "Jacobian" (online) mode: warp the moving IMAGE by the current field and RE-EXTRACT features
      // through the network with autograd, descending on the true loss F(warp(I)) whose gradient carries
      // the network term d(feature)/d(displacement). A model whose features downsample every axis
      // (full-dimension) is compared whole, its features matched to the fixed ones by RESAMPLING, like the
      // itkv4 metric; a sliced model (features keep the swept axis) is taken in chunks of slices along the
      // axis drawn for this iteration, to bound peak memory.
      if (sampledJacobian)
      {
        // elastix's Jacobian scheme on the dense field. The points of this iteration: that share of the fixed voxels,
        // drawn anew from the seeded generator, of which those whose every model patch fits in the image are kept
        // (elastix's SampleCheck).
        reg.backward();
        // The field the patches are warped by. Every batch below back-propagates into this detached leaf, and its
        // accumulated gradient crosses the smoothing passes to the control grid once, at the end: the smoothing's graph
        // is shared by every batch of every model, and a backward frees the graph it runs through.
        const torch::Tensor smoothedControl = smoothControl(theta); // graph to theta
        torch::Tensor       smoothed = smoothedControl.detach().requires_grad_(true);
        smoothed.mutable_grad() = torch::zeros_like(smoothed);
        int64_t             voxels = 1;
        for (const int64_t extent : spatial)
        {
          voxels *= extent;
        }
        const int64_t count = std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(voxels)));
        const torch::Tensor flat = torch::randint(voxels, { count }, torch::TensorOptions().dtype(torch::kLong));
        torch::Tensor       points = torch::empty({ count, static_cast<int64_t>(ImageDimension) }, torch::kLong);
        int64_t             stride = 1;
        for (unsigned int d = 0; d < ImageDimension; ++d) // ITK index, x fastest in the flat voxel number
        {
          const int64_t extent = spatial[ImageDimension - 1 - d];
          points.select(1, d).copy_(flat.div(stride, "trunc").remainder(extent));
          stride *= extent;
        }
        std::vector<double> lastIndex(ImageDimension), gridScale(ImageDimension);
        for (unsigned int d = 0; d < ImageDimension; ++d)
        {
          lastIndex[d] = static_cast<double>(spatial[ImageDimension - 1 - d] - 1);
          gridScale[d] = lastIndex[d] > 0.0 ? lastIndex[d] / 2.0 : 1.0;
        }
        const torch::Tensor upper = torch::tensor(lastIndex, torch::kFloat64);
        torch::Tensor       fits = torch::ones({ count }, torch::kBool);
        for (size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
        {
          const ImpactModelConfiguration & config = m_FixedModelsConfiguration[i];
          const torch::Tensor              corners = Impact::SampledPatchPositions<ImageDimension>(
            config, points, fixedSpacing, m_Seed, i, Impact::PatchCorners(config));
          fits = fits & ((corners >= 0.0) & (corners <= upper)).flatten(1).all(1);
        }
        if (fixedMask.defined()) // the points drawn in the fixed mask
        {
          fits = fits & fixedMask.flatten().index_select(0, flat.to(device)).to(torch::kCPU);
        }
        torch::Tensor       kept = points.index_select(0, fits.nonzero().flatten());
        const torch::Tensor toGrid = torch::tensor(gridScale, torch::kFloat64);
        if (movingMask.defined() && kept.size(0) > 0)
        {
          // ... that the moving mask, warped by the field, holds at their centre.
          torch::NoGradGuard   noGrad;
          std::vector<int64_t> where(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, n, Dim}
          where[ImageDimension] = kept.size(0);
          where[ImageDimension + 1] = ImageDimension;
          const torch::Tensor centres =
            (kept.to(torch::kFloat64) / toGrid - 1.0).to(torch::kFloat32).to(device).reshape(where);
          const torch::Tensor displacement = F::grid_sample(smoothedControl.detach(), centres, sampleOpts)
                                               .reshape({ static_cast<int64_t>(ImageDimension), kept.size(0) })
                                               .t();
          const torch::Tensor holds =
            F::grid_sample(movingMaskShare, centres + (displacement / scale).flip(-1).reshape(where), sampleOpts) >=
            0.5;
          kept = kept.index_select(0, holds.flatten().nonzero().flatten().to(torch::kCPU));
        }
        const int64_t n = kept.size(0);
        if (n == 0 && !masked)
        {
          itkExceptionMacro("ImpactFineRegistration: no drawn point has its model patch inside the image; the patch "
                            "(PatchSize x voxel size) is larger than the image, or SamplingPercentage too small.");
        }
        double simValue = 0.0; // no point in the masks: no data this iteration
        if (n > 0)
        {
          // Model i's patches of points [begin, end), {b, 1, reversed PatchSize...}: in the fixed image, or in the
          // moving image as the field warps it (differentiable, so the gradient reaches the control grid). The patch
          // voxels are placed a batch at a time, as grid_sample coordinates (x, y, z: ITK order).
          auto patchesOf = [&](size_t i, int64_t begin, int64_t end, bool moving) {
            const torch::Tensor grid =
              (Impact::SampledPatchPositions<ImageDimension>(
                 m_FixedModelsConfiguration[i], kept.narrow(0, begin, end - begin), fixedSpacing, m_Seed, i) /
                 toGrid -
               1.0)
                .to(torch::kFloat32)
                .to(device);
            const int64_t        samples = grid.size(0) * grid.size(1);
            std::vector<int64_t> where(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, samples, Dim}
            where[ImageDimension] = samples;
            where[ImageDimension + 1] = ImageDimension;
            torch::Tensor at = grid.reshape(where);
            if (moving)
            {
              const torch::Tensor displacement =
                F::grid_sample(smoothed, at, sampleOpts).reshape({ static_cast<int64_t>(ImageDimension), samples }).t();
              at = at + (displacement / scale).flip(-1).reshape(where);
            }
            std::vector<int64_t> shape{ end - begin, 1 };
            for (const int64_t extent : Impact::PatchTensorShape(m_FixedModelsConfiguration[i]))
            {
              shape.push_back(extent);
            }
            return F::grid_sample(moving ? movingT : fixedT, at, sampleOpts).reshape(shape);
          };
          // Every kept layer's centre feature vector {b, C} of a model run on `patches`, normalized, the voxel the
          // patch is centred on (GetCentersIndexLayers' size / 2).
          auto centresOf = [&](const ImpactModelConfiguration & config,
                               const torch::Tensor &            patches,
                               const ImpactImageMetadata &      metadata) {
            std::vector<int64_t> repeat(patches.dim(), 1);
            repeat[1] = static_cast<int64_t>(config.GetNumberOfChannels());
            std::vector<torch::jit::IValue> outputs =
              Forward(config, patches.repeat(repeat).to(GetModelDtype(config)), metadata);
            const std::vector<bool> &  mask = config.GetLayersMask();
            std::vector<torch::Tensor> layers;
            for (size_t it = 0; it < outputs.size() && it < mask.size(); ++it)
            {
              if (mask[it])
              {
                torch::Tensor layer = outputs[it].toTensor();
                while (layer.dim() > 2)
                {
                  layer = layer.select(2, layer.size(2) / 2);
                }
                layers.push_back(
                  Impact::NormalizeFeatureChannels(layer.to(torch::kFloat32), config.GetFeatureNormalization(), 1));
              }
            }
            return layers;
          };
          std::vector<size_t> firstLayer{ 0 }; // model i's kept layers start at firstLayer[i] in the flat list
          for (const auto & config : m_FixedModelsConfiguration)
          {
            firstLayer.push_back(firstLayer.back() + NumberOfKeptLayers(config));
          }
          // Every point's centre features on one side, without the graph, a batch of patches at a time.
          std::vector<torch::Tensor> fixedCentres(losses.size());
          std::vector<torch::Tensor> movingCentres(losses.size());
          auto                       gather = [&](bool moving) {
            torch::NoGradGuard noGrad;
            for (size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
            {
              const ImpactModelConfiguration & config = moving ? movingConfigs[i] : m_FixedModelsConfiguration[i];
              std::vector<std::vector<torch::Tensor>> pieces(firstLayer[i + 1] - firstLayer[i]);
              Impact::ForEachBatch(
                config,
                device,
                n,
                [&](int64_t begin, int64_t end) {
                  const std::vector<torch::Tensor> layers =
                    centresOf(config, patchesOf(i, begin, end, moving), moving ? movingMetadata : fixedMetadata);
                  for (size_t k = 0; k < pieces.size(); ++k)
                  {
                    pieces[k].push_back(layers[k]);
                  }
                },
                [] {});
              for (size_t k = 0; k < pieces.size(); ++k)
              {
                (moving ? movingCentres : fixedCentres)[firstLayer[i] + k] = torch::cat(pieces[k]);
              }
            }
          };
          gather(false);
          drawSubsets(fixedCentres);
          // PCA, a 3D model's rule on the points: the basis fitted on the fixed features of this iteration's points,
          // each side centred by its own mean over them -- the moving one taken without the graph, so that every point
          // keeps a gradient of its own.
          std::vector<torch::Tensor> basis(losses.size());
          std::vector<torch::Tensor> movingMean(losses.size());
          bool                       reduces = false;
          for (size_t l = 0; l < losses.size(); ++l)
          {
            if (const int64_t components = pcaComponents(l, fixedCentres[l].size(1)); components > 0)
            {
              basis[l] = Impact::PcaFit(fixedCentres[l].t(), components);
              fixedCentres[l] = Impact::PcaTransform(fixedCentres[l].t(), basis[l]).t();
              reduces = true;
            }
          }
          const bool latching = m_NormalizeLosses && !normalization.IsLatched(0);
          const bool perPoint =
            std::all_of(losses.begin(), losses.end(), [](const std::unique_ptr<Impact::Loss> & loss) {
              return loss->IsPerPointMean();
            });
          if (!perPoint || reduces || latching)
          {
            gather(true);
          }
          for (size_t l = 0; l < losses.size(); ++l)
          {
            if (basis[l].defined())
            {
              movingMean[l] = movingCentres[l].mean(0).unsqueeze(1);
            }
          }
          // A moving side's features {b, C} as layer l compares them.
          auto reduced = [&](size_t l, const torch::Tensor & centres) {
            return pick(l,
                        basis[l].defined() ? Impact::PcaTransform(centres.t(), basis[l], movingMean[l]).t() : centres);
          };
          if (latching)
          {
            for (size_t l = 0; l < losses.size(); ++l)
            {
              normalization.Latch(l,
                                  losses[l]
                                    ->forwardValue(pick(l, fixedCentres[l]), reduced(l, movingCentres[l]))
                                    .template item<double>());
            }
          }
          // A global distance (NCC) mixes the points: its gradient with respect to each point's features is only known
          // from the whole set, so it is taken there, on the features as a leaf, and seeds each batch's backward below.
          std::vector<torch::Tensor> seeds;
          if (!perPoint)
          {
            std::vector<torch::Tensor> leaves;
            torch::Tensor              total = torch::zeros({}, theta.options());
            for (size_t l = 0; l < losses.size(); ++l)
            {
              leaves.push_back(movingCentres[l].detach().requires_grad_(true));
              total = total + layerWeights[l] *
                                normalized(l, losses[l]->forwardValue(pick(l, fixedCentres[l]), reduced(l, leaves[l])));
            }
            seeds = torch::autograd::grad({ total }, leaves);
            simValue = total.template item<double>();
          }
          // The moving side again, with the graph, a batch at a time: each batch's backward reaches the smoothed field
          // and frees its network graph. A per-point distance's batch weighs its share of the points. A batch that runs
          // out of device memory is replayed at half the size, with the gradient it had added taken back.
          torch::Tensor gradientBefore;
          for (size_t i = 0; i < movingConfigs.size(); ++i)
          {
            Impact::ForEachBatch(
              movingConfigs[i],
              device,
              n,
              [&](int64_t begin, int64_t end) {
                gradientBefore = smoothed.grad().clone();
                const std::vector<torch::Tensor> layers =
                  centresOf(movingConfigs[i], patchesOf(i, begin, end, true), movingMetadata);
                torch::Tensor surrogate = torch::zeros({}, theta.options());
                for (size_t k = 0; k < layers.size(); ++k)
                {
                  const size_t l = firstLayer[i] + k;
                  if (perPoint)
                  {
                    const torch::Tensor value = losses[l]->forwardValue(
                      pick(l, fixedCentres[l].narrow(0, begin, end - begin)), reduced(l, layers[k]));
                    surrogate = surrogate + layerWeights[l] * normalized(l, value) *
                                              (static_cast<double>(end - begin) / static_cast<double>(n));
                  }
                  else
                  {
                    surrogate = surrogate + (layers[k] * seeds[l].narrow(0, begin, end - begin)).sum();
                  }
                }
                surrogate.backward();
                if (perPoint)
                {
                  simValue += surrogate.template item<double>();
                }
              },
              [&] { smoothed.mutable_grad().copy_(gradientBefore); });
          }
        }
        smoothedControl.backward(smoothed.grad()); // the smoothing passes, once: -> theta.grad
        lossValue = simValue + reg.template item<double>();
      }
      else if (jacSliced)
      {
        // Peak memory is bounded by taking the field's swept axis in chunks: each chunk backprops
        // into a detached grid leaf -- its network subgraph is then freed -- and the shared field->grid
        // graph is traversed once at the end. The per-point feature distance decomposes over voxels, so
        // weighting a chunk by its voxel fraction makes the accumulated gradient equal the whole mean's.
        const auto keep = static_cast<int64_t>(ImageDimension - 1 - sweptAxis); // tensor spatial index swept
        reg.backward();
        const torch::Tensor fullGrid = gridFromControl(theta); // {1, spatial..., Dim}, graph to theta
        torch::Tensor       gridLeaf = fullGrid.detach().clone();
        gridLeaf.set_requires_grad(true);
        const int64_t nLead = gridLeaf.size(1 + keep);
        const int64_t chunk = (m_FeatureChunkSize == 0) ? nLead : std::min<int64_t>(m_FeatureChunkSize, nLead);
        // A swept model's PCA centres the moving features by their own mean over the slices its basis was read on
        // (Impact::PcaSlices), which the chunks only give piece by piece: taken from those slices of the moving
        // image as the field warps it now, without the graph, as the FireANTs engine detaches it.
        std::vector<torch::Tensor> movingMeans(fixedSwept->basis.size());
        if (std::any_of(fixedSwept->basis.begin(), fixedSwept->basis.end(), [](const torch::Tensor & basis) {
              return basis.defined();
            }))
        {
          const torch::Tensor slices =
            F::grid_sample(movingT, Impact::PcaSlices(gridLeaf.detach(), 1 + keep), sampleOpts);
          const std::vector<torch::Tensor> ml = Impact::ExtractFeatureLayers<ImageDimension>(
            movingConfigs, slices, device, {}, false, {}, static_cast<int>(sweptAxis));
          for (size_t l = 0; l < movingMeans.size(); ++l)
          {
            if (fixedSwept->basis[l].defined())
            {
              movingMeans[l] = Impact::PcaSampleMean(ml[l].squeeze(0));
            }
          }
        }
        // With masks, the voxels each layer counts in the whole volume at this iteration, so that a chunk weighs its
        // share of them (instead of its share of the voxels).
        std::vector<torch::Tensor> counted, countedTotal;
        if (masked)
        {
          torch::Tensor warpedMoving;
          if (movingMask.defined())
          {
            torch::NoGradGuard noGrad;
            warpedMoving = F::grid_sample(movingMaskShare, fullGrid.detach(), sampleOpts) >= 0.5;
          }
          for (size_t l = 0; l < fixedSwept->layers.size(); ++l)
          {
            const std::vector<int64_t> ls(fixedSwept->layers[l].sizes().begin() + 2,
                                          fixedSwept->layers[l].sizes().end());
            torch::Tensor              voxels = fixedMask.defined() ? fixedSwept->masks[l] : torch::Tensor();
            if (warpedMoving.defined())
            {
              const torch::Tensor holds = Impact::MaskShare<ImageDimension>(warpedMoving, ls) >= 0.5;
              voxels = voxels.defined() ? voxels.logical_and(holds) : holds;
            }
            counted.push_back(voxels);
            countedTotal.push_back(voxels.sum().clamp_min(1).to(torch::kFloat32));
          }
        }
        auto chunkShare = [&](size_t l, int64_t i0, int64_t di) {
          return counted[l].narrow(2 + keep, i0, di).sum().to(torch::kFloat32) / countedTotal[l];
        };
        // Each layer's loss over the chunk [i0, i0 + di) of the swept axis.
        auto chunkLosses = [&](int64_t i0, int64_t di, const torch::Tensor & gz, bool withGrad) {
          torch::Tensor              movingChunk = F::grid_sample(movingT, gz, sampleOpts); // grad -> gridLeaf
          std::vector<torch::Tensor> ml = Impact::ExtractFeatureLayers<ImageDimension>(
            movingConfigs, movingChunk, device, {}, withGrad, {}, static_cast<int>(sweptAxis));
          std::vector<torch::Tensor> values;
          for (size_t l = 0; l < ml.size(); ++l)
          {
            torch::Tensor mll = ml[l];
            if (fixedSwept->basis[l].defined())
            {
              mll = Impact::PcaTransform(mll.squeeze(0), fixedSwept->basis[l], movingMeans[l]).unsqueeze(0);
            }
            mll = poolExcept(mll, keep); // the features keep the swept axis, so only the others are resampled
            values.push_back(layerLoss(l,
                                       pick(l, fixedSwept->layers[l].narrow(2 + keep, i0, di)),
                                       pick(l, mll),
                                       masked ? counted[l].narrow(2 + keep, i0, di) : torch::Tensor()));
          }
          return values;
        };
        if (m_NormalizeLosses && !normalization.IsLatched(0))
        {
          // The factors take each layer's value over the whole volume, which the chunks only give piece by
          // piece: one pass without the graph at the stage's starting field, the means recombined by the
          // chunks' voxel fractions (exact for these per-point means).
          torch::NoGradGuard  noGradPrepass;
          std::vector<double> whole;
          for (int64_t i0 = 0; i0 < nLead; i0 += chunk)
          {
            const int64_t              di = std::min(chunk, nLead - i0);
            std::vector<torch::Tensor> values = chunkLosses(i0, di, gridLeaf.detach().narrow(1 + keep, i0, di), false);
            whole.resize(values.size(), 0.0);
            for (size_t l = 0; l < values.size(); ++l)
            {
              whole[l] += masked
                            ? (values[l] * chunkShare(l, i0, di)).template item<double>()
                            : values[l].template item<double>() * static_cast<double>(di) / static_cast<double>(nLead);
            }
          }
          for (size_t l = 0; l < whole.size(); ++l)
          {
            normalization.Latch(l, whole[l]);
          }
        }
        double simValue = 0.0;
        for (int64_t i0 = 0; i0 < nLead; i0 += chunk)
        {
          const int64_t              di = std::min(chunk, nLead - i0);
          torch::Tensor              gz = gridLeaf.narrow(1 + keep, i0, di);
          std::vector<torch::Tensor> values = chunkLosses(i0, di, gz, true);
          torch::Tensor              simChunk = torch::zeros({}, theta.options());
          for (size_t l = 0; l < values.size(); ++l)
          {
            simChunk = masked ? simChunk + layerWeights[l] * normalized(l, values[l]) * chunkShare(l, i0, di)
                              : simChunk + layerWeights[l] * normalized(l, values[l]);
          }
          torch::Tensor weighted =
            masked ? simChunk : simChunk * (static_cast<double>(di) / static_cast<double>(nLead));
          weighted.backward(); // accumulates into gridLeaf.grad; this chunk's network graph is freed
          simValue += weighted.template item<double>();
        }
        fullGrid.backward(gridLeaf.grad()); // shared field->grid graph, once: gridLeaf.grad -> theta.grad
        lossValue = simValue + reg.template item<double>();
      }
      else
      {
        // Full-dimension model: warp the whole moving image, re-extract its features, and resample each
        // layer to the fixed layer's loss-grid resolution before comparing (a single backward pass).
        torch::Tensor              movingWarped = F::grid_sample(movingT, gridFromControl(theta), sampleOpts);
        std::vector<torch::Tensor> ml =
          Impact::ExtractFeatureLayers<ImageDimension>(movingConfigs, movingWarped, device, {}, true, fixedSpacing);
        torch::Tensor sim = torch::zeros({}, theta.options());
        for (size_t l = 0; l < ml.size(); ++l)
        {
          torch::Tensor mll = ml[l];
          if (l < pcaBasis.size() && pcaBasis[l].defined())
          {
            mll = Impact::PcaTransform(mll.squeeze(0), pcaBasis[l]).unsqueeze(0);
          }
          const std::vector<int64_t> tgt(fixedLayersOnline[l].sizes().begin() + 2, fixedLayersOnline[l].sizes().end());
          const std::vector<int64_t> cur(mll.sizes().begin() + 2, mll.sizes().end());
          if (cur != tgt)
          {
            if constexpr (ImageDimension == 3)
              mll = F::interpolate(mll, F::InterpolateFuncOptions().size(tgt).mode(torch::kTrilinear).align_corners(true));
            else
              mll = F::interpolate(mll, F::InterpolateFuncOptions().size(tgt).mode(torch::kBilinear).align_corners(true));
          }
          torch::Tensor counted;
          if (masked)
          {
            counted = countedVoxels(fixedLayerMask[l], movingLayerShare[l], [&] {
              return layerBaseGrid[l] +
                     (resizeField(smoothControl(theta.detach()), layerSpatials[l]).permute(toChannelLast) / scale)
                       .flip(-1);
            });
          }
          sim =
            sim + layerWeights[l] * normalized(l, layerLoss(l, pick(l, fixedLayersOnline[l]), pick(l, mll), counted));
        }
        torch::Tensor loss = sim + reg;
        loss.backward();
        lossValue = loss.template item<double>();
      }
    }
    else
    {
      // Warp by the residual control field (theta - thetaRef): the moving features were extracted
      // at thetaRef, so this is identity right after a refresh (and == theta when disabled).
      torch::Tensor similarity;
      if (featureMode)
      {
        // Build each layer's sampling grid directly at its resolution from the smoothed residual
        // (no full-res detour); smoothing is shared across layers.
        const torch::Tensor smoothedResidual = smoothControl(theta - thetaRef);
        // The moving mask follows the total field, which is the residual until a refresh.
        const torch::Tensor smoothedTotal =
          movingMask.defined() && refreshed ? smoothControl(theta).detach() : torch::Tensor();
        similarity = torch::zeros({}, theta.options());
        for (size_t l = 0; l < movingLayers.size(); ++l)
        {
          if (sampled)
          {
            similarity =
              similarity + layerWeights[l] * normalized(l, sampledLayerLoss(l, smoothedResidual, smoothedTotal));
            continue;
          }
          // Only the drawn channels are warped.
          const torch::Tensor grid = gridForLayerControl(smoothedResidual, l);
          torch::Tensor       warpedLayer = F::grid_sample(pick(l, movingLayers[l]), grid, sampleOpts);
          torch::Tensor       counted;
          if (masked)
          {
            counted = countedVoxels(fixedLayerMask[l], movingLayerShare[l], [&] {
              return smoothedTotal.defined() ? gridForLayerControl(smoothedTotal, l) : grid;
            });
          }
          similarity =
            similarity + layerWeights[l] * normalized(l, layerLoss(l, pick(l, fixedLayers[l]), warpedLayer, counted));
        }
      }
      else
      {
        // Build the grid directly at the control-grid resolution and warp the pooled moving image.
        torch::Tensor dd = smoothControl(theta - thetaRef).permute(toChannelLast);
        torch::Tensor grid = gridBaseCoarse + (dd / scale).flip(-1);
        torch::Tensor warpedImage = F::grid_sample(movingCoarse, grid, sampleOpts);
        if (!masked)
        {
          similarity = (warpedImage - fixedCoarse).pow(2).mean();
        }
        else
        {
          const torch::Tensor counted = countedVoxels(fixedCoarseMask, movingCoarseShare, [&] { return grid; });
          similarity =
            torch::where(counted, (warpedImage - fixedCoarse).pow(2), 0.0).sum() / counted.sum().clamp_min(1);
        }
      }
      torch::Tensor loss = similarity + reg;
      loss.backward();
      lossValue = loss.template item<double>();
    }
    optimizer.step();

    m_MetricValuesPerIteration.push_back(lossValue);

    // FeatureMapUpdateInterval: periodically re-extract the moving feature maps from the moving
    // image warped by the current total field, and reset the residual baseline. theta (and its
    // Adam moments) is untouched; only the reference frame of the moving features changes.
    if (featureMode && !jacobianMode && m_FeatureMapUpdateInterval > 0 &&
        (iteration + 1) % static_cast<unsigned int>(m_FeatureMapUpdateInterval) == 0 &&
        iteration + 1 < m_NumberOfIterations)
    {
      torch::NoGradGuard noGrad;
      thetaRef = theta.detach().clone();
      refreshed = true;
      torch::Tensor movingWarped = F::grid_sample(movingT, gridFromControl(thetaRef), sampleOpts);
      movingLayers = extractMovingFeatures(movingWarped);
      for (size_t l = 0; l < movingLayers.size(); ++l)
      {
        movingLayers[l] = poolLayerToLoss(movingLayers[l]).contiguous(); // match the pooled fixed layers
      }
    }
  }

  // Final forward pass (no grad) so `warped` reflects the converged field and so the
  // NumberOfIterations == 0 case (pure identity / geometry check) is well defined.
  {
    torch::NoGradGuard noGrad;
    warped = F::grid_sample(movingT, gridFromControl(theta), sampleOpts);
  }

  // ---- 5. Geometry-correct writeback (once), via the shared convention. ----
  DisplacementFieldType * output = this->GetOutput();
  output->SetRegions(m_FixedImage->GetLargestPossibleRegion());
  output->SetSpacing(m_FixedImage->GetSpacing());
  output->SetOrigin(m_FixedImage->GetOrigin());
  output->SetDirection(m_FixedImage->GetDirection());
  output->Allocate();

  Impact::WriteVoxelFieldToDisplacement<ImageDimension>(controlGridToFullField(theta.detach()) * unitToVoxels,
                                                        m_FixedImage->GetSpacing(),
                                                        m_FixedImage->GetDirection(),
                                                        output);

  // Wrap the field in a ready-to-use transform.
  m_DisplacementFieldTransform = DisplacementFieldTransformType::New();
  m_DisplacementFieldTransform->SetDisplacementField(output);

  // ---- 6. Warped moving image on the fixed grid (for inspection). ----
  torch::Tensor wCpu = warped.detach().squeeze(0).squeeze(0).contiguous().to(torch::kCPU); // {z,y,x}
  const float * wPtr = wCpu.data_ptr<float>();

  m_WarpedMovingImage = WarpedImageType::New();
  m_WarpedMovingImage->SetRegions(m_FixedImage->GetLargestPossibleRegion());
  m_WarpedMovingImage->SetSpacing(m_FixedImage->GetSpacing());
  m_WarpedMovingImage->SetOrigin(m_FixedImage->GetOrigin());
  m_WarpedMovingImage->SetDirection(m_FixedImage->GetDirection());
  m_WarpedMovingImage->Allocate();

  ImageRegionIteratorWithIndex<WarpedImageType> wit(m_WarpedMovingImage, m_WarpedMovingImage->GetLargestPossibleRegion());
  for (wit.GoToBegin(); !wit.IsAtEnd(); ++wit)
  {
    const auto idx = wit.GetIndex();
    size_t     base = 0;
    for (unsigned int t = 0; t < ImageDimension; ++t)
    {
      base = base * static_cast<size_t>(spatial[t]) + static_cast<size_t>(idx[ImageDimension - 1 - t]);
    }
    wit.Set(wPtr[base]);
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Device: " << m_Device << std::endl;
  os << indent << "NumberOfIterations: " << m_NumberOfIterations << std::endl;
  os << indent << "LearningRate: " << m_LearningRate << std::endl;
  os << indent << "RegularizationWeight: " << m_RegularizationWeight << std::endl;
  os << indent << "NormalizeLosses: " << (m_NormalizeLosses ? "on" : "off") << std::endl;
  os << indent << "FixedModelsConfiguration count: " << m_FixedModelsConfiguration.size() << std::endl;
}

} // end namespace itk

#endif // itkImpactFineRegistration_hxx

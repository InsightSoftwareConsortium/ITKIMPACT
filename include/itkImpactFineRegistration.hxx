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
#include "itkImpactRegionSampling.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkIdentityTransform.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <array>
#include <limits>
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
 * voxel sizes (the finest spacing when unset). `voxels`, when given, keeps those patch voxels only (P of them).
 * `points` stay on the host; the positions are formed with `options` (dtype and device), so that a registration on
 * the GPU places its patches there: only the per-point planes of a 2D model are drawn on the host. */
template <unsigned int Dim>
torch::Tensor
SampledPatchPositions(const ImpactModelConfiguration & config,
                      const torch::Tensor &            points,
                      const std::vector<double> &      spacing,
                      unsigned int                     seed,
                      size_t                           modelIndex,
                      const std::vector<int64_t> &     voxels = {},
                      const torch::TensorOptions &     options = torch::TensorOptions().dtype(torch::kFloat64))
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
  const torch::Tensor centres = points.to(options).unsqueeze(1); // {N, 1, Dim}
  const torch::Tensor perVoxel = 1.0 / torch::tensor(spacing, torch::kFloat64);
  if (modelDimension == Dim)
  {
    return centres + (offsets * perVoxel).to(options).unsqueeze(0);
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
  return centres + torch::einsum("njd,pd->npj", { axes.to(options), offsets.to(options) }) * perVoxel.to(options);
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
ImpactFineRegistration<TFixedImage, TMovingImage>::ImpactFineRegistration()
{
  this->DynamicMultiThreadingOn();
}

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
ImpactFineRegistration<TFixedImage, TMovingImage>::GetDisplacementFieldTransform() -> DisplacementFieldTransformType *
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
  const_cast<FixedImageType *>(m_FixedImage.GetPointer())->UpdateOutputInformation();
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
  if (m_RegionSize > 0)
  {
    // The out-of-core run: solved once, then the requested region of the dense field is interpolated from the
    // control grid (ImageSource's threaded GenerateData, through DynamicThreadedGenerateData).
    if (!m_Solved || m_SolvedTime < this->GetMTime())
    {
      this->SolveByRegions();
      m_Solved = true;
      m_SolvedTime = this->GetMTime();
    }
    Superclass::GenerateData();
    return;
  }
  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }
  // Feature (IMPACT) mode if any model is configured; otherwise raw-intensity MSE.
  const bool featureMode = !m_FixedModelsConfiguration.empty();

  const torch::Device device(m_Device);
  // The stage's own generator, on the CPU. torch::manual_seed reseeded every generator of the
  // process, a Python caller's included, and a draw on the device also followed whatever else draws
  // there, such as the batch-budget probe. Draws are made here and moved to the device.
  at::Generator generator = at::make_generator<at::CPUGeneratorImpl>(m_Seed);

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

  // ---- 2. Optimizable control grid {1, N, coarse z,y,x}; component order (z,y,x), in voxels, so that an Adam step
  // of LearningRate moves LearningRate voxels along every axis. The grid is GridShrinkFactor times coarser than the
  // image along every axis, whatever its spacing, as ConvexAdam's, and is upsampled to full resolution each
  // iteration. Warm-started from an initial field (e.g. the coarse stage) if provided, else zero. ----
  const std::vector<int64_t> shrink(ImageDimension, static_cast<int64_t>(std::max(1u, m_GridShrinkFactor)));
  std::vector<int64_t>       coarseSpatial(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    coarseSpatial[d] = std::max<int64_t>(1, spatial[d] / shrink[d]);
  }
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
        m_InitialDisplacementField, m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), device); // voxels
    if (coarseSpatial != spatial)
    {
      if constexpr (ImageDimension == 3)
        initField = torch::nn::functional::interpolate(initField,
                                                       torch::nn::functional::InterpolateFuncOptions()
                                                         .size(coarseSpatial)
                                                         .mode(torch::kTrilinear)
                                                         .align_corners(false));
      else
        initField = torch::nn::functional::interpolate(initField,
                                                       torch::nn::functional::InterpolateFuncOptions()
                                                         .size(coarseSpatial)
                                                         .mode(torch::kBilinear)
                                                         .align_corners(false));
    }
    theta = initField.set_requires_grad(true);
  }
  else
  {
    theta = torch::zeros(fieldShape, torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(true));
  }

  // ---- 3. Base identity sampling grid (normalized [-1,1], last-dim order x,y,z), align_corners=false: every grid,
  // sampling and resizing of this run follows ConvexAdam's convention. ----
  torch::Tensor idAffine = torch::eye(ImageDimension, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  idAffine = torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(ImageDimension), 1 }, idAffine.options()) }, 1)
               .unsqueeze(0); // {1, N, N+1}
  std::vector<int64_t> gridSize;
  gridSize.push_back(1);
  gridSize.push_back(1);
  for (auto s : spatial)
  {
    gridSize.push_back(s);
  }
  torch::Tensor grid0 = torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/false); // {1, z,y,x, N}

  // Per-component normalization in (z, y, x) order: the field units per grid_sample unit on the loss grid, as
  // ConvexAdam counts them: (cells - 1)/2 cells of shrink voxels. Its sampling grid (align_corners=false) spans
  // cells/2, so the field it returns is (cells - 1)/cells of the displacement it matched, and so is this one.
  std::vector<float> scaleValues(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    scaleValues[d] =
      (coarseSpatial[d] > 1) ? static_cast<float>(static_cast<double>(shrink[d]) * (coarseSpatial[d] - 1) / 2.0) : 1.0f;
  }
  torch::Tensor scale =
    torch::from_blob(scaleValues.data(), { static_cast<int64_t>(ImageDimension) }, torch::kFloat32).clone().to(device);
  // The same for a warp of the image itself (the Jacobian mode, a feature refresh, the warped output): half the
  // image's extent, what the returned field means.
  std::vector<float> imageScaleValues(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    imageScaleValues[d] = static_cast<float>(static_cast<double>(spatial[d]) / 2.0);
  }
  torch::Tensor imageScale =
    torch::from_blob(imageScaleValues.data(), { static_cast<int64_t>(ImageDimension) }, torch::kFloat32)
      .clone()
      .to(device);

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
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);
  // The control field read at a point: its border value past the first and last cell centres, as resizing it gives.
  const auto controlOpts =
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(false);

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
      return F::interpolate(field,
                            F::InterpolateFuncOptions().size(target).mode(torch::kTrilinear).align_corners(false));
    else
      return F::interpolate(field, F::InterpolateFuncOptions().size(target).mode(torch::kBilinear).align_corners(false));
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
    return grid0 + (dd / imageScale).flip(-1);
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
  std::vector<int64_t> cellVoxels(ImageDimension); // a cell of the loss grid in voxels, the image's size at most
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    cellVoxels[d] = std::min(shrink[d], spatial[d]);
  }
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
    if (std::vector<int64_t>(layer.sizes().begin() + 2, layer.sizes().end()) == spatial)
    {
      // A layer at the image's resolution: whole cells of `shrink` voxels, the tail dropped, as ConvexAdam pools.
      if constexpr (ImageDimension == 3)
        return F::avg_pool3d(layer, F::AvgPool3dFuncOptions(cellVoxels).stride(cellVoxels));
      else
        return F::avg_pool2d(layer, F::AvgPool2dFuncOptions(cellVoxels).stride(cellVoxels));
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
  std::vector<torch::Tensor>                 fixedLayers;
  std::vector<torch::Tensor>                 movingLayers;
  std::vector<torch::Tensor>                 fixedLayersOnline; // "Jacobian" mode: fixed features, pooled
  std::vector<torch::Tensor>                 pcaBasis;          // per kept layer; undefined entry = no PCA
  std::vector<std::unique_ptr<Impact::Loss>> losses;
  std::vector<float>                         layerWeights;
  // SubsetFeatures: per kept layer, that many of its channels drawn at random at every iteration (0 = all).
  std::vector<torch::Tensor> subsets; // per layer; an undefined entry keeps every channel
  auto                       drawSubsets = [&](const std::vector<torch::Tensor> & layers) {
    subsets.assign(layers.size(), torch::Tensor());
    for (size_t l = 0; l < layers.size(); ++l)
    {
      const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
      if (kept > 0 && kept < layers[l].size(1))
      {
        subsets[l] = torch::randperm(layers[l].size(1), generator, torch::kLong).narrow(0, 0, kept).to(device);
      }
    }
  };
  auto pick = [&](size_t l, const torch::Tensor & layer) -> torch::Tensor {
    return (l < subsets.size() && subsets[l].defined()) ? layer.index_select(1, subsets[l]) : layer;
  };
  // Layer l's loss between two feature maps on one grid ({1, C, spatial...}): a spatial distance (LNCC) reads the
  // maps whole (its window LNCCKernel voxels of the image along every axis), the others compare their feature vectors voxel by voxel; over the voxels `mask` keeps, when given.
  auto layerLoss = [&](size_t                l,
                       const torch::Tensor & fixedMap,
                       const torch::Tensor & movingMap,
                       const torch::Tensor & mask = {}) -> torch::Tensor {
    return Impact::MapLoss<ImageDimension>(
      *losses[l],
      fixedMap,
      movingMap,
      mask,
      losses[l]->IsSpatial() ? Impact::IsotropicWindow(spatial, fixedMap.sizes(), m_LNCCKernel)
                             : std::vector<int64_t>{});
  };
  // The masks on a grid of `size` voxels: the fixed one as bool (its share of each voxel >= 0.5), the moving one as its
  // share, to be warped; undefined when absent.
  // On the loss grid a mask's share is taken over the cells the features are pooled over.
  auto maskShareOn = [&](const torch::Tensor & mask, const std::vector<int64_t> & size) {
    return size == coarseSpatial ? poolLayerToLoss(mask.to(torch::kFloat32))
                                 : Impact::MaskShare<ImageDimension>(mask, size);
  };
  auto fixedMaskOn = [&](const std::vector<int64_t> & size) {
    return fixedMask.defined() ? maskShareOn(fixedMask, size) >= 0.5 : torch::Tensor();
  };
  auto movingShareOn = [&](const std::vector<int64_t> & size) {
    return movingMask.defined() ? maskShareOn(movingMask, size) : torch::Tensor();
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
  // by the residual field, interpolated there as gridForLayerControl's resizeField would.
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
        0, torch::randint(candidates, { count }, generator, torch::kLong).to(device));
    }
    else
    {
      drawn = torch::randint(voxels, { count }, generator, torch::kLong).to(device);
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
      const std::string name = m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)];
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
            layerBaseGrid.push_back(torch::affine_grid_generator(idAffine, gs, /*align_corners=*/false));
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
          layerBaseGrid[l] = torch::affine_grid_generator(idAffine, gs, /*align_corners=*/false);
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
    // Whole cells, the tail a size the shrink does not divide leaves dropped, as the masks and the feature layers.
    movingCoarse = poolLayerToLoss(movingT);
    fixedCoarse = poolLayerToLoss(fixedT);
    fixedCoarseMask = fixedMaskOn(coarseSpatial);
    movingCoarseShare = movingShareOn(coarseSpatial);
    std::vector<int64_t> gs{ 1, 1 };
    for (auto s : coarseSpatial)
    {
      gs.push_back(s);
    }
    gridBaseCoarse = torch::affine_grid_generator(idAffine, gs, /*align_corners=*/false);
  }

  // ---- 4. Adam loop (entirely on device; no host copies). ----
  torch::optim::Adam optimizer(
    { theta }, torch::optim::AdamOptions(m_LearningRate)
                 .betas(std::make_tuple(m_Beta1, m_Beta2))
                 // Epsilon weighs against the gradient per control cell, as in ConvexAdam, whose parameters count cells.
                 .eps(m_Epsilon / static_cast<double>(std::max(1u, m_GridShrinkFactor))));

  m_MetricValuesPerIteration.clear();
  m_MetricValuesPerIteration.reserve(m_NumberOfIterations);

  // Every layer's loss divided by its value at the stage's first iteration, so each starts at 1 and
  // LayersWeight weighs comparable quantities (see Impact::LossNormalization).
  Impact::LossNormalization normalization;
  auto                      normalized = [&](size_t l, const torch::Tensor & value) -> torch::Tensor {
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
      // The gradient per voxel: the field differentiated per control cell of shrink voxels.
      const double controlSpacing = static_cast<double>(shrink[ax]);
      reg = reg + (regField.narrow(tdim, 1, len - 1) - regField.narrow(tdim, 0, len - 1)).pow(2).mean() /
                    (controlSpacing * controlSpacing);
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
        int64_t voxels = 1;
        for (const int64_t extent : spatial)
        {
          voxels *= extent;
        }
        const int64_t count = std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(voxels)));
        const torch::Tensor flat = torch::randint(voxels, { count }, generator, torch::kLong);
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
          gridScale[d] = static_cast<double>(spatial[ImageDimension - 1 - d]) / 2.0;
        }
        // The patches are placed on the device, in single precision.
        const auto          onDevice = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const torch::Tensor upper = torch::tensor(lastIndex, torch::kFloat64).to(onDevice);
        torch::Tensor       fits = torch::ones({ count }, torch::TensorOptions().dtype(torch::kBool).device(device));
        for (size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
        {
          const ImpactModelConfiguration & config = m_FixedModelsConfiguration[i];
          const torch::Tensor              corners = Impact::SampledPatchPositions<ImageDimension>(
            config, points, fixedSpacing, m_Seed, i, Impact::PatchCorners(config), onDevice);
          fits = fits & ((corners >= 0.0) & (corners <= upper)).flatten(1).all(1);
        }
        if (fixedMask.defined()) // the points drawn in the fixed mask
        {
          fits = fits & fixedMask.flatten().index_select(0, flat.to(device));
        }
        torch::Tensor       kept = points.index_select(0, fits.nonzero().flatten().to(torch::kCPU));
        const torch::Tensor toGrid = torch::tensor(gridScale, torch::kFloat64);
        const torch::Tensor toGridOnDevice = toGrid.to(onDevice);
        if (movingMask.defined() && kept.size(0) > 0)
        {
          // ... that the moving mask, warped by the field, holds at their centre.
          torch::NoGradGuard   noGrad;
          std::vector<int64_t> where(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, n, Dim}
          where[ImageDimension] = kept.size(0);
          where[ImageDimension + 1] = ImageDimension;
          const torch::Tensor centres =
            ((kept.to(torch::kFloat64) + 0.5) / toGrid - 1.0).to(torch::kFloat32).to(device).reshape(where);
          const torch::Tensor displacement = F::grid_sample(smoothedControl.detach(), centres, controlOpts)
                                               .reshape({ static_cast<int64_t>(ImageDimension), kept.size(0) })
                                               .t();
          const torch::Tensor holds =
            F::grid_sample(movingMaskShare, centres + (displacement / imageScale).flip(-1).reshape(where), sampleOpts) >=
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
            const torch::Tensor grid = (Impact::SampledPatchPositions<ImageDimension>(m_FixedModelsConfiguration[i],
                                                                                     kept.narrow(0, begin, end - begin),
                                                                                     fixedSpacing,
                                                                                     m_Seed,
                                                                                     i,
                                                                                     {},
                                                                                     onDevice) +
                                          0.5) /
                                         toGridOnDevice -
                                       1.0;
            const int64_t        samples = grid.size(0) * grid.size(1);
            std::vector<int64_t> where(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, samples, Dim}
            where[ImageDimension] = samples;
            where[ImageDimension + 1] = ImageDimension;
            torch::Tensor at = grid.reshape(where);
            if (moving)
            {
              const torch::Tensor displacement =
                F::grid_sample(smoothed, at, controlOpts).reshape({ static_cast<int64_t>(ImageDimension), samples }).t();
              at = at + (displacement / imageScale).flip(-1).reshape(where);
            }
            std::vector<int64_t> shape{ end - begin, 1 };
            for (const int64_t extent : Impact::PatchTensorShape(m_FixedModelsConfiguration[i]))
            {
              shape.push_back(extent);
            }
            return F::grid_sample(moving ? movingT : fixedT, at, sampleOpts).reshape(shape);
          };
          // Every kept layer's centre feature vector {b, C} of a model run on `patches`, normalized, the voxel the
          // patch is centred on (size / 2 along every axis).
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
              const torch::Tensor holds = maskShareOn(warpedMoving, ls) >= 0.5;
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
              mll =
                F::interpolate(mll, F::InterpolateFuncOptions().size(tgt).mode(torch::kTrilinear).align_corners(true));
            else
              mll =
                F::interpolate(mll, F::InterpolateFuncOptions().size(tgt).mode(torch::kBilinear).align_corners(true));
          }
          torch::Tensor counted;
          if (masked)
          {
            counted = countedVoxels(fixedLayerMask[l], movingLayerShare[l], [&] {
              return layerBaseGrid[l] +
                     (resizeField(smoothControl(theta.detach()), layerSpatials[l]).permute(toChannelLast) / imageScale)
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

  Impact::WriteVoxelFieldToDisplacement<ImageDimension>(controlGridToFullField(theta.detach()),
                                                        m_FixedImage->GetSpacing(),
                                                        m_FixedImage->GetDirection(),
                                                        output);

  // Wrap the field in a ready-to-use transform.
  m_DisplacementFieldTransform = DisplacementFieldTransformType::New();
  m_DisplacementFieldTransform->SetDisplacementField(output);
  {
    torch::NoGradGuard noGrad;
    m_ControlGrid =
      Impact::BuildCellGrid<ImageDimension>(smoothControl(theta), m_FixedImage.GetPointer());
    m_ControlGridTransform = nullptr;
  }

  // ---- 6. Warped moving image on the fixed grid (for inspection). ----
  torch::Tensor wCpu = warped.detach().squeeze(0).squeeze(0).contiguous().to(torch::kCPU); // {z,y,x}
  const float * wPtr = wCpu.data_ptr<float>();

  m_WarpedMovingImage = WarpedImageType::New();
  m_WarpedMovingImage->SetRegions(m_FixedImage->GetLargestPossibleRegion());
  m_WarpedMovingImage->SetSpacing(m_FixedImage->GetSpacing());
  m_WarpedMovingImage->SetOrigin(m_FixedImage->GetOrigin());
  m_WarpedMovingImage->SetDirection(m_FixedImage->GetDirection());
  m_WarpedMovingImage->Allocate();

  ImageRegionIteratorWithIndex<WarpedImageType> wit(m_WarpedMovingImage,
                                                    m_WarpedMovingImage->GetLargestPossibleRegion());
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
auto
ImpactFineRegistration<TFixedImage, TMovingImage>::GetControlGridTransform() -> DisplacementFieldTransformType *
{
  if (m_ControlGrid.IsNull())
  {
    // Solve without allocating the dense field: request one voxel of it.
    this->UpdateOutputInformation();
    typename DisplacementFieldType::RegionType one = this->GetOutput()->GetLargestPossibleRegion();
    one.SetSize(DisplacementFieldType::SizeType::Filled(1));
    this->GetOutput()->SetRequestedRegion(one);
    this->Update();
  }
  if (m_ControlGridTransform.IsNull() || m_ControlGridTransform->GetDisplacementField() != m_ControlGrid.GetPointer())
  {
    m_ControlGridTransform = DisplacementFieldTransformType::New();
    m_ControlGridTransform->SetDisplacementField(m_ControlGrid);
  }
  return m_ControlGridTransform;
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::DynamicThreadedGenerateData(
  const typename Superclass::OutputImageRegionType & region)
{
  using InterpolatorType = VectorLinearInterpolateImageFunction<DisplacementFieldType, double>;
  auto interpolator = InterpolatorType::New();
  interpolator->SetInputImage(m_ControlGrid);
  const auto &                                        controlSize = m_ControlGrid->GetLargestPossibleRegion().GetSize();
  ImageRegionIteratorWithIndex<DisplacementFieldType> it(this->GetOutput(), region);
  typename FixedImageType::PointType                  point;
  typename InterpolatorType::ContinuousIndexType      index;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    m_FixedImage->TransformIndexToPhysicalPoint(it.GetIndex(), point);
    m_ControlGrid->TransformPhysicalPointToContinuousIndex(point, index);
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      // Round-off can put the last voxel a hair past the last control point.
      index[a] = std::clamp(index[a], 0.0, static_cast<double>(controlSize[a] - 1));
    }
    const auto u = interpolator->EvaluateAtContinuousIndex(index);
    VectorType value;
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      value[a] = static_cast<float>(u[a]);
    }
    it.Set(value);
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactFineRegistration<TFixedImage, TMovingImage>::SolveByRegions()
{
  namespace F = torch::nn::functional;
  constexpr unsigned int Dim = ImageDimension;
  using RegionType = typename FixedImageType::RegionType;
  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }
  const bool featureMode = !m_FixedModelsConfiguration.empty();
  if (featureMode && m_Mode == "Jacobian")
  {
    itkExceptionMacro("ImpactFineRegistration: RegionSize runs the Static mode; the Jacobian mode re-extracts the "
                      "features of the whole warped image.");
  }
  if (m_FeatureMapUpdateInterval > 0)
  {
    itkExceptionMacro("ImpactFineRegistration: FeatureMapUpdateInterval re-extracts the features of the whole warped "
                      "image; leave it at -1 with RegionSize.");
  }
  for (const unsigned int components : m_PCA)
  {
    if (components > 0)
    {
      itkExceptionMacro("ImpactFineRegistration: PCA fits its basis on the whole fixed feature map; set it to 0 with "
                        "RegionSize.");
    }
  }
  if (!(m_SamplingPercentage > 0.0 && m_SamplingPercentage <= 1.0))
  {
    itkExceptionMacro("ImpactFineRegistration: SamplingPercentage must lie in (0, 1], got " << m_SamplingPercentage
                                                                                            << ".");
  }

  const torch::Device device(m_Device);
  torch::manual_seed(m_Seed);
  const Impact::PythonGilReleaseGuard gilRelease;
  // What grows with the image when the pooled maps are not resident; declared before the tensors it maps.
  Impact::ScratchFiles scratch(m_ScratchDirectory);
  m_ScratchBytes = 0;
  m_BytesRead = 0;
  const auto                          onDevice = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto                          sampleOpts =
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);
  const auto nearestOpts =
    F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(true);

  // ---- Geometry. A reader's output knows its grid once its information is updated; nothing is read yet. ----
  auto * fixedSource = const_cast<FixedImageType *>(m_FixedImage.GetPointer());
  auto * movingSource = const_cast<MovingImageType *>(m_MovingImage.GetPointer());
  fixedSource->UpdateOutputInformation();
  movingSource->UpdateOutputInformation();
  const RegionType                           fixedRegion = m_FixedImage->GetLargestPossibleRegion();
  const typename MovingImageType::RegionType movingRegion = m_MovingImage->GetLargestPossibleRegion();
  std::vector<int64_t>                       spatial(Dim); // tensor order (z, y, x)
  for (unsigned int d = 0; d < Dim; ++d)
  {
    spatial[d] = static_cast<int64_t>(fixedRegion.GetSize()[Dim - 1 - d]);
  }
  const std::vector<int64_t>       shrink(Dim, static_cast<int64_t>(std::max(1u, m_GridShrinkFactor)));
  std::vector<int64_t>             cells(Dim);
  std::vector<float>               cellsPerVoxel(Dim); // a voxel of displacement in cells, by ConvexAdam's scale
  std::vector<Impact::PoolWindows> windows(Dim);       // the voxels each cell pools
  int64_t                          cellCountTotal = 1;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    cells[d] = std::max<int64_t>(1, spatial[d] / shrink[d]);
    cellCountTotal *= cells[d];
    // The dense run divides the field by shrink * (cells - 1) / 2 voxels a grid_sample unit of cells / 2 cells.
    cellsPerVoxel[d] = cells[d] > 1 ? static_cast<float>(static_cast<double>(cells[d]) /
                                                         (static_cast<double>(shrink[d]) * static_cast<double>(cells[d] - 1)))
                                    : 0.0f;
    windows[d] = Impact::UniformPoolWindows(cells[d], std::min(shrink[d], spatial[d]));
  }
  std::vector<int64_t> componentShape(Dim + 2, 1);
  componentShape[1] = static_cast<int64_t>(Dim);
  const torch::Tensor voxelsToCells = torch::tensor(cellsPerVoxel, torch::kFloat32).view(componentShape).to(device);
  const std::vector<int64_t> regionVoxels(Dim, static_cast<int64_t>(m_RegionSize));
  std::vector<int64_t>       perBox(Dim);
  for (unsigned int d = 0; d < Dim; ++d)
  {
    perBox[d] =
      std::max<int64_t>(1, std::llround(static_cast<double>(regionVoxels[d]) / static_cast<double>(shrink[d])));
  }
  const std::vector<Impact::CellBox> boxes = Impact::TileCells(cells, perBox);
  // A model's receptive field is in its input's voxels, on every axis whatever their spacing.
  const std::vector<int64_t> halo(Dim, static_cast<int64_t>(m_RegionHalo));
  // The cells' positions, {1, Dim, n...}, for the cells [c0, c1).
  auto cellPositions = [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1) {
    std::vector<torch::Tensor> axes;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      axes.push_back(torch::arange(c0[d], c1[d], onDevice));
    }
    return torch::stack(torch::meshgrid(axes, "ij"), 0).unsqueeze(0);
  };
  // The sampling grid of the positions `q` ({1, Dim, n...}, in cells) in a map over the cells [m0, m1):
  // normalized to that window, (x, y, z) last.
  auto gridIn = [&](const torch::Tensor & q, const std::vector<int64_t> & m0, const std::vector<int64_t> & m1) {
    std::vector<torch::Tensor> components;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      const int64_t extent = m1[d] - m0[d];
      components.push_back(extent > 1 ? (q.select(1, static_cast<int64_t>(d)) - static_cast<double>(m0[d])) *
                                            (2.0 / static_cast<double>(extent - 1)) -
                                          1.0
                                      : torch::zeros_like(q.select(1, static_cast<int64_t>(d))));
    }
    return torch::stack(components, -1).flip(-1); // {1, n..., Dim}
  };

  // ---- Masks, read by regions on their own image's grid. ----
  auto sameGrid = [](const auto * a, const auto * b) { return Impact::SameGrid(a, b); };
  if (m_FixedMask.IsNotNull())
  {
    const_cast<MaskImageType *>(m_FixedMask.GetPointer())->UpdateOutputInformation();
    if (!sameGrid(m_FixedMask.GetPointer(), m_FixedImage.GetPointer()))
    {
      itkExceptionMacro("ImpactFineRegistration: with RegionSize the fixed mask is read by regions on the fixed "
                        "image's grid; resample it there first.");
    }
  }
  if (m_MovingMask.IsNotNull())
  {
    const_cast<MaskImageType *>(m_MovingMask.GetPointer())->UpdateOutputInformation();
    if (!sameGrid(m_MovingMask.GetPointer(), m_MovingImage.GetPointer()))
    {
      itkExceptionMacro("ImpactFineRegistration: with RegionSize the moving mask is read by regions on the moving "
                        "image's grid; resample it there first.");
    }
  }

  // ---- Reads: each through the image's own pipeline, counted, telling whether the input streams. ----
  auto readFixed = [&](const RegionType & region) {
    return Impact::ReadRegionCounted(
      m_FixedImage.GetPointer(), region, device, m_BytesRead, m_FixedImage.GetPointer(), m_FixedImageStreamed);
  };
  auto readMoving = [&](const typename MovingImageType::RegionType & region) {
    return Impact::ReadRegionCounted(
      m_MovingImage.GetPointer(), region, device, m_BytesRead, m_MovingImage.GetPointer(), m_MovingImageStreamed);
  };
  auto readMask = [&](const MaskImageType * mask, const typename MaskImageType::RegionType & region) {
    m_BytesRead += region.GetNumberOfPixels();
    return (Impact::ReadRegion(mask, region) != 0).to(torch::kFloat32).to(device);
  };

  // ---- The moving transform, and where a fixed voxel reads the moving image. ----
  const bool identityMap =
    m_MovingTransform.IsNull() && sameGrid(m_MovingImage.GetPointer(), m_FixedImage.GetPointer());
  const bool linearMap = m_MovingTransform.IsNull() ||
                         m_MovingTransform->GetTransformCategory() == TransformType::TransformCategoryEnum::Linear;
  // The moving continuous index (tensor order) at every voxel of `window`, a region of the fixed grid: {1, Dim,
  // window...} on the device. Evaluated at the corners of a lattice of at most four voxels a step (the window's
  // corners alone for a linear map, which the interpolation then reproduces exactly) and interpolated linearly.
  auto movingIndexField = [&](const RegionType & window) -> torch::Tensor {
    std::vector<int64_t> size(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      size[d] = static_cast<int64_t>(window.GetSize()[Dim - 1 - d]);
    }
    if (identityMap)
    {
      std::vector<int64_t> start(Dim), stop(Dim);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        start[d] = static_cast<int64_t>(window.GetIndex()[Dim - 1 - d]);
        stop[d] = start[d] + size[d];
      }
      return cellPositions(start, stop);
    }
    std::vector<int64_t> knots(Dim);
    int64_t              total = 1;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      knots[d] = linearMap ? 2 : std::max<int64_t>(2, (size[d] - 1 + 3) / 4 + 1);
      total *= knots[d];
    }
    std::vector<float> buffer(static_cast<size_t>(total * Dim)); // {Dim, knots...}, last axis fastest
    for (int64_t flat = 0; flat < total; ++flat)
    {
      int64_t                      rest = flat;
      ContinuousIndex<double, Dim> fixedIndex;
      for (int d = static_cast<int>(Dim) - 1; d >= 0; --d)
      {
        const int64_t k = rest % knots[d];
        rest /= knots[d];
        const double t =
          size[d] > 1 ? static_cast<double>(k) * static_cast<double>(size[d] - 1) / static_cast<double>(knots[d] - 1)
                      : 0.0;
        fixedIndex[Dim - 1 - d] = static_cast<double>(window.GetIndex()[Dim - 1 - d]) + t;
      }
      typename FixedImageType::PointType point;
      m_FixedImage->TransformContinuousIndexToPhysicalPoint(fixedIndex, point);
      if (m_MovingTransform.IsNotNull())
      {
        point = m_MovingTransform->TransformPoint(point);
      }
      ContinuousIndex<double, Dim> movingIndex;
      m_MovingImage->TransformPhysicalPointToContinuousIndex(point, movingIndex);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        buffer[static_cast<size_t>(d) * total + flat] = static_cast<float>(movingIndex[Dim - 1 - d]);
      }
    }
    std::vector<int64_t> shape{ 1, static_cast<int64_t>(Dim) };
    shape.insert(shape.end(), knots.begin(), knots.end());
    const torch::Tensor lattice = torch::from_blob(buffer.data(), shape, torch::kFloat32).clone().to(device);
    if (knots == size)
    {
      return lattice;
    }
    if constexpr (Dim == 3)
      return F::interpolate(lattice,
                            F::InterpolateFuncOptions().size(size).mode(torch::kTrilinear).align_corners(true));
    else
      return F::interpolate(lattice, F::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(true));
  };

  // The full-resolution span of the cells [c0, c1), widened by `extra` voxels along each axis within the image;
  // `lowPad` is the widening the low side got.
  auto spanOf = [&](const std::vector<int64_t> & c0,
                    const std::vector<int64_t> & c1,
                    const std::vector<int64_t> & extra,
                    std::vector<int64_t> &       lowPad,
                    int64_t                      align = 1) {
    return Impact::CellSpan<RegionType>(c0, c1, extra, windows, spatial, fixedRegion.GetIndex(), lowPad, align);
  };
  // A layer of such a span ({1, C, samples...}, a mask: the image's lattice) pooled over the cells [c0, c1).
  auto toCells = [&](const torch::Tensor &        layer,
                     const Impact::LayerLattice & lattice,
                     const std::vector<int64_t> & lowPad,
                     const std::vector<int64_t> & c0,
                     const std::vector<int64_t> & c1) {
    return Impact::PoolLayerToCells<Dim>(layer, lattice, lowPad, c0, c1, windows, spatial, "ImpactFineRegistration");
  };
  const auto writeCells = Impact::WriteCells;
  const auto cellsOf = Impact::CellsOf;

  // ---- The feature models and distances. ----
  std::vector<std::unique_ptr<Impact::Loss>> losses;
  std::vector<float>                         layerWeights;
  bool                                       takesMetadata = false; // a model that reads the image's statistics
  if (featureMode)
  {
    const Impact::RegionModels models =
      Impact::CheckRegionModels(m_FixedModelsConfiguration, m_FixedImage->GetSpacing(), Dim, "ImpactFineRegistration");
    const size_t layerCount = models.layerCount;
    takesMetadata = models.takesMetadata;
    for (size_t l = 0; l < layerCount; ++l)
    {
      const std::string name = m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)];
      losses.push_back(Impact::LossFactory::Instance().Create(name));
      if (losses.back()->IsSpatial())
      {
        itkExceptionMacro("ImpactFineRegistration: distance '" << name
                                                               << "' correlates windows of whole maps; RegionSize "
                                                                  "compares the features cell by cell.");
      }
      if (!losses.back()->IsPerPointMean())
      {
        itkExceptionMacro("ImpactFineRegistration: distance '"
                          << name
                          << "' is a statistic over all the points, which the boxes of RegionSize do not have.");
      }
      layerWeights.push_back(l < m_LayersWeight.size() ? m_LayersWeight[l] : 1.0f);
    }
  }
  else
  {
    // The intensity similarity: the image is its own single feature, compared by its mean squared difference.
    losses.push_back(Impact::LossFactory::Instance().Create("L2"));
    layerWeights.push_back(1.0f);
  }
  const std::vector<ImpactModelConfiguration> & movingConfigs =
    m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
  // A metadata-aware model (MIND reads the image's range) takes the image's statistics: one pass over each image, all
  // its voxels, in blocks of the box size, as the dense run's StatisticsImageFilter takes them (the moving ones on
  // its own grid, where the dense run reads it resampled onto the fixed grid).
  ImpactImageMetadata fixedMetadata, movingMetadata;
  if (takesMetadata)
  {
    auto statistics = [&](const RegionType & whole, const auto & direction, auto && read) {
      return Impact::MetadataByBlocks(direction, Dim, [&](auto && take) {
        std::vector<int64_t> extent(Dim), step(Dim), counts(Dim);
        size_t               total = 1;
        for (unsigned int d = 0; d < Dim; ++d)
        {
          extent[d] = static_cast<int64_t>(whole.GetSize()[Dim - 1 - d]);
          step[d] = std::max<int64_t>(1, std::min(extent[d], regionVoxels[d]));
          counts[d] = (extent[d] + step[d] - 1) / step[d];
          total *= static_cast<size_t>(counts[d]);
        }
        for (size_t flat = 0; flat < total; ++flat)
        {
          RegionType block;
          size_t     rest = flat;
          for (int d = static_cast<int>(Dim) - 1; d >= 0; --d)
          {
            const int64_t i = static_cast<int64_t>(rest % static_cast<size_t>(counts[d]));
            rest /= static_cast<size_t>(counts[d]);
            const int64_t start = i * step[d];
            block.SetIndex(Dim - 1 - d, whole.GetIndex()[Dim - 1 - d] + static_cast<IndexValueType>(start));
            block.SetSize(Dim - 1 - d, static_cast<SizeValueType>(std::min(step[d], extent[d] - start)));
          }
          take(read(block));
        }
      });
    };
    fixedMetadata = statistics(fixedRegion, m_FixedImage->GetDirection(), readFixed);
    movingMetadata = statistics(movingRegion, m_MovingImage->GetDirection(), readMoving);
  }
  // The feature layers of an image region {1, 1, span...}, at its resolution, each model run on the region whole in
  // one forward (a region is sized to fit the device, so the tiler ExtractFeatureLayers measures the device for at
  // every call has nothing to cut); the region itself in intensity mode.
  auto layersOf = [&](const torch::Tensor & image, bool moving) {
    return Impact::RegionLayers<Dim>(featureMode ? (moving ? movingConfigs : m_FixedModelsConfiguration)
                                                 : std::vector<ImpactModelConfiguration>{},
                                     image,
                                     device,
                                     losses.size(),
                                     takesMetadata ? &(moving ? movingMetadata : fixedMetadata) : nullptr,
                                     "ImpactFineRegistration");
  };
  // A layer coarser than the image lies on the image's lattice every `align` voxels: the regions read for the
  // features start on it, and reach what pooling the whole layer to the cells reads past a cell's own voxels.
  int64_t                           align = 1;
  std::vector<int64_t>              reach = halo;
  std::vector<Impact::LayerLattice> lattices(losses.size(), Impact::ImageLattice(Dim));
  std::vector<Impact::LayerLattice> movingLattices = lattices; // the moving models' own, when they differ
  if (featureMode)
  {
    std::vector<int64_t> probe(Dim + 2, Impact::LatticeProbe);
    probe[0] = probe[1] = 1;
    lattices = Impact::LayerLattices<Dim>(layersOf(torch::zeros(probe, onDevice), false), "ImpactFineRegistration");
    movingLattices =
      m_MovingModelsConfiguration.empty()
        ? lattices
        : Impact::LayerLattices<Dim>(layersOf(torch::zeros(probe, onDevice), true), "ImpactFineRegistration");
    for (const auto * side : { &lattices, &movingLattices })
    {
      for (const Impact::LayerLattice & lattice : *side)
      {
        align = std::max(align, *std::max_element(lattice.stride.begin(), lattice.stride.end()));
      }
    }
    for (unsigned int d = 0; d < Dim && align > 1; ++d)
    {
      reach[d] += windows[d].ends[0] - windows[d].starts[0] + 2 * align;
    }
  }
  auto pooledToCells = [&](std::vector<torch::Tensor> &              layers,
                           const std::vector<Impact::LayerLattice> & on,
                           const std::vector<int64_t> &              lowPad,
                           const std::vector<int64_t> &              c0,
                           const std::vector<int64_t> &              c1) {
    for (size_t l = 0; l < layers.size(); ++l)
    {
      layers[l] = toCells(layers[l], on[l], lowPad, c0, c1);
    }
  };
  // The fixed features pooled over the cells [c0, c1): {1, C, n...} per layer.
  auto pooledFixed = [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1) {
    std::vector<int64_t>       lowPad(Dim);
    std::vector<torch::Tensor> layers = layersOf(readFixed(spanOf(c0, c1, reach, lowPad, align)), false);
    pooledToCells(layers, lattices, lowPad, c0, c1);
    return layers;
  };
  // The moving features pooled over the cells [c0, c1) of the fixed grid, the moving image brought there through
  // the moving transform (the moving mask with it, as its share of each cell): {1, C, n...} per layer, and the share
  // {1, 1, n...} (undefined without a moving mask).
  struct PooledMoving
  {
    std::vector<torch::Tensor> layers;
    torch::Tensor              share;
  };
  auto pooledMoving = [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1) -> PooledMoving {
    std::vector<int64_t> lowPad(Dim);
    const RegionType     span = spanOf(c0, c1, reach, lowPad, align);
    const torch::Tensor  movingIndex = movingIndexField(span); // {1, Dim, span...} on the device
    std::vector<int64_t> spanSize(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      spanSize[d] = static_cast<int64_t>(span.GetSize()[Dim - 1 - d]);
    }
    // The region of the moving image those indices read, one voxel around for the interpolation.
    typename MovingImageType::RegionType movingRead;
    std::vector<int64_t>                 readStart(Dim), readSize(Dim);
    bool                                 outside = false;
    const torch::Tensor                  bounds =
      torch::stack({ std::get<0>(movingIndex.flatten(2).min(2)), std::get<0>(movingIndex.flatten(2).max(2)) }, 0)
        .to(torch::kCPU); // {2, 1, Dim}
    for (unsigned int d = 0; d < Dim; ++d)
    {
      const auto    low = static_cast<int64_t>(std::floor(bounds[0][0][d].template item<double>())) - 1;
      const auto    high = static_cast<int64_t>(std::ceil(bounds[1][0][d].template item<double>())) + 1;
      const int64_t first = movingRegion.GetIndex()[Dim - 1 - d];
      const int64_t extent = static_cast<int64_t>(movingRegion.GetSize()[Dim - 1 - d]);
      const int64_t start = std::max(low, first);
      const int64_t end = std::min(high + 1, first + extent);
      outside = outside || end <= start;
      readStart[d] = start;
      readSize[d] = std::max<int64_t>(end - start, 1);
      movingRead.SetIndex(Dim - 1 - d, static_cast<IndexValueType>(start));
      movingRead.SetSize(Dim - 1 - d, static_cast<SizeValueType>(readSize[d]));
    }
    std::vector<int64_t> windowShape{ 1, 1 };
    windowShape.insert(windowShape.end(), spanSize.begin(), spanSize.end());
    torch::Tensor window, maskWindow;
    if (outside)
    {
      window = torch::zeros(windowShape, onDevice);
      if (m_MovingMask.IsNotNull())
      {
        maskWindow = window;
      }
    }
    else
    {
      std::vector<int64_t> readStop(Dim);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        readStop[d] = readStart[d] + readSize[d];
      }
      const torch::Tensor grid = gridIn(movingIndex, readStart, readStop);
      window = F::grid_sample(readMoving(movingRead), grid, sampleOpts);
      if (m_MovingMask.IsNotNull())
      {
        maskWindow = F::grid_sample(readMask(m_MovingMask.GetPointer(), movingRead), grid, nearestOpts);
      }
    }
    PooledMoving pooled;
    pooled.layers = layersOf(window, true);
    pooledToCells(pooled.layers, movingLattices, lowPad, c0, c1);
    if (maskWindow.defined())
    {
      pooled.share = toCells(maskWindow, Impact::ImageLattice(Dim), lowPad, c0, c1);
    }
    return pooled;
  };
  // SubsetFeatures: per kept layer, that many of its channels drawn at random at every iteration (0 = all).
  std::vector<torch::Tensor> subsets(losses.size());
  auto                       drawSubsets = [&](const std::vector<int64_t> & channels) {
    for (size_t l = 0; l < losses.size(); ++l)
    {
      const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
      subsets[l] =
        (kept > 0 && kept < channels[l])
                                ? torch::randperm(channels[l], torch::TensorOptions().dtype(torch::kLong)).narrow(0, 0, kept).to(device)
                                : torch::Tensor();
    }
  };
  auto pick = [&](size_t l, const torch::Tensor & layer) -> torch::Tensor {
    return subsets[l].defined() ? layer.index_select(1, subsets[l]) : layer;
  };
  // Every layer's loss divided by its value at the first iteration (feature mode; the intensity path is unscaled).
  const bool                normalizes = featureMode && m_NormalizeLosses;
  Impact::LossNormalization normalization;
  auto                      normalized = [&](size_t l, const torch::Tensor & value) -> torch::Tensor {
    return normalizes ? value * normalization.Factor(l) : value;
  };
  // Layer l's loss between the rows `rows` of two pooled maps {1, C, n...} (every row when undefined).
  auto rowsLoss = [&](size_t l, const torch::Tensor & fixed, const torch::Tensor & moving, const torch::Tensor & rows) {
    const int64_t channels = moving.size(1);
    torch::Tensor f = fixed.reshape({ channels, -1 }), m = moving.reshape({ channels, -1 });
    if (rows.defined())
    {
      f = f.index_select(1, rows);
      m = m.index_select(1, rows);
    }
    return losses[l]->forwardValue(f.t(), m.t());
  };
  // The rows that count among `counted` (bool {1, 1, n...}; every row when undefined), a share of them drawn at
  // random with SamplingPercentage < 1.
  auto drawRows = [&](const torch::Tensor & counted, int64_t count) -> torch::Tensor {
    torch::Tensor rows;
    if (counted.defined())
    {
      rows = counted.flatten().nonzero().flatten();
    }
    else if (m_SamplingPercentage < 1.0)
    {
      rows = torch::arange(count, torch::TensorOptions().dtype(torch::kLong).device(device));
    }
    if (m_SamplingPercentage < 1.0 && rows.numel() > 0)
    {
      const int64_t drawn =
        std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(rows.numel())));
      rows = rows.index_select(
        0, torch::randint(rows.numel(), { drawn }, torch::TensorOptions().dtype(torch::kLong).device(device)));
    }
    return rows;
  };

  // ---- The route: the pooled feature maps of both images, computed once by one pass over the boxes, are compared
  // whole at every iteration when they fit the device (the dense run on them), brick by brick when they do not. What
  // grows with the image is then at home on the device, or in mapped files of which an iteration brings the cells of
  // a brick at a time to the device. ----
  std::vector<int64_t> channels(losses.size(), 1);
  if (featureMode)
  {
    const std::vector<torch::Tensor> probe = pooledFixed(boxes[0].cell0, boxes[0].cell1);
    for (size_t l = 0; l < probe.size(); ++l)
    {
      channels[l] = probe[l].size(1);
    }
  }
  int64_t channelCount = 0;
  for (const int64_t c : channels)
  {
    channelCount += c;
  }
  const double residentBytes = static_cast<double>(cellCountTotal) * static_cast<double>(channelCount) * 4.0 * 2.0 +
                               (m_MovingMask.IsNotNull() ? static_cast<double>(cellCountTotal) * 4.0 : 0.0);
  double budget = static_cast<double>(m_ResidentFeatureBudget);
  if (m_ResidentFeatureBudget == 0)
  {
    // Of the memory left once the control grid and Adam's state, resident with the maps, are on the device.
    budget = Impact::ResidentBudget(device) -
             0.5 * static_cast<double>(cellCountTotal) * static_cast<double>(3 * Dim + 1) * 4.0;
  }
  m_ResidentFeatures = residentBytes <= budget;
  auto atHome = [&](const std::vector<int64_t> & shape) {
    return m_ResidentFeatures ? torch::zeros(shape, onDevice) : scratch.Zeros(shape);
  };

  // ---- Occupancy: each box's cells in the fixed mask (share of their voxels >= 0.5), all of them without one; at
  // home like the maps. ----
  const std::vector<int64_t> noHalo(Dim, 0);
  std::vector<int64_t>       countByBox(boxes.size());
  std::vector<size_t>        occupied;
  double                     totalCount = 0.0;
  std::vector<int64_t>       maskShape{ 1, 1 };
  maskShape.insert(maskShape.end(), cells.begin(), cells.end());
  torch::Tensor fixedOn; // bool {1, 1, cells...}; undefined without a fixed mask
  if (m_FixedMask.IsNotNull())
  {
    fixedOn = m_ResidentFeatures ? torch::zeros(maskShape, torch::TensorOptions().dtype(torch::kBool).device(device))
                                 : scratch.Zeros(maskShape, torch::kBool);
  }
  for (size_t b = 0; b < boxes.size(); ++b)
  {
    const Impact::CellBox & box = boxes[b];
    int64_t                 count = 1;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      count *= box.cell1[d] - box.cell0[d];
    }
    if (fixedOn.defined())
    {
      std::vector<int64_t> lowPad(Dim);
      const RegionType     span = spanOf(box.cell0, box.cell1, noHalo, lowPad);
      const torch::Tensor  on = toCells(readMask(m_FixedMask.GetPointer(), span),
                                       Impact::ImageLattice(Dim),
                                       lowPad,
                                       box.cell0,
                                       box.cell1) >= 0.5;
      writeCells(fixedOn, box.cell0, on);
      count = on.sum().template item<int64_t>();
    }
    countByBox[b] = count;
    if (count > 0)
    {
      occupied.push_back(b);
      totalCount += static_cast<double>(count);
    }
  }
  if (occupied.empty())
  {
    itkExceptionMacro("ImpactFineRegistration: the fixed mask holds no voxel.");
  }

  // ---- The control grid {1, Dim, cells...} in voxels, warm-started from the initial field read at the
  // control points (any grid), else zero. ----
  std::vector<int64_t> thetaShape{ 1, static_cast<int64_t>(Dim) };
  thetaShape.insert(thetaShape.end(), cells.begin(), cells.end());
  torch::Tensor       theta = atHome(thetaShape);
  std::vector<double> fixedSpacing(Dim);
  for (unsigned int a = 0; a < Dim; ++a)
  {
    fixedSpacing[a] = m_FixedImage->GetSpacing()[a];
  }
  if (m_InitialDisplacementField.IsNotNull())
  {
    m_InitialDisplacementField->Update();
    using FieldInterpolatorType = VectorLinearInterpolateImageFunction<DisplacementFieldType, double>;
    auto interpolator = FieldInterpolatorType::New();
    interpolator->SetInputImage(m_InitialDisplacementField);
    // Filled in host memory: the mapped grid itself, or a grid the device then takes.
    torch::Tensor start = m_ResidentFeatures ? torch::zeros(thetaShape, torch::kFloat32) : theta;
    float * const buffer = start.data_ptr<float>();
    const auto &  direction = m_FixedImage->GetDirection();
    for (int64_t flat = 0; flat < cellCountTotal; ++flat)
    {
      // Resizing the dense field to the control grid blends, along each axis, the two voxels around the cell's
      // centre. The field is read at those voxels: one given on a coarser grid is then its dense field, resized.
      int64_t                     rest = flat;
      std::array<int64_t, Dim>    below{};
      std::array<double, Dim>     share{};
      for (int d = static_cast<int>(Dim) - 1; d >= 0; --d)
      {
        const int64_t c = rest % cells[d];
        rest /= cells[d];
        const double centre = std::max(
          (static_cast<double>(c) + 0.5) * static_cast<double>(spatial[d]) / static_cast<double>(cells[d]) - 0.5, 0.0);
        below[Dim - 1 - d] = std::min(static_cast<int64_t>(centre), spatial[d] - 1);
        share[Dim - 1 - d] = centre - static_cast<double>(below[Dim - 1 - d]);
      }
      Vector<double, Dim> u; // millimetres, ITK order
      u.Fill(0.0);
      bool inside = true;
      for (unsigned int corner = 0; inside && corner < (1u << Dim); ++corner)
      {
        double                          weight = 1.0;
        typename FixedImageType::IndexType voxel;
        for (unsigned int a = 0; a < Dim; ++a)
        {
          const bool above = (corner >> a) & 1u;
          weight *= above ? share[a] : 1.0 - share[a];
          voxel[a] = fixedRegion.GetIndex()[a] +
                     std::min(below[a] + (above ? 1 : 0), static_cast<int64_t>(fixedRegion.GetSize()[a]) - 1);
        }
        if (weight == 0.0)
        {
          continue;
        }
        typename FixedImageType::PointType point;
        m_FixedImage->TransformIndexToPhysicalPoint(voxel, point);
        inside = interpolator->IsInsideBuffer(point);
        if (inside)
        {
          const auto value = interpolator->Evaluate(point);
          for (unsigned int r = 0; r < Dim; ++r)
          {
            u[r] += weight * static_cast<double>(value[r]);
          }
        }
      }
      if (!inside)
      {
        continue;
      }
      for (unsigned int d = 0; d < Dim; ++d)
      {
        const unsigned int a = Dim - 1 - d;
        double             voxels = 0.0;
        for (unsigned int r = 0; r < Dim; ++r)
        {
          voxels += direction[r][a] * static_cast<double>(u[r]);
        }
        voxels /= fixedSpacing[a];
        buffer[static_cast<size_t>(d) * cellCountTotal + flat] = static_cast<float>(voxels);
      }
    }
    theta = m_ResidentFeatures ? start.to(device) : start;
  }
  // Epsilon weighs against the gradient per control cell, as in the dense run.
  const double     adamEpsilon = m_Epsilon / static_cast<double>(std::max(1u, m_GridShrinkFactor));
  auto             smoothControl = [&](const torch::Tensor & control) -> torch::Tensor {
    torch::Tensor s = control;
    for (unsigned int k = 0; k < m_ControlGridSmoothingIterations; ++k)
    {
      if constexpr (Dim == 3)
        s = F::avg_pool3d(s, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
      else
        s = F::avg_pool2d(s, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    }
    return s;
  };
  // The diffusion regularizer over the differences along `ax` between `first` and `second` ({1, Dim, ...}, the
  // smoothed field at consecutive cells), over the count the dense mean divides by. `smoothed` holds the cells from
  // `origin` on.
  std::vector<double> differenceCount(Dim), differenceScale(Dim);
  for (unsigned int ax = 0; ax < Dim; ++ax)
  {
    differenceCount[ax] = static_cast<double>(Dim) * static_cast<double>(std::max<int64_t>(cells[ax] - 1, 1));
    for (unsigned int d = 0; d < Dim; ++d)
    {
      if (d != ax)
      {
        differenceCount[ax] *= static_cast<double>(cells[d]);
      }
    }
    const double controlSpacing = static_cast<double>(shrink[ax]);
    differenceScale[ax] = m_RegularizationWeight / (controlSpacing * controlSpacing);
  }
  auto regularizer = [&](const torch::Tensor &        smoothed,
                         const std::vector<int64_t> & c0,
                         const std::vector<int64_t> & c1,
                         const std::vector<int64_t> & origin) {
    torch::Tensor penalty = torch::zeros({}, onDevice);
    for (unsigned int ax = 0; ax < Dim; ++ax)
    {
      const int64_t len = cells[ax];
      const int64_t a0 = c0[ax];
      const int64_t a1 = std::min(c1[ax], len - 1); // the differences whose first cell lies in [c0, c1)
      if (len < 2 || a1 <= a0)
      {
        continue;
      }
      torch::Tensor along = smoothed;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        if (d != ax)
        {
          along = along.narrow(2 + static_cast<int64_t>(d), c0[d] - origin[d], c1[d] - c0[d]);
        }
      }
      const int64_t       tdim = 2 + static_cast<int64_t>(ax);
      const torch::Tensor first = along.narrow(tdim, a0 - origin[ax], a1 - a0);
      const torch::Tensor second = along.narrow(tdim, a0 + 1 - origin[ax], a1 - a0);
      penalty = penalty + (second - first).pow(2).sum() * (differenceScale[ax] / differenceCount[ax]);
    }
    return penalty;
  };
  const std::vector<int64_t> wholeStart(Dim, 0);

  // ---- One pass over the boxes, the only time the images are read: the pooled maps, on the device when they fit
  // its budget and in mapped files otherwise. ----
  std::vector<torch::Tensor> fixedMaps(losses.size()), movingMaps(losses.size());
  torch::Tensor              movingShare;
  for (size_t l = 0; l < losses.size(); ++l)
  {
    std::vector<int64_t> shape{ 1, channels[l] };
    shape.insert(shape.end(), cells.begin(), cells.end());
    fixedMaps[l] = atHome(shape);
    movingMaps[l] = atHome(shape);
  }
  if (m_MovingMask.IsNotNull())
  {
    movingShare = atHome(maskShape);
  }
  for (size_t b = 0; b < boxes.size(); ++b)
  {
    const Impact::CellBox & box = boxes[b];
    if (countByBox[b] > 0)
    {
      const std::vector<torch::Tensor> layers = pooledFixed(box.cell0, box.cell1);
      for (size_t l = 0; l < layers.size(); ++l)
      {
        writeCells(fixedMaps[l], box.cell0, layers[l]);
      }
    }
    // The moving map everywhere: the displacement reads it past the fixed mask.
    PooledMoving pooled = pooledMoving(box.cell0, box.cell1);
    for (size_t l = 0; l < pooled.layers.size(); ++l)
    {
      writeCells(movingMaps[l], box.cell0, pooled.layers[l]);
    }
    if (pooled.share.defined())
    {
      writeCells(movingShare, box.cell0, pooled.share);
    }
  }
  if (m_ResidentFeatures)
  {
    // ---- The dense iterations on the pooled maps. ----
    theta.set_requires_grad(true);
    Impact::CellAdam    adam{ torch::zeros_like(theta), torch::zeros_like(theta), m_LearningRate, m_Beta1, m_Beta2,
                           adamEpsilon };
    const torch::Tensor base = cellPositions(wholeStart, cells);
    m_MetricValuesPerIteration.clear();
    m_MetricValuesPerIteration.reserve(m_NumberOfIterations);
    for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
    {
      drawSubsets(channels);
      const torch::Tensor smoothed = smoothControl(theta);
      const torch::Tensor q = base + smoothed * voxelsToCells;
      const torch::Tensor grid = gridIn(q, wholeStart, cells);
      torch::Tensor       counted = fixedOn;
      if (movingShare.defined())
      {
        torch::NoGradGuard  noGrad;
        const torch::Tensor holds = F::grid_sample(movingShare, grid.detach(), sampleOpts) >= 0.5;
        counted = counted.defined() ? counted.logical_and(holds) : holds;
      }
      const torch::Tensor rows = drawRows(counted, cellCountTotal);
      torch::Tensor       total = torch::zeros({}, onDevice);
      if (!rows.defined() || rows.numel() > 0)
      {
        for (size_t l = 0; l < losses.size(); ++l)
        {
          const torch::Tensor moving = F::grid_sample(pick(l, movingMaps[l]), grid, sampleOpts);
          const torch::Tensor value = rowsLoss(l, pick(l, fixedMaps[l]), moving, rows);
          if (normalizes && !normalization.IsLatched(l))
          {
            normalization.Latch(l, value.template item<double>());
          }
          total = total + layerWeights[l] * normalized(l, value);
        }
      }
      const torch::Tensor loss = total + regularizer(smoothed, wholeStart, cells, wholeStart);
      if (loss.requires_grad()) // a constant when no cell counts on a grid without a difference
      {
        loss.backward();
      }
      if (theta.grad().defined())
      {
        adam.Advance();
        adam.StepCells(theta, theta.grad(), wholeStart, cells, device);
        theta.grad().zero_();
      }
      m_MetricValuesPerIteration.push_back(loss.template item<double>());
    }
  }
  else
  {
    // ---- By bricks: an iteration steps every cell, the whole-image run's schedule, a brick of boxes at a time. A
    // brick's loss is the dense loss restricted to its cells, weighed by its share of the counted cells, plus the
    // regularizer's differences whose first cell lies in it. An iteration brings to the device the cells its bricks
    // touch, one brick at a time: the control grid, its gradient and Adam's moments are mapped like the maps. ----
    torch::Tensor    gradient = scratch.Zeros(thetaShape);
    Impact::CellAdam adam{ scratch.Zeros(thetaShape), scratch.Zeros(thetaShape), m_LearningRate, m_Beta1, m_Beta2,
                           adamEpsilon };
    auto             cellCountOf = [&](const Impact::CellBox & box) {
      int64_t count = 1;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        count *= box.cell1[d] - box.cell0[d];
      }
      return count;
    };
    // As many boxes a side as BrickBytes holds: a cell brings its share of the mapped files, the graph on a brick's
    // cells taking a few times that.
    const double brickBytes =
      m_BrickBytes > 0 ? static_cast<double>(m_BrickBytes) : Impact::ResidentBudget(device) / 8.0;
    const double held = brickBytes * static_cast<double>(cellCountTotal) / static_cast<double>(scratch.GetBytes());
    auto         brickCells = [&](int64_t boxesASide) {
      double count = 1.0;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        count *= static_cast<double>(std::min(cells[d], boxesASide * perBox[d]));
      }
      return count;
    };
    int64_t boxesASide = 1;
    while (brickCells(boxesASide) < static_cast<double>(cellCountTotal) && brickCells(boxesASide + 1) <= held)
    {
      ++boxesASide;
    }
    std::vector<int64_t> perBrick(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      perBrick[d] = boxesASide * perBox[d];
    }
    const std::vector<Impact::CellBox> bricks = Impact::TileCells(cells, perBrick);
    std::vector<int64_t>               countByBrick; // the cells of each the fixed mask counts
    for (const Impact::CellBox & brick : bricks)
    {
      countByBrick.push_back(fixedOn.defined()
                               ? cellsOf(fixedOn, brick.cell0, brick.cell1).sum().template item<int64_t>()
                               : cellCountOf(brick));
    }
    // The cells a brick touches: its own, the regularizer's next one, and the smoothing's reach. Smoothed alone,
    // they give the smoothed field of the whole grid on the brick's cells and the regularizer's next one.
    const int64_t reach = 1 + static_cast<int64_t>(m_ControlGridSmoothingIterations);
    auto          touchedBy = [&](const Impact::CellBox & brick) {
      Impact::CellBox window = brick;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        window.cell0[d] = std::max<int64_t>(0, brick.cell0[d] - reach);
        window.cell1[d] = std::min<int64_t>(cells[d], brick.cell1[d] + reach);
      }
      return window;
    };
    // One brick's loss for the smoothed field `smoothed` of the cells from `origin` on; `raw`, when given, takes
    // each layer's unweighted value times the brick's share.
    auto brickLoss = [&](size_t                       b,
                         const torch::Tensor &        smoothed,
                         const std::vector<int64_t> & origin,
                         std::vector<double> *        raw) -> torch::Tensor {
      const Impact::CellBox & box = bricks[b];
      if (countByBrick[b] == 0)
      {
        // Nothing of it counts: its edges alone, nothing read.
        return regularizer(smoothed, box.cell0, box.cell1, origin);
      }
      // Where each cell reads the moving map, in cells: q = c + u (voxels -> cells), and the cells of
      // the moving map the box reads: what q reaches, one around for the interpolation and one spare, within the
      // lattice, at least two wide.
      torch::Tensor own = smoothed;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        own = own.narrow(2 + static_cast<int64_t>(d), box.cell0[d] - origin[d], box.cell1[d] - box.cell0[d]);
      }
      const torch::Tensor  q = cellPositions(box.cell0, box.cell1) + own * voxelsToCells;
      std::vector<int64_t> m0(Dim), m1(Dim);
      {
        torch::NoGradGuard  noGrad;
        const torch::Tensor reached = q.detach().flatten(2);
        const torch::Tensor bounds =
          torch::stack({ std::get<0>(reached.min(2)), std::get<0>(reached.max(2)) }, 0).to(torch::kCPU); // {2, 1, Dim}
        for (unsigned int d = 0; d < Dim; ++d)
        {
          const auto low = static_cast<int64_t>(std::floor(bounds[0][0][d].template item<double>())) - 1;
          const auto high = static_cast<int64_t>(std::ceil(bounds[1][0][d].template item<double>())) + 2;
          m0[d] = std::clamp<int64_t>(low, 0, cells[d] - 1);
          m1[d] = std::clamp<int64_t>(high, 1, cells[d]);
          if (m1[d] - m0[d] < 2)
          {
            if (m1[d] < cells[d])
              ++m1[d];
            else if (m0[d] > 0)
              --m0[d];
          }
        }
      }
      std::vector<torch::Tensor> fixedLayers(losses.size()), movingLayers(losses.size());
      for (size_t l = 0; l < losses.size(); ++l)
      {
        fixedLayers[l] = cellsOf(fixedMaps[l], box.cell0, box.cell1).to(device);
        movingLayers[l] = cellsOf(movingMaps[l], m0, m1).to(device);
      }
      const torch::Tensor grid = gridIn(q, m0, m1);
      torch::Tensor       counted;
      if (fixedOn.defined())
      {
        counted = cellsOf(fixedOn, box.cell0, box.cell1).to(device);
      }
      if (movingShare.defined())
      {
        torch::NoGradGuard  noGrad;
        const torch::Tensor holds =
          F::grid_sample(cellsOf(movingShare, m0, m1).to(device), grid.detach(), sampleOpts) >= 0.5;
        counted = counted.defined() ? counted.logical_and(holds) : holds;
      }
      const torch::Tensor rows = drawRows(counted, cellCountOf(box));
      const double        share = static_cast<double>(countByBrick[b]) / totalCount;
      torch::Tensor       total = torch::zeros({}, onDevice);
      if (!rows.defined() || rows.numel() > 0)
      {
        for (size_t l = 0; l < losses.size(); ++l)
        {
          const torch::Tensor sampled = F::grid_sample(pick(l, movingLayers[l]), grid, sampleOpts);
          const torch::Tensor value = rowsLoss(l, pick(l, fixedLayers[l]), sampled, rows);
          if (raw != nullptr)
          {
            (*raw)[l] += value.template item<double>() * share;
          }
          total = total + layerWeights[l] * normalized(l, value) * share;
        }
      }
      const torch::Tensor loss = total + regularizer(smoothed, box.cell0, box.cell1, origin);
      return loss;
    };
    m_MetricValuesPerIteration.clear();
    m_MetricValuesPerIteration.reserve(m_NumberOfIterations);
    for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
    {
      drawSubsets(channels);
      if (normalizes && !normalization.IsLatched(0))
      {
        // Each layer's value at the starting field, without the graph: the dense run's mean, summed over the bricks.
        torch::NoGradGuard  noGrad;
        std::vector<double> raw(losses.size(), 0.0);
        for (size_t b = 0; b < bricks.size(); ++b)
        {
          const Impact::CellBox window = touchedBy(bricks[b]);
          brickLoss(b, smoothControl(cellsOf(theta, window.cell0, window.cell1).to(device)), window.cell0, &raw);
        }
        for (size_t l = 0; l < losses.size(); ++l)
        {
          normalization.Latch(l, raw[l]);
        }
      }
      // Each brick back-propagates through the smoothing passes to the cells it touches; the control grid's gradient
      // is their sum over the bricks, which touch one another's cells.
      double lossValue = 0.0;
      for (size_t b = 0; b < bricks.size(); ++b)
      {
        const Impact::CellBox window = touchedBy(bricks[b]);
        const torch::Tensor   control = cellsOf(theta, window.cell0, window.cell1)
                                        .to(device, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true)
                                        .requires_grad_(true);
        const torch::Tensor loss = brickLoss(b, smoothControl(control), window.cell0, nullptr);
        if (loss.requires_grad()) // a constant for a last cell out of the fixed mask: no similarity, no difference
        {
          loss.backward();
        }
        lossValue += loss.template item<double>();
        if (control.grad().defined())
        {
          cellsOf(gradient, window.cell0, window.cell1).add_(control.grad().to(torch::kCPU));
        }
      }
      adam.Advance();
      for (const Impact::CellBox & brick : bricks)
      {
        adam.StepCells(theta, gradient, brick.cell0, brick.cell1, device);
        cellsOf(gradient, brick.cell0, brick.cell1).zero_();
      }
      m_MetricValuesPerIteration.push_back(lossValue);
    }
  }
  // ---- The control grid, smoothed, as the field on its own grid, written a box at a time from the cells its
  // smoothing reads: the host holds that image alone. The dense output is interpolated from it. ----
  {
    torch::NoGradGuard noGrad;
    m_ControlGrid = Impact::AllocateCellGrid<Dim>(cells, m_FixedImage.GetPointer());
    const auto smoothingReach = static_cast<int64_t>(m_ControlGridSmoothingIterations);
    for (const Impact::CellBox & box : boxes)
    {
      std::vector<int64_t> w0(Dim), w1(Dim), lo(Dim), hi(Dim);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        w0[d] = std::max<int64_t>(0, box.cell0[d] - smoothingReach);
        w1[d] = std::min<int64_t>(cells[d], box.cell1[d] + smoothingReach);
        lo[d] = box.cell0[d] - w0[d];
        hi[d] = box.cell1[d] - w0[d];
      }
      Impact::WriteCellsToCellGrid<Dim>(
        Impact::NarrowTo(smoothControl(cellsOf(theta, w0, w1).to(device)), lo, hi, 2),
        box.cell0,
        cells,
        m_FixedImage.GetPointer(),
        m_ControlGrid.GetPointer());
    }
  }
  m_ControlGridTransform = DisplacementFieldTransformType::New();
  m_ControlGridTransform->SetDisplacementField(m_ControlGrid);
  m_DisplacementFieldTransform = m_ControlGridTransform;
  m_WarpedMovingImage = nullptr;
  m_ScratchBytes = static_cast<SizeValueType>(scratch.GetBytes());
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
  os << indent << "RegionSize: " << m_RegionSize << std::endl;
  os << indent << "BrickBytes: " << m_BrickBytes << std::endl;
  os << indent << "RegionHalo: " << m_RegionHalo << std::endl;
  os << indent << "ResidentFeatureBudget: " << m_ResidentFeatureBudget << std::endl;
  os << indent << "ScratchDirectory: " << m_ScratchDirectory << std::endl;
}

} // end namespace itk

#endif // itkImpactFineRegistration_hxx

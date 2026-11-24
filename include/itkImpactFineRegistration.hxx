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

#include <itkImageAlgorithm.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
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

/** Every kept layer's centre feature vector {b, C} of `config` run on `patches` ({b, 1, patch...}), normalized: the
 * voxel each patch is centred on (size / 2 along every axis), which the sampled Jacobian mode compares. */
inline std::vector<torch::Tensor>
PatchCentreFeatures(const ImpactModelConfiguration & config,
                    const torch::Tensor &            patches,
                    const ImpactImageMetadata &      metadata)
{
  std::vector<int64_t> repeat(patches.dim(), 1);
  repeat[1] = static_cast<int64_t>(config.GetNumberOfChannels());
  std::vector<torch::jit::IValue> outputs = Forward(config, patches.repeat(repeat).to(GetModelDtype(config)), metadata);
  const std::vector<bool> &       mask = config.GetLayersMask();
  std::vector<torch::Tensor>      layers;
  for (size_t it = 0; it < outputs.size() && it < mask.size(); ++it)
  {
    if (mask[it])
    {
      torch::Tensor layer = outputs[it].toTensor();
      while (layer.dim() > 2)
      {
        layer = layer.select(2, layer.size(2) / 2);
      }
      layers.push_back(NormalizeFeatureChannels(layer.to(torch::kFloat32), config.GetFeatureNormalization(), 1));
    }
  }
  return layers;
}

/** `passes` averages over 3^Dim cells (zero-padded) of a control field ({1, Dim, cells...}), ConvexAdam's
 * smoothing; the field itself without a pass. */
template <unsigned int Dim>
torch::Tensor
SmoothControlField(torch::Tensor control, unsigned int passes)
{
  namespace F = torch::nn::functional;
  for (unsigned int k = 0; k < passes; ++k)
  {
    if constexpr (Dim == 3)
      control = F::avg_pool3d(control, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
    else
      control = F::avg_pool2d(control, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
  }
  return control;
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
    if (m_SolvedTime < this->GetMTime())
    {
      this->SolveByRegions();
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
    torch::Tensor initField = Impact::DisplacementToVoxelField<ImageDimension>(
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

  // ---- 3. The identity affine of the base sampling grids (normalized [-1,1], last-dim order x,y,z),
  // align_corners=false: every grid, sampling and resizing of this run follows ConvexAdam's convention. ----
  torch::Tensor idAffine = torch::eye(ImageDimension, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  idAffine = torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(ImageDimension), 1 }, idAffine.options()) }, 1)
               .unsqueeze(0); // {1, N, N+1}

  namespace F = torch::nn::functional;
  const auto sampleOpts =
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);
  // The control field read at a point: its border value past the first and last cell centres, as resizing it gives.
  const auto controlOpts =
    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(false);

  auto smoothControl = [&](const torch::Tensor & control) {
    return Impact::SmoothControlField<ImageDimension>(control, m_ControlGridSmoothingIterations);
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
      return F::interpolate(field,
                            F::InterpolateFuncOptions().size(target).mode(torch::kBilinear).align_corners(false));
  };

  // Map the (optionally low-resolution) control grid to a full-resolution displacement field:
  // smoothing passes then trilinear/bilinear upsampling to the full image grid. With
  // GridShrinkFactor==1 and no smoothing this is the identity.
  auto controlGridToFullField = [&](const torch::Tensor & control) -> torch::Tensor {
    return resizeField(smoothControl(control), spatial);
  };

  // The moving image (or its mask's share) warped by a full-resolution field in voxels, what the returned field means:
  // the Jacobian mode's warp, a feature refresh's and the warped output.
  const std::vector<int64_t>       imageOrigin(ImageDimension, 0);
  const std::vector<torch::Tensor> imageIndex = Impact::BoxIndices(imageOrigin, spatial, device);
  auto                             warpImage = [&](const torch::Tensor & image, const torch::Tensor & field) {
    return Impact::WarpMap<ImageDimension>(image, imageOrigin, spatial, imageIndex, field);
  };

  // A map on the grid of a feature layer warped by a (smoothed) control field, DIRECTLY at that layer's resolution: the
  // field resized to the layer size, its displacement in the layer's samples (Impact::WarpMap). Avoids the full-res
  // detour (upsample field, then downsample grid) for downsampled layers. `layerBaseGrid` (each layer's identity base
  // grid), `layerSpatials`, `layerIndex` and `layerScale` are filled once after feature extraction, below.
  std::vector<torch::Tensor>              layerBaseGrid; // per-layer identity base grid [1, layer..., Dim]
  std::vector<std::vector<int64_t>>       layerSpatials; // per-layer spatial size (z,y,x)
  std::vector<std::vector<torch::Tensor>> layerIndex;    // per-layer sample positions (Impact::BoxIndices)
  std::vector<torch::Tensor>              layerScale;    // per-layer samples a voxel of displacement, {1, Dim, 1...}
  const bool                              jacobianMode = featureMode && (m_Mode == "Jacobian");
  // A map's samples a voxel of displacement moves by. The loss grid reads the field as ConvexAdam counts it, (cells -
  // 1)/2 cells of shrink voxels for half its extent (Impact::SamplesPerVoxel), so the field it returns is (cells -
  // 1)/cells of the displacement it matched. The Jacobian mode warps the image itself: the map's samples over the
  // image's voxels.
  auto samplesPerVoxel = [&](const std::vector<int64_t> & extent) {
    std::vector<float> values(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      values[d] = jacobianMode ? static_cast<float>(static_cast<double>(extent[d]) / static_cast<double>(spatial[d]))
                               : Impact::SamplesPerVoxel(extent[d], coarseSpatial[d], shrink[d]);
    }
    std::vector<int64_t> shape(ImageDimension + 2, 1);
    shape[1] = static_cast<int64_t>(ImageDimension);
    return torch::tensor(values, torch::kFloat32).view(shape).to(device);
  };
  auto warpLayer = [&](const torch::Tensor & map, size_t l, const torch::Tensor & smoothedControl) -> torch::Tensor {
    return Impact::WarpMap<ImageDimension>(map,
                                           std::vector<int64_t>(ImageDimension, 0),
                                           layerSpatials[l],
                                           layerIndex[l],
                                           resizeField(smoothedControl, layerSpatials[l]) * layerScale[l]);
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
  // iterations. A full-dimension model may downsample every axis, so its features are compared whole, both sides
  // pooled to the loss grid.
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
  // maps whole (its window LNCCKernel voxels of the image along every axis), the others compare their feature vectors
  // voxel by voxel; over the voxels `mask` keeps, when given.
  auto layerLoss = [&](size_t                l,
                       const torch::Tensor & fixedMap,
                       const torch::Tensor & movingMap,
                       const torch::Tensor & mask = {}) -> torch::Tensor {
    return Impact::MapLoss<ImageDimension>(*losses[l],
                                           fixedMap,
                                           movingMap,
                                           mask,
                                           losses[l]->IsSpatial()
                                             ? Impact::IsotropicWindow(spatial, fixedMap.sizes(), m_LNCCKernel)
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
  // The voxels that count: `fixedOn`, and where the moving mask's share `movingShare` warped by `warp` (a function of
  // the share, only called with a moving mask) holds.
  auto countedVoxels = [&](const torch::Tensor & fixedOn, const torch::Tensor & movingShare, const auto & warp) {
    if (!movingShare.defined())
    {
      return fixedOn;
    }
    torch::NoGradGuard  noGrad;
    const torch::Tensor warped = warp(movingShare) >= 0.5;
    return fixedOn.defined() ? fixedOn.logical_and(warped) : warped;
  };
  // SamplingPercentage < 1: layer l's loss on that share of its voxels, drawn anew at every iteration. Only those
  // points are warped: the fixed features are read at the drawn voxels, the moving ones at the same voxels displaced
  // by the residual field, interpolated there as warpLayer's resizeField would.
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
    // The drawn points' samples along each axis, from their raster numbers, {1, [1,] 1, n}, and their displacement
    // there, {1, N, [1,] 1, n}.
    std::vector<int64_t> axisShape(ImageDimension + 1, 1);
    axisShape[ImageDimension] = count;
    std::vector<torch::Tensor> index(ImageDimension);
    int64_t                    stride = voxels;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      stride /= layerSpatials[l][d];
      index[d] = drawn.div(stride, "trunc").remainder(layerSpatials[l][d]).reshape(axisShape);
    }
    const torch::Tensor displacement =
      F::grid_sample(smoothedResidual, points.reshape(pointShape), sampleOpts) * layerScale[l];
    const torch::Tensor moving = Impact::WarpMap<ImageDimension>(
      pick(l, movingLayers[l]), std::vector<int64_t>(ImageDimension, 0), layerSpatials[l], index, displacement);
    const int64_t       channels = moving.size(1);
    const torch::Tensor fixed = pick(l, fixedLayers[l]).reshape({ channels, voxels }).index_select(1, drawn);
    if (!movingMask.defined())
    {
      return losses[l]->forwardValue(fixed.t(), moving.reshape({ channels, count }).t());
    }
    torch::Tensor rows;
    {
      torch::NoGradGuard  noGrad;
      const torch::Tensor atTotal =
        smoothedTotal.defined() ? F::grid_sample(smoothedTotal, points.reshape(pointShape), sampleOpts) * layerScale[l]
                                : displacement.detach();
      rows = (Impact::WarpMap<ImageDimension>(
                movingLayerShare[l], std::vector<int64_t>(ImageDimension, 0), layerSpatials[l], index, atTotal) >= 0.5)
               .flatten()
               .nonzero()
               .flatten();
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
        // re-extracted from the warped image every iteration in the Adam loop and pooled to this same
        // grid, so movingLayers (the frozen-warp machinery) are unused here; the masks are warped on it.
        fixedLayersOnline.resize(fixedLayers.size());
        for (size_t l = 0; l < fixedLayers.size(); ++l)
        {
          fixedLayersOnline[l] = poolLayerToLoss(fixedLayers[l]).contiguous();
          if (masked)
          {
            const std::vector<int64_t> ls(fixedLayersOnline[l].sizes().begin() + 2, fixedLayersOnline[l].sizes().end());
            fixedLayerMask.push_back(fixedMaskOn(ls));
            movingLayerShare.push_back(movingShareOn(ls));
            layerSpatials.push_back(ls);
            layerIndex.push_back(Impact::BoxIndices(imageOrigin, ls, device));
            layerScale.push_back(samplesPerVoxel(ls));
          }
        }
      }
    }
    else
    {
      // Static mode: downsample the fixed/moving feature layers to the Adam similarity resolution
      // (never upsampling), then precompute the identity base grid at each layer's resulting
      // resolution so the loss grid is built directly there.
      layerBaseGrid.resize(movingLayers.size());
      layerSpatials.resize(movingLayers.size());
      for (size_t l = 0; l < movingLayers.size(); ++l)
      {
        fixedLayers[l] = poolLayerToLoss(fixedLayers[l]).contiguous();
        movingLayers[l] = poolLayerToLoss(movingLayers[l]).contiguous();
        std::vector<int64_t> ls(movingLayers[l].sizes().begin() + 2, movingLayers[l].sizes().end());
        layerSpatials[l] = ls;
        layerIndex.push_back(Impact::BoxIndices(std::vector<int64_t>(ImageDimension, 0), ls, device));
        layerScale.push_back(samplesPerVoxel(ls));
        if (sampled) // the base grid places the points a share of the voxels draws
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
  torch::Tensor              movingCoarse, fixedCoarse, coarseScale, fixedCoarseMask, movingCoarseShare;
  std::vector<torch::Tensor> coarseIndex;
  if (!featureMode)
  {
    // Whole cells, the tail a size the shrink does not divide leaves dropped, as the masks and the feature layers.
    movingCoarse = poolLayerToLoss(movingT);
    fixedCoarse = poolLayerToLoss(fixedT);
    fixedCoarseMask = fixedMaskOn(coarseSpatial);
    movingCoarseShare = movingShareOn(coarseSpatial);
    coarseIndex = Impact::BoxIndices(std::vector<int64_t>(ImageDimension, 0), coarseSpatial, device);
    coarseScale = samplesPerVoxel(coarseSpatial);
  }

  // ---- 4. Adam loop (entirely on device; no host copies). ----
  torch::optim::Adam optimizer(
    { theta },
    torch::optim::AdamOptions(m_LearningRate)
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
      // (full-dimension) is compared whole, its features pooled to the loss grid as the fixed ones; a sliced
      // model (features keep the swept axis) is taken in chunks of slices along the axis drawn for this
      // iteration, to bound peak memory.
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
          const auto [index, moved] = Impact::PointWarp(kept.flip(-1).to(onDevice), displacement);
          const torch::Tensor holds =
            Impact::WarpMap<ImageDimension>(movingMaskShare, imageOrigin, spatial, index, moved) >= 0.5;
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
          // voxels are placed a batch at a time, the field read there through grid_sample coordinates (x, y, z: ITK
          // order).
          auto patchesOf = [&](size_t i, int64_t begin, int64_t end, bool moving) {
            const torch::Tensor positions = Impact::SampledPatchPositions<ImageDimension>(
              m_FixedModelsConfiguration[i], kept.narrow(0, begin, end - begin), fixedSpacing, m_Seed, i, {}, onDevice);
            const int64_t        samples = positions.size(0) * positions.size(1);
            std::vector<int64_t> where(ImageDimension + 2, 1); // grid_sample's {1, [1,] 1, samples, Dim}
            where[ImageDimension] = samples;
            where[ImageDimension + 1] = ImageDimension;
            const torch::Tensor displacement =
              moving ? F::grid_sample(smoothed, ((positions + 0.5) / toGridOnDevice - 1.0).reshape(where), controlOpts)
                         .reshape({ static_cast<int64_t>(ImageDimension), samples })
                         .t()
                     : torch::Tensor();
            const auto [index, moved] = Impact::PointWarp(
              positions.flip(-1).reshape({ samples, static_cast<int64_t>(ImageDimension) }), displacement);
            std::vector<int64_t> shape{ end - begin, 1 };
            for (const int64_t extent : Impact::PatchTensorShape(m_FixedModelsConfiguration[i]))
            {
              shape.push_back(extent);
            }
            return Impact::WarpMap<ImageDimension>(moving ? movingT : fixedT, imageOrigin, spatial, index, moved)
              .reshape(shape);
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
                  const std::vector<torch::Tensor> layers = Impact::PatchCentreFeatures(
                    config, patchesOf(i, begin, end, moving), moving ? movingMetadata : fixedMetadata);
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
          drawSubsets(fixedCentres); // among the components a layer compares
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
          // out of device memory is replayed at half the size, the gradient set back to the copy it started from: none
          // when it ran out making the copy, which then added nothing.
          torch::Tensor gradientBefore;
          for (size_t i = 0; i < movingConfigs.size(); ++i)
          {
            Impact::ForEachBatch(
              movingConfigs[i],
              device,
              n,
              [&](int64_t begin, int64_t end) {
                gradientBefore.reset();
                gradientBefore = smoothed.grad().clone();
                const std::vector<torch::Tensor> layers =
                  Impact::PatchCentreFeatures(movingConfigs[i], patchesOf(i, begin, end, true), movingMetadata);
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
              [&] {
                if (gradientBefore.defined())
                {
                  smoothed.mutable_grad().copy_(gradientBefore);
                }
              });
          }
        }
        smoothedControl.backward(smoothed.grad()); // the smoothing passes, once: -> theta.grad
        lossValue = simValue + reg.template item<double>();
      }
      else if (jacSliced)
      {
        // Peak memory is bounded by taking the field's swept axis in chunks: each chunk backprops
        // into a detached field leaf -- its network subgraph is then freed -- and the shared control->field
        // graph is traversed once at the end. The per-point feature distance decomposes over voxels, so
        // weighting a chunk by its voxel fraction makes the accumulated gradient equal the whole mean's.
        const auto keep = static_cast<int64_t>(ImageDimension - 1 - sweptAxis); // tensor spatial index swept
        reg.backward();
        const torch::Tensor fullField = controlGridToFullField(theta); // {1, Dim, spatial...}, graph to theta
        torch::Tensor       fieldLeaf = fullField.detach().clone();
        fieldLeaf.set_requires_grad(true);
        const int64_t nLead = spatial[keep];
        const int64_t chunk = (m_FeatureChunkSize == 0) ? nLead : std::min<int64_t>(m_FeatureChunkSize, nLead);
        // The moving image warped by `field` on the slices `pick` (a tensor and its axis to those of the swept axis it
        // reads) takes of the image.
        auto warpSlices = [&](const torch::Tensor & field, const auto & pick) {
          std::vector<torch::Tensor> index = imageIndex;
          index[keep] = pick(index[keep], 1 + keep);
          return Impact::WarpMap<ImageDimension>(movingT, imageOrigin, spatial, index, pick(field, 2 + keep));
        };
        // A swept model's PCA centres the moving features by their own mean over the slices its basis was read on
        // (Impact::PcaSlices), which the chunks only give piece by piece: taken from those slices of the moving
        // image as the field warps it now, without the graph, as the FireANTs engine detaches it.
        std::vector<torch::Tensor> movingMeans(fixedSwept->basis.size());
        if (std::any_of(fixedSwept->basis.begin(), fixedSwept->basis.end(), [](const torch::Tensor & basis) {
              return basis.defined();
            }))
        {
          const torch::Tensor              slices = warpSlices(fieldLeaf.detach(), &Impact::PcaSlices);
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
            warpedMoving = warpImage(movingMaskShare, fieldLeaf.detach()) >= 0.5;
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
        // Each layer's loss over the chunk [i0, i0 + di) of the swept axis, the image warped by `field`.
        auto chunkLosses = [&](int64_t i0, int64_t di, const torch::Tensor & field, bool withGrad) {
          const torch::Tensor movingChunk =
            warpSlices(field, [&](const torch::Tensor & tensor, int64_t axis) { return tensor.narrow(axis, i0, di); });
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
            std::vector<torch::Tensor> values = chunkLosses(i0, di, fieldLeaf.detach(), false);
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
          std::vector<torch::Tensor> values = chunkLosses(i0, di, fieldLeaf, true);
          torch::Tensor              simChunk = torch::zeros({}, theta.options());
          for (size_t l = 0; l < values.size(); ++l)
          {
            simChunk = masked ? simChunk + layerWeights[l] * normalized(l, values[l]) * chunkShare(l, i0, di)
                              : simChunk + layerWeights[l] * normalized(l, values[l]);
          }
          torch::Tensor weighted =
            masked ? simChunk : simChunk * (static_cast<double>(di) / static_cast<double>(nLead));
          weighted.backward(); // accumulates into fieldLeaf.grad; this chunk's network graph is freed
          simValue += weighted.template item<double>();
        }
        fullField.backward(fieldLeaf.grad()); // shared control->field graph, once: fieldLeaf.grad -> theta.grad
        lossValue = simValue + reg.template item<double>();
      }
      else
      {
        // Full-dimension model: warp the whole moving image, re-extract its features, and pool each layer to the
        // loss grid as the fixed one is pooled, so that a cell compares the same voxels on both sides (a single
        // backward pass).
        std::vector<torch::Tensor> ml = Impact::ExtractFeatureLayers<ImageDimension>(
          movingConfigs, warpImage(movingT, controlGridToFullField(theta)), device, {}, true, fixedSpacing);
        torch::Tensor sim = torch::zeros({}, theta.options());
        for (size_t l = 0; l < ml.size(); ++l)
        {
          torch::Tensor mll = ml[l];
          if (l < pcaBasis.size() && pcaBasis[l].defined())
          {
            // Centred by their own mean taken without the graph, as the swept and sampled modes and FireANTs take it.
            const torch::Tensor features = mll.squeeze(0);
            mll = Impact::PcaTransform(features, pcaBasis[l], Impact::PcaSampleMean(features).detach()).unsqueeze(0);
          }
          mll = poolLayerToLoss(mll);
          torch::Tensor counted;
          if (masked)
          {
            counted = countedVoxels(fixedLayerMask[l], movingLayerShare[l], [&](const torch::Tensor & share) {
              return warpLayer(share, l, smoothControl(theta.detach()));
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
          const torch::Tensor warpedLayer = warpLayer(pick(l, movingLayers[l]), l, smoothedResidual);
          torch::Tensor       counted;
          if (masked)
          {
            counted = countedVoxels(fixedLayerMask[l], movingLayerShare[l], [&](const torch::Tensor & share) {
              return warpLayer(share, l, smoothedTotal.defined() ? smoothedTotal : smoothedResidual.detach());
            });
          }
          similarity =
            similarity + layerWeights[l] * normalized(l, layerLoss(l, pick(l, fixedLayers[l]), warpedLayer, counted));
        }
      }
      else
      {
        // Warp the pooled moving image directly at the control-grid resolution.
        const torch::Tensor        displacement = smoothControl(theta - thetaRef) * coarseScale;
        const std::vector<int64_t> origin(ImageDimension, 0);
        const torch::Tensor        warpedImage =
          Impact::WarpMap<ImageDimension>(movingCoarse, origin, coarseSpatial, coarseIndex, displacement);
        if (!masked)
        {
          similarity = (warpedImage - fixedCoarse).pow(2).mean();
        }
        else
        {
          const torch::Tensor counted =
            countedVoxels(fixedCoarseMask, movingCoarseShare, [&](const torch::Tensor & share) {
              return Impact::WarpMap<ImageDimension>(share, origin, coarseSpatial, coarseIndex, displacement.detach());
            });
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
      movingLayers = extractMovingFeatures(warpImage(movingT, controlGridToFullField(thetaRef)));
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
    warped = warpImage(movingT, controlGridToFullField(theta));
  }

  // ---- 5. Geometry-correct writeback (once), via the shared convention. ----
  DisplacementFieldType * output = this->GetOutput();
  output->SetRegions(m_FixedImage->GetLargestPossibleRegion());
  output->SetSpacing(m_FixedImage->GetSpacing());
  output->SetOrigin(m_FixedImage->GetOrigin());
  output->SetDirection(m_FixedImage->GetDirection());
  output->Allocate();

  Impact::WriteVoxelFieldToDisplacement<ImageDimension>(
    controlGridToFullField(theta.detach()), m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), output);

  // Wrap the field in a ready-to-use transform.
  m_DisplacementFieldTransform = DisplacementFieldTransformType::New();
  m_DisplacementFieldTransform->SetDisplacementField(output);
  {
    torch::NoGradGuard noGrad;
    m_ControlGrid = Impact::BuildCellGrid<ImageDimension>(smoothControl(theta), m_FixedImage.GetPointer());
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
  Impact::InterpolateCellGrid<ImageDimension>(
    m_ControlGrid.GetPointer(), m_FixedImage.GetPointer(), this->GetOutput(), region);
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
  if (!(m_SamplingPercentage > 0.0 && m_SamplingPercentage <= 1.0))
  {
    itkExceptionMacro("ImpactFineRegistration: SamplingPercentage must lie in (0, 1], got " << m_SamplingPercentage
                                                                                            << ".");
  }
  // The Jacobian mode re-extracts the moving features from the image the field warps, a unit of the grid at a time,
  // in the three ways the whole-image run does: over the unit's voxels; on the patch around each point drawn
  // (SamplingPercentage < 1, elastix's scheme); or, with models of one dimension fewer, slice by slice along an axis
  // of the image drawn at every iteration.
  const bool jacobian = featureMode && m_Mode == "Jacobian";
  const bool sampledJacobian = jacobian && m_SamplingPercentage < 1.0;
  const bool swept = jacobian && !sampledJacobian && m_FixedModelsConfiguration[0].GetDimension() < Dim;
  // Static mode with models of one dimension fewer: they sweep the ITK axis closest to head-feet, as the whole-image
  // run sweeps them, and their maps are pooled to the cells along every axis.
  const int mapSweep = featureMode && !jacobian && m_FixedModelsConfiguration[0].GetDimension() < Dim
                         ? static_cast<int>(Impact::HeadFeetAxis(m_FixedImage->GetDirection(), Dim))
                         : -1;
  // FeatureMapUpdateInterval: Static mode pools its moving maps again, every that many iterations, from the moving
  // image the field warps, and reads them past that field.
  const bool refreshes = featureMode && !jacobian && m_FeatureMapUpdateInterval > 0;

  const torch::Device device(m_Device);
  torch::manual_seed(m_Seed);
  // What the whole-image run draws from its own generator, drawn in the same order: the channels of SubsetFeatures,
  // the points of the sampled Jacobian mode.
  at::Generator                       generator = at::make_generator<at::CPUGeneratorImpl>(m_Seed);
  const Impact::PythonGilReleaseGuard gilRelease;
  // What grows with the image when the pooled maps are not resident; declared before the tensors it maps.
  Impact::ScratchFiles scratch(m_ScratchDirectory);
  m_BytesRead = 0;
  const auto onDevice = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto sampleOpts =
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
  std::vector<double> fixedSpacing(Dim); // ITK order
  for (unsigned int a = 0; a < Dim; ++a)
  {
    fixedSpacing[a] = m_FixedImage->GetSpacing()[a];
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
    // The whole-image run's, for the same displacements in the same cells: ConvexAdam's scale, or, in the Jacobian
    // mode, which warps the image itself, half the image's voxels.
    cellsPerVoxel[d] = jacobian ? static_cast<float>(static_cast<double>(cells[d]) / static_cast<double>(spatial[d]))
                                : Impact::SamplesPerVoxel(cells[d], cells[d], shrink[d]);
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
  // A model that resamples the image to its voxel size sees it on a grid of its own (FeatureGrid), its layers pooled to
  // the cells over that grid's adaptive windows, as the whole-image run pools them; all the models see one grid.
  // Without a voxel size, the image's grid and the cells' windows.
  const Impact::FeatureGrid grid =
    featureMode && !sampledJacobian
      ? Impact::ModelFeatureGrid<Dim>({ &m_FixedModelsConfiguration, &m_MovingModelsConfiguration },
                                      spatial,
                                      fixedSpacing,
                                      windows,
                                      "ImpactFineRegistration")
      : Impact::FeatureGrid{ spatial, windows, false };
  const std::vector<int64_t> &             featureSpatial = grid.spatial;
  const std::vector<Impact::PoolWindows> & featureWindows = grid.windows;
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

  // ---- Masks, read by regions: the fixed one on the fixed grid, the moving one on the moving image's grid, read with
  // it. A mask on a grid of its own is brought onto the fixed grid (the moving one through the moving transform) by a
  // resampler, nearest neighbour, that asks its input for each region alone, as the whole-image run brings it there
  // (MaskOnGrid). ----
  std::vector<ProcessObject::Pointer> maskResamplers; // alive for as long as their outputs are read
  auto                                onFixedGrid = [&](const MaskImageType * mask, const TransformType * transform) {
    auto resample = ResampleImageFilter<MaskImageType, MaskImageType, double>::New();
    resample->SetInput(mask);
    if (transform != nullptr)
    {
      resample->SetTransform(transform);
    }
    resample->SetInterpolator(NearestNeighborInterpolateImageFunction<MaskImageType, double>::New());
    resample->SetUseReferenceImage(true);
    resample->SetReferenceImage(m_FixedImage);
    resample->UpdateOutputInformation();
    maskResamplers.push_back(resample.GetPointer());
    return typename MaskImageType::ConstPointer(resample->GetOutput());
  };
  typename MaskImageType::ConstPointer fixedMask = m_FixedMask, movingMaskOnFixed;
  if (m_FixedMask.IsNotNull())
  {
    const_cast<MaskImageType *>(m_FixedMask.GetPointer())->UpdateOutputInformation();
    if (!Impact::SameGrid(m_FixedMask.GetPointer(), m_FixedImage.GetPointer()))
    {
      fixedMask = onFixedGrid(m_FixedMask.GetPointer(), nullptr);
    }
  }
  if (m_MovingMask.IsNotNull())
  {
    const_cast<MaskImageType *>(m_MovingMask.GetPointer())->UpdateOutputInformation();
    if (!Impact::SameGrid(m_MovingMask.GetPointer(), m_MovingImage.GetPointer()))
    {
      movingMaskOnFixed = onFixedGrid(m_MovingMask.GetPointer(), m_MovingTransform.GetPointer());
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
    m_MovingTransform.IsNull() && Impact::SameGrid(m_MovingImage.GetPointer(), m_FixedImage.GetPointer());
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
      const auto movingIndex = m_MovingImage->template TransformPhysicalPointToContinuousIndex<double>(point);
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
  // A layer of such a span ({1, C, samples...}, a mask: the image's lattice) pooled over the cells [c0, c1); a swept
  // model's keeping the voxels along tensor axis `keep`.
  auto toCells = [&](const torch::Tensor &        layer,
                     const Impact::LayerLattice & lattice,
                     const std::vector<int64_t> & lowPad,
                     const std::vector<int64_t> & c0,
                     const std::vector<int64_t> & c1,
                     int64_t                      keep = -1) {
    return Impact::PoolLayerToCells<Dim>(
      layer, lattice, lowPad, c0, c1, windows, spatial, "ImpactFineRegistration", keep);
  };
  // The moving image (with `mask`, the moving mask by nearest neighbour) on the voxels of `span`, a region of the fixed
  // grid, through the moving transform: {1, 1, span...}, zeros where it reads past the moving image.
  auto movingOnFixed = [&](const RegionType & span, bool mask) -> torch::Tensor {
    if (mask && movingMaskOnFixed.IsNotNull())
    {
      return readMask(movingMaskOnFixed.GetPointer(), span);
    }
    if (identityMap)
    {
      return mask ? readMask(m_MovingMask.GetPointer(), span) : readMoving(span);
    }
    const torch::Tensor  movingIndex = movingIndexField(span); // {1, Dim, span...} on the device
    std::vector<int64_t> windowShape{ 1, 1 };
    for (unsigned int d = 0; d < Dim; ++d)
    {
      windowShape.push_back(static_cast<int64_t>(span.GetSize()[Dim - 1 - d]));
    }
    // The region of the moving image those indices read, one voxel around for the interpolation.
    typename MovingImageType::RegionType movingRead;
    std::vector<int64_t>                 readStart(Dim), readStop(Dim);
    const torch::Tensor                  bounds =
      torch::stack({ std::get<0>(movingIndex.flatten(2).min(2)), std::get<0>(movingIndex.flatten(2).max(2)) }, 0)
        .to(torch::kCPU); // {2, 1, Dim}
    for (unsigned int d = 0; d < Dim; ++d)
    {
      const auto    low = static_cast<int64_t>(std::floor(bounds[0][0][d].template item<double>())) - 1;
      const auto    high = static_cast<int64_t>(std::ceil(bounds[1][0][d].template item<double>())) + 1;
      const int64_t first = movingRegion.GetIndex()[Dim - 1 - d];
      const int64_t extent = static_cast<int64_t>(movingRegion.GetSize()[Dim - 1 - d]);
      readStart[d] = std::max(low, first);
      readStop[d] = std::min(high + 1, first + extent);
      if (readStop[d] <= readStart[d])
      {
        return torch::zeros(windowShape, onDevice);
      }
      movingRead.SetIndex(Dim - 1 - d, static_cast<IndexValueType>(readStart[d]));
      movingRead.SetSize(Dim - 1 - d, static_cast<SizeValueType>(readStop[d] - readStart[d]));
    }
    const torch::Tensor grid = gridIn(movingIndex, readStart, readStop);
    return mask ? F::grid_sample(readMask(m_MovingMask.GetPointer(), movingRead), grid, nearestOpts)
                : F::grid_sample(readMoving(movingRead), grid, sampleOpts);
  };
  auto readMovingImage = [&](const RegionType & region) { return movingOnFixed(region, false); };
  auto readMovingMask = [&](const RegionType & region) { return movingOnFixed(region, true); };
  auto readFixedMask = [&](const RegionType & region) { return readMask(fixedMask.GetPointer(), region); };
  // What `read` gives for a region of the fixed grid at the voxels `index` (per tensor axis, integers broadcasting to
  // {1, n...}) moved by `displacement` ({1, Dim, n...}, in voxels), as Impact::WarpMap reads the whole image: the
  // region read is the box of the voxels they read, cut at the image. {1, 1, n...}, differentiable to the displacement.
  auto readWarped = [&](auto && read, const std::vector<torch::Tensor> & index, const torch::Tensor & displacement) {
    // Points that read nothing of the image: zeros, as WarpMap gives past a map, on the graph of the displacement (with
    // a zero gradient), which a backward of a batch of them runs through.
    auto nothing = [&] { return displacement.narrow(1, 0, 1) * 0.0; };
    if (displacement.numel() == 0)
    {
      return nothing();
    }
    std::vector<torch::Tensor> ends;
    {
      torch::NoGradGuard noGrad;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        const torch::Tensor sample = index[d] + Impact::WarpBelow(displacement.select(1, d)).to(torch::kLong);
        ends.push_back(sample.min());
        ends.push_back(sample.max());
      }
    }
    const torch::Tensor  bounds = torch::stack(ends).to(torch::kCPU);
    std::vector<int64_t> lo(Dim);
    RegionType           region;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      lo[d] = std::max<int64_t>(bounds[2 * d].template item<int64_t>(), 0);
      const int64_t hi = std::min(bounds[2 * d + 1].template item<int64_t>() + 2, spatial[d]);
      if (hi <= lo[d])
      {
        return nothing();
      }
      region.SetIndex(Dim - 1 - d, fixedRegion.GetIndex()[Dim - 1 - d] + static_cast<IndexValueType>(lo[d]));
      region.SetSize(Dim - 1 - d, static_cast<SizeValueType>(hi - lo[d]));
    }
    return Impact::WarpMap<Dim>(read(region), lo, spatial, index, displacement);
  };
  // What `read` gives at the points `positions` ({n, Dim}, voxels of the fixed grid in tensor order) moved by `moved`
  // ({n, Dim}, undefined for none), as the whole-image run reads them (Impact::PointWarp). {n}.
  auto readAt = [&](auto && read, const torch::Tensor & positions, const torch::Tensor & moved) {
    const auto [index, displacement] = Impact::PointWarp(positions, moved);
    return readWarped(read, index, displacement).flatten();
  };
  // The field {1, Dim, n...} at the voxels [lo, hi) of the fixed grid (tensor order), in voxels: the smoothed control
  // field `smoothed` of the cells from `origin` on, resized to the image as the whole-image run resizes it.
  auto fieldOnVoxels = [&](const torch::Tensor &        smoothed,
                           const std::vector<int64_t> & origin,
                           const std::vector<int64_t> & lo,
                           const std::vector<int64_t> & hi) {
    return Impact::ResizeRegion<Dim>(smoothed, origin, cells, spatial, lo, hi, "ImpactFineRegistration");
  };
  // The field {n, Dim} at `positions` ({n, Dim}, voxels of the fixed grid in tensor order), in voxels: the smoothed
  // control field of the cells from `origin` on read there as the whole-image run reads it at a patch's voxels, its
  // values at the first and last cell centres past them.
  auto fieldAtPoints =
    [&](const torch::Tensor & smoothed, const std::vector<int64_t> & origin, const torch::Tensor & positions) {
      std::vector<int64_t> stop(Dim), shape(Dim + 2, 1);
      std::vector<double>  toCellsScale(Dim), last(Dim);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        stop[d] = origin[d] + smoothed.size(2 + static_cast<int64_t>(d));
        toCellsScale[d] = static_cast<double>(cells[d]) / static_cast<double>(spatial[d]);
        last[d] = static_cast<double>(cells[d] - 1);
      }
      shape[1] = static_cast<int64_t>(Dim);
      shape[Dim + 1] = positions.size(0);
      const torch::Tensor q = torch::minimum(
        ((positions + 0.5) * torch::tensor(toCellsScale, torch::kFloat64).to(onDevice) - 0.5).clamp_min(0.0),
        torch::tensor(last, torch::kFloat64).to(onDevice));
      return F::grid_sample(smoothed, gridIn(q.t().reshape(shape), origin, stop), sampleOpts)
        .reshape({ static_cast<int64_t>(Dim), positions.size(0) })
        .t();
    };
  // What `read` gives, warped by the field: at each voxel x of `span`, read at x + u(x). {1, 1, span...}.
  auto warpedOn =
    [&](auto && read, const RegionType & span, const torch::Tensor & smoothed, const std::vector<int64_t> & origin) {
      std::vector<int64_t> lo(Dim), hi(Dim), shape{ 1, 1 };
      for (unsigned int d = 0; d < Dim; ++d)
      {
        lo[d] = static_cast<int64_t>(span.GetIndex()[Dim - 1 - d] - fixedRegion.GetIndex()[Dim - 1 - d]);
        hi[d] = lo[d] + static_cast<int64_t>(span.GetSize()[Dim - 1 - d]);
        shape.push_back(hi[d] - lo[d]);
      }
      return readWarped(read, Impact::BoxIndices(lo, hi, device), fieldOnVoxels(smoothed, origin, lo, hi));
    };

  // ---- The feature models and distances. ----
  std::vector<std::unique_ptr<Impact::Loss>> losses;
  std::vector<float>                         layerWeights;
  bool                                       takesMetadata = false; // a model that reads the image's statistics
  if (featureMode)
  {
    const Impact::RegionModelRun run = sampledJacobian          ? Impact::RegionModelRun::Patches
                                       : swept || mapSweep >= 0 ? Impact::RegionModelRun::Swept
                                                                : Impact::RegionModelRun::Maps;
    // The models of Static mode and of the dense Jacobian mode may resample the image to their voxel size (`grid`,
    // above); the swept Jacobian mode compares slices index by index, as the whole-image run.
    const bool                 resampling = run == Impact::RegionModelRun::Maps || mapSweep >= 0;
    const Impact::RegionModels models = Impact::CheckRegionModels(
      m_FixedModelsConfiguration, m_FixedImage->GetSpacing(), Dim, "ImpactFineRegistration", run, resampling);
    const size_t layerCount = models.layerCount;
    // The moving models of their own are held to the same: the moving image reaches them on the fixed grid.
    const Impact::RegionModels movingModels = Impact::CheckRegionModels(
      m_MovingModelsConfiguration, m_FixedImage->GetSpacing(), Dim, "ImpactFineRegistration", run, resampling);
    takesMetadata = models.takesMetadata || movingModels.takesMetadata;
    for (size_t l = 0; l < layerCount; ++l)
    {
      const std::string name = m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)];
      losses.push_back(Impact::LossFactory::Instance().Create(name));
      if (losses.back()->IsSpatial() && m_SamplingPercentage < 1.0)
      {
        itkExceptionMacro("ImpactFineRegistration: distance '"
                          << name << "' correlates windows of dense maps, which the points SamplingPercentage draws "
                          << "do not have.");
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
  // every call has nothing to cut), a swept one slice by slice along ITK axis `axis`; the region itself in intensity
  // mode. With `withGrad`, differentiable to the region.
  // In Static mode the models take the statistics of their normalisations over the whole image (below), each model's
  // of each image; none until they are gathered, and none in the Jacobian mode, where every region keeps its own.
  std::vector<std::vector<torch::Tensor>> fixedNorms(m_FixedModelsConfiguration.size()),
    movingNorms(movingConfigs.size());
  auto layersOf = [&](const torch::Tensor & image, bool moving, bool withGrad = false, int axis = -1) {
    return Impact::RegionLayers<Dim>(featureMode ? (moving ? movingConfigs : m_FixedModelsConfiguration)
                                                 : std::vector<ImpactModelConfiguration>{},
                                     image,
                                     device,
                                     losses.size(),
                                     takesMetadata ? &(moving ? movingMetadata : fixedMetadata) : nullptr,
                                     "ImpactFineRegistration",
                                     withGrad,
                                     axis,
                                     moving ? &movingNorms : &fixedNorms);
  };
  std::vector<int64_t> cellVoxels(Dim),
    featureCellVoxels(Dim); // a cell's voxels, and its widest window on the models' grid
  for (unsigned int d = 0; d < Dim; ++d)
  {
    cellVoxels[d] = windows[d].ends[0] - windows[d].starts[0];
    for (int64_t c = 0; c < cells[d]; ++c)
    {
      featureCellVoxels[d] = std::max(featureCellVoxels[d], featureWindows[d].ends[c] - featureWindows[d].starts[c]);
    }
  }
  // Where the models' layers lie on the image, probed once, and once for each axis a swept model sweeps; the sampled
  // Jacobian mode reads no map.
  std::vector<Impact::RegionLattices> latticeSets;
  for (int axis = swept ? 0 : -1; !sampledJacobian && axis < (swept ? static_cast<int>(Dim) : 0); ++axis)
  {
    latticeSets.push_back(Impact::ProbeRegionLattices<Dim>(
      halo,
      featureCellVoxels,
      device,
      !m_MovingModelsConfiguration.empty(),
      [&](const torch::Tensor & image, bool moving) {
        return layersOf(image, moving, false, axis >= 0 ? axis : mapSweep);
      },
      "ImpactFineRegistration"));
  }
  // A map's lattices; those of a swept model's along ITK axis `axis`.
  auto latticesAlong = [&](int axis) -> const Impact::RegionLattices & {
    return latticeSets[axis < 0 ? 0 : static_cast<size_t>(axis)];
  };
  // The tensor spatial axis a swept model keeps the voxels of, sweeping ITK axis `axis`; -1 for a map.
  auto keepOf = [](int axis) { return axis < 0 ? int64_t{ -1 } : static_cast<int64_t>(Dim) - 1 - axis; };
  // The region of the fixed grid the features of the cells [c0, c1) are computed on: their voxels and the models'
  // reach around them, from the layers' lattice on, cut at the image; swept along ITK axis `axis`, the voxels past the
  // last cell along it with the last.
  auto featureSpan =
    [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1, std::vector<int64_t> & lowPad, int axis) {
      const Impact::RegionLattices & on = latticesAlong(axis);
      RegionType                     span = spanOf(c0, c1, on.reach, lowPad, on.align);
      if (axis >= 0 && c1[Dim - 1 - axis] == cells[Dim - 1 - axis])
      {
        span.SetSize(axis,
                     static_cast<SizeValueType>(fixedRegion.GetIndex()[axis] +
                                                static_cast<IndexValueType>(fixedRegion.GetSize()[axis]) -
                                                span.GetIndex()[axis]));
      }
      return span;
    };
  // The image a model runs on for the features of the cells [c0, c1), from `read` (a region of the fixed grid to its
  // {1, 1, region...}): their samples on the models' grid with the reach of the layers' lattices around
  // (FeatureRegion), with `toEnd` on to the grid's end along an axis the cells end it, swept along ITK axis `axis`
  // along it. `lowPad` is where the cells' samples start in it.
  auto featureInput = [&](const std::vector<int64_t> & c0,
                          const std::vector<int64_t> & c1,
                          std::vector<int64_t> &       lowPad,
                          int                          axis,
                          auto &&                      read,
                          bool                         toEnd = false) -> torch::Tensor {
    const Impact::RegionLattices & on = latticesAlong(axis);
    std::vector<bool>              ends(Dim, toEnd);
    if (axis >= 0)
    {
      ends[Dim - 1 - axis] = true;
    }
    const auto region =
      Impact::FeatureRegionOf<Dim, RegionType>(grid, spatial, fixedRegion.GetIndex(), c0, c1, on.reach, on.align, ends);
    lowPad = region.lowPad;
    return Impact::FeatureInput<Dim>(grid, region, read(region.span), spatial, "ImpactFineRegistration");
  };
  auto pooledToCells = [&](std::vector<torch::Tensor> &              layers,
                           const std::vector<Impact::LayerLattice> & on,
                           const std::vector<int64_t> &              lowPad,
                           const std::vector<int64_t> &              c0,
                           const std::vector<int64_t> &              c1,
                           int64_t                                   keep = -1) {
    for (size_t l = 0; l < layers.size(); ++l)
    {
      layers[l] = Impact::PoolLayerToCells<Dim>(
        layers[l], on[l], lowPad, c0, c1, featureWindows, featureSpatial, "ImpactFineRegistration", keep);
    }
  };
  // PCA: each layer's basis, and the mean each image's features are centred by before they are projected on it
  // (below), in Static mode; a swept model's for each axis it sweeps, and the moving mean of the iteration's axis.
  // Undefined for a layer without. Pooling commutes with the projection.
  std::vector<torch::Tensor> pcaBasis(losses.size()), fixedPcaMean(losses.size()), movingPcaMean(losses.size());
  std::vector<std::vector<torch::Tensor>> sweptBasis(Dim, std::vector<torch::Tensor>(losses.size())),
    sweptFixedMean(Dim, std::vector<torch::Tensor>(losses.size()));
  auto reduced = [&](std::vector<torch::Tensor> &       layers,
                     const std::vector<torch::Tensor> & bases,
                     const std::vector<torch::Tensor> & means) {
    for (size_t l = 0; l < layers.size(); ++l)
    {
      if (bases[l].defined())
      {
        layers[l] = Impact::PcaTransform(layers[l].squeeze(0), bases[l], means[l]).unsqueeze(0);
      }
    }
  };
  // Layer l's principal components, 0 for none (unset, or not fewer than its `channel` channels).
  auto componentsOf = [&](size_t l, int64_t channel) -> int64_t {
    const int64_t components = l < m_PCA.size() ? static_cast<int64_t>(m_PCA[l]) : 0;
    return components > 0 && components < channel ? components : 0;
  };
  // The fixed features pooled over the cells [c0, c1): {1, C, n...} per layer; swept along ITK axis `axis`, the
  // voxels along it kept.
  auto pooledFixed = [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1, int axis = -1) {
    std::vector<int64_t>       lowPad(Dim);
    std::vector<torch::Tensor> layers =
      layersOf(featureInput(c0, c1, lowPad, axis, readFixed), false, false, axis >= 0 ? axis : mapSweep);
    pooledToCells(layers, latticesAlong(axis).fixed, lowPad, c0, c1, keepOf(axis));
    if (axis >= 0)
    {
      reduced(layers, sweptBasis[axis], sweptFixedMean[axis]);
    }
    else
    {
      reduced(layers, pcaBasis, fixedPcaMean);
    }
    return layers;
  };
  // The moving features re-extracted from the moving image warped by the smoothed control field `smoothed` of the cells
  // from `origin` on, pooled likewise, with the graph when it is recorded: the Jacobian mode's, and a refresh's.
  auto pooledWarped = [&](const std::vector<int64_t> & c0,
                          const std::vector<int64_t> & c1,
                          const torch::Tensor &        smoothed,
                          const std::vector<int64_t> & origin,
                          int                          axis) {
    std::vector<int64_t> lowPad(Dim);
    const torch::Tensor  image = featureInput(
      c0, c1, lowPad, axis, [&](const RegionType & span) { return warpedOn(readMovingImage, span, smoothed, origin); });
    std::vector<torch::Tensor> layers =
      layersOf(image, true, torch::GradMode::is_enabled(), axis >= 0 ? axis : mapSweep);
    pooledToCells(layers, latticesAlong(axis).moving, lowPad, c0, c1, keepOf(axis));
    return layers;
  };
  // The moving mask's share of each of the cells [c0, c1), {1, 1, n...}, the mask brought onto the fixed grid.
  auto movingShareOf = [&](const std::vector<int64_t> & c0, const std::vector<int64_t> & c1) {
    std::vector<int64_t> lowPad(Dim);
    const RegionType     span = spanOf(c0, c1, std::vector<int64_t>(Dim, 0), lowPad);
    return toCells(movingOnFixed(span, true), Impact::ImageLattice(Dim), lowPad, c0, c1);
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
    PooledMoving         pooled;
    pooled.layers =
      layersOf(featureInput(c0, c1, lowPad, -1, [&](const RegionType & span) { return movingOnFixed(span, false); }),
               true,
               false,
               mapSweep);
    pooledToCells(pooled.layers, latticesAlong(-1).moving, lowPad, c0, c1);
    reduced(pooled.layers, pcaBasis, movingPcaMean);
    if (m_MovingMask.IsNotNull())
    {
      pooled.share = movingShareOf(c0, c1);
    }
    return pooled;
  };
  // SubsetFeatures: per kept layer, that many of the channels it compares (its components with PCA) drawn at random at
  // every iteration (0 = all).
  std::vector<torch::Tensor> subsets(losses.size());
  auto                       drawSubsets = [&](const std::vector<int64_t> & channels) {
    for (size_t l = 0; l < losses.size(); ++l)
    {
      const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
      const int64_t compared = componentsOf(l, channels[l]) > 0 ? componentsOf(l, channels[l]) : channels[l];
      subsets[l] = (kept > 0 && kept < compared)
                                           ? torch::randperm(compared, generator, torch::kLong).narrow(0, 0, kept).to(device)
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
  // The rows `rows` of two pooled maps {1, C, n...} (every row when undefined), {n, C} each.
  auto rowsOf = [&](const torch::Tensor & fixed, const torch::Tensor & moving, const torch::Tensor & rows) {
    const int64_t channels = moving.size(1);
    torch::Tensor f = fixed.reshape({ channels, -1 }), m = moving.reshape({ channels, -1 });
    if (rows.defined())
    {
      f = f.index_select(1, rows);
      m = m.index_select(1, rows);
    }
    return std::make_pair(f.t(), m.t());
  };
  // Layer l's loss between the rows `rows` of two pooled maps.
  auto rowsLoss = [&](size_t l, const torch::Tensor & fixed, const torch::Tensor & moving, const torch::Tensor & rows) {
    const auto [f, m] = rowsOf(fixed, moving, rows);
    return losses[l]->forwardValue(f, m);
  };
  // The rows that count among `counted` (bool {1, 1, n...}, of the cells [c0, c1); every row when undefined). With
  // SamplingPercentage < 1, each cell with that chance at every iteration (`drawIteration`), by a hash of its number on
  // the grid, the iteration and Seed: every pass of an iteration and every way of cutting the grid into units draw the
  // same cells, and a statistic over all the points (NCC) takes one draw of them.
  unsigned int drawIteration = 0;
  auto drawRows = [&](const torch::Tensor & counted, const std::vector<int64_t> & c0, const std::vector<int64_t> & c1) {
    torch::Tensor rows;
    if (m_SamplingPercentage < 1.0)
    {
      constexpr int64_t                modulus = 2147483647; // the Lehmer generator's, 2^31 - 1
      const std::vector<torch::Tensor> index = Impact::BoxIndices(c0, c1, device);
      torch::Tensor flat = torch::zeros({}, torch::TensorOptions().dtype(torch::kLong).device(device));
      for (unsigned int d = 0; d < Dim; ++d)
      {
        flat = flat * cells[d] + index[d];
      }
      torch::Tensor hash = (flat.flatten() + (static_cast<int64_t>(drawIteration) + 1) * 1000003 +
                            static_cast<int64_t>(m_Seed) * 7919 + 1) %
                           modulus;
      for (const int64_t shift : { 13, 7, 17 })
      {
        hash = (hash * 48271) % modulus;
        hash = hash.bitwise_xor(hash.bitwise_right_shift(shift));
      }
      const torch::Tensor drawn = hash.to(torch::kFloat64) < m_SamplingPercentage * static_cast<double>(modulus);
      rows = (counted.defined() ? drawn.logical_and(counted.flatten()) : drawn).nonzero().flatten();
    }
    else if (counted.defined())
    {
      rows = counted.flatten().nonzero().flatten();
    }
    return rows;
  };

  // ---- The route: the pooled feature maps of both images, computed once by one pass over the boxes, are compared
  // whole at every iteration when they fit the device (the dense run on them), brick by brick when they do not. What
  // grows with the image is then at home on the device, or in mapped files of which an iteration brings the cells of
  // a brick at a time to the device. ----
  std::vector<int64_t> channels(losses.size(), 1);
  if (sampledJacobian)
  {
    // Each model on a patch of zeros; the batches its patches go by are measured on the device, as the whole-image run
    // measures them.
    size_t layer = 0;
    for (const ImpactModelConfiguration & config : m_FixedModelsConfiguration)
    {
      ModelTo(config, device);
      std::vector<int64_t> shape{ 1, 1 };
      for (const int64_t extent : Impact::PatchTensorShape(config))
      {
        shape.push_back(extent);
      }
      for (const torch::Tensor & centre :
           Impact::PatchCentreFeatures(config, torch::zeros(shape, onDevice), fixedMetadata))
      {
        channels[layer++] = centre.size(1);
      }
    }
    for (const ImpactModelConfiguration & config : movingConfigs)
    {
      ModelTo(config, device);
    }
    Impact::ConfigureBatchSize(m_FixedModelsConfiguration, device, static_cast<int64_t>(m_BatchSize));
    if (!m_MovingModelsConfiguration.empty())
    {
      Impact::ConfigureBatchSize(m_MovingModelsConfiguration, device, static_cast<int64_t>(m_BatchSize));
    }
  }
  else if (featureMode)
  {
    const std::vector<torch::Tensor> probe = pooledFixed(boxes[0].cell0, boxes[0].cell1, swept ? 0 : -1);
    for (size_t l = 0; l < probe.size(); ++l)
    {
      channels[l] = probe[l].size(1);
    }
  }
  // The features of the cells of `box` at each layer's resolution, of the image `read` gives for a region of the fixed
  // grid, {C, n} per layer: the samples of their voxels, the voxels past the last cell along an axis with the last box,
  // so that the boxes share out every sample of the whole layer; a swept model's (along ITK axis `axis`, or the axis
  // Static mode sweeps) those of the slices PcaSlices reads.
  auto ownLayers = [&](const Impact::CellBox & box, bool moving, auto && read, int axis = -1) {
    const int                  sweep = axis >= 0 ? axis : mapSweep;
    std::vector<int64_t>       lowPad(Dim);
    std::vector<torch::Tensor> layers =
      layersOf(featureInput(box.cell0, box.cell1, lowPad, axis, read, true), moving, false, sweep);
    const std::vector<Impact::LayerLattice> & on = moving ? latticesAlong(axis).moving : latticesAlong(axis).fixed;
    for (size_t l = 0; l < layers.size(); ++l)
    {
      layers[l] = Impact::PcaOwnSamples<Dim>(layers[l],
                                             on[l],
                                             lowPad,
                                             box.cell0,
                                             box.cell1,
                                             featureWindows,
                                             featureSpatial,
                                             sweep >= 0 ? keepOf(sweep) : -1);
    }
    return layers;
  };
  // Each layer with a basis in `bases`: its mean {C, 1} over every sample of the image `read` gives (a function of the
  // box and the region), swept along `axis`.
  auto meansOver = [&](bool moving, const std::vector<torch::Tensor> & bases, auto && read, int axis = -1) {
    std::vector<Impact::PcaSums> sums(losses.size());
    for (const Impact::CellBox & box : boxes)
    {
      const std::vector<torch::Tensor> own =
        ownLayers(box, moving, [&](const RegionType & span) { return read(box, span); }, axis);
      for (size_t l = 0; l < own.size(); ++l)
      {
        if (bases[l].defined())
        {
          sums[l].Add(own[l]);
        }
      }
    }
    std::vector<torch::Tensor> means(losses.size());
    for (size_t l = 0; l < means.size(); ++l)
    {
      if (bases[l].defined())
      {
        means[l] = sums[l].Mean();
      }
    }
    return means;
  };
  // Each layer's PCA basis and fixed mean, fitted on the fixed features of every voxel (swept along `axis`: of the
  // slices PcaSlices reads), as PcaFit fits them on the whole layer, from their sums over the boxes.
  auto fitPca = [&](std::vector<torch::Tensor> & bases, std::vector<torch::Tensor> & means, int axis) {
    std::vector<Impact::PcaSums> sums(losses.size());
    for (const Impact::CellBox & box : boxes)
    {
      const std::vector<torch::Tensor> own = ownLayers(box, false, readFixed, axis);
      for (size_t l = 0; l < own.size(); ++l)
      {
        if (componentsOf(l, own[l].size(0)) > 0)
        {
          sums[l].Add(own[l]);
        }
      }
    }
    for (size_t l = 0; l < losses.size(); ++l)
    {
      if (sums[l].count > 0.0)
      {
        bases[l] = sums[l].Basis(m_PCA[l]);
        means[l] = sums[l].Mean();
        channels[l] = m_PCA[l];
      }
    }
  };
  // Static mode: the statistics of each model's normalisations over the whole image (GatherNorms), `read` giving the
  // image of a box's region.
  auto gatherNorms = [&](bool moving, auto && read) {
    return Impact::GatherNorms<Dim>(
      moving ? movingConfigs : m_FixedModelsConfiguration,
      takesMetadata ? (moving ? movingMetadata : fixedMetadata) : ImpactImageMetadata{},
      boxes,
      featureWindows,
      featureSpatial,
      device,
      [&](const Impact::CellBox & box, std::vector<int64_t> & lowPad) {
        return featureInput(
          box.cell0, box.cell1, lowPad, -1, [&](const RegionType & span) { return read(box, span); }, true);
      });
  };
  // A single box whose region is the whole image normalises it by its own statistics, the image's to the bit.
  auto spansImage = [&]() {
    const auto single = Impact::FeatureRegionOf<Dim, RegionType>(grid,
                                                                 spatial,
                                                                 fixedRegion.GetIndex(),
                                                                 boxes[0].cell0,
                                                                 boxes[0].cell1,
                                                                 latticesAlong(-1).reach,
                                                                 latticesAlong(-1).align,
                                                                 std::vector<bool>(Dim, false));
    return single.s0 == std::vector<int64_t>(Dim, 0) && single.s1 == featureSpatial;
  };
  const bool gathersNorms = featureMode && !jacobian && (boxes.size() > 1 || !spansImage());
  if (gathersNorms)
  {
    fixedNorms = gatherNorms(false, [&](const Impact::CellBox &, const RegionType & span) { return readFixed(span); });
    movingNorms =
      gatherNorms(true, [&](const Impact::CellBox &, const RegionType & span) { return movingOnFixed(span, false); });
  }
  const bool fitsPca = featureMode && !sampledJacobian &&
                       std::any_of(m_PCA.begin(), m_PCA.end(), [](unsigned int components) { return components > 0; });
  if (fitsPca && !jacobian)
  {
    // Static mode: each image's maps centred by its own features' mean.
    fitPca(pcaBasis, fixedPcaMean, -1);
    movingPcaMean = meansOver(
      true, pcaBasis, [&](const Impact::CellBox &, const RegionType & span) { return movingOnFixed(span, false); });
  }
  if (fitsPca && jacobian && !swept)
  {
    // The dense Jacobian mode: the fixed maps centred by their own features' mean, the moving ones by theirs at every
    // iteration.
    fitPca(pcaBasis, fixedPcaMean, -1);
  }
  for (unsigned int axis = 0; fitsPca && swept && axis < Dim; ++axis)
  {
    // A swept model: the basis of each axis it sweeps, the fixed maps centred by their own slices' mean; the moving
    // ones by theirs at every iteration.
    fitPca(sweptBasis[axis], sweptFixedMean[axis], static_cast<int>(axis));
  }
  int64_t channelCount = 0;
  for (const int64_t c : channels)
  {
    channelCount += c;
  }
  const bool drawsBoxes = m_RegionsPerIteration > 0; // an iteration steps those boxes alone
  // A spatial distance (LNCC) reads windows of the maps, of IsotropicWindow's samples on the whole map: a unit's terms
  // are taken on its cells and `lossHalo` around them, half the widest window in cells (a swept model's maps keep the
  // image's voxels along the axis swept), and kept on its own.
  const bool spatialLoss = std::any_of(
    losses.begin(), losses.end(), [](const std::unique_ptr<Impact::Loss> & loss) { return loss->IsSpatial(); });
  auto windowOf = [&](const torch::Tensor & map) {
    return Impact::IsotropicWindow(spatial, map.sizes(), m_LNCCKernel);
  };
  int64_t lossHalo = 0;
  for (int axis = swept ? 0 : -1; spatialLoss && axis < (swept ? static_cast<int>(Dim) : 0); ++axis)
  {
    const int64_t        keep = keepOf(axis);
    std::vector<int64_t> shape{ 1, 1 };
    shape.insert(shape.end(), cells.begin(), cells.end());
    if (keep >= 0)
    {
      shape[2 + keep] = spatial[keep];
    }
    const std::vector<int64_t> window = Impact::IsotropicWindow(spatial, shape, m_LNCCKernel);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      const int64_t side = window[d] / 2;
      lossHalo =
        std::max(lossHalo, static_cast<int64_t>(d) == keep ? (side + cellVoxels[d] - 1) / cellVoxels[d] : side);
    }
  }
  // What the run holds on the device at its peak with everything resident, in floats a cell. Stepping every cell
  // at once: an iteration's graph over the whole grid, Dim * (17 + the smoothing passes) for the field, its state and
  // the corners the warp reads, six a channel (the two maps, the features sampled, their difference and the gradients
  // back), and a channel and four more when the cells that count are drawn (a mask, SamplingPercentage). Otherwise,
  // the boxes drawn or every cell by bricks: the maps (both images' in Static mode; in the Jacobian mode the fixed
  // one's, one an axis for a swept model, with the image's voxels along it, and none on patches), the control grid,
  // its gradient, Adam's two moments and the masks' cells, and a brick or a box at a time beside them. Measured in 3D
  // on the CPU and on CUDA, on 1 to 48 channels and 0 to 3 passes: a run held between 0.77 and 0.96 of that count,
  // the corners of the warp aside (0.80 to 0.82 with them, on one channel and one pass).
  const auto smoothingPasses = static_cast<int64_t>(m_ControlGridSmoothingIterations);
  const bool drawsCells = m_FixedMask.IsNotNull() || m_MovingMask.IsNotNull() || m_SamplingPercentage < 1.0;
  double     mapFloats = jacobian ? (sampledJacobian || swept ? 0.0 : 1.0) * channelCount : 2.0 * channelCount;
  for (unsigned int d = 0; swept && d < Dim; ++d)
  {
    mapFloats += static_cast<double>(channelCount * spatial[d]) / static_cast<double>(cells[d]);
  }
  // A refresh's reference field, and its residual through the graph of a dense iteration.
  const double heldFloats = mapFloats + static_cast<double>((refreshes ? 5 : 4) * Dim) + 2.0;
  // A spatial distance's windows, a channel: the five moments, their combinations and the gradients back. With LNCC
  // (windows of 5 cells, 6 channels, on the CPU) a dense run held 0.78 of the count, a brick's step 0.80 of BrickBytes.
  const int64_t windowFloats = spatialLoss ? 8 : 0;
  const int64_t denseFloats = static_cast<int64_t>(Dim) * (17 + smoothingPasses + (refreshes ? 3 : 0)) +
                              (6 + windowFloats) * channelCount + (drawsCells ? channelCount + 4 : 0);
  const double budget =
    m_ResidentFeatureBudget > 0 ? static_cast<double>(m_ResidentFeatureBudget) : Impact::ResidentBudget(device);
  // In the Jacobian mode a unit's features are computed on its voxels and `spanExtra` around them (the models' reach,
  // the patches' half-diagonal and a voxel for the interpolation), and the field is read at each of those voxels.
  std::vector<int64_t> spanExtra(Dim, 0);
  int64_t              spanAlign = 1;
  if (sampledJacobian)
  {
    double radius = 0.0; // the patches' half-diagonal, in millimetres
    for (const ImpactModelConfiguration & config : m_FixedModelsConfiguration)
    {
      double squares = 0.0;
      for (unsigned int k = 0; k < config.GetDimension(); ++k)
      {
        const double step =
          k < config.GetVoxelSize().size() && config.GetVoxelSize()[k] > 0.0f
            ? config.GetVoxelSize()[k]
            : (config.GetDimension() == Dim ? fixedSpacing[k]
                                            : *std::min_element(fixedSpacing.begin(), fixedSpacing.end()));
        squares += std::pow(static_cast<double>(config.GetPatchSize()[k] / 2) * step, 2);
      }
      radius = std::max(radius, std::sqrt(squares));
    }
    for (unsigned int d = 0; d < Dim; ++d)
    {
      spanExtra[d] = static_cast<int64_t>(std::ceil(radius / fixedSpacing[Dim - 1 - d])) + 1;
    }
  }
  else if (jacobian || refreshes)
  {
    for (const Impact::RegionLattices & on : latticeSets)
    {
      spanAlign = std::max(spanAlign, on.align);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        spanExtra[d] = std::max(spanExtra[d], on.reach[d]);
      }
    }
  }
  // How many cells past a unit's own the field it is warped by is read: each voxel of its span reads its displacement
  // from the two cells around it. One at least, the regularizer's next cell.
  int64_t fieldReach = 1;
  for (unsigned int d = 0; (jacobian || refreshes) && d < Dim; ++d)
  {
    const float scale = static_cast<float>(cells[d]) / static_cast<float>(spatial[d]);
    auto        cellOf = [&](int64_t voxel) {
      return std::min(
        static_cast<int64_t>(std::floor(std::max(scale * (static_cast<float>(voxel) + 0.5f) - 0.5f, 0.0f))),
        cells[d] - 1);
    };
    for (int64_t c = 0; c < cells[d]; ++c)
    {
      // The samples of the models' grid a cell's features read (with a voxel size, the voxels those are resampled
      // from); the voxels past the last cell go with it, a swept axis's or a drawn point's.
      const int64_t margin = std::max<int64_t>(0, featureWindows[d].starts[c] - spanExtra[d]);
      const int64_t high =
        std::min(featureSpatial[d], (c == cells[d] - 1 ? featureSpatial[d] : featureWindows[d].ends[c]) + spanExtra[d]);
      const auto [low, end] = grid.Voxels(d, margin - margin % spanAlign, high, spatial[d]);
      fieldReach = std::max({ fieldReach, c - cellOf(low), cellOf(end - 1) + 1 - c });
    }
  }
  // What a unit's step reads around it: the field its cells and a spatial distance's halo read, in Static mode the
  // regularizer's next cell at least, with the smoothing's reach.
  const int64_t reach = (jacobian ? fieldReach + lossHalo : std::max<int64_t>(1, lossHalo)) +
                        static_cast<int64_t>(m_ControlGridSmoothingIterations);
  // What a Jacobian step holds on the device for each voxel of its span: the warp (the position, the displacement and
  // the value, with their gradients) and the network's activations, forward and backward, measured on CUDA on the
  // probe region, else taken as four floats a channel at each layer's samples. On patches, the two images' reads; the
  // patches' own go by batches measured on the device.
  double voxelBytes = sampledJacobian ? 16.0 : 0.0;
  if (jacobian && !sampledJacobian)
  {
    const int            axis = swept ? 0 : -1;
    std::vector<int64_t> shape(Dim + 2, Impact::LatticeProbe);
    shape[0] = shape[1] = 1;
    const torch::Tensor probe = torch::zeros(shape, onDevice).requires_grad_(true);
    const int64_t       measured = Impact::PeakDeviceBytes(device, [&] {
      torch::Tensor total = torch::zeros({}, onDevice);
      for (const torch::Tensor & layer : layersOf(probe, true, true, axis))
      {
        total = total + layer.sum();
      }
      torch::autograd::grad({ total }, { probe });
    });
    double              estimate = 0.0;
    for (size_t l = 0; l < channels.size(); ++l)
    {
      double samples = 4.0 * 4.0 * static_cast<double>(channels[l]);
      for (const int64_t stride : latticesAlong(axis).moving[l].stride)
      {
        samples /= static_cast<double>(stride);
      }
      estimate += samples;
    }
    voxelBytes = (measured > 0 ? static_cast<double>(measured) / static_cast<double>(probe.numel()) : estimate) +
                 4.0 * (4.0 * static_cast<double>(Dim) + 2.0);
  }
  // A brick: as many boxes a side as BrickBytes holds on the device. At its peak a step holds there, in floats,
  // Dim * (6 + the smoothing passes) for each cell the brick reads (the field along the graph) and four for each
  // channel of each of its own cells (the two maps, the features sampled and their gradient): measured in 3D on
  // 1 to 48 channels and 1 and 3 passes, a step held between 0.66 and 1.0 of that count. A drawn box is a unit
  // by itself.
  const double brickBytes = m_BrickBytes > 0 ? static_cast<double>(m_BrickBytes) : Impact::ResidentBudget(device) / 8.0;
  auto         brickCells = [&](int64_t boxesASide, int64_t around) {
    double count = 1.0;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      count *= static_cast<double>(std::min(cells[d], boxesASide * perBox[d] + 2 * around));
    }
    return count;
  };
  // A brick's loss is over its own cells and reads the field `reach` around them; a drawn box's loss is over the
  // cells within `reach` of it (every term that reads its cells), which read the field twice that around it. The
  // channels are read on the cells and a spatial distance's halo.
  const int64_t partReach = drawsBoxes ? reach : 0;
  const double  channelFloats = static_cast<double>(4 + windowFloats);
  auto          stepBytes = [&](int64_t boxesASide) {
    double spanVoxels = 1.0;
    for (unsigned int d = 0; d < Dim; ++d)
    {
      spanVoxels *= static_cast<double>(
        std::min(featureSpatial[d],
                 std::min(cells[d], boxesASide * perBox[d] + 2 * (partReach + lossHalo)) * featureCellVoxels[d] +
                   2 * spanExtra[d]));
    }
    return 4.0 * (static_cast<double>(Dim * (6 + m_ControlGridSmoothingIterations)) *
                    brickCells(boxesASide, partReach + reach) +
                  channelFloats * static_cast<double>(channelCount) * brickCells(boxesASide, partReach + lossHalo)) +
           voxelBytes * spanVoxels;
  };
  // The dense iterations when their graph fits; else the maps and the state stay resident when they fit with the
  // smallest step beside them (a box), and the iterations go by bricks (or drawn boxes) on them; else everything is in
  // files. The Jacobian mode, which reads the moving image at every iteration, goes by units.
  const bool dense = !jacobian && !drawsBoxes && static_cast<double>(cellCountTotal * denseFloats * 4) <= budget;
  m_ResidentBytes = static_cast<SizeValueType>(static_cast<double>(cellCountTotal) *
                                               (dense ? static_cast<double>(denseFloats) : heldFloats) * 4.0);
  m_ResidentFeatures = static_cast<double>(m_ResidentBytes) + (dense ? 0.0 : stepBytes(1)) <= budget;
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
    int64_t                 count = box.Count();
    if (fixedOn.defined())
    {
      std::vector<int64_t> lowPad(Dim);
      const RegionType     span = spanOf(box.cell0, box.cell1, noHalo, lowPad);
      const torch::Tensor  on =
        toCells(readMask(fixedMask.GetPointer(), span), Impact::ImageLattice(Dim), lowPad, box.cell0, box.cell1) >= 0.5;
      Impact::WriteCells(fixedOn, box.cell0, on);
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
  torch::Tensor theta = atHome(thetaShape);
  if (m_InitialDisplacementField.IsNotNull())
  {
    // Read by regions through the field's own pipeline, as the images are: for the cells of a box, the region of
    // the field around the fixed voxels they blend.
    DisplacementFieldType * const initial = m_InitialDisplacementField;
    initial->UpdateOutputInformation();
    using FieldInterpolatorType = VectorLinearInterpolateImageFunction<DisplacementFieldType, double>;
    auto interpolator = FieldInterpolatorType::New();
    // Filled in host memory: the mapped grid itself, or a grid the device then takes.
    torch::Tensor start = m_ResidentFeatures ? torch::zeros(thetaShape, torch::kFloat32) : theta;
    float * const buffer = start.data_ptr<float>();
    const auto &  direction = m_FixedImage->GetDirection();
    // Resizing the dense field to the control grid blends, along each axis, the two voxels around the cell's
    // centre. The field is read at those voxels: one given on a coarser grid is then its dense field, resized.
    // The lower of the two along tensor axis d for cell c, and the upper one's share.
    auto voxelBelow = [&](unsigned int d, int64_t c, double & share) {
      const double centre = std::max(
        (static_cast<double>(c) + 0.5) * static_cast<double>(spatial[d]) / static_cast<double>(cells[d]) - 0.5, 0.0);
      const int64_t below = std::min(static_cast<int64_t>(centre), spatial[d] - 1);
      share = centre - static_cast<double>(below);
      return below;
    };
    auto voxelAt = [&](unsigned int a, int64_t offset) {
      return fixedRegion.GetIndex()[a] + std::min(offset, static_cast<int64_t>(fixedRegion.GetSize()[a]) - 1);
    };
    for (const Impact::CellBox & box : boxes)
    {
      // The field's region that holds the fixed voxels the box's cells blend, and a sample around for round-off.
      RegionType voxels;
      for (unsigned int a = 0; a < Dim; ++a)
      {
        double             unused;
        const unsigned int d = Dim - 1 - a;
        const auto         first = voxelAt(a, voxelBelow(d, box.cell0[d], unused));
        voxels.SetIndex(a, first);
        voxels.SetSize(a,
                       static_cast<SizeValueType>(voxelAt(a, voxelBelow(d, box.cell1[d] - 1, unused) + 1) - first + 1));
      }
      auto read = ImageAlgorithm::EnlargeRegionOverBox(voxels, m_FixedImage.GetPointer(), initial);
      read.PadByRadius(1);
      if (!read.Crop(initial->GetLargestPossibleRegion()))
      {
        continue; // the field does not reach the box: its cells start at zero
      }
      initial->SetRequestedRegion(read);
      initial->Update();
      interpolator->SetInputImage(initial); // which takes the bounds of what is buffered
      std::vector<int64_t> cell = box.cell0;
      for (bool more = true; more;)
      {
        int64_t                  flat = 0;
        std::array<int64_t, Dim> below{};
        std::array<double, Dim>  share{};
        for (unsigned int d = 0; d < Dim; ++d)
        {
          flat = flat * cells[d] + cell[d];
          below[Dim - 1 - d] = voxelBelow(d, cell[d], share[Dim - 1 - d]);
        }
        Vector<double, Dim> u; // millimetres, ITK order
        u.Fill(0.0);
        bool inside = true;
        for (unsigned int corner = 0; inside && corner < (1u << Dim); ++corner)
        {
          double                             weight = 1.0;
          typename FixedImageType::IndexType voxel;
          for (unsigned int a = 0; a < Dim; ++a)
          {
            const bool above = (corner >> a) & 1u;
            weight *= above ? share[a] : 1.0 - share[a];
            voxel[a] = voxelAt(a, below[a] + (above ? 1 : 0));
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
        for (unsigned int d = 0; inside && d < Dim; ++d)
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
        more = false; // the next cell of the box, the last axis fastest
        for (int d = static_cast<int>(Dim) - 1; !more && d >= 0; --d)
        {
          more = ++cell[d] < box.cell1[d];
          if (!more)
          {
            cell[d] = box.cell0[d];
          }
        }
      }
    }
    theta = m_ResidentFeatures ? start.to(device) : start;
  }
  // A refresh's reference: the field the moving maps were pooled at, zero until the first.
  torch::Tensor thetaRef = refreshes ? atHome(thetaShape) : torch::Tensor();
  // Epsilon weighs against the gradient per control cell, as in the dense run.
  const double adamEpsilon = m_Epsilon / static_cast<double>(std::max(1u, m_GridShrinkFactor));
  auto         smoothControl = [&](const torch::Tensor & control) {
    return Impact::SmoothControlField<Dim>(control, m_ControlGridSmoothingIterations);
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

  // ---- One pass over the boxes, the only time the images are read in Static mode: the pooled maps, on the device
  // when they fit its budget and in mapped files otherwise. The Jacobian mode reads the fixed maps there (one an axis
  // for a swept model, none on patches) and the moving image at every iteration. ----
  std::vector<torch::Tensor> fixedMaps(losses.size()), movingMaps(losses.size());
  torch::Tensor              movingShare;
  // A swept model's fixed maps and fixed mask, one an axis it sweeps, with the image's voxels along it.
  std::vector<std::vector<torch::Tensor>> sweptMaps(swept ? Dim : 0);
  std::vector<torch::Tensor>              sweptOn(swept ? Dim : 0);
  auto                                    atHomeShaped = [&](int64_t channel, int axis, bool mask) {
    std::vector<int64_t> shape{ 1, channel };
    for (unsigned int d = 0; d < Dim; ++d)
    {
      shape.push_back(static_cast<int64_t>(d) == keepOf(axis) ? spatial[d] : cells[d]);
    }
    if (!mask)
    {
      return atHome(shape);
    }
    return m_ResidentFeatures ? torch::zeros(shape, torch::TensorOptions().dtype(torch::kBool).device(device))
                                                                 : scratch.Zeros(shape, torch::kBool);
  };
  for (size_t l = 0; l < losses.size() && !sampledJacobian && !swept; ++l)
  {
    fixedMaps[l] = atHomeShaped(channels[l], -1, false);
    if (!jacobian)
    {
      movingMaps[l] = atHomeShaped(channels[l], -1, false);
    }
  }
  for (unsigned int axis = 0; axis < sweptMaps.size(); ++axis)
  {
    for (size_t l = 0; l < losses.size(); ++l)
    {
      sweptMaps[axis].push_back(atHomeShaped(channels[l], static_cast<int>(axis), false));
    }
    if (m_FixedMask.IsNotNull())
    {
      sweptOn[axis] = atHomeShaped(1, static_cast<int>(axis), true);
    }
  }
  if (m_MovingMask.IsNotNull() && !sampledJacobian && !swept)
  {
    movingShare = atHome(maskShape);
  }
  // Where the points of the cells [c0, c1) start in a swept model's maps along `axis`.
  auto sweptStart = [&](const std::vector<int64_t> & c0, int axis) {
    std::vector<int64_t> start = c0;
    const int64_t        keep = keepOf(axis);
    start[keep] = windows[keep].starts[c0[keep]];
    return start;
  };
  for (size_t b = 0; b < boxes.size(); ++b)
  {
    const Impact::CellBox & box = boxes[b];
    if (countByBox[b] > 0 && !sampledJacobian && !swept)
    {
      const std::vector<torch::Tensor> layers = pooledFixed(box.cell0, box.cell1);
      for (size_t l = 0; l < layers.size(); ++l)
      {
        Impact::WriteCells(fixedMaps[l], box.cell0, layers[l]);
      }
    }
    for (int axis = 0; axis < static_cast<int>(sweptMaps.size()); ++axis)
    {
      // Every box: a point counts by its own share of the fixed mask, which pooling along fewer axes gives it.
      const std::vector<torch::Tensor> layers = pooledFixed(box.cell0, box.cell1, axis);
      for (size_t l = 0; l < layers.size(); ++l)
      {
        Impact::WriteCells(sweptMaps[axis][l], sweptStart(box.cell0, axis), layers[l]);
      }
      if (sweptOn[axis].defined())
      {
        std::vector<int64_t> lowPad(Dim);
        const RegionType     span = featureSpan(box.cell0, box.cell1, lowPad, axis);
        Impact::WriteCells(
          sweptOn[axis],
          sweptStart(box.cell0, axis),
          toCells(readFixedMask(span), Impact::ImageLattice(Dim), lowPad, box.cell0, box.cell1, keepOf(axis)) >= 0.5);
      }
    }
    if (!jacobian)
    {
      // The moving map everywhere: the displacement reads it past the fixed mask.
      PooledMoving pooled = pooledMoving(box.cell0, box.cell1);
      for (size_t l = 0; l < pooled.layers.size(); ++l)
      {
        Impact::WriteCells(movingMaps[l], box.cell0, pooled.layers[l]);
      }
      if (pooled.share.defined())
      {
        Impact::WriteCells(movingShare, box.cell0, pooled.share);
      }
    }
    else if (movingShare.defined())
    {
      Impact::WriteCells(movingShare, box.cell0, movingShareOf(box.cell0, box.cell1));
    }
  }
  // FeatureMapUpdateInterval, after the step of every that many iterations but the last: the field becomes the
  // reference the moving maps are read past, and they are pooled again from the moving image it warps, a box at a
  // time, as the whole-image run re-extracts them.
  auto refresh = [&](unsigned int iteration) {
    if (!refreshes || (iteration + 1) % static_cast<unsigned int>(m_FeatureMapUpdateInterval) != 0 ||
        iteration + 1 >= m_NumberOfIterations)
    {
      return;
    }
    torch::NoGradGuard noGrad;
    thetaRef.copy_(theta.detach());
    const std::vector<int64_t> around(Dim, fieldReach + static_cast<int64_t>(m_ControlGridSmoothingIterations));
    // The field of the cells around `box`, and where it starts.
    auto fieldAround = [&](const Impact::CellBox & box) {
      const Impact::CellBox window = Impact::CellsAround(box, around, cells);
      return std::make_pair(smoothControl(Impact::NarrowTo(thetaRef, window.cell0, window.cell1).to(device)),
                            window.cell0);
    };
    // The moving image the field now warps: its normalisations' statistics, and with PCA its features' mean.
    if (gathersNorms)
    {
      movingNorms = gatherNorms(true, [&](const Impact::CellBox & box, const RegionType & span) {
        const auto [smoothed, origin] = fieldAround(box);
        return warpedOn(readMovingImage, span, smoothed, origin);
      });
    }
    if (std::any_of(pcaBasis.begin(), pcaBasis.end(), [](const torch::Tensor & basis) { return basis.defined(); }))
    {
      movingPcaMean = meansOver(true, pcaBasis, [&](const Impact::CellBox & box, const RegionType & span) {
        const auto [smoothed, origin] = fieldAround(box);
        return warpedOn(readMovingImage, span, smoothed, origin);
      });
    }
    for (const Impact::CellBox & box : boxes)
    {
      const auto [smoothed, origin] = fieldAround(box);
      std::vector<torch::Tensor> layers = pooledWarped(box.cell0, box.cell1, smoothed, origin, -1);
      reduced(layers, pcaBasis, movingPcaMean);
      for (size_t l = 0; l < layers.size(); ++l)
      {
        Impact::WriteCells(movingMaps[l], box.cell0, layers[l]);
      }
    }
  };
  if (dense)
  {
    // ---- The dense iterations on the pooled maps. ----
    theta.set_requires_grad(true);
    Impact::CellAdam adam{
      torch::zeros_like(theta), torch::zeros_like(theta), m_LearningRate, m_Beta1, m_Beta2, adamEpsilon
    };
    const std::vector<torch::Tensor> base = Impact::BoxIndices(wholeStart, cells, device);
    m_MetricValuesPerIteration.clear();
    m_MetricValuesPerIteration.reserve(m_NumberOfIterations);
    for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
    {
      drawIteration = iteration;
      drawSubsets(channels);
      // The moving maps are read past a refresh's reference, the moving mask at the whole field.
      const torch::Tensor smoothed = smoothControl(theta);
      const torch::Tensor displacement = (refreshes ? smoothControl(theta - thetaRef) : smoothed) * voxelsToCells;
      torch::Tensor       counted = fixedOn;
      if (movingShare.defined())
      {
        torch::NoGradGuard  noGrad;
        const torch::Tensor holds =
          Impact::WarpMap<Dim>(movingShare, wholeStart, cells, base, smoothed.detach() * voxelsToCells) >= 0.5;
        counted = counted.defined() ? counted.logical_and(holds) : holds;
      }
      const torch::Tensor rows = drawRows(counted, wholeStart, cells);
      torch::Tensor       total = torch::zeros({}, onDevice);
      if (!rows.defined() || rows.numel() > 0)
      {
        for (size_t l = 0; l < losses.size(); ++l)
        {
          const torch::Tensor moving =
            Impact::WarpMap<Dim>(pick(l, movingMaps[l]), wholeStart, cells, base, displacement);
          const torch::Tensor value =
            losses[l]->IsSpatial()
              ? losses[l]->forwardSpatial(pick(l, fixedMaps[l]), moving, windowOf(fixedMaps[l]), counted)
              : rowsLoss(l, pick(l, fixedMaps[l]), moving, rows);
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
        ++adam.steps;
        adam.StepCells(theta, theta.grad(), wholeStart, cells, device);
        theta.grad().zero_();
      }
      m_MetricValuesPerIteration.push_back(loss.template item<double>());
      refresh(iteration);
    }
  }
  else
  {
    // ---- By units of the grid, brought to the device one at a time: the control grid, its gradient and Adam's
    // moments are at home like the maps, on the device when they are resident. RegionsPerIteration 0 steps every
    // cell at every iteration, the whole-image run's schedule, by bricks of boxes; otherwise an iteration steps the
    // cells of the boxes it draws. The loss of a set of cells is the dense loss restricted to them, weighed by their
    // share of the counted cells, plus the regularizer's differences whose first cell lies among them. ----
    torch::Tensor    gradient = atHome(thetaShape);
    Impact::CellAdam adam{ atHome(thetaShape), atHome(thetaShape), m_LearningRate, m_Beta1, m_Beta2, adamEpsilon };
    // The cells within `reach` of a box: the field its features read and the regularizer's next cell, with the
    // smoothing's reach. They are the cells the box's loss reads (smoothed alone, they give the smoothed field of the
    // whole grid where the box reads it), and the cells whose loss reads the box's.
    auto around = [&](const Impact::CellBox & box) {
      return Impact::CellsAround(box, std::vector<int64_t>(Dim, reach), cells);
    };
    int64_t      boxesASide = 1;
    const double brickRoom =
      m_ResidentFeatures ? std::min(brickBytes, budget - static_cast<double>(m_ResidentBytes)) : brickBytes;
    while (!drawsBoxes && brickCells(boxesASide, 0) < static_cast<double>(cellCountTotal) &&
           stepBytes(boxesASide + 1) <= brickRoom)
    {
      ++boxesASide;
    }
    // The units' counts are taken again where they change: with a moving mask the cells that count follow the field,
    // the sampled points change with every draw, and a swept model's points with the axis drawn, of which each has its
    // own set of counts.
    // Static mode's draws (SamplingPercentage < 1) change the cells that count at every iteration too.
    const bool recounts = movingShare.defined() || swept || sampledJacobian || m_SamplingPercentage < 1.0;
    std::vector<Impact::CellBox>      units;
    std::vector<int64_t>              perUnit(Dim), unitsAlong(Dim);
    std::vector<size_t>               order;
    std::vector<int64_t>              stepsOf;      // a unit's steps: Adam corrects its cells' moments by that count
    std::vector<double>               countedCells; // the cells that count over the grid, the dense mean's divisor
    std::vector<std::vector<int64_t>> survivors;    // each unit's, as of the field it was last counted at
    std::vector<bool>                 countedOnce;
    auto                              tile = [&](int64_t side, int64_t steps) {
      for (unsigned int d = 0; d < Dim; ++d)
      {
        perUnit[d] = side * perBox[d];
        unitsAlong[d] = (cells[d] + perUnit[d] - 1) / perUnit[d];
      }
      units = Impact::TileCells(cells, perUnit);
      order.resize(units.size());
      std::iota(order.begin(), order.end(), 0);
      stepsOf.assign(units.size(), steps);
      const size_t sets = swept ? Dim : 1;
      countedCells.assign(sets, recounts ? 0.0 : totalCount);
      survivors.assign(sets, std::vector<int64_t>(units.size(), 0));
      countedOnce.assign(sets, false);
    };
    tile(boxesASide, 0);
    unsigned int sweptAxis = 0; // the image axis a swept model sweeps at this iteration, drawn as the whole-image run
    std::mt19937 axisGenerator(m_Seed);
    std::uniform_int_distribution<unsigned int> axisDistribution(0, Dim - 1);
    auto countSet = [&] { return swept ? static_cast<size_t>(sweptAxis) : size_t{ 0 }; };
    // The points of `box` in a swept model's maps along the axis swept: its cells, with the image's voxels along it.
    auto sweptPoints = [&](const Impact::CellBox & box) {
      Impact::CellBox points = box;
      const int64_t   keep = keepOf(static_cast<int>(sweptAxis));
      points.cell0[keep] = windows[keep].starts[box.cell0[keep]];
      points.cell1[keep] = box.cell1[keep] == cells[keep] ? spatial[keep] : windows[keep].ends[box.cell1[keep] - 1];
      return points;
    };
    // How the cells of `box` read the moving map for the smoothed field `smoothed` of the cells from `origin` on:
    // the cells' samples and their displacement, the cells [m0, m1) of the map they read, and the cells that count, by
    // both masks at that field (`counted`, undefined when all do; `count` of them). A box the fixed mask leaves out
    // reads nothing. A swept model's points count by the fixed mask's share of each and the moving mask warped by the
    // field.
    struct Reading
    {
      std::vector<torch::Tensor> index;
      torch::Tensor              displacement;
      torch::Tensor              counted;
      std::vector<int64_t>       m0;
      std::vector<int64_t>       m1;
      int64_t                    count;
    };
    auto readingOf = [&](const Impact::CellBox &      box,
                         const torch::Tensor &        smoothed,
                         const std::vector<int64_t> & origin,
                         const torch::Tensor &        residual = {}) -> Reading {
      if (swept)
      {
        const Impact::CellBox points = sweptPoints(box);
        Reading               reading{ {}, {}, {}, {}, {}, points.Count() };
        if (m_FixedMask.IsNotNull())
        {
          reading.counted = Impact::NarrowTo(sweptOn[sweptAxis], points.cell0, points.cell1).to(device);
          reading.count = reading.counted.sum().template item<int64_t>();
        }
        if (reading.count > 0 && m_MovingMask.IsNotNull())
        {
          torch::NoGradGuard   noGrad;
          std::vector<int64_t> lowPad(Dim);
          const RegionType     span = featureSpan(box.cell0, box.cell1, lowPad, static_cast<int>(sweptAxis));
          const torch::Tensor  holds =
            toCells((warpedOn(readMovingMask, span, smoothed.detach(), origin) >= 0.5).to(torch::kFloat32),
                    Impact::ImageLattice(Dim),
                    lowPad,
                    box.cell0,
                    box.cell1,
                    keepOf(static_cast<int>(sweptAxis))) >= 0.5;
          reading.counted = reading.counted.defined() ? reading.counted.logical_and(holds) : holds;
          reading.count = reading.counted.sum().template item<int64_t>();
        }
        return reading;
      }
      Reading reading{ {}, {}, {}, std::vector<int64_t>(Dim), std::vector<int64_t>(Dim), box.Count() };
      if (fixedOn.defined())
      {
        reading.counted = Impact::NarrowTo(fixedOn, box.cell0, box.cell1).to(device);
        reading.count = reading.counted.sum().template item<int64_t>();
      }
      if (reading.count == 0)
      {
        return reading;
      }
      // Where each cell reads a map of the moving image by `field`: its displacement in cells (voxels -> cells), and
      // the cells [m0, m1) of the map the box reads: what c + u reaches, one around for the interpolation and one
      // spare, within the lattice, at least two wide. The moving maps are read past a refresh's reference
      // (`residual`), the moving mask at the whole field.
      reading.index = Impact::BoxIndices(box.cell0, box.cell1, device);
      auto readAt = [&](const torch::Tensor & field, std::vector<int64_t> & m0, std::vector<int64_t> & m1) {
        torch::Tensor own = field;
        for (unsigned int d = 0; d < Dim; ++d)
        {
          own = own.narrow(2 + static_cast<int64_t>(d), box.cell0[d] - origin[d], box.cell1[d] - box.cell0[d]);
        }
        const torch::Tensor displacement = own * voxelsToCells;
        const torch::Tensor q = cellPositions(box.cell0, box.cell1) + displacement;
        {
          torch::NoGradGuard  noGrad;
          const torch::Tensor reached = q.detach().flatten(2);
          const torch::Tensor bounds = torch::stack({ std::get<0>(reached.min(2)), std::get<0>(reached.max(2)) }, 0)
                                         .to(torch::kCPU); // {2, 1, Dim}
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
        return displacement;
      };
      reading.displacement = readAt(residual.defined() ? residual : smoothed, reading.m0, reading.m1);
      if (movingShare.defined())
      {
        torch::NoGradGuard   noGrad;
        std::vector<int64_t> s0 = reading.m0, s1 = reading.m1;
        const torch::Tensor at = residual.defined() ? readAt(smoothed.detach(), s0, s1) : reading.displacement.detach();
        const torch::Tensor holds =
          Impact::WarpMap<Dim>(Impact::NarrowTo(movingShare, s0, s1).to(device), s0, cells, reading.index, at) >= 0.5;
        reading.counted = reading.counted.defined() ? reading.counted.logical_and(holds) : holds;
        reading.count = reading.counted.sum().template item<int64_t>();
      }
      return reading;
    };
    // A global distance (NCC, OrganDice) is a function of its sums over the points that count (Impact::PointSums).
    // An iteration first gathers them over every unit without the graph (`gathering`), then takes the gradient of each
    // such layer's value with respect to them (`sumSeeds`); a unit's step back-propagates the seeds times its own sums,
    // a term of no value whose gradient is the unit's part of the whole one.
    auto isGlobal = [&](size_t l) { return !losses[l]->IsPerPointMean() && !losses[l]->IsSpatial(); };
    bool globalLoss = false;
    for (size_t l = 0; l < losses.size(); ++l)
    {
      globalLoss = globalLoss || isGlobal(l);
    }
    bool   gathering = false;
    double gatheredPoints = 0.0;
    // With PCA the moving features of the Jacobian mode are centred by their mean at the field of the iteration
    // (below): over every voxel, a swept model's over the slices PcaSlices reads along the axis of the iteration.
    std::vector<torch::Tensor> movingMeanNow(losses.size());
    // On patches, with PCA, the basis is fitted at every iteration on the fixed features of its points and each side is
    // centred by its own mean over them, as the whole-image run fits and centres them: their sums are gathered over
    // every unit first, without the graph (`gatheringPca`).
    bool                         gatheringPca = false;
    std::vector<Impact::PcaSums> pointFixedSums(losses.size()), pointMovingSums(losses.size());
    std::vector<torch::Tensor> pointBasis(losses.size()), pointFixedMean(losses.size()), pointMovingMean(losses.size());
    auto pointsProjected = [&](size_t l, const torch::Tensor & centres, const torch::Tensor & mean) {
      return pointBasis[l].defined() ? Impact::PcaTransform(centres.t(), pointBasis[l], mean).t() : centres;
    };
    std::vector<torch::Tensor> gatheredSums(losses.size()), sumSeeds(losses.size()), globalValues(losses.size());
    auto                       globalTerm = [&](size_t l, const torch::Tensor & sums) -> torch::Tensor {
      if (gathering)
      {
        gatheredSums[l] = gatheredSums[l].defined() ? gatheredSums[l] + sums : sums;
        return torch::zeros({}, onDevice);
      }
      return layerWeights[l] * normalized(l, (sumSeeds[l] * (sums - sums.detach())).sum());
    };
    // The loss of the cells of `box` for the smoothed field `smoothed` of the cells from `origin` on; `raw`, when
    // given, takes each layer's unweighted value times the box's share. The Jacobian mode compares the fixed maps with
    // the moving features re-extracted from the image the field warps. A spatial distance takes its terms on the block
    // of the box and `lossHalo` cells around it, and averages those of the box's points that count. A global distance's
    // layer gathers its sums, or takes its term of the gradient, and adds nothing to `raw`; gathering, the other
    // layers and the regularizer are left out.
    auto partLoss = [&](const Impact::CellBox &      box,
                        const torch::Tensor &        smoothed,
                        const std::vector<int64_t> & origin,
                        std::vector<double> *        raw,
                        const torch::Tensor &        residual) -> torch::Tensor {
      const Reading reading = readingOf(box, smoothed, origin, residual);
      if (reading.count == 0)
      {
        // Nothing of it counts: its edges alone, no map read.
        return regularizer(smoothed, box.cell0, box.cell1, origin);
      }
      const torch::Tensor rows = drawRows(reading.counted, box.cell0, box.cell1);
      const double        share =
        static_cast<double>(m_SamplingPercentage < 1.0 ? rows.numel() : reading.count) / countedCells[countSet()];
      const Impact::CellBox block = Impact::CellsAround(box, std::vector<int64_t>(Dim, lossHalo), cells);
      const Impact::CellBox points = swept ? sweptPoints(box) : box;
      if (gathering)
      {
        gatheredPoints += static_cast<double>(rows.defined() ? rows.numel() : points.Count());
      }
      const Impact::CellBox blockPoints = swept ? sweptPoints(block) : block;
      // The box's points within a map of the block's.
      auto ownOf = [&](torch::Tensor map) {
        for (unsigned int d = 0; d < Dim; ++d)
        {
          map = map.narrow(
            2 + static_cast<int64_t>(d), points.cell0[d] - blockPoints.cell0[d], points.cell1[d] - points.cell0[d]);
        }
        return map;
      };
      std::vector<torch::Tensor> warped =
        jacobian ? pooledWarped(block.cell0, block.cell1, smoothed, origin, swept ? static_cast<int>(sweptAxis) : -1)
                 : std::vector<torch::Tensor>{};
      if (jacobian)
      {
        reduced(warped, swept ? sweptBasis[sweptAxis] : pcaBasis, movingMeanNow);
      }
      const Reading blockReading = spatialLoss && !jacobian ? readingOf(block, smoothed, origin, residual) : Reading{};
      torch::Tensor kept; // the block's points a spatial distance averages: the box's that count
      if (spatialLoss)
      {
        std::vector<int64_t> shape{ 1, 1 };
        for (unsigned int d = 0; d < Dim; ++d)
        {
          shape.push_back(blockPoints.cell1[d] - blockPoints.cell0[d]);
        }
        kept = torch::zeros(shape, torch::TensorOptions().dtype(torch::kBool).device(device));
        ownOf(kept).fill_(true);
        if (reading.counted.defined())
        {
          ownOf(kept).logical_and_(reading.counted);
        }
      }
      // Static mode: layer l's moving map on the cells of a reading `at`, warped from the pooled maps.
      auto movingOn = [&](size_t l, const Reading & at) {
        return Impact::WarpMap<Dim>(
          pick(l, Impact::NarrowTo(movingMaps[l], at.m0, at.m1).to(device)), at.m0, cells, at.index, at.displacement);
      };
      torch::Tensor total = torch::zeros({}, onDevice);
      if (!rows.defined() || rows.numel() > 0)
      {
        for (size_t l = 0; l < losses.size(); ++l)
        {
          const torch::Tensor & fixedMap = swept ? sweptMaps[sweptAxis][l] : fixedMaps[l];
          if (isGlobal(l))
          {
            if (raw == nullptr)
            {
              const auto [f, m] = rowsOf(pick(l, Impact::NarrowTo(fixedMap, points.cell0, points.cell1).to(device)),
                                         jacobian ? pick(l, ownOf(warped[l])) : movingOn(l, reading),
                                         rows);
              total = total + globalTerm(l, Impact::PointSums(f, m));
            }
            continue;
          }
          if (gathering)
          {
            continue;
          }
          torch::Tensor value;
          if (losses[l]->IsSpatial())
          {
            const torch::Tensor fixedBlock =
              Impact::NarrowTo(fixedMap, blockPoints.cell0, blockPoints.cell1).to(device);
            value = losses[l]->forwardSpatial(
              pick(l, fixedBlock), jacobian ? pick(l, warped[l]) : movingOn(l, blockReading), windowOf(fixedMap), kept);
          }
          else
          {
            const torch::Tensor fixedLayer = Impact::NarrowTo(fixedMap, points.cell0, points.cell1).to(device);
            value = rowsLoss(l, pick(l, fixedLayer), jacobian ? pick(l, ownOf(warped[l])) : movingOn(l, reading), rows);
          }
          if (raw != nullptr)
          {
            (*raw)[l] += value.template item<double>() * share;
          }
          total = total + layerWeights[l] * normalized(l, value) * share;
        }
      }
      return gathering ? total : total + regularizer(smoothed, box.cell0, box.cell1, origin);
    };
    // The sampled Jacobian mode's draws of an iteration, as the whole-image run draws them (flat voxel numbers, ITK x
    // fastest), and grouped by the unit their voxel's cell lies in, the voxels past the last cell with it: unit u's are
    // [drawnStart[u], drawnStart[u + 1]) of `drawnByUnit`.
    // ponytail: an iteration's draws are held whole, 24 bytes a point at most, as the whole-image run holds them; a
    // generator per unit if they outgrow the host's memory, which gives up drawing what the whole-image run draws.
    torch::Tensor        draws, drawnByUnit;
    std::vector<int64_t> drawnStart;
    // The cell, along each tensor axis, of every voxel `flat` numbers ({n}, ITK x fastest): {n, Dim}, and the voxels'
    // ITK indices, {n, Dim}.
    auto cellsOfDraws = [&](const torch::Tensor & flat, torch::Tensor & indices) {
      torch::Tensor of = torch::empty({ flat.size(0), static_cast<int64_t>(Dim) }, torch::kLong);
      indices = torch::empty_like(of);
      int64_t stride = 1;
      for (unsigned int a = 0; a < Dim; ++a)
      {
        const unsigned int d = Dim - 1 - a;
        indices.select(1, a).copy_(flat.div(stride, "trunc").remainder(spatial[d]));
        of.select(1, d).copy_(indices.select(1, a).div(cellVoxels[d], "trunc").clamp_max(cells[d] - 1));
        stride *= spatial[d];
      }
      return of;
    };
    auto groupDraws = [&] {
      torch::Tensor       indices;
      const torch::Tensor of = cellsOfDraws(draws, indices);
      torch::Tensor       unit = torch::zeros({ draws.size(0) }, torch::kLong);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        unit = unit * unitsAlong[d] + of.select(1, d).div(perUnit[d], "trunc");
      }
      drawnByUnit = draws.index_select(0, std::get<1>(unit.sort(/*stable=*/true, /*dim=*/0, /*descending=*/false)));
      const torch::Tensor counts = torch::bincount(unit, {}, static_cast<int64_t>(units.size()));
      drawnStart.assign(1, 0);
      for (int64_t u = 0; u < counts.size(0); ++u)
      {
        drawnStart.push_back(drawnStart.back() + counts[u].template item<int64_t>());
      }
    };
    // The draws whose cell lies in `box`, as ITK indices {k, Dim} on the host.
    auto drawnIn = [&](const Impact::CellBox & box) -> torch::Tensor {
      std::vector<int64_t> u0(Dim), extent(Dim);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        u0[d] = box.cell0[d] / perUnit[d];
        extent[d] = (box.cell1[d] - 1) / perUnit[d] + 1 - u0[d];
      }
      std::vector<torch::Tensor> pieces;
      for (const Impact::CellBox & at : Impact::TileCells(extent, std::vector<int64_t>(Dim, 1)))
      {
        int64_t unit = 0;
        for (unsigned int d = 0; d < Dim; ++d)
        {
          unit = unit * unitsAlong[d] + u0[d] + at.cell0[d];
        }
        pieces.push_back(drawnByUnit.narrow(0, drawnStart[unit], drawnStart[unit + 1] - drawnStart[unit]));
      }
      torch::Tensor       indices;
      const torch::Tensor of = cellsOfDraws(torch::cat(pieces), indices);
      torch::Tensor       in = torch::ones({ of.size(0) }, torch::kBool);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        in = in & (of.select(1, d) >= box.cell0[d]) & (of.select(1, d) < box.cell1[d]);
      }
      return indices.index_select(0, in.nonzero().flatten());
    };
    std::vector<double> lastIndex(Dim);
    for (unsigned int a = 0; a < Dim; ++a)
    {
      lastIndex[a] = static_cast<double>(spatial[Dim - 1 - a] - 1);
    }
    const torch::Tensor upper = torch::tensor(lastIndex, torch::kFloat64).to(onDevice);
    // The sampled Jacobian mode's points in `box`, kept as the whole-image run keeps its draws: every model's patch
    // inside the image, the fixed mask holding at the point and the moving mask, warped by `smoothed` (the field of the
    // cells from `origin` on), at the point. {k, Dim}, ITK indices on the host.
    auto keptPoints =
      [&](const Impact::CellBox & box, const torch::Tensor & smoothed, const std::vector<int64_t> & origin) {
        torch::NoGradGuard noGrad;
        torch::Tensor      points = drawnIn(box);
        if (points.size(0) == 0)
        {
          return points;
        }
        torch::Tensor fits = torch::ones({ points.size(0) }, torch::TensorOptions().dtype(torch::kBool).device(device));
        for (size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
        {
          const ImpactModelConfiguration & config = m_FixedModelsConfiguration[i];
          const torch::Tensor              corners = Impact::SampledPatchPositions<Dim>(
            config, points, fixedSpacing, m_Seed, i, Impact::PatchCorners(config), onDevice);
          fits = fits & ((corners >= 0.0) & (corners <= upper)).flatten(1).all(1);
        }
        if (m_FixedMask.IsNotNull())
        {
          fits = fits & (readAt(readFixedMask, points.to(onDevice).flip(-1), {}) >= 0.5);
        }
        points = points.index_select(0, fits.nonzero().flatten().to(torch::kCPU));
        if (m_MovingMask.IsNotNull() && points.size(0) > 0)
        {
          const torch::Tensor centres = points.to(onDevice).flip(-1);
          const torch::Tensor holds = readAt(readMovingMask, centres, fieldAtPoints(smoothed, origin, centres)) >= 0.5;
          points = points.index_select(0, holds.nonzero().flatten().to(torch::kCPU));
        }
        return points;
      };
    std::vector<size_t> firstLayer{ 0 }; // model i's kept layers start at firstLayer[i] in the flat list
    for (const auto & config : m_FixedModelsConfiguration)
    {
      firstLayer.push_back(firstLayer.back() + NumberOfKeptLayers(config));
    }
    // The sampled Jacobian mode's loss of the points of `box` for the smoothed field `smoothed` of the cells from
    // `origin` on, as the whole-image run takes it: each model run on the patch around every point kept, cut in the
    // fixed image and in the moving image as the field warps it, a batch at a time, each batch back-propagated to
    // `smoothed` on its own; the regularizer's differences of `box` with them. `raw`, when given, takes the values
    // without the graph instead. The value of the loss.
    auto sampledPart = [&](const Impact::CellBox &      box,
                           const torch::Tensor &        smoothed,
                           const std::vector<int64_t> & origin,
                           std::vector<double> *        raw) -> double {
      const torch::Tensor reg = regularizer(smoothed, box.cell0, box.cell1, origin);
      const torch::Tensor kept = keptPoints(box, smoothed.detach(), origin);
      const int64_t       n = kept.size(0);
      const double        total = countedCells[0];
      double              value = reg.template item<double>();
      if (gathering)
      {
        gatheredPoints += static_cast<double>(n);
      }
      torch::Tensor leaf; // the field the patches are warped by: the batches' backwards meet there
      if (n > 0)
      {
        // Model i's patches of the points [begin, end), {b, 1, patch...}: in the fixed image, or in the moving image as
        // `field` warps it.
        auto patchesOf = [&](size_t i, int64_t begin, int64_t end, const torch::Tensor & field) {
          const ImpactModelConfiguration & config = m_FixedModelsConfiguration[i];
          const torch::Tensor              at = Impact::SampledPatchPositions<Dim>(
                                     config, kept.narrow(0, begin, end - begin), fixedSpacing, m_Seed, i, {}, onDevice)
                                     .flip(-1)
                                     .reshape({ -1, static_cast<int64_t>(Dim) });
          std::vector<int64_t> shape{ end - begin, 1 };
          for (const int64_t extent : Impact::PatchTensorShape(config))
          {
            shape.push_back(extent);
          }
          return (field.defined() ? readAt(readMovingImage, at, fieldAtPoints(field, origin, at))
                                  : readAt(readFixed, at, {}))
            .reshape(shape);
        };
        // A side's centre features of the batch [begin, end) of model i, as layer l compares them.
        std::vector<torch::Tensor> fixedCentres(losses.size());
        {
          torch::NoGradGuard noGrad;
          for (size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
          {
            std::vector<std::vector<torch::Tensor>> pieces(firstLayer[i + 1] - firstLayer[i]);
            Impact::ForEachBatch(
              m_FixedModelsConfiguration[i],
              device,
              n,
              [&](int64_t begin, int64_t end) {
                const std::vector<torch::Tensor> layers = Impact::PatchCentreFeatures(
                  m_FixedModelsConfiguration[i], patchesOf(i, begin, end, {}), fixedMetadata);
                for (size_t k = 0; k < pieces.size(); ++k)
                {
                  pieces[k].push_back(layers[k]);
                }
              },
              [] {});
            for (size_t k = 0; k < pieces.size(); ++k)
            {
              fixedCentres[firstLayer[i] + k] = torch::cat(pieces[k]);
            }
          }
        }
        if (gatheringPca)
        {
          // The sums of both sides' features of the points, for the iteration's basis and means.
          for (size_t l = 0; l < losses.size(); ++l)
          {
            if (componentsOf(l, fixedCentres[l].size(1)) > 0)
            {
              pointFixedSums[l].Add(fixedCentres[l].t());
            }
          }
          for (size_t i = 0; i < movingConfigs.size(); ++i)
          {
            Impact::ForEachBatch(
              movingConfigs[i],
              device,
              n,
              [&](int64_t begin, int64_t end) {
                const std::vector<torch::Tensor> layers =
                  Impact::PatchCentreFeatures(movingConfigs[i], patchesOf(i, begin, end, smoothed), movingMetadata);
                for (size_t k = 0; k < layers.size(); ++k)
                {
                  if (componentsOf(firstLayer[i] + k, layers[k].size(1)) > 0)
                  {
                    pointMovingSums[firstLayer[i] + k].Add(layers[k].t());
                  }
                }
              },
              [] {});
          }
          return value;
        }
        for (size_t l = 0; l < losses.size(); ++l)
        {
          fixedCentres[l] = pointsProjected(l, fixedCentres[l], pointFixedMean[l]);
        }
        if (raw == nullptr && !gathering)
        {
          leaf = smoothed.detach().requires_grad_(true);
          leaf.mutable_grad() = torch::zeros_like(leaf);
        }
        torch::Tensor gradientBefore;
        for (size_t i = 0; i < movingConfigs.size(); ++i)
        {
          Impact::ForEachBatch(
            movingConfigs[i],
            device,
            n,
            [&](int64_t begin, int64_t end) {
              const double weight = static_cast<double>(end - begin) / total;
              if (raw != nullptr || gathering)
              {
                torch::NoGradGuard               noGrad;
                const std::vector<torch::Tensor> layers =
                  Impact::PatchCentreFeatures(movingConfigs[i], patchesOf(i, begin, end, smoothed), movingMetadata);
                for (size_t k = 0; k < layers.size(); ++k)
                {
                  const size_t        l = firstLayer[i] + k;
                  const torch::Tensor centre = fixedCentres[l].narrow(0, begin, end - begin);
                  const torch::Tensor moving = pick(l, pointsProjected(l, layers[k], pointMovingMean[l]));
                  if (gathering && isGlobal(l))
                  {
                    globalTerm(l, Impact::PointSums(pick(l, centre), moving));
                  }
                  else if (!gathering && !isGlobal(l))
                  {
                    (*raw)[l] += losses[l]->forwardValue(pick(l, centre), moving).template item<double>() * weight;
                  }
                }
                return;
              }
              gradientBefore.reset(); // none to set back if the copy runs out of memory
              gradientBefore = leaf.grad().clone();
              const std::vector<torch::Tensor> layers =
                Impact::PatchCentreFeatures(movingConfigs[i], patchesOf(i, begin, end, leaf), movingMetadata);
              torch::Tensor surrogate = torch::zeros({}, onDevice);
              for (size_t k = 0; k < layers.size(); ++k)
              {
                const size_t        l = firstLayer[i] + k;
                const torch::Tensor centre = pick(l, fixedCentres[l].narrow(0, begin, end - begin));
                const torch::Tensor moving = pick(l, pointsProjected(l, layers[k], pointMovingMean[l]));
                surrogate =
                  surrogate + (isGlobal(l)
                                 ? globalTerm(l, Impact::PointSums(centre, moving))
                                 : layerWeights[l] * normalized(l, losses[l]->forwardValue(centre, moving)) * weight);
              }
              surrogate.backward();
              value += surrogate.template item<double>();
            },
            [&] {
              if (gradientBefore.defined())
              {
                leaf.mutable_grad().copy_(gradientBefore);
              }
            });
        }
      }
      if (raw == nullptr && !gathering)
      {
        const torch::Tensor loss = leaf.defined() ? reg + (smoothed * leaf.grad()).sum() : reg;
        if (loss.requires_grad()) // a constant for a last cell without a point
        {
          loss.backward();
        }
      }
      return value;
    };
    // How many of the cells (points) of `box` count, for the smoothed field `smoothed` of the cells from `origin` on.
    auto countOf =
      [&](const Impact::CellBox & box, const torch::Tensor & smoothed, const std::vector<int64_t> & origin) {
        if (sampledJacobian)
        {
          return keptPoints(box, smoothed, origin).size(0);
        }
        const Reading reading = readingOf(box, smoothed, origin);
        return m_SamplingPercentage < 1.0 && reading.count > 0 ? drawRows(reading.counted, box.cell0, box.cell1).numel()
                                                               : reading.count;
      };
    // The smoothed residual past a refresh's reference of the cells of `window` whose control cells are `control`:
    // the field the moving maps are read at; undefined without FeatureMapUpdateInterval.
    auto residualOf = [&](const Impact::CellBox & window, const torch::Tensor & control) {
      return refreshes ? smoothControl(control - Impact::NarrowTo(thetaRef, window.cell0, window.cell1).to(device))
                       : torch::Tensor();
    };
    // The loss of `part`, back-propagated to the control cells `control` of `window` it reads; its value. `raw`, when
    // given, takes each layer's value instead, without the graph.
    auto partValue = [&](const Impact::CellBox & part,
                         const Impact::CellBox & window,
                         const torch::Tensor &   control,
                         std::vector<double> *   raw) {
      const torch::Tensor smoothed = smoothControl(control);
      if (sampledJacobian)
      {
        return sampledPart(part, smoothed, window.cell0, raw);
      }
      const torch::Tensor loss = partLoss(part, smoothed, window.cell0, raw, residualOf(window, control));
      if (raw == nullptr && loss.requires_grad()) // a constant for a last cell out of the fixed mask
      {
        loss.backward();
      }
      return loss.template item<double>();
    };
    const bool masked = m_FixedMask.IsNotNull() || m_MovingMask.IsNotNull();
    int64_t    voxels = 1;
    for (const int64_t extent : spatial)
    {
      voxels *= extent;
    }
    // The schedule. RegionsPerIteration 0: every brick at every iteration, in the files' order. Otherwise that many
    // boxes, drawn afresh from Seed at every iteration, without one twice in an iteration: a draw moves its boxes
    // to the head of the list, so that it costs what they number.
    // ponytail: the files hold the cells in raster order, so a box read touches a page for each of its rows; files
    // laid out by boxes if boxes in a random order outgrow the page cache.
    std::mt19937 schedule(m_Seed);
    m_MetricValuesPerIteration.clear();
    m_MetricValuesPerIteration.reserve(m_NumberOfIterations);
    for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
    {
      drawIteration = iteration;
      const size_t drawn = drawsBoxes ? std::min<size_t>(m_RegionsPerIteration, order.size()) : 0;
      for (size_t i = 0; i < drawn; ++i)
      {
        std::swap(order[i], order[i + schedule() % (order.size() - i)]);
      }
      // What the whole-image run draws at the start of an iteration, in its order: the axis swept, the points, the
      // channels.
      if (swept)
      {
        sweptAxis = axisDistribution(axisGenerator);
      }
      if (sampledJacobian)
      {
        const int64_t count = std::max<int64_t>(1, std::llround(m_SamplingPercentage * static_cast<double>(voxels)));
        draws = torch::randint(voxels, { count }, generator, torch::kLong);
      }
      drawSubsets(channels);
      double  lossValue = 0.0;
      int64_t covered = 0;
      for (bool stepped = false; !stepped;)
      {
        try
        {
          if (sampledJacobian)
          {
            groupDraws();
          }
          const size_t perIteration = drawsBoxes ? drawn : units.size();
          const size_t set = countSet();
          if (recounts)
          {
            // The units of this iteration are counted again at the field they are about to be stepped from; the
            // others keep the count of their last step, the first iteration (of an axis swept) counting every unit.
            // With every unit at every iteration the divisor is the dense run's.
            torch::NoGradGuard noGrad;
            const bool         every = !countedOnce[set];
            for (size_t i = 0; i < (every ? units.size() : perIteration); ++i)
            {
              const size_t          u = every ? i : order[i];
              const Impact::CellBox window = around(units[u]);
              const int64_t         count = countOf(
                units[u], smoothControl(Impact::NarrowTo(theta, window.cell0, window.cell1).to(device)), window.cell0);
              countedCells[set] += static_cast<double>(count - survivors[set][u]);
              survivors[set][u] = count;
            }
            countedOnce[set] = true;
          }
          if (sampledJacobian && countedCells[0] == 0.0 && !masked)
          {
            itkExceptionMacro("ImpactFineRegistration: no drawn point has its model patch inside the image; the patch "
                              "(PatchSize x voxel size) is larger than the image, or SamplingPercentage too small.");
          }
          if (sampledJacobian && std::any_of(m_PCA.begin(), m_PCA.end(), [](unsigned int k) { return k > 0; }))
          {
            // The iteration's basis and means, from both sides' features of every unit's points.
            {
              torch::NoGradGuard noGrad;
              gatheringPca = true;
              std::fill(pointFixedSums.begin(), pointFixedSums.end(), Impact::PcaSums{});
              std::fill(pointMovingSums.begin(), pointMovingSums.end(), Impact::PcaSums{});
              std::fill(pointBasis.begin(), pointBasis.end(), torch::Tensor());
              for (const Impact::CellBox & unit : units)
              {
                const Impact::CellBox window = around(unit);
                partValue(unit, window, Impact::NarrowTo(theta, window.cell0, window.cell1).to(device), nullptr);
              }
              gatheringPca = false;
            }
            for (size_t l = 0; l < losses.size(); ++l)
            {
              if (pointFixedSums[l].count > 0.0)
              {
                pointBasis[l] = pointFixedSums[l].Basis(m_PCA[l]);
                pointFixedMean[l] = pointFixedSums[l].Mean();
                pointMovingMean[l] = pointMovingSums[l].Mean();
              }
            }
          }
          if (jacobian && fitsPca)
          {
            // The moving features' mean at the field of this iteration (a swept model's over the slices PcaSlices reads
            // along the axis swept), without the graph, as the whole-image run detaches it.
            torch::NoGradGuard noGrad;
            movingMeanNow = meansOver(
              true,
              swept ? sweptBasis[sweptAxis] : pcaBasis,
              [&](const Impact::CellBox & box, const RegionType & span) {
                const Impact::CellBox window = around(box);
                return warpedOn(readMovingImage,
                                span,
                                smoothControl(Impact::NarrowTo(theta, window.cell0, window.cell1).to(device)),
                                window.cell0);
              },
              swept ? static_cast<int>(sweptAxis) : -1);
          }
          if (globalLoss)
          {
            // A global distance's sums over every unit at the field of this iteration, whatever the schedule draws:
            // its value, and its gradient with respect to them.
            {
              torch::NoGradGuard noGrad;
              gathering = true;
              gatheredPoints = 0.0;
              std::fill(gatheredSums.begin(), gatheredSums.end(), torch::Tensor());
              for (const Impact::CellBox & unit : units)
              {
                const Impact::CellBox window = around(unit);
                partValue(unit, window, Impact::NarrowTo(theta, window.cell0, window.cell1).to(device), nullptr);
              }
              gathering = false;
            }
            for (size_t l = 0; l < losses.size(); ++l)
            {
              globalValues[l] = torch::zeros({}, onDevice); // no point counts: no term
              if (isGlobal(l) && gatheredSums[l].defined())
              {
                const torch::Tensor sums = gatheredSums[l].detach().requires_grad_(true);
                const torch::Tensor value = losses[l]->ValueFromSums(sums, gatheredPoints);
                sumSeeds[l] = torch::autograd::grad({ value }, { sums })[0];
                globalValues[l] = value.detach();
              }
            }
          }
          if (normalizes && !normalization.IsLatched(0))
          {
            // Each layer's value at the starting field, without the graph, summed over every unit: the dense run's
            // mean, whatever the schedule draws. The one pass of the iterations over the grid, a unit at a time.
            torch::NoGradGuard  noGrad;
            std::vector<double> raw(losses.size(), 0.0);
            for (const Impact::CellBox & unit : units)
            {
              const Impact::CellBox window = around(unit);
              partValue(unit, window, Impact::NarrowTo(theta, window.cell0, window.cell1).to(device), &raw);
            }
            for (size_t l = 0; l < losses.size(); ++l)
            {
              normalization.Latch(l, isGlobal(l) ? globalValues[l].template item<double>() : raw[l]);
            }
          }
          // The gradient of the units' cells. The bricks, which cover the grid, each back-propagate their own loss
          // through the smoothing passes to the cells it reads, and those gradients add up on the cells bricks share.
          // A drawn box has no such neighbours: it takes the loss of the cells around it, which holds every term of
          // the dense loss that reads its cells, and keeps the gradient of its own cells, the whole grid's.
          for (size_t i = 0; i < perIteration; ++i)
          {
            const Impact::CellBox & unit = units[order[i]];
            const Impact::CellBox   part = drawsBoxes ? around(unit) : unit;
            const Impact::CellBox   window = around(part);
            const Impact::CellBox & kept = drawsBoxes ? unit : window;
            const torch::Tensor     control = Impact::NarrowTo(theta, window.cell0, window.cell1)
                                            .to(device, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true)
                                            .requires_grad_(true);
            lossValue += partValue(part, window, control, nullptr);
            covered += part.Count();
            if (control.grad().defined())
            {
              Impact::NarrowTo(gradient, kept.cell0, kept.cell1)
                .add_(Impact::CropTo(control.grad(), window, kept).to(gradient.device()));
            }
          }
          stepped = true;
        }
        catch (const std::exception & error)
        {
          // A brick that runs out of device memory: the iteration again by bricks of half the side, down to a box.
          if (!Impact::IsDeviceOutOfMemory(error) || drawsBoxes || boxesASide == 1)
          {
            throw;
          }
          gradient.zero_();
          lossValue = 0.0;
          covered = 0;
          Impact::ReleaseCachedDeviceMemory(device);
          itkWarningMacro("ImpactFineRegistration: a brick of " << boxesASide << " boxes a side ran out of device "
                                                                << "memory; retrying with " << boxesASide / 2 << ".");
          boxesASide /= 2;
          tile(boxesASide, iteration);
        }
      }
      for (size_t i = 0; i < (drawsBoxes ? drawn : units.size()); ++i)
      {
        const Impact::CellBox & unit = units[order[i]];
        adam.steps = ++stepsOf[order[i]];
        adam.StepCells(theta, gradient, unit.cell0, unit.cell1, device);
        Impact::NarrowTo(gradient, unit.cell0, unit.cell1).zero_();
      }
      // The loss over the parts, scaled from the cells they hold to the grid: the dense loss when they tile it. A
      // global distance's value is the whole grid's.
      const double scale = static_cast<double>(covered) / static_cast<double>(cellCountTotal);
      for (size_t l = 0; l < losses.size(); ++l)
      {
        if (isGlobal(l))
        {
          lossValue += (layerWeights[l] * normalized(l, globalValues[l])).template item<double>() * scale;
        }
      }
      m_MetricValuesPerIteration.push_back(lossValue / scale);
      refresh(iteration);
    }
  }
  // ---- The control grid, smoothed, as the field on its own grid, written a box at a time from the cells its
  // smoothing reads: the host holds that image alone. The dense output is interpolated from it. ----
  {
    torch::NoGradGuard noGrad;
    m_ControlGrid = Impact::AllocateCellGrid<Dim>(
      cells,
      m_FixedImage.GetPointer(),
      m_ResidentFeatures ? nullptr : Impact::MapCellGridPixels<Dim>(cells, scratch).GetPointer());
    const std::vector<int64_t> smoothingReach(Dim, static_cast<int64_t>(m_ControlGridSmoothingIterations));
    for (const Impact::CellBox & box : boxes)
    {
      const Impact::CellBox window = Impact::CellsAround(box, smoothingReach, cells);
      Impact::WriteCellsToCellGrid<Dim>(
        Impact::CropTo(smoothControl(Impact::NarrowTo(theta, window.cell0, window.cell1).to(device)), window, box),
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
  os << indent << "RegionsPerIteration: " << m_RegionsPerIteration << std::endl;
  os << indent << "BrickBytes: " << m_BrickBytes << std::endl;
  os << indent << "RegionHalo: " << m_RegionHalo << std::endl;
  os << indent << "ResidentFeatureBudget: " << m_ResidentFeatureBudget << std::endl;
  os << indent << "ResidentBytes: " << m_ResidentBytes << std::endl;
  os << indent << "ScratchDirectory: " << m_ScratchDirectory << std::endl;
}

} // end namespace itk

#endif // itkImpactFineRegistration_hxx

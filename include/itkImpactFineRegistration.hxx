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
#include "ImpactLoss.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkIdentityTransform.h>

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace itk
{

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

  // Spatial sizes in torch (z, y, x) order.
  const auto &         fixedSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
  std::vector<int64_t> spatial(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    spatial[d] = static_cast<int64_t>(fixedSize[ImageDimension - 1 - d]);
  }

  // ---- 2. Optimizable control grid {1, N, coarse z,y,x}; component order (z,y,x), full-res
  // voxel units. With GridShrinkFactor>1 it lives at image-size / GridShrinkFactor and is
  // upsampled to full resolution each iteration (ConvexAdam-style); the values stay in full-res
  // voxel units. Warm-started from an initial field (e.g. the coarse stage) if provided, else zero. ----
  const int64_t        shrink = static_cast<int64_t>(std::max(1u, m_GridShrinkFactor));
  std::vector<int64_t> coarseSpatial(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    coarseSpatial[d] = std::max<int64_t>(1, spatial[d] / shrink);
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
      m_InitialDisplacementField, m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), device); // {1,N,z,y,x}
    if (shrink > 1)
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

  // Per-component normalization (size-1)/2 in (z, y, x) order (exact for align_corners=true).
  std::vector<float> scaleValues(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    scaleValues[d] = (spatial[d] > 1) ? static_cast<float>((spatial[d] - 1) / 2.0) : 1.0f;
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

  // Feature-mode setup: extract the fixed/moving feature layers (constants, not differentiated
  // through), optionally PCA-reduce them (fit on fixed), and build one loss per kept layer.
  // Intensity mode skips this and compares raw voxels.
  std::vector<torch::Tensor>                 fixedLayers;
  std::vector<torch::Tensor>                 movingLayers;
  std::vector<torch::Tensor>                 pcaBasis; // per kept layer; undefined entry = no PCA
  std::vector<std::unique_ptr<Impact::Loss>> losses;
  std::vector<float>                         layerWeights;
  const std::vector<ModelConfiguration> &    movingConfigs =
    m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
  // Extract the moving feature layers from an image tensor and project them onto the stored PCA
  // bases (used for the initial extraction and for the FeatureMapUpdateInterval re-extraction).
  auto extractMovingFeatures = [&](const torch::Tensor & imageTensor) -> std::vector<torch::Tensor> {
    auto layers = Impact::ExtractFeatureLayers<ImageDimension>(movingConfigs, imageTensor, device, m_SubsetFeatures);
    for (size_t l = 0; l < layers.size() && l < pcaBasis.size(); ++l)
    {
      if (pcaBasis[l].defined())
      {
        layers[l] = Impact::PcaTransform(layers[l].squeeze(0), pcaBasis[l]).unsqueeze(0).contiguous();
      }
    }
    return layers;
  };
  if (featureMode)
  {
    fixedLayers = Impact::ExtractFeatureLayers<ImageDimension>(m_FixedModelsConfiguration, fixedT, device, m_SubsetFeatures);
    movingLayers = Impact::ExtractFeatureLayers<ImageDimension>(movingConfigs, movingT, device, m_SubsetFeatures);
    if (fixedLayers.size() != movingLayers.size() || fixedLayers.empty())
    {
      itkExceptionMacro("ImpactFineRegistration: fixed and moving produced "
                        << fixedLayers.size() << " and " << movingLayers.size()
                        << " feature layers; they must match and be non-empty.");
    }
    // Optional per-layer PCA: fit the basis on the fixed features and project BOTH onto it so
    // they live in a consistent reduced space (PCA[l] components; 0 or >= channels = no-op).
    pcaBasis.resize(fixedLayers.size());
    for (size_t l = 0; l < fixedLayers.size(); ++l)
    {
      const int64_t components = (l < m_PCA.size()) ? static_cast<int64_t>(m_PCA[l]) : 0;
      if (components <= 0 || components >= fixedLayers[l].size(1))
      {
        continue;
      }
      pcaBasis[l] = Impact::PcaFit(fixedLayers[l].squeeze(0), components);
      fixedLayers[l] = Impact::PcaTransform(fixedLayers[l].squeeze(0), pcaBasis[l]).unsqueeze(0).contiguous();
      movingLayers[l] = Impact::PcaTransform(movingLayers[l].squeeze(0), pcaBasis[l]).unsqueeze(0).contiguous();
    }
    for (size_t l = 0; l < fixedLayers.size(); ++l)
    {
      const std::string name =
        m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)];
      losses.push_back(Impact::LossFactory::Instance().Create(name));
      layerWeights.push_back(l < m_LayersWeight.size() ? m_LayersWeight[l] : 1.0f);
    }
    // Downsample the fixed/moving feature layers to the Adam similarity resolution (never upsampling),
    // then precompute the identity base grid at each layer's resulting resolution so the loss grid is
    // built directly there (reuses grid0 for full-resolution layers).
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
    }
  }

  // Reference control field at which `movingLayers` were extracted (0 = un-warped). With
  // FeatureMapUpdateInterval > 0 the moving features are periodically re-extracted from the
  // moving image warped by the current total field, and the loop warps them by the residual
  // (theta - thetaRef); thetaRef stays 0 (residual == theta) when disabled.
  torch::Tensor thetaRef = torch::zeros_like(theta);

  // ConvexAdam-style intensity refinement: evaluate the similarity at the control-grid resolution
  // (coarseSpatial), so GridShrinkFactor plays the role of the reference's grid_sp_adam. Moving/fixed
  // are pooled once to that grid and the field is warped and compared there, instead of upsampling
  // the field and warping at full resolution every iteration. With GridShrinkFactor==1
  // coarseSpatial==spatial and this is bit-identical to the full-res path.
  torch::Tensor movingCoarse, fixedCoarse, gridBaseCoarse;
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

  torch::Tensor warped;
  for (unsigned int iteration = 0; iteration < m_NumberOfIterations; ++iteration)
  {
    optimizer.zero_grad();

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
      const double controlSpacing = static_cast<double>(spatial[ax]) / static_cast<double>(len);
      reg = reg + (regField.narrow(tdim, 1, len - 1) - regField.narrow(tdim, 0, len - 1)).pow(2).mean() /
                    (controlSpacing * controlSpacing);
    }
    reg = reg * m_RegularizationWeight;

    // Warp by the residual control field (theta - thetaRef): the moving features were extracted
    // at thetaRef, so this is identity right after a refresh (and == theta when disabled).
    torch::Tensor similarity;
    if (featureMode)
    {
      // Build each layer's sampling grid directly at its resolution from the smoothed residual
      // (no full-res detour); smoothing is shared across layers.
      const torch::Tensor smoothedResidual = smoothControl(theta - thetaRef);
      similarity = torch::zeros({}, theta.options());
      for (size_t l = 0; l < movingLayers.size(); ++l)
      {
        torch::Tensor warpedLayer =
          F::grid_sample(movingLayers[l], gridForLayerControl(smoothedResidual, l), sampleOpts);
        const int64_t channels = movingLayers[l].size(1);
        torch::Tensor fixedFlat = fixedLayers[l].permute(toChannelLast).reshape({ -1, channels });
        torch::Tensor warpedFlat = warpedLayer.permute(toChannelLast).reshape({ -1, channels });
        similarity = similarity + layerWeights[l] * losses[l]->forwardValue(fixedFlat, warpedFlat);
      }
    }
    else
    {
      // Build the grid directly at the control-grid resolution and warp the pooled moving image.
      torch::Tensor dd = smoothControl(theta - thetaRef).permute(toChannelLast);
      torch::Tensor grid = gridBaseCoarse + (dd / scale).flip(-1);
      torch::Tensor warpedImage = F::grid_sample(movingCoarse, grid, sampleOpts);
      similarity = (warpedImage - fixedCoarse).pow(2).mean();
    }
    torch::Tensor loss = similarity + reg;

    loss.backward();
    optimizer.step();

    m_MetricValuesPerIteration.push_back(loss.template item<double>());

    // FeatureMapUpdateInterval: periodically re-extract the moving feature maps from the moving
    // image warped by the current total field, and reset the residual baseline. theta (and its
    // Adam moments) is untouched; only the reference frame of the moving features changes.
    if (featureMode && m_FeatureMapUpdateInterval > 0 &&
        (iteration + 1) % static_cast<unsigned int>(m_FeatureMapUpdateInterval) == 0 &&
        iteration + 1 < m_NumberOfIterations)
    {
      torch::NoGradGuard noGrad;
      thetaRef = theta.detach().clone();
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

  Impact::WriteVoxelFieldToDisplacement<ImageDimension>(
    controlGridToFullField(theta.detach()), m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), output);

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
  os << indent << "FixedModelsConfiguration count: " << m_FixedModelsConfiguration.size() << std::endl;
}

} // end namespace itk

#endif // itkImpactFineRegistration_hxx

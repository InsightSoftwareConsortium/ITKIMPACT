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
#ifndef itkImpactCoarseRegistration_hxx
#define itkImpactCoarseRegistration_hxx

// LibTorch-backed implementation; included only when ITK_MANUAL_INSTANTIATION is undefined.

#include "itkImpactCoarseRegistration.h"
#include "itkImpactTorchRegistrationHelpers.h"

#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkIdentityTransform.h>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace itk
{
namespace Impact
{

/** The mean over the coarse stage's cell window: two 3^Dim box passes, zero-padded, as ConvexAdam smooths its cost. */
template <unsigned int Dim>
torch::Tensor
CoarseBoxWindow(const torch::Tensor & t)
{
  namespace F = torch::nn::functional;
  if constexpr (Dim == 3)
  {
    const auto box = F::AvgPool3dFuncOptions(3).stride(1).padding(1);
    return F::avg_pool3d(F::avg_pool3d(t, box), box);
  }
  else
  {
    const auto box = F::AvgPool2dFuncOptions(3).stride(1).padding(1);
    return F::avg_pool2d(F::avg_pool2d(t, box), box);
  }
}

/** One feature layer's coarse cost for one candidate displacement, {1, 1, coarse...}: `fixed` and `shifted` ({1, C,
 * coarse...}) are the two images' coarse features, the moving one shifted by the candidate. The cost at a coarse voxel
 * is `distance` over the cell window around it (CoarseBoxWindow): a point-wise distance's terms (Loss::forwardPoints)
 * averaged over the window -- for L2 exactly ConvexAdam's smoothed SSD --, NCC and LNCC each channel's correlation over
 * the window, from its weighted means of f, m, f^2, m^2 and f*m (the weight falling outside the image dropped). A
 * point-wise term that is not a number (the cosine of a zero vector) counts as no similarity, 1.
 *
 * `weight` ({1, 1, coarse...}), when given, weighs each cell in the window (the masks' shares, CoarseMaskWeight): every
 * mean is then a ratio of two window sums, and a coarse voxel whose window holds no weight costs 0 (see
 * CoarseMaskSupport). */
template <unsigned int Dim>
torch::Tensor
CoarseLayerCost(const Loss &          distance,
                const torch::Tensor & fixed,
                const torch::Tensor & shifted,
                const torch::Tensor & weight = {})
{
  if (!weight.defined())
  {
    if (distance.IsPerPointMean())
    {
      return CoarseBoxWindow<Dim>(torch::nan_to_num(distance.forwardPoints(fixed, shifted), 1.0).unsqueeze(1));
    }
    const torch::Tensor weights = CoarseBoxWindow<Dim>(torch::ones_like(fixed.narrow(1, 0, 1)));
    auto                mean = [&weights](const torch::Tensor & t) { return CoarseBoxWindow<Dim>(t) / weights; };
    const torch::Tensor meanFixed = mean(fixed);
    const torch::Tensor meanMoving = mean(shifted);
    const torch::Tensor meanFixedSquare = mean(fixed * fixed);
    const torch::Tensor meanMovingSquare = mean(shifted * shifted);
    const torch::Tensor meanProduct = mean(fixed * shifted);
    const torch::Tensor correlation =
      distance.IsSpatial()
        ? LNCC::SquaredCorrelation(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct)
        : NCC::Correlation(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct);
    return 1.0 - correlation.mean(1, /*keepdim=*/true);
  }
  const torch::Tensor total = CoarseBoxWindow<Dim>(weight);
  const torch::Tensor held = total > 0.0;
  const torch::Tensor divisor = torch::where(held, total, 1.0);
  if (distance.IsPerPointMean())
  {
    const torch::Tensor terms = torch::nan_to_num(distance.forwardPoints(fixed, shifted), 1.0).unsqueeze(1);
    return torch::where(held, CoarseBoxWindow<Dim>(terms * weight) / divisor, 0.0);
  }
  auto                mean = [&](const torch::Tensor & t) { return CoarseBoxWindow<Dim>(t * weight) / divisor; };
  const torch::Tensor meanFixed = mean(fixed);
  const torch::Tensor meanMoving = mean(shifted);
  const torch::Tensor meanFixedSquare = mean(fixed * fixed);
  const torch::Tensor meanMovingSquare = mean(shifted * shifted);
  const torch::Tensor meanProduct = mean(fixed * shifted);
  const torch::Tensor correlation =
    distance.IsSpatial()
      ? LNCC::SquaredCorrelation(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct)
      : NCC::Correlation(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct);
  return torch::where(held, 1.0 - correlation.mean(1, /*keepdim=*/true), 0.0);
}

/** The shift of `volume` ({1, C, coarse...}) by candidate `l` of the box of `halfWidths` (see AccumulateCoarseCost),
 * read in `padded`, the volume padded by the half-widths. */
template <unsigned int Dim>
torch::Tensor
CoarseCandidateShift(const torch::Tensor &        padded,
                     const std::vector<int64_t> & halfWidths,
                     int64_t                      l,
                     c10::IntArrayRef             coarse)
{
  torch::Tensor shifted = padded;
  int64_t       rem = l;
  for (int a = static_cast<int>(Dim) - 1; a >= 0; --a)
  {
    const int64_t side = 2 * halfWidths[a] + 1;
    const int64_t offset = (rem % side) - halfWidths[a];
    rem /= side;
    shifted = shifted.narrow(2 + a, halfWidths[a] + offset, coarse[2 + a]);
  }
  return shifted;
}

/** `volume` padded by `halfWidths` on each side of each axis: edge-replicated (features), or with zeros (a mask's
 * share: outside the image no voxel is in the mask). */
template <unsigned int Dim>
torch::Tensor
CoarsePad(const torch::Tensor & volume, const std::vector<int64_t> & halfWidths, bool replicate)
{
  namespace F = torch::nn::functional;
  std::vector<int64_t> padding; // F::pad takes its pairs from the last axis backwards
  for (int a = static_cast<int>(Dim) - 1; a >= 0; --a)
  {
    padding.push_back(halfWidths[a]);
    padding.push_back(halfWidths[a]);
  }
  return replicate ? F::pad(volume, F::PadFuncOptions(padding).mode(torch::kReplicate))
                   : F::pad(volume, F::PadFuncOptions(padding).mode(torch::kConstant).value(0.0));
}

/** The weight of each coarse cell for one candidate: the fixed cell's share of the fixed mask times the share of the
 * moving mask in the moving cell the candidate shifts it to (`shiftedMoving`); either alone when the other mask is
 * absent. */
inline torch::Tensor
CoarseMaskWeight(const torch::Tensor & fixedShare, const torch::Tensor & shiftedMoving)
{
  if (!fixedShare.defined())
  {
    return shiftedMoving;
  }
  return shiftedMoving.defined() ? fixedShare * shiftedMoving : fixedShare;
}

/** Add `weight` times one feature layer's coarse cost volume to `cost` ({L, coarse...}): for each of the L =
 * prod_a (2 halfWidths[a] + 1) candidate displacements -- `halfWidths` per tensor axis, in coarse voxels, the last
 * tensor axis (ITK x) running fastest, the order of the solver's displacement table --, the CoarseLayerCost of `moving`
 * shifted by it (edge-replicated) against `fixed`, both {1, C, coarse...}. With masks, `fixedShare` and `movingShare`
 * ({1, 1, coarse...}, either undefined when its mask is absent) weigh the cells, the moving one shifted with the
 * features (zero outside the image). */
template <unsigned int Dim>
void
AccumulateCoarseCost(torch::Tensor &              cost,
                     const Loss &                 distance,
                     const torch::Tensor &        fixed,
                     const torch::Tensor &        moving,
                     const std::vector<int64_t> & halfWidths,
                     double                       weight,
                     const torch::Tensor &        fixedShare = {},
                     const torch::Tensor &        movingShare = {})
{
  const torch::Tensor padded = CoarsePad<Dim>(moving, halfWidths, true);
  const torch::Tensor paddedShare =
    movingShare.defined() ? CoarsePad<Dim>(movingShare, halfWidths, false) : movingShare;
  const bool masked = fixedShare.defined() || movingShare.defined();
  for (int64_t l = 0; l < cost.size(0); ++l)
  {
    const torch::Tensor shifted = CoarseCandidateShift<Dim>(padded, halfWidths, l, fixed.sizes());
    if (!masked)
    {
      cost[l] += weight * CoarseLayerCost<Dim>(distance, fixed, shifted).squeeze(0).squeeze(0);
      continue;
    }
    const torch::Tensor cellWeight = CoarseMaskWeight(
      fixedShare,
      paddedShare.defined() ? CoarseCandidateShift<Dim>(paddedShare, halfWidths, l, fixed.sizes()) : paddedShare);
    cost[l] += weight * CoarseLayerCost<Dim>(distance, fixed, shifted, cellWeight).squeeze(0).squeeze(0);
  }
}

/** Where each candidate leaves the cell window around each coarse voxel some mask weight, {L, coarse...} bool: the
 * coarse voxels and candidates a masked cost has data for (see AccumulateCoarseCost). */
template <unsigned int Dim>
torch::Tensor
CoarseMaskSupport(const torch::Tensor &        fixedShare,
                  const torch::Tensor &        movingShare,
                  const std::vector<int64_t> & halfWidths,
                  c10::IntArrayRef             coarse,
                  int64_t                      candidates)
{
  const torch::Tensor paddedShare =
    movingShare.defined() ? CoarsePad<Dim>(movingShare, halfWidths, false) : movingShare;
  std::vector<int64_t> shape{ candidates };
  shape.insert(shape.end(), coarse.begin() + 2, coarse.end());
  torch::Tensor support = torch::empty(
    shape,
    torch::TensorOptions().dtype(torch::kBool).device((fixedShare.defined() ? fixedShare : movingShare).device()));
  for (int64_t l = 0; l < candidates; ++l)
  {
    const torch::Tensor cellWeight = CoarseMaskWeight(
      fixedShare, paddedShare.defined() ? CoarseCandidateShift<Dim>(paddedShare, halfWidths, l, coarse) : paddedShare);
    support[l] = (CoarseBoxWindow<Dim>(cellWeight) > 0.0).squeeze(0).squeeze(0);
  }
  return support;
}

/** A masked cost volume ({L, coarse...}) where `support` says it has no data: a candidate without data costs as much as
 * the worst candidate with data at that coarse voxel, and a coarse voxel with no data at all costs 0 whatever the
 * candidate, so that the coupling alone places it. */
inline torch::Tensor
FillCoarseCostWithoutData(const torch::Tensor & cost, const torch::Tensor & support)
{
  const torch::Tensor worst =
    torch::where(support, cost, -std::numeric_limits<float>::infinity()).amax(0, /*keepdim=*/true);
  return torch::where(support, cost, torch::where(support.any(0, /*keepdim=*/true), worst, 0.0));
}

} // namespace Impact

template <typename TFixedImage, typename TMovingImage>
ImpactCoarseRegistration<TFixedImage, TMovingImage>::ImpactCoarseRegistration() = default;

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::SetFixedImage(const FixedImageType * image)
{
  if (m_FixedImage.GetPointer() != image)
  {
    m_FixedImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::SetMovingImage(const MovingImageType * image)
{
  if (m_MovingImage.GetPointer() != image)
  {
    m_MovingImage = image;
    this->Modified();
  }
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GetDisplacementField() -> DisplacementFieldType *
{
  return this->GetOutput();
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GetDisplacementFieldTransform() -> DisplacementFieldTransformType *
{
  return m_DisplacementFieldTransform.GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GenerateOutputInformation()
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
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GenerateData()
{
  namespace F = torch::nn::functional;

  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }

  if constexpr (ImageDimension != 2 && ImageDimension != 3)
  {
    itkExceptionMacro("ImpactCoarseRegistration supports 2D and 3D images only.");
  }
  else
  {
    const bool featureMode = !m_FixedModelsConfiguration.empty();
    for (size_t l = 0; l < m_LayersWeight.size(); ++l)
    {
      if (!(m_LayersWeight[l] >= 0.0f)) // also rejects NaN
      {
        itkExceptionMacro("ImpactCoarseRegistration: LayersWeight[" << l << "] is " << m_LayersWeight[l]
                                                                    << "; layer weights must be non-negative.");
      }
    }

    const torch::Device device(m_Device);
    torch::manual_seed(m_Seed);
    torch::NoGradGuard noGrad; // the coarse stage is non-differentiable

    // ---- Fixed and moving onto the same (fixed) voxel grid; resample only if geometries differ. ----
    typename MovingImageType::ConstPointer movingOnFixed = m_MovingImage;
    {
      const auto & fSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
      const auto & mSize = m_MovingImage->GetLargestPossibleRegion().GetSize();
      const bool   sameGrid = (fSize == mSize) && (m_FixedImage->GetSpacing() == m_MovingImage->GetSpacing()) &&
                            (m_FixedImage->GetOrigin() == m_MovingImage->GetOrigin()) &&
                            (m_FixedImage->GetDirection() == m_MovingImage->GetDirection());
      if (!sameGrid)
      {
        using ResampleType = ResampleImageFilter<MovingImageType, MovingImageType, double>;
        auto resample = ResampleType::New();
        resample->SetInput(m_MovingImage);
        resample->SetTransform(IdentityTransform<double, ImageDimension>::New());
        resample->SetInterpolator(LinearInterpolateImageFunction<MovingImageType, double>::New());
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

    // Full-resolution spatial sizes in torch (z, y, x) order.
    const auto &         fixedSize = m_FixedImage->GetLargestPossibleRegion().GetSize();
    std::vector<int64_t> spatial(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      spatial[d] = static_cast<int64_t>(fixedSize[ImageDimension - 1 - d]);
    }

    // ---- Cost source: raw intensities, or IMPACT feature channels (per native-resolution layer). ----
    std::vector<torch::Tensor> fixedLayers; // empty in intensity mode
    std::vector<torch::Tensor> movingLayers;
    if (featureMode)
    {
      const std::vector<ImpactModelConfiguration> & movingConfigs =
        m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
      // Give metadata-aware models (nArgs>=4, e.g. SAM) each image's OWN normalisation stats
      // (min/max/mean/sigma + direction), not the undefined default. Fixed and moving configs added via
      // AddModelConfiguration share one impl, so set each image's stats immediately BEFORE extracting
      // that image: the extraction reads them there (serialized on the stream), and reassigning the
      // member for the next image does not touch the tensor the completed forward already captured.
      for (const auto & cfg : m_FixedModelsConfiguration)
        SetupImageMetadata<FixedImageType>(cfg, m_FixedImage);
      // Both tensors are on the fixed grid: each model sees them resampled from its spacing to its voxel size.
      std::vector<double> fixedSpacing(ImageDimension);
      for (unsigned int d = 0; d < ImageDimension; ++d)
      {
        fixedSpacing[d] = m_FixedImage->GetSpacing()[d];
      }
      // A 2D model sweeps the axis closest to head-feet; both images are on the fixed grid.
      const unsigned int sweepAxis = Impact::HeadFeetAxis(m_FixedImage->GetDirection(), ImageDimension);
      fixedLayers = Impact::ExtractFeatureLayers<ImageDimension>(
        m_FixedModelsConfiguration, fixedT, device, {}, false, fixedSpacing, static_cast<int>(sweepAxis));
      for (const auto & cfg : movingConfigs)
        SetupImageMetadata<MovingImageType>(cfg, movingOnFixed);
      movingLayers = Impact::ExtractFeatureLayers<ImageDimension>(
        movingConfigs, movingT, device, {}, false, fixedSpacing, static_cast<int>(sweepAxis));
      if (fixedLayers.empty() || fixedLayers.size() != movingLayers.size())
      {
        itkExceptionMacro("ImpactCoarseRegistration: fixed/moving produced "
                          << fixedLayers.size() << " and " << movingLayers.size() << " feature layers.");
      }
      const std::vector<int64_t> pcaSweep =
        Impact::PcaSweepDimensions<ImageDimension>(m_FixedModelsConfiguration, sweepAxis);
      for (size_t l = 0; l < fixedLayers.size(); ++l)
      {
        // PCA as the fine stage does it: the basis fitted on the fixed features, both projected onto it, so the
        // two stages compare the same channels.
        const int64_t components = l < m_PCA.size() ? static_cast<int64_t>(m_PCA[l]) : 0;
        if (components > 0 && components < fixedLayers[l].size(1))
        {
          torch::Tensor basis;
          const int64_t sweep = l < pcaSweep.size() ? pcaSweep[l] : 0;
          fixedLayers[l] =
            Impact::PcaReduce(fixedLayers[l].squeeze(0), basis, components, sweep).unsqueeze(0).contiguous();
          movingLayers[l] =
            Impact::PcaReduce(movingLayers[l].squeeze(0), basis, components, sweep).unsqueeze(0).contiguous();
        }
        // SubsetFeatures: that many of the layer's channels, drawn at random (from the seeded generator) once,
        // since the coarse search runs once.
        const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
        if (kept > 0 && kept < fixedLayers[l].size(1))
        {
          const torch::Tensor channels =
            torch::randperm(fixedLayers[l].size(1), torch::TensorOptions().dtype(torch::kLong))
              .narrow(0, 0, kept)
              .to(device);
          fixedLayers[l] = fixedLayers[l].index_select(1, channels).contiguous();
          movingLayers[l] = movingLayers[l].index_select(1, channels).contiguous();
        }
      }
    }

    // ---- The search, in units of the fixed image's finest voxel side s_min, derived per axis so that it is (nearly)
    // isotropic in millimetres: a cell of gs_a = max(1, round(GridSpacing * s_min / s_a)) voxels along axis a, and
    // along it hw_a = ceil(R / (gs_a * s_a)) cells each way, R = DisplacementHalfWidth * GridSpacing * s_min mm being
    // the capture range. An isotropic image keeps GridSpacing and DisplacementHalfWidth on every axis. ----
    const int64_t              gs = static_cast<int64_t>(m_GridSpacing);
    const int64_t              hw = static_cast<int64_t>(m_DisplacementHalfWidth);
    const std::vector<double>  voxelSide = Impact::TensorVoxelSides<ImageDimension>(m_FixedImage->GetSpacing());
    const double               finest = *std::min_element(voxelSide.begin(), voxelSide.end());
    const std::vector<int64_t> cellVoxels = Impact::IsotropicVoxelCounts(voxelSide, static_cast<double>(gs)); // gs_a
    const std::vector<int64_t> halfWidths = Impact::CaptureHalfWidths(voxelSide, cellVoxels, gs, hw);         // hw_a
    std::vector<double> cellWeight(ImageDimension); // (cell_a / cell_ref)^2, what a cell of axis a costs to cross
    int64_t             L = 1;
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      const double ratio = (static_cast<double>(cellVoxels[a]) * voxelSide[a]) / (static_cast<double>(gs) * finest);
      cellWeight[a] = ratio * ratio;
      L *= 2 * halfWidths[a] + 1;
    }

    // Dimension-generic avg-pool / replicate-pad / interpolate helpers (2D or 3D).
    auto poolCells = [&](const torch::Tensor & t) {
      if constexpr (ImageDimension == 3)
      {
        const std::vector<int64_t> k = cellVoxels;
        return F::avg_pool3d(t, F::AvgPool3dFuncOptions({ k[0], k[1], k[2] }).stride({ k[0], k[1], k[2] }));
      }
      else
      {
        const std::vector<int64_t> k = cellVoxels;
        return F::avg_pool2d(t, F::AvgPool2dFuncOptions({ k[0], k[1] }).stride({ k[0], k[1] }));
      }
    };
    auto smoothBox = [&](const torch::Tensor & t) {
      if constexpr (ImageDimension == 3)
        return F::avg_pool3d(t, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
      else
        return F::avg_pool2d(t, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    };

    // ---- Coarse grid: average-pool each cost source to the common coarse grid. ----
    // A native-resolution source (intensities, or a full-res feature layer) uses the exact
    // stride-gs_a pool of the cells. An already-downsampled backbone layer is adaptive-pooled
    // to the same coarse grid, so its features are never upsampled to full res.
    std::vector<int64_t> coarseSpatial(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      coarseSpatial[d] = spatial[d] / cellVoxels[d];
    }
    auto toCoarseGrid = [&](const torch::Tensor & t) -> torch::Tensor {
      const std::vector<int64_t> ts(t.sizes().begin() + 2, t.sizes().end());
      if (ts == spatial)
      {
        return poolCells(t);
      }
      if constexpr (ImageDimension == 3)
        return F::adaptive_avg_pool3d(
          t, F::AdaptiveAvgPool3dFuncOptions({ coarseSpatial[0], coarseSpatial[1], coarseSpatial[2] }));
      else
        return F::adaptive_avg_pool2d(t, F::AdaptiveAvgPool2dFuncOptions({ coarseSpatial[0], coarseSpatial[1] }));
    };
    // The cost sources on the coarse grid, one per kept layer (the intensities alone without a model), each compared
    // with its distance and weighed. With the normalization the weight also divides the layer by its cost at zero
    // displacement, so every layer starts at 1.
    std::vector<torch::Tensor>                 fixedCoarse;
    std::vector<torch::Tensor>                 movingCoarse;
    std::vector<std::unique_ptr<Impact::Loss>> distances;
    std::vector<double>                        weights;
    // With a mask, a cell's features are its masked voxels' mean (a ratio of two pooled sums, the mask brought to the
    // source's grid as its share), and each cell carries its share of the mask.
    auto toMaskedCoarseGrid = [&](const torch::Tensor & t, const torch::Tensor & mask) -> torch::Tensor {
      if (!mask.defined())
      {
        return toCoarseGrid(t);
      }
      const std::vector<int64_t> ts(t.sizes().begin() + 2, t.sizes().end());
      const torch::Tensor        share = Impact::MaskShare<ImageDimension>(mask, ts);
      const torch::Tensor        count = toCoarseGrid(share);
      return torch::where(count > 0.0, toCoarseGrid(t * share) / torch::where(count > 0.0, count, 1.0), 0.0);
    };
    const torch::Tensor fixedShare =
      fixedMask.defined() ? toCoarseGrid(fixedMask.to(torch::kFloat32)) : torch::Tensor();
    const torch::Tensor movingShare =
      movingMask.defined() ? toCoarseGrid(movingMask.to(torch::kFloat32)) : torch::Tensor();
    if (featureMode)
    {
      Impact::LossNormalization normalization;
      for (size_t l = 0; l < fixedLayers.size(); ++l)
      {
        fixedCoarse.push_back(toMaskedCoarseGrid(fixedLayers[l], fixedMask));
        movingCoarse.push_back(toMaskedCoarseGrid(movingLayers[l], movingMask));
        distances.push_back(Impact::LossFactory::Instance().Create(
          m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)]));
        double weight = l < m_LayersWeight.size() ? m_LayersWeight[l] : 1.0;
        if (m_NormalizeLosses && !m_BalanceLosses)
        {
          // At zero displacement: a point-wise distance latches the mean of its terms before the window averages
          // them (for L2 the value latched so far), NCC and LNCC the mean of their windowed cost.
          const Impact::Loss & distance = *distances.back();
          double               value;
          if (!masked)
          {
            value = distance.IsPerPointMean()
                      ? distance.forwardPoints(fixedCoarse.back(), movingCoarse.back())
                          .nan_to_num(1.0)
                          .mean()
                          .template item<double>()
                      : Impact::CoarseLayerCost<ImageDimension>(distance, fixedCoarse.back(), movingCoarse.back())
                          .mean()
                          .template item<double>();
          }
          else
          {
            // The same, weighed by the cells' mask shares at zero displacement.
            const torch::Tensor cellWeight = Impact::CoarseMaskWeight(fixedShare, movingShare);
            const torch::Tensor terms =
              distance.IsPerPointMean()
                ? distance.forwardPoints(fixedCoarse.back(), movingCoarse.back()).nan_to_num(1.0).unsqueeze(1)
                : Impact::CoarseLayerCost<ImageDimension>(
                    distance, fixedCoarse.back(), movingCoarse.back(), cellWeight);
            value = ((terms * cellWeight).sum() / cellWeight.sum().clamp_min(1e-12)).template item<double>();
          }
          weight *= normalization.Latch(l, value);
        }
        weights.push_back(weight);
      }
    }
    else
    {
      fixedCoarse.push_back(toMaskedCoarseGrid(fixedT, fixedMask));
      movingCoarse.push_back(toMaskedCoarseGrid(movingT, movingMask));
      distances.push_back(Impact::LossFactory::Instance().Create("L2"));
      weights.push_back(1.0);
    }
    std::vector<int64_t> coarse(ImageDimension);
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      coarse[d] = fixedCoarse[0].size(2 + static_cast<int64_t>(d));
      if (coarse[d] < 1)
      {
        itkExceptionMacro("ImpactCoarseRegistration: GridSpacing too large for the image size.");
      }
    }

    // ---- Candidate displacement table {Dim, L} (z,y,x), in cells of each axis, shared by forward & backward. ----
    std::vector<int64_t> ssdShape;
    ssdShape.push_back(L);
    for (auto c : coarse)
    {
      ssdShape.push_back(c);
    }
    std::vector<float> meshBuffer(static_cast<size_t>(ImageDimension) * static_cast<size_t>(L)); // {Dim, L}, z,y,x
    for (int64_t l = 0; l < L; ++l)
    {
      int64_t rem = l;
      for (int a = static_cast<int>(ImageDimension) - 1; a >= 0; --a) // axis 0 most significant (x fastest)
      {
        const int64_t side = 2 * halfWidths[a] + 1;
        meshBuffer[static_cast<size_t>(a) * L + l] = static_cast<float>((rem % side) - halfWidths[a]);
        rem /= side;
      }
    }
    torch::Tensor dispMesh =
      torch::from_blob(meshBuffer.data(), { static_cast<int64_t>(ImageDimension), L }, torch::kFloat32)
        .clone()
        .to(device);

    auto gather = [&](const torch::Tensor & indices) -> torch::Tensor {
      // indices {coarse...} long -> {1, Dim, coarse...}, coarse-voxel units, z,y,x.
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(ImageDimension));
      for (auto c : coarse)
      {
        shape.push_back(c);
      }
      return dispMesh.index_select(1, indices.reshape({ -1 })).reshape(shape).unsqueeze(0);
    };
    const std::vector<double> coeffs = { 0.003, 0.01, 0.03, 0.1, 0.3, 1.0 };
    std::vector<int64_t>      dmShape; // {Dim, L, 1, ..., 1} ((Dim-1) trailing singletons)
    dmShape.push_back(static_cast<int64_t>(ImageDimension));
    dmShape.push_back(L);
    for (unsigned int d = 1; d < ImageDimension; ++d)
    {
      dmShape.push_back(1);
    }
    // The coupling penalty weighs each axis's difference in cells by the cell's side in millimetres against cell_ref =
    // GridSpacing * s_min, so that the same distance costs the same along every axis: {Dim, 1, ..., 1}.
    std::vector<int64_t> weightShape(ImageDimension + 1, 1);
    weightShape[0] = static_cast<int64_t>(ImageDimension);
    const torch::Tensor penaltyWeight =
      torch::tensor(cellWeight, torch::kFloat64).to(torch::kFloat32).view(weightShape).to(device);

    // Solve the coarse problem for one (reference, to-shift) pair of cost sources: discrete cost volume over the dense
    // window (each layer's distance, weighed) + coupled-convex regularization -> {1, Dim, coarse...}.
    // BalanceLosses measures each layer's spread on the first volume (fixed onto moving): its own volume is built
    // apart, S_l read off it, and it is added weighed by 1 / S_l; the common factor sum_k S_k / L follows once every
    // layer is in. The backward problem reuses the weights.
    m_LayerSpreads.clear();
    const bool balance = featureMode && m_BalanceLosses;
    // With masks, the cells weigh by their shares (the moving one shifted with the features) and the volume is then
    // filled where it has no data (Impact::FillCoarseCostWithoutData); a coarse voxel with none starts at zero
    // displacement and follows its neighbours through the coupling.
    int64_t zeroCandidate = 0; // the candidate of zero displacement, in the table's order (x fastest)
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      zeroCandidate = zeroCandidate * (2 * halfWidths[a] + 1) + halfWidths[a];
    }
    auto solveCoarse = [&](const std::vector<torch::Tensor> & reference,
                           const std::vector<torch::Tensor> & toShift,
                           const torch::Tensor &              referenceShare,
                           const torch::Tensor &              shiftShare) -> torch::Tensor {
      torch::Tensor       ssd = torch::zeros(ssdShape, reference[0].options());
      const torch::Tensor support =
        masked
          ? Impact::CoarseMaskSupport<ImageDimension>(referenceShare, shiftShare, halfWidths, reference[0].sizes(), L)
          : torch::Tensor();
      if (balance && m_LayerSpreads.empty())
      {
        torch::Tensor layerCost = torch::empty_like(ssd);
        double        total = 0.0;
        size_t        count = 0;
        m_LayerSpreads.assign(reference.size(), 0.0);
        for (size_t l = 0; l < reference.size(); ++l)
        {
          if (weights[l] == 0.0)
          {
            continue;
          }
          layerCost.zero_();
          Impact::AccumulateCoarseCost<ImageDimension>(
            layerCost, *distances[l], reference[l], toShift[l], halfWidths, 1.0, referenceShare, shiftShare);
          const torch::Tensor measured = masked ? Impact::FillCoarseCostWithoutData(layerCost, support) : layerCost;
          const double        spread =
            (std::get<0>(measured.max(0)) - std::get<0>(measured.min(0))).mean().template item<double>();
          m_LayerSpreads[l] = spread;
          if (spread > 0.0 && std::isfinite(spread))
          {
            ssd.add_(layerCost, weights[l] / spread);
            total += spread;
            ++count;
          }
        }
        const double common = count > 0 ? total / static_cast<double>(count) : 1.0;
        ssd.mul_(common);
        for (size_t l = 0; l < reference.size(); ++l) // the weights the backward problem reuses
        {
          const double spread = m_LayerSpreads[l];
          weights[l] *= spread > 0.0 && std::isfinite(spread) ? common / spread : 0.0;
        }
      }
      else
      {
        for (size_t l = 0; l < reference.size(); ++l)
        {
          if (weights[l] != 0.0) // a layer weighed 0 drops out exactly
          {
            Impact::AccumulateCoarseCost<ImageDimension>(
              ssd, *distances[l], reference[l], toShift[l], halfWidths, weights[l], referenceShare, shiftShare);
          }
        }
      }
      torch::Tensor initial = torch::argmin(ssd, 0);
      if (masked)
      {
        ssd = Impact::FillCoarseCostWithoutData(ssd, support);
        initial = torch::where(support.any(0), torch::argmin(ssd, 0), zeroCandidate);
      }
      torch::Tensor disp = smoothBox(gather(initial));
      for (double coeff : coeffs)
      {
        // Tiled over the first coarse axis so the (L x grid) penalty transient stays bounded.
        torch::Tensor       argmin = torch::empty(coarse, torch::TensorOptions().dtype(torch::kLong).device(device));
        const torch::Tensor dm = dispMesh.view(dmShape);
        const torch::Tensor dispC = disp.squeeze(0);
        for (int64_t i = 0; i < coarse[0]; ++i)
        {
          torch::Tensor dvi = dispC.narrow(1, i, 1);
          torch::Tensor penalty = ((dm - dvi).pow(2) * penaltyWeight).sum(0);
          torch::Tensor coupled = ssd.select(1, i) + coeff * penalty;
          argmin.select(0, i).copy_(torch::argmin(coupled, 0));
        }
        disp = smoothBox(gather(argmin));
      }
      return disp;
    };

    torch::Tensor disp = solveCoarse(fixedCoarse, movingCoarse, fixedShare, movingShare); // forward: fixed -> moving

    // ---- Optional inverse consistency: also solve the backward (moving->fixed) problem and
    // symmetrize the two fields toward mutual inverses in normalized [-1,1] grid coordinates
    // (ConvexAdam-style), for a more diffeomorphic coarse initialization. ----
    if (m_InverseConsistency)
    {
      torch::Tensor dispBack = solveCoarse(movingCoarse, fixedCoarse, movingShare, fixedShare); // backward

      // Per-channel normalization (coarse_size - 1)/2, in z,y,x channel order.
      std::vector<float> scaleVals(ImageDimension);
      for (unsigned int c = 0; c < ImageDimension; ++c)
      {
        scaleVals[c] = (coarse[c] > 1) ? static_cast<float>((coarse[c] - 1) / 2.0) : 1.0f;
      }
      std::vector<int64_t> scaleShape(ImageDimension + 2, 1);
      scaleShape[1] = static_cast<int64_t>(ImageDimension);
      torch::Tensor scaleT =
        torch::from_blob(scaleVals.data(), { static_cast<int64_t>(ImageDimension) }, torch::kFloat32)
          .clone()
          .to(device)
          .reshape(scaleShape);

      // Normalize to [-1,1] and flip channel order z,y,x -> x,y,z (grid_sample order).
      torch::Tensor f1 = (disp / scaleT).flip(1);
      torch::Tensor f2 = (dispBack / scaleT).flip(1);

      // Identity sampling grid at coarse resolution, channel-first {1, Dim, coarse...}, x,y,z.
      torch::Tensor idAffine = torch::eye(ImageDimension, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      idAffine =
        torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(ImageDimension), 1 }, idAffine.options()) }, 1)
          .unsqueeze(0);
      std::vector<int64_t> gridSize;
      gridSize.push_back(1);
      gridSize.push_back(1);
      for (auto c : coarse)
      {
        gridSize.push_back(c);
      }
      torch::Tensor        idGrid = torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/true);
      std::vector<int64_t> toChannelFirst;
      toChannelFirst.push_back(0);
      toChannelFirst.push_back(idGrid.dim() - 1);
      for (int64_t dd = 1; dd < idGrid.dim() - 1; ++dd)
      {
        toChannelFirst.push_back(dd);
      }
      torch::Tensor identity = idGrid.permute(toChannelFirst).contiguous(); // {1, Dim, coarse...}

      std::vector<int64_t> toChannelLast;
      toChannelLast.push_back(0);
      for (unsigned int dd = 0; dd < ImageDimension; ++dd)
      {
        toChannelLast.push_back(2 + static_cast<int64_t>(dd));
      }
      toChannelLast.push_back(1);
      const auto icSampleOpts =
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true);

      // Symmetric fixed point: f1 <- 0.5 (f1 - f2 pulled back through id+f1), and vice versa.
      for (int it = 0; it < 15; ++it)
      {
        torch::Tensor f1i = f1.clone();
        torch::Tensor f2i = f2.clone();
        f1 = 0.5 * (f1i - F::grid_sample(f2i, (identity + f1i).permute(toChannelLast), icSampleOpts));
        f2 = 0.5 * (f2i - F::grid_sample(f1i, (identity + f2i).permute(toChannelLast), icSampleOpts));
      }
      // Back to coarse-voxel units, z,y,x channel order.
      disp = f1.flip(1) * scaleT;
    }

    // ---- Upsample to full resolution; coarse-voxel -> full-resolution voxel units. ----
    // Interpolate with an EXACT gs_a stretch (target coarse*gs_a) so the displacement-magnitude scale (gs_a)
    // matches the positional stretch. Otherwise, when a dimension is not divisible by gs_a, interpolate
    // would stretch by full/coarse != gs_a and shear the field toward the high-index border. Tail voxels
    // the coarse avg-pool dropped are then edge-padded.
    std::vector<int64_t> exactSize(ImageDimension);
    bool                 needPad = false;
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      exactSize[d] = coarse[d] * cellVoxels[d];
      if (exactSize[d] != spatial[d])
      {
        needPad = true;
      }
    }
    std::vector<int64_t> componentShape(ImageDimension + 2, 1); // {1, Dim, 1, ...}: one factor per component
    componentShape[1] = static_cast<int64_t>(ImageDimension);
    const torch::Tensor cellsToVoxels = torch::tensor(cellVoxels, torch::kLong).to(disp.options()).view(componentShape);
    torch::Tensor       dispFull;
    if constexpr (ImageDimension == 3)
      dispFull = F::interpolate(
        disp * cellsToVoxels, F::InterpolateFuncOptions().size(exactSize).mode(torch::kTrilinear).align_corners(false));
    else
      dispFull = F::interpolate(
        disp * cellsToVoxels, F::InterpolateFuncOptions().size(exactSize).mode(torch::kBilinear).align_corners(false));
    if (needPad)
    {
      // F::pad pads from the last (x) axis inward; high side only (coarse*gs <= full), replicate.
      std::vector<int64_t> pad;
      for (int a = static_cast<int>(ImageDimension) - 1; a >= 0; --a)
      {
        pad.push_back(0);
        pad.push_back(spatial[a] - exactSize[a]);
      }
      dispFull = F::pad(dispFull, F::PadFuncOptions(pad).mode(torch::kReplicate));
    } // dispFull: {1, Dim, spatial...}, full-res voxel units, z,y,x

    // ---- Geometry-correct writeback (shared convention). ----
    DisplacementFieldType * output = this->GetOutput();
    output->SetRegions(m_FixedImage->GetLargestPossibleRegion());
    output->SetSpacing(m_FixedImage->GetSpacing());
    output->SetOrigin(m_FixedImage->GetOrigin());
    output->SetDirection(m_FixedImage->GetDirection());
    output->Allocate();

    Impact::WriteVoxelFieldToDisplacement<ImageDimension>(
      dispFull, m_FixedImage->GetSpacing(), m_FixedImage->GetDirection(), output);

    m_DisplacementFieldTransform = DisplacementFieldTransformType::New();
    m_DisplacementFieldTransform->SetDisplacementField(output);
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Device: " << m_Device << std::endl;
  os << indent << "GridSpacing: " << m_GridSpacing << std::endl;
  os << indent << "DisplacementHalfWidth: " << m_DisplacementHalfWidth << std::endl;
  os << indent << "FixedModelsConfiguration count: " << m_FixedModelsConfiguration.size() << std::endl;
  os << indent << "NormalizeLosses: " << (m_NormalizeLosses ? "on" : "off") << std::endl;
  os << indent << "BalanceLosses: " << (m_BalanceLosses ? "on" : "off") << std::endl;
  os << indent << "LayersWeight:";
  for (const float weight : m_LayersWeight)
  {
    os << ' ' << weight;
  }
  os << std::endl;
}

} // end namespace itk

#endif // itkImpactCoarseRegistration_hxx

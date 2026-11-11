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
#include "itkImpactRegionSampling.h"
#include <itkImageRegionIteratorWithIndex.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>

#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
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
 * averaged over the window -- for L2 exactly ConvexAdam's smoothed SSD --, a global distance (NCC, LNCC, OrganDice)
 * each channel's similarity over the window (Loss::SimilarityFromMoments), from its weighted means of f, m, f^2, m^2
 * and f*m (the weight falling outside the image dropped). A point-wise term that is not a number (the cosine of a zero
 * vector) counts as no similarity, 1.
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
    const torch::Tensor similarity =
      distance.SimilarityFromMoments(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct);
    return 1.0 - similarity.mean(1, /*keepdim=*/true);
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
  const torch::Tensor similarity =
    distance.SimilarityFromMoments(meanFixed, meanMoving, meanFixedSquare, meanMovingSquare, meanProduct);
  return torch::where(held, 1.0 - similarity.mean(1, /*keepdim=*/true), 0.0);
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

/** `volume` padded with zeros by `halfWidths` on each side of each axis: the moving features as ConvexAdam pads them,
 * a mask's share because no voxel outside the image is in the mask. */
template <unsigned int Dim>
torch::Tensor
CoarsePad(const torch::Tensor & volume, const std::vector<int64_t> & halfWidths)
{
  namespace F = torch::nn::functional;
  std::vector<int64_t> padding; // F::pad takes its pairs from the last axis backwards
  for (int a = static_cast<int>(Dim) - 1; a >= 0; --a)
  {
    padding.push_back(halfWidths[a]);
    padding.push_back(halfWidths[a]);
  }
  return F::pad(volume, F::PadFuncOptions(padding).mode(torch::kConstant).value(0.0));
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
 * shifted by it (zeros past the image) against `fixed`, both {1, C, coarse...}. With masks, `fixedShare` and `movingShare`
 * ({1, 1, coarse...}, either undefined when its mask is absent) weigh the cells, the moving one shifted with the
 * features (zero outside the image). */
/** AccumulateCoarseCost with the moving features and share already padded by the half-widths (`padded`,
 * `paddedShare`): what a box of the grid hands over, its margins read from the grid where it has neighbours and
 * padded where the grid ends. */
template <unsigned int Dim>
void
AccumulateCoarseCostPadded(torch::Tensor &              cost,
                           const Loss &                 distance,
                           const torch::Tensor &        fixed,
                           const torch::Tensor &        padded,
                           const std::vector<int64_t> & halfWidths,
                           double                       weight,
                           const torch::Tensor &        fixedShare = {},
                           const torch::Tensor &        paddedShare = {})
{
  const bool masked = fixedShare.defined() || paddedShare.defined();
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
  AccumulateCoarseCostPadded<Dim>(cost,
                                  distance,
                                  fixed,
                                  CoarsePad<Dim>(moving, halfWidths),
                                  halfWidths,
                                  weight,
                                  fixedShare,
                                  movingShare.defined() ? CoarsePad<Dim>(movingShare, halfWidths) : movingShare);
}

/** Where each candidate leaves the cell window around each coarse voxel some mask weight, {L, coarse...} bool: the
 * coarse voxels and candidates a masked cost has data for (see AccumulateCoarseCost). */
/** CoarseMaskSupport with the moving share already padded by the half-widths (see AccumulateCoarseCostPadded). */
template <unsigned int Dim>
torch::Tensor
CoarseMaskSupportPadded(const torch::Tensor &        fixedShare,
                        const torch::Tensor &        paddedShare,
                        const std::vector<int64_t> & halfWidths,
                        c10::IntArrayRef             coarse,
                        int64_t                      candidates)
{
  std::vector<int64_t> shape{ candidates };
  shape.insert(shape.end(), coarse.begin() + 2, coarse.end());
  torch::Tensor support = torch::empty(
    shape,
    torch::TensorOptions().dtype(torch::kBool).device((fixedShare.defined() ? fixedShare : paddedShare).device()));
  for (int64_t l = 0; l < candidates; ++l)
  {
    const torch::Tensor cellWeight = CoarseMaskWeight(
      fixedShare, paddedShare.defined() ? CoarseCandidateShift<Dim>(paddedShare, halfWidths, l, coarse) : paddedShare);
    support[l] = (CoarseBoxWindow<Dim>(cellWeight) > 0.0).squeeze(0).squeeze(0);
  }
  return support;
}

template <unsigned int Dim>
torch::Tensor
CoarseMaskSupport(const torch::Tensor &        fixedShare,
                  const torch::Tensor &        movingShare,
                  const std::vector<int64_t> & halfWidths,
                  c10::IntArrayRef             coarse,
                  int64_t                      candidates)
{
  return CoarseMaskSupportPadded<Dim>(
    fixedShare,
    movingShare.defined() ? CoarsePad<Dim>(movingShare, halfWidths) : movingShare,
    halfWidths,
    coarse,
    candidates);
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
ImpactCoarseRegistration<TFixedImage, TMovingImage>::ImpactCoarseRegistration()
{
  this->DynamicMultiThreadingOn();
}

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
  const_cast<FixedImageType *>(m_FixedImage.GetPointer())->UpdateOutputInformation();
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

  if (m_RegionSize > 0)
  {
    // The out-of-core run: solved once, then the requested region of the dense field is interpolated from the coarse
    // grid (ImageSource's threaded GenerateData, through DynamicThreadedGenerateData).
    if (!m_Solved || m_SolvedTime < this->GetMTime())
    {
      this->SolveByRegions();
      m_Solved = true;
      m_SolvedTime = this->GetMTime();
    }
    Superclass::GenerateData();
    m_DisplacementFieldTransform = this->GetCoarseGridTransform();
    return;
  }

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
    // The stage's own generator, on the CPU. torch::manual_seed reseeded every generator of the
    // process, a Python caller's included, and a draw on the device also followed whatever else draws
    // there, such as the batch-budget probe. Draws are made here and moved to the device.
    at::Generator      generator = at::make_generator<at::CPUGeneratorImpl>(m_Seed);
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
            torch::randperm(fixedLayers[l].size(1), generator, torch::kLong).narrow(0, 0, kept).to(device);
          fixedLayers[l] = fixedLayers[l].index_select(1, channels).contiguous();
          movingLayers[l] = movingLayers[l].index_select(1, channels).contiguous();
        }
      }
    }

    // ---- The search, in voxels on every axis whatever its spacing, as ConvexAdam's: cells of GridSpacing voxels,
    // DisplacementHalfWidth cells each way. ----
    const int64_t              gs = static_cast<int64_t>(m_GridSpacing);
    const int64_t              hw = static_cast<int64_t>(m_DisplacementHalfWidth);
    const std::vector<int64_t> cellVoxels(ImageDimension, gs);
    const std::vector<int64_t> halfWidths(ImageDimension, hw);
    int64_t                    L = 1;
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      L *= 2 * hw + 1;
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
          torch::Tensor penalty = (dm - dvi).pow(2).sum(0);
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
      torch::Tensor        idGrid = torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/false);
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
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);

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

    // ---- Upsample to full resolution, as ConvexAdam does: cells to voxels, then straight to the image's size, which
    // stretches the cells over the tail a size that is no multiple of the spacing leaves. ----
    std::vector<int64_t> componentShape(ImageDimension + 2, 1); // {1, Dim, 1, ...}: one factor per component
    componentShape[1] = static_cast<int64_t>(ImageDimension);
    const torch::Tensor cellsToVoxels = torch::tensor(cellVoxels, torch::kLong).to(disp.options()).view(componentShape);
    m_CoarseGrid = Impact::BuildCellGrid<ImageDimension>(disp * cellsToVoxels, m_FixedImage.GetPointer());
    m_CoarseGridTransform = nullptr;
    torch::Tensor dispFull; // {1, Dim, spatial...}, full-res voxel units, z,y,x
    if constexpr (ImageDimension == 3)
      dispFull = F::interpolate(
        disp * cellsToVoxels, F::InterpolateFuncOptions().size(spatial).mode(torch::kTrilinear).align_corners(false));
    else
      dispFull = F::interpolate(
        disp * cellsToVoxels, F::InterpolateFuncOptions().size(spatial).mode(torch::kBilinear).align_corners(false));

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
auto
ImpactCoarseRegistration<TFixedImage, TMovingImage>::GetCoarseGridTransform() -> DisplacementFieldTransformType *
{
  if (m_CoarseGrid.IsNull())
  {
    // Solve without allocating the dense field: request one voxel of it.
    this->UpdateOutputInformation();
    typename DisplacementFieldType::RegionType one = this->GetOutput()->GetLargestPossibleRegion();
    one.SetSize(DisplacementFieldType::SizeType::Filled(1));
    this->GetOutput()->SetRequestedRegion(one);
    this->Update();
  }
  if (m_CoarseGridTransform.IsNull() || m_CoarseGridTransform->GetDisplacementField() != m_CoarseGrid.GetPointer())
  {
    m_CoarseGridTransform = DisplacementFieldTransformType::New();
    m_CoarseGridTransform->SetDisplacementField(m_CoarseGrid);
  }
  return m_CoarseGridTransform;
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactCoarseRegistration<TFixedImage, TMovingImage>::DynamicThreadedGenerateData(
  const typename Superclass::OutputImageRegionType & region)
{
  using InterpolatorType = VectorLinearInterpolateImageFunction<DisplacementFieldType, double>;
  auto interpolator = InterpolatorType::New();
  interpolator->SetInputImage(m_CoarseGrid);
  const auto &                                        gridSize = m_CoarseGrid->GetLargestPossibleRegion().GetSize();
  ImageRegionIteratorWithIndex<DisplacementFieldType> it(this->GetOutput(), region);
  typename FixedImageType::PointType                  point;
  typename InterpolatorType::ContinuousIndexType      index;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    m_FixedImage->TransformIndexToPhysicalPoint(it.GetIndex(), point);
    m_CoarseGrid->TransformPhysicalPointToContinuousIndex(point, index);
    for (unsigned int a = 0; a < ImageDimension; ++a)
    {
      // Round-off can put a voxel a hair past the padded grid.
      index[a] = std::clamp(index[a], 0.0, static_cast<double>(gridSize[a] - 1));
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
ImpactCoarseRegistration<TFixedImage, TMovingImage>::SolveByRegions()
{
  namespace F = torch::nn::functional;
  constexpr unsigned int Dim = ImageDimension;
  using RegionType = typename FixedImageType::RegionType;
  if (m_FixedImage.IsNull() || m_MovingImage.IsNull())
  {
    itkExceptionMacro("Both SetFixedImage() and SetMovingImage() are required.");
  }
  if constexpr (Dim != 2 && Dim != 3)
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
    for (const unsigned int components : m_PCA)
    {
      if (components > 0)
      {
        itkExceptionMacro("ImpactCoarseRegistration: PCA fits its basis on the whole fixed feature map; set it to 0 "
                          "with RegionSize.");
      }
    }
    m_BytesRead = 0;
    m_FixedImageStreamed = false;
    m_MovingImageStreamed = false;
    m_LayerSpreads.clear();

    const torch::Device device(m_Device);
    torch::manual_seed(m_Seed);
    torch::NoGradGuard                  noGrad; // the coarse stage is non-differentiable
    const Impact::PythonGilReleaseGuard gilRelease;
    const auto                          onDevice = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // ---- Geometry. A reader's output knows its grid once its information is updated; nothing is read yet. ----
    auto * fixedSource = const_cast<FixedImageType *>(m_FixedImage.GetPointer());
    auto * movingSource = const_cast<MovingImageType *>(m_MovingImage.GetPointer());
    fixedSource->UpdateOutputInformation();
    movingSource->UpdateOutputInformation();
    const RegionType     fixedRegion = m_FixedImage->GetLargestPossibleRegion();
    std::vector<int64_t> spatial(Dim); // tensor order (z, y, x)
    for (unsigned int d = 0; d < Dim; ++d)
    {
      spatial[d] = static_cast<int64_t>(fixedRegion.GetSize()[Dim - 1 - d]);
    }
    const int64_t              gs = static_cast<int64_t>(m_GridSpacing);
    const int64_t              hw = static_cast<int64_t>(m_DisplacementHalfWidth);
    const std::vector<int64_t> cellVoxels(Dim, gs);
    const std::vector<int64_t> halfWidths(Dim, hw);
    int64_t                    L = 1;
    for (unsigned int a = 0; a < Dim; ++a)
    {
      L *= 2 * hw + 1;
    }
    std::vector<int64_t> coarse(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      coarse[d] = spatial[d] / cellVoxels[d];
      if (coarse[d] < 1)
      {
        itkExceptionMacro("ImpactCoarseRegistration: GridSpacing too large for the image size.");
      }
    }
    // The boxes of cells read at a time: RegionSize voxels a side, whole cells.
    const std::vector<int64_t> regionVoxels(Dim, static_cast<int64_t>(m_RegionSize));
    std::vector<int64_t> perBox(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      perBox[d] = std::max<int64_t>(1, regionVoxels[d] / cellVoxels[d]);
    }
    const std::vector<Impact::CellBox> boxes = Impact::TileCells(coarse, perBox);
    // A model's receptive field is in its input's voxels, on every axis whatever their spacing.
    const std::vector<int64_t> halo(Dim, static_cast<int64_t>(m_RegionHalo));

    // ---- The moving image and its mask on the fixed grid, each region brought there through the moving transform
    // (the identity without one) by a resampler that asks its input for that region alone; a fixed mask on another
    // grid likewise through the identity. An image already on the grid, with nothing to read it through, is read as
    // it is. ----
    auto sameGrid = [](const auto * a, const auto * b) { return Impact::SameGrid(a, b); };
    const typename TransformType::ConstPointer identity = IdentityTransform<double, Dim>::New().GetPointer();
    std::vector<ProcessObject::Pointer>       resamplers; // alive for as long as their outputs are read
    auto onFixedGrid = [&](auto image, auto interpolator, const TransformType * transform) {
      using ImageType = std::remove_const_t<std::remove_pointer_t<decltype(image)>>;
      using ResampleType = ResampleImageFilter<ImageType, ImageType, double>;
      auto resample = ResampleType::New();
      resample->SetInput(image);
      resample->SetTransform(transform);
      resample->SetInterpolator(interpolator);
      resample->SetUseReferenceImage(true);
      resample->SetReferenceImage(m_FixedImage);
      resample->UpdateOutputInformation();
      resamplers.push_back(resample.GetPointer());
      return typename ImageType::ConstPointer(resample->GetOutput());
    };
    const TransformType * movingTransform = m_MovingTransform.IsNotNull() ? m_MovingTransform.GetPointer() : nullptr;
    typename MovingImageType::ConstPointer movingOnFixed = m_MovingImage;
    if (movingTransform != nullptr || !sameGrid(m_MovingImage.GetPointer(), m_FixedImage.GetPointer()))
    {
      movingOnFixed = onFixedGrid(m_MovingImage.GetPointer(),
                                  LinearInterpolateImageFunction<MovingImageType, double>::New(),
                                  movingTransform != nullptr ? movingTransform : identity.GetPointer());
    }
    auto maskOnFixed = [&](const MaskImageType * mask, const TransformType * transform) ->
      typename MaskImageType::ConstPointer {
        if (mask == nullptr)
        {
          return nullptr;
        }
        const_cast<MaskImageType *>(mask)->UpdateOutputInformation();
        if (transform == nullptr && sameGrid(mask, m_FixedImage.GetPointer()))
        {
          return mask;
        }
        return onFixedGrid(mask,
                           NearestNeighborInterpolateImageFunction<MaskImageType, double>::New(),
                           transform != nullptr ? transform : identity.GetPointer());
      };
    const typename MaskImageType::ConstPointer fixedMask = maskOnFixed(m_FixedMask.GetPointer(), nullptr);
    const typename MaskImageType::ConstPointer movingMask = maskOnFixed(m_MovingMask.GetPointer(), movingTransform);

    // ---- Reads: each through the image's own pipeline, counted; whether the first was served without the whole
    // image tells if the input streams. ----
    bool fixedProbed = false, movingProbed = false;
    auto readFixed = [&](const RegionType & region) {
      m_BytesRead += region.GetNumberOfPixels() * sizeof(typename FixedImageType::PixelType);
      torch::Tensor tensor = Impact::ReadRegion(m_FixedImage.GetPointer(), region);
      if (!fixedProbed)
      {
        m_FixedImageStreamed =
          m_FixedImage->GetBufferedRegion().GetNumberOfPixels() < fixedRegion.GetNumberOfPixels();
        fixedProbed = true;
      }
      return tensor.to(device);
    };
    auto readMoving = [&](const RegionType & region) {
      m_BytesRead += region.GetNumberOfPixels() * sizeof(typename MovingImageType::PixelType);
      torch::Tensor tensor = Impact::ReadRegion(movingOnFixed.GetPointer(), region);
      if (!movingProbed)
      {
        m_MovingImageStreamed = m_MovingImage->GetBufferedRegion().GetNumberOfPixels() <
                                m_MovingImage->GetLargestPossibleRegion().GetNumberOfPixels();
        movingProbed = true;
      }
      return tensor.to(device);
    };
    auto readMask = [&](const MaskImageType * mask, const RegionType & region) {
      m_BytesRead += region.GetNumberOfPixels();
      return (Impact::ReadRegion(mask, region) != 0).to(torch::kFloat32).to(device);
    };

    // ---- A box's span on the fixed grid: the cells [c0, c1) in voxels with `extra` around, clipped to the image;
    // the pooling of whole cells; the cells of a map. ----
    std::vector<Impact::PoolWindows> windows(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      windows[d] = Impact::UniformPoolWindows(coarse[d], cellVoxels[d]);
    }
    auto spanOf = [&](const std::vector<int64_t> & c0,
                      const std::vector<int64_t> & c1,
                      const std::vector<int64_t> & extra,
                      std::vector<int64_t> &       lowPad,
                      std::vector<int64_t> &       core,
                      int64_t                      align = 1) {
      return Impact::CellSpan<RegionType>(c0, c1, extra, windows, spatial, fixedRegion.GetIndex(), lowPad, core, align);
    };
    const auto narrowTo = Impact::NarrowTo;
    const auto writeCells = Impact::WriteCells;

    // ---- The feature models, run on each region whole at the image's resolution. ----
    size_t layerCount = 1;
    bool   takesMetadata = false; // a model that reads the image's statistics
    if (featureMode)
    {
      const Impact::RegionModels models = Impact::CheckRegionModels(
        m_FixedModelsConfiguration, m_FixedImage->GetSpacing(), Dim, "ImpactCoarseRegistration");
      layerCount = models.layerCount;
      takesMetadata = models.takesMetadata;
    }
    const std::vector<ImpactModelConfiguration> & movingConfigs =
      m_MovingModelsConfiguration.empty() ? m_FixedModelsConfiguration : m_MovingModelsConfiguration;
    // A metadata-aware model (MIND reads the image's range) takes the image's statistics: one pass over each image
    // by blocks, the moving one on the fixed grid, as the dense run's StatisticsImageFilter takes them.
    ImpactImageMetadata fixedMetadata, movingMetadata;
    if (takesMetadata)
    {
      auto blocksOf = [&](auto && take) {
        std::vector<int64_t> step(Dim), counts(Dim);
        size_t               total = 1;
        for (unsigned int d = 0; d < Dim; ++d)
        {
          step[d] = std::max<int64_t>(1, std::min(spatial[d], regionVoxels[d]));
          counts[d] = (spatial[d] + step[d] - 1) / step[d];
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
            block.SetIndex(Dim - 1 - d, fixedRegion.GetIndex()[Dim - 1 - d] + static_cast<IndexValueType>(start));
            block.SetSize(Dim - 1 - d, static_cast<SizeValueType>(std::min(step[d], spatial[d] - start)));
          }
          take(block);
        }
      };
      auto statistics = [&](auto && readBlock) {
        return Impact::MetadataByBlocks(m_FixedImage->GetDirection(), Dim, [&](auto && take) {
          blocksOf([&](const RegionType & block) { take(readBlock(block)); });
        });
      };
      fixedMetadata = statistics([&](const RegionType & block) { return readFixed(block); });
      movingMetadata = statistics([&](const RegionType & block) { return readMoving(block); });
    }
    // The feature layers of an image region {1, 1, span...}, at its resolution, each model run on the region whole
    // in one forward; the region itself in intensity mode.
    auto layersOf = [&](const torch::Tensor & image, bool moving) {
      return Impact::RegionLayers<Dim>(featureMode ? (moving ? movingConfigs : m_FixedModelsConfiguration)
                                                   : std::vector<ImpactModelConfiguration>{},
                                       image,
                                       device,
                                       layerCount,
                                       takesMetadata ? &(moving ? movingMetadata : fixedMetadata) : nullptr,
                                       "ImpactCoarseRegistration");
    };

    // ---- One pass over the boxes: each image's features pooled to the coarse grid, resident on the device, and
    // the masks' shares of the cells. SubsetFeatures draws its channels once, at the first box. ----
    std::vector<torch::Tensor> fixedMaps(layerCount), movingMaps(layerCount), subset(layerCount);
    torch::Tensor              fixedShare, movingShare;
    std::vector<int64_t>       shareShape{ 1, 1 };
    shareShape.insert(shareShape.end(), coarse.begin(), coarse.end());
    if (fixedMask.IsNotNull())
    {
      fixedShare = torch::zeros(shareShape, onDevice);
    }
    if (movingMask.IsNotNull())
    {
      movingShare = torch::zeros(shareShape, onDevice);
    }
    // A layer coarser than the image lies on the image's lattice every `align` voxels: the regions read for the
    // features start on it, and reach what pooling the whole layer to the cells reads past a cell's own voxels.
    int64_t                           align = 1;
    std::vector<int64_t>              reach = halo;
    std::vector<Impact::LayerLattice> lattices(layerCount, Impact::ImageLattice(Dim));
    std::vector<Impact::LayerLattice> movingLattices = lattices; // the moving models' own, when they differ
    if (featureMode)
    {
      std::vector<int64_t> probe(Dim + 2, Impact::LatticeProbe);
      probe[0] = probe[1] = 1;
      lattices = Impact::LayerLattices<Dim>(layersOf(torch::zeros(probe, onDevice), false), "ImpactCoarseRegistration");
      movingLattices =
        m_MovingModelsConfiguration.empty()
          ? lattices
          : Impact::LayerLattices<Dim>(layersOf(torch::zeros(probe, onDevice), true), "ImpactCoarseRegistration");
      for (const auto * side : { &lattices, &movingLattices })
      {
        for (const Impact::LayerLattice & lattice : *side)
        {
          align = std::max(align, *std::max_element(lattice.stride.begin(), lattice.stride.end()));
        }
      }
      for (unsigned int d = 0; d < Dim && align > 1; ++d)
      {
        reach[d] += cellVoxels[d] + 2 * align;
      }
    }
    for (size_t b = 0; b < boxes.size(); ++b)
    {
      const Impact::CellBox &    box = boxes[b];
      std::vector<int64_t>       lowPad(Dim), core(Dim), low(Dim);
      const RegionType           span = spanOf(box.cell0, box.cell1, reach, lowPad, core, align);
      const torch::Tensor        fixedSpan = readFixed(span);
      const torch::Tensor        movingSpan = readMoving(span);
      for (unsigned int d = 0; d < Dim; ++d)
      {
        low[d] = windows[d].starts[box.cell0[d]] - lowPad[d];
      }
      auto toCells = [&](const torch::Tensor & t, const Impact::LayerLattice & lattice) {
        return Impact::PoolLayerToCells<Dim>(
          t, lattice, lowPad, box.cell0, box.cell1, windows, spatial, "ImpactCoarseRegistration");
      };
      // A cell's features: the mean of the layer's samples it pools, of its masked ones with a mask (brought to the
      // layer's samples as its share; a cell without any: 0).
      auto pooledCells = [&](const torch::Tensor &        layer,
                             const Impact::LayerLattice & lattice,
                             const torch::Tensor &        mask) -> torch::Tensor {
        if (!mask.defined())
        {
          return toCells(layer, lattice);
        }
        const torch::Tensor share = Impact::SpanMaskShare<Dim>(mask, lattice, low, spatial);
        const torch::Tensor count = toCells(share, lattice);
        return torch::where(
          count > 0.0, toCells(layer * share, lattice) / torch::where(count > 0.0, count, 1.0), 0.0);
      };
      const Impact::LayerLattice voxels = Impact::ImageLattice(Dim);
      torch::Tensor fixedMaskBox, movingMaskBox;
      if (fixedMask.IsNotNull())
      {
        fixedMaskBox = readMask(fixedMask, span);
        writeCells(fixedShare, box.cell0, toCells(fixedMaskBox, voxels));
      }
      if (movingMask.IsNotNull())
      {
        movingMaskBox = readMask(movingMask, span);
        writeCells(movingShare, box.cell0, toCells(movingMaskBox, voxels));
      }
      const std::vector<torch::Tensor> fixedLayers = layersOf(fixedSpan, false);
      const std::vector<torch::Tensor> movingLayers = layersOf(movingSpan, true);
      if (b == 0)
      {
        for (size_t l = 0; l < layerCount; ++l)
        {
          const int64_t channels = fixedLayers[l].size(1);
          const int64_t kept = l < m_SubsetFeatures.size() ? static_cast<int64_t>(m_SubsetFeatures[l]) : 0;
          if (kept > 0 && kept < channels)
          {
            subset[l] = torch::randperm(channels, torch::TensorOptions().dtype(torch::kLong)).narrow(0, 0, kept).to(device);
          }
          std::vector<int64_t> shape{ 1, subset[l].defined() ? kept : channels };
          shape.insert(shape.end(), coarse.begin(), coarse.end());
          fixedMaps[l] = torch::zeros(shape, onDevice);
          movingMaps[l] = torch::zeros(shape, onDevice);
        }
      }
      for (size_t l = 0; l < layerCount; ++l)
      {
        torch::Tensor fixedLayer = fixedLayers[l];
        torch::Tensor movingLayer = movingLayers[l];
        if (subset[l].defined())
        {
          fixedLayer = fixedLayer.index_select(1, subset[l]);
          movingLayer = movingLayer.index_select(1, subset[l]);
        }
        writeCells(fixedMaps[l], box.cell0, pooledCells(fixedLayer, lattices[l], fixedMaskBox));
        writeCells(movingMaps[l], box.cell0, pooledCells(movingLayer, movingLattices[l], movingMaskBox));
      }
    }

    // ---- Distances and weights, the normalization at zero displacement on the resident maps, as the dense run. ----
    const bool                                 masked = fixedShare.defined() || movingShare.defined();
    std::vector<std::unique_ptr<Impact::Loss>> distances;
    std::vector<double>                        weights;
    if (featureMode)
    {
      Impact::LossNormalization normalization;
      for (size_t l = 0; l < layerCount; ++l)
      {
        distances.push_back(Impact::LossFactory::Instance().Create(
          m_Distance.empty() ? std::string("L2") : m_Distance[std::min(l, m_Distance.size() - 1)]));
        double weight = l < m_LayersWeight.size() ? m_LayersWeight[l] : 1.0;
        if (m_NormalizeLosses && !m_BalanceLosses)
        {
          const Impact::Loss & distance = *distances.back();
          double               value;
          if (!masked)
          {
            value = distance.IsPerPointMean()
                      ? distance.forwardPoints(fixedMaps[l], movingMaps[l]).nan_to_num(1.0).mean().template item<double>()
                      : Impact::CoarseLayerCost<Dim>(distance, fixedMaps[l], movingMaps[l]).mean().template item<double>();
          }
          else
          {
            const torch::Tensor cellWeights = Impact::CoarseMaskWeight(fixedShare, movingShare);
            const torch::Tensor terms =
              distance.IsPerPointMean()
                ? distance.forwardPoints(fixedMaps[l], movingMaps[l]).nan_to_num(1.0).unsqueeze(1)
                : Impact::CoarseLayerCost<Dim>(distance, fixedMaps[l], movingMaps[l], cellWeights);
            value = ((terms * cellWeights).sum() / cellWeights.sum().clamp_min(1e-12)).template item<double>();
          }
          weight *= normalization.Latch(l, value);
        }
        weights.push_back(weight);
      }
    }
    else
    {
      distances.push_back(Impact::LossFactory::Instance().Create("L2"));
      weights.push_back(1.0);
    }

    // ---- The candidate table {Dim, L} (z,y,x) in cells, and the coupling, as the dense run. ----
    std::vector<float> meshBuffer(static_cast<size_t>(Dim) * static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l)
    {
      int64_t rem = l;
      for (int a = static_cast<int>(Dim) - 1; a >= 0; --a) // axis 0 most significant (x fastest)
      {
        const int64_t side = 2 * halfWidths[a] + 1;
        meshBuffer[static_cast<size_t>(a) * L + l] = static_cast<float>((rem % side) - halfWidths[a]);
        rem /= side;
      }
    }
    const torch::Tensor dispMesh =
      torch::from_blob(meshBuffer.data(), { static_cast<int64_t>(Dim), L }, torch::kFloat32).clone().to(device);
    auto gather = [&](const torch::Tensor & indices) -> torch::Tensor {
      std::vector<int64_t> shape{ static_cast<int64_t>(Dim) };
      shape.insert(shape.end(), coarse.begin(), coarse.end());
      return dispMesh.index_select(1, indices.reshape({ -1 })).reshape(shape).unsqueeze(0);
    };
    auto smoothBox = [&](const torch::Tensor & t) -> torch::Tensor {
      if constexpr (Dim == 3)
        return F::avg_pool3d(t, F::AvgPool3dFuncOptions(3).stride(1).padding(1));
      else
        return F::avg_pool2d(t, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    };
    const std::vector<double> coeffs = { 0.003, 0.01, 0.03, 0.1, 0.3, 1.0 };
    std::vector<int64_t>      dmShape{ static_cast<int64_t>(Dim), L }; // {Dim, L, 1, ..., 1}
    for (unsigned int d = 1; d < Dim; ++d)
    {
      dmShape.push_back(1);
    }
    int64_t zeroCandidate = 0; // the candidate of zero displacement, in the table's order (x fastest)
    for (unsigned int a = 0; a < Dim; ++a)
    {
      zeroCandidate = zeroCandidate * (2 * halfWidths[a] + 1) + halfWidths[a];
    }
    const bool balance = featureMode && m_BalanceLosses;

    // ---- The cost volume by boxes of the coarse grid, never whole: a box's cost over every candidate is built from
    // the maps over the box with its margins, the cell window's two cells and the candidates' half-widths, read from
    // the grid where it has neighbours and padded where the grid ends as the whole map is (CoarsePad), then cropped
    // to the box. ----
    const int64_t margin = 2;
    const double  perCell = 8.0 * static_cast<double>(L); // the cost over the candidates, and as much in transients
    const double  side =
      std::pow(static_cast<double>(std::max<SizeValueType>(m_CostBoxBytes, 1)) / perCell, 1.0 / static_cast<double>(Dim)) -
      2.0 * static_cast<double>(margin);
    const std::vector<int64_t>         perCost(Dim, std::max<int64_t>(1, static_cast<int64_t>(std::floor(side))));
    const std::vector<Impact::CellBox> costBoxes = Impact::TileCells(coarse, perCost);
    // `only` >= 0: that layer alone, unweighed (BalanceLosses measures each layer's spread); `support` comes back as
    // the masked cost's support over the box.
    auto costOf = [&](const std::vector<torch::Tensor> & reference,
                      const std::vector<torch::Tensor> & toShift,
                      const torch::Tensor &              referenceShare,
                      const torch::Tensor &              shiftShare,
                      const Impact::CellBox &            box,
                      int                                only,
                      torch::Tensor &                    support) -> torch::Tensor {
      std::vector<int64_t> lo(Dim), hi(Dim), loM(Dim), hiM(Dim), padding;
      for (unsigned int d = 0; d < Dim; ++d)
      {
        lo[d] = std::max<int64_t>(0, box.cell0[d] - margin);
        hi[d] = std::min<int64_t>(coarse[d], box.cell1[d] + margin);
        loM[d] = std::max<int64_t>(0, lo[d] - halfWidths[d]);
        hiM[d] = std::min<int64_t>(coarse[d], hi[d] + halfWidths[d]);
      }
      for (int a = static_cast<int>(Dim) - 1; a >= 0; --a) // F::pad takes its pairs from the last axis backwards
      {
        padding.push_back(halfWidths[a] - (lo[a] - loM[a]));
        padding.push_back(halfWidths[a] - (hiM[a] - hi[a]));
      }
      auto shiftWindow = [&](const torch::Tensor & t) {
        return F::pad(narrowTo(t, loM, hiM, 2), F::PadFuncOptions(padding).mode(torch::kConstant).value(0.0));
      };
      const torch::Tensor  refShare = referenceShare.defined() ? narrowTo(referenceShare, lo, hi, 2) : referenceShare;
      const torch::Tensor  paddedShare = shiftShare.defined() ? shiftWindow(shiftShare) : shiftShare;
      std::vector<int64_t> shape{ L };
      for (unsigned int d = 0; d < Dim; ++d)
      {
        shape.push_back(hi[d] - lo[d]);
      }
      torch::Tensor cost = torch::zeros(shape, onDevice);
      for (size_t l = 0; l < reference.size(); ++l)
      {
        if (only >= 0 ? static_cast<size_t>(only) != l : weights[l] == 0.0)
        {
          continue;
        }
        Impact::AccumulateCoarseCostPadded<Dim>(cost,
                                                *distances[l],
                                                narrowTo(reference[l], lo, hi, 2),
                                                shiftWindow(toShift[l]),
                                                halfWidths,
                                                only >= 0 ? 1.0 : weights[l],
                                                refShare,
                                                paddedShare);
      }
      if (masked)
      {
        support = Impact::CoarseMaskSupportPadded<Dim>(
          refShare, paddedShare, halfWidths, narrowTo(reference[0], lo, hi, 2).sizes(), L);
        cost = Impact::FillCoarseCostWithoutData(cost, support);
      }
      for (unsigned int d = 0; d < Dim; ++d)
      {
        const int64_t axis = 1 + static_cast<int64_t>(d);
        cost = cost.narrow(axis, box.cell0[d] - lo[d], box.cell1[d] - box.cell0[d]);
        if (support.defined())
        {
          support = support.narrow(axis, box.cell0[d] - lo[d], box.cell1[d] - box.cell0[d]);
        }
      }
      return cost;
    };

    // ---- The coupled-convex solve on the resident maps: the argmin by boxes of the grid, the field smoothed
    // whole, the six couplings. ----
    auto solveCoarse = [&](const std::vector<torch::Tensor> & reference,
                           const std::vector<torch::Tensor> & toShift,
                           const torch::Tensor &              referenceShare,
                           const torch::Tensor &              shiftShare) -> torch::Tensor {
      if (balance && m_LayerSpreads.empty())
      {
        // Each layer's spread over the candidates, averaged over the grid, measured on its own volume by boxes; the
        // layers then weigh by 1 / S_l, times the common factor sum_k S_k / L, as the dense run.
        m_LayerSpreads.assign(reference.size(), 0.0);
        double total = 0.0;
        size_t count = 0;
        for (size_t l = 0; l < reference.size(); ++l)
        {
          if (weights[l] == 0.0)
          {
            continue;
          }
          double  sum = 0.0;
          int64_t cells = 0;
          for (const Impact::CellBox & box : costBoxes)
          {
            torch::Tensor       support;
            const torch::Tensor cost =
              costOf(reference, toShift, referenceShare, shiftShare, box, static_cast<int>(l), support);
            sum += (std::get<0>(cost.max(0)) - std::get<0>(cost.min(0))).sum().template item<double>();
            cells += cost.numel() / L;
          }
          const double spread = sum / static_cast<double>(std::max<int64_t>(cells, 1));
          m_LayerSpreads[l] = spread;
          if (spread > 0.0 && std::isfinite(spread))
          {
            total += spread;
            ++count;
          }
        }
        const double common = count > 0 ? total / static_cast<double>(count) : 1.0;
        for (size_t l = 0; l < reference.size(); ++l)
        {
          const double spread = m_LayerSpreads[l];
          weights[l] *= spread > 0.0 && std::isfinite(spread) ? common / spread : 0.0;
        }
      }
      torch::Tensor argmin = torch::empty(coarse, torch::TensorOptions().dtype(torch::kLong).device(device));
      auto          argminPass = [&](const torch::Tensor * disp, double coeff) {
        const torch::Tensor dm = dispMesh.view(dmShape);
        for (const Impact::CellBox & box : costBoxes)
        {
          torch::Tensor       support;
          const torch::Tensor cost = costOf(reference, toShift, referenceShare, shiftShare, box, -1, support);
          torch::Tensor       view = narrowTo(argmin, box.cell0, box.cell1, 0);
          if (disp == nullptr)
          {
            torch::Tensor best = torch::argmin(cost, 0);
            if (masked)
            {
              best = torch::where(support.any(0), best, zeroCandidate);
            }
            view.copy_(best);
            continue;
          }
          // Tiled over the box's first axis so the (L x box) penalty transient stays bounded, as the dense run.
          const torch::Tensor dispBox = narrowTo(disp->squeeze(0), box.cell0, box.cell1, 1);
          for (int64_t i = 0; i < dispBox.size(1); ++i)
          {
            const torch::Tensor dvi = dispBox.narrow(1, i, 1);
            const torch::Tensor penalty = (dm - dvi).pow(2).sum(0);
            const torch::Tensor coupled = cost.select(1, i) + coeff * penalty;
            view.select(0, i).copy_(torch::argmin(coupled, 0));
          }
        }
      };
      argminPass(nullptr, 0.0);
      torch::Tensor disp = smoothBox(gather(argmin));
      for (const double coeff : coeffs)
      {
        argminPass(&disp, coeff);
        disp = smoothBox(gather(argmin));
      }
      return disp;
    };

    torch::Tensor disp = solveCoarse(fixedMaps, movingMaps, fixedShare, movingShare); // forward: fixed -> moving

    // ---- Inverse consistency, as the dense run: the backward problem too, the two fields symmetrized toward
    // mutual inverses in normalized [-1,1] grid coordinates. ----
    if (m_InverseConsistency)
    {
      torch::Tensor dispBack = solveCoarse(movingMaps, fixedMaps, movingShare, fixedShare); // backward

      std::vector<float> scaleVals(Dim);
      for (unsigned int c = 0; c < Dim; ++c)
      {
        scaleVals[c] = (coarse[c] > 1) ? static_cast<float>((coarse[c] - 1) / 2.0) : 1.0f;
      }
      std::vector<int64_t> scaleShape(Dim + 2, 1);
      scaleShape[1] = static_cast<int64_t>(Dim);
      torch::Tensor scaleT = torch::from_blob(scaleVals.data(), { static_cast<int64_t>(Dim) }, torch::kFloat32)
                               .clone()
                               .to(device)
                               .reshape(scaleShape);
      torch::Tensor f1 = (disp / scaleT).flip(1);
      torch::Tensor f2 = (dispBack / scaleT).flip(1);
      torch::Tensor idAffine = torch::eye(Dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      idAffine = torch::cat({ idAffine, torch::zeros({ static_cast<int64_t>(Dim), 1 }, idAffine.options()) }, 1)
                   .unsqueeze(0);
      std::vector<int64_t> gridSize{ 1, 1 };
      gridSize.insert(gridSize.end(), coarse.begin(), coarse.end());
      torch::Tensor        idGrid = torch::affine_grid_generator(idAffine, gridSize, /*align_corners=*/false);
      std::vector<int64_t> toChannelFirst{ 0, idGrid.dim() - 1 };
      for (int64_t dd = 1; dd < idGrid.dim() - 1; ++dd)
      {
        toChannelFirst.push_back(dd);
      }
      torch::Tensor        identity = idGrid.permute(toChannelFirst).contiguous(); // {1, Dim, coarse...}
      std::vector<int64_t> toChannelLast{ 0 };
      for (unsigned int dd = 0; dd < Dim; ++dd)
      {
        toChannelLast.push_back(2 + static_cast<int64_t>(dd));
      }
      toChannelLast.push_back(1);
      const auto icSampleOpts =
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);
      for (int it = 0; it < 15; ++it)
      {
        torch::Tensor f1i = f1.clone();
        torch::Tensor f2i = f2.clone();
        f1 = 0.5 * (f1i - F::grid_sample(f2i, (identity + f1i).permute(toChannelLast), icSampleOpts));
        f2 = 0.5 * (f2i - F::grid_sample(f1i, (identity + f2i).permute(toChannelLast), icSampleOpts));
      }
      disp = f1.flip(1) * scaleT;
    }

    // ---- The field in voxels of the fixed grid on the coarse grid, kept as the padded coarse image the dense field
    // is interpolated from. ----
    std::vector<int64_t> componentShape(Dim + 2, 1);
    componentShape[1] = static_cast<int64_t>(Dim);
    const torch::Tensor cellsToVoxels =
      torch::tensor(cellVoxels, torch::kLong).to(disp.options()).view(componentShape);
    m_CoarseGrid = Impact::BuildCellGrid<Dim>(disp * cellsToVoxels, m_FixedImage.GetPointer());
    m_CoarseGridTransform = nullptr;
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
  os << indent << "RegionSize: " << m_RegionSize << std::endl;
  os << indent << "RegionHalo: " << m_RegionHalo << std::endl;
  os << indent << "CostBoxBytes: " << m_CostBoxBytes << std::endl;
  os << indent << "MovingTransform: " << (m_MovingTransform.IsNull() ? "none" : "set") << std::endl;
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

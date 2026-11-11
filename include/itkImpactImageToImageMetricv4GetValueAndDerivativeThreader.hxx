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

#ifndef itkImpactImageToImageMetricv4GetValueAndDerivativeThreader_hxx
#define itkImpactImageToImageMetricv4GetValueAndDerivativeThreader_hxx

// The online ("Jacobian") path runs the TorchScript model per point and backpropagates
// through it, so it needs the torch model accessors. This header is only compiled in a
// C++ build (guarded out of the castxml/wrapping parse by ITK_MANUAL_INSTANTIATION), so
// pulling in LibTorch here keeps the public metric header torch-free.
#include "itkImpactModelConfigurationDetail.h"
// PatchTensorShape(): the single place that knows a patch reaches the model with its extents
// reversed, shared with the elastix metric so both cut the same patch.
#include "itkImpactOnlineInference.h"
#include "itkImpactBatchBudget.h"

#include <mutex>

namespace itk
{

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  ImpactImageToImageMetricv4GetValueAndDerivativeThreader()
  : m_LossThreadStruct(nullptr)
  , m_ImpactAssociate(nullptr)
{}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  BeforeThreadedExecution()
{
  Superclass::BeforeThreadedExecution();

  /* Cache the associate pointer once to avoid dynamic_cast in tight loops. */
  this->m_ImpactAssociate = dynamic_cast<TImpactMetric *>(this->m_Associate);
  if (this->m_ImpactAssociate == nullptr)
  {
    itkExceptionMacro("Dynamic casting of associate pointer failed.");
  }

  /* Derivative size always comes from the moving transform parameters. */
  const NumberOfParametersType globalDerivativeSize = this->GetCachedNumberOfParameters();

  const ThreadIdType numWorkUnitsUsed = this->GetNumberOfWorkUnitsUsed();

  // The plane seed follows Seed while it is set, so SetSeed() takes effect whenever it is
  // called. A Seed of zero asks for run-to-run variation: draw the clock ONCE and keep it, so
  // the planes still do not move between two evaluations of the same parameters. The clock
  // value is forced odd so it never collides with the "not yet drawn" zero.
  if (const unsigned int seed = this->m_ImpactAssociate->GetSeed(); seed > 0)
  {
    this->m_PlaneSeed = seed;
  }
  else if (this->m_PlaneSeed == 0)
  {
    this->m_PlaneSeed = static_cast<unsigned int>(time(nullptr)) | 1u;
  }

  m_LossThreadStruct = make_unique_for_overwrite<AlignedLossPerThreadStruct[]>(numWorkUnitsUsed);
  this->m_OnlineSamples.assign(numWorkUnitsUsed, {});

  for (ThreadIdType i = 0; i < numWorkUnitsUsed; ++i)
  {
    this->m_LossThreadStruct[i].init(
      this->m_ImpactAssociate->GetDistance(), this->m_ImpactAssociate->GetLayersWeight(), globalDerivativeSize);
  }

  // The feature subsets follow the same seed as the planes above, drawn afresh from it on every
  // evaluation, so two evaluations of the same parameters compare the same channels.
  std::mt19937        generator(this->m_PlaneSeed);
  const torch::Device device(this->m_ImpactAssociate->GetDevice());
  this->m_SubsetsOfFeatures = this->m_ImpactAssociate->m_features_indexes;
  this->m_SubsetsOfFeaturesOnDevice.clear();
  for (size_t comparison = 0; comparison < this->m_SubsetsOfFeatures.size(); ++comparison)
  {
    std::vector<unsigned int> & subset = this->m_SubsetsOfFeatures[comparison];
    const unsigned int          kept = this->m_ImpactAssociate->GetSubsetFeatures()[comparison];
    if (kept < subset.size())
    {
      std::shuffle(subset.begin(), subset.end(), generator);
      subset.resize(kept);
    }
    this->m_SubsetsOfFeaturesOnDevice.push_back(
      torch::tensor(std::vector<int64_t>(subset.begin(), subset.end()), torch::kInt64).to(device));
  }
  // Static mode reads exactly these channels at every point: have the maps lay them side by side.
  auto & internals = *this->m_ImpactAssociate->m_Internals;
  for (size_t comparison = 0; comparison < internals.m_FixedFeaturesMaps.size(); ++comparison)
  {
    internals.m_FixedFeaturesMaps[comparison].m_FeaturesMapInterpolator->SetActiveChannels(
      this->m_SubsetsOfFeatures[comparison]);
    internals.m_MovingFeaturesMaps[comparison].m_FeaturesMapInterpolator->SetActiveChannels(
      this->m_SubsetsOfFeatures[comparison]);
  }
  // The fixed features kept over a sampled point set (Internals::m_FixedFeatures), sized here, on
  // one thread, and dropped when what they depend on has moved.
  if (this->m_ImpactAssociate->GetMode() == "Jacobian" && this->m_ImpactAssociate->GetUseSampledPointSet())
  {
    const auto points = static_cast<int64_t>(this->m_ImpactAssociate->GetNumberOfDomainPoints());
    const std::pair<unsigned int, ModifiedTimeType> stamp{ this->m_PlaneSeed,
                                                           this->m_ImpactAssociate->GetFixedTransform()->GetMTime() };
    if (internals.m_FixedFeaturesKnown.size() != static_cast<size_t>(points) || internals.m_FixedFeaturesStamp != stamp)
    {
      internals.m_FixedFeaturesKnown.assign(static_cast<size_t>(points), 0);
      internals.m_FixedFeatures.clear();
      for (const torch::Tensor & subset : this->m_SubsetsOfFeaturesOnDevice)
      {
        internals.m_FixedFeatures.push_back(torch::empty({ points, subset.numel() }, torch::kFloat32));
      }
      internals.m_FixedFeaturesStamp = stamp;
    }
  }
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  AfterThreadedExecution()
{
  // The reductions below, and the small operations of the online passes, are not worth a team.
  const Impact::OneOpenMPThread oneThread(this->GetNumberOfWorkUnitsUsed() > 1);

  // Let the base reduce the valid-point count (and per-point derivative contributions,
  // if any). It also sets m_Value from the per-point measures, which we leave at zero
  // and override below with the IMPACT loss value.
  Superclass::AfterThreadedExecution();

  if (this->m_ImpactAssociate->GetMode() == "Jacobian")
  {
    this->EvaluateOnline();
  }

  // Reduce the per-work-unit loss accumulators into work unit 0.
  const ThreadIdType numWorkUnitsUsed = this->GetNumberOfWorkUnitsUsed();
  for (ThreadIdType i = 1; i < numWorkUnitsUsed; ++i)
  {
    this->m_LossThreadStruct[0] += this->m_LossThreadStruct[i];
  }

  if (this->m_ImpactAssociate->GetNumberOfValidPoints() > 0)
  {
    itk::Impact::LossNormalization * normalization =
      this->m_ImpactAssociate->m_NormalizeLosses ? &this->m_ImpactAssociate->m_LossNormalization : nullptr;
    this->m_ImpactAssociate->m_Value = this->m_LossThreadStruct[0].GetValue(normalization);
    if (this->GetComputeDerivative())
    {
      // The losses accumulate the true gradient d(value)/dp. ITKv4 optimizers *add* the
      // metric derivative to the parameters (UpdateTransformParameters), so they expect
      // the descent direction -d(value)/dp. This matches the built-in v4 metrics: e.g.
      // MeanSquares accumulates +2*(fixed-moving)*dMoving/dp, which is -d(MSE)/dp. Hence
      // we write the negated IMPACT loss gradient here.
      const DerivativeType gradient = this->m_LossThreadStruct[0].GetDerivative(normalization);
      DerivativeType &     result = *(this->m_ImpactAssociate->m_DerivativeResult);
      for (SizeValueType parameter = 0; parameter < gradient.GetSize(); ++parameter)
      {
        result[parameter] = -gradient[parameter];
      }
    }
  }
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  ThreadedExecution(const DomainType & domain, const ThreadIdType threadId)
{
  // Gather this work unit's points first and evaluate them in batches, which is the shape the
  // elastix metric has: there the sampler hands its whole point set over in a single call.
  // Online, walking point by point would cost one forward and one backward pass each; Static, a
  // few small tensor operations each, and the framework's walk also computed an image gradient
  // per point that IMPACT never reads.
  const Impact::OneOpenMPThread oneThread(this->GetNumberOfWorkUnitsUsed() > 1);
  LossPerThreadStruct &         loss = this->m_LossThreadStruct[threadId];
  const bool                    online = this->m_ImpactAssociate->GetMode() == "Jacobian";

  // ProcessVirtualPoint() would have kept these counts and we bypassed it: the first is what
  // the base metric reduces into GetNumberOfValidPoints(), the second what normalizes the loss.
  auto count = [&](const std::vector<Sample> & samples) {
    this->m_GetValueAndDerivativePerThreadVariables[threadId].NumberOfValidPoints += samples.size();
    loss.m_numberOfPixelsCounted += samples.size();
  };
  // Static, a batch holds a [batch, channels, parameters] Jacobian: about 4 MB per work unit, so that a dense
  // domain under a B-spline transform stays bounded, and the points are evaluated as they are gathered, a batch
  // at a time. Online the work unit gathers them all, for EvaluateOnline().
  size_t channels = 1;
  for (const std::vector<unsigned int> & subset : this->m_SubsetsOfFeatures)
  {
    channels = std::max(channels, subset.size());
  }
  const size_t parameters = this->GetComputeDerivative() ? this->GetCachedNumberOfLocalParameters() : 1;
  const size_t batch = std::max<size_t>(1, (size_t{ 1 } << 20) / std::max<size_t>(1, channels * parameters));
  auto         evaluate = [&](std::vector<Sample> & samples) {
    count(samples);
    this->EvaluateStaticBatch(samples.begin(), samples.end(), threadId, loss);
    samples.clear();
  };
  std::vector<Sample> samples =
    this->CollectSamples(domain, online ? std::numeric_limits<size_t>::max() : batch, evaluate);
  if (online)
  {
    count(samples);
    this->m_OnlineSamples[threadId] = std::move(samples);
  }
  else if (!samples.empty())
  {
    evaluate(samples);
  }

  this->m_Associate->FinalizeThread(threadId);
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
bool
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  ResolveSample(const VirtualIndexType & virtualIndex, const VirtualPointType & virtualPoint, Sample & sample) const
{
  // The intensities these return are unused online -- the patch is sampled around the mapped
  // points -- but the calls are also what tests the point against each image's buffer.
  FixedImagePixelType  fixedValue;
  MovingImagePixelType movingValue;
  if (!this->m_ImpactAssociate->TransformAndEvaluateFixedPoint(virtualPoint, sample.m_FixedPoint, fixedValue) ||
      !this->m_ImpactAssociate->TransformAndEvaluateMovingPoint(virtualPoint, sample.m_MovingPoint, movingValue))
  {
    return false;
  }
  sample.m_VirtualIndex = virtualIndex;
  sample.m_VirtualPoint = virtualPoint;
  return true;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
auto
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  CollectSamples(const ImageRegion<ImageToImageMetricv4Type::VirtualImageDimension> & region,
                 size_t                                                               capacity,
                 const std::function<void(std::vector<Sample> &)> & whenFull) const -> std::vector<Sample>
{
  const typename VirtualImageType::ConstPointer virtualImage = this->m_ImpactAssociate->GetVirtualImage();

  std::vector<Sample> samples;
  samples.reserve(std::min<size_t>(region.GetNumberOfPixels(), capacity));
  Sample           sample;
  VirtualPointType virtualPoint;
  for (ImageRegionConstIteratorWithIndex<VirtualImageType> it(virtualImage, region); !it.IsAtEnd(); ++it)
  {
    virtualImage->TransformIndexToPhysicalPoint(it.GetIndex(), virtualPoint);
    if (this->ResolveSample(it.GetIndex(), virtualPoint, sample))
    {
      samples.push_back(sample);
      if (samples.size() == capacity)
      {
        whenFull(samples);
      }
    }
  }
  return samples;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
auto
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  CollectSamples(const ThreadedIndexedContainerPartitioner::DomainType & indexRange,
                 size_t                                                  capacity,
                 const std::function<void(std::vector<Sample> &)> &      whenFull) const -> std::vector<Sample>
{
  using PointIdentifierType = typename ImageToImageMetricv4Type::VirtualPointSetType::MeshTraits::PointIdentifier;
  const typename VirtualImageType::ConstPointer virtualImage = this->m_ImpactAssociate->GetVirtualImage();
  const auto                                    sampledPointSet = this->m_ImpactAssociate->GetVirtualSampledPointSet();

  const auto begin = static_cast<PointIdentifierType>(indexRange[0]);
  const auto end = static_cast<PointIdentifierType>(indexRange[1]);

  std::vector<Sample> samples;
  samples.reserve(std::min(static_cast<size_t>(end - begin + 1), capacity));
  Sample sample;
  for (PointIdentifierType i = begin; i <= end; ++i)
  {
    const VirtualPointType virtualPoint = sampledPointSet->GetPoint(i);
    if (this->ResolveSample(virtualImage->TransformPhysicalPointToIndex(virtualPoint), virtualPoint, sample))
    {
      sample.m_Id = static_cast<SizeValueType>(i);
      samples.push_back(sample);
      if (samples.size() == capacity)
      {
        whenFull(samples);
      }
    }
  }
  return samples;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
auto
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  PatchPlane(const VirtualIndexType & virtualIndex, size_t modelIndex, unsigned int modelDimension) const
  -> PatchPlaneType
{
  constexpr unsigned int fixedDimension = ImageToImageMetricv4Type::FixedImageDimension;

  PatchPlaneType plane;
  plane.SetIdentity();
  // A model of the image's own dimension spans every axis, so each patch axis is one image
  // axis and the plane is the identity. Only a plane in a volume is rotated: that is the case
  // a lower-dimensional model actually occurs in, and the one elastix rotates.
  if constexpr (fixedDimension == 3)
  {
    if (modelDimension == 2)
    {
      // Taking the same plane at every point would only ever show the network one orientation of
      // the anatomy, so it is drawn afresh for each sampled point and the metric integrates over
      // orientations as the sampler walks the image. This is what the elastix metric does through
      // Impact::GetPatchIndex. The plane is a function of the point (see Impact::PatchPlaneRotation).
      plane = Impact::PatchPlaneRotation(this->m_PlaneSeed, modelIndex, virtualIndex);
    }
  }
  return plane;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  SamplePatch(const Sample &  sample,
              size_t          modelIndex,
              int64_t         row,
              torch::Tensor & fixedPatches,
              torch::Tensor & movingPatches,
              torch::Tensor & movingGradients,
              bool            computeDerivative) const
{
  constexpr unsigned int fixedDimension = ImageToImageMetricv4Type::FixedImageDimension;
  constexpr unsigned int movingDimension = ImageToImageMetricv4Type::MovingImageDimension;
  // Central-difference step (physical units) linearizing the moving interpolator around each
  // patch voxel. Using the SAME interpolator that produces the value keeps the analytic
  // derivative consistent with finite differences of the value.
  constexpr double kGradientStep = 1e-3;

  const ImpactModelConfiguration & config = this->m_ImpactAssociate->GetFixedModelsConfiguration()[modelIndex];
  const std::vector<int64_t> &     patchSize = config.GetPatchSize();
  const std::vector<float> &       voxelSize = config.GetVoxelSize();
  const unsigned int               dim = config.GetDimension();

  // A patch step of one voxel along image axis d is the d-th column of the image's direction
  // matrix, scaled by the requested voxel size -- not world axis d. This is the same
  // index-to-physical map the Static path applies (through
  // TransformContinuousIndexToPhysicalPoint in ImageToTensorFilter), so a model of the image's
  // own dimension gets the same neighborhood in both modes; adding the offsets component-wise
  // in world coordinates is correct only for an identity direction and silently rotates the
  // patch otherwise.
  const auto &         fixedDirection = this->m_ImpactAssociate->GetFixedImage()->GetDirection();
  const auto &         movingDirection = this->m_ImpactAssociate->GetMovingImage()->GetDirection();
  const PatchPlaneType plane = this->PatchPlane(sample.m_VirtualIndex, modelIndex, dim);

  float * const fixedRow =
    fixedPatches.defined() ? fixedPatches.template data_ptr<float>() + row * fixedPatches.size(1) : nullptr;
  auto movingAccessor = movingPatches.accessor<float, 2>();
  auto gradientAccessor = movingGradients.accessor<float, 3>();

  for (int64_t flat = 0; flat < movingPatches.size(1); ++flat)
  {
    // Decode the multi-index with image axis 0 varying FASTEST, so that reshaping the row to
    // the reversed extents reproduces the Static path's tensor layout.
    int64_t              remainder = flat;
    FixedImagePointType  fixedPoint = sample.m_FixedPoint;
    MovingImagePointType movingPoint = sample.m_MovingPoint;
    for (unsigned int d = 0; d < dim; ++d)
    {
      const int64_t index = remainder % patchSize[d];
      remainder /= patchSize[d];
      // The centre is the voxel centerFeatures() reads back, patchSize[d] / 2, not the
      // geometric middle: those agree for an odd extent and differ by half a voxel for an even
      // one, where (n - 1) / 2 would hand the loss a feature half a voxel off its sample point.
      // The shared online inference spans its patch the same way.
      const double offset = (static_cast<double>(index) - static_cast<double>(patchSize[d] / 2)) * voxelSize[d];
      // Patch axis d points along the plane's d-th COLUMN, expressed in the image's own frame:
      // elastix spans its patch with matrix * (x, y, 0), whose basis vectors are the columns,
      // and the rows would span the transposed rotation's plane instead. With the identity
      // plane this is simply direction column d.
      for (unsigned int j = 0; j < fixedDimension; ++j)
      {
        const double component = plane[j][d] * offset;
        if (component == 0.0)
        {
          continue;
        }
        for (unsigned int r = 0; r < fixedDimension; ++r)
        {
          fixedPoint[r] += fixedDirection[r][j] * component;
        }
        if (j < movingDimension)
        {
          for (unsigned int r = 0; r < movingDimension; ++r)
          {
            movingPoint[r] += movingDirection[r][j] * component;
          }
        }
      }
    }

    if (fixedRow != nullptr && this->m_ImpactAssociate->m_FixedInterpolator->IsInsideBuffer(fixedPoint))
    {
      fixedRow[flat] = static_cast<float>(this->m_ImpactAssociate->m_FixedInterpolator->Evaluate(fixedPoint));
    }
    if (!this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(movingPoint))
    {
      continue;
    }
    const double movingValue =
      static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(movingPoint));
    movingAccessor[row][flat] = static_cast<float>(movingValue);
    if (!computeDerivative)
    {
      continue;
    }
    for (unsigned int d = 0; d < movingDimension; ++d)
    {
      MovingImagePointType plusPoint = movingPoint;
      MovingImagePointType minusPoint = movingPoint;
      plusPoint[d] += kGradientStep;
      minusPoint[d] -= kGradientStep;
      const double valuePlus =
        this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(plusPoint)
          ? static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(plusPoint))
          : movingValue;
      const double valueMinus =
        this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(minusPoint)
          ? static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(minusPoint))
          : movingValue;
      gradientAccessor[row][flat][d] = static_cast<float>((valuePlus - valueMinus) / (2.0 * kGradientStep));
    }
  }
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
auto
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  SampleBatch(typename std::vector<Sample>::const_iterator first,
              typename std::vector<Sample>::const_iterator last,
              bool                                         parallel,
              bool                                         derivative,
              bool                                         fixed) const -> Impact::OnlineBatch
{
  using JacobianType = typename TImageToImageMetric::JacobianType;
  constexpr unsigned int movingDimension = ImageToImageMetricv4Type::MovingImageDimension;

  const auto   size = static_cast<int64_t>(last - first);
  const auto & configs = this->m_ImpactAssociate->GetFixedModelsConfiguration();

  Impact::OnlineBatch batch;
  for (const ImpactModelConfiguration & config : configs)
  {
    const std::vector<int64_t> & patchSize = config.GetPatchSize();
    const int64_t voxels = std::accumulate(patchSize.begin(), patchSize.end(), int64_t{ 1 }, std::multiplies<>());
    batch.fixedPatches.push_back(fixed ? torch::zeros({ size, voxels }, torch::kFloat32) : torch::Tensor());
    batch.movingPatches.push_back(torch::zeros({ size, voxels }, torch::kFloat32));
    batch.movingGradients.push_back(torch::zeros({ size, voxels, movingDimension }, torch::kFloat32));
  }

  // d(moving coordinate)/d(parameter) at every point, and the global parameter index of each of
  // its columns.
  const auto parameters = static_cast<int64_t>(this->GetCachedNumberOfLocalParameters());
  float *    jacobianOut = nullptr;
  int64_t *  indicesOut = nullptr;
  if (derivative)
  {
    batch.transformJacobian = torch::empty({ size, movingDimension, parameters }, torch::kFloat32);
    batch.nonZeroJacobianIndices = torch::empty({ size, parameters }, torch::kInt64);
    jacobianOut = batch.transformJacobian.template data_ptr<float>();
    indicesOut = batch.nonZeroJacobianIndices.template data_ptr<int64_t>();
  }

  auto sampleRows = [&](int64_t begin, int64_t end) {
    JacobianType jacobian(movingDimension, parameters);
    JacobianType jacobianPositional(movingDimension, movingDimension);
    for (int64_t row = begin; row < end; ++row)
    {
      const Sample & sample = *(first + row);
      for (size_t i = 0; i < configs.size(); ++i)
      {
        this->SamplePatch(
          sample, i, row, batch.fixedPatches[i], batch.movingPatches[i], batch.movingGradients[i], derivative);
      }
      if (derivative)
      {
        this->FillTransformJacobian(sample,
                                    jacobian,
                                    jacobianPositional,
                                    jacobianOut + row * movingDimension * parameters,
                                    indicesOut + row * parameters);
      }
    }
  };
  if (!parallel)
  {
    sampleRows(0, size);
    return batch;
  }
  // The rows are independent: spread them over the threads, idle between two evaluations.
  MultiThreaderBase * const threader = this->GetMultiThreader();
  const auto                parts = std::min<int64_t>(size, threader->GetNumberOfWorkUnits());
  threader->ParallelizeArray(
    0, parts, [&](SizeValueType part) { sampleRows(size * part / parts, size * (part + 1) / parts); }, nullptr);
  return batch;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  EvaluateOnline()
{
  std::vector<Sample> samples;
  for (const std::vector<Sample> & unit : this->m_OnlineSamples)
  {
    samples.insert(samples.end(), unit.begin(), unit.end());
  }
  this->m_OnlineSamples.clear();

  // The points as Impact::EvaluateOnline asks for them. Over a sampled point set the fixed features are kept from one
  // evaluation to the next (Internals::m_FixedFeatures): a batch whose points all have theirs cuts no fixed patch.
  class Host : public Impact::OnlineHost
  {
  public:
    Host(const Self & threader, const std::vector<typename Self::Sample> & samples)
      : m_Threader(threader)
      , m_Samples(samples)
      , m_Internals(*threader.m_ImpactAssociate->m_Internals)
      , m_Keep(threader.m_ImpactAssociate->GetUseSampledPointSet())
    {}
    Impact::OnlineBatch
    Sample(int64_t begin, int64_t end, bool parallel, bool derivative, bool fixed) const override
    {
      const auto first = m_Samples.begin() + begin, last = m_Samples.begin() + end;
      fixed = fixed && !(m_Keep && std::all_of(first, last, [&](const auto & s) {
                           return m_Internals.m_FixedFeaturesKnown[s.m_Id] != 0;
                         }));
      return m_Threader.SampleBatch(first, last, parallel, derivative, fixed);
    }
    void
    Parallelize(int64_t parts, const std::function<void(int64_t)> & body) const override
    {
      m_Threader.GetMultiThreader()->ParallelizeArray(
        0, parts, [&](SizeValueType part) { body(static_cast<int64_t>(part)); }, nullptr);
    }
    std::vector<torch::Tensor>
    KeptFixedFeatures(int64_t begin, int64_t end) const override
    {
      std::vector<torch::Tensor> fixed;
      for (const torch::Tensor & kept : m_Internals.m_FixedFeatures)
      {
        fixed.push_back(kept.index_select(0, Ids(begin, end)));
      }
      return fixed;
    }
    void
    KeepFixedFeatures(int64_t begin, int64_t end, const std::vector<torch::Tensor> & fixed) const override
    {
      if (!m_Keep)
      {
        return;
      }
      const torch::Tensor ids = Ids(begin, end);
      for (size_t c = 0; c < fixed.size(); ++c)
      {
        m_Internals.m_FixedFeatures[c].index_copy_(0, ids, fixed[c]);
      }
      for (auto it = m_Samples.begin() + begin; it != m_Samples.begin() + end; ++it)
      {
        m_Internals.m_FixedFeaturesKnown[it->m_Id] = 1;
      }
    }

  private:
    torch::Tensor
    Ids(int64_t begin, int64_t end) const
    {
      torch::Tensor ids = torch::empty({ end - begin }, torch::kInt64);
      std::transform(m_Samples.begin() + begin, m_Samples.begin() + end, ids.data_ptr<int64_t>(), [](const auto & s) {
        return static_cast<int64_t>(s.m_Id);
      });
      return ids;
    }
    const Self &                               m_Threader;
    const std::vector<typename Self::Sample> & m_Samples;
    typename TImpactMetric::Internals &        m_Internals;
    const bool                                 m_Keep;
  };

  const Impact::OnlineSettings settings{ this->m_ImpactAssociate->GetFixedModelsConfiguration(),
                                         this->m_ImpactAssociate->GetMovingModelsConfiguration(),
                                         this->m_SubsetsOfFeaturesOnDevice,
                                         torch::Device(this->m_ImpactAssociate->GetDevice()),
                                         static_cast<int64_t>(this->m_ImpactAssociate->GetBatchSize()),
                                         static_cast<int64_t>(this->GetNumberOfWorkUnitsUsed()),
                                         &this->m_ImpactAssociate->m_Internals->m_FixedImageMetadata,
                                         &this->m_ImpactAssociate->m_Internals->m_MovingImageMetadata };
  Impact::EvaluateOnline(settings,
                         Host(*this, samples),
                         static_cast<int64_t>(samples.size()),
                         this->GetComputeDerivative(),
                         [&](int64_t part) -> std::vector<std::unique_ptr<Impact::Loss>> & {
                           return this->m_LossThreadStruct[part].m_losses;
                         });
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  FillTransformJacobian(const Sample &                               sample,
                        typename TImageToImageMetric::JacobianType & jacobian,
                        typename TImageToImageMetric::JacobianType & jacobianPositional,
                        float *                                      jacobianOut,
                        int64_t *                                    indicesOut) const
{
  using MovingTransformType = typename TImageToImageMetric::MovingTransformType;
  const auto transform = this->m_ImpactAssociate->GetMovingTransform();
  const auto parameters = static_cast<int64_t>(this->GetCachedNumberOfLocalParameters());
  transform->ComputeJacobianWithRespectToParametersCachedTemporaries(
    sample.m_VirtualPoint, jacobian, jacobianPositional);
  for (unsigned int d = 0; d < ImageToImageMetricv4Type::MovingImageDimension; ++d)
  {
    for (int64_t parameter = 0; parameter < parameters; ++parameter)
    {
      *jacobianOut++ = static_cast<float>(jacobian(d, parameter));
    }
  }
  // A global transform (affine, rigid) has offset 0; a local-support displacement field only
  // affects the parameter block of the point's own virtual voxel.
  const int64_t offset =
    transform->GetTransformCategory() == MovingTransformType::TransformCategoryEnum::DisplacementField
      ? static_cast<int64_t>(
          this->m_ImpactAssociate->ComputeParameterOffsetFromVirtualIndex(sample.m_VirtualIndex, parameters))
      : 0;
  for (int64_t parameter = 0; parameter < parameters; ++parameter)
  {
    *indicesOut++ = offset + parameter;
  }
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  EvaluateStaticBatch(typename std::vector<Sample>::const_iterator first,
                      typename std::vector<Sample>::const_iterator last,
                      const ThreadIdType                           threadId,
                      LossPerThreadStruct &                        loss) const
{
  // Interpolate the per-layer feature vectors at the mapped fixed and moving points and
  // accumulate the per-layer losses. When a derivative is requested, chain the moving
  // features' spatial gradient with the moving-transform Jacobian. Value and derivative are
  // reduced and normalized in AfterThreadedExecution.
  const auto             count = static_cast<int64_t>(last - first);
  const bool             computeDerivative = this->GetComputeDerivative();
  constexpr unsigned int movingDimension = ImageToImageMetricv4Type::MovingImageDimension;
  const auto &           internals = *this->m_ImpactAssociate->m_Internals;

  // A map rebuilt through T_k holds the features of M(T_k(p)): read it at x + (T(x) - T_k(x)),
  // which is x at the moment of the refresh and moves with the parameters exactly as T(x) does.
  // The map built at Initialize() holds the features of M itself and is read at T(x).
  std::vector<MovingImagePointType> movingReadPoints;
  movingReadPoints.reserve(count);
  for (auto sample = first; sample != last; ++sample)
  {
    MovingImagePointType movingReadPoint = sample->m_MovingPoint;
    if (internals.m_RefreshTransform.IsNotNull())
    {
      const auto anchored = internals.m_RefreshTransform->TransformPoint(sample->m_VirtualPoint);
      for (unsigned int d = 0; d < movingDimension; ++d)
      {
        movingReadPoint[d] = sample->m_VirtualPoint[d] + (sample->m_MovingPoint[d] - anchored[d]);
      }
    }
    movingReadPoints.push_back(movingReadPoint);
  }

  torch::Tensor transformJacobian;      // [B, MovingDim, P]
  torch::Tensor nonZeroJacobianIndices; // [B, P]
  if (computeDerivative)
  {
    using JacobianType = typename TImageToImageMetric::JacobianType;
    JacobianType & jacobian = this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobian;
    JacobianType & jacobianPositional =
      this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobianPositional;
    const auto parameters = static_cast<int64_t>(this->GetCachedNumberOfLocalParameters());

    transformJacobian = torch::empty({ count, movingDimension, parameters }, torch::kFloat32);
    nonZeroJacobianIndices = torch::empty({ count, parameters }, torch::kInt64);
    float *   jacobianOut = transformJacobian.data_ptr<float>();
    int64_t * indicesOut = nonZeroJacobianIndices.data_ptr<int64_t>();
    for (auto sample = first; sample != last; ++sample)
    {
      this->FillTransformJacobian(*sample, jacobian, jacobianPositional, jacobianOut, indicesOut);
      jacobianOut += movingDimension * parameters;
      indicesOut += parameters;
    }
  }

  for (size_t i = 0; i < internals.m_FixedFeaturesMaps.size(); ++i)
  {
    const std::vector<unsigned int> & subset = this->m_SubsetsOfFeatures[i];
    const auto                        channels = static_cast<int64_t>(subset.size());
    const auto &                      fixedMap = *internals.m_FixedFeaturesMaps[i].m_FeaturesMapInterpolator;
    const auto &                      movingMap = *internals.m_MovingFeaturesMaps[i].m_FeaturesMapInterpolator;

    torch::Tensor fixedFeatures = torch::empty({ count, channels }, torch::kFloat32);
    torch::Tensor movingFeatures = torch::empty({ count, channels }, torch::kFloat32);
    torch::Tensor movingDerivatives; // d(feature)/d(moving coordinate) : [B, C, MovingDim]
    float *       fixedOut = fixedFeatures.data_ptr<float>();
    float *       movingOut = movingFeatures.data_ptr<float>();
    float *       derivativeOut = nullptr;
    if (computeDerivative)
    {
      movingDerivatives = torch::empty({ count, channels, movingDimension }, torch::kFloat32);
      derivativeOut = movingDerivatives.data_ptr<float>();
    }
    int64_t row = 0;
    for (auto sample = first; sample != last; ++sample, ++row)
    {
      fixedMap.Evaluate(sample->m_FixedPoint, subset, fixedOut + row * channels);
      if (computeDerivative)
      {
        movingMap.EvaluateValueAndDerivative(
          movingReadPoints[row], subset, movingOut + row * channels, derivativeOut + row * channels * movingDimension);
      }
      else
      {
        movingMap.Evaluate(movingReadPoints[row], subset, movingOut + row * channels);
      }
    }

    if (computeDerivative)
    {
      // chained with d(moving coordinate)/d(parameter) : [B, C, P]
      torch::Tensor featureParameterJacobian = torch::bmm(movingDerivatives, transformJacobian);
      loss.m_losses[i]->updateValueAndDerivativeInStaticMode(
        fixedFeatures, movingFeatures, featureParameterJacobian, nonZeroJacobianIndices);
    }
    else
    {
      loss.m_losses[i]->updateValue(fixedFeatures, movingFeatures);
    }
  }
}

} // end namespace itk

#endif

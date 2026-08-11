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
#include "itkModelConfigurationDetail.h"
// PatchTensorShape(): the single place that knows a patch reaches the model with its extents
// reversed, shared with the elastix metric so both cut the same patch.
#include "itkImpactOnlineInference.h"

namespace itk
{

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<
  TDomainPartitioner,
  TImageToImageMetric,
  TImpactMetric>::ImpactImageToImageMetricv4GetValueAndDerivativeThreader()
  : m_LossThreadStruct(nullptr)
  , m_ImpactAssociate(nullptr)
{}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner,
                                                        TImageToImageMetric,
                                                        TImpactMetric>::BeforeThreadedExecution()
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

  for (ThreadIdType i = 0; i < numWorkUnitsUsed; ++i)
  {
    this->m_LossThreadStruct[i].init(this->m_ImpactAssociate->GetDistance(),
                                     this->m_ImpactAssociate->GetLayersWeight(),
                                     this->m_ImpactAssociate->GetSeed(),
                                     globalDerivativeSize);
  }
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner,
                                                        TImageToImageMetric,
                                                        TImpactMetric>::AfterThreadedExecution()
{
  // Let the base reduce the valid-point count (and per-point derivative contributions,
  // if any). It also sets m_Value from the per-point measures, which we leave at zero
  // and override below with the IMPACT loss value.
  Superclass::AfterThreadedExecution();

  // Reduce the per-work-unit loss accumulators into work unit 0.
  const ThreadIdType numWorkUnitsUsed = this->GetNumberOfWorkUnitsUsed();
  for (ThreadIdType i = 1; i < numWorkUnitsUsed; ++i)
  {
    this->m_LossThreadStruct[0] += this->m_LossThreadStruct[i];
  }

  if (this->m_ImpactAssociate->GetNumberOfValidPoints() > 0)
  {
    this->m_ImpactAssociate->m_Value = this->m_LossThreadStruct[0].GetValue();
    if (this->GetComputeDerivative())
    {
      // The losses accumulate the true gradient d(value)/dp. ITKv4 optimizers *add* the
      // metric derivative to the parameters (UpdateTransformParameters), so they expect
      // the descent direction -d(value)/dp. This matches the built-in v4 metrics: e.g.
      // MeanSquares accumulates +2*(fixed-moving)*dMoving/dp, which is -d(MSE)/dp. Hence
      // we write the negated IMPACT loss gradient here.
      const DerivativeType gradient = this->m_LossThreadStruct[0].GetDerivative();
      DerivativeType &     result = *(this->m_ImpactAssociate->m_DerivativeResult);
      for (SizeValueType parameter = 0; parameter < gradient.GetSize(); ++parameter)
      {
        result[parameter] = -gradient[parameter];
      }
    }
  }
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
std::vector<unsigned int>
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<
  TDomainPartitioner,
  TImageToImageMetric,
  TImpactMetric>::GetSubsetOfFeatures(const std::vector<unsigned int> & features_index,
                                      std::mt19937 &                    randomGenerator,
                                      int                               n) const
{
  if (features_index.size() == static_cast<size_t>(n))
    return features_index;

  std::vector<unsigned int> shuffled = features_index;
  std::shuffle(shuffled.begin(), shuffled.end(), randomGenerator);
  shuffled.resize(n);
  return shuffled;
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  ThreadedExecution(const DomainType & domain, const ThreadIdType threadId)
{
  // Static mode reads a precomputed feature map, so a point costs one interpolation and the
  // framework's own walk -- which ends in ProcessPoint() -- is the right granularity.
  if (this->m_ImpactAssociate->GetMode() != "Jacobian")
  {
    Superclass::ThreadedExecution(domain, threadId);
    return;
  }

  // Online mode instead runs the network on a patch cut around every point, so walking point
  // by point costs one forward and one backward pass each. Gather this work unit's points
  // first and push them through the models in batches, which is the shape the elastix metric
  // has: there the sampler hands its whole point set over in a single call.
  const std::vector<Sample> samples = this->CollectSamples(domain);
  LossPerThreadStruct &     loss = this->m_LossThreadStruct[threadId];

  const auto batchSize = static_cast<std::ptrdiff_t>(std::max(1u, this->m_ImpactAssociate->GetBatchSize()));
  for (auto first = samples.begin(); first != samples.end();)
  {
    const auto last = first + std::min(batchSize, samples.end() - first);
    this->EvaluateJacobianBatch(first, last, threadId, loss);
    first = last;
  }

  // ProcessVirtualPoint() would have kept these counts and we bypassed it: the first is what
  // the base metric reduces into GetNumberOfValidPoints(), the second what normalizes the loss.
  this->m_GetValueAndDerivativePerThreadVariables[threadId].NumberOfValidPoints += samples.size();
  loss.m_numberOfPixelsCounted += samples.size();

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
  CollectSamples(const ImageRegion<ImageToImageMetricv4Type::VirtualImageDimension> & region) const
  -> std::vector<Sample>
{
  const typename VirtualImageType::ConstPointer virtualImage = this->m_ImpactAssociate->GetVirtualImage();

  std::vector<Sample> samples;
  samples.reserve(region.GetNumberOfPixels());
  Sample           sample;
  VirtualPointType virtualPoint;
  for (ImageRegionConstIteratorWithIndex<VirtualImageType> it(virtualImage, region); !it.IsAtEnd(); ++it)
  {
    virtualImage->TransformIndexToPhysicalPoint(it.GetIndex(), virtualPoint);
    if (this->ResolveSample(it.GetIndex(), virtualPoint, sample))
    {
      samples.push_back(sample);
    }
  }
  return samples;
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
auto
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  CollectSamples(const ThreadedIndexedContainerPartitioner::DomainType & indexRange) const -> std::vector<Sample>
{
  using PointIdentifierType = typename ImageToImageMetricv4Type::VirtualPointSetType::MeshTraits::PointIdentifier;
  const typename VirtualImageType::ConstPointer virtualImage = this->m_ImpactAssociate->GetVirtualImage();
  const auto                                    sampledPointSet = this->m_ImpactAssociate->GetVirtualSampledPointSet();

  const auto begin = static_cast<PointIdentifierType>(indexRange[0]);
  const auto end = static_cast<PointIdentifierType>(indexRange[1]);

  std::vector<Sample> samples;
  samples.reserve(static_cast<size_t>(end - begin + 1));
  Sample sample;
  for (PointIdentifierType i = begin; i <= end; ++i)
  {
    const VirtualPointType virtualPoint = sampledPointSet->GetPoint(i);
    if (this->ResolveSample(virtualImage->TransformPhysicalPointToIndex(virtualPoint), virtualPoint, sample))
    {
      samples.push_back(sample);
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
  if (modelDimension != 2 || fixedDimension != 3)
  {
    return plane;
  }

  // Taking the same plane at every point would only ever show the network one orientation of
  // the anatomy, so it is drawn afresh for each sampled point and the metric integrates over
  // orientations as the sampler walks the image. This is what the elastix metric does through
  // Impact::GetPatchIndex.
  //
  // The generator is seeded from the sampled point itself rather than from the work unit's
  // running generator, so the plane depends on WHERE the point is and not on how many points
  // came before it. That is what keeps the metric a function of its parameters: a transform
  // perturbed by a finite-difference step pushes a few points out of the moving buffer, which
  // drops them from the sequence, and a running generator would then hand every later point a
  // different plane -- the two arms of the difference would sample different anatomy and the
  // derivative would mean nothing. It also makes the value independent of the work-unit
  // partition. Mixing the model index in keeps two models configured at the same point from
  // being handed the identical plane, as elastix draws one per (point, model).
  std::uint_fast32_t pointSeed = static_cast<std::uint_fast32_t>(this->m_PlaneSeed) + 0x9e3779b9u * (modelIndex + 1u);
  for (unsigned int d = 0; d < fixedDimension; ++d)
  {
    pointSeed = pointSeed * 2654435761u + static_cast<std::uint_fast32_t>(virtualIndex[d]);
  }
  std::mt19937                           pointGenerator(pointSeed);
  std::uniform_real_distribution<double> angles(0.0, 2.0 * itk::Math::pi);
  const double                           a = angles(pointGenerator);
  const double                           b = angles(pointGenerator);
  const double                           c = angles(pointGenerator);
  const double                           ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b);
  const double                           cc = std::cos(c), sc = std::sin(c);
  // Rz * Ry * Rx, as elastix composes them.
  plane[0][0] = cc * cb;
  plane[0][1] = cc * sb * sa - sc * ca;
  plane[0][2] = cc * sb * ca + sc * sa;
  plane[1][0] = sc * cb;
  plane[1][1] = sc * sb * sa + cc * ca;
  plane[1][2] = sc * sb * ca - cc * sa;
  plane[2][0] = -sb;
  plane[2][1] = cb * sa;
  plane[2][2] = cb * ca;
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

  const ModelConfiguration &   config = this->m_ImpactAssociate->GetFixedModelsConfiguration()[modelIndex];
  const std::vector<int64_t> & patchSize = config.GetPatchSize();
  const std::vector<float> &   voxelSize = config.GetVoxelSize();
  const unsigned int           dim = config.GetDimension();

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

  auto fixedAccessor = fixedPatches.accessor<float, 2>();
  auto movingAccessor = movingPatches.accessor<float, 2>();
  auto gradientAccessor = movingGradients.accessor<float, 3>();

  for (int64_t flat = 0; flat < fixedPatches.size(1); ++flat)
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
      const double offset = (static_cast<double>(index) - (patchSize[d] - 1) / 2.0) * voxelSize[d];
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

    if (this->m_ImpactAssociate->m_FixedInterpolator->IsInsideBuffer(fixedPoint))
    {
      fixedAccessor[row][flat] = static_cast<float>(this->m_ImpactAssociate->m_FixedInterpolator->Evaluate(fixedPoint));
    }
    if (!this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(movingPoint))
    {
      continue;
    }
    const double movingValue = static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(movingPoint));
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
void
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TImpactMetric>::
  EvaluateJacobianBatch(typename std::vector<Sample>::const_iterator first,
                        typename std::vector<Sample>::const_iterator last,
                        const ThreadIdType                           threadId,
                        LossPerThreadStruct &                        loss) const
{
  constexpr unsigned int movingDimension = ImageToImageMetricv4Type::MovingImageDimension;

  const auto          batchSize = static_cast<int64_t>(std::distance(first, last));
  const bool          computeDerivative = this->GetComputeDerivative();
  const torch::Device device(this->m_ImpactAssociate->GetDevice());

  // d(moving coordinate)/d(parameter) at every point of the batch, and the global parameter
  // index each of its columns belongs to.
  torch::Tensor transformJacobian;      // [B, MovingDim, P]
  torch::Tensor nonZeroJacobianIndices; // [B, P]
  if (computeDerivative)
  {
    using JacobianType = typename TImageToImageMetric::JacobianType;
    using MovingTransformType = typename TImageToImageMetric::MovingTransformType;
    JacobianType & jacobian = this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobian;
    JacobianType & jacobianPositional =
      this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobianPositional;

    const NumberOfParametersType parameterCount = this->GetCachedNumberOfLocalParameters();
    transformJacobian = torch::empty(
      { batchSize, static_cast<int64_t>(movingDimension), static_cast<int64_t>(parameterCount) }, torch::kFloat32);
    nonZeroJacobianIndices = torch::empty({ batchSize, static_cast<int64_t>(parameterCount) }, torch::kInt64);
    auto jacobianAccessor = transformJacobian.accessor<float, 3>();
    auto indexAccessor = nonZeroJacobianIndices.accessor<int64_t, 2>();

    for (int64_t row = 0; row < batchSize; ++row)
    {
      const Sample & sample = *(first + row);
      this->m_ImpactAssociate->GetMovingTransform()->ComputeJacobianWithRespectToParametersCachedTemporaries(
        sample.m_VirtualPoint, jacobian, jacobianPositional);
      for (unsigned int dimension = 0; dimension < movingDimension; ++dimension)
      {
        for (NumberOfParametersType parameter = 0; parameter < parameterCount; ++parameter)
        {
          jacobianAccessor[row][dimension][parameter] = static_cast<float>(jacobian(dimension, parameter));
        }
      }
      // Map local parameter indices to global ones. A global transform (affine, rigid) has
      // offset 0; a local-support displacement field only affects the parameter block of this
      // point's own virtual voxel.
      int64_t parameterOffset = 0;
      if (this->m_ImpactAssociate->GetMovingTransform()->GetTransformCategory() ==
          MovingTransformType::TransformCategoryEnum::DisplacementField)
      {
        parameterOffset = static_cast<int64_t>(
          this->m_ImpactAssociate->ComputeParameterOffsetFromVirtualIndex(sample.m_VirtualIndex, parameterCount));
      }
      for (NumberOfParametersType parameter = 0; parameter < parameterCount; ++parameter)
      {
        indexAccessor[row][parameter] = parameterOffset + static_cast<int64_t>(parameter);
      }
    }
  }

  // The layers the metric compares, in mask order: one feature map, one loss and one weight
  // each, appended to the same flat list across models that Static builds.
  auto keptLayers = [](std::vector<torch::jit::IValue> & outputs, const ModelConfiguration & config) {
    std::vector<torch::Tensor> layers;
    const auto &               mask = config.GetLayersMask();
    for (size_t it = 0; it < mask.size(); ++it)
    {
      if (mask[it])
      {
        layers.push_back(outputs[it].toTensor());
      }
    }
    return layers;
  };
  // The central feature vector of every point of a layer tensor [B, C, s...] -> [B, C].
  auto centerFeatures = [](torch::Tensor layer) {
    while (layer.dim() > 2)
    {
      layer = layer.select(2, layer.size(2) / 2);
    }
    return layer;
  };

  const auto & fixedConfigs = this->m_ImpactAssociate->GetFixedModelsConfiguration();
  const auto & movingConfigs = this->m_ImpactAssociate->GetMovingModelsConfiguration();

  size_t comparison = 0; // flat index over (model, kept layer), the one the losses are read at
  for (size_t i = 0; i < fixedConfigs.size(); ++i)
  {
    const ModelConfiguration & fixedConfig = fixedConfigs[i];
    const ModelConfiguration & movingConfig = movingConfigs[i];
    const auto                 channels = static_cast<int64_t>(fixedConfig.GetNumberOfChannels());

    const std::vector<int64_t> & patchSize = fixedConfig.GetPatchSize();
    const int64_t voxelCount = std::accumulate(patchSize.begin(), patchSize.end(), int64_t{ 1 }, std::multiplies<>());

    torch::Tensor fixedPatches = torch::zeros({ batchSize, voxelCount }, torch::kFloat32);
    torch::Tensor movingPatches = torch::zeros({ batchSize, voxelCount }, torch::kFloat32);
    torch::Tensor movingGradients =
      torch::zeros({ batchSize, voxelCount, static_cast<int64_t>(movingDimension) }, torch::kFloat32);
    for (int64_t row = 0; row < batchSize; ++row)
    {
      this->SamplePatch(*(first + row), i, row, fixedPatches, movingPatches, movingGradients, computeDerivative);
    }

    // [B, C, s...]. PatchTensorShape is the one place that knows the spatial extents are
    // reversed (image axis 0 innermost): SamplePatch fills each row with image axis 0 varying
    // fastest, which is the layout ImageToTensorFilter produces and the one the TorchScript
    // models are trained on. The single intensity channel is then repeated to the channel
    // count the model expects.
    const std::vector<int64_t> tensorShape = Impact::PatchTensorShape(fixedConfig);
    std::vector<int64_t>       patchShape{ batchSize, 1 };
    patchShape.insert(patchShape.end(), tensorShape.begin(), tensorShape.end());
    std::vector<int64_t> channelRepeat(patchShape.size(), 1);
    channelRepeat[1] = channels;

    auto toModelPatch = [&](const torch::Tensor & values, const ModelConfiguration & config) {
      return values.reshape(patchShape).repeat(channelRepeat).to(device).to(GetModelDtype(config));
    };
    // The feature vector of one kept layer, restricted to the channels this comparison samples.
    auto featuresOf = [&](const torch::Tensor & layer, const torch::Tensor & subsetIndices) {
      return centerFeatures(layer).index_select(1, subsetIndices).to(torch::kFloat32);
    };
    // Which channels of comparison `a` are compared. Drawn once per batch and per comparison,
    // from the work unit's own generator.
    auto subsetIndicesOf = [&](size_t a) {
      const std::vector<unsigned int> subsetOfFeatures =
        this->GetSubsetOfFeatures(this->m_ImpactAssociate->m_features_indexes[a],
                                  loss.m_randomGenerator,
                                  this->m_ImpactAssociate->GetSubsetFeatures()[a]);
      return torch::tensor(std::vector<int64_t>(subsetOfFeatures.begin(), subsetOfFeatures.end()), torch::kInt64)
        .to(device);
    };

    // No gradient ever flows through the fixed image.
    std::vector<torch::Tensor> fixedLayers;
    {
      torch::NoGradGuard noGrad;
      auto outputs = Forward(fixedConfig, toModelPatch(fixedPatches, fixedConfig));
      fixedLayers = keptLayers(outputs, fixedConfig);
    }
    const auto layerCount = static_cast<size_t>(fixedLayers.size());

    if (!computeDerivative)
    {
      torch::NoGradGuard noGrad;
      auto                             outputs = Forward(movingConfig, toModelPatch(movingPatches, movingConfig));
      const std::vector<torch::Tensor> movingLayers = keptLayers(outputs, movingConfig);
      for (size_t layer = 0; layer < layerCount; ++layer, ++comparison)
      {
        const torch::Tensor subsetIndices = subsetIndicesOf(comparison);
        torch::Tensor       fixedFeatures = featuresOf(fixedLayers[layer], subsetIndices).to(torch::kCPU);
        torch::Tensor       movingFeatures = featuresOf(movingLayers[layer], subsetIndices).to(torch::kCPU);
        loss.m_losses[comparison]->updateValue(fixedFeatures, movingFeatures);
      }
      continue;
    }

    // The patch values are the autograd leaf, as in the elastix metric: differentiating the
    // features with respect to them and contracting with the moving interpolator's spatial
    // gradients gives d(feature)/d(moving coordinate) for the whole batch at once. One forward
    // pass serves every kept layer, so the graph has to survive until the last of them is done.
    torch::Tensor movingPatch = toModelPatch(movingPatches, movingConfig).detach().set_requires_grad(true);
    auto                             movingOutputs = Forward(movingConfig, movingPatch);
    const std::vector<torch::Tensor> movingLayers = keptLayers(movingOutputs, movingConfig);
    // d(patch value)/d(moving coordinate), tiled once per repeated channel so that it lines up
    // with the flattened gradient below: summing over the copies sums over the channel axis,
    // which is exactly the derivative with respect to the underlying intensity.
    torch::Tensor patchGradients = movingGradients.to(device).repeat({ 1, channels, 1 }); // [B, C * nVox, MovingDim]

    // ... and then into d(.)/d(parameter), which is what the losses accumulate.
    auto toParameterJacobian = [&](const torch::Tensor & featureCoordinateJacobian) {
      return torch::bmm(featureCoordinateJacobian.to(torch::kCPU).to(torch::kFloat32), transformJacobian);
    };

    for (size_t layer = 0; layer < layerCount; ++layer, ++comparison)
    {
      const torch::Tensor subsetIndices = subsetIndicesOf(comparison);
      torch::Tensor       fixedFeatures = featuresOf(fixedLayers[layer], subsetIndices).to(torch::kCPU);
      torch::Tensor       movingFeatures = featuresOf(movingLayers[layer], subsetIndices); // [B, C], on the device
      torch::Tensor       movingFeaturesOnCpu = movingFeatures.detach().to(torch::kCPU);
      const bool          laterLayersFollow = layer + 1 < layerCount;

      // One backward pass over the batch, seeded by `seed`, contracted into d(.)/d(coordinate).
      auto backward = [&](const torch::Tensor & seed, bool retainGraph) {
        const torch::Tensor gradient =
          torch::autograd::grad({ movingFeatures }, { movingPatch }, { seed }, retainGraph, false)[0];
        return torch::bmm(gradient.flatten(1).unsqueeze(1).to(torch::kFloat32), patchGradients); // [B, 1, MovingDim]
      };

      if (loss.m_losses[comparison]->IsPerPointMean())
      {
        // The loss is a mean of independent per-point terms, so its gradient with respect to the
        // features is known before any backward pass: seeding the pass with it yields
        // d(loss)/d(moving coordinate) directly, in ONE pass for the whole batch. This is what
        // the elastix metric does.
        torch::Tensor modulator =
          loss.m_losses[comparison]->updateValueAndGetGradientModulator(fixedFeatures, movingFeaturesOnCpu);
        torch::Tensor jacobian =
          toParameterJacobian(backward(modulator.to(device).to(movingFeatures.dtype()), laterLayersFollow));
        loss.m_losses[comparison]->updateDerivativeInJacobianMode(jacobian, nonZeroJacobianIndices);
      }
      else
      {
        // A global statistic (NCC) mixes the points together: its gradient with respect to the
        // features is only known once every point has been seen, so it cannot seed the pass --
        // seeded per batch it would measure the correlation within that batch alone, which at one
        // point is a zero variance and hence an identically zero gradient. Recover the full
        // per-channel Jacobian instead -- one pass per kept channel, each still over the whole
        // batch -- and let the loss fold it into the statistics it accumulates itself. That is
        // the closed form Static mode uses, and it stays exact whatever the batch size and
        // however the work units split the domain.
        const int64_t              keptChannels = movingFeatures.size(1);
        std::vector<torch::Tensor> perChannel;
        perChannel.reserve(static_cast<size_t>(keptChannels));
        torch::Tensor seed = torch::zeros_like(movingFeatures);
        for (int64_t channel = 0; channel < keptChannels; ++channel)
        {
          seed.zero_();
          seed.select(1, channel).fill_(1.0);
          perChannel.push_back(backward(seed, channel + 1 < keptChannels || laterLayersFollow).squeeze(1));
        }
        torch::Tensor jacobian = toParameterJacobian(torch::stack(perChannel, 1)); // [B, C, MovingDim] -> [B, C, P]
        loss.m_losses[comparison]->updateValueAndDerivativeInStaticMode(
          fixedFeatures, movingFeaturesOnCpu, jacobian, nonZeroJacobianIndices);
      }
    }
  }
}


template <typename TDomainPartitioner, typename TImageToImageMetric, typename TImpactMetric>
bool
ImpactImageToImageMetricv4GetValueAndDerivativeThreader<
  TDomainPartitioner,
  TImageToImageMetric,
  TImpactMetric>::ProcessPoint(const VirtualIndexType & virtualIndex,
                               const VirtualPointType & virtualPoint,
                               const FixedImagePointType &  mappedFixedPoint,
                               const FixedImagePixelType &,
                               const FixedImageGradientType &,
                               const MovingImagePointType & mappedMovingPoint,
                               const MovingImagePixelType &,
                               const MovingImageGradientType &,
                               MeasureType &      metricValueReturn,
                               DerivativeType &,
                               const ThreadIdType threadId) const
{
  // Static mode only: interpolate the (optionally subsampled) per-layer feature vector at the
  // mapped fixed/moving point and accumulate the per-layer loss. When a derivative is
  // requested, chain the moving feature spatial gradient with the moving-transform Jacobian.
  // Value and derivative are reduced and normalized in AfterThreadedExecution. Online mode
  // never reaches here: ThreadedExecution() walks its points itself, in batches.
  LossPerThreadStruct & loss = this->m_LossThreadStruct[threadId];

  const bool computeDerivative = this->GetComputeDerivative();

  torch::Tensor transformJacobian;        // [1, MovingDim, P]
  torch::Tensor nonZeroJacobianIndices;   // [1, P]
  if (computeDerivative)
  {
    using JacobianType = typename TImageToImageMetric::JacobianType;
    JacobianType & jacobian = this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobian;
    JacobianType & jacobianPositional =
      this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobianPositional;
    // For a dense/global transform this is the full Jacobian and every parameter
    // contributes at every point (handled below via an identity index list).
    this->m_ImpactAssociate->GetMovingTransform()->ComputeJacobianWithRespectToParametersCachedTemporaries(
      virtualPoint, jacobian, jacobianPositional);

    constexpr unsigned int      movingDimension = ImageToImageMetricv4Type::MovingImageDimension;
    const NumberOfParametersType numberOfLocalParameters = this->GetCachedNumberOfLocalParameters();
    transformJacobian =
      torch::empty({ 1, static_cast<int64_t>(movingDimension), static_cast<int64_t>(numberOfLocalParameters) },
                   torch::kFloat32);
    auto jacobianAccessor = transformJacobian.accessor<float, 3>();
    for (unsigned int dimension = 0; dimension < movingDimension; ++dimension)
    {
      for (NumberOfParametersType parameter = 0; parameter < numberOfLocalParameters; ++parameter)
      {
        jacobianAccessor[0][dimension][parameter] = static_cast<float>(jacobian(dimension, parameter));
      }
    }
    // Map local parameter indices to global ones. A global transform (affine, rigid) has
    // offset 0. A local-support displacement field only affects the parameter block of
    // this point's own virtual voxel, at ComputeParameterOffsetFromVirtualIndex.
    int64_t parameterOffset = 0;
    using MovingTransformType = typename TImageToImageMetric::MovingTransformType;
    if (this->m_ImpactAssociate->GetMovingTransform()->GetTransformCategory() ==
        MovingTransformType::TransformCategoryEnum::DisplacementField)
    {
      parameterOffset = static_cast<int64_t>(
        this->m_ImpactAssociate->ComputeParameterOffsetFromVirtualIndex(virtualIndex, numberOfLocalParameters));
    }
    nonZeroJacobianIndices =
      (torch::arange(static_cast<int64_t>(numberOfLocalParameters), torch::kInt64) + parameterOffset).unsqueeze(0);
  }

  for (size_t i = 0; i < this->m_ImpactAssociate->m_Internals->m_FixedFeaturesMaps.size(); ++i)
  {
    const std::vector<unsigned int> subsetOfFeatures =
      this->GetSubsetOfFeatures(this->m_ImpactAssociate->m_features_indexes[i],
                                loss.m_randomGenerator,
                                this->m_ImpactAssociate->GetSubsetFeatures()[i]);

    torch::Tensor fixedFeatures =
      this->m_ImpactAssociate->m_Internals->m_FixedFeaturesMaps[i].m_FeaturesMapInterpolator->Evaluate(mappedFixedPoint, subsetOfFeatures)
        .unsqueeze(0);
    torch::Tensor movingFeatures =
      this->m_ImpactAssociate->m_Internals->m_MovingFeaturesMaps[i].m_FeaturesMapInterpolator->Evaluate(mappedMovingPoint, subsetOfFeatures)
        .unsqueeze(0);

    if (computeDerivative)
    {
      // d(feature)/d(moving coordinate) : [1, C, MovingDim]
      torch::Tensor movingFeatureDerivative =
        this->m_ImpactAssociate->m_Internals->m_MovingFeaturesMaps[i].m_FeaturesMapInterpolator
          ->EvaluateDerivative(mappedMovingPoint, subsetOfFeatures)
          .unsqueeze(0);
      // chain with d(moving coordinate)/d(parameter) : [1, C, P]
      torch::Tensor featureParameterJacobian = torch::bmm(movingFeatureDerivative, transformJacobian);
      loss.m_losses[i]->updateValueAndDerivativeInStaticMode(
        fixedFeatures, movingFeatures, featureParameterJacobian, nonZeroJacobianIndices);
    }
    else
    {
      loss.m_losses[i]->updateValue(fixedFeatures, movingFeatures);
    }
  }
  ++loss.m_numberOfPixelsCounted;

  metricValueReturn = MeasureType{};
  return true;
}

} // end namespace itk

#endif

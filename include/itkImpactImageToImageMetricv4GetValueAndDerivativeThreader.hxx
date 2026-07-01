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
  // Static mode (the default, handled by the loop below): interpolate the (optionally
  // subsampled) per-layer feature vector at the mapped fixed/moving point and accumulate
  // the per-layer loss. When a derivative is requested, chain the moving feature spatial
  // gradient with the moving-transform Jacobian. Value/derivative are reduced and
  // normalized in AfterThreadedExecution. Jacobian mode is handled in the block below.
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

  // ---- Online ("Jacobian") mode -------------------------------------------------
  // Extract an intensity patch around the point in each image, run the model online and
  // differentiate through it via the chain rule:
  //   d(loss)/d(param) = d(loss)/d(movingFeature)            [loss modulator]
  //                    x d(movingFeature)/d(movingCoord)     [autograd through model x interpolator gradient]
  //                    x d(movingCoord)/d(param)             [transform Jacobian].
  // The modulator+value math is identical to Static mode, so we build the feature/parameter
  // Jacobian by autograd here and hand it to updateValueAndDerivativeInStaticMode.
  if (this->m_ImpactAssociate->GetMode() == "Jacobian")
  {
    constexpr unsigned int movingDimension = ImageToImageMetricv4Type::MovingImageDimension;
    // Central-difference step (physical units) to linearize the moving interpolator around
    // the sample point. Using the SAME interpolator that produces the value keeps the
    // analytic derivative consistent with finite differences of the value.
    constexpr double kGradientStep = 1e-3;

    auto firstKeptLayer =
      [](std::vector<torch::jit::IValue> & outputs, const ModelConfiguration & cfg) -> torch::Tensor {
      const auto & mask = cfg.GetLayersMask();
      for (size_t it = 0; it < outputs.size(); ++it)
        if (it < mask.size() && mask[it])
          return outputs[it].toTensor();
      return outputs[0].toTensor();
    };
    // The central feature vector [C] of a layer tensor [1, C, s0, s1, ...].
    auto centerFeature = [](const torch::Tensor & layer) -> torch::Tensor {
      torch::Tensor t = layer.squeeze(0); // [C, s...]
      while (t.dim() > 1)
        t = t.select(1, t.size(1) / 2);
      return t; // [C]
    };

    for (size_t i = 0; i < this->m_ImpactAssociate->GetFixedModelsConfiguration().size(); ++i)
    {
      const ModelConfiguration &   fixedConfig = this->m_ImpactAssociate->GetFixedModelsConfiguration()[i];
      const ModelConfiguration &   movingConfig = this->m_ImpactAssociate->GetMovingModelsConfiguration()[i];
      const std::vector<int64_t> & patchSize = fixedConfig.GetPatchSize();
      const std::vector<float> &   voxelSize = fixedConfig.GetVoxelSize();
      const unsigned int           dim = fixedConfig.GetDimension();

      int64_t nVox = 1;
      for (unsigned int d = 0; d < dim; ++d)
        nVox *= patchSize[d];

      // Sample fixed-patch intensities, moving-patch intensities and the moving
      // interpolator's spatial gradient (central difference) at every patch voxel.
      std::vector<float> fixedVals(static_cast<size_t>(nVox), 0.0f);
      std::vector<float> movingVals(static_cast<size_t>(nVox), 0.0f);
      torch::Tensor movingGrad = torch::zeros({ nVox, static_cast<int64_t>(movingDimension) }, torch::kFloat32);
      auto          movingGradAccessor = movingGrad.accessor<float, 2>();

      for (int64_t flat = 0; flat < nVox; ++flat)
      {
        // Decode the row-major multi-index and build the physical offset along each axis.
        int64_t             rem = flat;
        FixedImagePointType  fixedPt = mappedFixedPoint;
        MovingImagePointType movingPt = mappedMovingPoint;
        for (int d = static_cast<int>(dim) - 1; d >= 0; --d)
        {
          const int64_t idx = rem % patchSize[d];
          rem /= patchSize[d];
          const double off = (static_cast<double>(idx) - (patchSize[d] - 1) / 2.0) * voxelSize[d];
          fixedPt[d] += off;
          movingPt[d] += off;
        }

        if (this->m_ImpactAssociate->m_FixedInterpolator->IsInsideBuffer(fixedPt))
          fixedVals[flat] = static_cast<float>(this->m_ImpactAssociate->m_FixedInterpolator->Evaluate(fixedPt));

        if (this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(movingPt))
        {
          movingVals[flat] = static_cast<float>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(movingPt));
          if (computeDerivative)
          {
            for (unsigned int d = 0; d < movingDimension; ++d)
            {
              MovingImagePointType plusPt = movingPt;
              MovingImagePointType minusPt = movingPt;
              plusPt[d] += kGradientStep;
              minusPt[d] -= kGradientStep;
              const double valuePlus =
                this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(plusPt)
                  ? static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(plusPt))
                  : static_cast<double>(movingVals[flat]);
              const double valueMinus =
                this->m_ImpactAssociate->m_MovingInterpolator->IsInsideBuffer(minusPt)
                  ? static_cast<double>(this->m_ImpactAssociate->m_MovingInterpolator->Evaluate(minusPt))
                  : static_cast<double>(movingVals[flat]);
              movingGradAccessor[flat][d] = static_cast<float>((valuePlus - valueMinus) / (2.0 * kGradientStep));
            }
          }
        }
      }

      // Patch spatial shape and the per-channel repeat the model expects: [1, C, s...].
      std::vector<int64_t> spatial;
      for (unsigned int d = 0; d < dim; ++d)
        spatial.push_back(patchSize[d]);
      std::vector<int64_t> channelRepeat(dim + 1, 1);
      channelRepeat[0] = fixedConfig.GetNumberOfChannels();

      const torch::Device device(this->m_ImpactAssociate->GetDevice());

      const std::vector<unsigned int> subsetOfFeatures =
        this->GetSubsetOfFeatures(this->m_ImpactAssociate->m_features_indexes[i],
                                  loss.m_randomGenerator,
                                  this->m_ImpactAssociate->GetSubsetFeatures()[i]);
      torch::Tensor subsetIdx =
        torch::tensor(std::vector<int64_t>(subsetOfFeatures.begin(), subsetOfFeatures.end()), torch::kInt64);

      // Fixed features (no gradient flows through the fixed image).
      torch::Tensor fixedFeatures;
      {
        torch::NoGradGuard ng;
        torch::Tensor      fixedPatch = torch::from_blob(fixedVals.data(), { nVox }, torch::kFloat32)
                                     .clone()
                                     .reshape(spatial)
                                     .unsqueeze(0)
                                     .repeat(channelRepeat)
                                     .unsqueeze(0)
                                     .to(device)
                                     .to(GetModelDtype(fixedConfig));
        auto          outputs = Forward(fixedConfig, fixedPatch);
        torch::Tensor layer = firstKeptLayer(outputs, fixedConfig);
        fixedFeatures =
          centerFeature(layer).unsqueeze(0).to(torch::kCPU).to(torch::kFloat32).index_select(1, subsetIdx); // [1, C]
      }

      if (computeDerivative)
      {
        // Rigidly shift the whole patch by an autograd leaf coordOffset (the per-point
        // moving-coordinate displacement), linearizing each intensity through the moving
        // interpolator's gradient so autograd can flow coordOffset -> feature.
        torch::Tensor coordOffset =
          torch::zeros({ static_cast<int64_t>(movingDimension) },
                       torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
        torch::Tensor movingBase = torch::from_blob(movingVals.data(), { nVox }, torch::kFloat32).clone();
        torch::Tensor movingFlat = movingBase + movingGrad.matmul(coordOffset); // [nVox], grad wrt coordOffset
        torch::Tensor movingPatch = movingFlat.reshape(spatial)
                                       .unsqueeze(0)
                                       .repeat(channelRepeat)
                                       .unsqueeze(0)
                                       .to(device)
                                       .to(GetModelDtype(movingConfig));
        auto          outputs = Forward(movingConfig, movingPatch);
        torch::Tensor layer = firstKeptLayer(outputs, movingConfig);
        torch::Tensor movingFeatures =
          centerFeature(layer).unsqueeze(0).to(torch::kCPU).to(torch::kFloat32).index_select(1, subsetIdx); // [1, C]

        // d(movingFeature_c)/d(coordOffset) per kept channel -> [C, movingDim].
        const int64_t channels = movingFeatures.size(1);
        torch::Tensor featureCoordJacobian =
          torch::zeros({ channels, static_cast<int64_t>(movingDimension) }, torch::kFloat32);
        for (int64_t c = 0; c < channels; ++c)
        {
          torch::Tensor scalar = movingFeatures.select(1, c).sum();
          auto          grads = torch::autograd::grad({ scalar },
                                              { coordOffset },
                                              /*grad_outputs=*/{},
                                              /*retain_graph=*/true,
                                              /*create_graph=*/false,
                                              /*allow_unused=*/true);
          if (grads[0].defined())
            featureCoordJacobian[c] = grads[0].to(torch::kCPU).to(torch::kFloat32);
        }
        // [1, C, movingDim] x [1, movingDim, P] -> [1, C, P]; fold in the loss modulator
        // and accumulate value+derivative exactly like Static mode.
        torch::Tensor featureParameterJacobian = torch::bmm(featureCoordJacobian.unsqueeze(0), transformJacobian);
        torch::Tensor movingFeaturesDetached = movingFeatures.detach();
        loss.m_losses[i]->updateValueAndDerivativeInStaticMode(
          fixedFeatures, movingFeaturesDetached, featureParameterJacobian, nonZeroJacobianIndices);
      }
      else
      {
        torch::NoGradGuard ng;
        torch::Tensor      movingPatch = torch::from_blob(movingVals.data(), { nVox }, torch::kFloat32)
                                      .clone()
                                      .reshape(spatial)
                                      .unsqueeze(0)
                                      .repeat(channelRepeat)
                                      .unsqueeze(0)
                                      .to(device)
                                      .to(GetModelDtype(movingConfig));
        auto          outputs = Forward(movingConfig, movingPatch);
        torch::Tensor layer = firstKeptLayer(outputs, movingConfig);
        torch::Tensor movingFeatures =
          centerFeature(layer).unsqueeze(0).to(torch::kCPU).to(torch::kFloat32).index_select(1, subsetIdx);
        loss.m_losses[i]->updateValue(fixedFeatures, movingFeatures);
      }
    }

    ++loss.m_numberOfPixelsCounted;
    metricValueReturn = MeasureType{};
    return true;
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

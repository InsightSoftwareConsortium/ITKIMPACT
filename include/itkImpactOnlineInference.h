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
#ifndef itkImpactOnlineInference_h
#define itkImpactOnlineInference_h

// Framework-neutral online ("Jacobian") inference shared one-way (Elastix -> ITKIMPACT):
// patch-offset sampling, batched TorchScript forward, center extraction, and the autograd
// value+Jacobian path used by the registration metrics. Depends only on the backend
// ImpactModelConfiguration (+ its torch accessors) and the IMPACT losses, talking to the host
// framework through point-sampling callbacks (ImagesPatchValues[AndJacobians]Evaluator).
// Pulls in LibTorch; never part of the castxml-parsed public surface.

#include "itkImpactModelConfiguration.h"
#include "itkImpactModelConfigurationDetail.h"
#include "itkImpactBatchBudget.h"
#include "ImpactLoss.h"

#include <itkMacro.h>
#include <itkMath.h>
#include <itkMatrix.h>
#include <itkPoint.h>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace itk
{
namespace Impact
{

/** The shape a sampled patch must be given as a tensor.
 *
 * The host framework fills a flat buffer in GetPatchIndex() order, which runs ITK axis 0
 * fastest. A row-major tensor over that buffer therefore has ITK x as its LAST, fastest axis:
 * the shape is the configured patch size reversed, the same convention ImageToTensorFilter and
 * the metric threader build their tensors with. Passing the un-reversed patch size is correct
 * only when its first and last entries happen to be equal -- an isotropic patch -- and
 * silently re-partitions the buffer otherwise. Callbacks are handed this, never the raw patch
 * size, so the convention lives in one place. */
inline std::vector<int64_t>
PatchTensorShape(const ImpactModelConfiguration & configuration)
{
  const std::vector<int64_t> & patchSize = configuration.GetPatchSize();
  return std::vector<int64_t>(patchSize.rbegin(), patchSize.rend());
}

/** The plane a model of dimension 2 cuts its patch on in a volume, at the point of index `index` (any type with
 * operator[] over three axes), as a rotation whose first two COLUMNS are the patch axes in the image's own frame:
 * three angles drawn uniform in [0, 2 pi) from a generator seeded from the point, composed Rz * Ry * Rx, as the
 * elastix metric draws them.
 *
 * The generator is seeded from the point itself rather than from a running generator, so the plane depends on WHERE
 * the point is and not on how many points came before it: a metric stays a function of its parameters when a
 * finite-difference step drops a few points, whatever the partition of its domain. Mixing the model index in keeps
 * two models at one point from being handed the identical plane. Shared by the itkv4 metric and the fine
 * registration stage's sampled Jacobian mode. */
template <typename TIndex>
Matrix<double, 3, 3>
PatchPlaneRotation(unsigned int seed, size_t modelIndex, const TIndex & index)
{
  std::uint_fast32_t pointSeed = static_cast<std::uint_fast32_t>(seed) + 0x9e3779b9u * (modelIndex + 1u);
  for (unsigned int d = 0; d < 3; ++d)
  {
    pointSeed = pointSeed * 2654435761u + static_cast<std::uint_fast32_t>(index[d]);
  }
  std::mt19937                           pointGenerator(pointSeed);
  std::uniform_real_distribution<double> angles(0.0, 2.0 * itk::Math::pi);
  const double                           a = angles(pointGenerator);
  const double                           b = angles(pointGenerator);
  const double                           c = angles(pointGenerator);
  const double                           ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b);
  const double                           cc = std::cos(c), sc = std::sin(c);
  Matrix<double, 3, 3>                   plane;
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

/** Callback evaluating a patch of image intensities around a point (value-only path). Its third
 * argument is the tensor shape the returned patch must have: see PatchTensorShape(). */
template <typename ImagePointType>
using ImagesPatchValuesEvaluator = std::function<
  torch::Tensor(const ImagePointType &, const std::vector<std::vector<float>> &, const std::vector<int64_t> &)>;

/** Callback evaluating a patch of image intensities and accumulating the per-voxel
 * intensity->coordinate Jacobian (derivative path). */
template <typename ImagePointType>
using ImagesPatchValuesAndJacobiansEvaluator = std::function<torch::Tensor(const ImagePointType &,
                                                                           torch::Tensor &,
                                                                           const std::vector<std::vector<float>> &,
                                                                           const std::vector<int64_t> &,
                                                                           int)>;

inline std::vector<torch::Tensor>
GetModelOutputsExample(std::vector<itk::ImpactModelConfiguration> & modelsConfig,
                       const std::string &                          modelType,
                       torch::Device                                device)
{

  // Run each model on a dummy patch to probe its output-layer structure.
  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard noGrad;
    for (int i = 0; i < modelsConfig.size(); ++i)
    {
      const auto &         config = modelsConfig[i];
      std::vector<int64_t> resizeVector(config.GetPatchSize().size() + 1, 1);
      resizeVector[0] = config.GetNumberOfChannels();
      std::vector<torch::jit::IValue> outputsList;
      const std::vector<int64_t>      tensorShape = PatchTensorShape(config);
      auto modelInput = torch::zeros({ torch::IntArrayRef(tensorShape) }, itk::GetModelDtype(config))
                          .unsqueeze(0)
                          .repeat({ torch::IntArrayRef(resizeVector) })
                          .unsqueeze(0)
                          .clone()
                          .to(device);
      try
      {
        outputsList = itk::Forward(config, modelInput);
      }
      catch (const std::exception & e)
      {
        itkGenericExceptionMacro(
          "ERROR: The " << modelType << " model " << i
                        << " configuration is invalid. The dimensions, number of channels, or patch size may "
                           "not meet the requirements of the model.\n"
                           "Details:\n"
                           " - Number of channels: "
                        << config.GetNumberOfChannels()
                        << "\n"
                           " - Patch size: "
                        << config.GetPatchSize()
                        << "\n"
                           " - Dimension: "
                        << config.GetDimension()
                        << "\n"
                           "Please verify the configuration to ensure compatibility with the model. \n Exception : "
                        << e.what());
      }
      if (config.GetLayersMask().size() != outputsList.size())
      {
        itkGenericExceptionMacro("Error: The number of " << modelType << " masks (" << config.GetLayersMask().size()
                                                         << ") does not match the number of layers ("
                                                         << outputsList.size()
                                                         << "). Please ensure that the configuration is consistent.");
      }

      for (int it = 0; it < outputsList.size(); ++it)
      {
        if (config.GetLayersMask()[it])
        {
          outputsTensor.push_back(outputsList[it].toTensor().to(torch::kCPU));
        }
      }
    }
    for (itk::ImpactModelConfiguration & config : modelsConfig)
    {
      std::vector<std::vector<torch::indexing::TensorIndex>> centersIndexLayers;
      for (const torch::Tensor & tensor : outputsTensor)
      {
        std::vector<torch::indexing::TensorIndex> centersIndexLayer;
        centersIndexLayer.push_back("...");
        for (int j = 2; j < tensor.dim(); ++j)
        {
          centersIndexLayer.push_back(tensor.size(j) / 2);
        }
        centersIndexLayers.push_back(centersIndexLayer);
      }
      itk::SetCentersIndexLayers(config, centersIndexLayers);
    }
  }
  // How many patches one online forward may hold on the device (itkImpactBatchBudget.h). Done
  // here, outside the no-grad scope, so every host of the online mode gets it from this call.
  ConfigureBatchSize(modelsConfig, device);
  return outputsTensor;
} // end GetModelOutputsExample

/** The offsets, in millimetres along the image's own axes, of model `modelConfiguration`'s patch voxels around the
 * point of fixed-grid index `index` (model axis 0 running fastest). A model of the image's dimension takes its
 * precomputed box (GetPatchIndex of the configuration). A 2D model in a volume takes the plane PatchPlaneRotation
 * draws from `seed`, `modelIndex` and the point, as the itkv4 metric and ImpactFineRegistration's sampled Jacobian mode
 * do, so that the three hosts cut the same patch for the same point, seed and model. */
template <typename TIndex>
std::vector<std::vector<float>>
GetPatchIndex(const itk::ImpactModelConfiguration & modelConfiguration,
              unsigned int                          seed,
              size_t                                modelIndex,
              const TIndex &                        index,
              unsigned int                          dimension)
{
  if (dimension == modelConfiguration.GetPatchSize().size())
  {
    return modelConfiguration.GetPatchIndex();
  }
  if (dimension != 3 || modelConfiguration.GetPatchSize().size() != 2)
  {
    itkGenericExceptionMacro("IMPACT: a " << modelConfiguration.GetPatchSize().size() << "D patch cannot be cut in a "
                                          << dimension << "D image; only a 2D model in a volume is swept.");
  }
  const Matrix<double, 3, 3>      plane = PatchPlaneRotation(seed, modelIndex, index);
  const std::vector<int64_t> &    patchSize = modelConfiguration.GetPatchSize();
  const std::vector<float> &      voxelSize = modelConfiguration.GetVoxelSize();
  std::vector<std::vector<float>> patchIndex;
  patchIndex.reserve(static_cast<size_t>(patchSize[0] * patchSize[1]));
  for (int64_t y = 0; y < patchSize[1]; ++y)
  {
    for (int64_t x = 0; x < patchSize[0]; ++x)
    {
      const double u = static_cast<double>(x - patchSize[0] / 2) * voxelSize[0];
      const double v = static_cast<double>(y - patchSize[1] / 2) * voxelSize[1];
      patchIndex.push_back({ static_cast<float>(plane[0][0] * u + plane[0][1] * v),
                             static_cast<float>(plane[1][0] * u + plane[1][1] * v),
                             static_cast<float>(plane[2][0] * u + plane[2][1] * v) });
    }
  }
  return patchIndex;
} // end GetPatchIndex

template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<itk::ImpactModelConfiguration> &                modelConfig,
                const std::vector<ImagePointType> &                               fixedPoints,
                const std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex,
                const std::vector<torch::Tensor>                                  subsetsOfFeatures,
                torch::Device                                                     device,
                const ImagesPatchValuesEvaluator<ImagePointType> &                imagesPatchValuesEvaluator)
{

  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard noGrad;
    const auto         nbSample = static_cast<int64_t>(fixedPoints.size());

    size_t a = 0;
    for (size_t i = 0; i < modelConfig.size(); ++i)
    {
      const auto & config = modelConfig[i];

      std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
      sizes[0] = nbSample;

      const std::vector<int64_t> tensorShape = PatchTensorShape(config);
      torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(tensorShape) }, itk::GetModelDtype(config))
                                         .unsqueeze(0)
                                         .expand(sizes)
                                         .unsqueeze(1)
                                         .clone();

      for (int64_t s = 0; s < nbSample; ++s)
      {
        patchValueTensor[s] =
          imagesPatchValuesEvaluator(fixedPoints[s], patchIndex[i][s], tensorShape).to(itk::GetModelDtype(config));
      }

      std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
      resizeVector[1] = config.GetNumberOfChannels();

      // Batched by the device budget (itkImpactBatchBudget.h); kept layers joined at the end.
      const std::vector<bool> &               mask = config.GetLayersMask();
      const auto                              kept = static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
      std::vector<std::vector<torch::Tensor>> batches(kept);
      ForEachBatch(
        config,
        device,
        nbSample,
        [&](int64_t begin, int64_t end) {
          torch::Tensor input = patchValueTensor.narrow(0, begin, end - begin)
                                  .to(device)
                                  .repeat({ torch::IntArrayRef(resizeVector) })
                                  .clone();
          std::vector<torch::jit::IValue> outputsList = itk::Forward(config, input);
          std::vector<torch::Tensor>      layers;
          layers.reserve(kept);
          for (size_t it = 0; it < outputsList.size(); ++it)
          {
            if (mask[it])
            {
              const size_t k = layers.size();
              // The point's feature vector normalized over all its channels, before the subset keeps a few.
              layers.push_back(
                Impact::NormalizeFeatureChannels(
                  outputsList[it].toTensor().index(itk::GetCentersIndexLayers(config)[a + k]).to(torch::kFloat32),
                  config.GetFeatureNormalization(),
                  1)
                  .index_select(1, subsetsOfFeatures[a + k]));
            }
          }
          for (size_t k = 0; k < layers.size(); ++k)
          {
            batches[k].push_back(layers[k]);
          }
        },
        [] {});
      for (size_t k = 0; k < kept; ++k)
      {
        outputsTensor.push_back(torch::cat(batches[k], 0));
      }
      a += kept;
    }
  }
  return outputsTensor;
} // end GenerateOutputs

template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputsAndJacobian(
  const std::vector<itk::ImpactModelConfiguration> &                modelConfig,
  const std::vector<ImagePointType> &                               fixedPoints,
  const std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex,
  std::vector<torch::Tensor>                                        subsetsOfFeatures,
  std::vector<torch::Tensor>                                        fixedOutputsTensor,
  torch::Device                                                     device,
  std::vector<std::unique_ptr<itk::Impact::Loss>> &                 losses,
  const ImagesPatchValuesAndJacobiansEvaluator<ImagePointType> &    imagesPatchValuesAndJacobiansEvaluator)
{
  std::vector<torch::Tensor> layersJacobian;

  const auto   nbSample = static_cast<int64_t>(fixedPoints.size());
  unsigned int dimension = fixedPoints[0].size();

  size_t a = 0;
  for (size_t i = 0; i < modelConfig.size(); ++i)
  {
    const auto & config = modelConfig[i];

    std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
    sizes[0] = nbSample;

    const std::vector<int64_t> tensorShape = PatchTensorShape(config);
    torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(tensorShape) }, itk::GetModelDtype(config))
                                       .unsqueeze(0)
                                       .expand(sizes)
                                       .unsqueeze(1)
                                       .clone();
    torch::Tensor imagesPatchesJacobians =
      torch::zeros({ nbSample, static_cast<int64_t>(patchIndex[i][0].size()), dimension }, torch::kFloat32);

    for (int64_t s = 0; s < nbSample; ++s)
    {
      patchValueTensor[s] =
        imagesPatchValuesAndJacobiansEvaluator(fixedPoints[s], imagesPatchesJacobians, patchIndex[i][s], tensorShape, s)
          .to(itk::GetModelDtype(config));
    }

    std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
    resizeVector[1] = config.GetNumberOfChannels();
    const auto channels = static_cast<int64_t>(config.GetNumberOfChannels());

    const std::vector<bool> & mask = config.GetLayersMask();
    const auto                kept = static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
    // The losses accumulate inside a batch; a replayed batch is rewound to what they held before.
    std::vector<std::vector<torch::Tensor>> batches(kept);
    std::vector<itk::Impact::Loss::State>   before(kept);
    auto                                    rewind = [&]() {
      for (size_t k = 0; k < kept; ++k)
      {
        losses[a + k]->RestoreState(before[k]);
      }
    };
    ForEachBatch(
      config,
      device,
      nbSample,
      [&](int64_t begin, int64_t end) {
        for (size_t k = 0; k < kept; ++k)
        {
          before[k] = losses[a + k]->SaveState();
        }
        const int64_t n = end - begin;
        torch::Tensor input = patchValueTensor.narrow(0, begin, n)
                                .to(device)
                                .repeat({ torch::IntArrayRef(resizeVector) })
                                .clone()
                                .set_requires_grad(true);
        torch::Tensor patchJacobians =
          imagesPatchesJacobians.narrow(0, begin, n).to(device).repeat({ 1, channels, 1 }).clone();

        std::vector<torch::jit::IValue> outputsList = itk::Forward(config, input);
        std::vector<torch::Tensor>      jacobians;
        jacobians.reserve(kept);
        for (size_t it = 0; it < outputsList.size(); ++it)
        {
          if (!mask[it])
          {
            continue;
          }
          const size_t k = jacobians.size();
          // Normalized inside the graph, so the Jacobian below carries the normalization too.
          torch::Tensor layer =
            Impact::NormalizeFeatureChannels(
              outputsList[it].toTensor().index(itk::GetCentersIndexLayers(config)[a + k]).to(torch::kFloat32),
              config.GetFeatureNormalization(),
              1)
              .index_select(1, subsetsOfFeatures[a + k]);
          torch::Tensor fixedLayer = fixedOutputsTensor[a + k].narrow(0, begin, n);
          torch::Tensor gradientModulator = losses[a + k]->updateValueAndGetGradientModulator(fixedLayer, layer);
          jacobians.push_back(
            torch::bmm(torch::autograd::grad({ layer }, { input }, { gradientModulator }, kept > 1, false)[0]
                         .flatten(1)
                         .unsqueeze(1)
                         .to(torch::kFloat32),
                       patchJacobians));
        }
        for (size_t k = 0; k < jacobians.size(); ++k)
        {
          batches[k].push_back(jacobians[k]);
        }
      },
      rewind);
    for (size_t k = 0; k < kept; ++k)
    {
      layersJacobian.push_back(torch::cat(batches[k], 0));
    }
    a += kept;
  }
  return layersJacobian;
} // end GenerateOutputsAndJacobian


} // namespace Impact
} // namespace itk

#endif // end #ifndef itkImpactOnlineInference_h

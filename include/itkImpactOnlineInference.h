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
// ModelConfiguration (+ its torch accessors) and the IMPACT losses, talking to the host
// framework through point-sampling callbacks (ImagesPatchValues[AndJacobians]Evaluator).
// Pulls in LibTorch; never part of the castxml-parsed public surface.

#include "itkModelConfiguration.h"
#include "itkModelConfigurationDetail.h"
#include "ImpactLoss.h"

#include <itkMacro.h>
#include <itkMatrix.h>
#include <itkPoint.h>

#include <torch/torch.h>

#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace itk
{
namespace Impact
{

/** Callback evaluating a patch of image intensities around a point (value-only path). */
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
GetModelOutputsExample(std::vector<itk::ModelConfiguration> & modelsConfig,
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
      auto modelInput = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, itk::GetModelDtype(config))
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
    for (itk::ModelConfiguration & config : modelsConfig)
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
  return outputsTensor;
} // end GetModelOutputsExample

inline std::vector<std::vector<float>>
GetPatchIndex(const itk::ModelConfiguration & modelConfiguration,
              std::mt19937 &                        randomGenerator,
              unsigned int                          dimension)
{
  if (dimension == modelConfiguration.GetPatchSize().size())
  {
    return modelConfiguration.GetPatchIndex();
  }
  else
  {

    using MatrixType = itk::Matrix<float, 3, 3>;
    using Point3D = itk::Point<float, 3>;
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);

    double radX = angleDist(randomGenerator);
    double radY = angleDist(randomGenerator);
    double radZ = angleDist(randomGenerator);

    MatrixType rotationX;
    MatrixType rotationY;
    MatrixType rotationZ;

    rotationX.SetIdentity();
    rotationY.SetIdentity();
    rotationZ.SetIdentity();

    rotationX[1][1] = cos(radX);
    rotationX[1][2] = -sin(radX);
    rotationX[2][1] = sin(radX);
    rotationX[2][2] = cos(radX);

    rotationY[0][0] = cos(radY);
    rotationY[0][2] = sin(radY);
    rotationY[2][0] = -sin(radY);
    rotationY[2][2] = cos(radY);

    rotationZ[0][0] = cos(radZ);
    rotationZ[0][1] = -sin(radZ);
    rotationZ[1][0] = sin(radZ);
    rotationZ[1][1] = cos(radZ);

    MatrixType                      matrix = rotationZ * rotationY * rotationX;
    std::vector<std::vector<float>> patchIndex;

    for (int y = 0; y < modelConfiguration.GetPatchSize()[1]; ++y)
    {
      for (int x = 0; x < modelConfiguration.GetPatchSize()[0]; ++x)
      {
        Point3D point({ (x - modelConfiguration.GetPatchSize()[0] / 2) * modelConfiguration.GetVoxelSize()[0],
                        (y - modelConfiguration.GetPatchSize()[1] / 2) * modelConfiguration.GetVoxelSize()[1],
                        0 });
        point = matrix * point;
        std::vector<float> vec(3);
        vec[0] = point[0];
        vec[1] = point[1];
        vec[2] = point[2];
        patchIndex.push_back(vec);
      }
    }
    return patchIndex;
  }
} // end GetPatchIndex

template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputs(const std::vector<itk::ModelConfiguration> &                    modelConfig,
                const std::vector<ImagePointType> &                                   fixedPoints,
                const std::vector<std::vector<std::vector<std::vector<float>>>> &     patchIndex,
                const std::vector<torch::Tensor>                                      subsetsOfFeatures,
                torch::Device                                                         device,
                const ImagesPatchValuesEvaluator<ImagePointType> & imagesPatchValuesEvaluator)
{

  std::vector<torch::Tensor> outputsTensor;
  {
    torch::NoGradGuard noGrad;
    unsigned int       nbSample = fixedPoints.size();

    int a = 0;
    for (int i = 0; i < modelConfig.size(); ++i)
    {
      const auto & config = modelConfig[i];

      std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
      sizes[0] = nbSample;

      torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, itk::GetModelDtype(config))
                                         .unsqueeze(0)
                                         .expand(sizes)
                                         .unsqueeze(1)
                                         .clone();

      for (unsigned int s = 0; s < nbSample; ++s)
      {
        patchValueTensor[s] =
          imagesPatchValuesEvaluator(fixedPoints[s], patchIndex[i][s], config.GetPatchSize()).to(itk::GetModelDtype(config));
      }

      std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
      resizeVector[1] = config.GetNumberOfChannels();
      std::vector<torch::jit::IValue> outputsList =
        itk::Forward(config, patchValueTensor.to(device).repeat({ torch::IntArrayRef(resizeVector) }).clone());

      for (int it = 0; it < outputsList.size(); ++it)
      {
        if (config.GetLayersMask()[it])
        {
          outputsTensor.push_back(outputsList[it]
                                    .toTensor()
                                    .index(itk::GetCentersIndexLayers(config)[a])
                                    .index_select(1, subsetsOfFeatures[a])
                                    .to(torch::kFloat32));
          a++;
        }
      }
    }
  }
  return outputsTensor;
} // end GenerateOutputs

template <typename ImagePointType>
std::vector<torch::Tensor>
GenerateOutputsAndJacobian(const std::vector<itk::ModelConfiguration> &                modelConfig,
                           const std::vector<ImagePointType> &                               fixedPoints,
                           const std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex,
                           std::vector<torch::Tensor>                                        subsetsOfFeatures,
                           std::vector<torch::Tensor>                                        fixedOutputsTensor,
                           torch::Device                                                     device,
                           std::vector<std::unique_ptr<itk::Impact::Loss>> &                  losses,
                           const ImagesPatchValuesAndJacobiansEvaluator<ImagePointType> &
                             imagesPatchValuesAndJacobiansEvaluator)
{
  std::vector<torch::Tensor> layersJacobian;

  unsigned int nbSample = fixedPoints.size();
  unsigned int dimension = fixedPoints[0].size();

  int a = 0;
  for (int i = 0; i < modelConfig.size(); ++i)
  {
    const auto & config = modelConfig[i];

    std::vector<int64_t> sizes(config.GetPatchSize().size() + 1, -1);
    sizes[0] = nbSample;

    torch::Tensor patchValueTensor = torch::zeros({ torch::IntArrayRef(config.GetPatchSize()) }, itk::GetModelDtype(config))
                                       .unsqueeze(0)
                                       .expand(sizes)
                                       .unsqueeze(1)
                                       .clone();
    torch::Tensor imagesPatchesJacobians =
      torch::zeros({ nbSample, static_cast<int64_t>(patchIndex[i][0].size()), dimension }, torch::kFloat32);

    for (unsigned int s = 0; s < nbSample; ++s)
    {
      patchValueTensor[s] = imagesPatchValuesAndJacobiansEvaluator(
                              fixedPoints[s], imagesPatchesJacobians, patchIndex[i][s], config.GetPatchSize(), s)
                              .to(itk::GetModelDtype(config));
    }


    std::vector<int64_t> resizeVector(patchValueTensor.dim(), 1);
    resizeVector[1] = config.GetNumberOfChannels();
    patchValueTensor =
      patchValueTensor.to(device).repeat({ torch::IntArrayRef(resizeVector) }).clone().set_requires_grad(true);
    imagesPatchesJacobians = imagesPatchesJacobians.to(device).repeat({ 1, config.GetNumberOfChannels(), 1 }).clone();

    std::vector<torch::jit::IValue> outputsList = itk::Forward(config, patchValueTensor);
    torch::Tensor                   layer, diffLayer, modelJacobian;
    for (int it = 0; it < outputsList.size(); ++it)
    {
      if (config.GetLayersMask()[it])
      {
        int nb = std::accumulate(config.GetLayersMask().begin(), config.GetLayersMask().end(), 0);

        layer = outputsList[it]
                  .toTensor()
                  .index(itk::GetCentersIndexLayers(config)[a])
                  .index_select(1, subsetsOfFeatures[a])
                  .to(torch::kFloat32);
        torch::Tensor gradientModulator = losses[a]->updateValueAndGetGradientModulator(fixedOutputsTensor[a], layer);
        std::vector<torch::Tensor> modelJacobians;
        layersJacobian.push_back(
          torch::bmm(torch::autograd::grad({ layer }, { patchValueTensor }, { gradientModulator }, nb > 1, false)[0]
                       .flatten(1)
                       .unsqueeze(1)
                       .to(torch::kFloat32),
                     imagesPatchesJacobians));

        a++;
      }
    }
  }
  return layersJacobian;
} // end GenerateOutputsAndJacobian


} // namespace Impact
} // namespace itk

#endif // end #ifndef itkImpactOnlineInference_h

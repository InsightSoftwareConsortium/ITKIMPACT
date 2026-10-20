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

// Framework-neutral online ("Jacobian") evaluation shared one-way (Elastix -> ITKIMPACT): patch
// offsets, batched TorchScript forward, center extraction and the autograd value+derivative path
// both registration metrics run (EvaluateOnline). Depends only on the backend
// ImpactModelConfiguration (+ its torch accessors) and the IMPACT losses, talking to the host
// metric through an OnlineHost that cuts its patches. Pulls in LibTorch; never part of the
// castxml-parsed public surface.

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
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace itk
{
namespace Impact
{

/** \class OneOpenMPThread
 * \ingroup Impact
 * \brief While it lives, the calling thread runs its LibTorch operations on one OpenMP thread.
 *
 * Only when \p shared, i.e. when other threads share the work out with the calling thread; on as
 * many as before once it is gone. LibTorch spreads an operation over as many OpenMP threads as the
 * calling thread allows, and threads that already share the work out each ran a team as large as
 * the machine: an online evaluation of the v4 metric took three times as long. A thread that works
 * alone keeps them all. LibTorch's native backend has one pool for the process instead, whose size
 * cannot change once it runs: there this does nothing. */
class OneOpenMPThread
{
public:
  explicit OneOpenMPThread(bool shared)
    : m_Previous(shared && AT_PARALLEL_OPENMP ? at::get_num_threads() : 1)
  {
    if (m_Previous != 1)
    {
      at::set_num_threads(1);
    }
  }
  ~OneOpenMPThread()
  {
    if (m_Previous != 1)
    {
      at::set_num_threads(m_Previous);
    }
  }
  OneOpenMPThread(const OneOpenMPThread &) = delete;
  OneOpenMPThread &
  operator=(const OneOpenMPThread &) = delete;

private:
  const int m_Previous;
};


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

/** A batch of points as the host of an online evaluation cuts them, on the CPU: per model, the fixed and moving
 * patches ([B, nVox], image axis 0 fastest, see PatchTensorShape) and the moving image's spatial gradient at every
 * patch voxel ([B, nVox, MovingDim]); the moving transform's Jacobian ([B, MovingDim, P]) and the global parameter
 * index of each of its columns ([B, P]). The gradients and the Jacobian only with a derivative; the fixed patches
 * undefined when the host keeps the fixed features of these points (OnlineHost::KeptFixedFeatures). */
struct OnlineBatch
{
  std::vector<torch::Tensor> fixedPatches;
  std::vector<torch::Tensor> movingPatches;
  std::vector<torch::Tensor> movingGradients;
  torch::Tensor              transformJacobian;
  torch::Tensor              nonZeroJacobianIndices;
};

/** Code built against several versions of this header (benchmarks/ab.sh) tests for OnlineHost with this. */
#define ITK_IMPACT_HAS_ONLINE_HOST 1

/** \class OnlineHost
 * \ingroup Impact
 * \brief What EvaluateOnline asks of the metric it evaluates: its points, by row, and its threads. */
class OnlineHost
{
public:
  virtual ~OnlineHost() = default;
  /** The points [begin, end), cut over all the host's threads when `parallel` (a thread already running its own share
   * leaves it off): the fixed patches with `fixed`, the gradients and the Jacobian with `derivative`. */
  virtual OnlineBatch
  Sample(int64_t begin, int64_t end, bool parallel, bool derivative, bool fixed) const = 0;
  /** Run `body(part)` for every part in [0, parts), on the host's threads. */
  virtual void
  Parallelize(int64_t parts, const std::function<void(int64_t)> & body) const = 0;
  /** The fixed features, [B, C] per comparison on the CPU, of the points [begin, end) of a batch the host cut without
   * fixed patches. */
  virtual std::vector<torch::Tensor>
  KeptFixedFeatures(int64_t, int64_t) const
  {
    return {};
  }
  /** The fixed features of the points [begin, end), computed from their patches, for a host that keeps them. */
  virtual void
  KeepFixedFeatures(int64_t, int64_t, const std::vector<torch::Tensor> &) const
  {}
};

/** The models and settings of an online evaluation. Both images are cut to the fixed configurations' patches. */
struct OnlineSettings
{
  const std::vector<ImpactModelConfiguration> & fixedConfigurations;
  const std::vector<ImpactModelConfiguration> & movingConfigurations;
  /** Per comparison (kept layer, in model then mask order), the channels compared, on the device. */
  const std::vector<torch::Tensor> & subsets;
  torch::Device                      device;
  /** Points per forward pass: on a GPU capped by the device budget (itkImpactBatchBudget.h), 0 = the budget alone; on
   * the CPU 0 = 64. */
  int64_t batchSize;
  /** How many parts the CPU splits the points into, one per thread, each with its own losses. */
  int64_t workUnits;
  /** The images' metadata the models take; null: each configuration's own. */
  const ImpactImageMetadata * fixedMetadata = nullptr;
  const ImpactImageMetadata * movingMetadata = nullptr;
};

/** A batch of patch values, [B, nVox], as model `model` of the fixed or the moving image takes them: [B, C, s...] on
 * the device. PatchTensorShape is the one place that knows the spatial extents are reversed; the single intensity
 * channel is repeated to the channel count the model expects. */
inline torch::Tensor
OnlineModelPatches(const OnlineSettings & settings, const torch::Tensor & values, size_t model, bool moving)
{
  const ImpactModelConfiguration & fixedConfig = settings.fixedConfigurations[model];
  const ImpactModelConfiguration & config = moving ? settings.movingConfigurations[model] : fixedConfig;
  const std::vector<int64_t>       spatial = PatchTensorShape(fixedConfig);
  std::vector<int64_t>             shape{ values.size(0), 1 };
  shape.insert(shape.end(), spatial.begin(), spatial.end());
  std::vector<int64_t> channels(shape.size(), 1);
  channels[1] = static_cast<int64_t>(fixedConfig.GetNumberOfChannels());
  return values.reshape(shape).repeat(channels).to(settings.device).to(GetModelDtype(config));
}

/** Model `model` of the fixed or the moving image run on `patches`: the central feature vector of every kept layer,
 * normalized as the configuration says, at the channels its comparison samples, [B, C] each on the device.
 * `comparison` is the model's first. */
inline std::vector<torch::Tensor>
OnlineFeatures(const OnlineSettings & settings,
               size_t                 model,
               bool                   moving,
               const torch::Tensor &  patches,
               size_t                 comparison)
{
  const ImpactModelConfiguration & config =
    moving ? settings.movingConfigurations[model] : settings.fixedConfigurations[model];
  const ImpactImageMetadata *     metadata = moving ? settings.movingMetadata : settings.fixedMetadata;
  std::vector<torch::jit::IValue> outputs = metadata ? Forward(config, patches, *metadata) : Forward(config, patches);
  const std::vector<bool> &       mask = config.GetLayersMask();
  std::vector<torch::Tensor>      features;
  for (size_t it = 0; it < mask.size(); ++it)
  {
    if (!mask[it])
    {
      continue;
    }
    torch::Tensor layer = outputs[it].toTensor(); // [B, C, s...] -> [B, C], the patch's central voxel
    while (layer.dim() > 2)
    {
      layer = layer.select(2, layer.size(2) / 2);
    }
    features.push_back(NormalizeFeatureChannels(layer.to(torch::kFloat32), config.GetFeatureNormalization(), 1)
                         .index_select(1, settings.subsets[comparison++]));
  }
  return features;
}

/** d(loss)/d(parameter) of a batch at every comparison, [B, 1, P] each on the CPU, to be added at the batch's
 * nonZeroJacobianIndices. The moving models run with gradient, and each comparison's backward pass is seeded with
 * `modulator(comparison, movingFeatures)`: d(loss)/d(feature) at the batch's points, [B, C]. */
inline std::vector<torch::Tensor>
OnlineBackpropagate(const OnlineSettings &                                        settings,
                    const OnlineBatch &                                           batch,
                    const std::function<torch::Tensor(size_t, torch::Tensor &)> & modulator)
{
  std::vector<torch::Tensor> jacobians; // one per comparison, in order
  for (size_t i = 0; i < batch.movingPatches.size(); ++i)
  {
    // The patch values are the autograd leaf: differentiating the features with respect to them and contracting with
    // the moving image's spatial gradients gives d(feature)/d(moving coordinate) for the whole batch at once. One
    // forward pass serves every kept layer, so the graph has to survive until the last of them is done.
    const torch::Tensor patches =
      OnlineModelPatches(settings, batch.movingPatches[i], i, true).detach().set_requires_grad(true);
    const std::vector<torch::Tensor> layers = OnlineFeatures(settings, i, true, patches, jacobians.size());
    // d(patch value)/d(moving coordinate), tiled once per repeated channel so that it lines up with the flattened
    // gradient below: summing over the copies sums over the channel axis, which is exactly the derivative with
    // respect to the underlying intensity.
    const torch::Tensor patchGradients =
      batch.movingGradients[i].to(settings.device).repeat({ 1, patches.size(1), 1 }); // [B, C * nVox, MovingDim]
    for (size_t k = 0; k < layers.size(); ++k)
    {
      torch::Tensor       features = layers[k].detach().to(torch::kCPU);
      const torch::Tensor seed = modulator(jacobians.size(), features).to(settings.device).to(layers[k].dtype());
      const torch::Tensor gradient =
        torch::autograd::grad({ layers[k] }, { patches }, { seed }, k + 1 < layers.size(), false)[0];
      // d(.)/d(moving coordinate), [B, 1, MovingDim], and then d(.)/d(parameter), [B, 1, P].
      const torch::Tensor coordinate = torch::bmm(gradient.flatten(1).unsqueeze(1).to(torch::kFloat32), patchGradients);
      jacobians.push_back(torch::bmm(coordinate.to(torch::kCPU), batch.transformJacobian));
    }
  }
  return jacobians;
}

/** The online (Jacobian) evaluation of an IMPACT metric over `count` points: their loss value and, with
 * `computeDerivative`, its derivative, accumulated into `lossesOf(part)`, the losses of part `part`, which the host
 * then sums over the parts.
 *
 * On the CPU the threads are the parallelism: each runs the models on its own share of the points, a batch at a time,
 * on one OpenMP thread. On a GPU one batch at a time, as large as the device holds, its patches cut by all the
 * threads; a batch that runs out of memory is halved and replayed after the losses have been rewound.
 *
 * A loss is exact whatever the batches: a mean of per-point terms is known at a point from that point alone, while a
 * global statistic (NCC, OrganDice) mixes the points, and d(loss)/d(feature) at one of them is only known once all
 * have been seen. A batch that holds every point, as one on a GPU often does, sees them all; otherwise a first pass,
 * without gradient, gathers the features of every point, from which each loss takes its value and that derivative,
 * and a second carries it to the parameters.
 *
 * The host calls it, and sums the parts' losses after it, on one OpenMP thread when it has several (OneOpenMPThread):
 * a derivative tens of thousands long, as a B-spline transform's, otherwise takes a team of threads for every
 * operation, which competes with the host's own. */
inline void
EvaluateOnline(const OnlineSettings &                                                    settings,
               const OnlineHost &                                                        host,
               int64_t                                                                   count,
               bool                                                                      computeDerivative,
               const std::function<std::vector<std::unique_ptr<Loss>> &(int64_t part)> & lossesOf)
{
  if (count == 0)
  {
    return;
  }
  const torch::Device & device = settings.device;
  auto &                losses = lossesOf(0);

  // Run `evaluate(begin, end, parallel, losses)` over the points a batch at a time, `losses` being those of the part
  // that runs it. With `whole`, in one batch or not at all: false, the losses as they were, when the device does not
  // take every point at once.
  auto inBatches = [&](const auto & evaluate, bool whole = false) {
    if (!device.is_cuda())
    {
      if (whole)
      {
        return false;
      }
      const int64_t batch = settings.batchSize > 0 ? settings.batchSize : 64;
      const int64_t parts = std::max<int64_t>(1, std::min<int64_t>(count, settings.workUnits));
      host.Parallelize(parts, [&](int64_t part) {
        const OneOpenMPThread oneThread(parts > 1);
        for (int64_t begin = count * part / parts, end = count * (part + 1) / parts; begin < end; begin += batch)
        {
          evaluate(begin, std::min(begin + batch, end), false, lossesOf(part));
        }
      });
      return true;
    }
    std::lock_guard<std::mutex> lock(DeviceForwardMutex());
    for (int64_t begin = 0; begin < count;)
    {
      const int64_t bound = EffectiveBatchSize(settings.movingConfigurations);
      int64_t       batch = settings.batchSize > 0 ? settings.batchSize : count;
      batch = std::max<int64_t>(bound > 0 ? std::min(batch, bound) : batch, 1);
      const int64_t end = std::min(begin + batch, count);
      if (whole && end - begin < count)
      {
        return false;
      }
      std::vector<Loss::State> before;
      for (const auto & loss : losses)
      {
        before.push_back(loss->SaveState());
      }
      try
      {
        evaluate(begin, end, true, losses);
        begin = end;
      }
      catch (const std::exception & error)
      {
        if (!IsDeviceOutOfMemory(error) || end - begin <= 1)
        {
          throw;
        }
        for (size_t l = 0; l < losses.size(); ++l)
        {
          losses[l]->RestoreState(before[l]);
        }
        for (const ImpactModelConfiguration & config : settings.movingConfigurations)
        {
          ShrinkBatchSize(config, end - begin, device);
        }
        if (whole)
        {
          return false;
        }
      }
    }
    return true;
  };
  // The feature vectors of one image of a batch at every comparison, [B, C] each on the CPU.
  auto featuresOf = [&](const OnlineBatch & batch, bool moving) {
    torch::NoGradGuard         noGrad;
    std::vector<torch::Tensor> features;
    for (size_t i = 0; i < settings.fixedConfigurations.size(); ++i)
    {
      const torch::Tensor & values = moving ? batch.movingPatches[i] : batch.fixedPatches[i];
      for (const torch::Tensor & layer :
           OnlineFeatures(settings, i, moving, OnlineModelPatches(settings, values, i, moving), features.size()))
      {
        features.push_back(layer.to(torch::kCPU));
      }
    }
    return features;
  };
  // The fixed feature vectors of a batch: those the host keeps when it cut no fixed patch, else the fixed models'.
  auto fixedOf = [&](const OnlineBatch & batch, int64_t begin, int64_t end) {
    if (!batch.fixedPatches.front().defined())
    {
      return host.KeptFixedFeatures(begin, end);
    }
    std::vector<torch::Tensor> fixed = featuresOf(batch, false);
    host.KeepFixedFeatures(begin, end, fixed);
    return fixed;
  };

  auto singlePass = [&](int64_t begin, int64_t end, bool parallel, std::vector<std::unique_ptr<Loss>> & partLosses) {
    const OnlineBatch          batch = host.Sample(begin, end, parallel, computeDerivative, true);
    std::vector<torch::Tensor> fixed = fixedOf(batch, begin, end);
    if (!computeDerivative)
    {
      std::vector<torch::Tensor> moving = featuresOf(batch, true);
      for (size_t c = 0; c < fixed.size(); ++c)
      {
        partLosses[c]->updateValue(fixed[c], moving[c]);
      }
      return;
    }
    // The loss gives d(loss)/d(feature) at the batch's points with the batch's value, and one backward pass a
    // comparison carries it to the parameters.
    std::vector<torch::Tensor> jacobians = OnlineBackpropagate(settings, batch, [&](size_t c, torch::Tensor & moving) {
      return partLosses[c]->updateValueAndGetGradientModulator(fixed[c], moving);
    });
    torch::Tensor              indices = batch.nonZeroJacobianIndices;
    for (size_t c = 0; c < jacobians.size(); ++c)
    {
      partLosses[c]->updateDerivativeInJacobianMode(jacobians[c], indices);
    }
  };
  const bool global =
    computeDerivative && std::any_of(losses.begin(), losses.end(), [](const auto & l) { return !l->IsPerPointMean(); });
  if (!global)
  {
    inBatches(singlePass);
    return;
  }
  if (inBatches(singlePass, /*whole*/ true))
  {
    return;
  }

  std::vector<torch::Tensor> fixedFeatures, movingFeatures;
  for (const torch::Tensor & subset : settings.subsets)
  {
    fixedFeatures.push_back(torch::empty({ count, subset.numel() }, torch::kFloat32));
    movingFeatures.push_back(torch::empty({ count, subset.numel() }, torch::kFloat32));
  }
  inBatches([&](int64_t begin, int64_t end, bool parallel, std::vector<std::unique_ptr<Loss>> &) {
    const OnlineBatch                batch = host.Sample(begin, end, parallel, false, true);
    const std::vector<torch::Tensor> fixed = fixedOf(batch, begin, end);
    const std::vector<torch::Tensor> moving = featuresOf(batch, true);
    for (size_t c = 0; c < fixed.size(); ++c)
    {
      fixedFeatures[c].narrow(0, begin, end - begin).copy_(fixed[c]);
      movingFeatures[c].narrow(0, begin, end - begin).copy_(moving[c]);
    }
  });
  std::vector<torch::Tensor> modulators;
  for (size_t c = 0; c < losses.size(); ++c)
  {
    modulators.push_back(losses[c]->updateValueAndGetGradientModulator(fixedFeatures[c], movingFeatures[c]));
  }
  std::mutex accumulating; // the CPU parts all add to part 0's derivative
  inBatches([&](int64_t begin, int64_t end, bool parallel, std::vector<std::unique_ptr<Loss>> &) {
    const OnlineBatch          batch = host.Sample(begin, end, parallel, true, false);
    std::vector<torch::Tensor> jacobians = OnlineBackpropagate(
      settings, batch, [&](size_t c, torch::Tensor &) { return modulators[c].narrow(0, begin, end - begin); });
    torch::Tensor               indices = batch.nonZeroJacobianIndices;
    std::lock_guard<std::mutex> lock(accumulating);
    for (size_t c = 0; c < jacobians.size(); ++c)
    {
      losses[c]->updateDerivativeInJacobianMode(jacobians[c], indices);
    }
  });
}


} // namespace Impact
} // namespace itk

#endif // end #ifndef itkImpactOnlineInference_h

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

#ifndef itkImpactModelConfigurationDetail_h
#define itkImpactModelConfigurationDetail_h

// Internal companion to itkImpactModelConfiguration.h: pulls in LibTorch, so it is included
// ONLY by translation units that already depend on torch (the feature-extraction
// internals and the compiled itkImpactModelConfiguration.cxx). Never part of the public,
// castxml-parsed surface.

#include "itkImpactModelConfiguration.h"
#include "itkStatisticsImageFilter.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <utility>
#include <vector>

namespace itk
{
/** What a metadata-aware model (Forward with four arguments) takes for the image it works on:
 * the intensity [min, max, mean, sigma] as float32, and the direction matrix as int16. A field
 * left default-constructed stands for "none", and Forward then hands the model the empty tensor
 * its schema defaults to. */
struct ImpactImageMetadata
{
  torch::Tensor stats;
  torch::Tensor direction;
};

namespace detail
{
/** The LibTorch state held opaquely by ImpactModelConfiguration. */
struct ImpactModelConfigurationImpl
{
  std::shared_ptr<torch::jit::script::Module> model;
  torch::ScalarType                           dtype{ torch::kFloat32 };

  /** Number of positional inputs forward expects, introspected from the model schema:
   * 1 => patch only; 2 => patch + nLayers; 4 => patch + nLayers + image stats + image
   * direction. */
  std::size_t   nArgs{ 1 };
  torch::Tensor nLayers; // int16 scalar = number of requested layers
  /** The image the model is run on, as a metadata-aware model takes it; set by SetupImageMetadata
   * for the stages that run the model on one image at a time. */
  ImpactImageMetadata imageMetadata;
  /** Per-layer center-extraction indices for the online inference (set by the caller). */
  std::vector<std::vector<torch::indexing::TensorIndex>> centersIndexLayers;
};
} // namespace detail

/** Access the loaded TorchScript module of a configuration. */
inline torch::jit::script::Module &
GetModel(const ImpactModelConfiguration & configuration)
{
  return *configuration.GetImpl()->model;
}

/** The scalar type the model runs in (float32, or float16 in mixed precision). */
inline torch::ScalarType
GetModelDtype(const ImpactModelConfiguration & configuration)
{
  return configuration.GetImpl()->dtype;
}

/** Move the model to a device (e.g. CPU or CUDA). */
inline void
ModelTo(const ImpactModelConfiguration & configuration, const torch::Device & device)
{
  configuration.GetImpl()->model->to(device);
}

/** Run the model's forward, assembling positional arguments per the introspected schema:
 * always the patch; plus the requested layer count when forward takes >= 2 inputs; plus the
 * stats and direction of `metadata` when it takes >= 4. Returns the per-layer output tensors. */
inline std::vector<torch::jit::IValue>
Forward(const ImpactModelConfiguration & configuration, torch::Tensor inputPatch, const ImpactImageMetadata & metadata)
{
  detail::ImpactModelConfigurationImpl * impl = configuration.GetImpl();
  std::vector<torch::jit::IValue>  args;
  args.reserve(impl->nArgs);
  args.emplace_back(inputPatch);
  if (impl->nArgs >= 2)
  {
    args.emplace_back(impl->nLayers);
  }
  if (impl->nArgs >= 4)
  {
    // A metadata-aware model (nArgs >= 4) takes the image's intensity statistics and direction.
    // Passing an *undefined* tensor into a model that actually reads it -- e.g. SAM's input
    // normalisation does aten::size on `stats` -- aborts LibTorch's alias analysis ("no op for
    // aten::size"). Fall back to the schema's empty-tensor default so the model takes its
    // no-metadata path instead of crashing.
    const torch::Tensor empty = torch::empty({ 0 }, inputPatch.options().dtype(torch::kFloat32));
    args.emplace_back(metadata.stats.defined() ? metadata.stats : empty);
    args.emplace_back(metadata.direction.defined() ? metadata.direction : empty);
  }
  return impl->model->forward(std::move(args)).toList().vec();
}

/** Forward with the metadata SetupImageMetadata stored in the configuration. A configuration is shared
 * by every side it was added to, so this serves one image at a time; a stage that runs the same
 * model on two images in one evaluation passes each image's metadata itself. */
inline std::vector<torch::jit::IValue>
Forward(const ImpactModelConfiguration & configuration, torch::Tensor inputPatch)
{
  return Forward(configuration, std::move(inputPatch), configuration.GetImpl()->imageMetadata);
}

/** The metadata of an image, as a metadata-aware model takes it: one pass of statistics over
 * the image, and its direction rounded to integers. Computed and returned; nothing is stored. */
template <typename TImage>
ImpactImageMetadata
ComputeImageMetadata(typename TImage::ConstPointer image)
{
  auto imageStats = itk::StatisticsImageFilter<TImage>::New();
  imageStats->SetInput(image);
  imageStats->Update();
  torch::Tensor statsTensor = torch::tensor({ static_cast<float>(imageStats->GetMinimum()),
                                              static_cast<float>(imageStats->GetMaximum()),
                                              static_cast<float>(imageStats->GetMean()),
                                              static_cast<float>(imageStats->GetSigma()) },
                                            torch::kFloat32);

  constexpr unsigned int imageDimension = TImage::ImageDimension;
  const auto &           imageDirection = image->GetDirection();
  torch::Tensor          directionTensor =
    torch::empty({ static_cast<int64_t>(imageDimension), static_cast<int64_t>(imageDimension) }, torch::kInt16);
  for (unsigned int r = 0; r < imageDimension; ++r)
  {
    for (unsigned int c = 0; c < imageDimension; ++c)
    {
      directionTensor[r][c] = static_cast<int16_t>(std::llround(imageDirection(r, c)));
    }
  }
  return { statsTensor, directionTensor };
}

/** Store an image's metadata in the configuration, for the stages that run the model on one
 * image at a time: the feature map, and the coarse and fine registrations. */
template <typename TImage>
void
SetupImageMetadata(const ImpactModelConfiguration & configuration, typename TImage::ConstPointer image)
{
  configuration.GetImpl()->imageMetadata = ComputeImageMetadata<TImage>(image);
}

/** Per-layer center-extraction indices used by the online inference (read/write). */
inline const std::vector<std::vector<torch::indexing::TensorIndex>> &
GetCentersIndexLayers(const ImpactModelConfiguration & configuration)
{
  return configuration.GetImpl()->centersIndexLayers;
}
inline void
SetCentersIndexLayers(const ImpactModelConfiguration &                         configuration,
                      std::vector<std::vector<torch::indexing::TensorIndex>> & centersIndexLayers)
{
  configuration.GetImpl()->centersIndexLayers = centersIndexLayers;
}

} // end namespace itk

#endif // end #ifndef itkImpactModelConfigurationDetail_h

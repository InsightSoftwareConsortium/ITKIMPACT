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
  /** Most patches one online forward may take at once; 0 = no bound (itkImpactBatchBudget.h). */
  int64_t batchSize{ 0 };
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

/** The stats and direction of `metadata` as a metadata-aware model takes them. Passing an *undefined* tensor into a
 * model that actually reads it -- e.g. SAM's input normalisation does aten::size on `stats` -- aborts LibTorch's alias
 * analysis ("no op for aten::size"): the schema's empty-tensor default stands for a field left undefined, so that the
 * model takes its no-metadata path instead of crashing. */
inline std::pair<torch::Tensor, torch::Tensor>
MetadataArguments(const ImpactImageMetadata & metadata, const torch::Tensor & input)
{
  const torch::Tensor empty = torch::empty({ 0 }, input.options().dtype(torch::kFloat32));
  return { metadata.stats.defined() ? metadata.stats : empty,
           metadata.direction.defined() ? metadata.direction : empty };
}

namespace detail
{
/** The gradient through a half-precision network, scaled on its way in and back on its way out: a loss over many
 * voxels has a derivative with respect to each feature below float16's range, which the network's backward pass would
 * round to zero. The first output's gradient to come back, in single precision, sets the scale, a power of two that
 * brings its largest magnitude near one, held in `scale` (a double on the host, 0 until set), which every output then
 * takes and the input's gradient is divided by, in single precision. */
struct HalfGradientOut : public torch::autograd::Function<HalfGradientOut>
{
  static torch::Tensor
  forward(torch::autograd::AutogradContext * ctx, const torch::Tensor & output, const torch::Tensor & scale)
  {
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["dtype"] = static_cast<int64_t>(output.scalar_type());
    return output.to(torch::kFloat32);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext * ctx, torch::autograd::variable_list gradients)
  {
    double & scale = ctx->saved_data["scale"].toTensor().data_ptr<double>()[0];
    if (scale == 0.0)
    {
      const double peak = gradients[0].abs().max().item<double>();
      scale = peak > 0.0 && std::isfinite(peak) ? std::exp2(-std::ceil(std::log2(peak))) : 0.0;
    }
    const auto dtype = static_cast<torch::ScalarType>(ctx->saved_data["dtype"].toInt());
    return { (gradients[0] * (scale > 0.0 ? scale : 1.0)).to(dtype), torch::Tensor() };
  }
};

struct HalfGradientIn : public torch::autograd::Function<HalfGradientIn>
{
  static torch::Tensor
  forward(torch::autograd::AutogradContext * ctx,
          const torch::Tensor &              input,
          const torch::Tensor &              scale,
          torch::ScalarType                  dtype)
  {
    ctx->saved_data["scale"] = scale;
    return input.to(dtype);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext * ctx, torch::autograd::variable_list gradients)
  {
    const double scale = ctx->saved_data["scale"].toTensor().data_ptr<double>()[0];
    return { gradients[0].to(torch::kFloat32) / (scale > 0.0 ? scale : 1.0), torch::Tensor(), torch::Tensor() };
  }
};
} // namespace detail

/** Run the model's forward, assembling positional arguments per the introspected schema:
 * always the patch; plus the requested layer count when forward takes >= 2 inputs; plus the
 * stats and direction of `metadata` when it takes >= 4; plus `norms` when given, the statistics of
 * its normalisations (NormCount) the model takes in place of its input's own. Returns the per-layer
 * output tensors. The input is brought to the model's precision; a half-precision model on the graph of a
 * single-precision input returns single-precision layers whose gradient goes through it scaled (HalfGradientOut). */
inline std::vector<torch::jit::IValue>
Forward(const ImpactModelConfiguration &   configuration,
        torch::Tensor                      inputPatch,
        const ImpactImageMetadata &        metadata,
        const std::vector<torch::Tensor> & norms = {})
{
  detail::ImpactModelConfigurationImpl * impl = configuration.GetImpl();
  const bool scaled = impl->dtype == torch::kFloat16 && torch::GradMode::is_enabled() && inputPatch.requires_grad() &&
                      inputPatch.scalar_type() == torch::kFloat32;
  const torch::Tensor scale = scaled ? torch::zeros({ 1 }, torch::kFloat64) : torch::Tensor();
  inputPatch = scaled ? detail::HalfGradientIn::apply(inputPatch, scale, impl->dtype) : inputPatch.to(impl->dtype);
  std::vector<torch::jit::IValue> args;
  args.reserve(impl->nArgs);
  args.emplace_back(inputPatch);
  if (impl->nArgs >= 2)
  {
    args.emplace_back(impl->nLayers);
  }
  if (impl->nArgs >= 4)
  {
    const auto [stats, direction] = MetadataArguments(metadata, inputPatch);
    args.emplace_back(stats);
    args.emplace_back(direction);
  }
  if (!norms.empty())
  {
    args.emplace_back(c10::List<torch::Tensor>(norms));
  }
  std::vector<torch::jit::IValue> outputs = impl->model->forward(std::move(args)).toList().vec();
  for (size_t i = 0; scaled && i < outputs.size(); ++i)
  {
    outputs[i] = detail::HalfGradientOut::apply(outputs[i].toTensor(), scale);
  }
  return outputs;
}

/** How many normalisations by its input's own statistics (TS's instance normalisation, MIND's bound on its
 * variance) the model applies up to its last requested layer: what its norm_count method says, 0 for a model without
 * one, which then always normalises by its input's own. */
inline int64_t
NormCount(const ImpactModelConfiguration & configuration)
{
  torch::jit::script::Module & model = GetModel(configuration);
  if (!model.find_method("norm_count"))
  {
    return 0;
  }
  return model.run_method("norm_count", configuration.GetImpl()->nLayers.item<int64_t>()).toInt();
}

/** The input of the model's normalisation number norms.size(), on the image's axes, those before it normalised by
 * `norms` (one [2, C] tensor each: the mean and the biased variance of its input). */
inline torch::Tensor
NormInput(const ImpactModelConfiguration &   configuration,
          const torch::Tensor &              input,
          const ImpactImageMetadata &        metadata,
          const std::vector<torch::Tensor> & norms)
{
  const auto [stats, direction] = MetadataArguments(metadata, input);
  return GetModel(configuration)
    .run_method("norm_input", input.to(GetModelDtype(configuration)), stats, direction, c10::List<torch::Tensor>(norms))
    .toTensor();
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

/** Most patches one online forward may take at once (0 = unbounded); see itkImpactBatchBudget.h. */
inline int64_t
GetBatchSize(const ImpactModelConfiguration & configuration)
{
  return configuration.GetImpl()->batchSize;
}
inline void
SetBatchSize(const ImpactModelConfiguration & configuration, int64_t batchSize)
{
  configuration.GetImpl()->batchSize = batchSize < 0 ? 0 : batchSize;
}

} // end namespace itk

#endif // end #ifndef itkImpactModelConfigurationDetail_h

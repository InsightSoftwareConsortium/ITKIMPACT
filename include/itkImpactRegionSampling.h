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
#ifndef itkImpactRegionSampling_h
#define itkImpactRegionSampling_h

// The pieces of ImpactFineRegistration's region-sampled (out-of-core) run that do not depend on the filter:
// reading a region through an image's own pipeline, the cell lattice its boxes tile, the pooling windows of the
// cells, and Adam on the cells an iteration touched. LibTorch-backed: included by the .hxx only.

#include "itkImpactModelConfiguration.h"
#include "itkImpactTorchRegistrationHelpers.h"

#include <itkImageRegion.h>
#include <itkImageRegionConstIterator.h>
#include <itkMacro.h>

#include <torch/torch.h>
#if __has_include(<c10/cuda/CUDAGuard.h>) && __has_include(<cuda_runtime_api.h>)
#  include <c10/cuda/CUDAGuard.h>
#  include <cuda_runtime_api.h>
#  define ITK_IMPACT_REGION_HAS_CUDA 1
#else
#  define ITK_IMPACT_REGION_HAS_CUDA 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#if !defined(_WIN32)
#  include <unistd.h>
#endif

namespace itk
{
namespace Impact
{

/** The windows PyTorch's adaptive average pooling takes an axis of `length` voxels into `cells` cells by: cell i
 * covers the voxels [floor(i * length / cells), ceil((i + 1) * length / cells)). Pooling a region of whole cells
 * with these windows gives the same cells as pooling the whole axis. */
struct PoolWindows
{
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
};

inline PoolWindows
AdaptivePoolWindows(int64_t length, int64_t cells)
{
  PoolWindows windows;
  windows.starts.resize(static_cast<size_t>(cells));
  windows.ends.resize(static_cast<size_t>(cells));
  for (int64_t i = 0; i < cells; ++i)
  {
    windows.starts[i] = (i * length) / cells;
    windows.ends[i] = ((i + 1) * length + cells - 1) / cells;
  }
  return windows;
}

/** `map` ({1, C, spatial...}) averaged over the windows `starts`/`ends` of each tensor axis, given in the map's own
 * voxel indices: {1, C, cells...}. Windows that are one stride wide and contiguous take an average pool, the others
 * cumulative sums along the axis. */
template <unsigned int Dim>
torch::Tensor
PoolByWindows(const torch::Tensor & map, const std::vector<PoolWindows> & windows)
{
  namespace F = torch::nn::functional;
  bool                 uniform = true;
  std::vector<int64_t> kernel(Dim);
  for (unsigned int d = 0; d < Dim && uniform; ++d)
  {
    const PoolWindows & axis = windows[d];
    kernel[d] = axis.ends[0] - axis.starts[0];
    uniform =
      axis.starts[0] == 0 && kernel[d] > 0 && map.size(2 + d) == kernel[d] * static_cast<int64_t>(axis.ends.size());
    for (size_t i = 0; i < axis.ends.size() && uniform; ++i)
    {
      uniform = axis.starts[i] == static_cast<int64_t>(i) * kernel[d] && axis.ends[i] == axis.starts[i] + kernel[d];
    }
  }
  if (uniform)
  {
    if constexpr (Dim == 3)
      return F::avg_pool3d(map, F::AvgPool3dFuncOptions({ kernel[0], kernel[1], kernel[2] }));
    else
      return F::avg_pool2d(map, F::AvgPool2dFuncOptions({ kernel[0], kernel[1] }));
  }
  torch::Tensor pooled = map;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    const int64_t       axis = 2 + static_cast<int64_t>(d);
    const PoolWindows & along = windows[d];
    // Inclusive prefix sums with a leading zero: the sum over [s, e) is prefix[e] - prefix[s].
    std::vector<int64_t> zeroShape(pooled.sizes().begin(), pooled.sizes().end());
    zeroShape[axis] = 1;
    const torch::Tensor prefix =
      torch::cat({ torch::zeros(zeroShape, pooled.options()), torch::cumsum(pooled, axis) }, axis);
    const auto           index = torch::TensorOptions().dtype(torch::kLong).device(pooled.device());
    const torch::Tensor  ends = torch::tensor(along.ends, index);
    const torch::Tensor  starts = torch::tensor(along.starts, index);
    std::vector<int64_t> lengthShape(pooled.dim(), 1);
    lengthShape[axis] = static_cast<int64_t>(along.ends.size());
    const torch::Tensor length = (ends - starts).to(pooled.dtype()).view(lengthShape);
    pooled = (prefix.index_select(axis, ends) - prefix.index_select(axis, starts)) / length;
  }
  return pooled;
}

/** The region `region` of `image`, read through the image's own pipeline (SetRequestedRegion then Update: a
 * streaming reader reads that region alone, an image in memory serves it from its buffer), as a {1, 1, spatial...}
 * float tensor on the host, in tensor axis order. */
template <typename TImage>
torch::Tensor
ReadRegion(const TImage * image, const typename TImage::RegionType & region)
{
  constexpr unsigned int Dim = TImage::ImageDimension;
  auto *                 source = const_cast<TImage *>(image);
  source->SetRequestedRegion(region);
  source->Update();
  std::vector<int64_t> shape(Dim);
  int64_t              voxels = 1;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    shape[d] = static_cast<int64_t>(region.GetSize()[Dim - 1 - d]);
    voxels *= shape[d];
  }
  std::vector<float>               buffer(static_cast<size_t>(voxels));
  ImageRegionConstIterator<TImage> it(image, region);
  size_t                           i = 0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) // raster order, ITK x fastest: matches the {z, y, x} tensor
  {
    buffer[i++] = static_cast<float>(it.Get());
  }
  return torch::from_blob(buffer.data(), shape, torch::kFloat32).clone().unsqueeze(0).unsqueeze(0);
}

/** The windows of `cells` cells of `voxelsPerCell` voxels each from the first voxel: the coarse stage's cells, the
 * tail voxels past the last one dropped. */
inline PoolWindows
UniformPoolWindows(int64_t cells, int64_t voxelsPerCell)
{
  PoolWindows windows;
  for (int64_t i = 0; i < cells; ++i)
  {
    windows.starts.push_back(i * voxelsPerCell);
    windows.ends.push_back((i + 1) * voxelsPerCell);
  }
  return windows;
}

/** The windows of the cells [c0, c1), relative to their span. */
inline std::vector<PoolWindows>
RelativeWindows(const std::vector<PoolWindows> & windows, const std::vector<int64_t> & c0, const std::vector<int64_t> & c1)
{
  std::vector<PoolWindows> relative(windows.size());
  for (size_t d = 0; d < windows.size(); ++d)
  {
    const int64_t base = windows[d].starts[c0[d]];
    for (int64_t c = c0[d]; c < c1[d]; ++c)
    {
      relative[d].starts.push_back(windows[d].starts[c] - base);
      relative[d].ends.push_back(windows[d].ends[c] - base);
    }
  }
  return relative;
}

/** Whether two images share one grid: size, spacing, origin and direction. */
template <typename TA, typename TB>
bool
SameGrid(const TA * a, const TB * b)
{
  return a->GetLargestPossibleRegion().GetSize() == b->GetLargestPossibleRegion().GetSize() &&
         a->GetSpacing() == b->GetSpacing() && a->GetOrigin() == b->GetOrigin() &&
         a->GetDirection() == b->GetDirection();
}

/** The region of a grid (`spatial` in tensor order, `index` its first index) that the cells [c0, c1) of `windows`
 * span, `extra` voxels around on each tensor axis, clipped to the grid: `lowPad` what the clipping left of the margin
 * on the low side, `core` the cells' own extent. */
template <typename TRegion>
TRegion
CellSpan(const std::vector<int64_t> &     c0,
         const std::vector<int64_t> &     c1,
         const std::vector<int64_t> &     extra,
         const std::vector<PoolWindows> & windows,
         const std::vector<int64_t> &     spatial,
         const typename TRegion::IndexType & index,
         std::vector<int64_t> &           lowPad,
         std::vector<int64_t> &           core)
{
  constexpr unsigned int Dim = TRegion::ImageDimension;
  typename TRegion::IndexType start;
  typename TRegion::SizeType  size;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    const int64_t first = windows[d].starts[c0[d]];
    const int64_t end = windows[d].ends[c1[d] - 1];
    const int64_t low = std::max<int64_t>(0, first - extra[d]);
    const int64_t high = std::min<int64_t>(spatial[d], end + extra[d]);
    start[Dim - 1 - d] = static_cast<IndexValueType>(low) + index[Dim - 1 - d];
    size[Dim - 1 - d] = static_cast<SizeValueType>(high - low);
    lowPad[d] = first - low;
    core[d] = end - first;
  }
  return TRegion(start, size);
}

/** `t` ({1, C, span...}) narrowed to the cells' own extent: `lowPad` dropped on the low side, `core` kept. */
inline torch::Tensor
CropCore(torch::Tensor t, const std::vector<int64_t> & lowPad, const std::vector<int64_t> & core)
{
  for (size_t d = 0; d < core.size(); ++d)
  {
    t = t.narrow(2 + static_cast<int64_t>(d), lowPad[d], core[d]);
  }
  return t;
}

/** `t` narrowed to [lo, hi) on each spatial axis, the first of them at `firstAxis`. */
inline torch::Tensor
NarrowTo(torch::Tensor t, const std::vector<int64_t> & lo, const std::vector<int64_t> & hi, int64_t firstAxis)
{
  for (size_t d = 0; d < lo.size(); ++d)
  {
    t = t.narrow(firstAxis + static_cast<int64_t>(d), lo[d], hi[d] - lo[d]);
  }
  return t;
}

/** `t` ({1, C, n...}) written over the cells [c0, c1) of `whole` ({1, C, cells...}). */
inline void
WriteCells(torch::Tensor & whole, const std::vector<int64_t> & c0, const torch::Tensor & t)
{
  torch::Tensor view = whole;
  for (size_t d = 0; d < c0.size(); ++d)
  {
    view = view.narrow(2 + static_cast<int64_t>(d), c0[d], t.size(2 + static_cast<int64_t>(d)));
  }
  view.copy_(t);
}

/** What the feature models of a region-sampled run must be: run at the image's resolution (no voxel size of
 * their own, whose lattice a region does not line up with) and of the image's dimension. The kept layers over the
 * models, and whether one reads the image's statistics. */
struct RegionModels
{
  size_t layerCount{ 0 };
  bool   takesMetadata{ false };
};

template <typename TSpacing>
RegionModels
CheckRegionModels(const std::vector<ImpactModelConfiguration> & configs,
                  const TSpacing &                              spacing,
                  unsigned int                                  dim,
                  const char *                                  who)
{
  RegionModels models;
  for (const ImpactModelConfiguration & config : configs)
  {
    models.layerCount += NumberOfKeptLayers(config);
    const std::vector<float> & voxel = config.GetVoxelSize();
    for (unsigned int a = 0; a < dim && voxel.size() == dim; ++a)
    {
      if (voxel[a] > 0.0f && std::abs(voxel[a] - spacing[a]) > 1e-6 * spacing[a])
      {
        itkGenericExceptionMacro(<< who << ": " << config.GetModelPath()
                                 << " resamples the image to its voxel size, whose lattice a region does not line up "
                                    "with; give it the image's spacing, or no voxel size, with RegionSize.");
      }
    }
    if (config.GetDimension() != dim)
    {
      itkGenericExceptionMacro(<< who << ": " << config.GetModelPath() << " is a " << config.GetDimension()
                               << "D model; RegionSize runs models of the image's dimension.");
    }
    models.takesMetadata = models.takesMetadata || config.GetImpl()->nArgs >= 4;
  }
  return models;
}

/** The image's statistics (minimum, maximum, mean, sigma, as StatisticsImageFilter takes them) accumulated over
 * the blocks `readBlocks` hands over (it calls its argument with each block's tensor), with `direction`: what a
 * metadata-aware model (MIND) reads. */
template <typename TDirection, typename TReadBlocks>
ImpactImageMetadata
MetadataByBlocks(const TDirection & direction, unsigned int dim, TReadBlocks && readBlocks)
{
  double  minimum = std::numeric_limits<double>::infinity(), maximum = -minimum, sum = 0.0, sumOfSquares = 0.0;
  int64_t count = 0;
  readBlocks([&](const torch::Tensor & block) {
    const torch::Tensor values = block.to(torch::kFloat64);
    minimum = std::min(minimum, values.min().template item<double>());
    maximum = std::max(maximum, values.max().template item<double>());
    sum += values.sum().template item<double>();
    sumOfSquares += (values * values).sum().template item<double>();
    count += values.numel();
  });
  const double mean = sum / static_cast<double>(std::max<int64_t>(count, 1));
  const double variance =
    count > 1 ? std::max(0.0, (sumOfSquares - sum * sum / static_cast<double>(count)) / static_cast<double>(count - 1))
              : 0.0;
  ImpactImageMetadata metadata;
  metadata.stats = torch::tensor({ static_cast<float>(minimum),
                                   static_cast<float>(maximum),
                                   static_cast<float>(mean),
                                   static_cast<float>(std::sqrt(variance)) },
                                 torch::kFloat32);
  metadata.direction = torch::empty({ static_cast<int64_t>(dim), static_cast<int64_t>(dim) }, torch::kInt16);
  for (unsigned int r = 0; r < dim; ++r)
  {
    for (unsigned int c = 0; c < dim; ++c)
    {
      metadata.direction[r][c] = static_cast<int16_t>(std::llround(direction(r, c)));
    }
  }
  return metadata;
}

/** The kept feature layers of an image region ({1, 1, span...}) at its resolution, each model run on the region
 * whole in one forward, `metadata` (when given) set on the configurations first; the region itself with no model.
 * `layerCount` is what the caller keeps, checked. */
template <unsigned int Dim>
std::vector<torch::Tensor>
RegionLayers(const std::vector<ImpactModelConfiguration> & configs,
             const torch::Tensor &                         image,
             const torch::Device &                         device,
             size_t                                        layerCount,
             const ImpactImageMetadata *                   metadata,
             const char *                                  who)
{
  if (configs.empty())
  {
    return { image };
  }
  std::vector<torch::Tensor> layers;
  torch::NoGradGuard         noGrad;
  for (const ImpactModelConfiguration & config : configs)
  {
    if (metadata != nullptr)
    {
      config.GetImpl()->imageMetadata = *metadata;
    }
    ModelTo(config, device);
    torch::Tensor input = image.to(GetModelDtype(config));
    if (config.GetNumberOfChannels() > 1)
    {
      std::vector<int64_t> repeats(Dim + 2, 1);
      repeats[1] = static_cast<int64_t>(config.GetNumberOfChannels());
      input = input.repeat(repeats);
    }
    const std::vector<torch::jit::IValue> outputs = Forward(config, input);
    const std::vector<bool> &             mask = config.GetLayersMask();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
      if (i < mask.size() && !mask[i])
      {
        continue;
      }
      const torch::Tensor layer =
        NormalizeFeatureChannels(outputs[i].toTensor().to(torch::kFloat32), config.GetFeatureNormalization(), 1);
      if (!std::equal(layer.sizes().begin() + 2, layer.sizes().end(), image.sizes().begin() + 2))
      {
        itkGenericExceptionMacro(<< who << ": a feature layer of another resolution than the image (" << layer.sizes()
                                 << " for " << image.sizes()
                                 << ") cannot be computed by regions, its lattice not lining up with the image's; "
                                    "RegionSize takes layers at the image's resolution (MIND).");
      }
      layers.push_back(layer.contiguous());
    }
  }
  if (layers.size() != layerCount)
  {
    itkGenericExceptionMacro(<< who << ": the models produced " << layers.size() << " feature layers where "
                             << layerCount << " are kept.");
  }
  return layers;
}

/** The boxes of whole cells tiling a lattice of `cells` cells by `perBox` cells a box, in tensor axis order; the
 * last box along an axis takes what is left. */
struct CellBox
{
  std::vector<int64_t> cell0;
  std::vector<int64_t> cell1;
};

inline std::vector<CellBox>
TileCells(const std::vector<int64_t> & cells, const std::vector<int64_t> & perBox)
{
  const size_t         dim = cells.size();
  std::vector<int64_t> count(dim);
  size_t               total = 1;
  for (size_t d = 0; d < dim; ++d)
  {
    count[d] = (cells[d] + perBox[d] - 1) / perBox[d];
    total *= static_cast<size_t>(count[d]);
  }
  std::vector<CellBox> boxes;
  boxes.reserve(total);
  for (size_t flat = 0; flat < total; ++flat)
  {
    CellBox box;
    box.cell0.resize(dim);
    box.cell1.resize(dim);
    size_t rest = flat;
    for (size_t d = dim; d-- > 0;) // the last tensor axis (ITK x) runs fastest
    {
      const int64_t i = static_cast<int64_t>(rest % static_cast<size_t>(count[d]));
      rest /= static_cast<size_t>(count[d]);
      box.cell0[d] = i * perBox[d];
      box.cell1[d] = std::min(cells[d], box.cell0[d] + perBox[d]);
    }
    boxes.push_back(std::move(box));
  }
  return boxes;
}

/** Adam (torch::optim::Adam's update, no weight decay) on the cells an iteration touched: the moments and the step
 * count of every other cell stay as they are, as if the cell had not been optimized that step. Touching every cell at
 * every step is torch::optim::Adam. `touched` is a bool {1, 1, cells...} broadcast over the components. */
struct LazyAdam
{
  torch::Tensor exponentialAverage;
  torch::Tensor exponentialAverageSquare;
  torch::Tensor steps;
  double        learningRate;
  double        beta1;
  double        beta2;
  double        epsilon;

  LazyAdam(const torch::Tensor & theta, double lr, double b1, double b2, double eps)
    : exponentialAverage(torch::zeros_like(theta))
    , exponentialAverageSquare(torch::zeros_like(theta))
    , learningRate(lr)
    , beta1(b1)
    , beta2(b2)
    , epsilon(eps)
  {
    std::vector<int64_t> shape(theta.sizes().begin(), theta.sizes().end());
    shape[1] = 1;
    steps = torch::zeros(shape, theta.options());
  }

  void
  Step(torch::Tensor & theta, const torch::Tensor & touched)
  {
    torch::NoGradGuard  noGrad;
    const torch::Tensor gradient = theta.grad().defined() ? theta.grad() : torch::zeros_like(theta);
    steps = torch::where(touched, steps + 1.0, steps);
    exponentialAverage =
      torch::where(touched, beta1 * exponentialAverage + (1.0 - beta1) * gradient, exponentialAverage);
    exponentialAverageSquare = torch::where(
      touched, beta2 * exponentialAverageSquare + (1.0 - beta2) * gradient * gradient, exponentialAverageSquare);
    // The bias corrections of a cell never touched are 0: kept out of the division by the mask.
    const torch::Tensor correction1 = (1.0 - torch::pow(beta1, steps)).clamp_min(1e-30);
    const torch::Tensor correction2 = (1.0 - torch::pow(beta2, steps)).clamp_min(1e-30);
    const torch::Tensor denominator = exponentialAverageSquare.sqrt() / correction2.sqrt() + epsilon;
    const torch::Tensor update = (learningRate / correction1) * exponentialAverage / denominator;
    theta.copy_(torch::where(touched, theta - update, theta));
    if (theta.grad().defined())
    {
      theta.grad().zero_();
    }
  }
};

/** What the pooled feature maps of a region-sampled run may take to stay resident on `device`, when the caller sets
 * no budget: half the device memory free now, or a quarter of the machine's memory on the CPU. */
inline double
ResidentBudget(const torch::Device & device)
{
#if ITK_IMPACT_REGION_HAS_CUDA
  if (device.is_cuda())
  {
    c10::cuda::CUDAGuard guard(device.index());
    size_t               freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess)
    {
      return static_cast<double>(freeBytes) * 0.5;
    }
  }
#endif
  if (device.is_cuda())
  {
    return 0.0; // the device memory could not be read: nothing stays resident
  }
#if defined(_WIN32)
  return 4.0 * 1024.0 * 1024.0 * 1024.0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES), pageSize = sysconf(_SC_PAGE_SIZE);
  return pages > 0 && pageSize > 0 ? static_cast<double>(pages) * static_cast<double>(pageSize) * 0.25
                                   : 4.0 * 1024.0 * 1024.0 * 1024.0;
#endif
}

} // namespace Impact
} // namespace itk

#endif // itkImpactRegionSampling_h

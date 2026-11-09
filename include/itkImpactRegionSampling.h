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

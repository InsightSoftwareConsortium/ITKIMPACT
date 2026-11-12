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
// cells, Adam stepped by parts of the cell grid, and the mapped files of what grows with the image. LibTorch-backed:
// included by the .hxx only.

#include "itkImpactBatchBudget.h" // whether the build reads the memory of a device
#include "itkImpactModelConfiguration.h"
#include "itkImpactTorchRegistrationHelpers.h"

#include <itkImageRegion.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImportImageContainer.h>
#include <itkMacro.h>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#include <vector>
#if !defined(_WIN32)
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <cerrno>
#  include <cstring>
#  include <fcntl.h>
#endif

namespace itk
{
namespace Impact
{

/** The voxels each cell of an axis pools: cell i covers [starts[i], ends[i]). */
struct PoolWindows
{
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
};

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

/** ReadRegion on `device`, `bytes` counting what is asked of the image. `streamed` tells whether `source`, the image
 * the read goes back to, served the region without buffering itself whole. */
template <typename TImage, typename TSource>
torch::Tensor
ReadRegionCounted(const TImage *                      image,
                  const typename TImage::RegionType & region,
                  const torch::Device &               device,
                  SizeValueType &                     bytes,
                  const TSource *                     source,
                  bool &                              streamed)
{
  bytes += region.GetNumberOfPixels() * sizeof(typename TImage::PixelType);
  const torch::Tensor tensor = ReadRegion(image, region);
  streamed = source->GetBufferedRegion().GetNumberOfPixels() < source->GetLargestPossibleRegion().GetNumberOfPixels();
  return tensor.to(device);
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

/** Whether two images share one grid: region (a box of one is read from the other by its indices), spacing, origin
 * and direction. */
template <typename TA, typename TB>
bool
SameGrid(const TA * a, const TB * b)
{
  return a->GetLargestPossibleRegion() == b->GetLargestPossibleRegion() && a->GetSpacing() == b->GetSpacing() &&
         a->GetOrigin() == b->GetOrigin() && a->GetDirection() == b->GetDirection();
}

/** The region of a grid (`spatial` in tensor order, `index` its first index) that the cells [c0, c1) of `windows`
 * span, `extra` voxels around on each tensor axis, clipped to the grid and starting on a multiple of `align` voxels
 * from the grid's first: `lowPad` what that left of the margin on the low side. */
template <typename TRegion>
TRegion
CellSpan(const std::vector<int64_t> &        c0,
         const std::vector<int64_t> &        c1,
         const std::vector<int64_t> &        extra,
         const std::vector<PoolWindows> &    windows,
         const std::vector<int64_t> &        spatial,
         const typename TRegion::IndexType & index,
         std::vector<int64_t> &              lowPad,
         int64_t                             align = 1)
{
  constexpr unsigned int      Dim = TRegion::ImageDimension;
  typename TRegion::IndexType start;
  typename TRegion::SizeType  size;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    const int64_t first = windows[d].starts[c0[d]];
    const int64_t end = windows[d].ends[c1[d] - 1];
    const int64_t margin = std::max<int64_t>(0, first - extra[d]);
    const int64_t low = margin - margin % align;
    const int64_t high = std::min<int64_t>(spatial[d], end + extra[d]);
    start[Dim - 1 - d] = static_cast<IndexValueType>(low) + index[Dim - 1 - d];
    size[Dim - 1 - d] = static_cast<SizeValueType>(high - low);
    lowPad[d] = first - low;
  }
  return TRegion(start, size);
}

/** `t` narrowed to [lo, hi) on each spatial axis, the first of them at `firstAxis`: the cells [lo, hi) of a map
 * ({1, C, cells...}), as a view. */
inline torch::Tensor
NarrowTo(torch::Tensor t, const std::vector<int64_t> & lo, const std::vector<int64_t> & hi, int64_t firstAxis = 2)
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

/** The cells from `c0` on of a field ({1, Dim, n...}, in voxels of the fixed grid, tensor order) written into the
 * image AllocateCellGrid made for `cells` cells, in millimetres, as BuildCellGrid writes the whole field: the cells on
 * a side of the grid replicated over its border. */
template <unsigned int Dim, typename TFixedImage>
void
WriteCellsToCellGrid(const torch::Tensor &            field,
                     const std::vector<int64_t> &     c0,
                     const std::vector<int64_t> &     cells,
                     const TFixedImage *              fixed,
                     Image<Vector<float, Dim>, Dim> * grid)
{
  namespace F = torch::nn::functional;
  using FieldType = Image<Vector<float, Dim>, Dim>;
  std::vector<int64_t>           padding; // F::pad takes its pairs from the last axis backwards
  typename FieldType::RegionType region;
  for (int d = static_cast<int>(Dim) - 1; d >= 0; --d)
  {
    const int64_t n = field.size(2 + d);
    const int64_t low = c0[d] == 0 ? CellGridBorder : 0;
    const int64_t high = c0[d] + n == cells[d] ? CellGridBorder : 0;
    padding.push_back(low);
    padding.push_back(high);
    region.SetIndex(Dim - 1 - d, static_cast<IndexValueType>(c0[d] + CellGridBorder - low));
    region.SetSize(Dim - 1 - d, static_cast<SizeValueType>(n + low + high));
  }
  const torch::Tensor values = F::pad(field, F::PadFuncOptions(padding).mode(torch::kReplicate))
                                 .squeeze(0)
                                 .permute(SqueezedChannelLastPermutation<Dim>())
                                 .contiguous()
                                 .to(torch::kCPU);
  const float *                  v = values.data_ptr<float>(); // components in (z, y, x) order
  const auto &                   spacing = fixed->GetSpacing();
  const auto &                   direction = fixed->GetDirection();
  ImageRegionIterator<FieldType> it(grid, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it, v += Dim) // raster order, ITK x fastest: the tensor's
  {
    Vector<float, Dim> physical;
    for (unsigned int r = 0; r < Dim; ++r)
    {
      double accumulator = 0.0;
      for (unsigned int c = 0; c < Dim; ++c)
      {
        accumulator += direction[r][c] * spacing[c] * static_cast<double>(v[Dim - 1 - c]);
      }
      physical[r] = accumulator;
    }
    it.Set(physical);
  }
}

/** Where a layer's samples lie among the image's voxels, along each tensor axis: every `stride` voxels from the
 * first, n voxels giving ceil(n / stride) samples when the last may cover fewer voxels (`partial`, a strided
 * convolution) and floor(n / stride) otherwise (a pooling). A layer at the image's resolution has stride 1. */
struct LayerLattice
{
  std::vector<int64_t> stride;
  std::vector<bool>    partial;

  int64_t
  Samples(size_t d, int64_t voxels) const
  {
    return (voxels + (partial[d] ? stride[d] - 1 : 0)) / stride[d];
  }
};

/** The lattice of the image itself, of a mask on its grid. */
inline LayerLattice
ImageLattice(size_t dim)
{
  return { std::vector<int64_t>(dim, 1), std::vector<bool>(dim, false) };
}

/** The lattices of the `layers` a model gave for a probe of `LatticeProbe` voxels along every axis, one more than a
 * power of two: a layer of 64 / s samples keeps whole samples every s voxels, one of 64 / s + 1 a partial last one. */
constexpr int64_t LatticeProbe = 65;

template <unsigned int Dim>
std::vector<LayerLattice>
LayerLattices(const std::vector<torch::Tensor> & layers, const char * who)
{
  auto                      powerOfTwo = [](int64_t n) { return n > 0 && (n & (n - 1)) == 0; };
  std::vector<LayerLattice> lattices;
  for (const torch::Tensor & layer : layers)
  {
    LayerLattice lattice = ImageLattice(Dim);
    for (unsigned int d = 0; d < Dim; ++d)
    {
      const int64_t samples = layer.size(2 + static_cast<int64_t>(d));
      if (samples == LatticeProbe)
      {
        continue;
      }
      const int64_t whole = LatticeProbe - 1;
      if (samples > 0 && whole % samples == 0 && powerOfTwo(whole / samples))
      {
        lattice.stride[d] = whole / samples;
      }
      else if (samples > 1 && whole % (samples - 1) == 0 && powerOfTwo(whole / (samples - 1)))
      {
        lattice.stride[d] = whole / (samples - 1);
        lattice.partial[d] = true;
      }
      else
      {
        itkGenericExceptionMacro(<< who << ": a feature layer of " << layer.sizes() << " for an image of "
                                 << LatticeProbe
                                 << " voxels a side lies on no lattice of the image's (a power of "
                                    "two of its voxels), which RegionSize needs to compute it by regions.");
      }
    }
    lattices.push_back(lattice);
  }
  return lattices;
}

/** The lattices of the kept layers of the fixed and of the moving models, and what the regions read for the features
 * take from them: a layer coarser than the image lies on the image's lattice every `align` voxels, so the regions
 * start on it, and they reach, past the models' halo, what pooling the whole layer to the cells reads beyond a cell's
 * own voxels. */
struct RegionLattices
{
  std::vector<LayerLattice> fixed;
  std::vector<LayerLattice> moving;
  int64_t                   align{ 1 };
  std::vector<int64_t>      reach;
};

/** Probed with `layersOf(image, moving)`, the layers of an image region (the image itself without a model); the
 * moving models are probed when they are not the fixed ones (`ownMovingModels`). `cellVoxels` is a cell's side. */
template <unsigned int Dim, typename TLayersOf>
RegionLattices
ProbeRegionLattices(const std::vector<int64_t> & halo,
                    const std::vector<int64_t> & cellVoxels,
                    const torch::Device &        device,
                    bool                         ownMovingModels,
                    TLayersOf &&                 layersOf,
                    const char *                 who)
{
  std::vector<int64_t> shape(Dim + 2, LatticeProbe);
  shape[0] = shape[1] = 1;
  const torch::Tensor probe = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  RegionLattices      lattices;
  lattices.fixed = LayerLattices<Dim>(layersOf(probe, false), who);
  lattices.moving = ownMovingModels ? LayerLattices<Dim>(layersOf(probe, true), who) : lattices.fixed;
  for (const auto * side : { &lattices.fixed, &lattices.moving })
  {
    for (const LayerLattice & lattice : *side)
    {
      lattices.align = std::max(lattices.align, *std::max_element(lattice.stride.begin(), lattice.stride.end()));
    }
  }
  lattices.reach = halo;
  for (unsigned int d = 0; d < Dim && lattices.align > 1; ++d)
  {
    lattices.reach[d] += cellVoxels[d] + 2 * lattices.align;
  }
  return lattices;
}

/** A layer ({1, C, samples...}) of a span of the image, pooled over the cells [c0, c1) as the whole-image run pools
 * the layer whole. A layer at the image's resolution: over the cells' voxels, `windows`. One coarser along any axis:
 * the whole layer is adaptive-average pooled to the cells along every axis, so each cell takes the samples that
 * pooling gives it. `lowPad` is where the cells start in the span, `spatial` the image; the span starts on the
 * layer's lattice and holds those samples (CellSpan's `align` and `extra`). */
template <unsigned int Dim>
torch::Tensor
PoolLayerToCells(torch::Tensor                    layer,
                 const LayerLattice &             lattice,
                 const std::vector<int64_t> &     lowPad,
                 const std::vector<int64_t> &     c0,
                 const std::vector<int64_t> &     c1,
                 const std::vector<PoolWindows> & windows,
                 const std::vector<int64_t> &     spatial,
                 const char *                     who)
{
  std::vector<PoolWindows> relative(Dim);
  const bool               atImage = lattice.stride == std::vector<int64_t>(Dim, 1);
  for (unsigned int d = 0; d < Dim; ++d)
  {
    const int64_t axis = 2 + static_cast<int64_t>(d);
    const int64_t stride = lattice.stride[d];
    const int64_t low = windows[d].starts[c0[d]] - lowPad[d]; // the span's first voxel in the image
    const int64_t cells = static_cast<int64_t>(windows[d].starts.size());
    const int64_t length = lattice.Samples(d, spatial[d]); // the whole layer's samples
    if (low % stride != 0 || (stride > 1 && length < cells))
    {
      itkGenericExceptionMacro(<< who << ": a feature layer " << stride << " times coarser than the image, of "
                               << length << " samples for " << cells
                               << " cells, cannot be pooled to the cells by regions: it must be no coarser than they.");
    }
    // Cell i's samples on the whole layer: [start(i), end(i)).
    auto start = [&](int64_t i) { return atImage ? windows[d].starts[i] : i * length / cells; };
    auto end = [&](int64_t i) { return atImage ? windows[d].ends[i] : ((i + 1) * length + cells - 1) / cells; };
    const int64_t first = start(c0[d]) - low / stride;
    const int64_t last = end(c1[d] - 1) - low / stride;
    if (first < 0 || last > layer.size(axis))
    {
      itkGenericExceptionMacro(<< who << ": the region read around the cells does not hold the samples of a layer of "
                               << layer.sizes() << " they pool; give RegionHalo more voxels.");
    }
    layer = layer.narrow(axis, first, last - first);
    for (int64_t i = c0[d]; i < c1[d]; ++i)
    {
      relative[d].starts.push_back(start(i) - start(c0[d]));
      relative[d].ends.push_back(end(i) - start(c0[d]));
    }
  }
  return PoolByWindows<Dim>(layer, relative);
}

/** The share of a mask ({1, 1, span...}, floats) in each sample a layer of `lattice` has over that span, as the
 * whole-image run brings the whole mask to the whole layer (adaptive average pooling over the image's `spatial`
 * voxels); the mask itself for a layer at the image's resolution. `low` is the span's first voxel in the image. */
template <unsigned int Dim>
torch::Tensor
SpanMaskShare(const torch::Tensor &        mask,
              const LayerLattice &         lattice,
              const std::vector<int64_t> & low,
              const std::vector<int64_t> & spatial)
{
  std::vector<PoolWindows> windows(Dim);
  bool                     same = true;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    const int64_t span = mask.size(2 + static_cast<int64_t>(d));
    const int64_t stride = lattice.stride[d];
    const int64_t length = lattice.Samples(d, spatial[d]);
    same = same && stride == 1;
    for (int64_t j = low[d] / stride; j < low[d] / stride + lattice.Samples(d, span); ++j)
    {
      // Sample j's voxels on the whole image, kept inside the span at its edges.
      const int64_t first = stride == 1 ? j : j * spatial[d] / length;
      const int64_t last = stride == 1 ? j + 1 : ((j + 1) * spatial[d] + length - 1) / length;
      windows[d].starts.push_back(std::clamp<int64_t>(first - low[d], 0, span - 1));
      windows[d].ends.push_back(std::clamp<int64_t>(last - low[d], windows[d].starts.back() + 1, span));
    }
  }
  return same ? mask : PoolByWindows<Dim>(mask, windows);
}

/** What the feature models of a region-sampled run must be: run at the image's resolution (no voxel size of
 * their own, whose lattice a region does not line up with), of the image's dimension and without PCA (`pca`, the
 * components kept per layer). The kept layers over the models, and whether one reads the image's statistics. */
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
                  const std::vector<unsigned int> &             pca,
                  const char *                                  who)
{
  if (std::any_of(pca.begin(), pca.end(), [](unsigned int components) { return components > 0; }))
  {
    itkGenericExceptionMacro(<< who
                             << ": PCA fits its basis on the whole fixed feature map; set it to 0 with "
                                "RegionSize.");
  }
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

/** The kept feature layers of an image region ({1, 1, span...}), each at its own resolution, each model run on the
 * region whole in one forward, `metadata` (when given) set on the configurations first; the region itself with no
 * model. `layerCount` is what the caller keeps, checked. */
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

/** The cells within `reach` of `box` in a grid of `cells` cells, slid into the grid where it ends, so that they keep
 * their count along each axis (the grid's when it has fewer). What reads `reach` cells around each cell of the box
 * finds them there, and a pooling of that half-width finds its kernel's cells at an end of the grid as the grid
 * whole gives them, however few cells the box has. */
inline CellBox
CellsAround(const CellBox & box, const std::vector<int64_t> & reach, const std::vector<int64_t> & cells)
{
  CellBox window = box;
  for (size_t d = 0; d < cells.size(); ++d)
  {
    const int64_t count = std::min(cells[d], box.cell1[d] - box.cell0[d] + 2 * reach[d]);
    window.cell0[d] = std::clamp<int64_t>(box.cell0[d] - reach[d], 0, cells[d] - count);
    window.cell1[d] = window.cell0[d] + count;
  }
  return window;
}

/** Adam (torch::optim::Adam's update, no weight decay) on a grid of cells stepped by parts: Advance() starts a step
 * of the grid, StepCells() takes it on the cells [c0, c1), each cell once. The parameters, their gradient and the
 * moments may be mapped files, of which a part's step reads and writes those cells alone, on `device`. */
struct CellAdam
{
  torch::Tensor exponentialAverage;
  torch::Tensor exponentialAverageSquare;
  double        learningRate;
  double        beta1;
  double        beta2;
  double        epsilon;
  int64_t       steps{ 0 };

  void
  Advance()
  {
    ++steps;
  }

  void
  StepCells(torch::Tensor &              theta,
            const torch::Tensor &        gradient,
            const std::vector<int64_t> & c0,
            const std::vector<int64_t> & c1,
            const torch::Device &        device)
  {
    torch::NoGradGuard noGrad;
    auto               onDevice = [&](const torch::Tensor & whole) { // a copy, never the mapped cells themselves
      return NarrowTo(whole, c0, c1).to(device, torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);
    };
    const torch::Tensor g = onDevice(gradient);
    const torch::Tensor average = onDevice(exponentialAverage).mul_(beta1).add_(g, 1.0 - beta1);
    const torch::Tensor averageSquare = onDevice(exponentialAverageSquare).mul_(beta2).addcmul_(g, g, 1.0 - beta2);
    const double        correction1 = 1.0 - std::pow(beta1, static_cast<double>(steps));
    const double        correction2 = 1.0 - std::pow(beta2, static_cast<double>(steps));
    const torch::Tensor denominator = (averageSquare.sqrt() / std::sqrt(correction2)).add_(epsilon);
    WriteCells(theta, c0, onDevice(theta).addcdiv_(average, denominator, -learningRate / correction1));
    WriteCells(exponentialAverage, c0, average);
    WriteCells(exponentialAverageSquare, c0, averageSquare);
  }
};

/** \class ScratchFiles
 * \ingroup Impact
 * \brief Tensors of zeros in files of a directory, mapped shared.
 *
 * The kernel pages them in and out, so what grows with the image need not fit the memory. The files go with the
 * object, which the tensors must not outlive, save one kept by its caller. */
class ScratchFiles
{
public:
  /** In `directory`, the system's temporary one when empty. */
  explicit ScratchFiles(const std::string & directory)
    : m_Directory(directory)
  {}
  ScratchFiles(const ScratchFiles &) = delete;
  ScratchFiles &
  operator=(const ScratchFiles &) = delete;
  ~ScratchFiles()
  {
    for (const std::filesystem::path & path : m_Paths)
    {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  }

  /** With `kept`, the file is the caller's from then on: it outlives this object, and the caller removes `*kept`
   * once the tensor is gone. */
  torch::Tensor
  Zeros(const std::vector<int64_t> & shape,
        torch::ScalarType            dtype = torch::kFloat32,
        std::filesystem::path *      kept = nullptr)
  {
    int64_t count = 1;
    for (const int64_t n : shape)
    {
      count *= n;
    }
    const std::uintmax_t bytes = static_cast<std::uintmax_t>(count) * torch::elementSize(dtype);
    std::error_code      error;
    if (m_Paths.empty())
    {
      if (m_Directory.empty())
      {
        m_Directory = std::filesystem::temp_directory_path(error);
      }
      m_Available = error ? 0 : std::filesystem::space(m_Directory, error).available;
    }
    if (error)
    {
      itkGenericExceptionMacro(<< "The scratch directory " << m_Directory << " cannot be used: " << error.message()
                               << ".");
    }
    // Against the space free when the run started: all that guards a system that cannot reserve a file's blocks.
    if (m_Bytes + bytes > m_Available)
    {
      itkGenericExceptionMacro(<< "The scratch directory " << m_Directory << " has " << m_Available
                               << " bytes free where the run maps " << m_Bytes + bytes
                               << "; set ScratchDirectory to a directory of a larger disk.");
    }
    std::random_device          random;
    const std::filesystem::path path = m_Directory / ("itkImpactScratch-" + std::to_string(random()) + "-" +
                                                      std::to_string(random()) + "-" + std::to_string(m_Paths.size()));
    if (std::filesystem::exists(path))
    {
      itkGenericExceptionMacro(<< "The scratch file " << path << " exists.");
    }
    m_Paths.push_back(path);
#if defined(__linux__)
    // A mapped file is sparse, and a file system that fills up under it kills the process at a write: its blocks
    // are reserved first, so that a full one is refused here.
    const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    const int failure = descriptor < 0 ? errno : ::posix_fallocate(descriptor, 0, static_cast<off_t>(bytes));
    if (descriptor >= 0)
    {
      ::close(descriptor);
    }
    if (failure != 0 && failure != EOPNOTSUPP && failure != EINVAL) // those two: a file system without the call
    {
      itkGenericExceptionMacro(<< "The scratch directory " << m_Directory << " cannot hold a file of " << bytes
                               << " bytes (" << std::strerror(failure)
                               << "); set ScratchDirectory to a directory of a larger disk.");
    }
#endif
    // A new file is zeros.
    torch::Tensor tensor = torch::from_file(path.string(), true, count, torch::TensorOptions().dtype(dtype));
#if !defined(_WIN32)
    std::filesystem::remove(path, error); // the mapping keeps it: a killed process leaves nothing behind
#endif
    if (kept != nullptr)
    {
      *kept = path;
      m_Paths.pop_back();
    }
    m_Bytes += bytes;
    return tensor.view(shape);
  }

private:
  std::filesystem::path              m_Directory;
  std::vector<std::filesystem::path> m_Paths;
  std::uintmax_t                     m_Available{ 0 };
  std::uintmax_t                     m_Bytes{ 0 };
};

/** \class MappedPixels
 * \ingroup Impact
 * \brief The pixels of an image in a file mapped to memory.
 *
 * The container holds the mapping, which lives as long as the image whose pixels it is and whatever holds that
 * image; the file goes with it. */
template <typename TPixel>
class MappedPixels : public ImportImageContainer<SizeValueType, TPixel>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MappedPixels);

  using Self = MappedPixels;
  using Superclass = ImportImageContainer<SizeValueType, TPixel>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(MappedPixels);

  /** `count` pixels of zeros in a file of `scratch`. */
  void
  Map(ScratchFiles & scratch, SizeValueType count)
  {
    m_Mapping = scratch.Zeros({ static_cast<int64_t>(count * sizeof(TPixel)) }, torch::kUInt8, &m_Path);
    this->SetImportPointer(static_cast<TPixel *>(m_Mapping.data_ptr()), count, /*LetContainerManageMemory=*/false);
  }

protected:
  MappedPixels() = default;
  ~MappedPixels() override
  {
    m_Mapping = torch::Tensor(); // unmapped before the file goes
    std::error_code ignored;
    std::filesystem::remove(m_Path, ignored);
  }

private:
  torch::Tensor         m_Mapping;
  std::filesystem::path m_Path;
};

/** The pixels of AllocateCellGrid's image for `cells` cells, in a file of `scratch`: of the grid the host memory
 * then holds the pages the system keeps of it. */
template <unsigned int Dim>
typename MappedPixels<Vector<float, Dim>>::Pointer
MapCellGridPixels(const std::vector<int64_t> & cells, ScratchFiles & scratch)
{
  SizeValueType count = 1;
  for (const int64_t n : cells)
  {
    count *= static_cast<SizeValueType>(n + 2 * CellGridBorder);
  }
  auto pixels = MappedPixels<Vector<float, Dim>>::New();
  pixels->Map(scratch, count);
  return pixels;
}

/** What the pooled feature maps of a region-sampled run may take to stay resident on `device`, when the caller sets
 * no budget: half the device memory free now, or a quarter of the machine's memory on the CPU. */
inline double
ResidentBudget(const torch::Device & device)
{
#if ITK_IMPACT_HAS_CUDA_MEMORY_STATS
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

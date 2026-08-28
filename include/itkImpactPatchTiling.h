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

#ifndef itkImpactPatchTiling_h
#define itkImpactPatchTiling_h

// Running a TorchScript model over a volume it does not fit in one pass: the patch grid, the
// blend windows, the accumulator that puts the answers back together, and RunTiledModel() which
// drives the three.
//
// It works on torch tensors and knows nothing about itk::Image, so both consumers share it --
// itk::ImageToFeaturesMap, which converts the result to an itk::VectorImage, and the
// registration filters, which keep it as a tensor on the device. There was one implementation
// per consumer before, and only one of them tiled.
//
// The reassembly is a port of KonfAI's (konfai/data/patching.py), so that running
// TotalSegmentator's or MRSegmentator's TorchScript export through it reproduces what those
// tools produce themselves.
//
// The PCA reduction both consumers apply to what it produces lives here too, for the same reason:
// there was a copy on each side, kept in step by a comment.
//
// Include only from translation units that already depend on torch.

#include "itkImpactModelConfigurationDetail.h"

#include <itkMacro.h>
#include <itkMath.h>

#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace itk
{

/** \class PathCombine
 * \brief The window that weights one patch when a tiled feature map is put back together.
 *
 * A patch's weight is the outer product of one 1-D window per axis. The factors are kept
 * rather than their product: the patch grid is a full per-axis product and the window is
 * separable, so the total weight covering a voxel factorises,
 *
 *   sum_p prod_d w_(d,p_d)(x_d)  ==  prod_d ( sum_k w_(d,k)(x_d) ),
 *
 * which makes the normaliser one vector per axis instead of a volume, and lets each patch be
 * normalised by its own share as it arrives rather than in a second pass.
 *
 * \ingroup Impact
 */
class PathCombine
{
public:
  virtual ~PathCombine() = default;

  /** Build the 1-D windows for a patch of `patchSize`, tapered over `overlaps` voxels per side
   * on each axis. An axis with no overlap is blended with nothing, so its window is flat. */
  void
  SetPatchConfig(const std::vector<int64_t> & patchSize, const std::vector<int64_t> & overlaps)
  {
    m_Overlaps = overlaps;
    m_Windows.clear();
    m_Windows.reserve(patchSize.size());
    for (std::size_t d = 0; d < patchSize.size(); ++d)
    {
      m_Windows.push_back(overlaps[d] > 0 ? this->MakeWindow(patchSize[d], overlaps[d])
                                          : torch::ones({ patchSize[d] }, torch::kFloat32));
    }
  }

  /** Whether the window selects one patch per voxel (values 0 or 1) instead of weighting
   * several. A selection makes the kept regions a partition, so assembly writes each one
   * instead of accumulating a weighted sum -- which is what carries a discrete output (a label
   * map, an arg-max) through reassembly, where a weighting invents values between its classes. */
  virtual bool
  Selects() const
  {
    return false;
  }

  /** The 1-D window along `axis` for the patch at grid `position` of `count`. A weighting is
   * the same wherever the patch sits; a selection opens its border patches (see PathTrim). */
  virtual torch::Tensor
  Window(unsigned int axis, std::size_t itkNotUsed(position), std::size_t itkNotUsed(count)) const
  {
    return m_Windows[axis];
  }

protected:
  /** The 1-D weight along one axis: `size` samples, `overlap` of them tapered per side. */
  virtual torch::Tensor
  MakeWindow(int64_t size, int64_t overlap) const = 0;

  std::vector<torch::Tensor> m_Windows;
  std::vector<int64_t>       m_Overlaps;
};

/** \class PathMean
 * \brief Plain average of the patches covering a voxel.
 *
 * The averaging is done by the assembly normalisation, so the window itself is flat.
 *
 * \ingroup Impact
 */
class PathMean : public PathCombine
{
protected:
  torch::Tensor
  MakeWindow(int64_t size, int64_t itkNotUsed(overlap)) const override
  {
    return torch::ones({ size }, torch::kFloat32);
  }
};

/** \class PathCosinus
 * \brief Raised-cosine (sin^2) taper: an exact partition of unity over the overlap.
 *
 * \ingroup Impact
 */
class PathCosinus : public PathCombine
{
protected:
  torch::Tensor
  MakeWindow(int64_t size, int64_t overlap) const override
  {
    torch::Tensor window = torch::ones({ size }, torch::kFloat32);
    // The neighbour's cos^2 ramp is this sin^2 ramp's complement, so the two sum to one across
    // the overlap. The +0.5 phase keeps both ends > 0, so a patch meeting no neighbour there
    // still carries the whole weight and recovers the raw value.
    const torch::Tensor ramp =
      torch::sin((torch::arange(overlap, torch::kFloat32) + 0.5) / overlap * (Math::pi / 2.0)).pow(2);
    window.narrow(0, 0, overlap).copy_(ramp);
    window.narrow(0, size - overlap, overlap).copy_(ramp.flip(0));
    return window;
  }
};

/** \class PathGaussian
 * \brief nnU-Net's Gaussian importance weighting -- what TotalSegmentator and MRSegmentator
 * reassemble with.
 *
 * Favours the patch centre, where the network saw context on every side. Not a partition of
 * unity; the assembly normalises by the accumulated weight.
 *
 * \ingroup Impact
 */
class PathGaussian : public PathCombine
{
public:
  explicit PathGaussian(double sigmaScale = 0.125)
    : m_SigmaScale(sigmaScale)
  {}

protected:
  torch::Tensor
  MakeWindow(int64_t size, int64_t itkNotUsed(overlap)) const override
  {
    // sigma is one eighth of the PATCH extent, not of the overlap: the window describes how far
    // the network's field of view is trusted, which does not depend on how far patches stepped.
    const double        sigma = std::max(size * m_SigmaScale, 1e-6);
    const torch::Tensor coordinates = torch::arange(size, torch::kFloat32) - (size - 1) / 2.0;
    return torch::exp(-coordinates.pow(2) / (2.0 * sigma * sigma));
  }

  double m_SigmaScale;
};

/** \class PathTrim
 * \brief Selection instead of weighting: each voxel comes from the patch holding it most
 * centrally.
 *
 * An interior patch keeps its central `patch - overlap` band, so the kept bands tile the axis;
 * the first and last patch of an axis open towards the volume edge.
 *
 * \ingroup Impact
 */
class PathTrim : public PathCombine
{
public:
  bool
  Selects() const override
  {
    return true;
  }

  torch::Tensor
  Window(unsigned int axis, std::size_t position, std::size_t count) const override
  {
    const int64_t         overlap = m_Overlaps[axis];
    const torch::Tensor & window = m_Windows[axis];
    if (overlap <= 0 || (position > 0 && position + 1 < count))
    {
      return window;
    }
    // A border patch has no neighbour to hand its trimmed band to, so it keeps it; otherwise
    // the first and last overlap/2 voxels of the axis would be written by nobody.
    torch::Tensor opened = window.clone();
    if (position == 0)
    {
      opened.narrow(0, 0, overlap / 2).fill_(1.0f);
    }
    if (position + 1 == count)
    {
      const int64_t trailing = overlap - overlap / 2;
      opened.narrow(0, opened.size(0) - trailing, trailing).fill_(1.0f);
    }
    return opened;
  }

protected:
  torch::Tensor
  MakeWindow(int64_t size, int64_t overlap) const override
  {
    if (overlap >= size)
    {
      // Trimming both sides would leave an empty band, and a patch that keeps nothing has no
      // box to write; keep it whole and let the axis fall back to last-write-wins.
      return torch::ones({ size }, torch::kFloat32);
    }
    torch::Tensor window = torch::zeros({ size }, torch::kFloat32);
    // Split an odd overlap so consecutive kept bands abut exactly:
    // k*stride + (size - hi) == (k+1)*stride + lo, with lo = overlap/2, hi = overlap - lo.
    window.narrow(0, overlap / 2, size - overlap).fill_(1.0f);
    return window;
  }
};

/** \class PathCombineMode
 * \brief The blend windows a caller can ask for by name.
 *
 * \ingroup Impact
 */
enum class PathCombineMode
{
  Mean,
  Cosinus,
  Trim,
  Gaussian
};

/** Parse a blend-window name (case-insensitive). An unknown name throws rather than falling
 * back to a default, so a typo cannot silently change the reassembly. */
inline PathCombineMode
PathCombineModeFromString(const std::string & name)
{
  std::string key;
  key.reserve(name.size());
  for (const char c : name)
  {
    key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (key == "mean")
  {
    return PathCombineMode::Mean;
  }
  if (key == "cosinus" || key == "cosine")
  {
    return PathCombineMode::Cosinus;
  }
  if (key == "trim")
  {
    return PathCombineMode::Trim;
  }
  if (key == "gaussian" || key == "gauss")
  {
    return PathCombineMode::Gaussian;
  }
  itkGenericExceptionMacro("Unknown patch-combine window '" << name
                                                            << "'; expected one of "
                                                               "mean, cosinus, trim, gaussian.");
}

inline std::shared_ptr<PathCombine>
MakePathCombine(PathCombineMode mode)
{
  switch (mode)
  {
    case PathCombineMode::Mean:
      return std::make_shared<PathMean>();
    case PathCombineMode::Trim:
      return std::make_shared<PathTrim>();
    case PathCombineMode::Gaussian:
      return std::make_shared<PathGaussian>();
    case PathCombineMode::Cosinus:
    default:
      return std::make_shared<PathCosinus>();
  }
}

/** \class PatchGrid
 * \brief Where the patches of a tiled volume start.
 *
 * `starts[axis]` are the ordered start offsets along one axis, `positions[patch]` the per-axis
 * index into them, so a patch is addressed either by its rank in the Cartesian product (what
 * the inference loop counts) or by its position on each axis (what the blend window needs).
 *
 * \ingroup Impact
 */
struct PatchGrid
{
  std::vector<std::vector<int64_t>>     starts;
  std::vector<std::vector<std::size_t>> positions;

  std::size_t
  size() const
  {
    return positions.size();
  }

  int64_t
  Start(std::size_t patch, unsigned int axis) const
  {
    return starts[axis][positions[patch][axis]];
  }
};

/** Cut `shape` into patches of `patchSize` advancing by `patchSize - overlaps` per step.
 *
 * The count follows the stride, not the patch: counting ceil(extent / patchSize) leaves the
 * last patch short of the extent as soon as there are three or more per axis, and the tail of
 * the volume is never handed to the model. Patches overshooting the far edge are expected --
 * the model gets a zero-padded patch and the accumulator crops it back -- so what matters is
 * that the grid covers the extent. This is KonfAI's rule, which likewise never pads the volume
 * before tiling it. */
inline PatchGrid
MakePatchGrid(const std::vector<int64_t> & shape,
              const std::vector<int64_t> & patchSize,
              const std::vector<int64_t> & overlaps)
{
  PatchGrid grid;
  grid.starts.resize(patchSize.size());
  for (std::size_t d = 0; d < patchSize.size(); ++d)
  {
    const int64_t extent = shape[d];
    const int64_t stride = patchSize[d] - overlaps[d];
    if (stride <= 0)
    {
      itkGenericExceptionMacro("ImageToFeaturesMap: overlap (" << overlaps[d]
                                                               << ") must be smaller than the patch size ("
                                                               << patchSize[d] << ") along dimension " << d
                                                               << "; otherwise consecutive patches never advance.");
    }
    int64_t count = 1;
    if (extent > patchSize[d])
    {
      count = (extent - patchSize[d] + stride - 1) / stride + 1;
    }
    for (int64_t step = 0; step < count; ++step)
    {
      grid.starts[d].push_back(stride * step);
    }
  }

  // The Cartesian product of the per-axis positions, counted like an odometer whose last axis
  // turns fastest, so the first axis is outermost.
  std::size_t count = 1;
  for (const std::vector<int64_t> & axisStarts : grid.starts)
  {
    count *= axisStarts.size();
  }
  grid.positions.reserve(count);
  std::vector<std::size_t> current(patchSize.size(), 0);
  for (std::size_t n = 0; n < count; ++n)
  {
    grid.positions.push_back(current);
    for (std::size_t axis = patchSize.size(); axis-- > 0;)
    {
      if (++current[axis] < grid.starts[axis].size())
      {
        break;
      }
      current[axis] = 0;
    }
  }
  return grid;
}

/** The same grid at a layer's own resolution: every patch keeps the position it was cut from,
 * taken to that layer's scale.
 *
 * Derived from the input grid rather than recomputed from the layer's extent, so the two can
 * never disagree on how many patches there are. Each start is scaled directly rather than
 * accumulated from a rounded stride, which keeps the placement error at half a layer voxel
 * instead of letting it grow with the patch index whenever the input stride does not scale to a
 * whole number of layer voxels. */
inline PatchGrid
ScalePatchGrid(const PatchGrid & grid, const std::vector<double> & scales)
{
  PatchGrid scaled;
  scaled.positions = grid.positions;
  scaled.starts.resize(grid.starts.size());
  for (std::size_t d = 0; d < grid.starts.size(); ++d)
  {
    for (const int64_t start : grid.starts[d])
    {
      scaled.starts[d].push_back(static_cast<int64_t>(std::llround(start * scales[d])));
    }
  }
  return scaled;
}

/** Cut patch `index` out of `tensor` ([channels, spatial...]) and zero-pad it up to `patchSize`.
 * The last patch of an axis is short, but the model was exported for a fixed input size; only
 * the in-volume part is blended back, so the padding never reaches the output. */
inline torch::Tensor
ExtractPatch(const torch::Tensor &        tensor,
             const PatchGrid &            grid,
             std::size_t                  index,
             const std::vector<int64_t> & patchSize)
{
  const int64_t        dimension = static_cast<int64_t>(patchSize.size());
  torch::Tensor        patch = tensor;
  std::vector<int64_t> padding;
  // constant_pad_nd takes its pairs from the last axis backwards.
  for (int64_t d = dimension - 1; d >= 0; --d)
  {
    const int64_t start = grid.Start(index, static_cast<unsigned int>(d));
    const int64_t extent = std::min(patchSize[d], tensor.size(d + 1) - start);
    padding.push_back(0);
    padding.push_back(patchSize[d] - extent);
  }
  for (int64_t d = 0; d < dimension; ++d)
  {
    const int64_t start = grid.Start(index, static_cast<unsigned int>(d));
    patch = patch.narrow(d + 1, start, std::min(patchSize[d], tensor.size(d + 1) - start));
  }
  return torch::constant_pad_nd(patch, padding, 0);
}

/** \class Accumulator
 * \brief Puts the model's per-patch answers back together into one feature map.
 *
 * Each patch is blended in on arrival and dropped: the overlap blend is a weighted sum, so
 * accumulating incrementally is equivalent, and keeping every patch until a final pass costs
 * more than the map they assemble into.
 *
 * \ingroup Impact
 */
class Accumulator
{
public:
  /** `shape` is the volume extent to assemble into, `patchSize` the extent of the patches the
   * model returns, `overlaps` their per-axis overlap, `combine` the blend window (null means
   * plain overwrite). */
  Accumulator(std::vector<int64_t>         shape,
              std::vector<int64_t>         patchSize,
              std::vector<int64_t>         overlaps,
              PatchGrid                    grid,
              std::shared_ptr<PathCombine> combine)
    : m_Shape(std::move(shape))
    , m_Grid(std::move(grid))
    , m_Combine(std::move(combine))
  {
    m_Done.assign(m_Grid.size(), false);
    if (m_Combine)
    {
      m_Combine->SetPatchConfig(patchSize, overlaps);
      this->ComputeWeightTotals();
    }
  }

  /** Blend one patch in and drop it. Re-adding an index is a no-op. */
  void
  AddLayer(std::size_t index, const torch::Tensor & layer)
  {
    if (m_Done[index])
    {
      return;
    }
    const std::size_t dimension = m_Shape.size();
    if (!m_Result.defined())
    {
      // Allocated to the volume extent, not to the extent the grid spans: the last patch of
      // each axis overshoots, and that tail is outside the volume.
      std::vector<int64_t> full;
      full.reserve(dimension + 1);
      full.push_back(layer.size(0));
      full.insert(full.end(), m_Shape.begin(), m_Shape.end());
      m_Result = at::zeros(full, layer.options());
    }
    else if (m_Result.size(0) != layer.size(0))
    {
      // Otherwise a channel count of 1 would broadcast over the map instead of failing.
      itkGenericExceptionMacro("ImageToFeaturesMap: patch " << index << " returned " << layer.size(0)
                                                            << " channels where the map holds " << m_Result.size(0)
                                                            << ".");
    }

    // Crop to the volume BEFORE weighting: the padded tail of a border patch has no share of
    // the blend weight to compute, and every index below then stays in range.
    torch::Tensor source = layer;
    torch::Tensor destination = m_Result;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      const int64_t start = m_Grid.Start(index, static_cast<unsigned int>(d));
      const int64_t extent = std::min(layer.size(static_cast<int64_t>(d) + 1), m_Shape[d] - start);
      source = source.narrow(static_cast<int64_t>(d) + 1, 0, extent);
      destination = destination.narrow(static_cast<int64_t>(d) + 1, start, extent);
    }

    if (!m_Combine)
    {
      destination.copy_(source);
    }
    else if (m_Combine->Selects())
    {
      // The kept regions partition the volume: writing the box this patch owns is the whole
      // operation, and it is what carries a discrete output through unchanged.
      for (std::size_t d = 0; d < dimension; ++d)
      {
        const std::pair<int64_t, int64_t> box = this->KeptBox(
          static_cast<unsigned int>(d), m_Grid.positions[index][d], source.size(static_cast<int64_t>(d) + 1));
        source = source.narrow(static_cast<int64_t>(d) + 1, box.first, box.second);
        destination = destination.narrow(static_cast<int64_t>(d) + 1, box.first, box.second);
      }
      destination.copy_(source);
    }
    else
    {
      // Scale by this patch's share of the weight. Normalising per patch rather than dividing
      // the assembled volume by a weight map drops both the volume-sized buffer and the final
      // division pass; the shares sum to one per voxel by construction, so the blend stays exact
      // at the border, where the raw window does not sum to one.
      //
      // The separable factors are combined into one patch-shaped weight (a few MB, no channel
      // axis) and applied with addcmul_, so no weighted copy of the patch is ever materialised.
      torch::Tensor weight = this->ShareView(0, index, source);
      for (std::size_t d = 1; d < dimension; ++d)
      {
        weight = weight * this->ShareView(static_cast<unsigned int>(d), index, source);
      }
      destination.addcmul_(source, weight);
    }

    m_Done[index] = true;
    ++m_Filled;
  }

  /** True once every patch of the grid has been blended in. */
  bool
  IsFull() const
  {
    return m_Filled == m_Grid.size();
  }

  /** Whether Assemble() has run; IsFull() cannot answer that once the result is handed out. */
  bool
  IsAssembled() const
  {
    return m_Assembled;
  }

  std::size_t
  NumberOfPatches() const
  {
    return m_Grid.size();
  }

  /** Hand out the assembled map and release it. Nothing to normalise and nothing to crop:
   * every patch was blended in with its share of the weight, and cropped to the volume first. */
  torch::Tensor
  Assemble()
  {
    if (!m_Result.defined())
    {
      itkGenericExceptionMacro("ImageToFeaturesMap: the feature map was assembled before any patch was blended in.");
    }
    torch::Tensor result = m_Result;
    m_Result = torch::Tensor();
    m_Assembled = true;
    return result;
  }

  /** Blend into `destination` ([channels, shape...]) instead of a buffer of the accumulator's
   * own, zeroed first. Lets the caller pass a view of the output image's own pixels so the
   * map -- by far the largest object here -- exists once rather than twice. */
  void
  SetDestination(const torch::Tensor & destination)
  {
    m_Result = destination;
    m_Result.zero_();
  }

private:
  /** Per axis, the blend weight summed over the grid: the denominator of every share. It
   * factorises (see PathCombine), so it is one vector per axis and never a volume. */
  void
  ComputeWeightTotals()
  {
    const std::size_t dimension = m_Shape.size();
    m_Totals.resize(dimension);
    m_Shares.assign(dimension, {});
    m_Boxes.assign(dimension, {});
    for (std::size_t d = 0; d < dimension; ++d)
    {
      const std::size_t count = m_Grid.starts[d].size();
      m_Shares[d].resize(count);
      m_Boxes[d].assign(count, { -1, -1 });
      m_Totals[d] = torch::zeros({ m_Shape[d] }, torch::kFloat32);
      for (std::size_t k = 0; k < count; ++k)
      {
        const torch::Tensor window = m_Combine->Window(static_cast<unsigned int>(d), k, count);
        const int64_t       start = m_Grid.starts[d][k];
        const int64_t       extent = std::min(window.size(0), m_Shape[d] - start);
        m_Totals[d].narrow(0, start, extent).add_(window.narrow(0, 0, extent));
      }
      // A voxel with no weight was never handed to the model: a coverage bug, not something to
      // clamp away. Checking per axis is exact -- the product is zero exactly when a factor is.
      if (!m_Combine->Selects() && m_Totals[d].min().item<float>() <= 0.0f)
      {
        itkGenericExceptionMacro("ImageToFeaturesMap: "
                                 << (m_Totals[d] <= 0.0f).sum().item<int64_t>() << " position(s) along dimension " << d
                                 << " of the assembled feature map carry no blending weight; the patch size and "
                                    "overlap do not tile the volume.");
      }
    }
  }

  /** This patch's fraction of the weight along one axis, `w / sum_k w`, cached per axis and
   * grid position (the cropped extent follows from the position). */
  const torch::Tensor &
  Share(unsigned int axis, std::size_t position)
  {
    torch::Tensor & share = m_Shares[axis][position];
    if (!share.defined())
    {
      const torch::Tensor window = m_Combine->Window(axis, position, m_Grid.starts[axis].size());
      const int64_t       start = m_Grid.starts[axis][position];
      const int64_t       extent = std::min(window.size(0), m_Shape[axis] - start);
      share = window.narrow(0, 0, extent) / m_Totals[axis].narrow(0, start, extent);
    }
    return share;
  }

  /** The share along `axis` shaped to broadcast against a [channels, spatial...] patch.
   *
   * The windows are built on the host -- they are a few hundred floats and the geometry is read
   * back from them -- so the share is brought to the patch's own device and dtype here. A caller
   * that accumulates on the GPU would otherwise multiply a CUDA patch by a CPU vector. */
  torch::Tensor
  ShareView(unsigned int axis, std::size_t index, const torch::Tensor & source)
  {
    std::vector<int64_t> view(source.dim(), 1);
    view[axis + 1] = -1;
    return this->Share(axis, m_Grid.positions[index][axis]).view(view).to(source.options());
  }

  /** The sub-box a selection keeps along one axis: the run of ones in its window, clipped to
   * the in-volume extent. Read from the host-side window and cached, so no patch needs a
   * device sync. */
  std::pair<int64_t, int64_t>
  KeptBox(unsigned int axis, std::size_t position, int64_t extent)
  {
    std::pair<int64_t, int64_t> & box = m_Boxes[axis][position];
    if (box.first < 0)
    {
      const torch::Tensor window = m_Combine->Window(axis, position, m_Grid.starts[axis].size());
      const torch::Tensor kept = window.narrow(0, 0, extent).nonzero().flatten();
      if (kept.numel() == 0)
      {
        box = { 0, 0 };
      }
      else
      {
        const int64_t first = kept[0].item<int64_t>();
        box = { first, kept[kept.numel() - 1].item<int64_t>() + 1 - first };
      }
    }
    return box;
  }

  std::vector<int64_t>                                  m_Shape;
  PatchGrid                                             m_Grid;
  std::shared_ptr<PathCombine>                          m_Combine;
  torch::Tensor                                         m_Result;
  std::vector<torch::Tensor>                            m_Totals;
  std::vector<std::vector<torch::Tensor>>               m_Shares;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> m_Boxes;
  std::vector<bool>                                     m_Done;
  std::size_t                                           m_Filled{ 0 };
  bool                                                  m_Assembled{ false };
};

/** Run `config` over `input` patch by patch, and return one assembled tensor per kept layer.
 *
 * `input` is [1, spatial...]; the number of spatial axes is the IMAGE dimension, which may exceed
 * the model's. A model that spans fewer axes is swept over the ones it does not: a 2D network on
 * a volume is run slice by slice along the leading tensor axes, which are the last ITK ones.
 *
 * Each patch is repeated to the model's channel count, moved to `device` and run; every kept
 * layer is blended back on `accumulateOn`, at its own resolution -- a /2 encoder yields a
 * half-size map. Keeping the accumulation device separate from the compute device is what lets
 * one caller assemble on the host while another stays on the GPU.
 *
 * `makeDestination`, when given, is called once per kept layer with its index and full shape and
 * may return a tensor to blend into, so a caller that already owns the memory (an image buffer)
 * is not made to copy. An undefined return, or no callback, lets the accumulator allocate.
 *
 * Runs under no-grad: a tiled pass would otherwise keep every patch's autograd graph alive until
 * the blend. A caller that needs the graph must run the model whole.
 */
inline std::vector<torch::Tensor>
RunTiledModel(const ImpactModelConfiguration &                                                config,
              const torch::Tensor &                                                           input,
              const torch::Device &                                                           device,
              const torch::Device &                                                           accumulateOn,
              PathCombineMode                                                                 combine,
              const std::function<torch::Tensor(std::size_t, const std::vector<int64_t> &)> & makeDestination = {})
{
  torch::NoGradGuard noGrad;

  const unsigned int dimension = config.GetDimension();
  const unsigned int imageDimension = static_cast<unsigned int>(input.dim()) - 1;
  if (dimension == 0 || dimension > imageDimension)
  {
    itkGenericExceptionMacro("IMPACT: the model " << config.GetModelPath() << " is configured for " << dimension
                             << " dimension(s), which a " << imageDimension << "D image cannot be tiled for.");
  }
  const unsigned int sweptAxes = imageDimension - dimension;

  const std::vector<int64_t> &      configuredPatchSize = config.GetPatchSize();
  const std::vector<unsigned int> & configuredOverlaps = config.GetOverlaps();

  std::vector<int64_t> sweptShape(sweptAxes);
  int64_t              numberOfSlices = 1;
  for (unsigned int a = 0; a < sweptAxes; ++a)
  {
    sweptShape[a] = input.size(a + 1);
    numberOfSlices *= sweptShape[a];
  }

  std::vector<int64_t> inputShape(dimension);
  std::vector<int64_t> patchSize(dimension);
  std::vector<int64_t> overlaps(dimension);
  for (unsigned int d = 0; d < dimension; ++d)
  {
    // `d` indexes the TENSOR axes the model spans, which come after the swept ones and run in the
    // reverse of ITK's order; the patch size and the overlap are declared in ITK order, like the
    // voxel size beside them, so they are reversed onto it here. A model that spans fewer axes
    // than the image keeps the leading ITK ones: a 2D patch is (x, y), swept along z.
    const unsigned int itkAxis = dimension - 1 - d;
    inputShape[d] = input.size(sweptAxes + d + 1);
    if (configuredPatchSize[itkAxis] > 0)
    {
      patchSize[d] = configuredPatchSize[itkAxis];
      overlaps[d] = itkAxis < configuredOverlaps.size() ? static_cast<int64_t>(configuredOverlaps[itkAxis]) : 0;
    }
    else
    {
      // A patch size of 0 means the whole extent on that axis: one patch, nothing to blend with,
      // so the overlap is inert there whatever it was configured to.
      patchSize[d] = inputShape[d];
      overlaps[d] = 0;
    }
  }

  const PatchGrid      grid = MakePatchGrid(inputShape, patchSize, overlaps);
  std::vector<int64_t> channelRepeat(dimension + 1, 1);
  channelRepeat[0] = config.GetNumberOfChannels();

  std::vector<Accumulator>   accumulators;
  std::vector<torch::Tensor> assembled;
  std::vector<int64_t>       layerShape;
  std::vector<int64_t>       layerPatchSize;
  std::vector<int64_t>       layerOverlaps;
  PatchGrid                  layerGrid;

  std::vector<int64_t> sweptIndex(sweptAxes, 0);
  for (int64_t slice = 0; slice < numberOfSlices; ++slice)
  {
    torch::Tensor sliceTensor = input;
    for (unsigned int a = 0; a < sweptAxes; ++a)
    {
      sliceTensor = sliceTensor.select(1, sweptIndex[a]);
    }

    accumulators.clear();

    for (std::size_t patchIndex = 0; patchIndex < grid.size(); ++patchIndex)
    {
      const torch::Tensor inputPatch = ExtractPatch(sliceTensor, grid, patchIndex, patchSize)
                                         .repeat({ torch::IntArrayRef(channelRepeat) })
                                         .unsqueeze(0)
                                         .to(device)
                                         .to(GetModelDtype(config));

      std::vector<torch::jit::IValue> outputsList;
      if (patchIndex == 0 && slice == 0)
      {
        try
        {
          outputsList = Forward(config, inputPatch);
        }
        catch (const std::exception & e)
        {
          itkGenericExceptionMacro(
            "IMPACT: the model " << config.GetModelPath()
            << " rejected its input. Check the number of channels (" << config.GetNumberOfChannels()
            << "), the patch size " << GetStringFromVector<int64_t>(config.GetPatchSize()) << " and the dimension ("
            << config.GetDimension() << ").\nDetails: " << e.what());
        }
        if (config.GetLayersMask().size() != outputsList.size())
        {
          itkGenericExceptionMacro("IMPACT: " << config.GetModelPath() << " declares " << config.GetLayersMask().size()
                                   << " layer mask entries but returned " << outputsList.size() << " layers.");
        }
      }
      else
      {
        outputsList = Forward(config, inputPatch);
      }

      std::size_t layerIndex = 0;
      for (std::size_t it = 0; it < outputsList.size(); ++it)
      {
        if (!config.GetLayersMask()[it])
        {
          continue;
        }
        const torch::Tensor layerPatch =
          outputsList[it].toTensor().squeeze(0).to(accumulateOn).to(torch::kFloat32);

        if (layerIndex == accumulators.size())
        {
          if (slice == 0)
          {
            // First patch of the first slice: the model has just told us what resolution it
            // returns this layer at, so size its grid now. Each axis is scaled by its own ratio --
            // a model may downsample anisotropically, and one scalar ratio would misplace the
            // other axes.
            layerPatchSize.assign(dimension, 0);
            layerOverlaps.assign(dimension, 0);
            layerShape.assign(dimension, 0);
            std::vector<double> scales(dimension);
            for (unsigned int d = 0; d < dimension; ++d)
            {
              layerPatchSize[d] = layerPatch.size(d + 1);
              scales[d] = static_cast<double>(layerPatchSize[d]) / static_cast<double>(patchSize[d]);
              // Rounded, not truncated, so the layer's stride is the input stride scaled; flooring
              // would let the two grids drift apart along the axis.
              layerOverlaps[d] = std::llround(overlaps[d] * scales[d]);
              if (layerOverlaps[d] >= layerPatchSize[d])
              {
                itkGenericExceptionMacro("IMPACT: the overlap (" << overlaps[d]
                                         << ") scaled to this layer's resolution (" << layerOverlaps[d]
                                         << ") is not smaller than the layer's patch extent ("
                                         << layerPatchSize[d] << ") along dimension " << d
                                         << ". Reduce the overlap or keep a finer layer.");
              }
            }
            layerGrid = ScalePatchGrid(grid, scales);
            for (unsigned int d = 0; d < dimension; ++d)
            {
              // The input extent at this layer's resolution, held to what its patches can cover:
              // at least one voxel of the last patch, at most the extent the grid spans.
              const int64_t lastStart = layerGrid.starts[d].back();
              const int64_t scaledExtent = static_cast<int64_t>(std::llround(inputShape[d] * scales[d]));
              layerShape[d] = std::min(std::max(scaledExtent, lastStart + 1), lastStart + layerPatchSize[d]);
            }

            std::vector<int64_t> shape{ layerPatch.size(0) };
            shape.insert(shape.end(), sweptShape.begin(), sweptShape.end());
            shape.insert(shape.end(), layerShape.begin(), layerShape.end());
            torch::Tensor destination = makeDestination ? makeDestination(layerIndex, shape) : torch::Tensor();
            if (!destination.defined())
            {
              destination = at::zeros(shape, layerPatch.options());
            }
            assembled.push_back(destination);
          }

          accumulators.emplace_back(
            layerShape, layerPatchSize, layerOverlaps, layerGrid, MakePathCombine(combine));

          // This slice's own sub-view of the layer's buffer.
          torch::Tensor destination = assembled[layerIndex];
          for (unsigned int a = 0; a < sweptAxes; ++a)
          {
            destination = destination.select(1, sweptIndex[a]);
          }
          accumulators.back().SetDestination(destination);
        }

        accumulators[layerIndex].AddLayer(patchIndex, layerPatch);
        ++layerIndex;
      }
    }

    for (std::size_t i = 0; i < accumulators.size(); ++i)
    {
      // A layer that did not see every patch would be handed back as a valid map of size zero.
      if (!accumulators[i].IsFull())
      {
        itkGenericExceptionMacro("IMPACT: layer " << i << " of " << config.GetModelPath()
                                 << " was never assembled: the accumulator expected more patches than the tiling "
                                    "produced. This is an internal inconsistency, not a configuration error.");
      }
    }

    // Advance the swept multi-index, last axis fastest.
    for (unsigned int a = sweptAxes; a-- > 0;)
    {
      if (++sweptIndex[a] < sweptShape[a])
      {
        break;
      }
      sweptIndex[a] = 0;
    }
  }

  return assembled;
}

namespace Impact
{

/** Fit a PCA basis on a feature tensor {C, spatial...} (no batch): channel-covariance
 * eigendecomposition, keep the top `newC` principal components. Returns the {C, newC} basis.
 * eigh returns ascending eigenvalues, so the largest components live at the end. */
inline torch::Tensor
PcaFit(const torch::Tensor & input, int64_t newC)
{
  const int64_t C = input.size(0);
  const int64_t N = input.numel() / C;
  torch::Tensor reshaped = input.reshape({ C, N });
  torch::Tensor centered = reshaped - reshaped.mean(1, /*keepdim=*/true);
  torch::Tensor covariance = torch::matmul(centered, centered.t()) / static_cast<double>(N - 1);
  torch::Tensor eigenvalues, eigenvectors;
  std::tie(eigenvalues, eigenvectors) = torch::linalg_eigh(covariance);
  return eigenvectors.narrow(1, C - newC, newC); // {C, newC}, largest-eigenvalue components
}

/** Project a feature tensor {C, spatial...} onto a PCA basis {C, K} -> {K, spatial...}.
 * The input is centred by its own channel mean, as PcaFit centred the data it fitted on. */
inline torch::Tensor
PcaTransform(const torch::Tensor & input, const torch::Tensor & basis)
{
  const int64_t C = input.size(0);
  const int64_t N = input.numel() / C;
  torch::Tensor reshaped = input.reshape({ C, N });
  torch::Tensor projected = torch::matmul(basis.t(), reshaped - reshaped.mean(1, /*keepdim=*/true)); // {K, N}
  std::vector<int64_t> shape;
  shape.push_back(basis.size(1));
  for (int64_t d = 1; d < input.dim(); ++d)
  {
    shape.push_back(input.size(d));
  }
  return projected.reshape(shape);
}

} // namespace Impact

} // namespace itk

#endif // itkImpactPatchTiling_h

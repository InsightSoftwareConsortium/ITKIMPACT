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
#ifndef itkImpactTorchRegistrationHelpers_h
#define itkImpactTorchRegistrationHelpers_h

// Shared LibTorch-backed helpers for the two Torch registration filters
// (ImpactFineRegistration fine stage and ImpactCoarseRegistration coarse stage):
//   - the itk image -> torch tensor bridge (axis-reversed, like itkImageToTensorFilter),
//   - the SINGLE SOURCE of the geometry convention mapping a voxel-space displacement tensor
//     {1, Dim, z,y,x} (component order z,y,x, units = fixed-grid voxels) to/from a
//     geometry-correct ITK displacement field (physical mm, ITK x,y,z, fixed->moving): the
//     x,y,z<->z,y,x axis reversal, the voxel<->mm scaling and the rotation by the fixed-image
//     direction cosines,
//   - whole-volume IMPACT feature extraction reusing itk::Forward.
// Pulls in LibTorch; included only from .hxx files, never part of the castxml surface.

#include "itkImpactPatchTiling.h"
#include "itkImpactModelConfigurationDetail.h"
#include "ImpactLoss.h"

#include <itkImage.h>
#include <itkVector.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkMacro.h>
#include <itkIdentityTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <tlhelp32.h> // enumerate loaded modules to find the Python C-API GIL primitives
#else
#  include <dlfcn.h> // runtime lookup of the optional Python C-API GIL primitives
#endif

namespace itk
{
namespace Impact
{

/** \class PythonGilReleaseGuard
 * \ingroup Impact
 * \brief RAII guard that releases the Python GIL for the lifetime of a LibTorch compute section.
 *
 * When the filters run under a Python interpreter, libtorch's default autograd engine is the
 * *Python* engine (libtorch_python is loaded), and it spawns worker threads that REQUIRE the
 * GIL to be released during loss.backward(); otherwise it aborts with "The autograd engine was
 * called while holding the GIL". The SWIG-wrapped Update() holds the GIL across GenerateData(),
 * so we drop it around our self-contained C++/torch work (which never calls back into Python)
 * and restore it on scope exit.
 *
 * The Python C-API GIL primitives are resolved at run time, so the module keeps NO build- or
 * link-time dependency on Python (and produces no undefined symbols on any linker). On POSIX
 * they are looked up in the process-global symbol table via dlsym; on Windows, where symbols
 * live per-module, the loaded modules are enumerated and the primitives are taken from whichever
 * one exports the CPython C-API (the hosting interpreter's pythonXY.dll). In a pure C++ process
 * (GTest, elastix, any non-Python consumer) libpython is not loaded, the lookups fail, and the
 * guard is a no-op; under a Python interpreter they resolve. PyThreadState* is ABI-compatible
 * with void* (a pointer), so no Python.h is needed. */
class PythonGilReleaseGuard
{
public:
  PythonGilReleaseGuard()
  {
#if defined(_WIN32)
    // Windows has no process-global symbol table to search, so walk the loaded modules and take
    // the GIL primitives from whichever one exports the CPython C-API (the interpreter's pythonXY.dll).
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snapshot != INVALID_HANDLE_VALUE)
    {
      MODULEENTRY32W entry;
      entry.dwSize = sizeof(entry);
      for (BOOL more = Module32FirstW(snapshot, &entry); more; more = Module32NextW(snapshot, &entry))
      {
        auto isInitialized = reinterpret_cast<int (*)()>(GetProcAddress(entry.hModule, "Py_IsInitialized"));
        auto saveThread = reinterpret_cast<void * (*)()>(GetProcAddress(entry.hModule, "PyEval_SaveThread"));
        auto restoreThread = reinterpret_cast<void (*)(void *)>(GetProcAddress(entry.hModule, "PyEval_RestoreThread"));
        if (isInitialized != nullptr && saveThread != nullptr && restoreThread != nullptr)
        {
          m_RestoreThread = restoreThread;
          if (isInitialized() != 0)
          {
            m_State = saveThread();
          }
          break;
        }
      }
      CloseHandle(snapshot);
    }
#else
    if (void * program = dlopen(nullptr, RTLD_LAZY))
    {
      auto isInitialized = reinterpret_cast<int (*)()>(dlsym(program, "Py_IsInitialized"));
      auto saveThread = reinterpret_cast<void * (*)()>(dlsym(program, "PyEval_SaveThread"));
      m_RestoreThread = reinterpret_cast<void (*)(void *)>(dlsym(program, "PyEval_RestoreThread"));
      dlclose(program);
      if (isInitialized != nullptr && saveThread != nullptr && m_RestoreThread != nullptr && isInitialized() != 0)
      {
        m_State = saveThread();
      }
    }
#endif
  }
  ~PythonGilReleaseGuard()
  {
    if (m_State != nullptr && m_RestoreThread != nullptr)
    {
      m_RestoreThread(m_State);
    }
  }
  PythonGilReleaseGuard(const PythonGilReleaseGuard &) = delete;
  PythonGilReleaseGuard & operator=(const PythonGilReleaseGuard &) = delete;

private:
  void * m_State{ nullptr };
  void (*m_RestoreThread)(void *){ nullptr };
};

/** Copy an itk image into a contiguous float CPU tensor of shape {spatial...} with the ITK
 * axis order reversed (so ITK x is the fastest/contiguous tensor axis), then add a leading
 * batch and channel dimension -> {1, 1, spatial...}. Matches itkImageToTensorFilter. */
template <typename TImage>
torch::Tensor
ImageToBatchTensor(const TImage * image)
{
  constexpr unsigned int Dim = TImage::ImageDimension;
  const auto             size = image->GetLargestPossibleRegion().GetSize();

  std::vector<int64_t> shape(Dim);
  int64_t              numberOfVoxels = 1;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    shape[d] = static_cast<int64_t>(size[Dim - 1 - d]); // reverse: tensor {z, y, x}
    numberOfVoxels *= shape[d];
  }

  std::vector<float>               buffer(static_cast<size_t>(numberOfVoxels));
  ImageRegionConstIterator<TImage> it(image, image->GetLargestPossibleRegion());
  size_t                           i = 0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) // raster order: ITK x varies fastest, matches buffer
  {
    buffer[i++] = static_cast<float>(it.Get());
  }

  // clone() so the tensor owns its memory once `buffer` goes out of scope.
  return torch::from_blob(buffer.data(), shape, torch::kFloat32).clone().unsqueeze(0).unsqueeze(0);
}

/** `image` ({1, C, spatial...}, spatial in tensor order, on a grid of `spacing` in ITK order) resampled to a model's
 * `voxelSize` (ITK order) before the model sees it, by linear interpolation and without smoothing: a Gaussian before
 * the downsampling was measured (elastix Static on MR/CT) to leave the Dice unchanged and fold more. Unchanged when no
 * voxel size is set (fewer than Dim entries, or one not positive) or when it gives the image's own size. The corners
 * stay aligned, which is how the registration filters place a feature layer of any size over the image. */
template <unsigned int Dim>
torch::Tensor
ResampleToVoxelSize(const torch::Tensor & image, const std::vector<double> & spacing, const std::vector<float> & voxelSize)
{
  namespace F = torch::nn::functional;
  if (spacing.size() != Dim || voxelSize.size() != Dim)
  {
    return image;
  }
  std::vector<int64_t> size(Dim);
  bool                 same = true;
  for (unsigned int i = 0; i < Dim; ++i) // tensor axis 2 + i holds ITK axis Dim - 1 - i
  {
    const double voxel = voxelSize[Dim - 1 - i];
    if (!(voxel > 0.0))
    {
      return image;
    }
    const int64_t extent = image.size(2 + static_cast<int64_t>(i));
    size[i] = std::max<int64_t>(1, std::llround(static_cast<double>(extent - 1) * spacing[Dim - 1 - i] / voxel) + 1);
    same = same && size[i] == extent;
  }
  if (same)
  {
    return image;
  }
  if constexpr (Dim == 3)
    return F::interpolate(image, F::InterpolateFuncOptions().size(size).mode(torch::kTrilinear).align_corners(true));
  else
    return F::interpolate(image, F::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(true));
}

/** Run the configured TorchScript models on a whole-volume image tensor and return the
 * selected (layersMask) feature-map layers, each {1, C, spatial...} float32 on `device`,
 * detached (the model is not differentiated through; only the warp is). Optionally selects a
 * channel subset. Layers are returned at their NATIVE model resolution -- a segmentation-style
 * backbone may emit downsampled deeper layers -- and the caller brings each to the grid it needs
 * (the fine filter resamples its sampling grid to the layer; the coarse stage pools each layer to
 * the common coarse grid), so features are never upsampled to full res here. Given the image's
 * `imageSpacing` (ITK order), each model sees the image resampled to its configured voxel size
 * (ResampleToVoxelSize); without it, or with no voxel size set, it sees the image as it is.
 * A model of lower dimension is swept along ITK axis `sweepAxis` (-1: the last one; see
 * HeadFeetAxis), and its layers keep that axis at the image's resolution. */
template <unsigned int Dim>
std::vector<torch::Tensor>
ExtractFeatureLayers(const std::vector<ImpactModelConfiguration> & configs,
                     const torch::Tensor &                         imageTensor, // {1,1,spatial...} on device
                     const torch::Device &                         device,
                     const std::vector<unsigned int> &             subset,
                     bool                                          withGrad = false,
                     const std::vector<double> &                   imageSpacing = {},
                     int                                           sweepAxis = -1)
{
  // withGrad=false (default): inference, features are constants (coarse stage, frozen-feature fine).
  // withGrad=true: keep the autograd graph so a caller can backpropagate a loss THROUGH the network to
  // the warped input (differentiable-feature fine mode), i.e. minimise F(warp(I)) not warp(F(I)).
  torch::AutoGradMode        gradMode(withGrad);
  std::vector<torch::Tensor> layers;

  torch::Tensor subsetIndex;
  if (!subset.empty())
  {
    std::vector<int64_t> idx(subset.begin(), subset.end());
    subsetIndex =
      torch::from_blob(idx.data(), { static_cast<int64_t>(idx.size()) }, torch::kLong).clone().to(device);
  }

  // Keep only the requested channels of a layer, validating the indices first.
  auto keepSubset = [&](torch::Tensor layer) {
    if (!subsetIndex.defined())
    {
      return layer;
    }
    // SubsetFeatures is an explicit list of channel INDICES (0-based). A raw index_select with an
    // out-of-range index triggers an asynchronous, context-poisoning CUDA device-side assert
    // ("srcIndex < srcSelectDimSize") that is impossible to attribute. Validate against this layer's
    // channel count first and fail with an actionable message instead.
    const int64_t channels = layer.size(1);
    for (const unsigned int index : subset)
    {
      if (static_cast<int64_t>(index) >= channels)
      {
        itkGenericExceptionMacro("ImpactRegistration: SubsetFeatures index "
                                 << index << " is out of range for a feature layer with " << channels
                                 << " channel(s). SubsetFeatures lists 0-based channel indices; leave it "
                                    "empty to keep all channels.");
      }
    }
    return layer.index_select(1, subsetIndex);
  };

  for (const auto & config : configs)
  {
    ModelTo(config, device);
    const torch::Tensor modelInput = ResampleToVoxelSize<Dim>(imageTensor, imageSpacing, config.GetVoxelSize());

    // A declared patch size means the model is meant to see the volume in pieces -- its trained
    // field of view, and bounded memory on a volume that does not fit whole. The tiling and its
    // blend are shared with itk::ImageToFeaturesMap (itkImpactPatchTiling.h) rather than written
    // twice. Not available with a live autograd graph: a tiled pass would hold every patch's
    // graph until the blend, so the differentiable-feature mode keeps the whole-volume forward.
    // With no patch declared the tiler runs the volume whole, and cuts it down if that does not fit.
    if (!withGrad)
    {
      for (torch::Tensor & map : RunTiledModel(config,
                                               modelInput.squeeze(0), // {1,C,spatial...} -> {C,spatial...}
                                               device,
                                               device, // stays on the device; no host round-trip
                                               PatchCombineModeFromString(config.GetPatchCombine()),
                                               {},
                                               sweepAxis))
      {
        layers.push_back(
          keepSubset(Impact::NormalizeFeatureChannels(map.unsqueeze(0), config.GetFeatureNormalization(), 1)).contiguous());
      }
      continue;
    }

    const int64_t numberOfChannels = static_cast<int64_t>(config.GetNumberOfChannels());

    torch::Tensor input = modelInput.to(GetModelDtype(config));
    if (numberOfChannels > 1)
    {
      std::vector<int64_t> repeats(Dim + 2, 1);
      repeats[1] = numberOfChannels;
      input = input.repeat(repeats);
    }

    // A model of lower dimension than the image (a 2D backbone like SAM on a 3D volume) cannot take the
    // whole {1,C,z,y,x} tensor -- it expects {N,C,H,W}. Run it slice-by-slice over the leading spatial
    // axes it does not cover and stack the per-slice feature layers back.
    //
    // RunTiledModel sweeps the same axes and would do this, and with a patch size of 0 it is exact
    // to the voxel -- a single patch whose blend weight is one everywhere. It is deliberately not
    // used here: this branch also serves withGrad, whose caller re-runs the network inside the Adam
    // loop, and routing it through the accumulator would add a map-sized allocation and a second
    // pass over every feature per iteration to save thirty lines. The two sweeps must stay in step.
    const unsigned int              modelDim = config.GetDimension();
    std::vector<torch::jit::IValue> outputs;
    if (modelDim < Dim)
    {
      // Tensor dimension of the sweep axis in {1,C,z,y,x}: 2 for the last ITK axis.
      const int64_t sliceAxis =
        2 + static_cast<int64_t>(Dim) - 1 - (sweepAxis < 0 ? static_cast<int64_t>(Dim) - 1 : sweepAxis);
      const int64_t                           nSlices = input.size(sliceAxis);
      std::vector<std::vector<torch::Tensor>> perLayer; // [layer][slice], native model resolution
      for (int64_t s = 0; s < nSlices; ++s)
      {
        std::vector<torch::jit::IValue> sliceOut = Forward(config, input.select(sliceAxis, s));
        if (perLayer.empty())
        {
          perLayer.resize(sliceOut.size());
        }
        for (size_t i = 0; i < sliceOut.size(); ++i)
        {
          perLayer[i].push_back(sliceOut[i].toTensor().to(torch::kFloat32));
        }
      }
      // Restack each layer's slices along the sliced axis: {1,Cf,y',x'} * z -> {1,Cf,z,y',x'}.
      for (auto & slices : perLayer)
      {
        outputs.emplace_back(torch::stack(slices, sliceAxis));
      }
    }
    else
    {
      outputs = Forward(config, input);
    }
    const std::vector<bool> &       mask = config.GetLayersMask();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
      if (i < mask.size() && !mask[i])
      {
        continue;
      }
      torch::Tensor layer = keepSubset(
        Impact::NormalizeFeatureChannels(outputs[i].toTensor().to(torch::kFloat32), config.GetFeatureNormalization(), 1));
      // Keep the layer at its NATIVE resolution (see the function doc); the consumer resamples it.
      // Detach only in the no-grad path -- withGrad must preserve the graph back to `imageTensor`.
      layers.push_back(withGrad ? layer.contiguous() : layer.detach().contiguous());
    }
  }
  return layers;
}

/** Per kept layer of `configs`, in their flat order, the dimension PcaReduce reads a {C, spatial...}
 * layer's slices along: `Dim - sweepAxis` for a model swept along ITK axis `sweepAxis`, 0 for a
 * model of the image's dimension. */
template <unsigned int Dim>
std::vector<int64_t>
PcaSweepDimensions(const std::vector<ImpactModelConfiguration> & configs, unsigned int sweepAxis)
{
  std::vector<int64_t> dimensions;
  for (const auto & config : configs)
  {
    dimensions.insert(dimensions.end(),
                      NumberOfKeptLayers(config),
                      config.GetDimension() < Dim ? static_cast<int64_t>(Dim - sweepAxis) : 0);
  }
  return dimensions;
}

/** \name Voxel counts read in units of the finest voxel side
 * The registration stages count their cells, windows and steps in voxels of the fixed image's finest axis, s_min, and
 * derive each axis's own count from it, so that they are (nearly) isotropic in millimetres. On an isotropic image the
 * derived counts are the given ones on every axis. Every vector here runs in tensor order (z, y, x). */
/** @{ */
/** The voxel sides of `spacing` (ITK order, x first) in tensor order (z, y, x). */
template <unsigned int Dim, typename TSpacing>
std::vector<double>
TensorVoxelSides(const TSpacing & spacing)
{
  std::vector<double> sides(Dim);
  for (unsigned int a = 0; a < Dim; ++a)
  {
    sides[a] = spacing[Dim - 1 - a];
  }
  return sides;
}

/** Per axis, `count` voxels of the finest axis in voxels of axis a, max(1, round(count * s_min / s_a)). */
inline std::vector<int64_t>
IsotropicVoxelCounts(const std::vector<double> & sides, double count)
{
  const double         finest = *std::min_element(sides.begin(), sides.end());
  std::vector<int64_t> counts(sides.size());
  for (size_t a = 0; a < sides.size(); ++a)
  {
    counts[a] = std::max<int64_t>(1, std::llround(count * finest / sides[a]));
  }
  return counts;
}

/** Per axis, the number of cells of `cells[a]` voxels a search of `halfWidth` cells of `cellSize` voxels of the finest
 * axis reaches each way: ceil(R / (cells[a] * s_a)), R = halfWidth * cellSize * s_min mm. */
inline std::vector<int64_t>
CaptureHalfWidths(const std::vector<double> &  sides,
                  const std::vector<int64_t> & cells,
                  int64_t                      cellSize,
                  int64_t                      halfWidth)
{
  const double         finest = *std::min_element(sides.begin(), sides.end());
  std::vector<int64_t> halfWidths(sides.size());
  for (size_t a = 0; a < sides.size(); ++a)
  {
    // cell_ref / cell_a, exactly 1 on an isotropic image; the tolerance keeps 1 from rounding up.
    const double ratio = (static_cast<double>(cellSize) * finest) / (static_cast<double>(cells[a]) * sides[a]);
    halfWidths[a] = static_cast<int64_t>(std::ceil(static_cast<double>(halfWidth) * ratio - 1e-9));
  }
  return halfWidths;
}

/** Per axis, a window of `kernel` voxels along the finest axis in millimetres of a map of shape `mapShape` ({1, C,
 * spatial...}) over an image of `spatial` voxels of `sides` mm: along each axis the odd number of voxels nearest the
 * same length, the map's voxel side being the image's times its rounded pooling factor. `kernel` on every axis of an
 * isotropic map. */
inline std::vector<int64_t>
IsotropicWindow(const std::vector<double> &  sides,
                const std::vector<int64_t> & spatial,
                c10::IntArrayRef             mapShape,
                int64_t                      kernel)
{
  std::vector<double> side(sides.size());
  for (size_t a = 0; a < sides.size(); ++a)
  {
    const int64_t pooling =
      std::max<int64_t>(1, std::llround(static_cast<double>(spatial[a]) / static_cast<double>(mapShape[2 + a])));
    side[a] = sides[a] * static_cast<double>(pooling);
  }
  const double         smallest = *std::min_element(side.begin(), side.end());
  std::vector<int64_t> window(sides.size());
  for (size_t a = 0; a < sides.size(); ++a)
  {
    window[a] = 2 * static_cast<int64_t>(std::floor(static_cast<double>(kernel) * (smallest / side[a]) / 2.0)) + 1;
  }
  return window;
}
/** @} */

/** \name Masks
 * A voxel counts where the fixed mask and the moving mask warped by the current field both hold (>= 0.5), as the
 * FireANTs engine counts it. Masks travel as bool tensors {1, 1, spatial...}; the shares of a mask a coarser grid
 * pools are floats. */
/** @{ */
/** A mask image as a bool tensor {1, 1, spatial...} (tensor axis order) on `device`: in where the image is not 0. */
template <typename TMask>
torch::Tensor
MaskToTensor(const TMask * mask, const torch::Device & device)
{
  constexpr unsigned int Dim = TMask::ImageDimension;
  const auto             size = mask->GetLargestPossibleRegion().GetSize();
  std::vector<int64_t>   shape(Dim);
  int64_t                voxels = 1;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    shape[d] = static_cast<int64_t>(size[Dim - 1 - d]);
    voxels *= shape[d];
  }
  std::vector<uint8_t>            buffer(static_cast<size_t>(voxels));
  ImageRegionConstIterator<TMask> it(mask, mask->GetLargestPossibleRegion());
  size_t                          i = 0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) // raster order, ITK x fastest
  {
    buffer[i++] = it.Get() != 0;
  }
  // to() copies, so the tensor owns its memory once `buffer` is gone.
  return torch::from_blob(buffer.data(), shape, torch::kUInt8).to(device, torch::kBool).unsqueeze(0).unsqueeze(0);
}

/** `mask` as MaskToTensor gives it on `reference`'s grid, resampled there by nearest neighbour through the identity
 * when its own grid differs; undefined for no mask. */
template <typename TMask, typename TReference>
torch::Tensor
MaskOnGrid(const TMask * mask, const TReference * reference, const torch::Device & device)
{
  if (mask == nullptr)
  {
    return {};
  }
  if (mask->GetLargestPossibleRegion().GetSize() == reference->GetLargestPossibleRegion().GetSize() &&
      mask->GetSpacing() == reference->GetSpacing() && mask->GetOrigin() == reference->GetOrigin() &&
      mask->GetDirection() == reference->GetDirection())
  {
    return MaskToTensor(mask, device);
  }
  auto resample = ResampleImageFilter<TMask, TMask, double>::New();
  resample->SetInput(mask);
  resample->SetTransform(IdentityTransform<double, TMask::ImageDimension>::New());
  resample->SetInterpolator(NearestNeighborInterpolateImageFunction<TMask, double>::New());
  resample->SetUseReferenceImage(true);
  resample->SetReferenceImage(reference);
  resample->Update();
  return MaskToTensor(resample->GetOutput(), device);
}

/** The share of `mask` ({1, 1, spatial...}, bool) in each voxel of a grid of `size` voxels over the same extent
 * (adaptive average pooling), as floats; the mask itself, as floats, on its own grid. */
template <unsigned int Dim>
torch::Tensor
MaskShare(const torch::Tensor & mask, const std::vector<int64_t> & size)
{
  namespace F = torch::nn::functional;
  const torch::Tensor share = mask.to(torch::kFloat32);
  if (std::equal(size.begin(), size.end(), share.sizes().begin() + 2))
  {
    return share;
  }
  if constexpr (Dim == 3)
    return F::adaptive_avg_pool3d(share, F::AdaptiveAvgPool3dFuncOptions({ size[0], size[1], size[2] }));
  else
    return F::adaptive_avg_pool2d(share, F::AdaptiveAvgPool2dFuncOptions({ size[0], size[1] }));
}

/** `loss` between two feature maps on one grid ({1, C, spatial...}) over the voxels `mask` ({1, 1, spatial...}, bool)
 * keeps: a point-wise distance's terms averaged over them (a ratio of two sums), NCC's correlation over them, LNCC's
 * local terms averaged over them. 0 when the mask keeps none. Without a mask, the loss over every voxel. */
template <unsigned int Dim>
torch::Tensor
MapLoss(const Loss &                 loss,
        const torch::Tensor &        fixedMap,
        const torch::Tensor &        movingMap,
        const torch::Tensor &        mask,
        const std::vector<int64_t> & kernel)
{
  if (loss.IsSpatial())
  {
    return loss.forwardSpatial(fixedMap, movingMap, kernel, mask);
  }
  const int64_t channels = fixedMap.size(1);
  if (!mask.defined())
  {
    std::vector<int64_t> toChannelLast{ 0 };
    for (unsigned int d = 0; d < Dim; ++d)
    {
      toChannelLast.push_back(2 + static_cast<int64_t>(d));
    }
    toChannelLast.push_back(1);
    return loss.forwardValue(fixedMap.permute(toChannelLast).reshape({ -1, channels }),
                             movingMap.permute(toChannelLast).reshape({ -1, channels }));
  }
  if (loss.IsPerPointMean())
  {
    const torch::Tensor keep = mask.squeeze(1);
    return torch::where(keep, loss.forwardPoints(fixedMap, movingMap), 0.0).sum() / keep.sum().clamp_min(1);
  }
  // A statistic over the points (NCC): the rows of the kept voxels.
  const torch::Tensor rows = mask.flatten().nonzero().flatten();
  if (rows.numel() == 0)
  {
    return (movingMap * 0.0).sum();
  }
  return loss.forwardValue(fixedMap.reshape({ channels, -1 }).index_select(1, rows).t(),
                           movingMap.reshape({ channels, -1 }).index_select(1, rows).t());
}
/** @} */

/** Channel-first {1, Dim, z,y,x} -> channel-last {z,y,x, Dim} permutation (drops batch). */
template <unsigned int Dim>
inline std::vector<int64_t>
SqueezedChannelLastPermutation()
{
  std::vector<int64_t> perm;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    perm.push_back(1 + static_cast<int64_t>(d));
  }
  perm.push_back(0);
  return perm;
}

/** Write a voxel-space displacement tensor {1, Dim, z,y,x} (component order z,y,x, units =
 * fixed-grid voxels) into an allocated ITK displacement field on the fixed grid, converting
 * to physical millimetres in ITK x,y,z order: u = Direction * (Spacing .* voxelOffset_xyz),
 * where voxelOffset_xyz reverses the z,y,x components. This is the fixed->moving convention. */
template <unsigned int Dim>
void
WriteVoxelFieldToDisplacement(const torch::Tensor &                                            voxelField,
                              const typename Image<Vector<float, Dim>, Dim>::SpacingType &     spacing,
                              const typename Image<Vector<float, Dim>, Dim>::DirectionType &   direction,
                              Image<Vector<float, Dim>, Dim> *                                 output)
{
  using FieldType = Image<Vector<float, Dim>, Dim>;
  using VectorType = Vector<float, Dim>;

  const std::vector<int64_t> spatial(voxelField.sizes().begin() + 2, voxelField.sizes().end());
  torch::Tensor              dCpu =
    voxelField.detach().squeeze(0).permute(SqueezedChannelLastPermutation<Dim>()).contiguous().to(torch::kCPU);
  const float * dPtr = dCpu.data_ptr<float>();

  ImageRegionIteratorWithIndex<FieldType> oit(output, output->GetLargestPossibleRegion());
  for (oit.GoToBegin(); !oit.IsAtEnd(); ++oit)
  {
    const auto idx = oit.GetIndex();
    size_t     base = 0;
    for (unsigned int t = 0; t < Dim; ++t)
    {
      base = base * static_cast<size_t>(spatial[t]) + static_cast<size_t>(idx[Dim - 1 - t]);
    }
    const float * v = dPtr + base * Dim; // components in (z, y, x) order

    VectorType physical;
    for (unsigned int r = 0; r < Dim; ++r)
    {
      double accumulator = 0.0;
      for (unsigned int c = 0; c < Dim; ++c)
      {
        accumulator += direction[r][c] * spacing[c] * static_cast<double>(v[Dim - 1 - c]);
      }
      physical[r] = accumulator;
    }
    oit.Set(physical);
  }
}

/** Inverse of WriteVoxelFieldToDisplacement: read an ITK displacement field (physical mm,
 * ITK x,y,z) into a voxel-space tensor {1, Dim, z,y,x} (component order z,y,x, units =
 * fixed-grid voxels) on `device`. voxelOffset = Direction^T * u / Spacing (Direction is
 * orthonormal). Used to ingest an initial/warm-start field. */
template <unsigned int Dim>
torch::Tensor
DisplacementToVoxelField(const Image<Vector<float, Dim>, Dim> *                          field,
                         const typename Image<Vector<float, Dim>, Dim>::SpacingType &    spacing,
                         const typename Image<Vector<float, Dim>, Dim>::DirectionType &  direction,
                         const torch::Device &                                           device)
{
  using FieldType = Image<Vector<float, Dim>, Dim>;
  const auto size = field->GetLargestPossibleRegion().GetSize();

  std::vector<int64_t> spatial(Dim);
  int64_t              numberOfVoxels = 1;
  for (unsigned int d = 0; d < Dim; ++d)
  {
    spatial[d] = static_cast<int64_t>(size[Dim - 1 - d]); // {z, y, x}
    numberOfVoxels *= spatial[d];
  }

  // Channel-last {z,y,x, Dim} buffer (component order z,y,x), x fastest.
  std::vector<float>                           buffer(static_cast<size_t>(numberOfVoxels) * Dim);
  ImageRegionConstIteratorWithIndex<FieldType> it(field, field->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto idx = it.GetIndex();
    size_t     base = 0;
    for (unsigned int t = 0; t < Dim; ++t)
    {
      base = base * static_cast<size_t>(spatial[t]) + static_cast<size_t>(idx[Dim - 1 - t]);
    }
    const auto u = it.Get(); // physical mm, ITK x,y,z
    for (unsigned int ch = 0; ch < Dim; ++ch)
    {
      const unsigned int c = Dim - 1 - ch; // tensor channel (z,y,x) -> ITK axis (x,y,z)
      double             accumulator = 0.0;
      for (unsigned int r = 0; r < Dim; ++r)
      {
        accumulator += direction[r][c] * static_cast<double>(u[r]); // (Direction^T u)_c
      }
      buffer[base * Dim + ch] = static_cast<float>(accumulator / spacing[c]);
    }
  }

  std::vector<int64_t> channelLastShape(spatial);
  channelLastShape.push_back(static_cast<int64_t>(Dim));
  torch::Tensor channelLast = torch::from_blob(buffer.data(), channelLastShape, torch::kFloat32).clone();

  // {z,y,x, Dim} -> {Dim, z,y,x} -> {1, Dim, z,y,x}
  std::vector<int64_t> toChannelFirst;
  toChannelFirst.push_back(static_cast<int64_t>(Dim));
  for (unsigned int d = 0; d < Dim; ++d)
  {
    toChannelFirst.push_back(static_cast<int64_t>(d));
  }
  return channelLast.permute(toChannelFirst).unsqueeze(0).contiguous().to(device);
}

} // namespace Impact
} // namespace itk

#endif // itkImpactTorchRegistrationHelpers_h

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

#include "itkModelConfigurationDetail.h"

#include <itkImage.h>
#include <itkVector.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkMacro.h>

#include <torch/torch.h>

#include <vector>

#if !defined(_MSC_VER)
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
 * The Python C-API GIL primitives are resolved at run time via dlsym, so the module keeps NO
 * build- or link-time dependency on Python (and produces no undefined symbols on any linker).
 * In a pure C++ process (GTest, elastix, any non-Python consumer) libpython is not loaded, the
 * lookups return null, and the guard is a no-op; under a Python interpreter they resolve
 * globally. PyThreadState* is ABI-compatible with void* (a pointer), so no Python.h is needed.
 * On MSVC the guard is unconditionally a no-op. */
class PythonGilReleaseGuard
{
public:
  PythonGilReleaseGuard()
  {
#if !defined(_MSC_VER)
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
#if !defined(_MSC_VER)
    if (m_State != nullptr && m_RestoreThread != nullptr)
    {
      m_RestoreThread(m_State);
    }
#endif
  }
  PythonGilReleaseGuard(const PythonGilReleaseGuard &) = delete;
  PythonGilReleaseGuard & operator=(const PythonGilReleaseGuard &) = delete;

private:
  void * m_State{ nullptr };
#if !defined(_MSC_VER)
  void (*m_RestoreThread)(void *){ nullptr };
#endif
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

/** Run the configured TorchScript models on a whole-volume image tensor and return the
 * selected (layersMask) feature-map layers, each {1, C, spatial...} float32 on `device`,
 * detached (the model is not differentiated through; only the warp is). Optionally selects a
 * channel subset. Layers are returned at their NATIVE model resolution -- a segmentation-style
 * backbone may emit downsampled deeper layers -- and the caller brings each to the grid it needs
 * (the fine filter resamples its sampling grid to the layer; the coarse stage pools each layer to
 * the common coarse grid), so features are never upsampled to full res here. */
template <unsigned int Dim>
std::vector<torch::Tensor>
ExtractFeatureLayers(const std::vector<ModelConfiguration> & configs,
                     const torch::Tensor &                   imageTensor, // {1,1,spatial...} on device
                     const torch::Device &                   device,
                     const std::vector<unsigned int> &       subset)
{
  torch::NoGradGuard         noGrad;
  std::vector<torch::Tensor> layers;

  torch::Tensor subsetIndex;
  if (!subset.empty())
  {
    std::vector<int64_t> idx(subset.begin(), subset.end());
    subsetIndex =
      torch::from_blob(idx.data(), { static_cast<int64_t>(idx.size()) }, torch::kLong).clone().to(device);
  }

  for (const auto & config : configs)
  {
    ModelTo(config, device);
    const int64_t numberOfChannels = static_cast<int64_t>(config.GetNumberOfChannels());

    torch::Tensor input = imageTensor.to(GetModelDtype(config));
    if (numberOfChannels > 1)
    {
      std::vector<int64_t> repeats(Dim + 2, 1);
      repeats[1] = numberOfChannels;
      input = input.repeat(repeats);
    }

    std::vector<torch::jit::IValue> outputs = Forward(config, input);
    const std::vector<bool> &       mask = config.GetLayersMask();
    for (size_t i = 0; i < outputs.size(); ++i)
    {
      if (i < mask.size() && !mask[i])
      {
        continue;
      }
      torch::Tensor layer = outputs[i].toTensor().to(torch::kFloat32);
      if (subsetIndex.defined())
      {
        layer = layer.index_select(1, subsetIndex);
      }
      // Keep the layer at its NATIVE resolution (see the function doc); the consumer resamples it.
      layers.push_back(layer.detach().contiguous());
    }
  }
  return layers;
}

/** Fit a PCA basis on a feature tensor {C, spatial...} (no batch): channel-covariance
 * eigendecomposition, keep the top `newC` principal components. Returns the {C, newC} basis.
 * Matches itkImageToFeaturesMap::pca_fit (eigh is ascending, so the largest live at the end). */
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
 * Matches itkImageToFeaturesMap::pca_transform (input centered by its own channel mean). */
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

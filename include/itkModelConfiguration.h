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

#ifndef itkModelConfiguration_h
#define itkModelConfiguration_h

// Intentionally free of any LibTorch dependency so castxml can parse it and expose the
// class to Python (WrapITK). The TorchScript model and all other torch state live in an
// opaque ModelConfigurationImpl, reached only from translation units that already depend
// on LibTorch via itkModelConfigurationDetail.h.

#include "ImpactExport.h" // Impact_EXPORT: export ModelConfiguration from the Impact shared lib (Windows DLL declspec)

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

/**
 * ******************* GetStringFromVector ***********************
 */
template <typename T>
std::string
GetStringFromVector(const std::vector<T> & vec)
{
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < vec.size(); ++i)
  {
    ss << vec[i];
    if (i != vec.size() - 1)
    {
      ss << " ";
    }
  }
  ss << ")";
  return ss.str();
} // end GetStringFromVector

namespace itk
{
namespace detail
{
struct ModelConfigurationImpl;
} // namespace detail

/**
 * Configuration for a TorchScript model used to extract semantic features.
 *
 * Holds the model path, input channel count, patch size, voxel size, overlap and the
 * per-layer mask, plus an opaque handle to the loaded TorchScript module. The public
 * interface uses only POD/STL types so the class is wrappable to Python; the LibTorch
 * module is reached through the accessors in itkModelConfigurationDetail.h.
 */
struct Impact_EXPORT ModelConfiguration
{
public:
  /** Load a TorchScript model and store its configuration. The model is loaded on the
   * CPU; move it with ModelTo() (itkModelConfigurationDetail.h). */
  ModelConfiguration(std::string               modelPath,
                     unsigned int              dimension,
                     unsigned int              numberOfChannels,
                     std::vector<unsigned int> patchSize,
                     std::vector<float>        voxelSize,
                     unsigned int              overlap,
                     std::vector<bool>         layersMask,
                     bool                      useMixedPrecision);

  /** An empty configuration with no associated model. */
  ModelConfiguration() = default;

  bool
  operator==(const ModelConfiguration & rhs) const
  {
    return m_modelPath == rhs.m_modelPath && m_dimension == rhs.m_dimension &&
           m_numberOfChannels == rhs.m_numberOfChannels && m_patchSize == rhs.m_patchSize &&
           m_voxelSize == rhs.m_voxelSize && m_layersMask == rhs.m_layersMask;
  }

  friend std::ostream &
  operator<<(std::ostream & os, const ModelConfiguration & config)
  {
    os << "\t\tPath : " << config.m_modelPath << "\n\t\tDimension : " << config.m_dimension
       << "\n\t\tNumberOfChannels : " << config.m_numberOfChannels
       << "\n\t\tPatchSize : " << GetStringFromVector<int64_t>(config.m_patchSize)
       << "\n\t\tVoxelSize : " << GetStringFromVector<float>(config.m_voxelSize)
       << "\n\t\tLayersMask : " << GetStringFromVector<bool>(config.m_layersMask);
    return os;
  }

  const std::string &
  GetModelPath() const
  {
    return m_modelPath;
  }
  unsigned int
  GetDimension() const
  {
    return m_dimension;
  }
  unsigned int
  GetNumberOfChannels() const
  {
    return m_numberOfChannels;
  }
  const std::vector<int64_t> &
  GetPatchSize() const
  {
    return m_patchSize;
  }
  const std::vector<float> &
  GetVoxelSize() const
  {
    return m_voxelSize;
  }
  const unsigned int &
  GetOverlap() const
  {
    return m_overlap;
  }
  const std::vector<bool> &
  GetLayersMask() const
  {
    return m_layersMask;
  }

  /** Precomputed physical patch offsets for the online (non-Static) inference: one inner
   * vector of length Dimension per patch voxel, holding (index - patchSize/2) * voxelSize
   * along each axis (x fastest). Filled by the constructor. POD only, so it stays
   * torch-free; the torch-side helpers live in itkModelConfigurationDetail.h. */
  const std::vector<std::vector<float>> &
  GetPatchIndex() const
  {
    return m_patchIndex;
  }

  /** Opaque handle to the loaded TorchScript module and dtype; nullptr for a
   * default-constructed configuration. Prefer the accessors in
   * itkModelConfigurationDetail.h; this pointer exists only to keep this header
   * free of LibTorch types. */
  detail::ModelConfigurationImpl *
  GetImpl() const
  {
    return m_impl.get();
  }

private:
  std::string          m_modelPath;
  unsigned int         m_dimension{ 0 };
  unsigned int         m_numberOfChannels{ 0 };
  std::vector<int64_t> m_patchSize;
  std::vector<float>   m_voxelSize;
  unsigned int         m_overlap{ 0 };
  std::vector<bool>    m_layersMask;
  // Precomputed physical patch offsets (torch-free); see GetPatchIndex().
  std::vector<std::vector<float>> m_patchIndex;
  // shared_ptr keeps the special members usable with an incomplete impl type, so this
  // header needs no out-of-line destructor and stays torch-free.
  std::shared_ptr<detail::ModelConfigurationImpl> m_impl;
};


} // end namespace itk

#endif // end #ifndef itkModelConfiguration_h

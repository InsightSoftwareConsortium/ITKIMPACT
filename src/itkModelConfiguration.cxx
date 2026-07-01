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

// The only LibTorch-dependent part of ModelConfiguration: loading the TorchScript model.
// Keeping it here (compiled, linking LibTorch) lets itkModelConfiguration.h stay
// torch-free and therefore Python-wrappable.

#include "itkModelConfigurationDetail.h"

namespace itk
{

ModelConfiguration::ModelConfiguration(std::string               modelPath,
                                       unsigned int              dimension,
                                       unsigned int              numberOfChannels,
                                       std::vector<unsigned int> patchSize,
                                       std::vector<float>        voxelSize,
                                       unsigned int              overlap,
                                       std::vector<bool>         layersMask,
                                       bool                      useMixedPrecision)
  : m_modelPath(modelPath)
  , m_dimension(dimension)
  , m_numberOfChannels(numberOfChannels)
  , m_patchSize(patchSize.begin(), patchSize.end())
  , m_voxelSize(voxelSize)
  , m_overlap(overlap)
  , m_layersMask(layersMask)
  , m_impl(std::make_shared<detail::ModelConfigurationImpl>())
{
  m_impl->dtype = useMixedPrecision ? torch::kFloat16 : torch::kFloat32;
  // The model runs with the default JIT graph-executor optimization (fusion) enabled.
  // Export models WITHOUT data-dependent control flow (e.g. a shape test feeding a
  // `prim::If`): such a graph can abort the profiling executor's TensorExpr fuser on some
  // LibTorch builds. Disabling the optimization globally to tolerate that slowed all
  // inference ~2x, so it is left on.
  m_impl->model = std::make_shared<torch::jit::script::Module>(
    torch::jit::load(m_modelPath, torch::Device(torch::kCPU)));
  m_impl->model->eval();
  m_impl->model->to(m_impl->dtype);

  // Introspect how many positional inputs forward expects (besides self) so Forward()
  // (itkModelConfigurationDetail.h) can pass the optional nLayers / image-metadata
  // arguments to metadata-aware models. The requested layer count is the mask size.
  m_impl->nArgs = m_impl->model->get_method("forward").function().getSchema().arguments().size() - 1;
  m_impl->nLayers = torch::tensor(static_cast<int64_t>(m_layersMask.size()), torch::kInt16);

  // Precompute the physical patch offsets for the online (non-Static) inference: a grid
  // over the patch, axis 0 varying fastest, each offset = (index - patchSize/2) * voxelSize
  // along that axis. Left empty when any patch dimension is 0 (i.e. Static mode).
  int64_t patchVoxels = 1;
  for (std::size_t d = 0; d < m_patchSize.size(); ++d)
  {
    patchVoxels *= m_patchSize[d];
  }
  if (patchVoxels > 0 && m_patchSize.size() == m_voxelSize.size())
  {
    m_patchIndex.reserve(static_cast<std::size_t>(patchVoxels));
    for (int64_t flat = 0; flat < patchVoxels; ++flat)
    {
      std::vector<float> offset(m_patchSize.size());
      int64_t            rem = flat;
      for (std::size_t d = 0; d < m_patchSize.size(); ++d) // axis 0 fastest
      {
        const int64_t idx = rem % m_patchSize[d];
        rem /= m_patchSize[d];
        offset[d] = static_cast<float>((idx - m_patchSize[d] / 2)) * m_voxelSize[d];
      }
      m_patchIndex.push_back(std::move(offset));
    }
  }
}

} // end namespace itk

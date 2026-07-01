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

#ifndef itkImageToFeaturesMapInternals_h
#define itkImageToFeaturesMapInternals_h

// Internal companion to itkImageToFeaturesMap.h. Pulls in LibTorch, the patch machinery
// (Patch/Accumulator/PathCombine, etc.) and the opaque per-instance torch state, keeping the
// public header torch-free and castxml/Python wrappable. Include ONLY from translation units
// that already depend on torch (itkImageToFeaturesMap.hxx, the metric .hxx, the C++ tests).

#include "itkImageToFeaturesMap.h"
#include "itkImageToTensorFilter.h"
#include "itkTensorToImageFilter.h"
#include <itkDanielssonDistanceMapImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

#include <torch/torch.h>

namespace itk
{

class PathCombine
{
public:
  using Slice = torch::indexing::Slice;
  using TensorIndex = torch::indexing::Slice;

  PathCombine(std::vector<int64_t> patchSize, unsigned int overlap)
    : m_PatchSize(patchSize)
    , m_Overlap(overlap)
  {}

  void
  setPatchConfig()
  {
    const size_t         dim = m_PatchSize.size();
    std::vector<int64_t> coreSize(dim);
    std::transform(m_PatchSize.begin(), m_PatchSize.end(), coreSize.begin(), [this](unsigned int s) {
      return static_cast<int64_t>(s - 2 * this->m_Overlap);
    });
    auto                 core = torch::ones(coreSize, torch::kFloat32);
    std::vector<int64_t> pad; // padding spec: [pad_left_D, pad_right_D, ..., pad_left_1, pad_right_1]
    for (auto it = m_PatchSize.rbegin(); it != m_PatchSize.rend(); ++it)
    {
      pad.push_back(static_cast<int64_t>(m_Overlap)); // pad left
      pad.push_back(static_cast<int64_t>(m_Overlap)); // pad right
    }

    m_Data = torch::constant_pad_nd(core, pad, 0.0f);
    m_Data = setFunction(m_Data);

    Slice A = Slice(0, m_Overlap);
    Slice B = Slice(-m_Overlap, torch::indexing::None);
    Slice C = Slice(m_Overlap, -m_Overlap);

    for (int i = 0; i < dim; ++i)
    {
      std::vector<std::vector<TensorIndex>> ab_opts(dim - i, { A, B });
      auto                                  slices_badge = cartesian_product(ab_opts);
      auto                                  combs = combinations(dim, i);

      for (const auto & idx_comb : combs)
      {
        std::vector<std::vector<at::indexing::TensorIndex>> full_slices;
        for (const auto & partial : slices_badge)
        {
          std::vector<at::indexing::TensorIndex> full;
          int                                    p = 0;
          for (int d = 0; d < dim; ++d)
          {
            if (std::find(idx_comb.begin(), idx_comb.end(), d) != idx_comb.end())
            {
              full.push_back(C);
            }
            else
            {
              full.push_back(partial[p++]);
            }
          }
          full_slices.push_back(full);
        }

        std::vector<torch::Tensor> patches;
        for (const auto & s : full_slices)
        {
          patches.push_back(m_Data.index(s));
        }


        auto normed = normalise(patches);
        for (size_t j = 0; j < full_slices.size(); ++j)
          m_Data.index_put_(full_slices[j], normed[j]);
      }
    }
  }


  virtual ~PathCombine() = default;

  torch::Tensor
  operator()(const torch::Tensor & input) const
  {
    std::vector<int64_t> repeatShape(input.dim(), 1);
    repeatShape[0] = input.size(0);
    auto weight = m_Data.unsqueeze(0).repeat({ torch::IntArrayRef(repeatShape) });
    return input * weight.to(input.device());
  }

protected:
  virtual torch::Tensor
  setFunction(const torch::Tensor & tensor) = 0;

  std::vector<std::vector<int>>
  combinations(int n, int k)
  {
    std::vector<std::vector<int>> result;
    std::vector<bool>             v(n);
    std::fill(v.begin(), v.begin() + k, true);
    do
    {
      std::vector<int> comb;
      for (int i = 0; i < n; ++i)
        if (v[i])
          comb.push_back(i);
      result.push_back(comb);
    } while (std::prev_permutation(v.begin(), v.end()));
    return result;
  }

  std::vector<torch::Tensor>
  normalise(const std::vector<torch::Tensor> & patches)
  {
    torch::Tensor              stacked = torch::stack(patches, 0);
    torch::Tensor              data_sum = torch::sum(stacked, 0);
    std::vector<torch::Tensor> normalized;
    normalized.reserve(patches.size());
    for (const auto & patch : patches)
    {
      normalized.push_back(patch / data_sum);
    }
    return normalized;
  }


  std::vector<std::vector<TensorIndex>>
  cartesian_product(const std::vector<std::vector<TensorIndex>> & options)
  {
    std::vector<std::vector<TensorIndex>>                          result;
    std::function<void(std::vector<TensorIndex>, size_t)>          recurse;

    recurse = [&](std::vector<TensorIndex> current, size_t depth) {
      if (depth == options.size())
      {
        result.push_back(current);
        return;
      }
      for (const auto & opt : options[depth])
      {
        auto next = current;
        next.push_back(opt);
        recurse(next, depth + 1);
      }
    };

    recurse({}, 0);
    return result;
  }

  torch::Tensor        m_Data;
  int64_t              m_Overlap;
  std::vector<int64_t> m_PatchSize;
};

class PathMean : public PathCombine
{
public:
  PathMean(const std::vector<int64_t> & patchSize, unsigned int overlap)
    : PathCombine(patchSize, overlap)
  {}

protected:
  torch::Tensor
  setFunction(const torch::Tensor & tensor) override
  {
    return torch::ones_like(tensor);
  }
};

template <unsigned int VImageDimension = 3>
class PathCosinus : public PathCombine
{
public:
  PathCosinus(const std::vector<int64_t> & patchSize, unsigned int overlap)
    : PathCombine(patchSize, overlap)
  {}

protected:
  torch::Tensor
  functionSides(const torch::Tensor & x, int overlap)
  {
    float factor = M_PI / (2.0f * (overlap + 1));
    return torch::clamp(torch::cos(factor * x), 0, 1);
  }

  torch::Tensor
  setFunction(const torch::Tensor & tensor) override
  {
    using InputImageType = itk::Image<float, VImageDimension>;
    using InputVectorImageType = itk::VectorImage<float, VImageDimension>;

    using InterpolatorType = itk::BSplineInterpolateImageFunction<InputImageType, double>;

    using TensorToImageFilterType = itk::TensorToImageFilter<VImageDimension>;
    using ImageToTensorFilterType = itk::ImageToTensorFilter<InputImageType, InterpolatorType>;


    auto tensorToImageFilter = TensorToImageFilterType::New();
    tensorToImageFilter->SetTensor(tensor.unsqueeze(0));
    tensorToImageFilter->Update();

    auto selector = itk::VectorIndexSelectionCastImageFilter<InputVectorImageType, InputImageType>::New();
    selector->SetInput(tensorToImageFilter->GetOutput());
    selector->SetIndex(0);
    selector->Update();

    auto filter = itk::DanielssonDistanceMapImageFilter<InputImageType, InputImageType>::New();
    filter->SetInput(selector->GetOutput());
    filter->Update();

    auto distanceImage = filter->GetOutput();

    auto converter = ImageToTensorFilterType::New();

    converter->AddInput(distanceImage);
    auto interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(3);

    converter->SetInterpolator(interpolator);
    converter->Update();
    torch::Tensor distanceTensor = converter->GetTensor();
    return functionSides(distanceTensor, this->m_Overlap);
  }
};

inline void
generateCartesianProduct(const std::vector<std::vector<int>> & startIndex,
                         std::vector<int> &                    current,
                         unsigned int                          depth,
                         std::vector<std::vector<int>> &       result)
{
  // Recursive function to compute the Cartesian product of multiple 1D index sets
  if (depth == startIndex.size())
  {
    result.push_back(current);
    return;
  }

  for (unsigned int val : startIndex[depth])
  {
    current[depth] = val;
    generateCartesianProduct(startIndex, current, depth + 1, result);
  }
} // end generateCartesianProduct

inline std::vector<std::vector<at::indexing::TensorIndex>>
getSlices(std::vector<int64_t> shape, std::vector<int64_t> patchSize, unsigned int overlap)
{
  std::vector<std::vector<int>> inputStartIndices(patchSize.size());
  for (unsigned int dim = 0; dim < patchSize.size(); ++dim)
  {
    for (int step = 0;
         step < std::ceil(shape[shape.size() - patchSize.size() + dim] / static_cast<float>(patchSize[dim]));
         ++step)
    {
      inputStartIndices[dim].push_back((patchSize[dim] - overlap) * step);
    }
  }
  std::vector<std::vector<int>> startIndices;
  std::vector<int>              inputCurrent(patchSize.size());
  generateCartesianProduct(inputStartIndices, inputCurrent, 0, startIndices);
  std::vector<std::vector<at::indexing::TensorIndex>> indices;
  for (auto & slice : startIndices)
  {
    std::vector<at::indexing::TensorIndex> indice = {
      torch::indexing::Slice(),
      torch::indexing::Slice(static_cast<int>(slice[0]), static_cast<int>(slice[0] + patchSize[0])),
      torch::indexing::Slice(static_cast<int>(slice[1]), static_cast<int>(slice[1] + patchSize[1]))
    };

    if (patchSize.size() == 3)
    {
      indice.push_back(torch::indexing::Slice(static_cast<int>(slice[2]), static_cast<int>(slice[2] + patchSize[2])));
    }
    indices.push_back(indice);
  }
  return indices;
}

class Patch
{
public:
  Patch(std::vector<int64_t> shape, std::vector<int64_t> patchSize, unsigned int overlap)
    : m_PatchSize(patchSize)
  {
    m_Slices = getSlices(shape, patchSize, overlap);
  }

  unsigned int
  size()
  {
    return m_Slices.size();
  }


  torch::Tensor
  GetData(torch::Tensor tensor, unsigned int index)
  {
    torch::Tensor        patch;
    std::vector<int64_t> padding;

    patch = tensor.index(m_Slices[index]);
    // Pad the patch if it's smaller than the expected size
    for (int i = tensor.dim() - 2; i >= 0; i--)
    {
      padding.push_back(0);
      padding.push_back(m_PatchSize[i] - patch.size(i + 1));
    }
    return torch::constant_pad_nd(patch, padding, 0);
  }

private:
  std::vector<int64_t>                                m_PatchSize;
  std::vector<std::vector<at::indexing::TensorIndex>> m_Slices;
};

class Accumulator
{
public:
  using Slice = at::indexing::Slice;
  using PatchSlice = std::vector<Slice>;

  Accumulator(std::vector<int64_t> shape, std::vector<int64_t> patchSize, unsigned int overlap, bool cosine)
    : m_Shape(shape)
    , m_Overlap(overlap)
  {
    m_Slices = getSlices(shape, patchSize, overlap);
    if (overlap > 0)
    {
      if (cosine)
      {
        if (shape.size() == 2)
        {
          m_PatchCombine = std::make_shared<PathCosinus<2>>(patchSize, overlap);
        }
        else
        {
          m_PatchCombine = std::make_shared<PathCosinus<3>>(patchSize, overlap);
        }
      }
      else
      {
        m_PatchCombine = std::make_shared<PathMean>(patchSize, overlap);
      }
      m_PatchCombine->setPatchConfig();
    }
    m_LayerAccumulator.resize(m_Slices.size(), {});
  }

  void
  addLayer(unsigned int index, const at::Tensor & layer)
  {
    m_LayerAccumulator[index] = layer;
  }

  bool
  isFull() const
  {
    return std::all_of(
      m_LayerAccumulator.begin(), m_LayerAccumulator.end(), [](const at::Tensor & t) { return t.defined(); });
  }

  torch::Tensor
  assemble()
  {
    const torch::Tensor & first = m_LayerAccumulator[0];
    auto                  device = first.device();
    auto                  dtype = first.dtype();

    std::vector<int64_t> fullShape;
    fullShape.reserve(1 + m_Shape.size());
    fullShape.push_back(first.size(0));
    fullShape.insert(fullShape.end(), m_Shape.begin(), m_Shape.end());

    torch::Tensor result = at::zeros(fullShape, at::TensorOptions().dtype(dtype).device(device));
    for (size_t i = 0; i < m_Slices.size(); ++i)
    {
      std::vector<at::indexing::TensorIndex> outputSlice = m_Slices[i];
      torch::Tensor                          data = m_LayerAccumulator[i];
      std::vector<at::indexing::TensorIndex> patchSlice;
      patchSlice.push_back(torch::indexing::Slice());
      for (int u = 0; u < fullShape.size() - 1; u++)
      {

        if (outputSlice[u + 1].slice().stop() > fullShape[u + 1])
        {
          patchSlice.push_back(torch::indexing::Slice(0, fullShape[u + 1] - outputSlice[u + 1].slice().start()));
        }
        else
        {
          patchSlice.push_back(torch::indexing::Slice());
        }
      }

      if (m_PatchCombine)
      {
        result.index_put_(outputSlice, result.index(outputSlice) + (*m_PatchCombine)(data).index(patchSlice));
      }
      else
      {
        result.index_put_(outputSlice, data.index(patchSlice));
      }
    }
    m_LayerAccumulator.clear();

    std::vector<at::indexing::TensorIndex> unpadding;
    unpadding.push_back(Slice());
    for (int it = 0; it < fullShape.size() - 1; ++it)
      // With overlap == 0 the crop must be the identity: Slice(0, -0) is Slice(0, 0), a
      // zero-length range in libtorch (not "keep all"), which would collapse the feature
      // map to size 0. Only strip the overlap margins when there is an overlap.
      unpadding.push_back(m_Overlap > 0 ? Slice(m_Overlap, -m_Overlap) : Slice());

    return result.index(unpadding);
  }

private:
  std::vector<std::vector<at::indexing::TensorIndex>> m_Slices;

  std::vector<at::Tensor>       m_LayerAccumulator;
  std::shared_ptr<PathCombine>  m_PatchCombine;
  std::vector<int64_t>          m_Shape;
  int64_t                       m_Overlap;
};

namespace detail
{
/** Per-instance LibTorch state held opaquely by ImageToFeaturesMap. */
struct ImageToFeaturesMapInternals
{
  std::vector<torch::Tensor> inputsTensor;
  std::vector<torch::Tensor> principalComponents;
};
} // namespace detail

/** The interpolated input tensors (one per AddInput call). */
template <typename TInputImage, typename TInterpolator>
inline torch::Tensor
GetTensorInput(const ImageToFeaturesMap<TInputImage, TInterpolator> & fe, unsigned int index)
{
  return fe.GetInternals()->inputsTensor[index];
}

/** Per-output-layer PCA bases (empty when PCA is disabled or not yet fitted). */
template <typename TInputImage, typename TInterpolator>
inline const std::vector<torch::Tensor> &
GetPrincipalComponents(const ImageToFeaturesMap<TInputImage, TInterpolator> & fe)
{
  return fe.GetInternals()->principalComponents;
}

/** Inject PCA bases so a moving image reuses the basis fitted on the fixed image. */
template <typename TInputImage, typename TInterpolator>
inline void
SetPrincipalComponents(ImageToFeaturesMap<TInputImage, TInterpolator> & fe,
                       const std::vector<torch::Tensor> &               principalComponents)
{
  fe.GetInternals()->principalComponents = principalComponents;
}

} // end namespace itk

#endif // end #ifndef itkImageToFeaturesMapInternals_h

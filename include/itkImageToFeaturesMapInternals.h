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

// Internal companion to itkImageToFeaturesMap.h: the opaque per-instance torch state and the
// bridge from an assembled feature map to an itk::VectorImage, so the public header stays
// torch-free and castxml/Python wrappable. Include ONLY from translation units that already
// depend on torch (itkImageToFeaturesMap.hxx, the metric .hxx, the C++ tests).
//
// The tiling itself lives in itkImpactPatchTiling.h, which the registration filters share.

#include "itkImpactPatchTiling.h"
#include "itkImageToTensorFilter.h"
#include "itkTensorToImageFilter.h"
#include <itkImage.h>
#include <itkMath.h>
#include <itkVectorImage.h>

#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace itk
{

/** Give `image` the geometry a feature map of `shape` ([channels, spatial...]) has on
 * `reference`'s grid, allocate it zeroed, and return a torch view of its pixel buffer in the
 * tensor's layout, so a producer can write the map in place.
 *
 * The geometry rule is TensorToImageFilter's and must stay it: spacing is the reference's
 * physical extent over the map's own size, which is how a model's stride is recovered (a /2
 * encoder halves the size, so the spacing doubles). ImageToTensorFilter samples at exactly that
 * step. A test pins the equivalence. */
template <unsigned int VImageDimension, typename TReferenceImage>
inline torch::Tensor
AllocateFeatureImage(const TReferenceImage *               reference,
                     const std::vector<int64_t> &          shape,
                     VectorImage<float, VImageDimension> * image)
{
  using ImageType = VectorImage<float, VImageDimension>;

  typename ImageType::SizeType size;
  for (unsigned int s = 0; s < VImageDimension; ++s)
  {
    // The tensor's axis order is the reverse of ITK's (see ImageToTensorFilter).
    size[s] = static_cast<SizeValueType>(shape[VImageDimension - s]);
  }
  image->SetRegions(typename ImageType::RegionType(size));
  image->SetVectorLength(static_cast<unsigned int>(shape[0]));

  typename ImageType::SpacingType spacing;
  const auto &                    referenceSize = reference->GetLargestPossibleRegion().GetSize();
  const auto &                    referenceSpacing = reference->GetSpacing();
  for (unsigned int i = 0; i < VImageDimension; ++i)
  {
    spacing[i] = referenceSize[i] * referenceSpacing[i] / size[i];
  }
  image->SetSpacing(spacing);
  image->SetOrigin(reference->GetOrigin());
  image->SetDirection(reference->GetDirection());
  // Zeroed because the blend adds into it; Allocate() alone leaves the buffer indeterminate.
  image->Allocate(true);

  // A VectorImage stores the component index as its fastest-varying axis, so its buffer is a
  // contiguous [spatial..., channels] tensor; permuting gives the [channels, spatial...] layout
  // the model works in. from_blob does not own the buffer: the view must not outlive the image.
  std::vector<int64_t> bufferShape;
  bufferShape.reserve(VImageDimension + 1);
  for (unsigned int s = 0; s < VImageDimension; ++s)
  {
    bufferShape.push_back(shape[1 + s]);
  }
  bufferShape.push_back(shape[0]);

  std::vector<int64_t> order;
  order.reserve(VImageDimension + 1);
  order.push_back(static_cast<int64_t>(VImageDimension));
  for (int64_t d = 0; d < static_cast<int64_t>(VImageDimension); ++d)
  {
    order.push_back(d);
  }
  return torch::from_blob(image->GetBufferPointer(), bufferShape, torch::kFloat32).permute(order);
}

namespace detail
{
/** Per-instance LibTorch state held opaquely by ImageToFeaturesMap. */
struct ImageToFeaturesMapInternals
{
  std::vector<torch::Tensor> inputsTensor;
  std::vector<torch::Tensor> principalComponents;
};
} // namespace detail

} // namespace itk

// itkImageToFeaturesMap.h pulls its .hxx, which needs the patch machinery and the complete
// Internals struct defined above -- include it only after those exist, so this header stays
// self-contained when it is included before itkImageToFeaturesMap.h.
#include "itkImageToFeaturesMap.h"

namespace itk
{

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

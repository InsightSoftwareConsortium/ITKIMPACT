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

#ifndef itkImageToTensorFilter_hxx
#define itkImageToTensorFilter_hxx

#include <vector>

namespace itk
{

template <typename TInputImage, typename TInterpolator>
ImageToTensorFilter<TInputImage, TInterpolator>::ImageToTensorFilter()
{
  // The single output is a decorator that carries the torch::Tensor handle.
  this->SetNumberOfRequiredOutputs(1);
  this->SetNthOutput(0, this->MakeOutput(0));
}

template <typename TInputImage, typename TInterpolator>
void
ImageToTensorFilter<TInputImage, TInterpolator>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "OutputSpacing: " << m_OutputSpacing << std::endl;
  os << indent << "Interpolator: " << m_Interpolator.GetPointer() << std::endl;
  os << indent << "Transform set: " << (m_Transform ? "true" : "false") << std::endl;
}

template <typename TInputImage, typename TInterpolator>
void
ImageToTensorFilter<TInputImage, TInterpolator>::VerifyPreconditions() const
{
  Superclass::VerifyPreconditions();

  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  if (numberOfInputs == 0)
  {
    itkExceptionMacro("ImageToTensorFilter requires at least one input image.");
  }

  if (!m_Interpolator)
  {
    itkExceptionMacro("Interpolator must be set before Update().");
  }

  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    if (m_OutputSpacing[d] <= 0.0)
    {
      itkExceptionMacro("OutputSpacing[" << d << "] must be > 0, got " << m_OutputSpacing[d]);
    }
  }

  const auto * referenceImage = static_cast<const TInputImage *>(this->GetInput(0));
  const auto & referenceRegion = referenceImage->GetLargestPossibleRegion();
  const auto & referenceSpacing = referenceImage->GetSpacing();
  const auto & referenceOrigin = referenceImage->GetOrigin();
  const auto & referenceDirection = referenceImage->GetDirection();

  for (unsigned int i = 1; i < numberOfInputs; ++i)
  {
    const auto * image = static_cast<const TInputImage *>(this->GetInput(i));
    if (!image)
    {
      itkExceptionMacro("Input image " << i << " is null.");
    }

    if (image->GetLargestPossibleRegion().GetSize() != referenceRegion.GetSize())
    {
      itkExceptionMacro("All input images must have the same region size.");
    }

    if (image->GetSpacing() != referenceSpacing || image->GetOrigin() != referenceOrigin ||
        image->GetDirection() != referenceDirection)
    {
      itkWarningMacro("Input image " << i << " has different geometry; this is not handled explicitly.");
    }
  }
}

template <typename TInputImage, typename TInterpolator>
void
ImageToTensorFilter<TInputImage, TInterpolator>::GenerateData()
{
  const InputImageType * inputImage = this->GetInputImage(0);
  m_Interpolator->SetInputImage(inputImage);

  const typename InputImageType::SizeType    oldSize = inputImage->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::SpacingType oldSpacing = inputImage->GetSpacing();
  const typename InputImageType::PointType   origin = inputImage->GetOrigin();

  // Resampled grid size from the target voxel spacing.
  typename InputImageType::SizeType newSize;
  SizeValueType                     numberOfVoxels = 1;
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    newSize[d] = static_cast<SizeValueType>(oldSize[d] * oldSpacing[d] / m_OutputSpacing[d] + 0.5);
    numberOfVoxels *= newSize[d];
  }

  // Interpolated intensities, filled with the first image axis varying fastest.
  std::vector<float> buffer(static_cast<size_t>(numberOfVoxels), 0.0f);

  // Physical points are formed axis-aligned (origin + index * spacing), matching the Elastix
  // ImpactTensorUtils::ImageToTensor reference. Direction cosines are intentionally not applied,
  // to preserve numerical parity with that reference feature extraction.
  typename InputImageType::IndexType index;
  index.Fill(0);
  typename InputImageType::PointType point;
  for (SizeValueType linearIndex = 0; linearIndex < numberOfVoxels; ++linearIndex)
  {
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      point[d] = origin[d] + index[d] * m_OutputSpacing[d];
    }

    const auto sampledPoint = m_Transform ? m_Transform(point) : point;
    if (m_Interpolator->IsInsideBuffer(sampledPoint))
    {
      buffer[linearIndex] = static_cast<float>(m_Interpolator->Evaluate(sampledPoint));
    }

    // Increment the multi-index with the first axis varying fastest.
    for (unsigned int d = 0; d < ImageDimension; ++d)
    {
      if (++index[d] < static_cast<IndexValueType>(newSize[d]))
      {
        break;
      }
      index[d] = 0;
    }
  }

  // Tensor shape reverses the ITK index order (e.g. {z, y, x}) so the first image axis (x)
  // is the contiguous/fastest tensor dimension, consistent with the buffer fill order.
  std::vector<int64_t> tensorShape(ImageDimension);
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    tensorShape[d] = static_cast<int64_t>(newSize[ImageDimension - 1 - d]);
  }

  // clone() so the tensor owns its memory after `buffer` goes out of scope.
  torch::Tensor tensor = torch::from_blob(buffer.data(), tensorShape, torch::kFloat32).clone();

  this->GetOutputTensorDataObject()->Set(std::make_shared<TensorType>(tensor));
}

} // namespace itk

#endif

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

#include <itkContinuousIndex.h>
#include <itkMultiThreaderBase.h>

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

  // Resampled grid size from the target voxel spacing, and the spacing that grid actually
  // has. The two differ: newSize is rounded, so an integer number of samples cannot span the
  // input's extent at exactly m_OutputSpacing. Sample the grid newSize describes -- step
  // oldSize*oldSpacing/newSize -- rather than m_OutputSpacing, so that the grid is consistent
  // with its own size. TensorToImageFilter must derive the feature map's spacing as
  // extent/size (that is how a model's stride is recovered: a /2 encoder halves the size and
  // the spacing correctly doubles), so a sampler that steps by m_OutputSpacing instead makes
  // it declare a spacing the samples were never taken at: the error is zero at the origin and
  // grows linearly, up to half a feature voxel at the far edge, sliding the features off the
  // anatomy they were computed from. The price is that m_OutputSpacing is honoured only to
  // within that same rounding (relative error <= 1/(2*newSize)), and exactly when it divides
  // the extent -- notably whenever it equals the input spacing.
  typename InputImageType::SizeType    newSize;
  typename InputImageType::SpacingType newSpacing;
  SizeValueType                        numberOfVoxels = 1;
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    newSize[d] = static_cast<SizeValueType>(oldSize[d] * oldSpacing[d] / m_OutputSpacing[d] + 0.5);
    numberOfVoxels *= newSize[d];
    newSpacing[d] = newSize[d] > 0 ? oldSize[d] * oldSpacing[d] / newSize[d] : m_OutputSpacing[d];
  }

  // Interpolated intensities, filled with the first image axis varying fastest.
  std::vector<float> buffer(static_cast<size_t>(numberOfVoxels), 0.0f);

  // Sampling points go through the image's own index-to-physical map, so the direction cosines
  // are applied. The resampled grid shares the input's origin and orientation and differs only
  // in spacing, which is what scaling the continuous index by newSpacing/oldSpacing expresses.
  // Forming the points axis-aligned instead (origin + index * spacing) is correct only for an
  // identity direction, and fails silently otherwise: the points walk off along the wrong
  // physical axis, IsInsideBuffer() rejects them, and the model is handed an empty volume.
  //
  // The coordinate type comes from the image's own PointType, so the continuous index and the
  // point handed to TransformContinuousIndexToPhysicalPoint agree on it.
  //
  // Threaded over the samples -- the dominant cost of feeding a model a resampled volume. Each
  // sample derives its own index from the flat sample number instead of carrying one, so they
  // are independent; Evaluate() is called concurrently on one shared interpolator, which is the
  // contract every ITK InterpolateImageFunction meets (ResampleImageFilter does the same).
  using ContinuousIndexType = ContinuousIndex<typename InputImageType::PointType::ValueType, ImageDimension>;
  MultiThreaderBase::New()->ParallelizeArray(
    0,
    numberOfVoxels,
    [&](SizeValueType linearIndex) {
      ContinuousIndexType continuousIndex;
      SizeValueType       remainder = linearIndex;
      for (unsigned int d = 0; d < ImageDimension; ++d) // first axis varying fastest
      {
        continuousIndex[d] = (remainder % newSize[d]) * newSpacing[d] / oldSpacing[d];
        remainder /= newSize[d];
      }
      typename InputImageType::PointType point;
      inputImage->TransformContinuousIndexToPhysicalPoint(continuousIndex, point);

      const auto sampledPoint = m_Transform ? m_Transform(point) : point;
      if (m_Interpolator->IsInsideBuffer(sampledPoint))
      {
        buffer[linearIndex] = static_cast<float>(m_Interpolator->Evaluate(sampledPoint));
      }
    },
    nullptr);

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

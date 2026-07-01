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

#ifndef itkTensorToImageFilter_hxx
#define itkTensorToImageFilter_hxx

#include <itkObjectFactory.h>
#include <itkMath.h>

namespace itk
{

template <unsigned int VImageDimension>
void
TensorToImageFilter<VImageDimension>::PrintSelf(std::ostream & os, Indent indent) const
{
  int i;
  Superclass::PrintSelf(os, indent);
  os << indent << "Spacing: [";
  for (i = 0; i < static_cast<int>(VImageDimension) - 1; ++i)
  {
    os << m_Spacing[i] << ", ";
  }
  os << m_Spacing[i] << ']' << std::endl;

  os << indent << "Origin: [";
  for (i = 0; i < static_cast<int>(VImageDimension) - 1; ++i)
  {
    os << m_Origin[i] << ", ";
  }
  os << m_Origin[i] << ']' << std::endl;
  os << indent << "Size: [";
  for (i = 0; i < static_cast<int>(VImageDimension) - 1; ++i)
  {
    os << m_Size[i] << ", ";
  }
  os << m_Size[i] << ']' << std::endl;
  os << indent << "Direction: " << std::endl << this->GetDirection() << std::endl;
}

template <unsigned int VImageDimension>
void
TensorToImageFilter<VImageDimension>::SetTensor(torch::Tensor tensor)
{
  m_tensor = tensor;
  this->Modified();
}

template <unsigned int VImageDimension>
torch::Tensor
TensorToImageFilter<VImageDimension>::GetTensor()
{
  return m_tensor;
}

template <unsigned int VImageDimension>
void
TensorToImageFilter<VImageDimension>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  OutputImagePointer outputPtr = this->GetOutput();

  OriginType origin;
  SpacingType spacing;
  DirectionType direction;
  RegionType region;
  SizeType size;
  
  for (int s = 0; s < VImageDimension; ++s)
  {
    size[s] = m_tensor.size(VImageDimension - s);
  }
  region.SetSize(size);
  outputPtr->SetRegions(region);
  outputPtr->SetVectorLength(m_tensor.size(0));
  if (m_Size[0] > 0){
    for (int i = 0; i < VImageDimension; ++i)
    {
      spacing[i] = m_Size[i] * m_Spacing[i] / size[i];
    }
    outputPtr->SetSpacing(spacing);
  } else {
    outputPtr->SetSpacing(m_Spacing);
  }
  
  outputPtr->SetOrigin(m_Origin);
  outputPtr->SetDirection(m_Direction);
  outputPtr->Allocate();
}

template <unsigned int VImageDimension>
void
TensorToImageFilter<VImageDimension>::GenerateData()
{
  OutputImagePointer outputPtr = this->GetOutput();
  SizeType size = outputPtr->GetLargestPossibleRegion().GetSize();

  std::vector<int64_t> dims;
  for (int64_t i = 1; i < m_tensor.dim(); ++i)
  {
    dims.push_back(i);
  }
  dims.push_back(0);

  torch::Tensor layers = m_tensor.permute(dims).contiguous().to(torch::kFloat32);

  
  unsigned int numberOfChannels = outputPtr->GetVectorLength();
  const float *      layersData = layers.data_ptr<float>();
  const unsigned int rowStride = size[0] * numberOfChannels;
  const unsigned int sliceStride = size[0] * size[1] * numberOfChannels;

  // Write each pixel vector from the tensor into the ITK vector image format
  if (VImageDimension == 2)
  {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int x = 0; x < size[1]; ++x)
    {
      for (int y = 0; y < size[0]; ++y)
      {
        const float *                    pixelPtr = layersData + x * rowStride + y * numberOfChannels;
        itk::VariableLengthVector<float> variableLengthVector(numberOfChannels);
        for (unsigned int i = 0; i < numberOfChannels; ++i)
        {
          variableLengthVector[i] = pixelPtr[i];
        }
        IndexType index;
        index[0] = y;
        index[1] = x;
        outputPtr->SetPixel(index, variableLengthVector);
      }
    }
  }
  else
  {
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int x = 0; x < size[2]; ++x)
    {
      for (int y = 0; y < size[1]; ++y)
      {
        for (int z = 0; z < size[0]; ++z)
        {
          const float * pixelPtr = layersData + x * sliceStride + y * rowStride + z * numberOfChannels;
          VariableLengthVector<float> variableLengthVector(numberOfChannels);
          for (unsigned int i = 0; i < numberOfChannels; ++i)
          {
            variableLengthVector[i] = pixelPtr[i];
          }
          IndexType index;
          index[0] = z;
          index[1] = y;
          index[2] = x;
          outputPtr->SetPixel(index, variableLengthVector);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
template <unsigned int VImageDimension>
void
TensorToImageFilter<VImageDimension>::SetDirection(const DirectionType & direction)
{
  bool modified = false;

  for (unsigned int r = 0; r < VImageDimension; ++r)
  {
    for (unsigned int c = 0; c < VImageDimension; ++c)
    {
      if (Math::NotExactlyEquals(m_Direction[r][c], direction[r][c]))
      {
        m_Direction[r][c] = direction[r][c];
        modified = true;
      }
    }
  }
  if (modified)
  {
    this->Modified();
  }
}
} // end namespace itk

#endif

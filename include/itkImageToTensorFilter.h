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

#ifndef itkImageToTensorFilter_h
#define itkImageToTensorFilter_h

#include <itkProcessObject.h>
#include <torch/torch.h>
#include <itkSimpleDataObjectDecorator.h>

namespace itk
{
/** \class ImageToTensorFilter
 * \brief Resample an ITK image to a target voxel spacing and export it as a torch::Tensor.
 *
 * An optional spatial transform is applied to each point before interpolation.
 *
 * \ingroup Impact
 */

template <typename TInputImage, typename TInterpolator>
class ITK_TEMPLATE_EXPORT ImageToTensorFilter : public ProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageToTensorFilter);

  /** Standard class type aliases. */
  using Self = ImageToTensorFilter;
  using Superclass = ProcessObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(ImageToTensorFilter);

  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  using InputImageType = TInputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;

  using TensorType       = torch::Tensor;
  using TensorHandleType = std::shared_ptr<TensorType>;
  using TensorDataObject = itk::SimpleDataObjectDecorator<TensorHandleType>;
    
  using InterpolatorPointer = typename TInterpolator::Pointer;

  using InputImagePixelType = typename InputImageType::PixelType;
  using InputImageIndexType = typename InputImageType::IndexType;
  using InputImagePointType = typename InputImageType::PointType;
  using InputSpacingType = typename InputImageType::SpacingType;

  using Superclass::AddInput;
  using Superclass::MakeOutput;

  void
  AddInput(const InputImageType * image)
  {
    Superclass::AddInput(const_cast<InputImageType *>(image));
  }

  const InputImageType *
  GetInputImage(unsigned int idx) const
  {
    return static_cast<const InputImageType *>(this->GetInput(idx));
  }

  InputImageType *
  GetInputImage(unsigned int idx)
  {
    return static_cast<InputImageType *>(this->GetInput(idx));
  }

  itkSetMacro(OutputSpacing, InputSpacingType);
  itkGetConstReferenceMacro(OutputSpacing, InputSpacingType);
  itkSetVectorMacro(OutputSpacing, const float, ImageDimension);
  
  void SetInterpolator(typename TInterpolator::Pointer interp) { m_Interpolator = interp; }
  void SetTransform(std::function<InputImagePointType(const InputImagePointType &)> fct)
  {
    m_Transform = fct;
  }

  DataObject::Pointer
  MakeOutput(DataObjectPointerArraySizeType idx) override
  {
    if (idx == 0)
    {
      auto output = TensorDataObject::New();
      return output.GetPointer();
    }
    return nullptr;
  }

  TensorDataObject *
  GetOutputTensorDataObject()
  {
    return static_cast<TensorDataObject *>(this->GetOutput(0));
  }

  const TensorDataObject *
  GetOutputTensorDataObject() const
  {
    return static_cast<const TensorDataObject *>(this->GetOutput(0));
  }

  TensorType &
  GetTensor()
  {
    auto * obj = this->GetOutputTensorDataObject();
    auto & handle = obj->Get();
    return *handle;
  }

  const TensorType &
  GetTensor() const
  {
    const auto * obj = this->GetOutputTensorDataObject();
    const auto & handle = obj->Get();
    return *handle;
  }

protected:
  ImageToTensorFilter();
  ~ImageToTensorFilter() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;
  
  void
  VerifyPreconditions() const override;

  void GenerateData() override;

private:
  InterpolatorPointer m_Interpolator;
  InputSpacingType   m_OutputSpacing{ MakeFilled<InputSpacingType>(1.0) };
  std::function<InputImagePointType(const InputImagePointType &)> m_Transform;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageToTensorFilter.hxx"
#endif

#endif

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

#ifndef itkTensorToImageFilter_h
#define itkTensorToImageFilter_h

#include <itkImageSource.h>
#include <itkVectorImage.h>
#include <torch/torch.h>

namespace itk
{
/** \class TensorToImageFilter
 * \brief Wrap a torch::Tensor as an itk::VectorImage without copying the geometry from a tensor.
 *
 * The channel axis of the tensor becomes the pixel vector; the remaining axes become the spatial
 * grid. Geometry (spacing, origin, direction, size) is supplied explicitly or copied from a
 * reference image. As an ImageSource it behaves like any other pipeline object.
 *
 * \ingroup Impact
 */
template <unsigned int VImageDimension = 2>
class ITK_TEMPLATE_EXPORT TensorToImageFilter : public ImageSource<VectorImage<float, VImageDimension>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TensorToImageFilter);

  /** Typedef for the output image.   */
  using OutputImageType = VectorImage<float, VImageDimension>;
  using ReferenceImageType = ImageBase<VImageDimension>; // geometry only -> accepts any pixel type
  using OutputImagePointer = typename OutputImageType::Pointer;
  using ReferenceImageTypeConstPointer = typename ReferenceImageType::ConstPointer;
  using SpacingType = typename OutputImageType::SpacingType;
  using OriginType = typename OutputImageType::PointType;

  /** Standard class type aliases. */
  using Self = TensorToImageFilter;
  using Superclass = ImageSource<OutputImageType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(TensorToImageFilter);

  /** Index type alias support An index is used to access pixel values. */
  using IndexType = Index<VImageDimension>;

  /** Size type alias support A size is used to define region bounds. */
  using SizeType = Size<VImageDimension>;

  /** Region type alias support A region is used to specify a
   * subset of an image. */
  using RegionType = ImageRegion<VImageDimension>;

  /** Type of the output image pixel type. */
  using OutputImagePixelType = float;

  /** Get the tensor that will be wrapped as the output image. */
  torch::Tensor GetTensor();

  /** Set the tensor to wrap. Layout is (channels, spatial...); the channel axis becomes the
   * pixel vector length and the trailing axes the spatial grid. */
  void
  SetTensor(torch::Tensor tensor);

  /** Set the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  void
  SetRegion(const RegionType & region)
  {
    if (m_Region != region)
    {
      m_Region = region;
      this->Modified();
    }
  }

  /** Get the region object that defines the size and starting index
   * for the imported image. This will serve as the LargestPossibleRegion,
   * the BufferedRegion, and the RequestedRegion.
   * \sa ImageRegion */
  const RegionType &
  GetRegion() const
  {
    return m_Region;
  }

  void SetReferenceImage(ReferenceImageTypeConstPointer referenceImage)
  {
    m_Origin = referenceImage->GetOrigin();
    m_Spacing = referenceImage->GetSpacing();
    m_Direction = referenceImage->GetDirection();
    m_Size = referenceImage->GetLargestPossibleRegion().GetSize();
  }

  /** Set the spacing (size of a pixel) of the image.
   * \sa GetSpacing() */
  itkSetMacro(Spacing, SpacingType);
  itkGetConstReferenceMacro(Spacing, SpacingType);
  itkSetVectorMacro(Spacing, const float, VImageDimension);

  /** Set the origin of the image.
   * \sa GetOrigin() */
  itkSetMacro(Origin, OriginType);
  itkGetConstReferenceMacro(Origin, OriginType);
  itkSetVectorMacro(Origin, const float, VImageDimension);

  /** Set the size of the image.
   * \sa GetSize() */
  itkSetMacro(Size, SizeType);
  itkGetConstReferenceMacro(Size, SizeType);
  itkSetVectorMacro(Size, const float, VImageDimension);

  using DirectionType = Matrix<SpacePrecisionType, VImageDimension, VImageDimension>;
  
  /** Set the direction of the image
   * \sa GetDirection() */
  virtual void
  SetDirection(const DirectionType & direction);

  /**  Get the direction of the image
   * \sa SetDirection */
  itkGetConstReferenceMacro(Direction, DirectionType);

protected:
  TensorToImageFilter() = default;
  ~TensorToImageFilter() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** This filter does not actually "produce" any data, rather it "wraps"
   * the user supplied data into an itk::Image.  */
  void
  GenerateData() override;

  /** This is a source, so it must set the spacing, size, and largest possible
   * region for the output image that it will produce.
   * \sa ProcessObject::GenerateOutputInformation() */
  void
  GenerateOutputInformation() override;

private:
  RegionType    m_Region{};
  SpacingType   m_Spacing{ MakeFilled<SpacingType>(1.0) };
  OriginType    m_Origin{MakeFilled<SpacingType>(0)};
  DirectionType m_Direction{ DirectionType::GetIdentity() };
  SizeType      m_Size{};
  torch::Tensor m_tensor;
  
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTensorToImageFilter.hxx"
#endif

#endif

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

/**
 * \class InterpolateVectorImageFunction
 * \ingroup Impact
 * \brief Interpolate each channel of a VectorImage with its own B-Spline interpolator.
 *
 * ITK's BSplineInterpolateImageFunction does not support VectorImages, so one interpolator is
 * built per channel. Evaluation at arbitrary physical points returns the interpolated values or
 * spatial derivatives as torch tensors.
 */
#ifndef itkInterpolateVectorImageFunction_h
#define itkInterpolateVectorImageFunction_h

#include <itkVectorImage.h>
#include <torch/torch.h>
#include <itkInterpolateImageFunction.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

namespace itk
{


template <typename TImage, typename TInterpolator>
class ITK_TEMPLATE_EXPORT InterpolateVectorImageFunction 
  : public InterpolateImageFunction<typename TInterpolator::InputImageType, typename TInterpolator::CoordRepType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(InterpolateVectorImageFunction);

  /** Standard class type aliases. */
  using Self = InterpolateVectorImageFunction;
  using Superclass = InterpolateImageFunction<typename TInterpolator::InputImageType, typename TInterpolator::CoordRepType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(InterpolateVectorImageFunction);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  using ImageType = TImage;
  using InterpolatorType = TInterpolator;
  using InterpolatorPointer = typename InterpolatorType::Pointer;
  using ImagePointer = typename ImageType::Pointer;
  using PixelType = typename ImageType::PixelType;
  using ImagePointType = typename ImageType::PointType;

  /** Dimension underlying input image. */
  static constexpr unsigned int ImageDimension = Superclass::ImageDimension;
  
  using CovariantVectorType = itk::CovariantVector<float, ImageDimension>;
  

  InterpolateVectorImageFunction() = default;

  using Superclass::Evaluate;
  using Superclass::SetInputImage;

  /**
   * \brief Initialize one B-Spline interpolator per feature channel of the input VectorImage.
   *
   * \param vectorImage The input VectorImage whose channels are interpolated.
   */
  void
  SetInputImage(ImagePointer vectorImage);

  /**
   * \brief Interpolates the selected feature channels at a given physical point.
   *
   * \param point The physical coordinate where interpolation is performed.
   * \param subsetOfFeatures Indices of feature channels to interpolate.
   *
   * \return A 1D torch::Tensor containing interpolated values for the requested channels.
   */
  torch::Tensor
  Evaluate(ImagePointType point, std::vector<unsigned int> subsetOfFeatures) const;

  /**
   * \brief Evaluates the spatial derivative of selected features at a given point.
   *
   * Computes gradients of the selected feature channels with respect to spatial dimensions
   * using the underlying B-Spline interpolators.
   *
   * \param point The physical coordinate at which derivatives are computed.
   * \param subsetOfFeatures Indices of feature channels to differentiate.
   *
   * \return A 2D torch::Tensor (Channels × SpatialDimension) with spatial gradients per feature.
   */
  torch::Tensor
  EvaluateDerivative(ImagePointType point, std::vector<unsigned int> subsetOfFeatures) const;

  using OutputType = typename Superclass::OutputType;
using ContinuousIndexType = typename Superclass::ContinuousIndexType;
using SizeType = typename Superclass::SizeType;

  OutputType EvaluateAtContinuousIndex(const ContinuousIndexType &) const override
  {
    itkExceptionMacro("EvaluateAtContinuousIndex is not implemented.");
  }

  SizeType GetRadius() const override
  {
    return SizeType{};
  }

private:
  ImagePointer                     m_VectorImage;
  std::vector<InterpolatorPointer> m_Interpolators;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkInterpolateVectorImageFunction.hxx"
#endif

#endif // end #ifndef itkInterpolateVectorImageFunction_h

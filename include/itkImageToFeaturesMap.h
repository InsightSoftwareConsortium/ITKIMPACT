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

#ifndef itkImageToFeaturesMap_h
#define itkImageToFeaturesMap_h

// This header is intentionally free of any LibTorch dependency so castxml can parse it and
// WrapITK can expose it to Python. The TorchScript inference, the patch machinery and the
// per-instance torch state live in itkImageToFeaturesMapInternals.h, included only by the
// .hxx and other torch-aware translation units.

#include <itkProcessObject.h>
#include <itkVectorImage.h>
#include <itkModelConfiguration.h>
#include "itkBSplineInterpolateImageFunction.h"

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace itk
{
namespace detail
{
struct ImageToFeaturesMapInternals;
} // namespace detail

/**
 * \class ImageToFeaturesMap
 * \brief Applies a TorchScript model to an ITK image and outputs a feature map as itk::VectorImage.
 *
 * The filter wraps an ImageToTensor conversion, a TorchScript inference and a
 * TensorToImageFilter to obtain semantic feature maps. Its interface uses only ITK/STL
 * types (the model is configured through ModelConfiguration and the device through a
 * string), so the class is wrappable to Python; the torch state is held opaquely.
 *
 * \ingroup Impact
 */
template <typename TInputImage, typename TInterpolator>
class ImageToFeaturesMap : public ProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImageToFeaturesMap);

  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  using InputImageType = TInputImage;
  using OutputImageType = VectorImage<float, ImageDimension>;
  /** Standard class aliases. */
  using Self = ImageToFeaturesMap<InputImageType, TInterpolator>;
  using Superclass = ProcessObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information. */
  itkOverrideGetNameOfClassMacro(ImageToFeaturesMap);
  /** Standard New macro. */
  itkNewMacro(Self);

  using InterpolatorPointer = typename TInterpolator::Pointer;
  using InputImagePointType = typename InputImageType::PointType;
  using InputImageSpacingType = typename InputImageType::SpacingType;

  using OutputImageTypePointer = typename OutputImageType::Pointer;
  using Slice = std::vector<int>;


  void
  SetInterpolator(typename TInterpolator::Pointer interp)
  {
    m_Interpolator = interp;
  }

  /** Optional point transform applied while resampling the input into the model's patch
   * grid (e.g. to sample a moving image at transformed points). Set before AddInput().
   * Torch-free (only ITK point types), so this header stays Python-wrappable. */
  void
  SetTransform(std::function<InputImagePointType(const InputImagePointType &)> transform)
  {
    m_Transform = transform;
  }

  void
  SetModelConfiguration(const itk::ModelConfiguration & config)
  {
    m_ModelConfiguration = config;

    unsigned int nbOutputs =
      std::accumulate(m_ModelConfiguration.GetLayersMask().begin(), m_ModelConfiguration.GetLayersMask().end(), 0);

    this->SetNumberOfRequiredOutputs(nbOutputs);
    for (unsigned int i = 0; i < nbOutputs; ++i)
    {
      this->SetNthOutput(i, OutputImageType::New());
    }

    this->Modified();
  }

  itkGetConstReferenceMacro(ModelConfiguration, ModelConfiguration);

  /** The device the model runs on, as a string ("cpu", "cuda", "cuda:0", ...). */
  itkSetMacro(Device, std::string);
  itkGetConstMacro(Device, std::string);

  itkSetMacro(PCA, unsigned int);
  itkGetConstMacro(PCA, unsigned int);

  using Superclass::AddInput;

  void
  AddInput(const TInputImage * input);

  const TInputImage *
  GetInput(unsigned int index) const
  {
    return dynamic_cast<const TInputImage *>(this->ProcessObject::GetInput(index));
  }

  typename OutputImageType::ConstPointer
  GetOutput(unsigned int idx)
  {
    return static_cast<const OutputImageType *>(this->ProcessObject::GetOutput(idx));
  }

  typename OutputImageType::ConstPointer
  GetOutput(unsigned int idx) const
  {
    return static_cast<const OutputImageType *>(this->ProcessObject::GetOutput(idx));
  }

  /** Opaque handle to the per-instance LibTorch state. Use the accessors in
   * itkImageToFeaturesMapInternals.h rather than this pointer directly; it exists only
   * so this header carries no torch types. */
  detail::ImageToFeaturesMapInternals *
  GetInternals() const
  {
    return m_Internals.get();
  }

protected:
  ImageToFeaturesMap();

  ~ImageToFeaturesMap() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;


  void
  VerifyPreconditions() const override;

  /** Deliberately suppress the default output-information generation.
   *
   * ProcessObject::GenerateOutputInformation() copies the primary input's geometry
   * (origin/spacing/direction AND LargestPossibleRegion) onto every output. That is
   * wrong for this filter: the outputs are feature maps whose size and spacing are
   * data-dependent (decided by the model) and are set authoritatively in
   * GenerateData() by grafting the TensorToImageFilter result. Worse, when a
   * downstream filter re-propagates UpdateOutputInformation() up this filter (e.g. the
   * per-channel VectorIndexSelectionCast in InterpolateVectorImageFunction::SetInputImage),
   * the default would reset each output's LargestPossibleRegion back to the (larger)
   * input image size while the buffer still holds the smaller feature map -- the
   * consumer then iterates an input-sized region over the smaller buffer and throws
   * "Region ... is outside of buffered region". Overriding with an empty body keeps the
   * grafted regions intact. */
  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;


private:
  InterpolatorPointer m_Interpolator;
  ModelConfiguration  m_ModelConfiguration;
  std::string         m_Device = "cpu";
  unsigned int        m_PCA = 0;
  std::function<InputImagePointType(const InputImagePointType &)> m_Transform;
  // shared_ptr to an incomplete type keeps all special members usable in this
  // torch-free header; the struct is defined in itkImageToFeaturesMapInternals.h.
  std::shared_ptr<detail::ImageToFeaturesMapInternals> m_Internals;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageToFeaturesMap.hxx"
#endif

#endif // itkImageToFeaturesMap

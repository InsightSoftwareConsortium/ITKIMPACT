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

#ifndef itkImageToFeaturesMap_hxx
#define itkImageToFeaturesMap_hxx

#include "itkImpactModelConfigurationDetail.h"
#include "itkImageToFeaturesMapInternals.h"

#include <algorithm>
#include <cmath>

namespace itk
{

template <typename TInputImage, typename TInterpolator>
ImageToFeaturesMap<TInputImage, TInterpolator>::ImageToFeaturesMap()
  : m_Internals(std::make_shared<detail::ImageToFeaturesMapInternals>())
{
  this->SetNumberOfRequiredOutputs(1); // minimal; raised by SetModelConfiguration
  this->SetNthOutput(0, OutputImageType::New());
  // Default interpolator, following the ITK convention that a filter which resamples
  // provides one (cf. ResampleImageFilter). The wrapped instantiations use a cubic
  // B-spline, which is what every call site used to set by hand. This also keeps the
  // filter usable from Python: SetInterpolator() takes a type wrapped in another module
  // (ITKImageFunction), and WrapITK does not share that SWIG type across module
  // boundaries, so a Python-created interpolator cannot be passed in.
  //
  // TInterpolator may be an abstract interpolator base (Elastix instantiates the filter
  // on itk::InterpolateImageFunction and injects its own concrete interpolator through
  // SetInterpolator); such a type has no New(), so only default-construct a concrete one.
  if constexpr (!std::is_abstract_v<TInterpolator>)
  {
    m_Interpolator = TInterpolator::New();
  }
}

template <typename TInputImage, typename TInterpolator>
void
ImageToFeaturesMap<TInputImage, TInterpolator>
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TInputImage, typename TInterpolator>
void 
ImageToFeaturesMap<TInputImage, TInterpolator>::AddInput(const TInputImage * input)
{
  if (!m_Interpolator)
    itkExceptionMacro("Interpolator must be set before SetInput()");
  if (m_ModelConfiguration.GetModelPath().empty())
    itkExceptionMacro("ImpactModelConfiguration must be set before SetInput()");

  using ImageToTensorFilterType = itk::ImageToTensorFilter<TInputImage, TInterpolator>;
  auto converter = ImageToTensorFilterType::New();

  converter->AddInput(input);
  converter->SetInterpolator(m_Interpolator);
  if (m_Transform)
  {
    converter->SetTransform(m_Transform);
  }

  // The resampling grid is the image's, so the voxel size is needed on every image axis -- one
  // entry per image dimension, not per model dimension. A model swept over a volume (a 2D model
  // run slice by slice) still needs the swept axis's spacing to place its slices.
  if (m_ModelConfiguration.GetVoxelSize().size() < TInputImage::ImageDimension)
  {
    itkExceptionMacro("ImageToFeaturesMap: the voxel size has "
                      << m_ModelConfiguration.GetVoxelSize().size() << " entries but the input image has "
                      << TInputImage::ImageDimension
                      << " dimension(s); give one voxel size per image axis, whatever the model's dimension.");
  }
  InputImageSpacingType outputSpacing;
  for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i)
  {
    outputSpacing[i] = m_ModelConfiguration.GetVoxelSize()[i];
  }

  converter->SetOutputSpacing(outputSpacing);
  converter->Update();

  m_Internals->inputsTensor.push_back(converter->GetTensor().unsqueeze(0));
  this->PushBackInput(const_cast<TInputImage *>(input));
}


template <typename TInputImage, typename TInterpolator>
void
ImageToFeaturesMap<TInputImage, TInterpolator>::VerifyPreconditions() const
{
  Superclass::VerifyPreconditions();
}

template <typename TInputImage, typename TInterpolator>
void
ImageToFeaturesMap<TInputImage, TInterpolator>::GenerateOutputInformation()
{
  // Intentionally empty: do NOT chain to Superclass (ProcessObject), whose default
  // copies the primary input's geometry -- including its LargestPossibleRegion -- onto
  // the outputs. The feature-map geometry is data-dependent and is set in GenerateData()
  // via graft; letting the default run (e.g. on a downstream UpdateOutputInformation
  // re-propagation) would reset LargestPossibleRegion to the input image size while the
  // buffer holds the smaller feature map, making a consumer iterate out of bounds. See
  // the declaration in itkImageToFeaturesMap.h for the full rationale.
}


template <typename TInputImage, typename TInterpolator>
void
ImageToFeaturesMap<TInputImage, TInterpolator>::GenerateData()
{
  using TensorToImageFilterType = itk::TensorToImageFilter<ImageDimension>;
  const torch::Device device(m_Device);

  // Pure inference: no gradient flows back through the model here. Without the guard every
  // patch's forward builds an autograd graph that the layer tensors keep alive -- .to(kCPU)
  // does not release it -- so device activations accumulate over the patches.
  torch::NoGradGuard noGrad;


  // The input's intensity statistics and direction matrix, for models whose forward takes
  // them: TotalSegmentator's exports reorient the patch from the direction and normalise from
  // the statistics. Without this they take their no-metadata path and their lateralised classes
  // come out mirrored on any image whose direction is not the identity.
  SetupImageMetadata<TInputImage>(m_ModelConfiguration, this->GetInput(0));

  // Run the model on the configured device; input patches are moved to it and
  // outputs are pulled back to the CPU below.
  ModelTo(m_ModelConfiguration, device);

  // Everything below is the tiling, which itkImpactPatchTiling.h owns and the registration
  // filters share. This filter only decides where the map lands: with PCA it has to exist as a
  // tensor to be projected, so the accumulator allocates; without it, the map is blended straight
  // into the output image's pixels and never exists twice.
  const bool blendInPlace = (m_PCA == 0);

  std::vector<typename OutputImageType::Pointer> images;
  auto makeDestination = [&](std::size_t layerIndex, const std::vector<int64_t> & shape) -> torch::Tensor {
    if (!blendInPlace)
    {
      return {};
    }
    auto * output = static_cast<OutputImageType *>(this->ProcessObject::GetOutput(layerIndex));
    return AllocateFeatureImage<ImageDimension>(this->GetInput(0), shape, output);
  };

  const std::vector<torch::Tensor> maps =
    RunTiledModel(m_ModelConfiguration,
                  m_Internals->inputsTensor[0],
                  device,
                  torch::Device(torch::kCPU), // the map ends up in an ITK buffer, which is on the host
                  PathCombineModeFromString(m_ModelConfiguration.GetPatchCombine()),
                  makeDestination);

  if (blendInPlace)
  {
    // The output images already hold the maps -- `maps` are views of their pixels.
    return;
  }

  if (m_ModelConfiguration.GetDimension() != ImageDimension)
  {
    itkGenericExceptionMacro("ImageToFeaturesMap: PCA needs the whole map as one tensor, which a model of lower "
                             "dimension than the image never forms -- it is run slice by slice.");
  }
  if (m_Internals->principalComponents.size() < maps.size())
  {
    m_Internals->principalComponents.resize(maps.size());
  }
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    // A fresh converter per layer: a shared one would let a later layer's Update() overwrite the
    // buffer already grafted for an earlier layer.
    auto tensorToImageFilter = TensorToImageFilterType::New();
    tensorToImageFilter->SetReferenceImage(this->GetInput(0));
    // Fit the PCA basis once (e.g. on the fixed image); reuse it when one was injected (e.g. on
    // the moving image) so both share the same basis.
    if (!m_Internals->principalComponents[i].defined())
    {
      m_Internals->principalComponents[i] = Impact::PcaFit(maps[i], m_PCA);
    }
    tensorToImageFilter->SetTensor(Impact::PcaTransform(maps[i], m_Internals->principalComponents[i]));
    tensorToImageFilter->Update();
    this->ProcessObject::GetOutput(i)->Graft(tensorToImageFilter->GetOutput());
  }
}

} // end namespace itk

#endif // itkImageToFeaturesMap_hxx

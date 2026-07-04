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

#include "itkModelConfigurationDetail.h"
#include "itkImageToFeaturesMapInternals.h"

namespace itk
{

template <typename TInputImage, typename TInterpolator>
ImageToFeaturesMap<TInputImage, TInterpolator>::ImageToFeaturesMap()
  : m_Internals(std::make_shared<detail::ImageToFeaturesMapInternals>())
{
  this->SetNumberOfRequiredOutputs(1); // minimal; raised by SetModelConfiguration
  this->SetNthOutput(0, OutputImageType::New());
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
    itkExceptionMacro("ModelConfiguration must be set before SetInput()");

  using ImageToTensorFilterType = itk::ImageToTensorFilter<TInputImage, TInterpolator>;
  auto converter = ImageToTensorFilterType::New();

  converter->AddInput(input);
  converter->SetInterpolator(m_Interpolator);
  if (m_Transform)
  {
    converter->SetTransform(m_Transform);
  }

  InputImageSpacingType outputSpacing;
  for (unsigned int i = 0; i < TInputImage::ImageDimension; ++i) {
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

/**
 * ******************* pca_fit ***********************
 */
inline torch::Tensor
pca_fit(torch::Tensor input, int new_C)
{

  int     C = input.size(0);
  int64_t N = std::accumulate(input.sizes().begin() + 1, input.sizes().end(), 1LL, std::multiplies<int64_t>());
  
  // Flatten spatial dimensions to compute PCA across feature channels
  torch::Tensor reshaped = input.view({ C, N });
  
  torch::Tensor centered = reshaped - reshaped.mean(1, true);
  // Channel-wise covariance matrix of the centered data.
  torch::Tensor covariance = torch::matmul(centered, centered.t()) / (N - 1);

  torch::Tensor eigenvalues, eigenvectors;
  std::tie(eigenvalues, eigenvectors) = torch::linalg_eigh(covariance);
  // Select top-k eigenvectors as principal components
  return eigenvectors.narrow(1, C - new_C, new_C);
} // end pca_fit

/**
 * ******************* pca_transform ***********************
 */
inline torch::Tensor
pca_transform(torch::Tensor input, torch::Tensor principal_components)
{
  int           C = input.size(0);
  int64_t       N = std::accumulate(input.sizes().begin() + 1, input.sizes().end(), 1LL, std::multiplies<int64_t>());
  torch::Tensor reshaped = input.view({ C, N });
  torch::Tensor projected = torch::matmul(principal_components.t(), reshaped - reshaped.mean(1, true));

  std::vector<int64_t> final_shape = { principal_components.size(1) };
  final_shape.insert(final_shape.end(), input.sizes().begin() + 1, input.sizes().end());
  return projected.view(final_shape);
} // end pca_transform

template <typename TInputImage, typename TInterpolator>
void
ImageToFeaturesMap<TInputImage, TInterpolator>::GenerateData()
{
  using TensorToImageFilterType = itk::TensorToImageFilter<ImageDimension>;
  const torch::Device device(m_Device);

  // Run the model on the configured device; input patches are moved to it and
  // outputs are pulled back to the CPU below.
  ModelTo(m_ModelConfiguration, device);

  torch::Tensor tensorTmp;
  std::vector<int64_t> padding;
  for (int it = m_ModelConfiguration.GetDimension()*2 - 1; it >= 0; it--) padding.push_back(m_ModelConfiguration.GetOverlap());
  torch::Tensor inputTensor = torch::constant_pad_nd(m_Internals->inputsTensor[0], padding, 0);
  std::vector<int64_t> patchSize = m_ModelConfiguration.GetPatchSize();
  std::vector<int64_t> inputShape;
  for (unsigned int dim = 0; dim < m_ModelConfiguration.GetDimension(); ++dim){
    if (m_ModelConfiguration.GetPatchSize()[dim] <= 0) 
      patchSize[dim] = inputTensor.size(inputTensor.dim() - m_ModelConfiguration.GetDimension() + dim);
    inputShape.push_back(inputTensor.size(dim+1));
  }
      
  Patch patch = Patch(inputShape, patchSize, m_ModelConfiguration.GetOverlap());
  
  std::vector<Accumulator> accumulators;

  std::vector<int64_t> channelRepeat(m_ModelConfiguration.GetDimension() + 1, 1);
  channelRepeat[0] = m_ModelConfiguration.GetNumberOfChannels();
  
  for (unsigned int sliceIndex = 0; sliceIndex < patch.size(); ++sliceIndex)
  {
    torch::Tensor inputPatch = patch.GetData(inputTensor, sliceIndex)
                                  .repeat({ torch::IntArrayRef(channelRepeat) })
                                  .unsqueeze(0)
                                  .to(device)
                                  .to(GetModelDtype(m_ModelConfiguration));
    std::vector<torch::jit::IValue> outputsList;
    if (accumulators.empty()){
      try
      {
        outputsList = GetModel(m_ModelConfiguration).forward({inputPatch}).toList().vec();
      }
      catch (const std::exception & e)
      {
        itkGenericExceptionMacro(
          "ERROR: The model " << m_ModelConfiguration.GetModelPath()
                        << " configuration is invalid. The dimensions, number of channels, or patch size may "
                            "not meet the requirements of the model.\n"
                            "Details:\n"
                            " - Number of channels: "
                        << m_ModelConfiguration.GetNumberOfChannels()
                        << "\n"
                            " - Patch size: "
                        << m_ModelConfiguration.GetPatchSize()
                        << "\n"
                            " - Dimension: "
                        << m_ModelConfiguration.GetDimension()
                        << "\n"
                            "Please verify the configuration to ensure compatibility with the model. \n Exception : "
                        << e.what());
      }
      if (m_ModelConfiguration.GetLayersMask().size() != outputsList.size())
      {
        itkGenericExceptionMacro("Error: The number of " << m_ModelConfiguration.GetModelPath() << " masks (" << m_ModelConfiguration.GetLayersMask().size()
                                                          << ") does not match the number of layers ("
                                                          << outputsList.size()
                                                          << "). Please ensure that the configuration is consistent.");
      }
      for (int index=0, it = 0; it < outputsList.size(); ++it)
      {
        if (m_ModelConfiguration.GetLayersMask()[it])
        {
          torch::Tensor layerPatch = outputsList[it].toTensor().squeeze(0).to(torch::kCPU).to(torch::kFloat32);
          float ratio = static_cast<float>(layerPatch.size(1))/static_cast<float>(patchSize[0]);
          std::vector<int64_t> outputShape;
          std::vector<int64_t> outputPatchSize;
          for(int d = 0; d < patchSize.size(); d++){
            outputShape.push_back(inputTensor.size(d+1)*ratio);
            outputPatchSize.push_back(layerPatch.size(d+1));
          }
          accumulators.push_back(Accumulator(outputShape, outputPatchSize, m_ModelConfiguration.GetOverlap()*ratio, true));
          accumulators[index].addLayer(sliceIndex, layerPatch);
          index++;
        }
      }
    } else {
      outputsList = GetModel(m_ModelConfiguration).forward({inputPatch}).toList().vec();
      for (int i = 0, index=0, it = 0; it < outputsList.size(); ++it)
      {
        if (m_ModelConfiguration.GetLayersMask()[it])
        {
          torch::Tensor layerPatch = outputsList[it].toTensor().squeeze(0).to(torch::kCPU).to(torch::kFloat32);
          accumulators[index++].addLayer(sliceIndex, layerPatch);
        }
      }
    }

      
    for(int i = 0; i < accumulators.size(); ++i){
      if(accumulators[i].isFull()){
        // A fresh converter per layer: the output is grafted into GetOutput(i), so a
        // shared instance would let a later layer's Update() overwrite the buffer
        // already grafted for an earlier layer, corrupting multi-layer feature maps.
        auto tensorToImageFilter = TensorToImageFilterType::New();
        tensorToImageFilter->SetReferenceImage(this->GetInput(0));
        torch::Tensor result = accumulators[i].assemble().contiguous();
        if (m_PCA > 0){
          if (m_Internals->principalComponents.size() < accumulators.size())
            m_Internals->principalComponents.resize(accumulators.size());
          // Fit the PCA basis once (e.g. on the fixed image); reuse it when one was
          // injected (e.g. on the moving image) so both share the same basis.
          if (!m_Internals->principalComponents[i].defined())
            m_Internals->principalComponents[i] = pca_fit(result, m_PCA);
          tensorToImageFilter->SetTensor(pca_transform(result, m_Internals->principalComponents[i]));
        } else {
          tensorToImageFilter->SetTensor(result);
        }
        tensorToImageFilter->Update();
        this->ProcessObject::GetOutput(i)->Graft(tensorToImageFilter->GetOutput());
      }
    }
  }
}

} // end namespace itk

#endif // itkImageToFeaturesMap_hxx

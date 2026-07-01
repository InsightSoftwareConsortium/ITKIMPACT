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

#ifndef itkInterpolateVectorImageFunction_hxx
#define itkInterpolateVectorImageFunction_hxx

namespace itk
{
/**
 * ******************* SetInputImage ***********************
 */
template <typename TImage, typename TInterpolator>
void
InterpolateVectorImageFunction<TImage, TInterpolator>::SetInputImage(ImagePointer vectorImage)
{
  // One scalar image + interpolator per channel.
  for (unsigned int i = 0; i < vectorImage->GetVectorLength(); ++i)
  {
    auto selector = itk::VectorIndexSelectionCastImageFilter<TImage, itk::Image<float, TImage::ImageDimension>>::New();
    selector->SetInput(vectorImage);
    selector->SetIndex(i);
    selector->Update();

    auto interpolator = TInterpolator::New();
    interpolator->SetInputImage(selector->GetOutput());
    interpolator->SetSplineOrder(3);
    this->m_Interpolators.push_back(interpolator);
  }
} // end SetInputImage

/**
 * ******************* Evaluate ***********************
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
InterpolateVectorImageFunction<TImage, TInterpolator>::Evaluate(ImagePointType point,
                                                                       std::vector<unsigned int> subsetOfFeatures) const
{
  std::vector<float> result;
  for (const unsigned int feature : subsetOfFeatures)
  {
    result.push_back(this->m_Interpolators[feature]->Evaluate(point));
  }
  return torch::from_blob(result.data(), { static_cast<int64_t>(result.size()) }, torch::kFloat32).clone();
} // end Evaluate

/**
 * ******************* EvaluateDerivative ***********************
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
InterpolateVectorImageFunction<TImage, TInterpolator>::EvaluateDerivative(
  ImagePointType point,
  std::vector<unsigned int>     subsetOfFeatures) const
{


  std::vector<float>  derivative(subsetOfFeatures.size() * TImage::ImageDimension, 0.0f);
  CovariantVectorType dev;
  // Fill the derivative tensor with directional gradients for each selected feature
  for (int i = 0; i < subsetOfFeatures.size(); ++i)
  {
    dev = this->m_Interpolators[subsetOfFeatures[i]]->EvaluateDerivative(point);
    for (unsigned int it = 0; it < TImage::ImageDimension; ++it)
    {
      derivative[i * TImage::ImageDimension + it] = static_cast<float>(dev[it]);
    }
  }
  return torch::from_blob(derivative.data(),
                          { static_cast<int64_t>(subsetOfFeatures.size()), TImage::ImageDimension },
                          torch::kFloat32)
    .clone();
} // end EvaluateDerivative
} // namespace itk
#endif // end #ifndef itkInterpolateVectorImageFunction_hxx

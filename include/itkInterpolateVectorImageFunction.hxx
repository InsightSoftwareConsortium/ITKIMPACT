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
  // Each channel's coefficients as TInterpolator computes them, one channel after the other: an
  // evaluation reads only the channels it compares.
  using ChannelImageType = itk::Image<float, TImage::ImageDimension>;
  const unsigned int channels = vectorImage->GetVectorLength();
  const size_t       voxels = vectorImage->GetBufferedRegion().GetNumberOfPixels();
  std::vector<float> coefficients(voxels * channels);
  for (unsigned int channel = 0; channel < channels; ++channel)
  {
    auto selector = itk::VectorIndexSelectionCastImageFilter<TImage, ChannelImageType>::New();
    selector->SetInput(vectorImage);
    selector->SetIndex(channel);
    auto decomposition = TInterpolator::CoefficientFilter::New();
    decomposition->SetInput(selector->GetOutput());
    decomposition->SetSplineOrder(3);
    decomposition->Update();
    const auto * channelCoefficients = decomposition->GetOutput()->GetBufferPointer();
    std::copy(channelCoefficients, channelCoefficients + voxels, coefficients.begin() + channel * voxels);
  }
  this->m_VectorImage = vectorImage;
  this->m_Coefficients = std::move(coefficients);
  this->m_ActiveChannels.clear();
  this->m_ActiveCoefficients.clear();
} // end SetInputImage

/**
 * ******************* SetActiveChannels ***********************
 */
template <typename TImage, typename TInterpolator>
void
InterpolateVectorImageFunction<TImage, TInterpolator>::SetActiveChannels(
  const std::vector<unsigned int> & subsetOfFeatures)
{
  if (subsetOfFeatures == this->m_ActiveChannels)
  {
    return;
  }
  const size_t voxels = this->m_VectorImage->GetBufferedRegion().GetNumberOfPixels();
  const size_t channels = subsetOfFeatures.size();
  this->m_ActiveCoefficients.resize(voxels * channels);
  for (size_t c = 0; c < channels; ++c)
  {
    const float * channel = this->m_Coefficients.data() + subsetOfFeatures[c] * voxels;
    for (size_t voxel = 0; voxel < voxels; ++voxel)
    {
      this->m_ActiveCoefficients[voxel * channels + c] = channel[voxel];
    }
  }
  this->m_ActiveChannels = subsetOfFeatures;
} // end SetActiveChannels

/**
 * ******************* EvaluateChannels ***********************
 */
template <typename TImage, typename TInterpolator>
template <bool VDerivative>
void
InterpolateVectorImageFunction<TImage, TInterpolator>::EvaluateChannels(
  const ImagePointType &            point,
  const std::vector<unsigned int> & subsetOfFeatures,
  float *                           values,
  float *                           derivatives) const
{
  // BSplineInterpolateImageFunction, order 3, step by step and in the same order of operations:
  // Evaluate(), DetermineRegionOfSupport(), SetInterpolationWeights(), SetDerivativeWeights(),
  // ApplyMirrorBoundaryConditions(), then EvaluateValueAndDerivativeAtContinuousIndexInternal().
  constexpr unsigned int Dimension = ImageDimension;
  constexpr unsigned int Support = 4;
  using CoordRepType = typename TInterpolator::CoordRepType;
  const Point<CoordRepType, Dimension> coordinatePoint(point);
  const auto x = this->m_VectorImage->template TransformPhysicalPointToContinuousIndex<CoordRepType>(coordinatePoint);

  const auto & region = this->m_VectorImage->GetBufferedRegion();
  size_t       offsets[Dimension][Support];
  double       weights[Dimension][Support];
  double       derivativeWeights[Dimension][Support];
  size_t       stride = 1;
  for (unsigned int n = 0; n < Dimension; ++n)
  {
    long index[Support];
    index[0] = static_cast<long>(std::floor(static_cast<float>(x[n]))) - 1;
    for (unsigned int k = 1; k < Support; ++k)
    {
      index[k] = index[k - 1] + 1;
    }

    double w = x[n] - static_cast<double>(index[1]);
    weights[n][3] = (1.0 / 6.0) * w * w * w;
    weights[n][0] = (1.0 / 6.0) + 0.5 * w * (w - 1.0) - weights[n][3];
    weights[n][2] = w + weights[n][0] - 2.0 * weights[n][3];
    weights[n][1] = 1.0 - weights[n][0] - weights[n][2] - weights[n][3];
    if constexpr (VDerivative)
    {
      w = x[n] + .5 - static_cast<double>(index[2]);
      const double w2 = 0.75 - w * w;
      const double w3 = 0.5 * (w - w2 + 1.0);
      const double w1 = 1.0 - w2 - w3;
      derivativeWeights[n][0] = 0.0 - w1;
      derivativeWeights[n][1] = w1 - w2;
      derivativeWeights[n][2] = w2 - w3;
      derivativeWeights[n][3] = w3;
    }

    const long start = region.GetIndex(n);
    const long end = start + static_cast<long>(region.GetSize(n)) - 1;
    for (unsigned int k = 0; k < Support; ++k)
    {
      if (region.GetSize(n) == 1)
      {
        index[k] = start;
      }
      else
      {
        if (index[k] < start)
        {
          index[k] = start + (start - index[k]);
        }
        if (index[k] >= end)
        {
          index[k] = end - (index[k] - end);
        }
      }
      offsets[n][k] = static_cast<size_t>(index[k] - start) * stride;
    }
    stride *= region.GetSize(n);
  }

  // Every point of the support, axis 0 fastest, accumulated in double; the channels a block at a
  // time, so that the accumulators stay on the stack.
  constexpr size_t Block = 64;
  const size_t     voxels = stride; // the coefficients of one channel
  const bool       active = subsetOfFeatures == this->m_ActiveChannels;
  const auto &     spacing = this->m_VectorImage->GetSpacing();
  const auto &     direction = this->m_VectorImage->GetDirection();
  for (size_t first = 0; first < subsetOfFeatures.size(); first += Block)
  {
    const size_t channels = std::min(Block, subsetOfFeatures.size() - first);
    const auto * channel = subsetOfFeatures.data() + first;
    double       value[Block] = {};
    double       derivative[Dimension][Block] = {};
    unsigned int k[Dimension] = {};
    for (unsigned int p = 0; p < (1u << (2 * Dimension)); ++p)
    {
      double w = weights[0][k[0]];
      size_t offset = offsets[0][k[0]];
      for (unsigned int n = 1; n < Dimension; ++n)
      {
        w *= weights[n][k[n]];
        offset += offsets[n][k[n]];
      }
      double derivativeW[Dimension];
      if constexpr (VDerivative)
      {
        for (unsigned int n = 0; n < Dimension; ++n)
        {
          derivativeW[n] = 1.0;
          for (unsigned int n1 = 0; n1 < Dimension; ++n1)
          {
            derivativeW[n] *= n1 == n ? derivativeWeights[n1][k[n1]] : weights[n1][k[n1]];
          }
        }
      }
      auto accumulate = [&](size_t c, double coefficient) {
        value[c] += w * coefficient;
        if constexpr (VDerivative)
        {
          for (unsigned int n = 0; n < Dimension; ++n)
          {
            derivative[n][c] += derivativeW[n] * coefficient;
          }
        }
      };
      if (active)
      {
        const float * coefficients = this->m_ActiveCoefficients.data() + offset * subsetOfFeatures.size() + first;
        for (size_t c = 0; c < channels; ++c)
        {
          accumulate(c, coefficients[c]);
        }
      }
      else
      {
        const float * coefficients = this->m_Coefficients.data() + offset;
        for (size_t c = 0; c < channels; ++c)
        {
          accumulate(c, coefficients[channel[c] * voxels]);
        }
      }
      for (unsigned int n = 0; n < Dimension && ++k[n] == Support; ++n)
      {
        k[n] = 0;
      }
    }

    for (size_t c = 0; c < channels; ++c)
    {
      values[first + c] = static_cast<float>(value[c]);
      if constexpr (VDerivative)
      {
        double local[Dimension];
        for (unsigned int n = 0; n < Dimension; ++n)
        {
          local[n] = derivative[n][c] / spacing[n];
        }
        for (unsigned int i = 0; i < Dimension; ++i)
        {
          double sum = 0.0;
          for (unsigned int j = 0; j < Dimension; ++j)
          {
            sum += direction[i][j] * local[j];
          }
          derivatives[(first + c) * Dimension + i] = static_cast<float>(sum);
        }
      }
    }
  }
} // end EvaluateChannels

/**
 * ******************* Evaluate ***********************
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
InterpolateVectorImageFunction<TImage, TInterpolator>::Evaluate(
  const ImagePointType &            point,
  const std::vector<unsigned int> & subsetOfFeatures) const
{
  torch::Tensor values = torch::empty({ static_cast<int64_t>(subsetOfFeatures.size()) }, torch::kFloat32);
  this->Evaluate(point, subsetOfFeatures, values.data_ptr<float>());
  return values;
} // end Evaluate

template <typename TImage, typename TInterpolator>
void
InterpolateVectorImageFunction<TImage, TInterpolator>::Evaluate(const ImagePointType &            point,
                                                                const std::vector<unsigned int> & subsetOfFeatures,
                                                                float *                           values) const
{
  this->template EvaluateChannels<false>(point, subsetOfFeatures, values, nullptr);
} // end Evaluate

/**
 * ******************* EvaluateValueAndDerivative ***********************
 */
template <typename TImage, typename TInterpolator>
void
InterpolateVectorImageFunction<TImage, TInterpolator>::EvaluateValueAndDerivative(
  const ImagePointType &            point,
  const std::vector<unsigned int> & subsetOfFeatures,
  float *                           values,
  float *                           derivatives) const
{
  this->template EvaluateChannels<true>(point, subsetOfFeatures, values, derivatives);
} // end EvaluateValueAndDerivative

/**
 * ******************* EvaluateDerivative ***********************
 */
template <typename TImage, typename TInterpolator>
torch::Tensor
InterpolateVectorImageFunction<TImage, TInterpolator>::EvaluateDerivative(
  const ImagePointType &            point,
  const std::vector<unsigned int> & subsetOfFeatures) const
{
  const auto         channels = static_cast<int64_t>(subsetOfFeatures.size());
  std::vector<float> values(channels);
  torch::Tensor      derivatives = torch::empty({ channels, TImage::ImageDimension }, torch::kFloat32);
  this->EvaluateValueAndDerivative(point, subsetOfFeatures, values.data(), derivatives.data_ptr<float>());
  return derivatives;
} // end EvaluateDerivative
} // namespace itk
#endif // end #ifndef itkInterpolateVectorImageFunction_hxx

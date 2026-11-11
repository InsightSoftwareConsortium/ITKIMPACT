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

// Self-contained: build a small synthetic VectorImage and check that interpolating its channels
// together gives, to the bit, what ITK's BSplineInterpolateImageFunction gives channel by channel:
// values and derivatives, on a region that does not start at zero, with anisotropic spacing and a
// rotated direction, at points up to the borders, for more channels than one block of 64, and for
// the active channels as for any other subset.
#include "itkInterpolateVectorImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkVectorImage.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int
itkInterpolateVectorImageFunctionTest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  constexpr unsigned int Channels = 70;
  using PixelType = float;
  using ChannelImageType = itk::Image<PixelType, Dimension>;
  using FeaturesImageType = itk::VectorImage<PixelType, Dimension>;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<ChannelImageType, float, float>;
  using VectorInterpolatorType = itk::InterpolateVectorImageFunction<FeaturesImageType, InterpolatorType>;

  auto                          image = FeaturesImageType::New();
  FeaturesImageType::RegionType region;
  region.SetIndex({ { 2, -3, 1 } });
  region.SetSize({ { 9, 10, 11 } });
  image->SetRegions(region);
  image->SetNumberOfComponentsPerPixel(Channels);
  FeaturesImageType::SpacingType spacing;
  spacing[0] = 0.7;
  spacing[1] = 1.3;
  spacing[2] = 2.1;
  image->SetSpacing(spacing);
  FeaturesImageType::PointType origin;
  origin[0] = -4.0;
  origin[1] = 2.5;
  origin[2] = 10.0;
  image->SetOrigin(origin);
  FeaturesImageType::DirectionType direction;
  const double                     angle = 0.4;
  direction.SetIdentity();
  direction[0][0] = std::cos(angle);
  direction[0][1] = -std::sin(angle);
  direction[1][0] = std::sin(angle);
  direction[1][1] = std::cos(angle);
  image->SetDirection(direction);
  image->Allocate();
  std::mt19937                          generator(3);
  std::uniform_real_distribution<float> uniform(-1.f, 1.f);
  for (size_t i = 0; i < image->GetPixelContainer()->Size(); ++i)
  {
    image->GetBufferPointer()[i] = uniform(generator);
  }

  auto interpolator = VectorInterpolatorType::New();
  interpolator->SetInputImage(image);
  std::vector<InterpolatorType::Pointer> references;
  for (unsigned int channel = 0; channel < Channels; ++channel)
  {
    auto selector = itk::VectorIndexSelectionCastImageFilter<FeaturesImageType, ChannelImageType>::New();
    selector->SetInput(image);
    selector->SetIndex(channel);
    selector->Update();
    references.push_back(InterpolatorType::New());
    references.back()->SetInputImage(selector->GetOutput());
  }

  std::vector<unsigned int> subset(Channels);
  std::iota(subset.begin(), subset.end(), 0u);
  std::shuffle(subset.begin(), subset.end(), generator);
  interpolator->SetActiveChannels(subset);
  const std::vector<unsigned int> other(subset.rbegin(), subset.rbegin() + 20); // not the active channels

  std::vector<float> values(Channels), derivatives(Channels * Dimension), alone(Channels);
  std::vector<float> otherValues(other.size()), otherDerivatives(other.size() * Dimension);
  unsigned int       mismatches = 0;
  for (int pointNumber = 0; pointNumber < 500; ++pointNumber)
  {
    itk::ContinuousIndex<double, Dimension> index;
    for (unsigned int d = 0; d < Dimension; ++d)
    {
      const double first = region.GetIndex(d) - 0.5;
      index[d] = first + (region.GetSize(d)) * (0.5 + 0.5 * uniform(generator));
    }
    FeaturesImageType::PointType point;
    image->TransformContinuousIndexToPhysicalPoint(index, point);

    interpolator->EvaluateValueAndDerivative(point, subset, values.data(), derivatives.data());
    interpolator->Evaluate(point, subset, alone.data());
    const torch::Tensor tensor = interpolator->Evaluate(point, subset);
    const torch::Tensor tensorDerivative = interpolator->EvaluateDerivative(point, subset);
    interpolator->EvaluateValueAndDerivative(point, other, otherValues.data(), otherDerivatives.data());
    for (size_t c = 0; c < other.size(); ++c)
    {
      const auto & reference = references[other[c]];
      const auto   derivative = reference->EvaluateDerivative(point);
      bool         same = otherValues[c] == static_cast<float>(reference->Evaluate(point));
      for (unsigned int d = 0; d < Dimension; ++d)
      {
        same = same && otherDerivatives[c * Dimension + d] == static_cast<float>(derivative[d]);
      }
      mismatches += same ? 0 : 1;
    }
    for (unsigned int c = 0; c < Channels; ++c)
    {
      const auto & reference = references[subset[c]];
      const auto   value = static_cast<float>(reference->Evaluate(point));
      const auto   derivative = reference->EvaluateDerivative(point);
      bool         same = values[c] == value && alone[c] == value && tensor[c].item<float>() == value;
      for (unsigned int d = 0; d < Dimension; ++d)
      {
        same = same && derivatives[c * Dimension + d] == static_cast<float>(derivative[d]) &&
               tensorDerivative[c][d].item<float>() == static_cast<float>(derivative[d]);
      }
      if (!same && ++mismatches <= 5)
      {
        std::cerr << "channel " << subset[c] << " at " << index << ": " << values[c] << " vs " << value << ", "
                  << derivatives[c * Dimension] << " vs " << derivative[0] << std::endl;
      }
    }
  }
  if (mismatches > 0)
  {
    std::cerr << mismatches << " channel evaluations differ from BSplineInterpolateImageFunction" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

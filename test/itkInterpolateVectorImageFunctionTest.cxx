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

// Self-contained: build a small synthetic VectorImage and interpolate a subset of
// its channels at a physical point. No external data required.

#include "itkInterpolateVectorImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkVectorImage.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <torch/torch.h>
#include <iostream>
#include <vector>

int
itkInterpolateVectorImageFunctionTest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using FeaturesImageType = itk::VectorImage<PixelType, Dimension>;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<itk::Image<PixelType, Dimension>, float, float>;
  using VectorInterpolatorType = itk::InterpolateVectorImageFunction<FeaturesImageType, InterpolatorType>;

  auto                          image = FeaturesImageType::New();
  FeaturesImageType::SizeType   size;
  size.Fill(8);
  FeaturesImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->SetNumberOfComponentsPerPixel(3);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<FeaturesImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto                       i = it.GetIndex();
    itk::VariableLengthVector<float> px(3);
    px[0] = static_cast<float>(i[0]);
    px[1] = static_cast<float>(i[1]);
    px[2] = static_cast<float>(i[2]);
    it.Set(px);
  }

  auto interpolator = VectorInterpolatorType::New();
  interpolator->SetInputImage(image);

  FeaturesImageType::PointType point;
  point[0] = 3.5;
  point[1] = 3.5;
  point[2] = 3.5;
  const torch::Tensor values = interpolator->Evaluate(point, std::vector<unsigned int>{ 0, 1, 2 });

  std::cout << "interpolated values: " << values << std::endl;
  if (values.numel() != 3)
  {
    std::cerr << "InterpolateVectorImageFunction returned an unexpected number of values" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

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

// Self-contained: a small synthetic ramp image is resampled into a torch tensor.
// No external data is required.

#include "itkImageToTensorFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <torch/torch.h>
#include <iostream>

int
itkImageToTensorFilterTest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double>;
  using ImageToTensorFilterType = itk::ImageToTensorFilter<ImageType, InterpolatorType>;

  auto                  image = ImageType::New();
  ImageType::SizeType   size;
  size.Fill(8);
  ImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto i = it.GetIndex();
    image->SetPixel(i, static_cast<float>(i[0] + 10 * i[1] + 100 * i[2]));
  }

  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  auto filter = ImageToTensorFilterType::New();
  filter->SetInterpolator(interpolator);
  ImageType::SpacingType outputSpacing;
  outputSpacing.Fill(1.0);
  filter->SetOutputSpacing(outputSpacing);
  filter->AddInput(image);
  filter->Update();

  const torch::Tensor t = filter->GetTensor();
  std::cout << "tensor sizes: " << t.sizes() << std::endl;
  if (t.numel() == 0)
  {
    std::cerr << "ImageToTensorFilter produced an empty tensor" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

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

// Self-contained: run the committed tiny TorchScript toy model on a small synthetic
// image and check the produced feature layers. The model path (the committed
// Data/ImpactToyModel.pt) is passed as the single argument.

#include "itkImageToFeaturesMap.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkModelConfiguration.h"

#include <iostream>

int
itkImageToFeaturesMapTest(int argc, char * argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: itkImageToFeaturesMapTest <ImpactToyModel.pt>" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double>;
  using ImageToFeaturesMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;

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

  // Toy model: single-channel input, returns 2 layers (4-channel conv, 2-channel passthrough).
  itk::ModelConfiguration modelConfiguration(argv[1], 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, { true, true }, false);

  auto filter = ImageToFeaturesMapType::New();
  filter->SetModelConfiguration(modelConfiguration);
  filter->SetInterpolator(interpolator);
  filter->AddInput(image);
  filter->SetPCA(0);
  filter->SetDevice("cpu");
  filter->Update();

  const auto components = filter->GetOutput(0)->GetNumberOfComponentsPerPixel();
  std::cout << "feature layer 0 components: " << components << std::endl;
  if (components != 4)
  {
    std::cerr << "ImageToFeaturesMap produced an unexpected number of feature channels" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

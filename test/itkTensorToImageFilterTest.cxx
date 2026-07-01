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

// Self-contained: wrap a small synthetic (channels, spatial...) tensor as an
// itk::VectorImage and check the resulting geometry. No external data required.

#include "itkTensorToImageFilter.h"
#include "itkImage.h"

#include <torch/torch.h>
#include <iostream>

int
itkTensorToImageFilterTest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;
  using TensorToImageFilterType = itk::TensorToImageFilter<Dimension>;

  // Reference image only supplies geometry (origin/spacing/direction).
  auto                  reference = ImageType::New();
  ImageType::SizeType   size;
  size.Fill(8);
  ImageType::RegionType region;
  region.SetSize(size);
  reference->SetRegions(region);
  reference->Allocate();
  reference->FillBuffer(0.0f);

  // Layout is (channels, spatial...); here 2 channels over an 8^3 grid.
  const torch::Tensor tensor = torch::rand({ 2, 8, 8, 8 });

  auto filter = TensorToImageFilterType::New();
  filter->SetTensor(tensor);
  filter->SetOrigin(reference->GetOrigin());
  filter->SetSpacing(reference->GetSpacing());
  filter->SetDirection(reference->GetDirection());
  filter->SetSize(reference->GetLargestPossibleRegion().GetSize());
  filter->SetReferenceImage(reference);
  filter->Update();

  auto       output = filter->GetOutput();
  const auto outSize = output->GetLargestPossibleRegion().GetSize();
  std::cout << "output size " << outSize << " components " << output->GetNumberOfComponentsPerPixel() << std::endl;

  if (output->GetNumberOfComponentsPerPixel() != 2 || outSize[0] != 8 || outSize[1] != 8 || outSize[2] != 8)
  {
    std::cerr << "TensorToImageFilter produced unexpected geometry" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

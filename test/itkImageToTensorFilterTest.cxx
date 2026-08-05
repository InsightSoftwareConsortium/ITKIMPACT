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
//
// The second half is a regression test for the image direction. The resampled grid follows
// the input's own index-to-physical map, so with the output spacing equal to the input's the
// tensor must depend only on the voxel content -- not on where or how the image sits in
// physical space. Forming the sampling points axis-aligned (origin + index * spacing)
// instead breaks that as soon as the direction is not the identity: the points walk off
// along the wrong physical axis, IsInsideBuffer() rejects them, and the tensor comes back
// (almost) all zeros while still having the right shape.

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

  // Anchor the values absolutely, not just against each other: the direction test below
  // compares the rotated tensor to this one, and a defect that empties BOTH would satisfy a
  // purely relative comparison -- which is exactly the production failure mode (a buffer left
  // at its zero fill because every sample was rejected). The tensor axes reverse the ITK index
  // order, so t[k][j][i] must carry the ramp sampled at ITK index (i, j, k).
  std::vector<float> expectedBuffer;
  expectedBuffer.reserve(8 * 8 * 8);
  for (int k = 0; k < 8; ++k)
  {
    for (int j = 0; j < 8; ++j)
    {
      for (int i = 0; i < 8; ++i)
      {
        expectedBuffer.push_back(static_cast<float>(i + 10 * j + 100 * k));
      }
    }
  }
  const torch::Tensor expected =
    torch::from_blob(expectedBuffer.data(), { 8, 8, 8 }, torch::kFloat32).clone();
  if (!torch::allclose(t, expected, 1e-4, 1e-4))
  {
    std::cerr << "The tensor does not reproduce the input ramp: max |difference| "
              << (t - expected).abs().max().item<float>() << ", non-zero voxels "
              << t.count_nonzero().item<int64_t>() << " of " << t.numel() << std::endl;
    return EXIT_FAILURE;
  }

  // Same voxels, but rotated a quarter turn about z and moved elsewhere in physical space.
  auto rotated = ImageType::New();
  rotated->SetRegions(region);
  rotated->Allocate();
  rotated->CopyInformation(image);
  ImageType::DirectionType direction;
  direction.SetIdentity();
  direction(0, 0) = 0.0;
  direction(0, 1) = -1.0;
  direction(1, 0) = 1.0;
  direction(1, 1) = 0.0;
  rotated->SetDirection(direction);
  ImageType::PointType rotatedOrigin;
  rotatedOrigin[0] = 5.0;
  rotatedOrigin[1] = -3.0;
  rotatedOrigin[2] = 2.0;
  rotated->SetOrigin(rotatedOrigin);
  itk::ImageRegionIteratorWithIndex<ImageType> rotatedIt(rotated, region);
  for (rotatedIt.GoToBegin(); !rotatedIt.IsAtEnd(); ++rotatedIt)
  {
    rotatedIt.Set(image->GetPixel(rotatedIt.GetIndex()));
  }

  auto rotatedInterpolator = InterpolatorType::New();
  rotatedInterpolator->SetSplineOrder(3);
  auto rotatedFilter = ImageToTensorFilterType::New();
  rotatedFilter->SetInterpolator(rotatedInterpolator);
  rotatedFilter->SetOutputSpacing(outputSpacing);
  rotatedFilter->AddInput(rotated);
  rotatedFilter->Update();

  const torch::Tensor rotatedTensor = rotatedFilter->GetTensor();
  if (rotatedTensor.sizes() != t.sizes())
  {
    std::cerr << "Rotating the image changed the tensor shape: " << t.sizes() << " vs "
              << rotatedTensor.sizes() << std::endl;
    return EXIT_FAILURE;
  }
  if (!torch::allclose(rotatedTensor, t, 1e-4, 1e-4))
  {
    std::cerr << "The image direction was not applied when forming the sampling points: the "
                 "rotated image gave a different tensor.\n"
              << "  max |difference| : " << (rotatedTensor - t).abs().max().item<float>() << '\n'
              << "  non-zero voxels  : " << rotatedTensor.count_nonzero().item<int64_t>() << " of "
              << rotatedTensor.numel() << " (identity image: " << t.count_nonzero().item<int64_t>() << ')'
              << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "direction is honoured: the rotated image gives the same tensor" << std::endl;

  return EXIT_SUCCESS;
}

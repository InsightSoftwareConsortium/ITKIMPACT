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

// The region-sampled run of ImpactCoarseRegistration against its whole-image run, on the intensity similarity: the
// pooled maps built by boxes and the cost volume built by boxes of the coarse grid give the dense run's field, with
// masks and with a moving image on another grid too; a streaming reader gives the same field as the image in memory,
// is read by regions, and the dense field is produced for the region asked.

#include "itkImpactCoarseRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTestingMacros.h"
#include "itkTranslationTransform.h"

#include <cmath>
#include <string>

namespace
{
using ImageType = itk::Image<float, 3>;
using MaskType = itk::Image<unsigned char, 3>;
using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
using FieldType = CoarseType::DisplacementFieldType;

// A smooth texture, f(x + shift): waves a dozen voxels long and a blob, so that the registration has structure
// everywhere and a capture range of a few voxels.
ImageType::Pointer
Texture(const ImageType::SizeType & size, double sx, double sy, double sz)
{
  auto                  image = ImageType::New();
  ImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto   index = it.GetIndex();
    const double x = index[0] + sx, y = index[1] + sy, z = index[2] + sz;
    const double waves =
      std::sin(x / 2.3) * std::cos(y / 3.1) + 0.5 * std::sin((x + y + 2 * z) / 4.0) + 0.7 * std::cos(z / 2.7 + x / 5.0);
    const double blob = 2.0 * std::exp(-((x - 20) * (x - 20) + (y - 18) * (y - 18) + (z - 14) * (z - 14)) / 60.0);
    it.Set(static_cast<float>(waves + blob));
  }
  return image;
}

// A mask on `image`'s grid: the voxels whose `axis` index is below `below`.
MaskType::Pointer
HalfMask(const ImageType * image, unsigned int axis, long below)
{
  auto mask = MaskType::New();
  mask->SetRegions(image->GetLargestPossibleRegion());
  mask->SetSpacing(image->GetSpacing());
  mask->SetOrigin(image->GetOrigin());
  mask->SetDirection(image->GetDirection());
  mask->Allocate();
  itk::ImageRegionIteratorWithIndex<MaskType> it(mask, mask->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    it.Set(it.GetIndex()[axis] < below ? 1 : 0);
  }
  return mask;
}

void
Configure(CoarseType * coarse)
{
  coarse->SetGridSpacing(2);
  coarse->SetDisplacementHalfWidth(2); // 125 candidates
  coarse->SetDevice("cpu");
  coarse->SetSeed(3);
}

// Boxes of 8 cells of the coarse grid: the cost over 125 candidates on a box of 8 + 2 * 2 cells a side, and as much
// again in transients.
constexpr itk::SizeValueType CostBoxBytes = 125 * 8 * 12 * 12 * 12;

double
MaxDifference(const FieldType * a, const FieldType * b, const FieldType::RegionType & region)
{
  double                                            largest = 0.0;
  itk::ImageRegionConstIteratorWithIndex<FieldType> it(a, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto va = it.Get();
    const auto vb = b->GetPixel(it.GetIndex());
    for (unsigned int c = 0; c < 3; ++c)
    {
      largest = std::max(largest, static_cast<double>(std::abs(va[c] - vb[c])));
    }
  }
  return largest;
}

// The dense run and the region-sampled run of one configuration: the largest difference of their coarse grids and of
// their dense fields, printed under `label`; EXIT_FAILURE past `tolerance`.
int
CompareRuns(CoarseType * dense, CoarseType * regions, const char * label, double tolerance)
{
  ITK_TRY_EXPECT_NO_EXCEPTION(dense->Update());
  ITK_TRY_EXPECT_NO_EXCEPTION(regions->Update());
  const FieldType * denseGrid = dense->GetCoarseGridTransform()->GetDisplacementField();
  const FieldType * regionGrid = regions->GetCoarseGridTransform()->GetDisplacementField();
  if (denseGrid->GetLargestPossibleRegion() != regionGrid->GetLargestPossibleRegion())
  {
    std::cerr << label << ": the coarse grids differ in size: " << denseGrid->GetLargestPossibleRegion() << " vs "
              << regionGrid->GetLargestPossibleRegion() << std::endl;
    return EXIT_FAILURE;
  }
  const double gridDifference = MaxDifference(denseGrid, regionGrid, denseGrid->GetLargestPossibleRegion());
  const double fieldDifference =
    MaxDifference(dense->GetOutput(), regions->GetOutput(), dense->GetOutput()->GetLargestPossibleRegion());
  std::cout << label << ": coarse grids max |difference| = " << gridDifference
            << " mm, dense fields max |difference| = " << fieldDifference << " mm" << std::endl;
  if (gridDifference > tolerance || fieldDifference > tolerance)
  {
    std::cerr << label << ": the region-sampled run differs from the dense one." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
} // namespace

int
itkImpactCoarseRegistrationRegionTest(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv)
              << " <temporary directory> [<model with half-resolution layers>]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string         temporary = argv[1];
  const ImageType::SizeType size = { { 48, 40, 32 } }; // whole cells of 2 voxels, whole boxes of 16
  ImageType::Pointer        fixed = Texture(size, 0.0, 0.0, 0.0);
  ImageType::Pointer        moving = Texture(size, 1.5, -1.0, 0.5);

  // 1. The whole-image run against the run by regions: the pooled maps by boxes of 16 voxels, the cost volume by
  // boxes of 8 cells (3 x 3 x 2 of them on the 24 x 20 x 16 grid).
  auto dense = CoarseType::New();
  ITK_EXERCISE_BASIC_OBJECT_METHODS(dense, ImpactCoarseRegistration, ImageSource);
  dense->SetFixedImage(fixed);
  dense->SetMovingImage(moving);
  Configure(dense);
  auto regions = CoarseType::New();
  regions->SetFixedImage(fixed);
  regions->SetMovingImage(moving);
  Configure(regions);
  regions->SetRegionSize(16);
  regions->SetCostBoxBytes(CostBoxBytes);
  ITK_TEST_SET_GET_VALUE(16u, regions->GetRegionSize());
  ITK_TEST_SET_GET_VALUE(CostBoxBytes, regions->GetCostBoxBytes());
  if (CompareRuns(dense, regions, "in memory", 1e-3) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if (regions->GetFixedImageStreamed() || regions->GetMovingImageStreamed())
  {
    std::cerr << "An image in memory was reported as streamed." << std::endl;
    return EXIT_FAILURE;
  }
  const FieldType * regionField = regions->GetOutput();
  {
    // The field moves the moving image onto the fixed one: moving(y) = f(y + s) asks for u = -s.
    FieldType::RegionType centre = fixed->GetLargestPossibleRegion();
    for (unsigned int a = 0; a < 3; ++a)
    {
      centre.SetIndex(a, 8);
      centre.SetSize(a, centre.GetSize()[a] - 16);
    }
    itk::Vector<double, 3> mean;
    mean.Fill(0.0);
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(regionField, centre);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      mean += it.Get();
    }
    mean /= static_cast<double>(centre.GetNumberOfPixels());
    std::cout << "mean displacement " << mean << " mm for a shift of (-1.5, 1, -0.5)" << std::endl;
    const double expected[3] = { -1.5, 1.0, -0.5 };
    for (unsigned int c = 0; c < 3; ++c)
    {
      if (std::abs(mean[c] - expected[c]) > 1.0)
      {
        std::cerr << "The coarse stage did not recover the shift." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  // The coarse grid transform is the same map: at a voxel, its displacement is the field's.
  {
    CoarseType::DisplacementFieldTransformType * grid = regions->GetCoarseGridTransform();
    const ImageType::IndexType                   index = { { 21, 17, 13 } };
    ImageType::PointType                         point;
    fixed->TransformIndexToPhysicalPoint(index, point);
    const auto mapped = grid->TransformPoint(point);
    const auto field = regionField->GetPixel(index);
    for (unsigned int c = 0; c < 3; ++c)
    {
      if (std::abs((mapped[c] - point[c]) - field[c]) > 1e-4)
      {
        std::cerr << "The coarse grid transform displaces by " << (mapped[c] - point[c]) << " where the field holds "
                  << field[c] << " (component " << c << ")." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // 2. With masks: the cells' shares and the cost's support by boxes.
  {
    MaskType::Pointer fixedMask = HalfMask(fixed, 0, 36);
    MaskType::Pointer movingMask = HalfMask(moving, 1, 30);
    auto              maskedDense = CoarseType::New();
    maskedDense->SetFixedImage(fixed);
    maskedDense->SetMovingImage(moving);
    maskedDense->SetFixedMask(fixedMask);
    maskedDense->SetMovingMask(movingMask);
    Configure(maskedDense);
    auto maskedRegions = CoarseType::New();
    maskedRegions->SetFixedImage(fixed);
    maskedRegions->SetMovingImage(moving);
    maskedRegions->SetFixedMask(fixedMask);
    maskedRegions->SetMovingMask(movingMask);
    Configure(maskedRegions);
    maskedRegions->SetRegionSize(16);
    maskedRegions->SetCostBoxBytes(CostBoxBytes);
    if (CompareRuns(maskedDense, maskedRegions, "with masks", 1e-3) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }
  }

  // 3. The moving image on another grid: brought onto the fixed one region by region.
  {
    ImageType::Pointer   shifted = Texture(size, 1.5, -1.0, 0.5);
    ImageType::PointType origin;
    origin[0] = 0.5;
    origin[1] = -0.25;
    origin[2] = 0.0;
    shifted->SetOrigin(origin);
    auto gridDense = CoarseType::New();
    gridDense->SetFixedImage(fixed);
    gridDense->SetMovingImage(shifted);
    Configure(gridDense);
    auto gridRegions = CoarseType::New();
    gridRegions->SetFixedImage(fixed);
    gridRegions->SetMovingImage(shifted);
    Configure(gridRegions);
    gridRegions->SetRegionSize(16);
    gridRegions->SetCostBoxBytes(CostBoxBytes);
    if (CompareRuns(gridDense, gridRegions, "moving on another grid", 1e-3) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }
  }

  // 4. A transform on the moving side, outside the field: the translation undoing the shift leaves little to find.
  {
    ImageType::Pointer     far = Texture(size, 5.5, -1.0, 0.5);
    auto                   shift = itk::TranslationTransform<double, 3>::New();
    itk::Vector<double, 3> offset;
    offset[0] = -5.5;
    offset[1] = 1.0;
    offset[2] = -0.5;
    shift->SetOffset(offset);
    auto undone = CoarseType::New();
    undone->SetFixedImage(fixed);
    undone->SetMovingImage(far);
    undone->SetMovingTransform(shift);
    Configure(undone);
    undone->SetRegionSize(16);
    undone->SetCostBoxBytes(CostBoxBytes);
    ITK_TRY_EXPECT_NO_EXCEPTION(undone->Update());
    FieldType::RegionType centre = fixed->GetLargestPossibleRegion();
    for (unsigned int a = 0; a < 3; ++a)
    {
      centre.SetIndex(a, 8);
      centre.SetSize(a, centre.GetSize()[a] - 16);
    }
    double                                            sum = 0.0;
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(undone->GetOutput(), centre);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      sum += it.Get().GetNorm();
    }
    const double mean = sum / static_cast<double>(centre.GetNumberOfPixels());
    std::cout << "moving read through the undoing translation: mean |displacement| " << mean << " mm" << std::endl;
    if (mean > 1.0)
    {
      std::cerr << "The moving transform was not applied: the field still carries the shift." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 5. The pair streamed from disk, by regions: the same field, read in pieces, produced for the region asked.
  const std::string fixedFile = temporary + "/itkImpactCoarseRegistrationRegionTest_fixed.mha";
  const std::string movingFile = temporary + "/itkImpactCoarseRegistrationRegionTest_moving.mha";
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(fixed, fixedFile));
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(moving, movingFile));
  using ReaderType = itk::ImageFileReader<ImageType>;
  auto fixedReader = ReaderType::New();
  fixedReader->SetFileName(fixedFile);
  fixedReader->UseStreamingOn();
  auto movingReader = ReaderType::New();
  movingReader->SetFileName(movingFile);
  movingReader->UseStreamingOn();
  auto streamed = CoarseType::New();
  streamed->SetFixedImage(fixedReader->GetOutput());
  streamed->SetMovingImage(movingReader->GetOutput());
  Configure(streamed);
  streamed->SetRegionSize(16);
  streamed->SetCostBoxBytes(CostBoxBytes);
  streamed->UpdateOutputInformation();
  FieldType::RegionType slab = streamed->GetOutput()->GetLargestPossibleRegion();
  slab.SetIndex(2, 10);
  slab.SetSize(2, 5);
  streamed->GetOutput()->SetRequestedRegion(slab);
  ITK_TRY_EXPECT_NO_EXCEPTION(streamed->Update());
  if (streamed->GetOutput()->GetBufferedRegion() != slab)
  {
    std::cerr << "The dense field was produced over " << streamed->GetOutput()->GetBufferedRegion()
              << " where the slab " << slab << " was requested." << std::endl;
    return EXIT_FAILURE;
  }
  const double streamedDifference = MaxDifference(streamed->GetOutput(), regionField, slab);
  std::cout << "regions in memory vs streamed: max |difference| = " << streamedDifference << " mm" << std::endl;
  if (streamedDifference > 1e-4)
  {
    std::cerr << "The streamed run differs from the in-memory one by " << streamedDifference << " mm." << std::endl;
    return EXIT_FAILURE;
  }
  if (!streamed->GetFixedImageStreamed() || !streamed->GetMovingImageStreamed())
  {
    std::cerr << "The readers served the whole images where regions were requested." << std::endl;
    return EXIT_FAILURE;
  }
  if (fixedReader->GetOutput()->GetBufferedRegion() == fixed->GetLargestPossibleRegion())
  {
    std::cerr << "The fixed reader holds the whole image after the run." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "bytes read over the streamed run: " << streamed->GetBytesRead() << std::endl;
  if (streamed->GetBytesRead() == 0)
  {
    std::cerr << "No bytes were counted." << std::endl;
    return EXIT_FAILURE;
  }

  // A model whose layers are half the image's resolution, on a pair of odd sizes, whose whole layers the cells pool
  // by uneven windows: by regions as whole, without a mask and with one.
  if (argc > 2)
  {
    const ImageType::SizeType           odd = { { 49, 41, 33 } };
    ImageType::Pointer                  oddFixed = Texture(odd, 0.0, 0.0, 0.0);
    ImageType::Pointer                  oddMoving = Texture(odd, 1.5, -1.0, 0.5);
    const itk::ImpactModelConfiguration model(
      argv[2], 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, true }, false);
    for (const bool masked : { false, true })
    {
      auto run = [&](bool byRegions) {
        auto coarse = CoarseType::New();
        coarse->SetFixedImage(oddFixed);
        coarse->SetMovingImage(oddMoving);
        coarse->AddModelConfiguration(model);
        coarse->SetDistance({ "L2", "L2" });
        coarse->SetLayersWeight({ 1.f, 1.f });
        if (masked)
        {
          coarse->SetFixedMask(HalfMask(oddFixed, 0, 30));
        }
        Configure(coarse);
        if (byRegions)
        {
          coarse->SetRegionSize(16);
          coarse->SetRegionHalo(2);
          coarse->SetCostBoxBytes(CostBoxBytes);
        }
        return coarse;
      };
      if (CompareRuns(run(false), run(true), masked ? "half-resolution layers, masked" : "half-resolution layers", 1e-3) !=
          EXIT_SUCCESS)
      {
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}

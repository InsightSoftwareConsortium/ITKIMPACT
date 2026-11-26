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
// masks and with a moving image on another grid too; with the maps in mapped files, every step of the solve taken by
// those boxes, it gives the field of the maps resident; a streaming reader gives the same field as the image in
// memory, is read by regions, and the dense field is produced for the region asked. Toy models by regions as whole:
// layers at half the image's resolution, PCA, a model of one dimension fewer, a voxel size of the model's own,
// instance normalisations, and layers on lattices of 3, 6 and 16 voxels, coarser than the cells.

#include "itkImpactCoarseRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTestingMacros.h"
#include "itkTranslationTransform.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
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
      const double difference = std::abs(va[c] - vb[c]);
      if (!std::isfinite(difference))
      {
        return std::numeric_limits<double>::infinity(); // which std::max would drop
      }
      largest = std::max(largest, difference);
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

// `mapped`, configured as `resident` (a run by regions, updated), run with its maps in files of `scratch`: the
// largest difference of the two coarse grids, printed under `label`.
int
CompareRoutes(CoarseType * resident, CoarseType * mapped, const std::string & scratch, const char * label)
{
  mapped->SetResidentFeatureBudget(1); // nothing fits
  mapped->SetScratchDirectory(scratch);
  ITK_TRY_EXPECT_NO_EXCEPTION(mapped->Update());
  if (!resident->GetResidentFeatures() || mapped->GetResidentFeatures())
  {
    std::cerr << label << ": the maps of a small pair should stay resident, and a budget of one byte leave none."
              << std::endl;
    return EXIT_FAILURE;
  }
  const FieldType * residentGrid = resident->GetCoarseGridTransform()->GetDisplacementField();
  const double      difference = MaxDifference(
    residentGrid, mapped->GetCoarseGridTransform()->GetDisplacementField(), residentGrid->GetLargestPossibleRegion());
  std::cout << label << ": coarse grids, mapped against resident: max |difference| = " << difference << " mm"
            << std::endl;
  // A brick takes the steps the grid takes whole, on the same values; the inverse consistency alone reads its
  // window in the window's own coordinates, a rounding apart: 8.7e-6 mm at most over the cases of this test, and
  // none without it or where a window is the grid.
  if (!(difference < 5e-5))
  {
    std::cerr << label << ": the run from mapped files differs from the resident one." << std::endl;
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
              << " <temporary directory> [<model with half-resolution layers> <3D model> <2D model> <model with "
                 "instance normalisations> <model with layers of strides 3, 6 and 16>]"
              << std::endl;
    return EXIT_FAILURE;
  }
  const std::string temporary = argv[1];
  const std::string scratch = temporary + "/itkImpactCoarseRegistrationRegionTest_scratch";
  std::filesystem::create_directories(scratch);
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
  // The same from mapped files, whose coarse grid is one of them, kept by the grid past its filter; the files go
  // with the run and with the grid.
  {
    CoarseType::Pointer mapped = CoarseType::New();
    mapped->SetFixedImage(fixed);
    mapped->SetMovingImage(moving);
    Configure(mapped);
    mapped->SetRegionSize(16);
    mapped->SetCostBoxBytes(CostBoxBytes);
    ITK_TEST_SET_GET_VALUE(0u, mapped->GetResidentFeatureBudget());
    if (CompareRoutes(regions, mapped, scratch, "in memory") != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }
    ITK_TEST_SET_GET_VALUE(scratch, std::string(mapped->GetScratchDirectory()));
    CoarseType::DisplacementFieldTransformType::Pointer grid = mapped->GetCoarseGridTransform();
    mapped = nullptr;
    const FieldType * residentGrid = regions->GetCoarseGridTransform()->GetDisplacementField();
    if (!(MaxDifference(residentGrid, grid->GetDisplacementField(), residentGrid->GetLargestPossibleRegion()) < 1e-4))
    {
      std::cerr << "The mapped coarse grid did not outlive its filter." << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (!std::filesystem::is_empty(scratch))
  {
    std::cerr << "The run left files in its scratch directory." << std::endl;
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
    auto maskedMapped = CoarseType::New();
    maskedMapped->SetFixedImage(fixed);
    maskedMapped->SetMovingImage(moving);
    maskedMapped->SetFixedMask(fixedMask);
    maskedMapped->SetMovingMask(movingMask);
    Configure(maskedMapped);
    maskedMapped->SetRegionSize(16);
    maskedMapped->SetCostBoxBytes(CostBoxBytes);
    if (CompareRoutes(maskedRegions, maskedMapped, scratch, "with masks") != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }
  }

  // The least boxes. A grid of 7 x 6 x 5 cells, read by regions of 2: with a cost volume by boxes of one cell
  // (CostBoxBytes 1, the least), then of 3, whose last along an axis is one cell, every step of the mapped solve
  // takes bricks narrower than its kernels, at the ends of the grid too, and gives the resident run's field.
  {
    const ImageType::SizeType few = { { 14, 12, 10 } };
    ImageType::Pointer        fewFixed = Texture(few, 0.0, 0.0, 0.0);
    ImageType::Pointer        fewMoving = Texture(few, 1.5, -1.0, 0.5);
    for (const itk::SizeValueType costBoxBytes : { 1, 400000 }) // 400000: (7.4 - 4) cells a side
    {
      auto run = [&]() {
        auto coarse = CoarseType::New();
        coarse->SetFixedImage(fewFixed);
        coarse->SetMovingImage(fewMoving);
        Configure(coarse);
        coarse->SetRegionSize(4);
        coarse->SetCostBoxBytes(costBoxBytes);
        return coarse;
      };
      const auto resident = run();
      ITK_TRY_EXPECT_NO_EXCEPTION(resident->Update());
      if (CompareRoutes(resident, run(), scratch, costBoxBytes == 1 ? "boxes of one cell" : "a last box of one cell") !=
          EXIT_SUCCESS)
      {
        return EXIT_FAILURE;
      }
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
    ImageType::Pointer     shifted = Texture(size, 5.5, -1.0, 0.5);
    auto                   shift = itk::TranslationTransform<double, 3>::New();
    itk::Vector<double, 3> offset;
    offset[0] = -5.5;
    offset[1] = 1.0;
    offset[2] = -0.5;
    shift->SetOffset(offset);
    auto undone = CoarseType::New();
    undone->SetFixedImage(fixed);
    undone->SetMovingImage(shifted);
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
      const char * label = masked ? "half-resolution layers, masked" : "half-resolution layers";
      const auto   byRegions = run(true);
      if (CompareRuns(run(false), byRegions, label, 1e-3) != EXIT_SUCCESS ||
          CompareRoutes(byRegions, run(true), scratch, label) != EXIT_SUCCESS)
      {
        return EXIT_FAILURE;
      }
    }
  }

  // The models' features by regions as whole on the same pair: PCA (on layers at the image's resolution and at half
  // of it), a 2D model swept along the axis closest to head-feet, a voxel size finer and coarser than the image's, a
  // model with instance normalisations; with both masks, and with a 2D model resampling the image. Each against the
  // whole-image run and, on the maps from mapped files, against its run on the resident ones. BalanceLosses measures
  // each layer's spread over the candidates, which follows the pooled features where the grid's choices may not.
  if (argc > 5)
  {
    const ImageType::SizeType odd = { { 49, 41, 33 } };
    ImageType::Pointer        oddFixed = Texture(odd, 0.0, 0.0, 0.0);
    ImageType::Pointer        oddMoving = Texture(odd, 1.5, -1.0, 0.5);
    struct Variant
    {
      const char *              label;
      const char *              model;
      unsigned int              dimension;
      float                     voxel;
      std::vector<unsigned int> pca;
      bool                      masked;
    };
    for (const Variant & variant :
         { Variant{ "PCA", argv[3], 3, 0.f, { 2, 1 }, false },
           Variant{ "PCA, masked", argv[3], 3, 0.f, { 2, 1 }, true },
           Variant{ "PCA of half-resolution layers", argv[2], 3, 0.f, { 0, 2 }, false },
           Variant{ "a 2D model", argv[4], 2, 0.f, {}, false },
           Variant{ "a 2D model, masked", argv[4], 2, 0.f, {}, true },
           Variant{ "a 2D model, PCA", argv[4], 2, 0.f, { 2, 1 }, false },
           Variant{ "a 2D model, a voxel size of 1.5", argv[4], 2, 1.5f, {}, false },
           Variant{ "a voxel size of 0.75", argv[3], 3, 0.75f, {}, false },
           Variant{ "a voxel size of 1.5", argv[3], 3, 1.5f, {}, false },
           Variant{ "a voxel size of 1.5, masked, PCA", argv[3], 3, 1.5f, { 2, 1 }, true },
           Variant{ "instance normalisations", argv[5], 3, 0.f, {}, false },
           Variant{ "instance normalisations, masked, PCA", argv[5], 3, 0.f, { 2, 1 }, true } })
    {
      auto run = [&](bool byRegions) {
        auto coarse = CoarseType::New();
        coarse->SetFixedImage(oddFixed);
        coarse->SetMovingImage(oddMoving);
        const std::vector<unsigned int> zeros(variant.dimension, 0u);
        coarse->AddModelConfiguration(itk::ImpactModelConfiguration(variant.model,
                                                                    variant.dimension,
                                                                    1,
                                                                    zeros,
                                                                    { variant.voxel, variant.voxel, variant.voxel },
                                                                    zeros,
                                                                    { true, true },
                                                                    false));
        coarse->SetDistance({ "L2", "L2" });
        coarse->SetLayersWeight({ 1.f, 1.f });
        coarse->SetPCA(variant.pca);
        if (variant.masked)
        {
          coarse->SetFixedMask(HalfMask(oddFixed, 0, 30));
          coarse->SetMovingMask(HalfMask(oddMoving, 1, 25));
        }
        coarse->SetBalanceLosses(true);
        Configure(coarse);
        if (byRegions)
        {
          coarse->SetRegionSize(16);
          coarse->SetRegionHalo(3);
          coarse->SetCostBoxBytes(CostBoxBytes);
        }
        return coarse;
      };
      const auto whole = run(false);
      const auto byRegions = run(true);
      if (CompareRuns(whole, byRegions, variant.label, 1e-3) != EXIT_SUCCESS ||
          CompareRoutes(byRegions, run(true), scratch, variant.label) != EXIT_SUCCESS)
      {
        return EXIT_FAILURE;
      }
      const std::vector<double> & spreads = whole->GetLayerSpreads();
      double                      spreadDifference = spreads.size() == 2 ? 0.0 : 1.0;
      for (size_t l = 0; l < spreads.size() && l < byRegions->GetLayerSpreads().size(); ++l)
      {
        spreadDifference =
          std::max(spreadDifference, std::abs(byRegions->GetLayerSpreads()[l] - spreads[l]) / std::abs(spreads[l]));
      }
      std::cout << variant.label << ": the layers' spreads within " << spreadDifference << " of theirs" << std::endl;
      if (!(spreadDifference < 1e-5))
      {
        std::cerr << variant.label << ": the layers' spreads by regions differ from the whole-image run's."
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // A model whose layers lie on lattices of 3, 6 and 16 voxels (strided convolutions, their last samples partial, and
  // an average pooling), by regions as whole: the layers of strides 3 and 6 on cells of 4 voxels, the second coarser
  // than they, each region starting on a multiple of 6 voxels; and the layer of stride 16, a backbone's deepest, on
  // cells of 2 voxels, each of which pools one or two of its samples, beside the 3x3x3 convolution's layer: alone, it
  // gives eight cells a side the same features, whose candidates tie and fall to round-off (0.9 mm apart, its pooled
  // maps 6e-8 apart). Without a mask and with both.
  if (argc > 6)
  {
    struct Case
    {
      const char *        label;
      unsigned int        gridSpacing;
      bool                beside;
      std::vector<bool>   layers;
      ImageType::SizeType size;
      bool                masked;
    };
    const ImageType::SizeType odd = { { 49, 41, 33 } }, large = { { 97, 81, 65 } };
    for (const Case & lattice :
         { Case{ "layers of strides 3 and 6", 4, false, { true, true, false }, odd, false },
           Case{ "layers of strides 3 and 6, masked", 4, false, { true, true, false }, odd, true },
           Case{ "a layer of stride 16", 2, true, { false, false, true }, large, false },
           Case{ "a layer of stride 16, masked", 2, true, { false, false, true }, large, true } })
    {
      const ImageType::Pointer latticeFixed = Texture(lattice.size, 0.0, 0.0, 0.0);
      const ImageType::Pointer latticeMoving = Texture(lattice.size, 1.5, -1.0, 0.5);
      const auto               kept =
        static_cast<size_t>(std::count(lattice.layers.begin(), lattice.layers.end(), true)) + (lattice.beside ? 1 : 0);
      auto run = [&](bool byRegions) {
        auto coarse = CoarseType::New();
        coarse->SetFixedImage(latticeFixed);
        coarse->SetMovingImage(latticeMoving);
        if (lattice.beside)
        {
          coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
            argv[3], 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, false }, false));
        }
        coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
          argv[6], 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, lattice.layers, false));
        coarse->SetDistance(std::vector<std::string>(kept, "L2"));
        coarse->SetLayersWeight(std::vector<float>(kept, 1.f));
        if (lattice.masked)
        {
          coarse->SetFixedMask(HalfMask(latticeFixed, 0, 30));
          coarse->SetMovingMask(HalfMask(latticeMoving, 1, 25));
        }
        Configure(coarse);
        coarse->SetGridSpacing(lattice.gridSpacing);
        if (byRegions)
        {
          coarse->SetRegionSize(16);
          coarse->SetRegionHalo(2);
          coarse->SetCostBoxBytes(CostBoxBytes);
        }
        return coarse;
      };
      if (CompareRuns(run(false), run(true), lattice.label, 1e-3) != EXIT_SUCCESS)
      {
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}

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

// The region-sampled run of ImpactFineRegistration against its whole-image run, on the intensity similarity: it
// reproduces the dense field, with the pooled maps resident and by bricks from mapped files that the run removes; a
// streaming reader gives the same field as the image in memory and is read by regions; a transform set on the moving
// side is applied outside the field; stepping the boxes drawn at each iteration is that run when every box is
// drawn, and tends to it with a few as the iterations grow.

#include "itkImpactFineRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTestingMacros.h"
#include "itkTranslationTransform.h"
#include "itkVectorLinearInterpolateImageFunction.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>

namespace
{
using ImageType = itk::Image<float, 3>;
using FineType = itk::ImpactFineRegistration<ImageType>;
using FieldType = FineType::DisplacementFieldType;

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

// A smooth field of a fraction of a voxel on `image`'s grid, for the runs to start from. From a zero field every
// cell reads the moving features exactly at a sample, where linear interpolation has two slopes and float round-off
// picks one: two runs then differ by whole steps there, whatever their agreement elsewhere.
FieldType::Pointer
StartingField(const ImageType * image)
{
  auto field = FieldType::New();
  field->SetRegions(image->GetLargestPossibleRegion());
  field->Allocate();
  itk::ImageRegionIteratorWithIndex<FieldType> it(field, field->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto           index = it.GetIndex();
    FieldType::PixelType value;
    value[0] = static_cast<float>(0.31 + 0.2 * std::sin(index[1] / 7.0));
    value[1] = static_cast<float>(-0.27 + 0.2 * std::cos(index[2] / 5.0));
    value[2] = static_cast<float>(0.23 + 0.2 * std::sin(index[0] / 9.0));
    it.Set(value);
  }
  return field;
}

void
Configure(FineType * fine, unsigned int iterations, double epsilon = 1.0)
{
  fine->SetGridShrinkFactor(2);
  fine->SetControlGridSmoothingIterations(1);
  fine->SetNumberOfIterations(iterations);
  fine->SetLearningRate(0.5);
  fine->SetRegularizationWeight(1.25);
  // Adam's first steps are lr * g / (|g| + eps): with the default eps a gradient near zero takes a full step whose
  // sign float round-off decides. A unit eps makes the step proportional to the gradient, so that two runs whose
  // gradients agree to float precision take the same path.
  fine->SetEpsilon(epsilon);
  fine->SetDevice("cpu");
  fine->SetSeed(3);
}

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
} // namespace

int
itkImpactFineRegistrationRegionTest(int argc, char * argv[])
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
  const FieldType::Pointer  start = StartingField(fixed);

  // 1. The whole-image run.
  auto dense = FineType::New();
  ITK_EXERCISE_BASIC_OBJECT_METHODS(dense, ImpactFineRegistration, ImageSource);
  dense->SetFixedImage(fixed);
  dense->SetMovingImage(moving);
  dense->SetInitialDisplacementField(start);
  Configure(dense, 6);
  ITK_TRY_EXPECT_NO_EXCEPTION(dense->Update());
  const FieldType * denseField = dense->GetOutput();

  // 2. By regions, the pooled maps resident (one pass over the boxes, then the dense iterations on them): the same
  // field. Then with nothing resident, every cell at every iteration by bricks of one box, from files mapped in a
  // scratch directory and gone with the run: the same field again.
  auto regions = FineType::New();
  regions->SetFixedImage(fixed);
  regions->SetMovingImage(moving);
  regions->SetInitialDisplacementField(start);
  Configure(regions, 6);
  regions->SetRegionSize(16);
  ITK_TEST_SET_GET_VALUE(16u, regions->GetRegionSize());
  ITK_TRY_EXPECT_NO_EXCEPTION(regions->Update());
  if (!regions->GetResidentFeatures())
  {
    std::cerr << "The pooled maps of a small pair should stay resident." << std::endl;
    return EXIT_FAILURE;
  }
  const FieldType * regionField = regions->GetOutput();
  FineType::Pointer bricks = FineType::New();
  bricks->SetFixedImage(fixed);
  bricks->SetMovingImage(moving);
  bricks->SetInitialDisplacementField(start);
  Configure(bricks, 6);
  bricks->SetRegionSize(16);
  bricks->SetResidentFeatureBudget(1); // nothing fits: by bricks
  bricks->SetBrickBytes(1);            // of one box
  ITK_TEST_SET_GET_VALUE(1u, bricks->GetResidentFeatureBudget());
  ITK_TEST_SET_GET_VALUE(1u, bricks->GetBrickBytes());
  const std::string scratch = temporary + "/itkImpactFineRegistrationRegionTest_scratch";
  std::filesystem::create_directories(scratch);
  bricks->SetScratchDirectory(scratch);
  ITK_TEST_SET_GET_VALUE(scratch, std::string(bricks->GetScratchDirectory()));
  ITK_TRY_EXPECT_NO_EXCEPTION(bricks->Update());
  if (bricks->GetResidentFeatures())
  {
    std::cerr << "A budget of one byte should leave nothing resident." << std::endl;
    return EXIT_FAILURE;
  }
  if (bricks->GetBytesRead() != regions->GetBytesRead())
  {
    std::cerr << "By bricks, the images should be read once, as with the maps resident: " << bricks->GetBytesRead()
              << " bytes against " << regions->GetBytesRead() << "." << std::endl;
    return EXIT_FAILURE;
  }
  const double brickDifference = MaxDifference(denseField, bricks->GetOutput(), fixed->GetLargestPossibleRegion());
  std::cout << "dense vs every cell by bricks: max |difference| = " << brickDifference << " mm" << std::endl;
  if (brickDifference > 1e-5)
  {
    std::cerr << "The field of every cell by bricks differs from the dense one by " << brickDifference << " mm."
              << std::endl;
    return EXIT_FAILURE;
  }
  // The control grid of the run by bricks is a file mapped to memory, which the grid keeps: it outlives the filter,
  // and the file goes with it.
  {
    FineType::DisplacementFieldTransformType::Pointer mapped = bricks->GetControlGridTransform();
    bricks = nullptr;
    const FieldType * residentGrid = regions->GetControlGridTransform()->GetDisplacementField();
    const double      gridDifference =
      MaxDifference(mapped->GetDisplacementField(), residentGrid, residentGrid->GetLargestPossibleRegion());
    std::cout << "control grid by bricks, past its filter, vs resident: max |difference| = " << gridDifference << " mm"
              << std::endl;
    if (gridDifference > 1e-5)
    {
      std::cerr << "The control grid of the run by bricks is not the resident run's." << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (!std::filesystem::is_empty(scratch))
  {
    std::cerr << "The run left files in its scratch directory." << std::endl;
    return EXIT_FAILURE;
  }
  const double difference = MaxDifference(denseField, regionField, fixed->GetLargestPossibleRegion());
  std::cout << "dense vs regions: max |difference| = " << difference << " mm" << std::endl;
  if (difference > 1e-5)
  {
    std::cerr << "The region-sampled field differs from the dense one by " << difference << " mm." << std::endl;
    return EXIT_FAILURE;
  }
  const auto & denseValues = dense->GetMetricValuesPerIteration();
  const auto & regionValues = regions->GetMetricValuesPerIteration();
  for (size_t i = 0; i < denseValues.size(); ++i)
  {
    if (std::abs(denseValues[i] - regionValues[i]) > 1e-4 * std::max(1.0, std::abs(denseValues[i])))
    {
      std::cerr << "Iteration " << i << ": dense loss " << denseValues[i] << ", region loss " << regionValues[i]
                << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (regions->GetFixedImageStreamed() || regions->GetMovingImageStreamed())
  {
    std::cerr << "An image in memory was reported as streamed." << std::endl;
    return EXIT_FAILURE;
  }
  // The control grid is the same map: at a voxel, its transform's displacement is the field's.
  FineType::DisplacementFieldTransformType * control = regions->GetControlGridTransform();
  if (control->GetDisplacementField()->GetLargestPossibleRegion().GetSize()[0] != 28)
  {
    std::cerr << "The control grid has " << control->GetDisplacementField()->GetLargestPossibleRegion().GetSize()
              << " samples where 28 x 24 x 20 were expected: 24 x 20 x 16 cells, two of border each side." << std::endl;
    return EXIT_FAILURE;
  }
  {
    const ImageType::IndexType index = { { 21, 17, 13 } };
    ImageType::PointType       point;
    fixed->TransformIndexToPhysicalPoint(index, point);
    const auto mapped = control->TransformPoint(point);
    const auto field = regionField->GetPixel(index);
    for (unsigned int c = 0; c < 3; ++c)
    {
      if (std::abs((mapped[c] - point[c]) - field[c]) > 1e-4)
      {
        std::cerr << "The control grid transform displaces by " << (mapped[c] - point[c]) << " where the field holds "
                  << field[c] << " (component " << c << ")." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // 3. The pair and the starting field streamed from disk, by regions: the same field, read in pieces, produced for
  // the region asked.
  const std::string fixedFile = temporary + "/itkImpactFineRegistrationRegionTest_fixed.mha";
  const std::string movingFile = temporary + "/itkImpactFineRegistrationRegionTest_moving.mha";
  const std::string startFile = temporary + "/itkImpactFineRegistrationRegionTest_start.mha";
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(fixed, fixedFile));
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(moving, movingFile));
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(start, startFile));
  using ReaderType = itk::ImageFileReader<ImageType>;
  auto fixedReader = ReaderType::New();
  fixedReader->SetFileName(fixedFile);
  fixedReader->UseStreamingOn();
  auto movingReader = ReaderType::New();
  movingReader->SetFileName(movingFile);
  movingReader->UseStreamingOn();
  auto startReader = itk::ImageFileReader<FieldType>::New();
  startReader->SetFileName(startFile);
  startReader->UseStreamingOn();
  auto streamed = FineType::New();
  streamed->SetFixedImage(fixedReader->GetOutput());
  streamed->SetMovingImage(movingReader->GetOutput());
  streamed->SetInitialDisplacementField(startReader->GetOutput());
  Configure(streamed, 6);
  streamed->SetRegionSize(16);
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
  if (fixedReader->GetOutput()->GetBufferedRegion() == fixed->GetLargestPossibleRegion() ||
      startReader->GetOutput()->GetBufferedRegion() == fixed->GetLargestPossibleRegion())
  {
    std::cerr << "A reader holds its whole image after the run." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "bytes read over the streamed run: " << streamed->GetBytesRead() << std::endl;
  if (streamed->GetBytesRead() == 0)
  {
    std::cerr << "No bytes were counted." << std::endl;
    return EXIT_FAILURE;
  }

  // 4. The moving transform compensates the shift outside the field: the starting loss is near zero with it.
  {
    ImageType::Pointer shifted = Texture(size, 3.0, 0.0, 0.0);
    using TranslationType = itk::TranslationTransform<double, 3>;
    auto                              translation = TranslationType::New();
    TranslationType::OutputVectorType offset;
    offset[0] = -3.0;
    offset[1] = 0.0;
    offset[2] = 0.0;
    translation->SetOffset(offset);
    auto with = FineType::New();
    with->SetFixedImage(fixed);
    with->SetMovingImage(shifted);
    with->SetMovingTransform(translation);
    Configure(with, 1);
    with->SetRegionSize(16);
    ITK_TRY_EXPECT_NO_EXCEPTION(with->Update());
    auto without = FineType::New();
    without->SetFixedImage(fixed);
    without->SetMovingImage(shifted);
    Configure(without, 1);
    without->SetRegionSize(16);
    ITK_TRY_EXPECT_NO_EXCEPTION(without->Update());
    const double compensated = with->GetMetricValuesPerIteration().front();
    const double raw = without->GetMetricValuesPerIteration().front();
    std::cout << "starting loss with the moving transform " << compensated << ", without " << raw << std::endl;
    if (!(compensated < 0.05 * raw))
    {
      std::cerr << "The moving transform did not bring the moving image onto the fixed one." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 5. An initial field on a coarser grid starts the control grid as its dense field starts the whole-image run.
  {
    auto                  coarse = FieldType::New();
    FieldType::RegionType coarseRegion;
    coarseRegion.SetSize({ { 8, 7, 6 } });
    coarse->SetRegions(coarseRegion);
    FieldType::SpacingType coarseSpacing;
    coarseSpacing.Fill(6.0);
    coarse->SetSpacing(coarseSpacing);
    FieldType::PointType coarseOrigin;
    coarseOrigin.Fill(2.5);
    coarse->SetOrigin(coarseOrigin);
    coarse->Allocate();
    itk::ImageRegionIteratorWithIndex<FieldType> cell(coarse, coarseRegion);
    for (cell.GoToBegin(); !cell.IsAtEnd(); ++cell)
    {
      const auto           index = cell.GetIndex();
      FieldType::PixelType value;
      value[0] = static_cast<float>(2.0 * std::sin(index[0] / 1.3));
      value[1] = static_cast<float>(2.0 * std::cos(index[1] / 1.7));
      value[2] = static_cast<float>(std::sin((index[0] + index[2]) / 1.1));
      cell.Set(value);
    }
    auto interpolator = itk::VectorLinearInterpolateImageFunction<FieldType, double>::New();
    interpolator->SetInputImage(coarse);
    auto resized = FieldType::New();
    resized->SetRegions(fixed->GetLargestPossibleRegion());
    resized->Allocate();
    itk::ImageRegionIteratorWithIndex<FieldType> voxel(resized, resized->GetLargestPossibleRegion());
    for (voxel.GoToBegin(); !voxel.IsAtEnd(); ++voxel)
    {
      FieldType::PointType point;
      resized->TransformIndexToPhysicalPoint(voxel.GetIndex(), point);
      const auto           value = interpolator->Evaluate(point);
      FieldType::PixelType pixel;
      for (unsigned int c = 0; c < 3; ++c)
      {
        pixel[c] = static_cast<float>(value[c]);
      }
      voxel.Set(pixel);
    }
    auto whole = FineType::New();
    whole->SetFixedImage(fixed);
    whole->SetMovingImage(moving);
    whole->SetInitialDisplacementField(resized);
    Configure(whole, 0);
    ITK_TRY_EXPECT_NO_EXCEPTION(whole->Update());
    auto byRegions = FineType::New();
    byRegions->SetFixedImage(fixed);
    byRegions->SetMovingImage(moving);
    byRegions->SetInitialDisplacementField(coarse);
    Configure(byRegions, 0);
    byRegions->SetRegionSize(16);
    ITK_TRY_EXPECT_NO_EXCEPTION(byRegions->Update());
    const double difference =
      MaxDifference(byRegions->GetOutput(), whole->GetOutput(), fixed->GetLargestPossibleRegion());
    std::cout << "initial field on a coarser grid, by regions against its dense field whole: max " << difference
              << std::endl;
    if (!(difference < 1e-4))
    {
      std::cerr << "An initial field on a coarser grid is not read as its dense field." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 6. A pair of odd sizes, whose tail the control cells leave out: by regions as whole, on the intensities and on a
  // model whose layers are half the image's resolution (an average pooling and a strided convolution), whose whole
  // layers the cells pool by uneven windows.
  for (const bool withModel : { false, true })
  {
    if (withModel && argc < 3)
    {
      continue;
    }
    const ImageType::SizeType odd = { { 49, 41, 33 } };
    ImageType::Pointer        oddFixed = Texture(odd, 0.0, 0.0, 0.0);
    ImageType::Pointer        oddMoving = Texture(odd, 1.5, -1.0, 0.5);
    const FieldType::Pointer  oddStart = StartingField(oddFixed);
    auto                      run = [&](bool byRegions) {
      auto fine = FineType::New();
      fine->SetFixedImage(oddFixed);
      fine->SetMovingImage(oddMoving);
      fine->SetInitialDisplacementField(oddStart);
      if (withModel)
      {
        fine->AddModelConfiguration(itk::ImpactModelConfiguration(
          argv[2], 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, true }, false));
        fine->SetDistance({ "L2", "L2" });
        fine->SetLayersWeight({ 1.f, 1.f });
      }
      Configure(fine, 6);
      if (byRegions)
      {
        fine->SetRegionSize(16);
        fine->SetRegionHalo(2);
      }
      fine->Update();
      return fine;
    };
    const auto   whole = run(false);
    const auto   byRegions = run(true);
    const char * label = withModel ? "half-resolution layers" : "intensities";
    const double moved = MaxDifference(whole->GetOutput(), oddStart, oddFixed->GetLargestPossibleRegion());
    const double difference =
      MaxDifference(whole->GetOutput(), byRegions->GetOutput(), oddFixed->GetLargestPossibleRegion());
    std::cout << "odd sizes, " << label << ", by regions against whole: max |difference| = " << difference
              << " mm, for a field moved by " << moved << " mm" << std::endl;
    if (!(difference < 1e-5) || !(moved > 1e-3))
    {
      std::cerr << "On odd sizes, " << label << " by regions differ from the whole-image run." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 7. The least boxes: a control grid of 9 x 7 x 5 cells by boxes of 2, whose last along each axis is one cell,
  // narrower than the smoothing's kernel. With the maps resident and by bricks of one box (BrickBytes 1, the
  // least): the whole-image run's field.
  {
    const ImageType::SizeType few = { { 18, 14, 10 } };
    ImageType::Pointer        fewFixed = Texture(few, 0.0, 0.0, 0.0);
    ImageType::Pointer        fewMoving = Texture(few, 1.5, -1.0, 0.5);
    const FieldType::Pointer  fewStart = StartingField(fewFixed);
    auto                      run = [&](unsigned int regionSize, bool mapped) {
      auto fine = FineType::New();
      fine->SetFixedImage(fewFixed);
      fine->SetMovingImage(fewMoving);
      fine->SetInitialDisplacementField(fewStart);
      Configure(fine, 6);
      fine->SetRegionSize(regionSize);
      if (mapped)
      {
        fine->SetResidentFeatureBudget(1);
        fine->SetBrickBytes(1);
        fine->SetScratchDirectory(scratch);
      }
      fine->Update();
      return fine;
    };
    const auto   whole = run(0, false);
    const double residentDifference =
      MaxDifference(whole->GetOutput(), run(4, false)->GetOutput(), fewFixed->GetLargestPossibleRegion());
    const double mappedDifference =
      MaxDifference(whole->GetOutput(), run(4, true)->GetOutput(), fewFixed->GetLargestPossibleRegion());
    std::cout << "boxes whose last is one cell, against whole: max |difference| = " << residentDifference
              << " mm with the maps resident, " << mappedDifference << " mm by bricks of one box" << std::endl;
    if (!(residentDifference < 1e-5) || !(mappedDifference < 1e-5))
    {
      std::cerr << "By boxes whose last is one cell, the field differs from the whole-image run's." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 8. RegionsPerIteration: an iteration steps the cells of the boxes it draws, with the whole grid's gradient.
  {
    auto drawing = [&](unsigned int regionsPerIteration, unsigned int iterations, double epsilon, bool mapped) {
      auto fine = FineType::New();
      fine->SetFixedImage(fixed);
      fine->SetMovingImage(moving);
      fine->SetInitialDisplacementField(start);
      Configure(fine, iterations, epsilon);
      fine->SetRegionSize(16);
      fine->SetRegionsPerIteration(regionsPerIteration);
      if (mapped)
      {
        fine->SetResidentFeatureBudget(1);
        fine->SetScratchDirectory(scratch);
      }
      fine->Update();
      return fine;
    };
    auto differenceOf = [&](const FineType * a, const FieldType * b) {
      return MaxDifference(a->GetOutput(), b, fixed->GetLargestPossibleRegion());
    };
    // All 18 boxes at every iteration: the run that steps every cell, with the maps resident and in mapped files.
    const auto everyResident = drawing(18, 6, 1.0, false);
    ITK_TEST_SET_GET_VALUE(18u, everyResident->GetRegionsPerIteration());
    const auto   everyMapped = drawing(1000, 6, 1.0, true);
    const double residentDifference = differenceOf(everyResident, denseField);
    const double mappedDifference = differenceOf(everyMapped, denseField);
    std::cout << "every box drawn at every iteration, against the dense run: max |difference| = " << residentDifference
              << " mm with the maps resident, " << mappedDifference << " mm from mapped files" << std::endl;
    if (everyResident->GetMetricValuesPerIteration().size() != 6 || !everyResident->GetResidentFeatures() ||
        everyMapped->GetResidentFeatures() || !(residentDifference < 1e-5) || !(mappedDifference < 1e-5))
    {
      std::cerr << "Drawing every box at every iteration is not the run that steps every cell." << std::endl;
      return EXIT_FAILURE;
    }
    // Three boxes an iteration: one field for one seed, from mapped files as with the maps resident.
    const auto   fewMapped = drawing(3, 36, 1.0, true);
    const double againDifference = differenceOf(drawing(3, 36, 1.0, true), fewMapped->GetOutput());
    const double routeDifference = differenceOf(drawing(3, 36, 1.0, false), fewMapped->GetOutput());
    std::cout << "three boxes an iteration: max |difference| = " << againDifference << " mm between two runs, "
              << routeDifference << " mm between mapped files and resident maps" << std::endl;
    if (fewMapped->GetMetricValuesPerIteration().size() != 36 || againDifference != 0.0 || !(routeDifference < 1e-5))
    {
      std::cerr << "Three boxes an iteration do not give one field for one seed." << std::endl;
      return EXIT_FAILURE;
    }
    // And the field tends to that of every cell at every iteration: at as many steps a cell on average (six
    // iterations of 3 boxes for one of all 18), the two fields close in as the steps grow. Under an epsilon of the
    // gradients' order, with which Adam's steps die out near the minimum both runs go to. Measured: 0.40, 0.17 and
    // 0.044 mm at 15, 45 and 135 steps a cell, for fields that moved by 2.2 mm.
    double previous = std::numeric_limits<double>::infinity();
    for (const unsigned int steps : { 15u, 45u, 135u })
    {
      const double distance =
        differenceOf(drawing(3, 6 * steps, 1e-4, false), drawing(0, steps, 1e-4, false)->GetOutput());
      std::cout << "three boxes an iteration against every cell, " << steps
                << " steps a cell: max |difference| = " << distance << " mm" << std::endl;
      if (!(distance < previous / 1.5))
      {
        std::cerr << "The field of three boxes an iteration does not close in on that of every cell." << std::endl;
        return EXIT_FAILURE;
      }
      previous = distance;
    }
    if (!std::filesystem::is_empty(scratch))
    {
      std::cerr << "The runs that draw their boxes left files in the scratch directory." << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

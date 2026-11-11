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

// The region-sampled run of ImpactFineRegistration against its whole-image run, on the intensity similarity:
// every box at every iteration reproduces the dense field; a streaming reader gives the same field as the image in
// memory and is read by regions; a transform set on the moving side is applied outside the field.

#include "itkImpactFineRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTestingMacros.h"
#include "itkTranslationTransform.h"
#include "itkVectorLinearInterpolateImageFunction.h"

#include <cmath>
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
      largest = std::max(largest, static_cast<double>(std::abs(va[c] - vb[c])));
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
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv) << " <temporary directory>" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string         temporary = argv[1];
  const ImageType::SizeType size = { { 48, 40, 32 } }; // whole cells of 2 voxels, whole boxes of 16
  ImageType::Pointer        fixed = Texture(size, 0.0, 0.0, 0.0);
  ImageType::Pointer        moving = Texture(size, 1.5, -1.0, 0.5);

  // 1. The whole-image run.
  auto dense = FineType::New();
  ITK_EXERCISE_BASIC_OBJECT_METHODS(dense, ImpactFineRegistration, ImageSource);
  dense->SetFixedImage(fixed);
  dense->SetMovingImage(moving);
  Configure(dense, 6);
  ITK_TRY_EXPECT_NO_EXCEPTION(dense->Update());
  const FieldType * denseField = dense->GetOutput();

  // 2. By regions, the pooled maps resident (one pass over the boxes, then the dense iterations on them): the same
  // field. Then box by box, every box at every iteration: the same field again.
  auto regions = FineType::New();
  regions->SetFixedImage(fixed);
  regions->SetMovingImage(moving);
  Configure(regions, 6);
  regions->SetRegionSize(16);
  regions->SetRegionsPerIteration(1000);
  ITK_TEST_SET_GET_VALUE(16u, regions->GetRegionSize());
  ITK_TEST_SET_GET_VALUE(1000u, regions->GetRegionsPerIteration());
  ITK_TRY_EXPECT_NO_EXCEPTION(regions->Update());
  if (!regions->GetResidentFeatures())
  {
    std::cerr << "The pooled maps of a small pair should stay resident." << std::endl;
    return EXIT_FAILURE;
  }
  const FieldType * regionField = regions->GetOutput();
  auto              boxes = FineType::New();
  boxes->SetFixedImage(fixed);
  boxes->SetMovingImage(moving);
  Configure(boxes, 6);
  boxes->SetRegionSize(16);
  boxes->SetRegionsPerIteration(1000);
  boxes->SetResidentFeatureBudget(1); // nothing fits: box by box
  ITK_TEST_SET_GET_VALUE(1u, boxes->GetResidentFeatureBudget());
  ITK_TRY_EXPECT_NO_EXCEPTION(boxes->Update());
  if (boxes->GetResidentFeatures())
  {
    std::cerr << "A budget of one byte should leave nothing resident." << std::endl;
    return EXIT_FAILURE;
  }
  if (boxes->GetBytesRead() != regions->GetBytesRead())
  {
    std::cerr << "Box by box, the images should be read once, as with the maps resident: " << boxes->GetBytesRead()
              << " bytes against " << regions->GetBytesRead() << "." << std::endl;
    return EXIT_FAILURE;
  }
  const double boxDifference = MaxDifference(denseField, boxes->GetOutput(), fixed->GetLargestPossibleRegion());
  std::cout << "dense vs box by box (every box): max |difference| = " << boxDifference << " mm" << std::endl;
  if (boxDifference > 1e-3)
  {
    std::cerr << "The box-by-box field differs from the dense one by " << boxDifference << " mm." << std::endl;
    return EXIT_FAILURE;
  }
  {
    const auto & a = dense->GetMetricValuesPerIteration();
    const auto & b = regions->GetMetricValuesPerIteration();
    for (size_t i = 0; i < a.size(); ++i)
    {
      std::cout << "iteration " << i << ": dense " << a[i] << " regions " << b[i] << std::endl;
    }
    const double controlDifference =
      MaxDifference(dense->GetControlGridTransform()->GetDisplacementField(),
                    regions->GetControlGridTransform()->GetDisplacementField(),
                    dense->GetControlGridTransform()->GetDisplacementField()->GetLargestPossibleRegion());
    std::cout << "control grids: max |difference| = " << controlDifference << " mm" << std::endl;
    {
      const FieldType *                                 ca = dense->GetControlGridTransform()->GetDisplacementField();
      const FieldType *                                 cb = regions->GetControlGridTransform()->GetDisplacementField();
      double                                            sum = 0.0;
      size_t                                            large = 0, count = 0;
      itk::ImageRegionConstIteratorWithIndex<FieldType> it(ca, ca->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
        const auto va = it.Get(), vb = cb->GetPixel(it.GetIndex());
        for (unsigned int c = 0; c < 3; ++c)
        {
          const double d = std::abs(va[c] - vb[c]);
          sum += d;
          large += d > 0.05;
          ++count;
        }
      }
      std::cout << "control grids: mean |difference| = " << sum / count << " mm, " << large << " of " << count
                << " entries differ by more than 0.05 mm" << std::endl;
    }
  }
  const double difference = MaxDifference(denseField, regionField, fixed->GetLargestPossibleRegion());
  std::cout << "dense vs regions (every box): max |difference| = " << difference << " mm" << std::endl;
  if (difference > 1e-3)
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
              << " samples where 28 x 24 x 20 were expected: 24 x 20 x 16 cells, two of border each side."
              << std::endl;
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

  // 3. The pair streamed from disk, by regions: the same field, read in pieces, produced for the region asked.
  const std::string fixedFile = temporary + "/itkImpactFineRegistrationRegionTest_fixed.mha";
  const std::string movingFile = temporary + "/itkImpactFineRegistrationRegionTest_moving.mha";
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(fixed, fixedFile));
  ITK_TRY_EXPECT_NO_EXCEPTION(itk::WriteImage(moving, movingFile));
  using ReaderType = itk::ImageFileReader<ImageType>;
  auto fixedReader = ReaderType::New();
  fixedReader->SetFileName(fixedFile);
  fixedReader->UseStreamingOn();
  auto movingReader = ReaderType::New();
  movingReader->SetFileName(movingFile);
  movingReader->UseStreamingOn();
  auto streamed = FineType::New();
  streamed->SetFixedImage(fixedReader->GetOutput());
  streamed->SetMovingImage(movingReader->GetOutput());
  Configure(streamed, 6);
  streamed->SetRegionSize(16);
  streamed->SetRegionsPerIteration(1000);
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

  // 4. Box by box, a share of the boxes at every iteration, every box visited 10 times: converges towards the shift,
  // moving(x + u) = fixed(x) with moving(y) = f(y + s) asking for u = -s.
  auto partial = FineType::New();
  partial->SetFixedImage(fixed);
  partial->SetMovingImage(moving);
  Configure(partial, 10, 1e-8); // Adam as configured by default: full steps
  partial->SetRegionSize(16);
  partial->SetRegionsPerIteration(3);
  partial->SetResidentFeatureBudget(1);
  ITK_TRY_EXPECT_NO_EXCEPTION(partial->Update());
  {
    FieldType::RegionType centre = fixed->GetLargestPossibleRegion();
    for (unsigned int a = 0; a < 3; ++a)
    {
      centre.SetIndex(a, 8);
      centre.SetSize(a, centre.GetSize()[a] - 16);
    }
    itk::Vector<double, 3> mean;
    mean.Fill(0.0);
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(partial->GetOutput(), centre);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      mean += it.Get();
    }
    mean /= static_cast<double>(centre.GetNumberOfPixels());
    std::cout << "partial schedule: mean displacement " << mean << " mm for a shift of (-1.5, 1, -0.5)" << std::endl;
    const double expected[3] = { -1.5, 1.0, -0.5 };
    for (unsigned int c = 0; c < 3; ++c)
    {
      if (std::abs(mean[c] - expected[c]) > 0.6)
      {
        std::cerr << "The partial schedule did not recover the shift." << std::endl;
        return EXIT_FAILURE;
      }
    }
    const auto & values = partial->GetMetricValuesPerIteration();
    if (!(values.back() < values.front()))
    {
      std::cerr << "The loss did not decrease: " << values.front() << " -> " << values.back() << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 5. The moving transform compensates the shift outside the field: the starting loss is near zero with it.
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
    with->SetRegionsPerIteration(1000);
    ITK_TRY_EXPECT_NO_EXCEPTION(with->Update());
    auto without = FineType::New();
    without->SetFixedImage(fixed);
    without->SetMovingImage(shifted);
    Configure(without, 1);
    without->SetRegionSize(16);
    without->SetRegionsPerIteration(1000);
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

  // 6. An initial field on a coarser grid starts the control grid as its dense field starts the whole-image run.
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
      const auto            index = cell.GetIndex();
      FieldType::PixelType  value;
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

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

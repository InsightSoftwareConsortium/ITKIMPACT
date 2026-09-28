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

#include "gtest/gtest.h"

#include <chrono>
#include <thread>

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkImageToTensorFilter.h"
#include "itkImpactModelConfiguration.h"
#include "itkImageToFeaturesMap.h"
#include "itkImageToFeaturesMapInternals.h"
#include "itkImpactOnlineInference.h"
#include "itkImpactBatchBudget.h"
#include "ImpactLoss.h"
#include "itkImpactImageToImageMetricv4.h"
#include "itkImpactFineRegistration.h"
#include "itkImpactCoarseRegistration.h"
#include "itkImageFileReader.h"
#include "itkMetaImageIO.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include <torch/torch.h>
#include "vnl/vnl_det.h"
#include "itkAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkDisplacementFieldTransform.h"
#include "itkTranslationTransform.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace
{
using ImageType = itk::Image<float, 3>;
using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double>;

ImageType::Pointer
MakeRampImage(const ImageType::SizeType & size, int mode = 0)
{
  auto                  image = ImageType::New();
  ImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto i = it.GetIndex();
    image->SetPixel(it.GetIndex(),
                    (mode == 0) ? static_cast<float>(i[0] + 10 * i[1] + 100 * i[2])
                                : static_cast<float>(2 * i[0] + i[1] * i[1] + 3 * i[2]));
  }
  return image;
}

ImageType::Pointer
MakeRampImage(unsigned int n, int mode = 0)
{
  ImageType::SizeType size;
  size.Fill(n);
  return MakeRampImage(size, mode);
}

// A smooth Gaussian blob centered at (n/2 + c*) -- structured content so that a
// translation has a well-defined optimal alignment (unlike a linear ramp).
ImageType::Pointer
MakeBlobImage(unsigned int n, double cx, double cy, double cz, double sigma)
{
  auto                  image = ImageType::New();
  ImageType::SizeType   size;
  size.Fill(n);
  ImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->Allocate();
  const double                                 centerX = n / 2.0 + cx;
  const double                                 centerY = n / 2.0 + cy;
  const double                                 centerZ = n / 2.0 + cz;
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto   i = it.GetIndex();
    const double d2 = (i[0] - centerX) * (i[0] - centerX) + (i[1] - centerY) * (i[1] - centerY) +
                      (i[2] - centerZ) * (i[2] - centerZ);
    image->SetPixel(i, static_cast<float>(std::exp(-d2 / (2.0 * sigma * sigma))));
  }
  return image;
}

std::string
ToyModelPath()
{
  return std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel.pt";
}
} // namespace

// --- A1: ImageToTensorFilter resamples an image into a torch tensor ----------
TEST(ImpactBackend, ImageToTensorFilterResampling)
{
  using FilterType = itk::ImageToTensorFilter<ImageType, InterpolatorType>;
  const ImageType::SizeType size = { { 8, 6, 4 } }; // distinct dims to verify axis order
  auto image = MakeRampImage(size);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  auto filter = FilterType::New();
  filter->AddInput(image);
  filter->SetInterpolator(interpolator);
  FilterType::InputSpacingType outSpacing;
  outSpacing.Fill(1.0);
  filter->SetOutputSpacing(outSpacing);
  ASSERT_NO_THROW(filter->Update());

  const torch::Tensor t = filter->GetTensor();
  ASSERT_EQ(t.dim(), 3);
  // torch axis order is the reverse of the ITK index order: {z, y, x}.
  EXPECT_EQ(t.size(0), 4);
  EXPECT_EQ(t.size(1), 6);
  EXPECT_EQ(t.size(2), 8);
  // t[z][y][x] == ramp(x, y, z)
  EXPECT_NEAR(t[2][3][5].item<float>(), 235.0f, 1e-2f);
}

// --- A4: Dice loss is soft, NaN-safe and does not mutate its inputs ----------
TEST(ImpactBackend, ImpactLossFactoryHasAllLosses)
{
  for (const char * name : { "L1", "L2", "Dice", "L1Cosine", "Cosine", "NCC" })
  {
    std::unique_ptr<itk::Impact::Loss> loss;
    ASSERT_NO_THROW(loss = itk::Impact::LossFactory::Instance().Create(name)) << name;
    EXPECT_NE(loss, nullptr) << name;
  }
  EXPECT_THROW(itk::Impact::LossFactory::Instance().Create("DoesNotExist"), std::runtime_error);
  // Unbounded, so it could not start at 1 like the others: removed.
  EXPECT_THROW(itk::Impact::LossFactory::Instance().Create("DotProduct"), std::runtime_error);
}

TEST(ImpactBackend, ImpactLossDiceNoMutationAndNaNSafe)
{
  // Does not mutate its inputs.
  {
    auto          dice = itk::Impact::LossFactory::Instance().Create("Dice");
    dice->SetNumberOfParameters(1);
    torch::Tensor f = torch::rand({ 4, 3 });
    torch::Tensor m = torch::rand({ 4, 3 });
    torch::Tensor fClone = f.clone();
    torch::Tensor mClone = m.clone();
    dice->updateValue(f, m);
    EXPECT_TRUE(torch::equal(f, fClone));
    EXPECT_TRUE(torch::equal(m, mClone));
  }
  // Identical inputs => perfect overlap (Dice == 1) => value == -1.
  {
    auto          dice = itk::Impact::LossFactory::Instance().Create("Dice");
    dice->SetNumberOfParameters(1);
    torch::Tensor f = torch::ones({ 4, 3 });
    torch::Tensor m = f.clone();
    dice->updateValue(f, m);
    EXPECT_NEAR(dice->GetValue(4.0), 0.0, 1e-5);
  }
  // Empty/empty (all zeros) => finite value and gradient (no 0/0 NaN).
  {
    auto          dice = itk::Impact::LossFactory::Instance().Create("Dice");
    dice->SetNumberOfParameters(1);
    torch::Tensor f = torch::zeros({ 4, 3 });
    torch::Tensor m = torch::zeros({ 4, 3 });
    torch::Tensor grad = dice->updateValueAndGetGradientModulator(f, m);
    EXPECT_NEAR(dice->GetValue(4.0), 0.0, 1e-5);
    EXPECT_FALSE(torch::isnan(grad).any().item<bool>());
    EXPECT_FALSE(torch::isinf(grad).any().item<bool>());
  }
}

// --- A2/A3: ImageToFeaturesMap inference on CPU, and the PCA basis is cached --
TEST(ImpactBackend, ImageToFeaturesMapProducesFeatureLayers)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  auto image = MakeRampImage(8);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  // Toy model returns 2 layers: passthrough (2 channels) and conv features (4).
  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 2, 2, 2 }, { true, true }, /*mixedPrecision*/ false);

  auto filter = FeatMapType::New();
  filter->SetModelConfiguration(config);
  filter->SetInterpolator(interpolator);
  filter->AddInput(image);
  filter->SetPCA(0);
  filter->SetDevice("cpu");
  ASSERT_NO_THROW(filter->Update());

  EXPECT_EQ(filter->GetOutput(0)->GetNumberOfComponentsPerPixel(), 4u); // conv features
  EXPECT_EQ(filter->GetOutput(1)->GetNumberOfComponentsPerPixel(), 2u); // passthrough
}

// --- Regression: overlap == 0 must NOT collapse the assembled feature map -----------
// Accumulator::assemble() cropped with Slice(m_Overlap, -m_Overlap); for overlap == 0
// that is Slice(0, -0) == Slice(0, 0), a ZERO-length range in libtorch (not "keep all"),
// which collapsed the feature map to spatial size 0 and made downstream interpolation
// produce NaNs. The full-image (patchSize 0) + overlap-0 path must keep the input size.
TEST(ImpactBackend, ImageToFeaturesMapOverlapZeroKeepsSize)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  auto image = MakeRampImage(8);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);
  // overlap = 0 (6th ctor arg), patchSize {0,0,0} = full image -> single patch.
  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false);
  auto                          filter = FeatMapType::New();
  filter->SetModelConfiguration(config);
  filter->SetInterpolator(interpolator);
  filter->AddInput(image);
  filter->SetPCA(0);
  filter->SetDevice("cpu");
  ASSERT_NO_THROW(filter->Update());
  // The toy conv keeps the spatial size, so the feature map must stay 8^3, not collapse to 0.
  const auto size = filter->GetOutput(0)->GetLargestPossibleRegion().GetSize();
  EXPECT_EQ(size[0], 8u);
  EXPECT_EQ(size[1], 8u);
  EXPECT_EQ(size[2], 8u);
}

TEST(ImpactBackend, ImageToFeaturesMapReusesInjectedPcaBasis)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  auto runOn = [&](ImageType::Pointer image, const std::vector<torch::Tensor> * inject) {
    itk::ImpactModelConfiguration config(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 2, 2, 2 }, { true, true }, false);
    auto filter = FeatMapType::New();
    filter->SetModelConfiguration(config);
    filter->SetInterpolator(interpolator);
    filter->AddInput(image);
    filter->SetPCA(2); // reduce the 4-channel layer to 2 components
    filter->SetDevice("cpu");
    if (inject)
    {
      itk::SetPrincipalComponents(*filter, *inject);
    }
    filter->Update();
    return itk::GetPrincipalComponents(*filter);
  };

  const std::vector<torch::Tensor> basisA = runOn(MakeRampImage(8, 0), nullptr);
  const std::vector<torch::Tensor> basisBReused = runOn(MakeRampImage(8, 1), &basisA);
  const std::vector<torch::Tensor> basisBRefit = runOn(MakeRampImage(8, 1), nullptr);

  ASSERT_EQ(basisA.size(), 2u);
  // Layer 0 is the 4-channel conv feature map, reduced 4 -> 2.
  ASSERT_TRUE(basisA[0].defined());
  EXPECT_EQ(basisA[0].size(0), 4);
  EXPECT_EQ(basisA[0].size(1), 2);
  // Injecting A's basis reuses it verbatim; refitting on B yields a different one.
  EXPECT_TRUE(torch::equal(basisBReused[0], basisA[0]));
  EXPECT_FALSE(torch::equal(basisBRefit[0], basisA[0]));
}

// --- A2: the model and patches run on CUDA when a GPU is available ------------
TEST(ImpactBackend, ImageToFeaturesMapOnCudaWhenAvailable)
{
  if (!torch::cuda::is_available())
  {
    GTEST_SKIP() << "no CUDA device available";
  }
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  auto image = MakeRampImage(8);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 2, 2, 2 }, { true, true }, false);

  auto filter = FeatMapType::New();
  filter->SetModelConfiguration(config);
  filter->SetInterpolator(interpolator);
  filter->AddInput(image);
  filter->SetPCA(0);
  filter->SetDevice("cuda:0"); // model is moved to the GPU by A2
  ASSERT_NO_THROW(filter->Update());
  EXPECT_EQ(filter->GetOutput(0)->GetNumberOfComponentsPerPixel(), 4u); // conv features
}

// Static extracts the moving features once, from the image as acquired, and then samples that
// frozen map at the transformed point -- so the network never sees the anatomy as it currently
// aligns. SetFeaturesMapUpdateInterval() re-extracts the map with the transform of the moment
// applied. The knob existed and was read by nothing, so the map never moved.
TEST(ImpactMetric, StaticFeaturesMapRefreshFollowsTheTransform)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  // 24 voxels: the 3-voxel shift below keeps the blob and its tails inside the rebuilt map,
  // so what the round trip reads back is the blob and not the map's edge.
  auto fixed = MakeBlobImage(24, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(24, 1.2, -0.8, 0.5, 2.0);

  // The passthrough layer is the one to watch: its "features" ARE the intensities, so a map
  // that followed the transform is directly readable as the resampled image.
  auto build = [&](int interval, TransformType * transform) {
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 2, 2, 2 },
                         std::vector<bool>{ false, true }, // passthrough only
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ "L2" });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 2 });
    metric->SetPCA({ 0 });
    metric->SetMode("Static");
    metric->SetSeed(1);
    metric->SetDevice("cpu");
    metric->SetFeaturesMapUpdateInterval(interval);
    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
    metric->SetMovingTransform(transform);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();
    return metric;
  };

  MetricType::MeasureType    value;
  MetricType::DerivativeType derivative;

  // Refresh disabled: the map is built once at Initialize() and must never move again, however
  // far the transform travels.
  auto transformOff = TransformType::New();
  transformOff->SetIdentity();
  auto off = build(0, transformOff);
  off->GetValueAndDerivative(value, derivative);
  const double offBefore = static_cast<double>(off->GetValue());
  MetricType::ParametersType shifted = transformOff->GetParameters();
  shifted[0] += 3.0;
  transformOff->SetParameters(shifted);
  off->GetValueAndDerivative(value, derivative); // would refresh, if the interval allowed it
  transformOff->SetParameters(MetricType::ParametersType(shifted.GetSize(), 0.0));
  EXPECT_DOUBLE_EQ(static_cast<double>(off->GetValue()), offBefore)
    << "with the refresh disabled the moving map must be exactly the one built at Initialize()";

  // Refresh every evaluation: the map is rebuilt while the transform is displaced, and read at
  // the residual point afterwards. For a translation the model is equivariant, so the round trip
  // comes back to the same value up to one extra resampling; only the refresh test below can
  // tell a rebuilt map from a stale one, at the moment of the refresh.
  auto transformOn = TransformType::New();
  transformOn->SetIdentity();
  auto on = build(1, transformOn);
  on->GetValueAndDerivative(value, derivative);
  const double onBefore = static_cast<double>(on->GetValue());
  MetricType::ParametersType shiftedOn = transformOn->GetParameters();
  shiftedOn[0] += 3.0;
  transformOn->SetParameters(shiftedOn);
  on->GetValueAndDerivative(value, derivative); // rebuilds the map at the displaced transform
  transformOn->SetParameters(MetricType::ParametersType(shiftedOn.GetSize(), 0.0));
  EXPECT_NEAR(static_cast<double>(on->GetValue()), onBefore, 0.02 * onBefore)
    << "a translation round trip through a refresh must come back to the same value";

  // And the refresh must not cost the derivative: the moving features are still read at the
  // transformed point, which is the only parameter-dependent point there is.
  double norm = 0.0;
  for (unsigned int j = 0; j < derivative.GetSize(); ++j)
    norm += derivative[j] * derivative[j];
  EXPECT_GT(norm, 1e-12) << "a refreshed map must still produce a gradient";
}

// A refreshed map is built by resampling the moving image through the transform of the moment,
// so its value at p is M(T_k(p)). The header promises the approximation is exact at the moment
// of the refresh. Reading that map at the transformed point T(x) gives M(T_k(T(x))): with T == T_k
// the transform is applied twice, and the value at the aligning parameters is the value of the
// unrefreshed metric at twice the shift. Blob images and the passthrough layer make the metric
// steep enough in the translation for the two readings to be an order of magnitude apart.
TEST(ImpactMetric, StaticRefreshIsExactAtTheMomentOfTheRefresh)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  auto fixed = MakeBlobImage(12, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(12, 3.0, 0.0, 0.0, 2.0); // the same blob, 3 voxels further along x

  auto build = [&](int interval, TransformType * transform) {
    auto                                       metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 2, 2, 2 },
                         std::vector<bool>{ false, true }, // passthrough: the features ARE the intensities
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ "L2" });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 2 });
    metric->SetPCA({ 0 });
    metric->SetMode("Static");
    metric->SetSeed(1);
    metric->SetDevice("cpu");
    metric->SetFeaturesMapUpdateInterval(interval);
    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
    metric->SetMovingTransform(transform);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();
    return metric;
  };
  auto shift = [](double x) {
    MetricType::ParametersType p(3);
    p.Fill(0.0);
    p[0] = x;
    return p;
  };

  // The unrefreshed metric, at the aligning shift and at twice that shift.
  auto reference = TransformType::New();
  auto plain = build(-1, reference);
  reference->SetParameters(shift(3.0));
  const double atAligned = static_cast<double>(plain->GetValue());
  reference->SetParameters(shift(6.0));
  const double atTwice = static_cast<double>(plain->GetValue());
  ASSERT_GT(atTwice, 3.0 * atAligned) << "the fixture must make the two readings distinguishable";

  // Refresh every evaluation: one GetValueAndDerivative at the aligning shift rebuilds the map
  // there, and the value read right after must still be the aligned one.
  auto candidate = TransformType::New();
  auto refreshing = build(1, candidate);
  candidate->SetParameters(shift(3.0));
  MetricType::MeasureType    value;
  MetricType::DerivativeType derivative;
  refreshing->GetValueAndDerivative(value, derivative);
  const double afterRefresh = static_cast<double>(refreshing->GetValue());

  EXPECT_NEAR(afterRefresh, atAligned, 0.05 * (atTwice - atAligned))
    << "after a refresh at the aligning shift the metric reads " << afterRefresh << ", the unrefreshed metric reads "
    << atAligned << " there and " << atTwice << " at twice the shift";
}

// --- B: the Static-mode metric value is ~0 for identical images, > 0 otherwise
TEST(ImpactMetric, StaticValueZeroForIdenticalPositiveForDifferent)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeRampImage(8);

  auto buildAndEvaluate = [&](ImageType::Pointer moving) -> double {
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 2, 2, 2 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ "Cosine" });   // non-normalized: value is the actual similarity
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 4 });     // use all 4 conv feature channels deterministically
    metric->SetPCA({ 0 });
    metric->SetMode("Static");
    metric->SetSeed(1);
    metric->SetFeaturesMapUpdateInterval(-1);
    metric->SetDevice("cpu");

    auto identity = itk::IdentityTransform<double, 3>::New();
    auto affine = itk::AffineTransform<double, 3>::New();
    affine->SetIdentity();

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identity);
    metric->SetMovingTransform(affine);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(1); // single-threaded (B-spline interpolators)
    metric->Initialize();
    return static_cast<double>(metric->GetValue());
  };

  const double valueSame = buildAndEvaluate(fixed);
  EXPECT_NEAR(valueSame, 0.0, 1e-3) << "identical features => cosine 1 => value 0";

  const double valueDifferent = buildAndEvaluate(MakeRampImage(8, 1));
  EXPECT_GT(valueDifferent, valueSame) << "a different image is less similar => higher loss";
  EXPECT_LE(valueDifferent, 2.0 + 1e-6);
}

// --- C: the Static-mode derivative matches finite differences of the value ----
TEST(ImpactMetric, StaticDerivativeMatchesFiniteDifferences)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1); // different => non-trivial gradient

  auto checkLoss = [&](const char * lossName) {
    SCOPED_TRACE(std::string("loss ") + lossName);
    SCOPED_TRACE(lossName);
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 2, 2, 2 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ std::string(lossName) });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 4 }); // all features => deterministic (no random subset)
    metric->SetPCA({ 0 });
    metric->SetMode("Static");
    metric->SetSeed(1);
    metric->SetFeaturesMapUpdateInterval(-1);
    metric->SetDevice("cpu");

    auto identityFixed = itk::IdentityTransform<double, 3>::New();
    auto affine = itk::AffineTransform<double, 3>::New();
    affine->SetIdentity();

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identityFixed);
    metric->SetMovingTransform(affine);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();

    MetricType::MeasureType    value;
    MetricType::DerivativeType analyticDerivative;
    metric->GetValueAndDerivative(value, analyticDerivative);
    ASSERT_EQ(analyticDerivative.GetSize(), affine->GetNumberOfParameters());

    const double                     h = 1e-3;
    const MetricType::ParametersType params0 = affine->GetParameters();
    for (unsigned int j = 0; j < affine->GetNumberOfParameters(); ++j)
    {
      MetricType::ParametersType pPlus = params0;
      pPlus[j] += h;
      affine->SetParameters(pPlus);
      const double vPlus = static_cast<double>(metric->GetValue());

      MetricType::ParametersType pMinus = params0;
      pMinus[j] -= h;
      affine->SetParameters(pMinus);
      const double vMinus = static_cast<double>(metric->GetValue());

      affine->SetParameters(params0);

      const double fd = (vPlus - vMinus) / (2.0 * h);
      // The metric reports the descent direction (-gradient), hence compare to -fd.
      const double analytic = static_cast<double>(analyticDerivative[j]);
      EXPECT_NEAR(analytic, -fd, 1e-2 + 0.05 * std::abs(fd)) << "parameter " << j;
    }
    // The derivative is not identically zero (the images are misaligned).
    double norm = 0.0;
    for (unsigned int j = 0; j < analyticDerivative.GetSize(); ++j)
      norm += analyticDerivative[j] * analyticDerivative[j];
    EXPECT_GT(norm, 1e-8) << "derivative should be non-zero for misaligned images";
  };

  checkLoss("Cosine"); // d(1 - cosine)/dm with the full dot-product cross term
  checkLoss("NCC");        // cross-point statistics path (closed-form derivative)
  checkLoss("L1");
  checkLoss("L2");
  checkLoss("L1Cosine");
  checkLoss("Dice");
}

// With Seed left at zero the metric asks for run-to-run variation, not for variation between
// two evaluations of the same parameters. The feature subset used to be drawn from a generator
// reseeded with the clock on every evaluation, so crossing a second boundary between two calls
// changed which channels were compared and moved the value; an optimizer's line search then
// compared numbers that did not come from the same function. The subset follows the plane seed,
// drawn once, so the value is a function of the parameters for the life of the metric.
// Each layer starts at 1 at the first evaluation after Initialize(); later evaluations follow the transform (the
// normalisation removed on 2026-08-05 re-latched at every evaluation and stayed at 1); a new Initialize() starts over.
TEST(ImpactMetric, NormalizedValueStartsAtOneAndFollowsTheTransform)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto                                       metric = MetricType::New();
  std::vector<itk::ImpactModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       std::vector<unsigned int>{ 2, 2, 2 },
                       std::vector<bool>{ true, false },
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "L2" });
  metric->SetLayersWeight({ 2.f });
  metric->SetSubsetFeatures({ 4 });
  metric->SetPCA({ 0 });
  metric->SetMode("Static");
  metric->SetSeed(1);
  metric->SetFeaturesMapUpdateInterval(-1);
  metric->SetDevice("cpu");
  auto identity = itk::IdentityTransform<double, 3>::New();
  auto affine = itk::AffineTransform<double, 3>::New();
  affine->SetIdentity();
  metric->SetFixedImage(MakeRampImage(8));
  metric->SetMovingImage(MakeRampImage(8, 1));
  metric->SetFixedTransform(identity);
  metric->SetMovingTransform(affine);
  metric->SetUseFixedImageGradientFilter(false);
  metric->SetUseMovingImageGradientFilter(false);
  metric->SetMaximumNumberOfWorkUnits(1);
  ASSERT_TRUE(metric->GetNormalizeLosses()) << "on by default";

  metric->Initialize();
  const double first = static_cast<double>(metric->GetValue());
  EXPECT_NEAR(first, 2.0, 1e-6) << "the layer starts at 1, times its weight";

  auto parameters = affine->GetParameters();
  parameters[9] += 1.5; // translate along x
  affine->SetParameters(parameters);
  EXPECT_GT(std::abs(static_cast<double>(metric->GetValue()) - first), 1e-3) << "the value follows the transform";

  metric->Initialize();
  EXPECT_NEAR(static_cast<double>(metric->GetValue()), 2.0, 1e-6) << "a new level starts at 1 again";

  metric->NormalizeLossesOff();
  metric->Initialize();
  EXPECT_GT(std::abs(static_cast<double>(metric->GetValue()) - 2.0), 1e-3) << "off, the raw value comes back";
}

TEST(ImpactMetric, SampledPointsRefuseTheLocalNCC)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  auto                                       metric = MetricType::New();
  std::vector<itk::ImpactModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       std::vector<unsigned int>{ 2, 2, 2 },
                       std::vector<bool>{ true, false },
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "LNCC" });
  metric->SetLayersWeight({ 1.f });
  metric->SetSubsetFeatures({ 4 });
  metric->SetPCA({ 0 });
  metric->SetMode("Static");
  metric->SetDevice("cpu");
  auto affine = itk::AffineTransform<double, 3>::New();
  affine->SetIdentity();
  metric->SetFixedImage(MakeRampImage(8));
  metric->SetMovingImage(MakeRampImage(8, 1));
  metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
  metric->SetMovingTransform(affine);
  EXPECT_THROW(metric->Initialize(), itk::ExceptionObject);
}

TEST(ImpactMetric, FeatureSubsetDoesNotFollowTheClock)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeBlobImage(12, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(12, 1.2, -0.8, 0.5, 2.0);

  auto                                       metric = MetricType::New();
  std::vector<itk::ImpactModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       std::vector<unsigned int>{ 2, 2, 2 },
                       std::vector<bool>{ true, false }, // the 4-channel conv layer
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "L2" });
  metric->SetLayersWeight({ 1.f });
  metric->SetSubsetFeatures({ 2 }); // a strict subset: the random draw actually happens
  metric->SetPCA({ 0 });
  metric->SetMode("Static");
  metric->SetSeed(0); // the clock policy is what is under test
  metric->SetDevice("cpu");
  metric->SetFeaturesMapUpdateInterval(-1);
  metric->SetFixedImage(fixed);
  metric->SetMovingImage(moving);
  metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
  metric->SetMovingTransform(itk::IdentityTransform<double, 3>::New());
  metric->SetMaximumNumberOfWorkUnits(1);
  metric->Initialize();

  const double first = static_cast<double>(metric->GetValue());
  // time(nullptr) has one-second resolution: make sure the second call sits in a later second.
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  const double second = static_cast<double>(metric->GetValue());

  EXPECT_DOUBLE_EQ(first, second) << "the feature subset must not change between two evaluations";
}

// --- C2: the derivative is correct for a local-support displacement field -----
TEST(ImpactMetric, StaticDerivativeMatchesFiniteDifferencesDisplacementField)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using DisplacementTransformType = itk::DisplacementFieldTransform<double, 3>;
  using FieldType = DisplacementTransformType::DisplacementFieldType;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1);

  auto                                 metric = MetricType::New();
  std::vector<itk::ImpactModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       std::vector<unsigned int>{ 2, 2, 2 },
                       std::vector<bool>{ true, false },
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "L2" });
  metric->SetLayersWeight({ 1.f });
  metric->SetSubsetFeatures({ 4 });
  metric->SetPCA({ 0 });
  metric->SetMode("Static");
  metric->SetSeed(1);
  metric->SetFeaturesMapUpdateInterval(-1);
  metric->SetDevice("cpu");
  metric->NormalizeLossesOff(); // the raw metric: its gradients are compared with a fixed threshold

  // Zero displacement field over the fixed (virtual) domain.
  auto field = FieldType::New();
  field->SetRegions(fixed->GetLargestPossibleRegion());
  field->SetOrigin(fixed->GetOrigin());
  field->SetSpacing(fixed->GetSpacing());
  field->SetDirection(fixed->GetDirection());
  field->Allocate();
  FieldType::PixelType zeroVector;
  zeroVector.Fill(0.0);
  field->FillBuffer(zeroVector);
  auto displacement = DisplacementTransformType::New();
  displacement->SetDisplacementField(field);

  auto identityFixed = itk::IdentityTransform<double, 3>::New();
  metric->SetFixedImage(fixed);
  metric->SetMovingImage(moving);
  metric->SetFixedTransform(identityFixed);
  metric->SetMovingTransform(displacement);
  metric->SetUseFixedImageGradientFilter(false);
  metric->SetUseMovingImageGradientFilter(false);
  metric->SetMaximumNumberOfWorkUnits(1);
  metric->Initialize();

  MetricType::MeasureType    value;
  MetricType::DerivativeType analyticDerivative;
  metric->GetValueAndDerivative(value, analyticDerivative);
  ASSERT_EQ(analyticDerivative.GetSize(), displacement->GetNumberOfParameters());

  // The field has thousands of parameters; spot-check a few interior voxels.
  // Parameters are laid out [vox0_x, vox0_y, vox0_z, vox1_x, ...].
  const double                          h = 1e-3;
  const MetricType::ParametersType      params0 = displacement->GetParameters();
  const std::vector<unsigned int>       indices = { 3 * 100, 3 * 100 + 1, 3 * 100 + 2, 3 * 200 + 1, 3 * 250 + 2 };
  unsigned int                          nonZeroChecks = 0;
  for (unsigned int j : indices)
  {
    if (j >= displacement->GetNumberOfParameters())
      continue;
    MetricType::ParametersType pPlus = params0;
    pPlus[j] += h;
    displacement->SetParameters(pPlus);
    const double vPlus = static_cast<double>(metric->GetValue());

    MetricType::ParametersType pMinus = params0;
    pMinus[j] -= h;
    displacement->SetParameters(pMinus);
    const double vMinus = static_cast<double>(metric->GetValue());

    displacement->SetParameters(params0);

    const double fd = (vPlus - vMinus) / (2.0 * h);
    // The metric reports the descent direction (-gradient), hence compare to -fd.
    const double analytic = static_cast<double>(analyticDerivative[j]);
    EXPECT_NEAR(analytic, -fd, 1e-2 + 0.05 * std::abs(fd)) << "parameter " << j;
    if (std::abs(fd) > 1e-4)
      ++nonZeroChecks;
  }
  EXPECT_GT(nonZeroChecks, 0u) << "expected some non-zero displacement-field gradients";
}

// --- E: the online ("Jacobian") mode derivative matches finite differences --------
// In Jacobian mode no feature maps are precomputed: every sampled point extracts a small
// intensity patch, runs the model online and backpropagates through it. The analytic
// derivative (autograd feature Jacobian x moving-interpolator gradient x transform
// Jacobian) must match central finite differences of the online value, up to the
// descent-sign convention (the metric reports -gradient, hence compare to -fd).
TEST(ImpactMetric, JacobianDerivativeMatchesFiniteDifferences)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1); // different => non-trivial gradient

  auto checkLoss = [&](const char * lossName) {
    SCOPED_TRACE(std::string("loss ") + lossName);
    SCOPED_TRACE(lossName);
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    // A strictly positive 3^3 patch at the image spacing: the toy conv (kernel 3,
    // padding 1) computes the exact neighborhood feature at the patch center.
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 3, 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ std::string(lossName) });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 4 }); // all 4 conv channels => deterministic (no random subset)
    metric->SetPCA({ 0 });
    metric->SetMode("Jacobian");
    metric->SetSeed(1);
    metric->SetDevice("cpu");

    // A smooth B-spline moving interpolator so its (central-difference) spatial gradient,
    // used by the analytic derivative, is consistent with finite differences of the value.
    auto movingInterp = itk::BSplineInterpolateImageFunction<ImageType, double>::New();
    movingInterp->SetSplineOrder(3);
    metric->SetMovingInterpolator(movingInterp);

    auto identityFixed = itk::IdentityTransform<double, 3>::New();
    auto transform = TransformType::New();
    transform->SetIdentity();

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identityFixed);
    metric->SetMovingTransform(transform);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();

    MetricType::MeasureType    value;
    MetricType::DerivativeType analyticDerivative;
    metric->GetValueAndDerivative(value, analyticDerivative);
    ASSERT_EQ(analyticDerivative.GetSize(), transform->GetNumberOfParameters());
    EXPECT_TRUE(std::isfinite(static_cast<double>(value))) << "online value must be finite";

    const double                     h = 1e-2;
    const MetricType::ParametersType params0 = transform->GetParameters();
    for (unsigned int j = 0; j < transform->GetNumberOfParameters(); ++j)
    {
      MetricType::ParametersType pPlus = params0;
      pPlus[j] += h;
      transform->SetParameters(pPlus);
      const double vPlus = static_cast<double>(metric->GetValue());

      MetricType::ParametersType pMinus = params0;
      pMinus[j] -= h;
      transform->SetParameters(pMinus);
      const double vMinus = static_cast<double>(metric->GetValue());

      transform->SetParameters(params0);

      const double fd = (vPlus - vMinus) / (2.0 * h);
      const double analytic = static_cast<double>(analyticDerivative[j]);
      EXPECT_NEAR(analytic, -fd, 2e-2 + 0.05 * std::abs(fd)) << "parameter " << j;
    }
    double norm = 0.0;
    for (unsigned int j = 0; j < analyticDerivative.GetSize(); ++j)
      norm += analyticDerivative[j] * analyticDerivative[j];
    EXPECT_GT(norm, 1e-8) << "derivative should be non-zero for misaligned images";
  };

  checkLoss("Cosine"); // normalized cross term
  checkLoss("NCC");        // cross-point statistics path
  checkLoss("L1");
  checkLoss("L2");
  checkLoss("L1Cosine");
  checkLoss("Dice");
}

// Online mode runs the model on a batch of points at a time, so what a point contributes must not
// depend on which batch it happened to land in. For a loss that is a mean of per-point terms that
// is arithmetic. For NCC it is not: NCC correlates the sampled points with one another, and it
// holds only because the derivative is accumulated as sums and combined in closed form once every
// point has been seen. Seeding the backward pass with NCC's own gradient instead would measure the
// correlation WITHIN a batch -- which at a batch of one is a zero variance, hence an identically
// zero gradient, silently.
TEST(ImpactMetric, JacobianResultIsIndependentOfBatchSize)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1); // different => non-trivial gradient
  auto identityFixed = itk::IdentityTransform<double, 3>::New();

  auto evaluate = [&](const char * lossName, unsigned int batchSize, MetricType::DerivativeType & derivative) {
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 3, 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ std::string(lossName) });
    metric->SetLayersWeight({ 1.f });
    // All four channels: the feature subset is then drawn without consuming the generator, so a
    // different number of batches cannot shift the random sequence and confound the comparison.
    metric->SetSubsetFeatures({ 4 });
    metric->SetPCA({ 0 });
    metric->SetMode("Jacobian");
    metric->SetSeed(1);
    metric->SetDevice("cpu");
    metric->SetBatchSize(batchSize);

    auto movingInterp = itk::BSplineInterpolateImageFunction<ImageType, double>::New();
    movingInterp->SetSplineOrder(3);
    metric->SetMovingInterpolator(movingInterp);

    auto transform = TransformType::New();
    transform->SetIdentity();

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identityFixed);
    metric->SetMovingTransform(transform);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();

    MetricType::MeasureType value;
    metric->GetValueAndDerivative(value, derivative);
    return static_cast<double>(value);
  };

  // The bars are relative rather than exact: batching changes how the per-point terms are grouped
  // before they are summed, and float addition is not associative. Anything that actually depends
  // on the batch moves far more than this.
  for (const char * lossName : { "L2", "NCC" })
  {
    SCOPED_TRACE(lossName);
    MetricType::DerivativeType perPoint, ragged, single;
    // 7 divides neither the 512 sampled points nor anything else here, so the last batch is
    // short -- the case a batch-boundary mistake shows up in.
    const double perPointValue = evaluate(lossName, 1, perPoint);
    const double raggedValue = evaluate(lossName, 7, ragged);
    const double singleValue = evaluate(lossName, 4096, single);

    EXPECT_NEAR(raggedValue, perPointValue, 1e-6 * std::abs(perPointValue));
    EXPECT_NEAR(singleValue, perPointValue, 1e-6 * std::abs(perPointValue));

    double norm = 0.0;
    for (unsigned int j = 0; j < perPoint.GetSize(); ++j)
    {
      norm += perPoint[j] * perPoint[j];
    }
    ASSERT_GT(norm, 1e-8) << "a derivative of zero would satisfy the comparisons below vacuously";

    for (unsigned int j = 0; j < perPoint.GetSize(); ++j)
    {
      EXPECT_NEAR(ragged[j], perPoint[j], 1e-4 * std::abs(perPoint[j])) << "parameter " << j;
      EXPECT_NEAR(single[j], perPoint[j], 1e-4 * std::abs(perPoint[j])) << "parameter " << j;
    }
  }
}

// LayersMask is what asks for several depths of the same network to be compared at once -- an
// early layer answers for texture, a late one for structure. Each kept layer therefore gets its
// own feature map, its own loss and its own weight, and the value is their weighted sum. The
// metric used to extract every kept layer and then compare only the first, silently: the rest
// were computed, allocated and dropped.
TEST(ImpactMetric, EveryKeptLayerIsCompared)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1); // different => non-trivial gradient
  auto identityFixed = itk::IdentityTransform<double, 3>::New();

  auto evaluate = [&](const char *                 mode,
                      const std::vector<bool> &    layersMask,
                      const std::vector<float> &   layersWeight,
                      MetricType::DerivativeType & derivative) {
    const auto layers = static_cast<size_t>(std::count(layersMask.begin(), layersMask.end(), true));
    auto       metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 3, 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         layersMask,
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance(std::vector<std::string>(layers, "L2"));
    metric->SetLayersWeight(layersWeight);
    // Larger than either layer has channels: Initialize clamps it, and a subset that is the
    // whole set is drawn without touching the generator, so the comparison stays deterministic
    // even though the two layers are not the same width.
    metric->SetSubsetFeatures(std::vector<unsigned int>(layers, 1000));
    metric->SetPCA({ 0 }); // one entry per MODEL, not per layer
    metric->SetMode(mode);
    metric->SetSeed(1);
    metric->SetDevice("cpu");
    metric->NormalizeLossesOff(); // each layer's raw value is compared

    auto movingInterp = itk::BSplineInterpolateImageFunction<ImageType, double>::New();
    movingInterp->SetSplineOrder(3);
    metric->SetMovingInterpolator(movingInterp);

    auto transform = TransformType::New();
    transform->SetIdentity();

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identityFixed);
    metric->SetMovingTransform(transform);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();

    MetricType::MeasureType value;
    metric->GetValueAndDerivative(value, derivative);
    return static_cast<double>(value);
  };

  for (const char * mode : { "Static", "Jacobian" })
  {
    SCOPED_TRACE(mode);
    MetricType::DerivativeType firstOnly, secondOnly, both;
    const double               firstValue = evaluate(mode, { true, false }, { 1.f }, firstOnly);
    const double               secondValue = evaluate(mode, { false, true }, { 1.f }, secondOnly);
    const double               bothValue = evaluate(mode, { true, true }, { 1.f, 1.f }, both);

    // The toy model's layer 1 is a passthrough and layer 0 a convolution, so the two disagree.
    // Without this the sum below would hold just as well if the second layer were the first.
    EXPECT_GT(std::abs(secondValue - firstValue), 1e-6 * std::abs(firstValue)) << "the two layers must differ";
    EXPECT_NEAR(bothValue, firstValue + secondValue, 1e-5 * std::abs(firstValue + secondValue));

    ASSERT_EQ(both.GetSize(), firstOnly.GetSize());
    for (unsigned int j = 0; j < both.GetSize(); ++j)
    {
      const double expected = firstOnly[j] + secondOnly[j];
      EXPECT_NEAR(both[j], expected, 1e-4 * std::abs(expected) + 1e-9) << "parameter " << j;
    }
  }
}

// Every per-layer vector is indexed by the loop over compared layers with no bound check in the
// hot path, so a size that does not match has to be refused up front: too short reads past the
// end, too long sums losses that no point ever updated -- over a value that was never even
// initialized. Too long is exactly what a configuration written for the elastix component looks
// like, where these vectors are already per kept layer.
TEST(ImpactMetric, PerLayerVectorsMustMatchTheNumberOfComparedLayers)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1);

  // One model keeping both layers, so two layers are compared: two distances, two weights, two
  // subsets -- and one PCA, which belongs to the model.
  auto build = [&](size_t distances, size_t weights, size_t subsets, size_t pca) {
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 3, 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, true },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance(std::vector<std::string>(distances, "L2"));
    metric->SetLayersWeight(std::vector<float>(weights, 1.f));
    metric->SetSubsetFeatures(std::vector<unsigned int>(subsets, 1000));
    metric->SetPCA(std::vector<unsigned int>(pca, 0));
    metric->SetMode("Static");
    metric->SetDevice("cpu");

    auto transform = TransformType::New();
    transform->SetIdentity();
    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
    metric->SetMovingTransform(transform);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    return metric;
  };

  EXPECT_NO_THROW(build(2, 2, 2, 1)->Initialize());
  EXPECT_THROW(build(1, 2, 2, 1)->Initialize(), itk::ExceptionObject) << "one distance for two layers";
  EXPECT_THROW(build(2, 1, 2, 1)->Initialize(), itk::ExceptionObject) << "one weight for two layers";
  EXPECT_THROW(build(2, 2, 1, 1)->Initialize(), itk::ExceptionObject) << "one subset for two layers";
  EXPECT_THROW(build(3, 3, 3, 1)->Initialize(), itk::ExceptionObject) << "three entries for two layers";
  EXPECT_THROW(build(2, 2, 2, 2)->Initialize(), itk::ExceptionObject) << "PCA is per model, not per layer";
}


// A 2D model on a volume samples a plane through each point, and that plane is drawn at random so
// the metric integrates over orientations instead of always showing the network the same one.
// Random, but a function of where the point is -- so the value is reproducible, which is what
// makes the analytic derivative comparable to a finite difference of it at all.
TEST(ImpactMetric, TwoDimensionalModelJacobianDerivativeMatchesFiniteDifferences)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  // Blobs rather than ramps: a rotated patch reaches diagonally, so its corners can leave the
  // buffer, where a sample is clamped to zero. A ramp is at its largest exactly at the border, so
  // the clamp is a large jump; a blob has decayed to nothing by then and the jump with it.
  auto fixed = MakeBlobImage(12, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(12, 1.0, -0.7, 0.6, 2.0);

  auto identityFixed = itk::IdentityTransform<double, 3>::New();
  auto transform = TransformType::New();
  transform->SetIdentity();

  auto makeMetric = [&](unsigned int workUnits, unsigned int seed) {
    auto                                 metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    // A 2D patch on a 3D image. The voxel size carries an entry per image axis because
    // ImpactModelConfiguration pairs the two when it precomputes its patch offsets; Jacobian mode
    // reads only the two in-plane steps.
    configs.emplace_back(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                         2,
                         1,
                         std::vector<unsigned int>{ 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ "L2" });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 4 }); // all 4 conv channels => no random subset on top
    metric->SetPCA({ 0 });
    metric->SetMode("Jacobian");
    metric->SetSeed(seed);
    metric->SetDevice("cpu");
    metric->NormalizeLossesOff(); // raw values are compared across seeds

    auto movingInterp = itk::BSplineInterpolateImageFunction<ImageType, double>::New();
    movingInterp->SetSplineOrder(3);
    metric->SetMovingInterpolator(movingInterp);

    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(identityFixed);
    metric->SetMovingTransform(transform);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetMaximumNumberOfWorkUnits(workUnits);

    // Sample the interior only. A rotated 3x3 patch reaches sqrt(2) voxels diagonally, so
    // sampling up to the border would put some of its corners outside the moving buffer, where a
    // sample is clamped to zero. Which corners are out then changes with the transform, making
    // the metric genuinely discontinuous there -- real, but not what this test is measuring.
    MetricType::VirtualRegionType virtualRegion;
    virtualRegion.SetSize({ { 8, 8, 8 } });
    MetricType::VirtualPointType virtualOrigin;
    virtualOrigin.Fill(2.0);
    metric->SetVirtualDomain(fixed->GetSpacing(), virtualOrigin, fixed->GetDirection(), virtualRegion);
    metric->Initialize();
    return metric;
  };

  auto                       metric = makeMetric(1, 1);
  MetricType::MeasureType    value;
  MetricType::DerivativeType analyticDerivative;
  metric->GetValueAndDerivative(value, analyticDerivative);
  ASSERT_EQ(analyticDerivative.GetSize(), transform->GetNumberOfParameters());
  EXPECT_TRUE(std::isfinite(static_cast<double>(value)));
  EXPECT_EQ(metric->GetNumberOfValidPoints(), 8u * 8u * 8u);

  const double                     h = 1e-3;
  const MetricType::ParametersType params0 = transform->GetParameters();
  for (unsigned int j = 0; j < transform->GetNumberOfParameters(); ++j)
  {
    MetricType::ParametersType pPlus = params0;
    pPlus[j] += h;
    transform->SetParameters(pPlus);
    const double vPlus = static_cast<double>(metric->GetValue());

    MetricType::ParametersType pMinus = params0;
    pMinus[j] -= h;
    transform->SetParameters(pMinus);
    const double vMinus = static_cast<double>(metric->GetValue());

    transform->SetParameters(params0);

    const double fd = (vPlus - vMinus) / (2.0 * h);
    // Purely relative: an absolute floor would make the comparison vacuous here, where the
    // derivative of a blob pair is three orders of magnitude smaller than the ramp's.
    EXPECT_GT(std::abs(fd), 1e-5) << "parameter " << j << ": nothing to compare against";
    EXPECT_NEAR(static_cast<double>(analyticDerivative[j]), -fd, 0.02 * std::abs(fd)) << "parameter " << j;
  }

  // The plane a point gets is a function of that point, not of how many points came before it,
  // so splitting the domain across work units must not change the value. Drawing from a running
  // per-work-unit generator instead would give each unit its own sequence and move the value.
  //
  // The bar is a tight relative tolerance rather than exact equality. Each work unit accumulates
  // its own partial sum and those are reduced afterwards, so the order of the additions follows
  // the partition and the last bits of the total move with it: on arm64 the two totals
  // here differ by 2e-9 relative. A plane that actually changed would move the value by orders of
  // magnitude more, which is what this is here to catch.
  const double single = static_cast<double>(makeMetric(4, 1)->GetValue());
  EXPECT_NEAR(single, static_cast<double>(value), 1e-6 * std::abs(single))
    << "the sampled planes must not depend on the work-unit partition";

  // This is the assertion that carries the feature. Every layer is kept and the feature subset
  // is the whole channel set, so the seed reaches nothing but the plane: were the plane left
  // axis-aligned, the seed would be dead and these two values would be bit-identical. They must
  // still land close together, since both integrate the same images over orientations.
  const double otherSeed = static_cast<double>(makeMetric(1, 7)->GetValue());
  EXPECT_NE(otherSeed, static_cast<double>(value)) << "a different seed must draw different planes";
  EXPECT_NEAR(otherSeed, static_cast<double>(value), 0.05 * std::abs(static_cast<double>(value)))
    << "different planes over the same images must still agree on roughly the same value";
}

// A metadata-aware model (Forward with four arguments) takes the image's intensity range, mean
// and sigma from the caller and normalizes with them; without them it falls back to its input's
// own range, as ImpactLoss builds its models. The toy below puts the sigma it received into its
// second layer, and zero when it received none, so an L2 distance between two images read through
// that layer is (sigmaF - sigmaM)^2 exactly when each side was handed its own stats: 0 when
// neither was, and one image's sigma^2 when only one was. The Static path stores the stats in
// the configuration through ImageToFeaturesMap; the online path did not hand any over, and a
// model that normalizes by its stats then saw every patch scaled by its own range.
TEST(ImpactMetric, OnlineModeHandsTheModelEachImagesMetadata)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeBlobImage(12, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(12, 1.2, -0.8, 0.5, 3.0); // a wider blob, so the two sigmas differ

  // The sigma each side's stats carry, from the filter ComputeImageMetadata runs, as float32.
  auto sigmaOf = [](const ImageType::Pointer & image) {
    auto statistics = itk::StatisticsImageFilter<ImageType>::New();
    statistics->SetInput(image);
    statistics->Update();
    return static_cast<double>(static_cast<float>(statistics->GetSigma()));
  };
  const double sigmaFixed = sigmaOf(fixed);
  const double sigmaMoving = sigmaOf(moving);
  const double expected = (sigmaFixed - sigmaMoving) * (sigmaFixed - sigmaMoving);
  // Each wrong wiring reads a value of its own, so the assertion below tells them apart.
  ASSERT_GT(expected, 1e-6);
  ASSERT_GT(std::abs(expected - sigmaFixed * sigmaFixed), 1e-6);
  ASSERT_GT(std::abs(expected - sigmaMoving * sigmaMoving), 1e-6);

  auto                                       metric = MetricType::New();
  std::vector<itk::ImpactModelConfiguration> configs;
  configs.emplace_back(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModelMetadata.pt",
                       3,
                       1,
                       std::vector<unsigned int>{ 3, 3, 3 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<bool>{ false, true }, // the sigma layer only
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "L2" });
  metric->SetLayersWeight({ 1.f });
  metric->SetSubsetFeatures({ 1 });
  metric->SetPCA({ 0 });
  metric->SetMode("Jacobian");
  metric->SetSeed(1);
  metric->SetDevice("cpu");
  metric->NormalizeLossesOff(); // the raw value is compared with a closed form
  metric->SetFixedImage(fixed);
  metric->SetMovingImage(moving);
  metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
  metric->SetMovingTransform(itk::TranslationTransform<double, 3>::New());
  metric->SetMaximumNumberOfWorkUnits(1);
  metric->Initialize();

  const double tolerance = 1e-4 * expected;
  EXPECT_NEAR(static_cast<double>(metric->GetValue()), expected, tolerance)
    << "sigma " << sigmaFixed << " on the fixed side and " << sigmaMoving << " on the moving side";

  // The derivative runs the moving model through autograd, a call of its own.
  MetricType::MeasureType    value;
  MetricType::DerivativeType derivative;
  metric->GetValueAndDerivative(value, derivative);
  EXPECT_NEAR(static_cast<double>(value), expected, tolerance);
  EXPECT_EQ(derivative.two_norm(), 0.0) << "a layer read off the stats does not move with the patch";
}

// Only one voxel of a patch ever reaches the loss: centerFeatures() reads index size/2 of the
// model output. The patch must therefore be spanned around that same voxel, so a passthrough
// layer returns the intensity AT the sample point whatever the patch extent. Centring on the
// geometric middle, (n - 1) / 2, agrees for an odd extent and lands half a voxel away for an
// even one, which handed the loss a feature taken beside its own point.
TEST(ImpactMetric, AnEvenPatchIsCentredOnTheVoxelThatIsReadBack)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeBlobImage(16, 0.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(16, 1.3, -0.7, 0.4, 2.0);

  // The toy's second layer is the intensity passthrough, so the compared feature is the image
  // value at whatever point the patch centre resolves to.
  auto valueForPatchExtent = [&](unsigned int extent) {
    auto                                       metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ extent, extent, extent },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ false, true }, // the passthrough layer only
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ "L2" });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 2 });
    metric->SetPCA({ 0 });
    metric->SetMode("Jacobian");
    metric->SetSeed(1);
    metric->SetDevice("cpu");
    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
    auto transform = itk::TranslationTransform<double, 3>::New();
    transform->SetIdentity();
    metric->SetMovingTransform(transform);
    metric->SetMaximumNumberOfWorkUnits(1);
    metric->Initialize();
    return static_cast<double>(metric->GetValue());
  };

  // A single-voxel patch IS the sample point, so it pins what every other extent must return.
  const double atOneVoxel = valueForPatchExtent(1);
  ASSERT_GT(atOneVoxel, 1e-6) << "the two blobs must differ for this to test anything";
  EXPECT_NEAR(valueForPatchExtent(3), atOneVoxel, 1e-9 * atOneVoxel) << "odd extent";
  EXPECT_NEAR(valueForPatchExtent(4), atOneVoxel, 1e-9 * atOneVoxel) << "even extent";
  EXPECT_NEAR(valueForPatchExtent(5), atOneVoxel, 1e-9 * atOneVoxel) << "odd extent";
}

// --- End-to-end: a real ITK optimizer reduces the metric and recovers a shift -
// Every distance must drive the optimizer downhill, not just the one the test happened to pick.
// L1 and L2 used to be rescaled by the inverse of their own first value, which made their reported
// energy exactly 1.0 at every transform: this test passed on NCC and would have failed on L1.
TEST(ImpactMetric, TranslationRegistrationConvergesForEveryDistance)
{
  for (const std::string distance : { "NCC", "L1", "L2", "Cosine", "L1Cosine" })
  {
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;
  using TransformType = itk::TranslationTransform<double, 3>;

  // Two sharp blobs straddling the image center, offset by +4 voxels in x (fixed at
  // 6, moving at 10 in a size-16 volume). Both blobs sit well inside the image so the
  // feature cross-correlation is not biased by boundary truncation. The optimal moving
  // translation that brings the moving blob onto the fixed one is exactly (+4, 0, 0).
  auto fixed = MakeBlobImage(16, -2.0, 0.0, 0.0, 2.0);
  auto moving = MakeBlobImage(16, 2.0, 0.0, 0.0, 2.0);

    auto                                       metric = MetricType::New();
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 2, 2, 2 },
                         std::vector<bool>{ true, false },
                         false);
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ distance });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 4 });
    metric->SetPCA({ 0 });
    metric->SetMode("Static");
    metric->SetSeed(1);
    metric->SetFeaturesMapUpdateInterval(-1);
    metric->SetDevice("cpu");

  auto identityFixed = itk::IdentityTransform<double, 3>::New();
  auto transform = TransformType::New();
  transform->SetIdentity();

  metric->SetFixedImage(fixed);
  metric->SetMovingImage(moving);
  metric->SetFixedTransform(identityFixed);
  metric->SetMovingTransform(transform);
  metric->SetUseFixedImageGradientFilter(false);
  metric->SetUseMovingImageGradientFilter(false);
  metric->SetMaximumNumberOfWorkUnits(1);
  metric->Initialize();

  const double initialValue = static_cast<double>(metric->GetValue());

  // Drive a real ITK optimizer with the metric.
  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  using ScalesEstimatorType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
  auto scales = ScalesEstimatorType::New();
  scales->SetMetric(metric);
  auto optimizer = OptimizerType::New();
  optimizer->SetMetric(metric);
  optimizer->SetScalesEstimator(scales);
  optimizer->SetNumberOfIterations(300);
  optimizer->SetLearningRate(2.0);
  optimizer->SetMinimumStepLength(1e-4);
  optimizer->SetRelaxationFactor(0.8);
  optimizer->StartOptimization();

  // The metric, through its analytic derivative, must drive the optimizer downhill.
  const double finalValue = static_cast<double>(metric->GetValue());
  EXPECT_LT(finalValue, initialValue) << distance << ": optimization should reduce the metric";

  // The optimizer recovers the true geometric alignment: bringing the moving blob
  // (x=10) onto the fixed one (x=6) is a +4 voxel translation in x, with no y/z motion.
  const TransformType::ParametersType recovered = transform->GetParameters();
  EXPECT_NEAR(recovered[0], 4.0, 0.5) << distance << ": recovered x-translation should reach +4";
  EXPECT_LT(std::abs(recovered[1]), 0.2) << distance << ": motion should stay on the x axis";
  EXPECT_LT(std::abs(recovered[2]), 0.2) << distance << ": motion should stay on the x axis";
  }
}

// --- Regression: extracting a later layer must not corrupt an earlier one --------
// A shared TensorToImageFilter whose output was grafted into each GetOutput(i) used
// to let layer 1's Update() overwrite the buffer already grafted for layer 0, halving
// the spatial content of every layer but the last. The layer-0 feature map must be
// identical whether or not layer 1 is also extracted.
TEST(ImpactBackend, MultiLayerFeatureMapsAreIndependent)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  auto img = MakeBlobImage(16, 2.0, 0.0, 0.0, 1.0);
  auto build = [&](std::vector<bool> mask) {
    auto interp = InterpolatorType::New();
    interp->SetSplineOrder(3);
    itk::ImpactModelConfiguration cfg(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 2, 2, 2 }, mask, false);
    auto f = FeatMapType::New();
    f->SetModelConfiguration(cfg);
    f->SetInterpolator(interp);
    f->AddInput(img);
    f->SetPCA(0);
    f->SetDevice("cpu");
    f->Update();
    return f;
  };
  auto convOnly = build({ true, false }); // only layer 0 extracted
  auto both = build({ true, true });      // layers 0 and 1 extracted

  auto map0a = convOnly->GetOutput(0);
  auto map0b = both->GetOutput(0);
  ASSERT_EQ(map0a->GetLargestPossibleRegion().GetSize(), map0b->GetLargestPossibleRegion().GetSize());
  ASSERT_EQ(map0a->GetNumberOfComponentsPerPixel(), map0b->GetNumberOfComponentsPerPixel());

  itk::ImageRegionConstIteratorWithIndex<std::remove_reference_t<decltype(*map0a)>> it(
    map0a, map0a->GetLargestPossibleRegion());
  double maxDiff = 0.0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto va = it.Get();
    const auto vb = map0b->GetPixel(it.GetIndex());
    for (unsigned int c = 0; c < va.GetSize(); ++c)
      maxDiff = std::max(maxDiff, std::abs(static_cast<double>(va[c]) - static_cast<double>(vb[c])));
  }
  EXPECT_LT(maxDiff, 1e-5) << "layer-0 feature map must not depend on whether layer 1 is also extracted";
}

// --- Tiling: a blended map must reconstruct the input, whatever the overlap -------
// The toy model's layer 1 is a passthrough, so with a voxel size equal to the input's the
// assembled map is the input itself -- exactly, for every window, because each one is
// normalised into a partition of unity over the patches covering a voxel. Anything that shifts
// the tiling (a padding collar, a mis-cropped border, a stride that does not match the patch
// count) breaks the identity, and so does dropping the direction cosines: the sampler then
// walks off the volume and returns zeros.
//
// The image deliberately has a non-identity direction and a size that is not a multiple of the
// patch on any axis; on an axis-aligned volume of a multiple size, both defects are invisible.
TEST(ImpactBackend, TiledFeatureMapReconstructsInputForEveryWindow)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const ImageType::SizeType size = { { 20, 17, 13 } }; // no axis is a multiple of the patch
  auto                      img = MakeRampImage(size);
  ImageType::DirectionType  direction;
  const double              angle = 30.0 * itk::Math::pi / 180.0;
  direction.SetIdentity();
  direction[0][0] = std::cos(angle);
  direction[0][1] = -std::sin(angle);
  direction[1][0] = std::sin(angle);
  direction[1][1] = std::cos(angle);
  img->SetDirection(direction);
  ImageType::PointType origin;
  origin[0] = -3.5;
  origin[1] = 7.25;
  origin[2] = 1.0;
  img->SetOrigin(origin);

  for (const std::string window : { "mean", "cosinus", "gaussian", "trim" })
  {
    for (const unsigned int overlap : { 0u, 3u, 5u })
    {
      itk::ImpactModelConfiguration cfg(
        ToyModelPath(), 3, 1, { 8, 8, 8 }, { 1.f, 1.f, 1.f }, { overlap, overlap, overlap }, { true, true }, false);
      cfg.SetPatchCombine(window);
      auto interp = InterpolatorType::New();
      interp->SetSplineOrder(3);
      auto f = FeatMapType::New();
      f->SetModelConfiguration(cfg);
      f->SetInterpolator(interp);
      f->AddInput(img);
      f->SetPCA(0);
      f->SetDevice("cpu");
      ASSERT_NO_THROW(f->Update()) << window << " / overlap " << overlap;

      auto map = f->GetOutput(1); // passthrough
      const std::string where = window + " / overlap " + std::to_string(overlap);
      ASSERT_EQ(map->GetLargestPossibleRegion().GetSize(), size) << where << ": map must lie on the input grid";
      for (unsigned int d = 0; d < 3; ++d)
      {
        EXPECT_NEAR(map->GetSpacing()[d], 1.0, 1e-6) << where;
        EXPECT_NEAR(map->GetOrigin()[d], origin[d], 1e-6) << where;
      }
      EXPECT_EQ(map->GetDirection(), direction) << where;

      itk::ImageRegionConstIteratorWithIndex<ImageType> it(img, img->GetLargestPossibleRegion());
      double                                            maxDiff = 0.0;
      for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
        maxDiff = std::max(maxDiff, std::abs(static_cast<double>(map->GetPixel(it.GetIndex())[0]) - it.Get()));
      }
      EXPECT_LT(maxDiff, 1e-3) << where << ": the blend must reconstruct the input exactly";
    }
  }
}

// The overlap is settable per axis; an anisotropic one must reconstruct just as exactly, and
// a value at least as large as the patch is a configuration error rather than a silent stall.
TEST(ImpactBackend, PerAxisOverlapBlendsAndRejectsAnOverlapAsLargeAsThePatch)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const ImageType::SizeType size = { { 20, 17, 13 } };
  auto                      img = MakeRampImage(size);

  auto run = [&](const std::vector<unsigned int> & overlaps) {
    itk::ImpactModelConfiguration cfg(
      ToyModelPath(), 3, 1, { 8, 8, 8 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false);
    cfg.SetOverlaps(overlaps);
    EXPECT_EQ(cfg.GetOverlaps().size(), 3u);
    auto interp = InterpolatorType::New();
    interp->SetSplineOrder(3);
    auto f = FeatMapType::New();
    f->SetModelConfiguration(cfg);
    f->SetInterpolator(interp);
    f->AddInput(img);
    f->SetPCA(0);
    f->SetDevice("cpu");
    f->Update();
    return f;
  };

  auto              f = run({ 6, 0, 3 });
  auto              map = f->GetOutput(1);
  itk::ImageRegionConstIteratorWithIndex<ImageType> it(img, img->GetLargestPossibleRegion());
  double            maxDiff = 0.0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    maxDiff = std::max(maxDiff, std::abs(static_cast<double>(map->GetPixel(it.GetIndex())[0]) - it.Get()));
  }
  EXPECT_LT(maxDiff, 1e-3) << "an anisotropic overlap must blend as exactly as a uniform one";

  // An overlap of a whole patch would leave the stride at zero, so the patches never advance.
  EXPECT_THROW(run({ 8, 8, 8 }), itk::ExceptionObject);
}

// An anisotropic patch is declared in ITK axis order, like the voxel size beside it: entry 0 is
// x. The reconstruction test above cannot see this -- every window reconstructs a passthrough
// whatever the layout -- so it is pinned here on the convolution layer, where the patch seams
// are visible.
TEST(ImpactBackend, AnisotropicPatchIsTiledInItkAxisOrder)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const ImageType::SizeType size = { { 20, 17, 13 } };
  auto                      img = MakeRampImage(size);

  auto run = [&](const std::vector<unsigned int> & patch) {
    itk::ImpactModelConfiguration cfg(
      ToyModelPath(), 3, 1, patch, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false);
    auto                    interp = InterpolatorType::New();
    interp->SetSplineOrder(3);
    auto f = FeatMapType::New();
    f->SetModelConfiguration(cfg);
    f->SetInterpolator(interp);
    f->AddInput(img);
    f->SetPCA(0);
    f->SetDevice("cpu");
    f->Update();
    return f;
  };

  // The 8 is patch entry 0, so it tiles x (extent 20): patches at 0, 8 and 16. y takes its full
  // 17 and z its full 13, one patch each. The whole-image run is the seam-free reference.
  auto tiled = run({ 8, 17, 13 });
  auto whole = run({ 0, 0, 0 });
  auto tiledMap = tiled->GetOutput(0);
  auto wholeMap = whole->GetOutput(0);
  ASSERT_EQ(tiledMap->GetLargestPossibleRegion().GetSize(), size);

  // Single voxels, not planes: every x-plane crosses the z seam, so only a voxel interior to
  // its patch on every axis isolates the axis under test.
  auto diffAt = [&](itk::IndexValueType x, itk::IndexValueType y, itk::IndexValueType z) {
    const ImageType::IndexType index = { { x, y, z } };
    const auto                 a = tiledMap->GetPixel(index);
    const auto                 b = wholeMap->GetPixel(index);
    double                     worst = 0.0;
    for (unsigned int c = 0; c < a.GetSize(); ++c)
    {
      worst = std::max(worst, std::abs(static_cast<double>(a[c]) - static_cast<double>(b[c])));
    }
    return worst;
  };

  // x's second patch starts at 8, so the convolution there sees its zero padding.
  EXPECT_GT(diffAt(8, 8, 8), 1.0) << "patch entry 0 tiles x: a seam is expected at x = 8";
  // x = 4 is interior to the first patch, and y and z hold a single patch each. Were the patch
  // consumed in the tensor's order the 8 would tile z instead, and this voxel would be clean
  // while z = 8 carried the seam.
  EXPECT_LT(diffAt(4, 8, 8), 1e-3) << "x = 4 is interior to its patch: no seam may appear";
}

// Online mode runs its samples through the model in batches bounded by the device budget
// (itkImpactBatchBudget.h); nothing may depend on where the batches are cut.
namespace
{
struct OnlineFixture
{
  std::vector<itk::ImpactModelConfiguration>                configs;
  std::vector<itk::Point<double, 3>>                        points;
  std::vector<std::vector<std::vector<std::vector<float>>>> patchIndex{ 1 };
  std::vector<torch::Tensor>                                subset{ torch::arange(4, torch::kInt64),
                                                                    torch::arange(1, torch::kInt64) };
  torch::Device                                             device{ torch::kCPU };

  explicit OnlineFixture(size_t samples)
  {
    configs.emplace_back(ToyModelPath(), 3, 1, std::vector<unsigned int>{ 3, 3, 3 }, std::vector<float>{ 1.f, 1.f, 1.f },
                         std::vector<unsigned int>{ 0, 0, 0 }, std::vector<bool>{ true, true }, false);
    itk::Impact::GetModelOutputsExample(configs, "fixed", device);
    for (size_t s = 0; s < samples; ++s)
    {
      points.emplace_back(itk::MakePoint(static_cast<double>(s), 0.0, 0.0)); // the sample rides on x
      patchIndex[0].push_back(configs[0].GetPatchIndex());
    }
  }
  // Intensities that depend on the sample, so batches cannot be confused by accident.
  static torch::Tensor
  patch(const itk::Point<double, 3> & point, const std::vector<int64_t> & shape, double offset)
  {
    torch::Tensor values = torch::zeros({ torch::IntArrayRef(shape) }, torch::kFloat32);
    return (values + torch::arange(values.numel(), torch::kFloat32).reshape(values.sizes()) * 0.1f +
            static_cast<float>(point[0] + offset))
      .sin()
      .unsqueeze(0);
  }
};
} // namespace

TEST(ImpactBackend, OnlineInferenceIsIndependentOfTheBatchBound)
{
  constexpr size_t samples = 7;
  OnlineFixture    fx(samples);
  auto             values = [](const itk::Point<double, 3> & point,
                   const std::vector<std::vector<float>> &,
                   const std::vector<int64_t> & shape) { return OnlineFixture::patch(point, shape, 0.0); };
  auto             valuesAndJacobians = [](const itk::Point<double, 3> &           point,
                               torch::Tensor &                         jacobians,
                               const std::vector<std::vector<float>> & index,
                               const std::vector<int64_t> &            shape,
                               int                                     s) {
    jacobians[s] = torch::arange(static_cast<int64_t>(index.size()), torch::kFloat32).unsqueeze(1).repeat({ 1, 3 }) *
                     0.01f * static_cast<float>(s + 1) +
                   0.5f;
    return OnlineFixture::patch(point, shape, 1.0);
  };
  auto run = [&](int64_t bound) {
    itk::SetBatchSize(fx.configs[0], bound);
    std::vector<torch::Tensor> fixedOutputs = itk::Impact::GenerateOutputs<itk::Point<double, 3>>(
      fx.configs, fx.points, fx.patchIndex, fx.subset, fx.device, values);
    std::vector<std::unique_ptr<itk::Impact::Loss>> losses;
    for (size_t i = 0; i < fixedOutputs.size(); ++i)
    {
      losses.push_back(itk::Impact::LossFactory::Instance().Create("L2"));
      losses.back()->SetNumberOfParameters(1);
    }
    std::vector<torch::Tensor> jacobians = itk::Impact::GenerateOutputsAndJacobian<itk::Point<double, 3>>(
      fx.configs, fx.points, fx.patchIndex, fx.subset, fixedOutputs, fx.device, losses, valuesAndJacobians);
    std::vector<double> lossValues;
    for (const auto & loss : losses)
    {
      lossValues.push_back(loss->GetValue(samples));
    }
    return std::make_tuple(fixedOutputs, jacobians, lossValues);
  };

  const auto [outputsAll, jacobiansAll, valuesAll] = run(0);
  ASSERT_EQ(outputsAll.size(), 2u);
  ASSERT_EQ(outputsAll[0].sizes(), (std::vector<int64_t>{ samples, 4 }));
  EXPECT_GT(valuesAll[0], 0.0);
  for (const int64_t bound : { 3, 1, 100 })
  {
    const auto [outputs, jacobians, lossValues] = run(bound);
    for (size_t k = 0; k < 2; ++k)
    {
      EXPECT_TRUE(torch::allclose(outputs[k], outputsAll[k], 1e-6, 1e-6)) << "layer " << k << " bound " << bound;
      EXPECT_TRUE(torch::allclose(jacobians[k], jacobiansAll[k], 1e-5, 1e-6)) << "layer " << k << " bound " << bound;
      EXPECT_NEAR(lossValues[k], valuesAll[k], 1e-6 * std::abs(valuesAll[k]) + 1e-9) << "layer " << k;
    }
  }
}

// The budget is measured on CUDA and left unbounded on the CPU; a batch the device cannot
// allocate is halved and replayed after the rollback, whether the failure is the allocator's
// own c10::OutOfMemoryError or the std::runtime_error the TorchScript interpreter wraps it in.
TEST(ImpactBackend, BatchBudgetIsMeasuredAndABatchOutOfMemoryIsHalved)
{
  OnlineFixture fx(1);
  EXPECT_EQ(itk::Impact::MeasureBatchBudget(fx.configs[0], torch::Device(torch::kCPU)), 0);
  itk::Impact::ConfigureBatchSize(fx.configs, torch::Device(torch::kCPU), 5);
  EXPECT_EQ(itk::GetBatchSize(fx.configs[0]), 5);
  if (torch::cuda::is_available() && ITK_IMPACT_HAS_CUDA_MEMORY_STATS)
  {
    EXPECT_GT(itk::Impact::MeasureBatchBudget(fx.configs[0], torch::Device(torch::kCUDA, 0)), 1);
    itk::ModelTo(fx.configs[0], torch::Device(torch::kCPU));
  }

  std::vector<std::pair<int64_t, int64_t>> ranges;
  int                                      rollbacks = 0;
  itk::SetBatchSize(fx.configs[0], 16);
  itk::Impact::ForEachBatch(
    fx.configs[0],
    fx.device,
    16,
    [&](int64_t begin, int64_t end) {
      if (end - begin > 8)
      {
        TORCH_CHECK_WITH(OutOfMemoryError, false, "simulated");
      }
      if (end - begin > 4)
      {
        throw std::runtime_error("The following operation failed in the TorchScript interpreter.\n"
                                 "RuntimeError: CUDA out of memory. Tried to allocate 432.00 MiB.");
      }
      ranges.emplace_back(begin, end);
    },
    [&] { ++rollbacks; });
  EXPECT_EQ(rollbacks, 2) << "16 -> 8 -> 4";
  EXPECT_EQ(itk::GetBatchSize(fx.configs[0]), 4);
  ASSERT_EQ(ranges.size(), 4u);
  EXPECT_EQ(ranges.back().second, 16);
  for (size_t i = 1; i < ranges.size(); ++i)
  {
    EXPECT_EQ(ranges[i].first, ranges[i - 1].second);
  }

  // Anything else, and a batch of one that still fails, must surface untouched.
  auto other = [](int64_t, int64_t) { throw std::runtime_error("shape mismatch"); };
  EXPECT_THROW(itk::Impact::ForEachBatch(fx.configs[0], fx.device, 4, other, [] {}), std::runtime_error);
  EXPECT_EQ(itk::GetBatchSize(fx.configs[0]), 4);
  auto oom = [](int64_t, int64_t) { TORCH_CHECK_WITH(OutOfMemoryError, false, "simulated"); };
  EXPECT_THROW(itk::Impact::ForEachBatch(fx.configs[0], fx.device, 1, oom, [] {}), c10::OutOfMemoryError);
}

// A whole-image (0) axis that does not fit on the device is halved, largest first, until the
// model fits; declared axes are never touched (itkImpactPatchTiling.h RunTiledModel).
TEST(ImpactBackend, WholeImagePatchShrinksLargestZeroAxisFirst)
{
  const std::vector<int64_t> configured{ 0, 64, 0 };
  std::vector<int64_t>       current{ 300, 64, 212 };
  ASSERT_TRUE(itk::Impact::ShrinkWholeImagePatch(configured, current));
  EXPECT_EQ(current, (std::vector<int64_t>{ 150, 64, 212 }));
  ASSERT_TRUE(itk::Impact::ShrinkWholeImagePatch(configured, current));
  EXPECT_EQ(current, (std::vector<int64_t>{ 150, 64, 106 }));
  while (itk::Impact::ShrinkWholeImagePatch(configured, current))
  {
  }
  EXPECT_EQ(current, (std::vector<int64_t>{ 1, 64, 1 })) << "declared axes stay; zero axes stop at one voxel";
  std::vector<int64_t> declared{ 32, 32, 32 };
  EXPECT_FALSE(itk::Impact::ShrinkWholeImagePatch(declared, declared)) << "nothing to shrink without a zero axis";
}

// A model of lower dimension than the image is swept over the axes it does not span: a 2D model
// on a volume is run slice by slice, along the last ITK axis. The passthrough layer must
// therefore reconstruct the volume exactly -- every slice is blended into a partition of unity
// of its own -- and the swept axis must keep its extent and its spacing.
// The online path exists only for the elastix metric, which samples a patch into a flat buffer
// in GetPatchIndex() order -- ITK axis 0 fastest -- and hands it back as a tensor. A row-major
// tensor over that buffer therefore has ITK x as its LAST, fastest axis, so its shape is the
// patch size reversed. Give it the un-reversed shape and every axis but the middle one is
// misread: for [3, 5, 7] the model is not handed a transposed patch but a re-partitioned one.
// Isotropic patches hide this completely, which is why it needs a test of its own.
TEST(ImpactBackend, OnlineInferenceShapesAnAnisotropicPatchForTheModel)
{
  const std::vector<unsigned int> patch = { 3, 5, 7 }; // x, y, z -- no two axes equal
  const int64_t                   px = 3, py = 5, pz = 7;
  itk::ImpactModelConfiguration   config(ToyModelPath(),
                                       3,
                                       1,
                                       patch,
                                       std::vector<float>{ 1.f, 1.f, 1.f },
                                       std::vector<unsigned int>{ 0, 0, 0 },
                                       std::vector<bool>{ true, false }, // keep the conv: it is not symmetric
                                       false);

  const torch::Device                  device(torch::kCPU);
  std::vector<itk::ImpactModelConfiguration> configs{ config };
  // Probes the model and records where the center of each output layer is; GenerateOutputs
  // needs it, and it is the same call the metric makes before its first evaluation.
  ASSERT_NO_THROW(itk::Impact::GetModelOutputsExample(configs, "fixed", device));

  // An intensity that depends on all three axes differently, so a re-partitioned patch cannot
  // coincide with a correct one.
  auto intensity = [](int64_t ix, int64_t iy, int64_t iz) {
    return static_cast<float>(std::sin(0.7 * ix) + 0.3 * iy - 0.11 * iz * iz);
  };

  // The reference patch, built directly at the shape the model expects: ITK x innermost.
  torch::Tensor reference = torch::zeros({ 1, 1, pz, py, px }, torch::kFloat32);
  auto          referenceAccessor = reference.accessor<float, 5>();
  for (int64_t iz = 0; iz < pz; ++iz)
    for (int64_t iy = 0; iy < py; ++iy)
      for (int64_t ix = 0; ix < px; ++ix)
        referenceAccessor[0][0][iz][iy][ix] = intensity(ix, iy, iz);

  torch::Tensor referenceCenter;
  {
    torch::NoGradGuard ng;
    referenceCenter = itk::Forward(configs[0], reference)[0].toTensor().index({ "...", pz / 2, py / 2, px / 2 });
  }

  // What the elastix evaluator does: fill a flat buffer in patch-offset order, then wrap it in
  // a tensor at the shape it was handed. The offsets carry (index - patchSize/2) at unit voxel
  // size, so the index each one came from is recoverable.
  auto evaluator = [&](const itk::Point<double, 3> &,
                       const std::vector<std::vector<float>> & patchIndex,
                       const std::vector<int64_t> &            shape) {
    std::vector<float> values(patchIndex.size(), 0.0f);
    for (size_t i = 0; i < patchIndex.size(); ++i)
    {
      values[i] = intensity(static_cast<int64_t>(patchIndex[i][0]) + px / 2,
                            static_cast<int64_t>(patchIndex[i][1]) + py / 2,
                            static_cast<int64_t>(patchIndex[i][2]) + pz / 2);
    }
    return torch::from_blob(values.data(), { torch::IntArrayRef(shape) }, torch::kFloat32).unsqueeze(0).clone();
  };

  std::vector<itk::Point<double, 3>>                        fixedPoints(1);
  std::vector<std::vector<std::vector<std::vector<float>>>> patchIndex(1);
  patchIndex[0].push_back(configs[0].GetPatchIndex());
  ASSERT_EQ(patchIndex[0][0].size(), static_cast<size_t>(px * py * pz));

  const std::vector<torch::Tensor> subset{ torch::arange(4, torch::kInt64) };
  std::vector<torch::Tensor>       outputs = itk::Impact::GenerateOutputs<itk::Point<double, 3>>(
    configs, fixedPoints, patchIndex, subset, device, evaluator);

  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_EQ(outputs[0].sizes(), (std::vector<int64_t>{ 1, 4 }));
  for (int64_t c = 0; c < 4; ++c)
  {
    EXPECT_NEAR(outputs[0][0][c].item<float>(), referenceCenter[0][c].item<float>(), 1e-4)
      << "channel " << c << ": the online patch must be the one the model would see directly";
  }
}

TEST(ImpactBackend, TwoDimensionalModelIsSweptOverAVolume)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const std::string model2d = std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt";
  const ImageType::SizeType size = { { 20, 17, 6 } }; // x, y tiled; z swept
  auto                      img = MakeRampImage(size);
  ImageType::SpacingType    spacing;
  spacing[0] = 1.0;
  spacing[1] = 1.0;
  spacing[2] = 2.5; // the swept axis is anisotropic, and must stay so
  img->SetSpacing(spacing);

  // The patch is 2D but the voxel size covers every image axis: the resampling grid is the
  // image's, and the swept axis needs a spacing to place its slices.
  itk::ImpactModelConfiguration cfg(
    model2d, 2, 1, { 8, 8 }, { 1.f, 1.f, 2.5f }, { 3, 3, 3 }, { true, true }, /*mixedPrecision*/ false);
  auto interp = InterpolatorType::New();
  interp->SetSplineOrder(3);
  auto f = FeatMapType::New();
  f->SetModelConfiguration(cfg);
  f->SetInterpolator(interp);
  f->AddInput(img);
  f->SetPCA(0);
  f->SetDevice("cpu");
  ASSERT_NO_THROW(f->Update());

  auto map = f->GetOutput(1); // passthrough
  EXPECT_EQ(map->GetLargestPossibleRegion().GetSize(), size) << "the swept axis must keep its extent";
  EXPECT_EQ(map->GetNumberOfComponentsPerPixel(), 2u);
  for (unsigned int d = 0; d < 3; ++d)
  {
    EXPECT_NEAR(map->GetSpacing()[d], spacing[d], 1e-6) << "axis " << d;
  }

  itk::ImageRegionConstIteratorWithIndex<ImageType> it(img, img->GetLargestPossibleRegion());
  double                                            maxDiff = 0.0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    maxDiff = std::max(maxDiff, std::abs(static_cast<double>(map->GetPixel(it.GetIndex())[0]) - it.Get()));
  }
  EXPECT_LT(maxDiff, 1e-3) << "each swept slice must be blended into a partition of unity";

  // The 4-channel convolution layer keeps the same grid.
  EXPECT_EQ(f->GetOutput(0)->GetLargestPossibleRegion().GetSize(), size);
  EXPECT_EQ(f->GetOutput(0)->GetNumberOfComponentsPerPixel(), 4u);
}

namespace
{
// Uniform noise on `size`, with `direction`: every voxel differs, so any axis mix-up shows.
ImageType::Pointer
MakeNoiseImage(const ImageType::SizeType & size, const ImageType::DirectionType & direction, unsigned int seed)
{
  auto image = ImageType::New();
  image->SetRegions(ImageType::RegionType(size));
  image->SetDirection(direction);
  image->Allocate();
  std::mt19937                                 generator(seed);
  std::uniform_real_distribution<float>        uniform(0.0f, 1.0f);
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, image->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    it.Set(uniform(generator));
  }
  return image;
}

// A feature map as a [channels, z, y, x] tensor (a copy).
torch::Tensor
FeatureMapTensor(const itk::VectorImage<float, 3> * map)
{
  const auto size = map->GetLargestPossibleRegion().GetSize();
  return torch::from_blob(const_cast<float *>(map->GetBufferPointer()),
                          { static_cast<int64_t>(size[2]),
                            static_cast<int64_t>(size[1]),
                            static_cast<int64_t>(size[0]),
                            static_cast<int64_t>(map->GetNumberOfComponentsPerPixel()) },
                          torch::kFloat32)
    .permute({ 3, 0, 1, 2 })
    .clone();
}

// Directions whose image axis j points head-feet (row 2 is the S component of each image axis).
ImageType::DirectionType
HeadFeetAlong(unsigned int axis)
{
  ImageType::DirectionType direction;
  direction.Fill(0.0);
  // A cyclic permutation: image axis `axis` -> S, the next -> L, the one after -> P.
  direction[2][axis] = 1.0;
  direction[0][(axis + 1) % 3] = 1.0;
  direction[1][(axis + 2) % 3] = 1.0;
  return direction;
}
} // namespace

// A 2D model is swept along the image axis closest to head-feet, argmax_j |Direction[2][j]|, a tie going to the
// larger j, whatever the acquisition plane: on a sagittal or coronal image it is not the last axis. A voxel changed
// in the input then changes the map in its own slice across that axis only -- the model's 3x3 convolution mixes
// the two other axes. Checked through ImageToFeaturesMap (elastix Static), and through the registration stages'
// extraction with and without the autograd graph.
TEST(ImpactBackend, TwoDimensionalModelSweepsTheHeadFeetAxis)
{
  ImageType::DirectionType identity;
  identity.SetIdentity();
  EXPECT_EQ(itk::Impact::HeadFeetAxis(identity, 3), 2u);
  for (unsigned int axis = 0; axis < 3; ++axis)
  {
    EXPECT_EQ(itk::Impact::HeadFeetAxis(HeadFeetAlong(axis), 3), axis);
  }
  ImageType::DirectionType oblique = identity; // 45 degrees about x: axes 1 and 2 tie
  const double             c = std::sqrt(0.5);
  oblique[1][1] = c;
  oblique[1][2] = -c;
  oblique[2][1] = c;
  oblique[2][2] = c;
  EXPECT_EQ(itk::Impact::HeadFeetAxis(oblique, 3), 2u) << "a tie goes to the larger axis";
  itk::Matrix<double, 2, 2> planar;
  planar.SetIdentity();
  EXPECT_EQ(itk::Impact::HeadFeetAxis(planar, 2), 1u);

  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const std::string             model2d = std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt";
  const ImageType::SizeType     size = { { 9, 8, 7 } };
  const ImageType::IndexType    marked = { { 4, 3, 3 } };
  itk::ImpactModelConfiguration config(
    model2d, 2, 1, { 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0 }, { true, false }, /*mixedPrecision*/ false);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  // The voxels (ITK index) where two feature tensors [C, z, y, x] differ.
  auto changed = [](const torch::Tensor & a, const torch::Tensor & b) {
    const torch::Tensor                 where = ((a - b).abs().amax(0) > 1e-4).nonzero(); // rows (z, y, x)
    std::vector<std::array<int64_t, 3>> voxels;
    for (int64_t r = 0; r < where.size(0); ++r)
    {
      voxels.push_back({ where[r][2].item<int64_t>(), where[r][1].item<int64_t>(), where[r][0].item<int64_t>() });
    }
    return voxels;
  };
  // The change stays in the marked voxel's slice across `axis` and spreads along both other axes.
  auto expectSliceOf = [&](const std::vector<std::array<int64_t, 3>> & voxels, unsigned int axis, const char * what) {
    ASSERT_FALSE(voxels.empty()) << what;
    std::array<bool, 3> spread{ false, false, false };
    for (const auto & v : voxels)
    {
      EXPECT_EQ(v[axis], marked[axis]) << what << ": a change left the slice across axis " << axis;
      for (unsigned int d = 0; d < 3; ++d)
      {
        spread[d] = spread[d] || v[d] != marked[d];
      }
    }
    for (unsigned int d = 0; d < 3; ++d)
    {
      if (d != axis)
      {
        EXPECT_TRUE(spread[d]) << what << ": the model did not span axis " << d;
      }
    }
  };

  for (unsigned int axis = 0; axis < 3; ++axis)
  {
    const ImageType::DirectionType direction = HeadFeetAlong(axis);
    auto                           image = MakeNoiseImage(size, direction, 3);
    auto                           touched = MakeNoiseImage(size, direction, 3);
    touched->SetPixel(marked, touched->GetPixel(marked) + 5.0f);

    auto mapOf = [&](ImageType::Pointer input) {
      auto filter = FeatMapType::New();
      filter->SetModelConfiguration(config);
      filter->SetInterpolator(interpolator);
      filter->AddInput(input);
      filter->SetDevice("cpu");
      filter->Update();
      return FeatureMapTensor(filter->GetOutput(0));
    };
    expectSliceOf(changed(mapOf(image), mapOf(touched)), axis, "ImageToFeaturesMap");

    const std::vector<itk::ImpactModelConfiguration> configs{ config };
    for (const bool withGrad : { false, true })
    {
      auto layerOf = [&](ImageType::Pointer input) {
        const torch::Tensor tensor = itk::Impact::ImageToBatchTensor(input.GetPointer());
        return itk::Impact::ExtractFeatureLayers<3>(
                 configs, tensor, torch::kCPU, {}, withGrad, {}, static_cast<int>(axis))[0]
          .squeeze(0)
          .detach();
      };
      expectSliceOf(changed(layerOf(image), layerOf(touched)),
                    axis,
                    withGrad ? "ExtractFeatureLayers with the graph" : "ExtractFeatureLayers");
    }
  }
}

// A 2D model's PCA, FireANTs' rule: the basis is fitted on the FIXED features of at most 32 slices along the sweep
// axis, at round(linspace(0, n - 1, min(n, 32))), each image centred by its own mean over those slices, and every
// slice projected. Here 40 slices lie along image axis 0, which the direction points head-feet.
TEST(ImpactBackend, TwoDimensionalModelPcaReadsSlicesAlongTheSweepAxis)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const std::string             model2d = std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt";
  const ImageType::SizeType     size = { { 40, 6, 5 } };
  const auto                    direction = HeadFeetAlong(0);
  auto                          fixed = MakeNoiseImage(size, direction, 1);
  auto                          moving = MakeNoiseImage(size, direction, 2);
  itk::ImpactModelConfiguration config(
    model2d, 2, 1, { 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0 }, { true, false }, /*mixedPrecision*/ false);
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);
  auto run = [&](ImageType::Pointer image, unsigned int pca, const std::vector<torch::Tensor> * basis) {
    auto filter = FeatMapType::New();
    filter->SetModelConfiguration(config);
    filter->SetInterpolator(interpolator);
    filter->AddInput(image);
    filter->SetPCA(pca);
    filter->SetDevice("cpu");
    if (basis)
    {
      itk::SetPrincipalComponents(*filter, *basis);
    }
    filter->Update();
    return std::make_pair(FeatureMapTensor(filter->GetOutput(0)), itk::GetPrincipalComponents(*filter));
  };

  const torch::Tensor fixedMap = run(fixed, 0, nullptr).first; // [4, z, y, x]
  const torch::Tensor movingMap = run(moving, 0, nullptr).first;
  const auto [fixedReduced, basis] = run(fixed, 2, nullptr);
  const torch::Tensor movingReduced = run(moving, 2, &basis).first;

  // The reference, written as the FireANTs engine computes it: the sweep axis (x) is tensor dimension 3.
  const torch::Tensor index = torch::linspace(0, 39, 32).round().to(torch::kLong);
  auto sampleOf = [&](const torch::Tensor & map) { return map.index_select(3, index).reshape({ 4, -1 }); };
  const torch::Tensor centred = sampleOf(fixedMap) - sampleOf(fixedMap).mean(1, true);
  const torch::Tensor covariance = centred.matmul(centred.t()) / (centred.size(1) - 1);
  const torch::Tensor reference = std::get<1>(torch::linalg_eigh(covariance)).narrow(1, 2, 2);
  auto                project = [&](const torch::Tensor & map) {
    return torch::einsum("cn,ck->kn", { map.reshape({ 4, -1 }) - sampleOf(map).mean(1, true), reference });
  };
  const torch::Tensor expectedFixed = project(fixedMap);
  const torch::Tensor expectedMoving = project(movingMap);
  for (int64_t k = 0; k < 2; ++k)
  {
    // An eigenvector is defined up to its sign.
    const double sign = (basis[0].select(1, k) * reference.select(1, k)).sum().item<double>() < 0 ? -1.0 : 1.0;
    EXPECT_LT((fixedReduced.select(0, k).flatten() - sign * expectedFixed.select(0, k)).abs().max().item<double>(),
              1e-4)
      << "fixed component " << k;
    EXPECT_LT((movingReduced.select(0, k).flatten() - sign * expectedMoving.select(0, k)).abs().max().item<double>(),
              1e-4)
      << "moving component " << k << " (the fixed basis, the moving image's own mean)";
  }
  // Centred by its own mean over the sampled slices: each image's projection averages to zero there.
  EXPECT_LT(fixedReduced.index_select(3, index).mean({ 1, 2, 3 }).abs().max().item<double>(), 1e-4);
  EXPECT_LT(movingReduced.index_select(3, index).mean({ 1, 2, 3 }).abs().max().item<double>(), 1e-4);
}

// The in-place blend gives the output image its geometry directly; it must agree with what
// TensorToImageFilter would have produced from the same tensor, since the PCA path still
// goes through that filter.
TEST(ImpactBackend, InPlaceFeatureImageGeometryMatchesTensorToImageFilter)
{
  const ImageType::SizeType size = { { 20, 17, 13 } };
  auto                      reference = MakeRampImage(size);
  ImageType::SpacingType    spacing;
  spacing[0] = 0.8;
  spacing[1] = 1.25;
  spacing[2] = 3.0;
  reference->SetSpacing(spacing);
  ImageType::DirectionType direction;
  direction.SetIdentity();
  direction[1][1] = -1.0;
  reference->SetDirection(direction);

  // A map at half the input resolution on one axis, so the derived spacing is exercised.
  const std::vector<int64_t> shape = { 5, 7, 17, 10 }; // [channels, z, y, x]

  auto inPlace = itk::VectorImage<float, 3>::New();
  itk::AllocateFeatureImage<3>(reference.GetPointer(), shape, inPlace.GetPointer());

  auto converter = itk::TensorToImageFilter<3>::New();
  converter->SetReferenceImage(reference.GetPointer());
  converter->SetTensor(torch::zeros({ shape[0], shape[1], shape[2], shape[3] }));
  converter->Update();
  auto viaFilter = converter->GetOutput();

  EXPECT_EQ(inPlace->GetLargestPossibleRegion().GetSize(), viaFilter->GetLargestPossibleRegion().GetSize());
  EXPECT_EQ(inPlace->GetNumberOfComponentsPerPixel(), viaFilter->GetNumberOfComponentsPerPixel());
  EXPECT_EQ(inPlace->GetDirection(), viaFilter->GetDirection());
  for (unsigned int d = 0; d < 3; ++d)
  {
    EXPECT_NEAR(inPlace->GetSpacing()[d], viaFilter->GetSpacing()[d], 1e-9);
    EXPECT_NEAR(inPlace->GetOrigin()[d], viaFilter->GetOrigin()[d], 1e-9);
  }
}

// --- A downsampling encoder yields a feature map that still overlays the input ----
// The feature-map spacing/size are derived from the input/output tensor size ratio
// (TensorToImageFilter: spacing = refExtent/outputSize). A /2 encoder must therefore
// produce a half-size, double-spacing map whose content still sits at the same
// physical location as the input, with no half-voxel drift. Two downsampled layers
// also confirm multi-layer independence under downsampling.
TEST(ImpactBackend, DownsamplingEncoderFeatureMapOverlaysInput)
{
  using FeatMapType = itk::ImageToFeaturesMap<ImageType, InterpolatorType>;
  const std::string downModel = std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModelDown.pt";
  // Sharp marker at input physical [10, 8, 8] (n=16).
  auto img = MakeBlobImage(16, 2.0, 0.0, 0.0, 1.0);

  auto interp = InterpolatorType::New();
  interp->SetSplineOrder(3);
  itk::ImpactModelConfiguration cfg(
    downModel, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 2, 2, 2 }, { true, true }, false);
  auto f = FeatMapType::New();
  f->SetModelConfiguration(cfg);
  f->SetInterpolator(interp);
  f->AddInput(img);
  f->SetPCA(0);
  f->SetDevice("cpu");
  ASSERT_NO_THROW(f->Update());

  // Layer 0 is an AvgPool (position-preserving), so its peak marks the true location.
  auto map0 = f->GetOutput(0);
  const auto size = map0->GetLargestPossibleRegion().GetSize();
  const auto spacing = map0->GetSpacing();
  const auto origin = map0->GetOrigin();
  for (unsigned int d = 0; d < 3; ++d)
  {
    EXPECT_EQ(size[d], 8u) << "downsampled feature map should be half the 16-voxel input";
    EXPECT_NEAR(spacing[d], 2.0, 1e-6) << "spacing should double to span the same physical extent";
    EXPECT_NEAR(origin[d], 0.0, 1e-6) << "origin should match the input (no half-voxel shift)";
  }

  itk::ImageRegionConstIteratorWithIndex<std::remove_reference_t<decltype(*map0)>> it(
    map0, map0->GetLargestPossibleRegion());
  double best = -1; std::remove_reference_t<decltype(*map0)>::IndexType bi{};
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) { const double v = std::abs(it.Get()[0]); if (v > best) { best = v; bi = it.GetIndex(); } }
  std::remove_reference_t<decltype(*map0)>::PointType bp;
  map0->TransformIndexToPhysicalPoint(bi, bp);
  // The downsampled marker must still be at the input's physical position (within one
  // downsampled voxel), confirming the feature map physically overlays the input.
  EXPECT_NEAR(bp[0], 10.0, 1.5) << "marker x should overlay the input";
  EXPECT_NEAR(bp[1], 8.0, 1.5) << "marker y should overlay the input";
  EXPECT_NEAR(bp[2], 8.0, 1.5) << "marker z should overlay the input";
}

// Validate every loss's non-mutating forwardValue() (the differentiable path the Torch-Adam
// optimizer backpropagates through) against central finite differences, in double precision.
TEST(ImpactLoss, NormalizationLatchesEachLayerOnceALevel)
{
  itk::Impact::LossNormalization normalization;
  EXPECT_DOUBLE_EQ(normalization.Factor(0), 1.0) << "no factor before the first value";
  EXPECT_DOUBLE_EQ(normalization.Latch(0, 4.0), 0.25);
  EXPECT_DOUBLE_EQ(normalization.Latch(0, 2.0), 0.25) << "latched once a level";
  EXPECT_DOUBLE_EQ(normalization.Latch(1, 0.0), 1.0) << "an already matched layer keeps the factor 1";
  EXPECT_DOUBLE_EQ(normalization.Latch(2, std::nan("")), 1.0);
  normalization.Reset();
  EXPECT_DOUBLE_EQ(normalization.Latch(0, 2.0), 0.5) << "a new level latches again";
}

TEST(ImpactLoss, FeatureNormalizationScalesEachVector)
{
  torch::manual_seed(3);
  const torch::Tensor features = torch::rand({ 1, 5, 4, 4, 4 }, torch::kFloat64) * 7.0 + 0.5;
  const torch::Tensor l2 = itk::Impact::NormalizeFeatureChannels(features, "l2", 1);
  EXPECT_LT((l2.norm(2, 1) - 1.0).abs().max().item<double>(), 1e-9) << "l2: unit vectors";
  const torch::Tensor standardized = itk::Impact::NormalizeFeatureChannels(features, "standardized", 1);
  EXPECT_LT(standardized.mean(1).abs().max().item<double>(), 1e-9) << "standardized: zero mean";
  EXPECT_LT((standardized.std(1, true) - 1.0).abs().max().item<double>(), 1e-6) << "and unit deviation";
  EXPECT_TRUE(torch::equal(itk::Impact::NormalizeFeatureChannels(features, "none", 1), features));
  EXPECT_THROW(itk::Impact::NormalizeFeatureChannels(features, "L2", 1), std::runtime_error);
  itk::ImpactModelConfiguration configuration;
  EXPECT_EQ(configuration.GetFeatureNormalization(), "none");
  EXPECT_THROW(configuration.SetFeatureNormalization("unit"), std::invalid_argument);
}

// LNCC compares windows: 0 for identical maps, positive otherwise, a gradient that matches finite differences,
// and no meaning at sampled points.
TEST(ImpactLoss, LocalNCCComparesWindowsOfDenseMaps)
{
  torch::manual_seed(5);
  auto                loss = itk::Impact::LossFactory::Instance().Create("LNCC");
  const auto          options = torch::TensorOptions().dtype(torch::kFloat64);
  const torch::Tensor fixed = torch::rand({ 1, 3, 9, 9, 9 }, options);
  ASSERT_TRUE(loss->IsSpatial());
  const std::vector<int64_t> window{ 3, 3, 3 };
  EXPECT_NEAR(loss->forwardSpatial(fixed, fixed, window).item<double>(), 0.0, 1e-6);
  const torch::Tensor moving0 = torch::rand({ 1, 3, 9, 9, 9 }, options);
  torch::Tensor       moving = moving0.clone().set_requires_grad(true);
  const torch::Tensor value = loss->forwardSpatial(fixed, moving, window);
  EXPECT_NE(loss->forwardSpatial(fixed, moving0, { 3, 1, 3 }).item<double>(), value.item<double>())
    << "the window has a side per axis";
  EXPECT_GT(value.item<double>(), 0.1);
  value.backward();
  for (const int64_t flat : { 0L, 400L, 1500L })
  {
    torch::Tensor plus = moving0.clone(), minus = moving0.clone();
    plus.view(-1)[flat] += 1e-6;
    minus.view(-1)[flat] -= 1e-6;
    const double fd = (loss->forwardSpatial(fixed, plus, window).item<double>() -
                       loss->forwardSpatial(fixed, minus, window).item<double>()) /
                      2e-6;
    EXPECT_NEAR(moving.grad().view(-1)[flat].item<double>(), fd, 1e-5 + 1e-3 * std::abs(fd)) << "element " << flat;
  }
  torch::Tensor points = torch::rand({ 10, 3 }, options);
  EXPECT_THROW(loss->forwardValue(points, points), std::runtime_error);
}

TEST(ImpactLoss, ForwardValueGradientFiniteDifference)
{
  torch::manual_seed(7);
  const int64_t N = 48, C = 6;
  const auto    opts = torch::TensorOptions().dtype(torch::kFloat64);
  for (const std::string name : { "L1", "L2", "Cosine", "L1Cosine", "Dice", "NCC" })
  {
    auto          loss = itk::Impact::LossFactory::Instance().Create(name);
    torch::Tensor fixed = torch::rand({ N, C }, opts) + 0.2;  // strictly positive (Dice-safe)
    torch::Tensor moving0 = torch::rand({ N, C }, opts) + 0.2;
    torch::Tensor moving = moving0.clone().set_requires_grad(true);

    torch::Tensor value = loss->forwardValue(fixed, moving);
    value.backward();
    torch::Tensor grad = moving.grad().clone();

    const double eps = 1e-6;
    int          checked = 0;
    for (int64_t i = 0; i < N && checked < 6; i += 7)
      for (int64_t j = 0; j < C && checked < 6; j += 2, ++checked)
      {
        torch::Tensor mp = moving0.clone();
        mp[i][j] += eps;
        torch::Tensor mm = moving0.clone();
        mm[i][j] -= eps;
        double vp, vm;
        {
          torch::NoGradGuard ng;
          vp = loss->forwardValue(fixed, mp).item<double>();
          vm = loss->forwardValue(fixed, mm).item<double>();
        }
        const double fd = (vp - vm) / (2 * eps);
        const double analytic = grad[i][j].item<double>();
        EXPECT_NEAR(analytic, fd, 1e-4 * (1.0 + std::abs(fd)))
          << name << " forwardValue gradient mismatch at (" << i << "," << j << ")";
      }
  }
}

// --- ImpactFineRegistration: fast Torch-backed Adam dense registration ---------
namespace
{
using TorchAdamFilterType = itk::ImpactFineRegistration<ImageType>;
using TorchAdamFieldType = TorchAdamFilterType::DisplacementFieldType;

float
TorchAdamPattern(double x, double y, double z, unsigned int n)
{
  const double pi = 4.0 * std::atan(1.0);
  // Low, equal spatial frequency (~1.3 cycles) and equal amplitude on every axis: long
  // wavelength avoids periodic translation aliasing on small grids, and the isotropic
  // gradient gives every axis an equally well-posed alignment (distinct phases break
  // axis symmetry).
  return static_cast<float>(std::sin(2 * pi * (1.3 * x / n + 0.10)) + std::sin(2 * pi * (1.3 * y / n + 0.37)) +
                            std::sin(2 * pi * (1.3 * z / n + 0.21)));
}

// Textured pattern translated by (tx,ty,tz) index voxels, on a given geometry.
ImageType::Pointer
MakeTorchAdamPattern(unsigned int                     n,
                     double                           tx,
                     double                           ty,
                     double                           tz,
                     const ImageType::SpacingType &   spacing,
                     const ImageType::DirectionType & direction)
{
  auto                  image = ImageType::New();
  ImageType::SizeType   size;
  size.Fill(n);
  ImageType::RegionType region;
  region.SetSize(size);
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetDirection(direction);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto i = it.GetIndex();
    image->SetPixel(i, TorchAdamPattern(i[0] - tx, i[1] - ty, i[2] - tz, n));
  }
  return image;
}

itk::Vector<double, 3>
InteriorMeanError(const TorchAdamFieldType * field, const itk::Vector<double, 3> & expected, int margin)
{
  itk::Vector<double, 3> acc;
  acc.Fill(0.0);
  const auto size = field->GetLargestPossibleRegion().GetSize();
  long       count = 0;
  itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType> it(field, field->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto idx = it.GetIndex();
    bool       interior = true;
    for (unsigned int d = 0; d < 3; ++d)
      if (idx[d] < margin || idx[d] >= static_cast<long>(size[d]) - margin)
        interior = false;
    if (!interior)
      continue;
    const auto u = it.Get();
    for (unsigned int d = 0; d < 3; ++d)
      acc[d] += std::abs(u[d] - expected[d]);
    ++count;
  }
  if (count > 0)
    for (unsigned int d = 0; d < 3; ++d)
      acc[d] /= count;
  return acc;
}
} // namespace

// With no optimization the field must be exactly zero and the identity grid_sample must
// reproduce the input, for both an axis-aligned and an oblique/anisotropic geometry.
// (Guards the z,y,x<->x,y,z, voxel<->mm and direction-rotation writeback in isolation.)
TEST(ImpactTorchAdam, ZeroIterationIdentity)
{
  ImageType::DirectionType oblique;
  oblique.SetIdentity();
  const double a = 0.30;
  oblique[0][0] = std::cos(a);
  oblique[0][1] = -std::sin(a);
  oblique[1][0] = std::sin(a);
  oblique[1][1] = std::cos(a);

  for (int variant = 0; variant < 2; ++variant)
  {
    ImageType::SpacingType   spacing;
    ImageType::DirectionType direction;
    if (variant == 0)
    {
      spacing.Fill(1.0);
      direction.SetIdentity();
    }
    else
    {
      spacing[0] = 1.3;
      spacing[1] = 0.8;
      spacing[2] = 1.1;
      direction = oblique;
    }
    auto fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, direction);

    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(fixed);
    filter->SetMovingImage(fixed);
    filter->SetNumberOfIterations(0);
    filter->Update();

    double maxField = 0.0;
    itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType> it(
      filter->GetDisplacementField(), filter->GetDisplacementField()->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      for (unsigned int d = 0; d < 3; ++d)
        maxField = std::max(maxField, static_cast<double>(std::abs(it.Get()[d])));
    EXPECT_LT(maxField, 1e-6) << "field must be exactly 0 with no optimization (variant " << variant << ")";

    double      maxWarpDiff = 0.0;
    const auto * warped = filter->GetWarpedMovingImage();
    itk::ImageRegionConstIteratorWithIndex<ImageType> wi(warped, warped->GetLargestPossibleRegion());
    for (wi.GoToBegin(); !wi.IsAtEnd(); ++wi)
      maxWarpDiff =
        std::max(maxWarpDiff, std::abs(static_cast<double>(wi.Get() - fixed->GetPixel(wi.GetIndex()))));
    EXPECT_LT(maxWarpDiff, 1e-4) << "identity grid_sample should reproduce the input (variant " << variant << ")";
  }
}

// Recover a known translation from raw-intensity MSE, including the direction-rotated
// physical displacement under an oblique/anisotropic geometry.
TEST(ImpactTorchAdam, TranslationRecoveryIntensity)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::DirectionType oblique;
  oblique.SetIdentity();
  const double a = 0.30;
  oblique[0][0] = std::cos(a);
  oblique[0][1] = -std::sin(a);
  oblique[1][0] = std::sin(a);
  oblique[1][1] = std::cos(a);

  for (int variant = 0; variant < 2; ++variant)
  {
    ImageType::SpacingType   spacing;
    ImageType::DirectionType direction;
    if (variant == 0)
    {
      spacing.Fill(1.0);
      direction.SetIdentity();
    }
    else
    {
      spacing[0] = 1.3;
      spacing[1] = 0.8;
      spacing[2] = 1.1;
      direction = oblique;
    }
    auto fixed = MakeTorchAdamPattern(20, 0, 0, 0, spacing, direction);
    auto moving = MakeTorchAdamPattern(20, tx, ty, tz, spacing, direction);

    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(fixed);
    filter->SetMovingImage(moving);
    filter->SetNumberOfIterations(300);
    filter->SetLearningRate(0.2);
    filter->SetRegularizationWeight(0.02);
    filter->Update();

    itk::Vector<double, 3> expected;
    for (unsigned int r = 0; r < 3; ++r)
    {
      double v = 0.0;
      for (unsigned int c = 0; c < 3; ++c)
        v += direction[r][c] * spacing[c] * ((c == 0) ? tx : (c == 1) ? ty : tz);
      expected[r] = v;
    }
    const auto err = InteriorMeanError(filter->GetDisplacementField(), expected, 6);
    const auto & hist = filter->GetMetricValuesPerIteration();

    EXPECT_LT(hist.back(), 0.1 * hist.front()) << "loss should drop (variant " << variant << ")";
    // Tolerance is in mm; under the oblique/anisotropic geometry the (sub-0.05-voxel)
    // residual is rotated and spacing-scaled, so allow a slightly looser bound.
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_LT(err[d], 0.3) << "axis " << d << " displacement off (variant " << variant << ")";
  }
}

// Recover the same translation, but with the similarity computed on IMPACT deep features
// (toy TorchScript model) via itk::Forward + ImpactLoss::forwardValue.
TEST(ImpactTorchAdam, TranslationRecoveryFeatures)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();

  auto fixed = MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false);
  filter->AddModelConfiguration(config);
  filter->SetDistance({ "L2", "L2" });
  filter->SetLayersWeight({ 1.f, 1.f });
  filter->SetNumberOfIterations(300);
  filter->SetLearningRate(0.2);
  filter->SetRegularizationWeight(0.02);
  filter->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto   err = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  const auto & hist = filter->GetMetricValuesPerIteration();

  EXPECT_LT(hist.back(), 0.3 * hist.front()) << "feature loss should drop";
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 0.3) << "feature-mode axis " << d << " displacement off";
}

// Per-layer PCA reduces the toy model's 4-channel layer to 2 components (fit on the fixed
// features, both projected onto the same basis); a known translation must still be recovered.
// Normalized, every layer starts at 1: the first recorded value is the sum of the layer weights (the regulariser
// is 0 on the zero starting field), and it drops as the field moves.
TEST(ImpactTorchAdam, NormalizedLossStartsAtTheSumOfTheWeights)
{
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity));
  filter->SetMovingImage(MakeTorchAdamPattern(20, 1.5, -2.0, 1.0, spacing, identity));
  filter->AddModelConfiguration(itk::ImpactModelConfiguration(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false));
  filter->SetDistance({ "L2", "L2" });
  filter->SetLayersWeight({ 1.f, 0.5f });
  filter->SetNumberOfIterations(60);
  filter->SetLearningRate(0.2);
  filter->SetRegularizationWeight(0.02);
  filter->Update();

  const auto & hist = filter->GetMetricValuesPerIteration();
  EXPECT_NEAR(hist.front(), 1.5, 1e-4);
  EXPECT_LT(hist.back(), 0.5 * hist.front());
}

// The fine stage takes the settings every IMPACT backend shares: a random channel subset redrawn at every
// iteration, the LNCC distance, a voxel size the image is resampled to before the model, and a per-voxel
// normalization of the features; each still recovers a known translation.
TEST(ImpactTorchAdam, SharedFeatureSettingsStillRecoverATranslation)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto run = [&](const std::string & distance,
                 unsigned int        subset,
                 float               voxel,
                 const std::string & normalization) {
    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity));
    filter->SetMovingImage(MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity));
    itk::ImpactModelConfiguration config(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { voxel, voxel, voxel }, { 0, 0, 0 }, { true, false }, false);
    config.SetFeatureNormalization(normalization);
    filter->AddModelConfiguration(config);
    filter->SetDistance({ distance });
    filter->SetLayersWeight({ 1.f });
    filter->SetSubsetFeatures({ subset });
    filter->SetNumberOfIterations(300);
    filter->SetLearningRate(0.2);
    filter->SetRegularizationWeight(0.02);
    filter->Update();
    itk::Vector<double, 3> expected;
    expected[0] = tx;
    expected[1] = ty;
    expected[2] = tz;
    return InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  };
  for (const auto & [label, error] :
       std::vector<std::pair<std::string, itk::Vector<double, 3>>>{ { "subset 2 of 4", run("L2", 2, 0.f, "none") },
                                                                     { "LNCC", run("LNCC", 0, 0.f, "none") },
                                                                     { "voxel size 2", run("L2", 0, 2.f, "none") },
                                                                     { "l2 normalization", run("L2", 0, 0.f, "l2") } })
  {
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_LT(error[d], 0.5) << label << ": axis " << d << " off";
  }
}

// SamplingPercentage reads a random share of each layer's voxels at every iteration, warping only those points:
// the stochastic estimate still recovers a known translation. (A 20-voxel cube has 8000 of them: at 5 %, 400 points
// an iteration, Adam's constant step left the end 0.6 voxel off; a real image has far more at any share.)
TEST(ImpactTorchAdam, SampledVoxelsStillRecoverATranslation)
{
  const double           tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  for (const double share : { 0.25, 0.1 })
  {
    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity));
    filter->SetMovingImage(MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity));
    filter->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, false }, false));
    filter->SetDistance({ "L2" });
    filter->SetSamplingPercentage(share);
    filter->SetNumberOfIterations(300);
    filter->SetLearningRate(0.2);
    filter->SetRegularizationWeight(0.02);
    filter->Update();
    itk::Vector<double, 3> expected;
    expected[0] = tx;
    expected[1] = ty;
    expected[2] = tz;
    const itk::Vector<double, 3> error = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_LT(error[d], 0.5) << share << " of the voxels: axis " << d << " off";
  }
}

// A share of the voxels only means something to a feature similarity read at points: outside (0, 1], with LNCC
// (windows of whole maps), without a model, or in Jacobian mode without a patch to run the model on, it is refused.
TEST(ImpactTorchAdam, SamplingRefusesWhatReadsWholeImages)
{
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto attempt =
    [&](double share, const std::string & distance, const std::string & mode, bool model, unsigned int patch = 0) {
      auto filter = TorchAdamFilterType::New();
      filter->SetFixedImage(MakeTorchAdamPattern(12, 0, 0, 0, spacing, identity));
      filter->SetMovingImage(MakeTorchAdamPattern(12, 1, 0, 0, spacing, identity));
      if (model)
      {
        filter->AddModelConfiguration(itk::ImpactModelConfiguration(
          ToyModelPath(), 3, 1, { patch, patch, patch }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, false }, false));
        filter->SetDistance({ distance });
        filter->SetMode(mode);
      }
      filter->SetSamplingPercentage(share);
      filter->SetNumberOfIterations(2);
      filter->Update();
    };
  EXPECT_NO_THROW(attempt(0.5, "L2", "Static", true));
  EXPECT_THROW(attempt(0.0, "L2", "Static", true), itk::ExceptionObject);
  EXPECT_THROW(attempt(1.5, "L2", "Static", true), itk::ExceptionObject);
  EXPECT_THROW(attempt(0.5, "LNCC", "Static", true), itk::ExceptionObject);
  EXPECT_THROW(attempt(0.5, "L2", "Jacobian", true), itk::ExceptionObject) << "no patch";
  EXPECT_NO_THROW(attempt(0.5, "L2", "Jacobian", true, 3));
  EXPECT_NO_THROW(attempt(0.5, "NCC", "Jacobian", true, 3));
  EXPECT_THROW(attempt(0.5, "LNCC", "Jacobian", true, 3), itk::ExceptionObject);
  EXPECT_THROW(attempt(0.5, "L2", "Static", false), itk::ExceptionObject);
}

// The plane a 2D model's patch is cut on around a sampled point is the metric's (PatchPlaneRotation: a rotation, a
// function of the seed, the model and the point), and the fine stage's sampled Jacobian mode lays the patch on its
// first two columns, stepped by the model's voxel sizes, the voxel read back on the point; a 3D model's patch is the
// box of its voxel sizes along the image axes. In voxels of an anisotropic grid.
TEST(ImpactTorchAdam, SampledJacobianPatchesLieOnTheMetricsPlanes)
{
  const std::array<int64_t, 3> p{ 4, 5, 6 };
  const std::array<int64_t, 3> q{ 4, 5, 7 };
  const auto                   rotation = itk::Impact::PatchPlaneRotation(7, 0, p);
  const auto                   orthogonality = rotation.GetTranspose() * rotation.GetVnlMatrix();
  for (unsigned int r = 0; r < 3; ++r)
    for (unsigned int c = 0; c < 3; ++c)
      EXPECT_NEAR(orthogonality(r, c), r == c ? 1.0 : 0.0, 1e-12);
  EXPECT_NEAR(vnl_det(rotation.GetVnlMatrix()), 1.0, 1e-12);
  EXPECT_EQ(rotation, itk::Impact::PatchPlaneRotation(7, 0, p)) << "a function of the point";
  EXPECT_NE(rotation, itk::Impact::PatchPlaneRotation(7, 0, q));
  EXPECT_NE(rotation, itk::Impact::PatchPlaneRotation(7, 1, p));
  EXPECT_NE(rotation, itk::Impact::PatchPlaneRotation(8, 0, p));

  const std::vector<double>     spacing{ 0.5, 0.8, 1.25 };
  const torch::Tensor           points = torch::tensor({ 4, 5, 6, 2, 3, 1 }, torch::kLong).reshape({ 2, 3 });
  itk::ImpactModelConfiguration plane(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                                      2,
                                      1,
                                      { 5, 3 },
                                      { 0.5f, 0.75f, 1.f },
                                      { 0, 0 },
                                      { true, false },
                                      false);
  const torch::Tensor           planar = itk::Impact::SampledPatchPositions<3>(plane, points, spacing, 7, 0);
  ASSERT_EQ(planar.sizes(), torch::IntArrayRef({ 2, 15, 3 }));
  for (int64_t n = 0; n < 2; ++n)
  {
    const std::array<int64_t, 3> point{ points[n][0].item<int64_t>(),
                                        points[n][1].item<int64_t>(),
                                        points[n][2].item<int64_t>() };
    const auto                   axes = itk::Impact::PatchPlaneRotation(7, 0, point);
    for (int64_t k = 0; k < 15; ++k)
    {
      const double u = (k % 5 - 2) * 0.5, v = (k / 5 - 1) * 0.75; // mm along the two patch axes
      for (unsigned int j = 0; j < 3; ++j)
      {
        EXPECT_NEAR(planar[n][k][j].item<double>(), point[j] + (axes[j][0] * u + axes[j][1] * v) / spacing[j], 1e-9)
          << "point " << n << ", patch voxel " << k << ", axis " << j;
      }
    }
  }
  // The SampleCheck reads the patch's corners only: a patch is the image of a box.
  const std::vector<int64_t> corners = itk::Impact::PatchCorners(plane);
  EXPECT_EQ(corners, (std::vector<int64_t>{ 0, 4, 10, 14 }));
  EXPECT_TRUE(torch::allclose(itk::Impact::SampledPatchPositions<3>(plane, points, spacing, 7, 0, corners),
                              planar.index_select(1, torch::tensor(corners, torch::kLong))));
  itk::ImpactModelConfiguration box(
    ToyModelPath(), 3, 1, { 3, 3, 3 }, { 0.f, 1.f, 2.5f }, { 0, 0, 0 }, { true, false }, false);
  const torch::Tensor boxed = itk::Impact::SampledPatchPositions<3>(box, points, spacing, 7, 0);
  const double        step[3] = { 1.0, 1.25, 2.0 }; // unset = the spacing, then 1 mm / 0.8 and 2.5 mm / 1.25
  for (int64_t k = 0; k < 27; ++k)
  {
    const int64_t index[3] = { k % 3, (k / 3) % 3, k / 9 };
    for (unsigned int j = 0; j < 3; ++j)
      EXPECT_NEAR(boxed[0][k][j].item<double>(), points[0][j].item<double>() + (index[j] - 1) * step[j], 1e-12);
  }
}

// elastix's IMPACT metric cuts a 2D model's patch through Impact::GetPatchIndex: for the same point, seed and model
// it lays the patch on the plane the itkv4 metric (PatchPlaneRotation) and ImpactFineRegistration's sampled Jacobian
// mode (SampledPatchPositions) use, so the three hosts draw identical planes. A 3D model keeps its precomputed box.
TEST(ImpactBackend, GetPatchIndexCutsTheSamePlaneAsTheOtherHosts)
{
  itk::ImpactModelConfiguration plane(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                                      2,
                                      1,
                                      { 5, 3 },
                                      { 0.5f, 0.75f, 1.f },
                                      { 0, 0 },
                                      { true, false },
                                      false);
  const std::vector<double>     millimetre{ 1.0, 1.0, 1.0 };
  for (const std::array<int64_t, 3> point : { std::array<int64_t, 3>{ 4, 5, 6 }, std::array<int64_t, 3>{ 2, 3, 1 } })
  {
    for (const size_t model : { size_t{ 0 }, size_t{ 1 } })
    {
      const std::vector<std::vector<float>> patch = itk::Impact::GetPatchIndex(plane, 7, model, point, 3);
      const torch::Tensor centre = torch::tensor({ point[0], point[1], point[2] }, torch::kLong).reshape({ 1, 3 });
      const torch::Tensor positions = itk::Impact::SampledPatchPositions<3>(plane, centre, millimetre, 7, model);
      const auto          rotation = itk::Impact::PatchPlaneRotation(7, model, point);
      ASSERT_EQ(patch.size(), 15u);
      for (size_t k = 0; k < 15; ++k)
      {
        const double u = (static_cast<double>(k % 5) - 2) * 0.5, v = (static_cast<double>(k / 5) - 1) * 0.75;
        for (unsigned int j = 0; j < 3; ++j)
        {
          EXPECT_NEAR(patch[k][j], rotation[j][0] * u + rotation[j][1] * v, 1e-6) << "PatchPlaneRotation";
          EXPECT_NEAR(patch[k][j], positions[0][static_cast<int64_t>(k)][j].item<double>() - point[j], 1e-6)
            << "the sampled Jacobian mode's patch";
        }
      }
    }
  }
  EXPECT_NE(itk::Impact::GetPatchIndex(plane, 7, 0, std::array<int64_t, 3>{ 4, 5, 6 }, 3),
            itk::Impact::GetPatchIndex(plane, 7, 0, std::array<int64_t, 3>{ 4, 5, 7 }, 3))
    << "a plane per point";
  itk::ImpactModelConfiguration box(
    ToyModelPath(), 3, 1, { 3, 3, 3 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false);
  EXPECT_EQ(itk::Impact::GetPatchIndex(box, 7, 0, std::array<int64_t, 3>{ 4, 5, 6 }, 3), box.GetPatchIndex());
}

// The sampled Jacobian mode's gradient (the drawn points' patches warped by the field, run through the network, their
// centre features compared, backpropagated batch by batch to the control grid) against finite differences of its
// value, for a per-point distance and for NCC, whose gradient is seeded from the whole point set. The gradient is read
// off the first Adam step: without momentum it moves each control voxel by -lr g / (|g| + eps), -g for lr = eps. The
// field starts off the voxel grid, so no patch sample crosses a voxel boundary under the finite-difference step.
TEST(ImpactTorchAdam, SampledJacobianGradientMatchesFiniteDifferences)
{
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto         fixed = MakeTorchAdamPattern(12, 0, 0, 0, spacing, identity);
  auto         moving = MakeTorchAdamPattern(12, 1.0, -0.5, 0.7, spacing, identity);
  const double start[3] = { 0.3, -0.2, 0.25 };
  auto         fieldWith = [&](const TorchAdamFieldType::IndexType & where, unsigned int component, double delta) {
    auto field = TorchAdamFieldType::New();
    field->SetRegions(fixed->GetLargestPossibleRegion());
    field->CopyInformation(fixed);
    field->Allocate();
    TorchAdamFieldType::PixelType value;
    for (unsigned int d = 0; d < 3; ++d)
      value[d] = static_cast<float>(start[d]);
    field->FillBuffer(value);
    value[component] += static_cast<float>(delta);
    field->SetPixel(where, value);
    return field;
  };
  auto run = [&](TorchAdamFieldType::Pointer field, const std::string & distance) {
    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(fixed);
    filter->SetMovingImage(moving);
    filter->SetInitialDisplacementField(field);
    filter->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 5, 5, 5 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false));
    filter->SetDistance({ distance });
    filter->SetMode("Jacobian");
    filter->SetSamplingPercentage(0.3);
    filter->SetBatchSize(64); // several batches
    filter->NormalizeLossesOff();
    filter->SetRegularizationWeight(0.0);
    filter->SetNumberOfIterations(1);
    filter->SetSeed(3);
    filter->SetBeta1(0.0);
    filter->SetBeta2(0.0);
    filter->SetEpsilon(1e4);
    filter->SetLearningRate(1e4);
    filter->Update();
    return filter;
  };
  const TorchAdamFieldType::IndexType origin{ { 0, 0, 0 } };
  for (const std::string distance : { "L2", "NCC" })
  {
    auto initial = fieldWith(origin, 0, 0.0);
    auto step = run(initial, distance);
    ASSERT_EQ(step->GetMetricValuesPerIteration().size(), 1u);
    // The three largest gradient entries.
    std::vector<std::tuple<double, TorchAdamFieldType::IndexType, unsigned int>> entries;
    itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType>                   it(step->GetDisplacementField(),
                                                                  fixed->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      for (unsigned int c = 0; c < 3; ++c)
        entries.emplace_back(-(it.Get()[c] - initial->GetPixel(it.GetIndex())[c]), it.GetIndex(), c);
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
      return std::abs(std::get<0>(a)) > std::abs(std::get<0>(b));
    });
    ASSERT_GT(std::abs(std::get<0>(entries[0])), 1e-4) << distance << ": no gradient";
    const double h = 0.05;
    for (size_t e = 0; e < 3; ++e)
    {
      const auto & [gradient, where, component] = entries[e];
      const double plus = run(fieldWith(where, component, h), distance)->GetMetricValuesPerIteration()[0];
      const double minus = run(fieldWith(where, component, -h), distance)->GetMetricValuesPerIteration()[0];
      const double difference = (plus - minus) / (2.0 * h);
      EXPECT_NEAR(gradient, difference, 0.03 * std::abs(difference))
        << distance << ": voxel " << where << ", component " << component;
    }
  }
}

// The sampled Jacobian mode as the presets run it: two models, a smoothed half-resolution control grid, the
// normalization on, and several batches of points. Every batch back-propagates through the same smoothing of the
// control grid, which a backward frees: the gradient is gathered on the smoothed field and crosses the smoothing once.
// The first iteration's value is then each layer normalized to 1, and the gradient (read off a momentum-free first
// Adam step, as above) does not depend on how the points are batched.
TEST(ImpactTorchAdam, SampledJacobianBatchesShareTheSmoothedControlGrid)
{
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(12, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(12, 1.0, -0.5, 0.7, spacing, identity);
  auto run = [&](unsigned int batch, unsigned int iterations, bool readGradient) {
    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(fixed);
    filter->SetMovingImage(moving);
    filter->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 3, 3, 3 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false));
    filter->AddModelConfiguration(
      itk::ImpactModelConfiguration(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                                    2,
                                    1,
                                    { 5, 5 },
                                    { 1.f, 1.f, 1.f },
                                    { 0, 0 },
                                    { false, true },
                                    false));
    filter->SetDistance({ "L2", "L2" });
    filter->SetMode("Jacobian");
    filter->SetSamplingPercentage(0.3);
    filter->SetBatchSize(batch);
    filter->SetGridShrinkFactor(2);
    filter->SetControlGridSmoothingIterations(3);
    filter->SetRegularizationWeight(0.0);
    filter->SetNumberOfIterations(iterations);
    filter->SetSeed(3);
    if (readGradient)
    {
      filter->SetBeta1(0.0);
      filter->SetBeta2(0.0);
      filter->SetEpsilon(1e4);
      filter->SetLearningRate(1e4);
    }
    filter->Update();
    return filter;
  };
  auto batched = run(64, 1, true);
  auto whole = run(0, 1, true);
  ASSERT_EQ(batched->GetMetricValuesPerIteration().size(), 1u);
  EXPECT_NEAR(batched->GetMetricValuesPerIteration()[0], 2.0, 1e-5) << "two layers, each normalized to 1";
  double                                                     step = 0.0, difference = 0.0;
  itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType> it(whole->GetDisplacementField(),
                                                                fixed->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    for (unsigned int c = 0; c < 3; ++c)
    {
      step = std::max(step, std::abs(static_cast<double>(it.Get()[c])));
      difference = std::max(
        difference,
        std::abs(static_cast<double>(it.Get()[c]) - batched->GetDisplacementField()->GetPixel(it.GetIndex())[c]));
    }
  }
  ASSERT_GT(step, 1e-6) << "no gradient";
  EXPECT_LT(difference, 1e-3 * step) << "the batches must add up to the whole set's gradient";
  EXPECT_NO_THROW(run(64, 3, false)) << "later iterations";
}

// The sampled Jacobian mode registers: a translation between two patterns is recovered from a quarter of the voxels.
// Each point constrains the field through its whole patch, which a full-resolution unsmoothed control grid leaves
// under-determined (the dense Jacobian mode as well): the grid is the presets' kind, half the resolution, smoothed.
TEST(ImpactTorchAdam, SampledJacobianRecoversATranslation)
{
  const double           tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity));
  filter->SetMovingImage(MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity));
  filter->AddModelConfiguration(itk::ImpactModelConfiguration(
    ToyModelPath(), 3, 1, { 3, 3, 3 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false));
  filter->SetDistance({ "L2" });
  filter->SetMode("Jacobian");
  filter->SetSamplingPercentage(0.25);
  filter->SetGridShrinkFactor(2);
  filter->SetControlGridSmoothingIterations(1);
  filter->SetNumberOfIterations(300);
  filter->SetLearningRate(0.2);
  filter->SetRegularizationWeight(0.02);
  filter->Update();
  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const itk::Vector<double, 3> error = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(error[d], 0.5) << "axis " << d << " off";
}

// A model of lower dimension is swept slice by slice and, in Jacobian mode, taken in z-chunks matched with the
// fixed features: a voxel size that resamples the image would hand back chunks of another grid, so it is refused.
TEST(ImpactTorchAdam, ChunkedJacobianRefusesAResamplingModel)
{
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(MakeTorchAdamPattern(16, 0, 0, 0, spacing, identity));
  filter->SetMovingImage(MakeTorchAdamPattern(16, 1, 0, 0, spacing, identity));
  filter->AddModelConfiguration(itk::ImpactModelConfiguration(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                                                              2, 1, { 0, 0 }, { 2.f, 2.f, 2.f }, { 0, 0 }, { true, false }, false));
  filter->SetDistance({ "L2" });
  filter->SetMode("Jacobian");
  filter->SetFeatureChunkSize(4);
  filter->SetNumberOfIterations(2);
  EXPECT_THROW(filter->Update(), itk::ExceptionObject);
}

// Jacobian mode with a 2D model sweeps ONE axis of the image per iteration, drawn at random from the seed, chunked
// along it: with the field held at zero (learning rate 0), every iteration's value is the images' features swept
// along one of the three axes and compared whole, PCA-reduced on slices along that axis when asked; the three axes
// all come up, and the same seed draws them in the same order.
TEST(ImpactTorchAdam, TwoDimensionalJacobianSweepsARandomAxisEachIteration)
{
  const ImageType::SizeType size = { { 10, 9, 8 } };
  ImageType::DirectionType  identity;
  identity.SetIdentity();
  auto                                             fixed = MakeNoiseImage(size, identity, 11);
  auto                                             moving = MakeNoiseImage(size, identity, 12);
  itk::ImpactModelConfiguration                    config(std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModel2D.pt",
                                       2,
                                       1,
                                                          { 0, 0 },
                                                          { 0.f, 0.f, 0.f },
                                                          { 0, 0 },
                                                          { true, false },
                                       false);
  const std::vector<itk::ImpactModelConfiguration> configs{ config };
  auto                                             run = [&](unsigned int pca, unsigned int seed) {
    auto filter = TorchAdamFilterType::New();
    filter->SetFixedImage(fixed);
    filter->SetMovingImage(moving);
    filter->AddModelConfiguration(config);
    filter->SetDistance({ "L2" });
    filter->SetMode("Jacobian");
    filter->SetFeatureChunkSize(3);
    filter->SetPCA({ pca });
    filter->NormalizeLossesOff();
    filter->SetLearningRate(0.0);
    filter->SetRegularizationWeight(0.0);
    filter->SetNumberOfIterations(20);
    filter->SetSeed(seed);
    filter->Update();
    return filter->GetMetricValuesPerIteration();
  };
  for (const unsigned int pca : { 0u, 2u })
  {
    std::array<double, 3> expected{};
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
      auto layerOf = [&](ImageType::Pointer image) {
        return itk::Impact::ExtractFeatureLayers<3>(configs,
                                                    itk::Impact::ImageToBatchTensor(image.GetPointer()),
                                                    torch::kCPU,
                                                    {},
                                                    false,
                                                    {},
                                                    static_cast<int>(axis))[0]
          .squeeze(0);
      };
      torch::Tensor f = layerOf(fixed);
      torch::Tensor m = layerOf(moving);
      if (pca > 0)
      {
        torch::Tensor basis;
        f = itk::Impact::PcaReduce(f, basis, pca, 3 - axis);
        m = itk::Impact::PcaReduce(m, basis, pca, 3 - axis);
      }
      expected[axis] = (f - m).pow(2).sum(0).mean().item<double>();
    }
    const std::vector<double> values = run(pca, 5);
    ASSERT_EQ(values.size(), 20u);
    std::array<int, 3> drawn{ 0, 0, 0 };
    for (const double value : values)
    {
      int axis = -1;
      for (int a = 0; a < 3; ++a)
      {
        if (std::abs(value - expected[a]) <= 1e-4 * expected[a])
        {
          axis = a;
        }
      }
      ASSERT_GE(axis, 0) << "PCA " << pca << ": an iteration's value " << value << " is no axis' sweep (" << expected[0]
                         << ", " << expected[1] << ", " << expected[2] << ")";
      ++drawn[axis];
    }
    for (unsigned int a = 0; a < 3; ++a)
    {
      EXPECT_GT(drawn[a], 0) << "PCA " << pca << ": axis " << a << " never drawn in 20 iterations";
    }
    if (pca == 0)
    {
      EXPECT_EQ(run(pca, 5), values) << "the same seed must draw the same axes";
      EXPECT_NE(run(pca, 6), values) << "another seed must draw other axes";
    }
  }
}

TEST(ImpactTorchAdam, FeaturePCA)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->NormalizeLossesOff(); // tolerances set on the raw loss
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false);
  filter->AddModelConfiguration(config);
  filter->SetDistance({ "L2" });
  filter->SetLayersWeight({ 1.f });
  filter->SetPCA({ 2 }); // 4-channel layer -> 2 principal components
  filter->SetNumberOfIterations(250);
  filter->SetLearningRate(0.3);
  filter->SetRegularizationWeight(0.02);
  filter->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto   err = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  const auto & hist = filter->GetMetricValuesPerIteration();
  EXPECT_LT(hist.back(), 0.5 * hist.front()) << "PCA-feature loss should drop";
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 0.3) << "PCA-feature axis " << d << " displacement off";
}

// FeatureMapUpdateInterval periodically re-extracts the moving features from the current warp;
// the refresh path must run without error and a translation must still be recovered.
TEST(ImpactTorchAdam, FeatureMapUpdateInterval)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(20, tx, ty, tz, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->NormalizeLossesOff(); // tolerances set on the raw loss
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false);
  filter->AddModelConfiguration(config);
  filter->SetDistance({ "L2" });
  filter->SetLayersWeight({ 1.f });
  filter->SetFeatureMapUpdateInterval(50); // re-extract moving features every 50 iterations
  filter->SetNumberOfIterations(200);
  filter->SetLearningRate(0.3);
  filter->SetRegularizationWeight(0.08); // scaled with the 4-channel layer's now channel-summed L2
  filter->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto   err = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  const auto & hist = filter->GetMetricValuesPerIteration();
  EXPECT_LT(hist.back(), 0.4 * hist.front()) << "feature loss should drop with periodic refresh";
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 0.3) << "refresh axis " << d << " displacement off";
}

// A model with downsampled feature layers (ImpactToyModelDown halves BOTH layers) must work:
// the layers are upsampled back to the input resolution and a translation is still recovered.
TEST(ImpactTorchAdam, DownsampledFeatureLayers)
{
  const std::string        downModel = std::string(IMPACT_TEST_DATA_DIR) + "/ImpactToyModelDown.pt";
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(24, tx, ty, tz, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ImpactModelConfiguration config(
    downModel, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, true }, false);
  filter->AddModelConfiguration(config);
  filter->SetDistance({ "L2", "L2" });
  filter->SetLayersWeight({ 1.f, 1.f });
  filter->SetNumberOfIterations(250);
  filter->SetLearningRate(0.3);
  filter->SetRegularizationWeight(0.02);
  filter->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto   err = InteriorMeanError(filter->GetDisplacementField(), expected, 5);
  const auto & hist = filter->GetMetricValuesPerIteration();
  EXPECT_LT(hist.back(), 0.6 * hist.front()) << "downsampled-feature loss should drop";
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 0.6) << "downsampled-feature axis " << d << " displacement off (coarse features)";
}

// The low-resolution control grid (GridShrinkFactor>1) optimizes the field on a coarser grid
// and trilinearly upsamples it each iteration; a known translation must still be recovered (a
// constant field is representable at any control-grid resolution), validating the upsample +
// units path and the control-grid smoothing.
TEST(ImpactTorchAdam, LowResControlGrid)
{
  const double             tx = 1.5, ty = -2.0, tz = 1.0;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();

  auto fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(24, tx, ty, tz, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  filter->SetGridShrinkFactor(2);               // optimize a 12^3 control grid
  filter->SetControlGridSmoothingIterations(1); // B-spline-like smoothing each step
  filter->SetNumberOfIterations(300);
  filter->SetLearningRate(0.3);
  filter->SetRegularizationWeight(0.02);
  filter->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto   err = InteriorMeanError(filter->GetDisplacementField(), expected, 6);
  const auto & hist = filter->GetMetricValuesPerIteration();

  EXPECT_LT(hist.back(), 0.1 * hist.front()) << "loss should drop on the low-res control grid";
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 0.3) << "low-res axis " << d << " displacement off";
}

// A GridShrinkFactor large enough to collapse a control-grid axis to size 1 must still produce
// a finite field: the diffusion regularizer must skip the degenerate (no-forward-difference)
// axis rather than averaging over zero elements (NaN).
TEST(ImpactTorchAdam, DegenerateControlGridFinite)
{
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(16, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(16, 1.0, -1.0, 0.5, spacing, identity);

  auto filter = TorchAdamFilterType::New();
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  filter->SetGridShrinkFactor(16); // control grid = 1 voxel on every axis
  filter->SetNumberOfIterations(10);
  filter->SetLearningRate(0.2);
  filter->SetRegularizationWeight(1.0);
  filter->Update();

  bool                                                      finite = true;
  itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType> it(
    filter->GetDisplacementField(), filter->GetDisplacementField()->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    for (unsigned int d = 0; d < 3; ++d)
      if (!std::isfinite(it.Get()[d]))
        finite = false;
  EXPECT_TRUE(finite) << "displacement field must be finite even with a size-1 control grid";
}

// --- ImpactCoarseRegistration: coarse stage feeding the Adam refinement ----------
// Two-stage pipeline on a translation too large for a single confident grid_sample step:
//   (1) the coarse cost-volume + coupled-convex stage must clearly improve over a zero
//       initial field, and (2) the Adam refinement initialized from that coarse field
//       must improve further. Exercises the coarse filter and the warm-start hand-off
//       (geometry-correct round trip ITK field -> voxel tensor -> ITK field).
TEST(ImpactConvexAdam, CoarseInitThenAdamRefines)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;

  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();

  const double tx = 5.0, ty = -4.0, tz = 3.0; // larger shift -> coarse stage matters
  auto         fixed = MakeTorchAdamPattern(32, 0, 0, 0, spacing, identity);
  auto         moving = MakeTorchAdamPattern(32, tx, ty, tz, spacing, identity);

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;

  // Zero-displacement baseline error is just |expected| per axis.
  itk::Vector<double, 3> zeroErr;
  for (unsigned int d = 0; d < 3; ++d)
    zeroErr[d] = std::abs(expected[d]);

  // Stage 1: coarse initializer.
  auto coarse = CoarseType::New();
  coarse->SetFixedImage(fixed);
  coarse->SetMovingImage(moving);
  coarse->SetGridSpacing(2);
  coarse->SetDisplacementHalfWidth(4); // +/- 8 voxels captures the shift
  coarse->Update();
  const auto coarseErr = InteriorMeanError(coarse->GetDisplacementField(), expected, 8);

  // Stage 2: Adam refinement, warm-started from the coarse field.
  auto fine = TorchAdamFilterType::New();
  fine->SetFixedImage(fixed);
  fine->SetMovingImage(moving);
  fine->SetInitialDisplacementField(coarse->GetDisplacementField());
  fine->SetNumberOfIterations(400);
  fine->SetLearningRate(0.1); // warm-started -> small lr converges tight instead of wandering
  fine->SetRegularizationWeight(0.02);
  fine->Update();
  const auto adamErr = InteriorMeanError(fine->GetDisplacementField(), expected, 8);

  double meanZero = 0, meanCoarse = 0, meanAdam = 0;
  for (unsigned int d = 0; d < 3; ++d)
  {
    meanZero += zeroErr[d] / 3.0;
    meanCoarse += coarseErr[d] / 3.0;
    meanAdam += adamErr[d] / 3.0;
  }
  std::cout << "[pipeline] mean |err| zero=" << meanZero << " coarse=" << meanCoarse << " adam=" << meanAdam << "\n"
            << "           coarse per-axis = " << coarseErr << "\n"
            << "           adam   per-axis = " << adamErr << std::endl;

  EXPECT_LT(meanCoarse, 0.6 * meanZero) << "coarse stage should clearly improve over zero displacement";
  EXPECT_LT(meanAdam, meanCoarse) << "Adam refinement should improve further from the coarse init";
  // Final accuracy: tolerance in mm; the z-axis pattern has the weakest amplitude (and so
  // the loosest gradient constraint), hence a per-axis bound rather than sub-0.1.
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(adamErr[d], 0.4) << "refined axis " << d << " displacement off";
}

// Inverse-consistency: the coarse stage also solves the backward problem and symmetrizes the
// two fields. A known translation must still be recovered (forward +t and backward -t are
// already mutual inverses, so symmetrization preserves them) and the path must run.
TEST(ImpactConvexAdam, CoarseInverseConsistency)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  ImageType::SpacingType   spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  const double tx = 4.0, ty = -2.0, tz = 2.0; // all even -> exact at GridSpacing 2
  auto         fixed = MakeTorchAdamPattern(32, 0, 0, 0, spacing, identity);
  auto         moving = MakeTorchAdamPattern(32, tx, ty, tz, spacing, identity);

  auto coarse = CoarseType::New();
  coarse->SetFixedImage(fixed);
  coarse->SetMovingImage(moving);
  coarse->SetGridSpacing(2);
  coarse->SetDisplacementHalfWidth(4);
  coarse->InverseConsistencyOn();
  coarse->Update();

  itk::Vector<double, 3> expected;
  expected[0] = tx;
  expected[1] = ty;
  expected[2] = tz;
  const auto err = InteriorMeanError(coarse->GetDisplacementField(), expected, 8);
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(err[d], 1.5) << "inverse-consistent coarse axis " << d << " off";
}

// A layer's weight scales its share of the coarse SSD cost. A zero weight must give exactly the
// field of the other layer alone (the cost then adds exact zeros for it), and a negative one is
// refused. The toy model's layer 0 is a 4-channel convolution, its layer 1 the image twice.
TEST(ImpactConvexAdam, CoarseLayerWeightScalesItsCost)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  using FieldType = CoarseType::DisplacementFieldType;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(24, 3.0, -2.5, 1.5, spacing, identity);

  auto run = [&](const std::vector<bool> & mask, const std::vector<float> & weights) {
    auto coarse = CoarseType::New();
    coarse->SetFixedImage(fixed);
    coarse->SetMovingImage(moving);
    coarse->AddModelConfiguration(
      itk::ImpactModelConfiguration(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, mask, false));
    coarse->SetLayersWeight(weights);
    coarse->SetGridSpacing(2);
    coarse->SetDisplacementHalfWidth(3);
    coarse->Update();
    return FieldType::Pointer(coarse->GetDisplacementField());
  };
  auto maxDifference = [](const FieldType * a, const FieldType * b) {
    double                                            worst = 0;
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(a, a->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      for (unsigned int d = 0; d < 3; ++d)
        worst = std::max(worst, std::abs(static_cast<double>(it.Get()[d]) - b->GetPixel(it.GetIndex())[d]));
    return worst;
  };

  const auto convolutionOnly = run({ true, false }, {});
  const auto imageOnly = run({ false, true }, {});
  ASSERT_GT(maxDifference(convolutionOnly, imageOnly), 0.0) << "the layers must disagree for a weight to show";
  EXPECT_EQ(maxDifference(run({ true, true }, { 0.f, 1.f }), imageOnly), 0.0) << "weight 0 must drop layer 0";
  EXPECT_EQ(maxDifference(run({ true, true }, { 1.f, 0.f }), convolutionOnly), 0.0) << "weight 0 must drop layer 1";
  EXPECT_THROW(run({ true, true }, { -1.f, 1.f }), itk::ExceptionObject);
}

// Normalized, a layer's cost starts at 1 whatever the range of its features: images ten times brighter give the
// toy model's image layer (the image twice) a cost a hundred times larger, and the same field. Unnormalized, the
// coupling schedule, whose coefficients are absolute, weighs that cost differently and the field moves.
TEST(ImpactConvexAdam, NormalizedCoarseCostIgnoresTheFeatureRange)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  using FieldType = CoarseType::DisplacementFieldType;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto scaled = [&](double factor, double tx, double ty, double tz) {
    auto image = MakeTorchAdamPattern(24, tx, ty, tz, spacing, identity);
    itk::ImageRegionIteratorWithIndex<ImageType> it(image, image->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      it.Set(static_cast<float>(it.Get() * factor));
    return image;
  };
  auto run = [&](double factor, bool normalize) {
    auto coarse = CoarseType::New();
    coarse->SetFixedImage(scaled(factor, 0, 0, 0));
    coarse->SetMovingImage(scaled(factor, 3.0, -2.5, 1.5));
    coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { false, true }, false));
    coarse->SetNormalizeLosses(normalize);
    coarse->SetGridSpacing(2);
    coarse->SetDisplacementHalfWidth(3);
    coarse->Update();
    return FieldType::Pointer(coarse->GetDisplacementField());
  };
  auto meanDifference = [](const FieldType * a, const FieldType * b) {
    double                                             sum = 0;
    long                                               n = 0;
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(a, a->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++n)
      for (unsigned int d = 0; d < 3; ++d)
        sum += std::abs(static_cast<double>(it.Get()[d]) - b->GetPixel(it.GetIndex())[d]);
    return sum / std::max<long>(n, 1);
  };

  ASSERT_GT(meanDifference(run(1.0, false), run(10.0, false)), 1e-3) << "unnormalized, the range must show";
  EXPECT_LT(meanDifference(run(1.0, true), run(10.0, true)), 1e-4) << "normalized, it must not";
}

// The coarse cost volume compares each layer with its distance, over the cell window around every coarse voxel --
// the two 3^3 box passes ConvexAdam smooths its cost with --, for every candidate displacement of the edge-replicated
// moving features, in a box of its own half-width per axis. Checked for every distance against loops in double
// precision: a point-wise distance's terms averaged over the window, NCC and LNCC correlated over it from the
// window's weighted moments.
TEST(ImpactConvexAdam, CoarseCostVolumeMatchesABruteForceForEveryDistance)
{
  const int64_t              C = 3, Z = 5, Y = 4, X = 6;
  const std::vector<int64_t> halfWidths{ 1, 2, 1 }; // z, y, x
  const int64_t              sx = 3, sy = 5, L = 45;
  torch::manual_seed(3);
  const torch::Tensor fixed = torch::rand({ 1, C, Z, Y, X }) + 0.1;
  const torch::Tensor moving = torch::rand({ 1, C, Z, Y, X }) + 0.1;
  auto                F = fixed.accessor<float, 5>();
  auto                M = moving.accessor<float, 5>();
  using Volume = std::vector<double>; // z, y, x
  auto at = [&](int64_t z, int64_t y, int64_t x) { return (z * Y + y) * X + x; };
  // One zero-padded 3^3 box pass (the voxels outside the volume count as 0, and every window divides by 27).
  auto box = [&](const Volume & q) {
    Volume out(q.size(), 0.0);
    for (int64_t z = 0; z < Z; ++z)
      for (int64_t y = 0; y < Y; ++y)
        for (int64_t x = 0; x < X; ++x)
        {
          double sum = 0;
          for (int64_t dz = -1; dz <= 1; ++dz)
            for (int64_t dy = -1; dy <= 1; ++dy)
              for (int64_t dx = -1; dx <= 1; ++dx)
              {
                const int64_t u = z + dz, v = y + dy, w = x + dx;
                if (u >= 0 && u < Z && v >= 0 && v < Y && w >= 0 && w < X)
                  sum += q[at(u, v, w)];
              }
          out[at(z, y, x)] = sum / 27.0;
        }
    return out;
  };
  auto window = [&](const Volume & q) { return box(box(q)); };
  auto clampTo = [](int64_t i, int64_t n) { return std::min<int64_t>(std::max<int64_t>(i, 0), n - 1); };

  for (const std::string name : { "L1", "L2", "Cosine", "L1Cosine", "Dice", "NCC", "LNCC" })
  {
    const auto    distance = itk::Impact::LossFactory::Instance().Create(name);
    torch::Tensor cost = torch::zeros({ L, Z, Y, X });
    itk::Impact::AccumulateCoarseCost<3>(cost, *distance, fixed, moving, halfWidths, 0.5);
    double worst = 0;
    for (int64_t l = 0; l < L; ++l)
    {
      // The candidate's offsets, x fastest.
      const int64_t dx = l % sx - halfWidths[2], dy = (l / sx) % sy - halfWidths[1], dz = l / (sx * sy) - halfWidths[0];
      auto          m = [&](int64_t c, int64_t z, int64_t y, int64_t x) {
        return static_cast<double>(M[0][c][clampTo(z + dz, Z)][clampTo(y + dy, Y)][clampTo(x + dx, X)]);
      };
      auto   f = [&](int64_t c, int64_t z, int64_t y, int64_t x) { return static_cast<double>(F[0][c][z][y][x]); };
      Volume expected(Z * Y * X, 0.0);
      if (name == "NCC" || name == "LNCC")
      {
        const Volume weight = window(Volume(Z * Y * X, 1.0));
        for (int64_t c = 0; c < C; ++c)
        {
          Volume vf(Z * Y * X), vm(Z * Y * X), vff(Z * Y * X), vmm(Z * Y * X), vfm(Z * Y * X);
          for (int64_t z = 0; z < Z; ++z)
            for (int64_t y = 0; y < Y; ++y)
              for (int64_t x = 0; x < X; ++x)
              {
                const double  a = f(c, z, y, x), b = m(c, z, y, x);
                const int64_t i = at(z, y, x);
                vf[i] = a;
                vm[i] = b;
                vff[i] = a * a;
                vmm[i] = b * b;
                vfm[i] = a * b;
              }
          const Volume mf = window(vf), mm = window(vm), mff = window(vff), mmm = window(vmm), mfm = window(vfm);
          for (size_t i = 0; i < expected.size(); ++i)
          {
            const double w = weight[i];
            const double meanF = mf[i] / w, meanM = mm[i] / w;
            const double covariance = mfm[i] / w - meanF * meanM;
            const double varianceF = mff[i] / w - meanF * meanF, varianceM = mmm[i] / w - meanM * meanM;
            const double correlation =
              name == "NCC"
                ? covariance / std::max(std::sqrt(std::max(varianceF, 0.0)) * std::sqrt(std::max(varianceM, 0.0)), 1e-8)
                : covariance * covariance / (std::max(varianceF, 1e-5) * std::max(varianceM, 1e-5));
            expected[i] += (1.0 - correlation) / C;
          }
        }
      }
      else
      {
        Volume terms(Z * Y * X, 0.0);
        for (int64_t z = 0; z < Z; ++z)
          for (int64_t y = 0; y < Y; ++y)
            for (int64_t x = 0; x < X; ++x)
            {
              double l1 = 0, l2 = 0, dot = 0, ff = 0, mm = 0, sum = 0;
              for (int64_t c = 0; c < C; ++c)
              {
                const double a = f(c, z, y, x), b = m(c, z, y, x);
                l1 += std::abs(a - b);
                l2 += (a - b) * (a - b);
                dot += a * b;
                ff += a * a;
                mm += b * b;
                sum += a + b;
              }
              const double cosine = dot / std::sqrt(ff * mm);
              double       damped = 0;
              for (int64_t c = 0; c < C; ++c)
                damped += cosine * std::exp(-0.1 * std::abs(f(c, z, y, x) - m(c, z, y, x))) / C;
              terms[at(z, y, x)] = name == "L1"       ? l1
                                   : name == "L2"     ? l2
                                   : name == "Cosine" ? 1.0 - cosine
                                   : name == "Dice"   ? 1.0 - 2.0 * dot / sum
                                                      : 1.0 - damped;
            }
        expected = window(terms);
      }
      for (int64_t z = 0; z < Z; ++z)
        for (int64_t y = 0; y < Y; ++y)
          for (int64_t x = 0; x < X; ++x)
            worst = std::max(worst, std::abs(cost[l][z][y][x].item<double>() - 0.5 * expected[at(z, y, x)]));
    }
    EXPECT_LT(worst, 1e-5) << name;
  }
}

// Every distance finds the translation between two patterns in the coarse stage.
TEST(ImpactConvexAdam, CoarseRecoversATranslationWithEveryDistance)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto                   fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto                   moving = MakeTorchAdamPattern(24, 3.0, -2.5, 1.5, spacing, identity);
  itk::Vector<double, 3> expected;
  expected[0] = 3.0;
  expected[1] = -2.5;
  expected[2] = 1.5;
  for (const std::string name : { "L1", "L2", "Cosine", "L1Cosine", "NCC", "LNCC" })
  {
    auto coarse = CoarseType::New();
    coarse->SetFixedImage(fixed);
    coarse->SetMovingImage(moving);
    coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false));
    coarse->SetDistance({ name });
    coarse->SetGridSpacing(2);
    coarse->SetDisplacementHalfWidth(3);
    coarse->Update();
    const auto error = InteriorMeanError(coarse->GetDisplacementField(), expected, 8);
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_LT(error[d], 1.5) << name << ", axis " << d;
  }
}

// ConvexAdam's knobs are counted in voxels of the fixed image's finest axis, s_min, and derived per axis so that they
// are (nearly) isotropic in millimetres: on an isotropic image every axis keeps the numbers as they are, whatever the
// spacing; on a 1 x 1 x 3 mm or a 1 x 1 x 2.5 mm image the thick axis takes fewer voxels. All in tensor order (z, y,
// x).
TEST(ImpactConvexAdam, KnobsAreCountedInUnitsOfTheFinestVoxelSide)
{
  using Counts = std::vector<int64_t>;
  for (const double side : { 1.0, 2.0, 0.7 })
  {
    const std::vector<double> sides{ side, side, side };
    EXPECT_EQ(itk::Impact::IsotropicVoxelCounts(sides, 6), (Counts{ 6, 6, 6 })) << side << " mm";
    EXPECT_EQ(itk::Impact::CaptureHalfWidths(sides, { 6, 6, 6 }, 6, 4), (Counts{ 4, 4, 4 })) << side << " mm";
    EXPECT_EQ(itk::Impact::IsotropicWindow(sides, { 24, 24, 24 }, { 1, 2, 12, 12, 12 }, 5), (Counts{ 5, 5, 5 }));
  }
  ImageType::SpacingType thick;
  thick[0] = 1.0;
  thick[1] = 1.0;
  thick[2] = 3.0;
  const std::vector<double> sides = itk::Impact::TensorVoxelSides<3>(thick);
  EXPECT_EQ(sides, (std::vector<double>{ 3.0, 1.0, 1.0 }));
  const Counts cells = itk::Impact::IsotropicVoxelCounts(sides, 6); // coarse cells of 6 mm
  EXPECT_EQ(cells, (Counts{ 2, 6, 6 }));
  EXPECT_EQ(itk::Impact::CaptureHalfWidths(sides, cells, 6, 4), (Counts{ 4, 4, 4 }))
    << "24 mm each way: 9^3 candidates";
  EXPECT_EQ(itk::Impact::IsotropicVoxelCounts(sides, 2), (Counts{ 1, 2, 2 })) << "the fine control grid's shrink";
  const std::vector<double> sides25{ 2.5, 1.0, 1.0 };
  const Counts              cells25 = itk::Impact::IsotropicVoxelCounts(sides25, 6); // round(2.4) = 2: 5 mm along z
  EXPECT_EQ(cells25, (Counts{ 2, 6, 6 }));
  EXPECT_EQ(itk::Impact::CaptureHalfWidths(sides25, cells25, 6, 4), (Counts{ 5, 4, 4 }))
    << "24 mm: 5 cells of 5 mm along z, 4 of 6 mm in-plane, 11 x 9 x 9 candidates";
  // The LNCC window, on a map at the image's resolution and on one pooled to the fine grid of shrink (1, 2, 2).
  EXPECT_EQ(itk::Impact::IsotropicWindow(sides, { 24, 24, 24 }, { 1, 2, 24, 24, 24 }, 5), (Counts{ 1, 5, 5 }));
  EXPECT_EQ(itk::Impact::IsotropicWindow(sides, { 24, 24, 24 }, { 1, 2, 24, 12, 12 }, 5), (Counts{ 3, 5, 5 }));
}

// On an isotropic image, the coarse and the fine stage work in voxels whatever the spacing: the same images at 2 mm
// give exactly twice the field in millimetres they give at 1 mm, through every knob (cells, capture range, shrink,
// smoothing, learning rate, regularization, the LNCC window) and the warm start.
TEST(ImpactConvexAdam, IsotropicSpacingOnlyScalesTheField)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  using FieldType = CoarseType::DisplacementFieldType;
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto run = [&](double side) {
    ImageType::SpacingType spacing;
    spacing.Fill(side);
    auto fixed = MakeTorchAdamPattern(20, 0, 0, 0, spacing, identity);
    auto moving = MakeTorchAdamPattern(20, 2.5, -1.5, 1.0, spacing, identity);
    auto coarse = CoarseType::New();
    coarse->SetFixedImage(fixed);
    coarse->SetMovingImage(moving);
    coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, true }, false));
    coarse->SetDistance({ "L2", "NCC" });
    coarse->SetGridSpacing(2);
    coarse->SetDisplacementHalfWidth(3);
    coarse->Update();
    auto fine = TorchAdamFilterType::New();
    fine->SetFixedImage(fixed);
    fine->SetMovingImage(moving);
    fine->SetInitialDisplacementField(coarse->GetDisplacementField());
    fine->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, true }, false));
    fine->SetDistance({ "L2", "LNCC" });
    fine->SetGridShrinkFactor(2);
    fine->SetControlGridSmoothingIterations(2);
    fine->SetNumberOfIterations(15);
    fine->SetLearningRate(0.5);
    fine->SetRegularizationWeight(1.0);
    fine->Update();
    return std::make_pair(FieldType::Pointer(coarse->GetDisplacementField()),
                          FieldType::Pointer(fine->GetDisplacementField()));
  };
  const auto [coarse1, fine1] = run(1.0);
  const auto [coarse2, fine2] = run(2.0);
  for (const auto & [one, two] : { std::make_pair(coarse1, coarse2), std::make_pair(fine1, fine2) })
  {
    long                                              differing = 0;
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(one, one->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      for (unsigned int d = 0; d < 3; ++d)
        differing += 2.0f * it.Get()[d] != two->GetPixel(it.GetIndex())[d];
    EXPECT_EQ(differing, 0) << (one == coarse1 ? "coarse" : "fine") << " field";
  }
}

// On a 1 x 1 x 3 mm image, the coarse and the fine stage recover a translation along the thick axis as well as
// in-plane, in millimetres.
TEST(ImpactConvexAdam, AnisotropicSpacingRecoversADisplacementAlongEveryAxis)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  ImageType::DirectionType identity;
  identity.SetIdentity();
  ImageType::SpacingType spacing;
  spacing[0] = 1.0;
  spacing[1] = 1.0;
  spacing[2] = 3.0;
  // 3 mm along x, -2 mm along y, 6 mm (2 voxels) along z.
  auto                   fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto                   moving = MakeTorchAdamPattern(24, 3.0, -2.0, 2.0, spacing, identity);
  itk::Vector<double, 3> expected;
  expected[0] = 3.0;
  expected[1] = -2.0;
  expected[2] = 6.0;
  const itk::ImpactModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 0.f, 0.f, 0.f }, { 0, 0, 0 }, { true, false }, false);
  auto coarse = CoarseType::New();
  coarse->SetFixedImage(fixed);
  coarse->SetMovingImage(moving);
  coarse->AddModelConfiguration(config);
  coarse->SetGridSpacing(2);
  coarse->SetDisplacementHalfWidth(4);
  coarse->Update();
  const auto coarseError = InteriorMeanError(coarse->GetDisplacementField(), expected, 6);
  auto       fine = TorchAdamFilterType::New();
  fine->SetFixedImage(fixed);
  fine->SetMovingImage(moving);
  fine->SetInitialDisplacementField(coarse->GetDisplacementField());
  fine->AddModelConfiguration(config);
  fine->SetDistance({ "L2" });
  fine->SetGridShrinkFactor(2);
  fine->SetControlGridSmoothingIterations(1);
  fine->SetNumberOfIterations(100);
  fine->SetLearningRate(0.2);
  fine->SetRegularizationWeight(0.1);
  fine->Update();
  const auto fineError = InteriorMeanError(fine->GetDisplacementField(), expected, 6);
  for (unsigned int d = 0; d < 3; ++d)
  {
    EXPECT_LT(coarseError[d], 1.5) << "coarse, axis " << d << " (mm)";
    EXPECT_LT(fineError[d], 0.6) << "fine, axis " << d << " (mm)";
  }
}

// PCA and the channel subset reach the coarse stage too: a subset drawn from the seeded generator gives the same
// field twice, a subset of every channel changes nothing, and a PCA as wide as the layer changes nothing either.
TEST(ImpactConvexAdam, CoarseTakesPCAAndASeededChannelSubset)
{
  using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
  using FieldType = CoarseType::DisplacementFieldType;
  ImageType::SpacingType spacing;
  spacing.Fill(1.0);
  ImageType::DirectionType identity;
  identity.SetIdentity();
  auto fixed = MakeTorchAdamPattern(24, 0, 0, 0, spacing, identity);
  auto moving = MakeTorchAdamPattern(24, 3.0, -2.5, 1.5, spacing, identity);
  auto run = [&](std::vector<unsigned int> subset, std::vector<unsigned int> pca) {
    auto coarse = CoarseType::New();
    coarse->SetFixedImage(fixed);
    coarse->SetMovingImage(moving);
    coarse->AddModelConfiguration(itk::ImpactModelConfiguration(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false));
    coarse->SetSubsetFeatures(subset);
    coarse->SetPCA(pca);
    coarse->SetSeed(7);
    coarse->SetGridSpacing(2);
    coarse->SetDisplacementHalfWidth(3);
    coarse->Update();
    return FieldType::Pointer(coarse->GetDisplacementField());
  };
  auto maxDifference = [](const FieldType * a, const FieldType * b) {
    double                                             worst = 0;
    itk::ImageRegionConstIteratorWithIndex<FieldType> it(a, a->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      for (unsigned int d = 0; d < 3; ++d)
        worst = std::max(worst, std::abs(static_cast<double>(it.Get()[d]) - b->GetPixel(it.GetIndex())[d]));
    return worst;
  };
  const auto all = run({}, {});
  EXPECT_EQ(maxDifference(run({ 2 }, {}), run({ 2 }, {})), 0.0) << "a seeded subset is reproducible";
  EXPECT_EQ(maxDifference(run({ 4 }, {}), all), 0.0) << "every channel = no subset";
  EXPECT_EQ(maxDifference(run({}, { 4 }), all), 0.0) << "a PCA as wide as the layer = none";
  itk::Vector<double, 3> expected;
  expected[0] = 3.0;
  expected[1] = -2.5;
  expected[2] = 1.5;
  const auto error = InteriorMeanError(run({}, { 2 }), expected, 8);
  for (unsigned int d = 0; d < 3; ++d)
    EXPECT_LT(error[d], 1.5) << "PCA to 2 components, axis " << d;
}

namespace
{
// NCC of two same-grid images over the interior (margin away from borders).
double
InteriorNCC(const ImageType * a, const ImageType * b, int margin)
{
  const auto                                            size = a->GetLargestPossibleRegion().GetSize();
  double                                                sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  long                                                  n = 0;
  itk::ImageRegionConstIteratorWithIndex<ImageType> it(a, a->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto idx = it.GetIndex();
    bool       interior = true;
    for (unsigned int d = 0; d < 3; ++d)
      if (idx[d] < margin || idx[d] >= static_cast<long>(size[d]) - margin)
        interior = false;
    if (!interior)
      continue;
    const double va = it.Get();
    const double vb = b->GetPixel(idx);
    sa += va;
    sb += vb;
    saa += va * va;
    sbb += vb * vb;
    sab += va * vb;
    ++n;
  }
  if (n == 0)
    return 0.0;
  const double num = n * sab - sa * sb;
  const double den = std::sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
  return (den > 0) ? num / den : 0.0;
}

// Resample 'moving' onto the 'reference' grid with an identity transform (the unregistered baseline).
ImageType::Pointer
ResampleOnto(const ImageType * moving, const ImageType * reference)
{
  using ResampleType = itk::ResampleImageFilter<ImageType, ImageType, double>;
  auto r = ResampleType::New();
  r->SetInput(moving);
  r->SetTransform(itk::IdentityTransform<double, 3>::New());
  r->SetUseReferenceImage(true);
  r->SetReferenceImage(const_cast<ImageType *>(reference));
  r->Update();
  return r->GetOutput();
}

// Fraction of interior voxels whose deformation (identity + field) Jacobian determinant <= 0.
double
FoldedFraction(const TorchAdamFieldType * field, int margin)
{
  const auto size = field->GetLargestPossibleRegion().GetSize();
  const auto spacing = field->GetSpacing();
  long       folded = 0, total = 0;
  itk::ImageRegionConstIteratorWithIndex<TorchAdamFieldType> it(field, field->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto idx = it.GetIndex();
    bool       interior = true;
    for (unsigned int d = 0; d < 3; ++d)
      if (idx[d] < std::max(1, margin) || idx[d] >= static_cast<long>(size[d]) - std::max(1, margin))
        interior = false;
    if (!interior)
      continue;
    double jac[3][3];
    for (unsigned int r = 0; r < 3; ++r)
      for (unsigned int c = 0; c < 3; ++c)
      {
        TorchAdamFieldType::IndexType ip = idx, im = idx;
        ip[c] += 1;
        im[c] -= 1;
        const double du = static_cast<double>(field->GetPixel(ip)[r]) - static_cast<double>(field->GetPixel(im)[r]);
        jac[r][c] = (r == c ? 1.0 : 0.0) + du / (2.0 * spacing[c]);
      }
    const double det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1]) -
                       jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0]) +
                       jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
    if (det <= 0)
      ++folded;
    ++total;
  }
  return total ? static_cast<double>(folded) / total : 0.0;
}
} // namespace

// End-to-end on the committed real lung-CT pair: coarse (intensity SSD) warm-start -> Adam
// refinement on M258 deep features. No synthetic ground truth, so assert IMPROVEMENT +
// REGULARITY: the feature loss drops, the warped moving correlates better with the fixed than
// the unregistered moving, and the deformation has negligible folding. This is the load-bearing
// validation on real anatomy (anisotropic spacing, a genuine inter-scan deformation) that the
// synthetic constant-translation tests cannot provide.
TEST(ImpactConvexAdam, RealLungCTCoarseInitThenAdamRefines)
{
  // These real-CT inputs are heavy binaries kept out of git. ExternalData fetches them into
  // the build tree (IMPACT_EXTERNAL_DATA_DIR) from the committed .sha512 links; prefer that
  // copy, fall back to a local copy in the source Data dir, and skip (rather than fail) if
  // neither is available. See test/Data/README.md.
  const auto resolveRealCT = [](const std::string & name) -> std::string {
    const std::string fetched = std::string(IMPACT_EXTERNAL_DATA_DIR) + "/" + name;
    if (std::ifstream(fetched).good())
      return fetched;
    const std::string local = std::string(IMPACT_TEST_DATA_DIR) + "/" + name;
    if (std::ifstream(local).good())
      return local;
    return {};
  };
  const std::string baselinePath = resolveRealCT("3DCT_lung_baseline.mha");
  const std::string followupPath = resolveRealCT("3DCT_lung_followup.mha");
  const std::string modelPath = resolveRealCT("M258_2_Layers.pt");
  if (baselinePath.empty() || followupPath.empty() || modelPath.empty())
    GTEST_SKIP() << "real-CT test data not available";

  using ReaderType = itk::ImageFileReader<ImageType>;
  auto fr = ReaderType::New();
  fr->SetFileName(baselinePath);
  fr->SetImageIO(itk::MetaImageIO::New()); // the GTest driver does not auto-register IO factories
  auto mr = ReaderType::New();
  mr->SetFileName(followupPath);
  mr->SetImageIO(itk::MetaImageIO::New());

  const bool         haveCuda = torch::cuda::is_available();
  const std::string  device = haveCuda ? "cuda:0" : "cpu";
  const unsigned int shrink = haveCuda ? 2u : 3u;

  using ShrinkType = itk::ShrinkImageFilter<ImageType, ImageType>;
  auto fs = ShrinkType::New();
  fs->SetInput(fr->GetOutput());
  fs->SetShrinkFactors(shrink);
  fs->Update();
  auto ms = ShrinkType::New();
  ms->SetInput(mr->GetOutput());
  ms->SetShrinkFactors(shrink);
  ms->Update();
  ImageType::Pointer fixed = fs->GetOutput();
  ImageType::Pointer moving = ms->GetOutput();

  // Coarse stage: intensity SSD cost volume (scale-invariant, cheap, captures the gross shift).
  auto coarse = itk::ImpactCoarseRegistration<ImageType>::New();
  coarse->SetFixedImage(fixed);
  coarse->SetMovingImage(moving);
  coarse->SetDevice(device);
  coarse->SetGridSpacing(4);
  coarse->SetDisplacementHalfWidth(5);
  coarse->Update();

  // Fine stage: Adam refinement on the full-resolution 32-channel M258 feature layer.
  auto fine = TorchAdamFilterType::New();
  fine->SetFixedImage(fixed);
  fine->SetMovingImage(moving);
  fine->SetDevice(device);
  fine->SetInitialDisplacementField(coarse->GetDisplacementField());
  itk::ImpactModelConfiguration cfg(
    modelPath, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, { 0, 0, 0 }, { true, false }, false);
  fine->AddModelConfiguration(cfg);
  fine->SetDistance({ "L2" });
  fine->SetLayersWeight({ 1.f });
  fine->SetSubsetFeatures({ 0, 1, 2, 3, 4, 5, 6, 7 }); // 8 of 32 channels keeps CPU runtime sane
  fine->SetNumberOfIterations(haveCuda ? 150u : 60u);
  fine->SetLearningRate(0.1); // warm-started -> small lr
  fine->SetRegularizationWeight(0.5);
  fine->Update();

  const auto & hist = fine->GetMetricValuesPerIteration();
  ASSERT_FALSE(hist.empty());

  auto         movingOnFixed = ResampleOnto(moving, fixed);
  const double s0 = InteriorNCC(fixed, movingOnFixed, 4);
  const double s1 = InteriorNCC(fixed, fine->GetWarpedMovingImage(), 4);
  const double folded = FoldedFraction(fine->GetDisplacementField(), 2);

  std::cout << "[real-CT] device=" << device << " shrink=" << shrink << " loss " << hist.front() << " -> "
            << hist.back() << " | NCC moving=" << s0 << " warped=" << s1 << " | foldedFraction=" << folded
            << std::endl;

  EXPECT_LT(hist.back(), 0.85 * hist.front()) << "feature loss should drop";
  EXPECT_GT(s1, s0 + 0.02) << "warped moving should correlate better with fixed than the unregistered moving";
  EXPECT_LT(folded, 0.05) << "deformation should have negligible folding";
}

// 2D coarse initializer: a known translation must be recovered to the coarse-grid resolution,
// exercising the dimension-generic cost-volume / coupled_convex / upsample (avg_pool2d, bilinear).
TEST(ImpactConvexAdam, Coarse2D)
{
  using Image2D = itk::Image<float, 2>;
  using Coarse2D = itk::ImpactCoarseRegistration<Image2D>;
  const unsigned int n = 40;
  const double       tx = 4.0, ty = -3.0; // index shift (x, y)

  auto make = [&](double sx, double sy) {
    auto                img = Image2D::New();
    Image2D::SizeType   size;
    size.Fill(n);
    Image2D::RegionType region;
    region.SetSize(size);
    img->SetRegions(region);
    img->Allocate();
    const double                               pi = 4.0 * std::atan(1.0);
    itk::ImageRegionIteratorWithIndex<Image2D> it(img, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      const auto   i = it.GetIndex();
      const double x = i[0] - sx, y = i[1] - sy;
      it.Set(static_cast<float>(std::sin(2 * pi * (1.3 * x / n)) + std::sin(2 * pi * (1.3 * y / n + 0.3))));
    }
    return img;
  };
  auto fixed = make(0, 0);
  auto moving = make(tx, ty);

  auto coarse = Coarse2D::New();
  coarse->SetFixedImage(fixed);
  coarse->SetMovingImage(moving);
  coarse->SetGridSpacing(2);
  coarse->SetDisplacementHalfWidth(4);
  coarse->Update();

  auto         field = coarse->GetDisplacementField();
  const auto   size = field->GetLargestPossibleRegion().GetSize();
  double       ex = 0, ey = 0;
  long         cnt = 0;
  itk::ImageRegionConstIteratorWithIndex<Coarse2D::DisplacementFieldType> it(field, field->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto i = it.GetIndex();
    if (i[0] < 8 || i[0] >= static_cast<long>(size[0]) - 8 || i[1] < 8 || i[1] >= static_cast<long>(size[1]) - 8)
      continue;
    ex += std::abs(static_cast<double>(it.Get()[0]) - tx);
    ey += std::abs(static_cast<double>(it.Get()[1]) - ty);
    ++cnt;
  }
  ex /= cnt;
  ey /= cnt;
  std::cout << "[2D coarse] mean|err| = (" << ex << ", " << ey << ") expected (" << tx << ", " << ty << ")\n";
  EXPECT_LT(ex, 1.5) << "2D coarse x displacement off";
  EXPECT_LT(ey, 1.5) << "2D coarse y displacement off";
}

// The driver provides its own main so that it always wins over the main() that
// ITK's vendored NrrdIO (sampleIO.c) exports from the shared ITK libraries; a
// direct object's main takes precedence over a shared-library-exported one.
int
main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

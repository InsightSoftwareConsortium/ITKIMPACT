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

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkImageToTensorFilter.h"
#include "itkModelConfiguration.h"
#include "itkImageToFeaturesMap.h"
#include "itkImageToFeaturesMapInternals.h"
#include "ImpactLoss.h"
#include "itkImpactImageToImageMetricv4.h"
#include "itkImpactFineRegistration.h"
#include "itkImpactCoarseRegistration.h"
#include "itkImageFileReader.h"
#include "itkMetaImageIO.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
#include <torch/torch.h>
#include "itkAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkDisplacementFieldTransform.h"
#include "itkTranslationTransform.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
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
  for (const char * name : { "L1", "L2", "Dice", "L1Cosine", "Cosine", "DotProduct", "NCC" })
  {
    std::unique_ptr<itk::Impact::Loss> loss;
    ASSERT_NO_THROW(loss = itk::Impact::LossFactory::Instance().Create(name)) << name;
    EXPECT_NE(loss, nullptr) << name;
  }
  EXPECT_THROW(itk::Impact::LossFactory::Instance().Create("DoesNotExist"), std::runtime_error);
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
    EXPECT_NEAR(dice->GetValue(4.0), -1.0, 1e-5);
  }
  // Empty/empty (all zeros) => finite value and gradient (no 0/0 NaN).
  {
    auto          dice = itk::Impact::LossFactory::Instance().Create("Dice");
    dice->SetNumberOfParameters(1);
    torch::Tensor f = torch::zeros({ 4, 3 });
    torch::Tensor m = torch::zeros({ 4, 3 });
    torch::Tensor grad = dice->updateValueAndGetGradientModulator(f, m);
    EXPECT_NEAR(dice->GetValue(4.0), -1.0, 1e-5);
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
  itk::ModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, { true, true }, /*mixedPrecision*/ false);

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
  itk::ModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, true }, false);
  auto filter = FeatMapType::New();
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
    itk::ModelConfiguration config(
      ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, { true, true }, false);
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

  itk::ModelConfiguration config(
    ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, { true, true }, false);

  auto filter = FeatMapType::New();
  filter->SetModelConfiguration(config);
  filter->SetInterpolator(interpolator);
  filter->AddInput(image);
  filter->SetPCA(0);
  filter->SetDevice("cuda:0"); // model is moved to the GPU by A2
  ASSERT_NO_THROW(filter->Update());
  EXPECT_EQ(filter->GetOutput(0)->GetNumberOfComponentsPerPixel(), 4u); // conv features
}

// --- B: the Static-mode metric value is ~0 for identical images, > 0 otherwise
TEST(ImpactMetric, StaticValueZeroForIdenticalPositiveForDifferent)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeRampImage(8);

  auto buildAndEvaluate = [&](ImageType::Pointer moving) -> double {
    auto                                 metric = MetricType::New();
    std::vector<itk::ModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         2,
                         std::vector<bool>{ true, true },
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
  EXPECT_NEAR(valueSame, -1.0, 1e-3) << "identical features => cosine 1 => value -1";

  const double valueDifferent = buildAndEvaluate(MakeRampImage(8, 1));
  EXPECT_GT(valueDifferent, valueSame) << "a different image is less similar => higher (less negative) loss";
  EXPECT_LE(valueDifferent, 1.0 + 1e-6);
}

// --- C: the Static-mode derivative matches finite differences of the value ----
TEST(ImpactMetric, StaticDerivativeMatchesFiniteDifferences)
{
  using VirtualImageType = itk::Image<double, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType, VirtualImageType, double>;

  auto fixed = MakeRampImage(8, 0);
  auto moving = MakeRampImage(8, 1); // different => non-trivial gradient

  auto checkLoss = [&](const char * lossName) {
    SCOPED_TRACE(lossName);
    auto                                 metric = MetricType::New();
    std::vector<itk::ModelConfiguration> configs;
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         2,
                         std::vector<bool>{ true, true },
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

  checkLoss("DotProduct"); // simple gradient: -f
  checkLoss("Cosine");     // d(-cosine)/dm with the full dot-product cross term
  checkLoss("NCC");        // cross-point statistics path (closed-form derivative)
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
  std::vector<itk::ModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       2,
                       std::vector<bool>{ true, true },
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "DotProduct" });
  metric->SetLayersWeight({ 1.f });
  metric->SetSubsetFeatures({ 4 });
  metric->SetPCA({ 0 });
  metric->SetMode("Static");
  metric->SetSeed(1);
  metric->SetFeaturesMapUpdateInterval(-1);
  metric->SetDevice("cpu");

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
    SCOPED_TRACE(lossName);
    auto                                 metric = MetricType::New();
    std::vector<itk::ModelConfiguration> configs;
    // A strictly positive 3^3 patch at the image spacing: the toy conv (kernel 3,
    // padding 1) computes the exact neighborhood feature at the patch center.
    configs.emplace_back(ToyModelPath(),
                         3,
                         1,
                         std::vector<unsigned int>{ 3, 3, 3 },
                         std::vector<float>{ 1.f, 1.f, 1.f },
                         0,
                         std::vector<bool>{ true, true },
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

  checkLoss("DotProduct"); // simplest: linear feature, modulator = -fixed
  checkLoss("Cosine");     // normalized cross term
  checkLoss("NCC");        // cross-point statistics path
}

// --- End-to-end: a real ITK optimizer reduces the metric and recovers a shift -
TEST(ImpactMetric, TranslationRegistrationConverges)
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

  auto                                 metric = MetricType::New();
  std::vector<itk::ModelConfiguration> configs;
  configs.emplace_back(ToyModelPath(),
                       3,
                       1,
                       std::vector<unsigned int>{ 0, 0, 0 },
                       std::vector<float>{ 1.f, 1.f, 1.f },
                       2,
                       std::vector<bool>{ true, true },
                       false);
  metric->SetModelsConfiguration(configs);
  metric->SetDistance({ "NCC" });
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
  EXPECT_LT(finalValue, initialValue) << "optimization should reduce the metric";

  // The optimizer recovers the true geometric alignment: bringing the moving blob
  // (x=10) onto the fixed one (x=6) is a +4 voxel translation in x, with no y/z motion.
  const TransformType::ParametersType recovered = transform->GetParameters();
  EXPECT_NEAR(recovered[0], 4.0, 0.5) << "recovered x-translation should reach the +4 alignment";
  EXPECT_LT(std::abs(recovered[1]), 0.2) << "motion should stay on the x axis";
  EXPECT_LT(std::abs(recovered[2]), 0.2) << "motion should stay on the x axis";
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
    itk::ModelConfiguration cfg(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, mask, false);
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
  itk::ModelConfiguration cfg(downModel, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 2, { true, true }, false);
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
TEST(ImpactLoss, ForwardValueGradientFiniteDifference)
{
  torch::manual_seed(7);
  const int64_t N = 48, C = 6;
  const auto    opts = torch::TensorOptions().dtype(torch::kFloat64);
  for (const std::string name : { "L1", "L2", "Cosine", "DotProduct", "L1Cosine", "Dice", "NCC" })
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
  itk::ModelConfiguration config(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, true }, false);
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
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ModelConfiguration config(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, false }, false);
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
  filter->SetFixedImage(fixed);
  filter->SetMovingImage(moving);
  itk::ModelConfiguration config(ToyModelPath(), 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, false }, false);
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
  itk::ModelConfiguration config(downModel, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, true }, false);
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
  itk::ModelConfiguration cfg(modelPath, 3, 1, { 0, 0, 0 }, { 1.f, 1.f, 1.f }, 0, { true, false }, false);
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

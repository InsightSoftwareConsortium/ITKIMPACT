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

// IMPACT: semantic similarity registration from pretrained TorchScript features.
//
// This example mirrors ImpactMetricExample.py and shows the two ITK-facing pieces of
// the module. No torch type ever crosses the API: it speaks only ITK/STL types (images,
// a model-path string, a device string).
//
//   1. the core:   itk::ImageToFeaturesMap extracts a feature map from an image;
//   2. the metric: itk::ImpactImageToImageMetricv4 plugs into
//                  itk::ImageRegistrationMethodv4 to register two images by comparing
//                  those features instead of raw intensities.

#include "itkImageToFeaturesMap.h"
#include "itkImpactImageToImageMetricv4.h"
#include "itkModelConfiguration.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationMethodv4.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkTranslationTransform.h>
#include <itkResampleImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>

int
main(int argc, char * argv[])
{
  if (argc < 5)
  {
    std::cerr << "IMPACT: semantic similarity registration from pretrained features.\n";
    std::cerr << "Usage: " << argv[0] << " model.pt fixedImage movingImage outputWarpedImage [device]\n";
    std::cerr << "  device: \"cpu\" (default), \"cuda\", \"cuda:0\", ...\n";
    return EXIT_FAILURE;
  }
  const char * const  modelPath = argv[1];
  const char * const  fixedPath = argv[2];
  const char * const  movingPath = argv[3];
  const char * const  outputPath = argv[4];
  const std::string   device = (argc > 5) ? argv[5] : "cpu";

  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;

  const auto fixed = itk::ReadImage<ImageType>(fixedPath);
  const auto moving = itk::ReadImage<ImageType>(movingPath);

  // A TorchScript model configuration: (path, dimension, channels, patchSize, voxelSize,
  // overlap, layersMask, mixedPrecision). Only POD/STL types cross the API.
  const itk::ModelConfiguration config(
    modelPath, Dimension, 1, { 0, 0, 0 }, { 1.0f, 1.0f, 1.0f }, 2, { true }, false);

  // --- 1. core: extract a dense feature map from the fixed image --------------------
  using InterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double>;
  auto interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(3);

  auto features = itk::ImageToFeaturesMap<ImageType, InterpolatorType>::New();
  features->SetModelConfiguration(config);
  features->SetInterpolator(interpolator);
  features->SetDevice(device);
  features->AddInput(fixed);
  features->Update();
  const auto featureMap = features->GetOutput(0); // itk::VectorImage<float, 3>
  std::cout << "feature map: " << featureMap->GetLargestPossibleRegion().GetSize()
            << "  channels: " << featureMap->GetNumberOfComponentsPerPixel() << std::endl;

  // --- 2. metric: register moving onto fixed by comparing anatomical features -------
  using MetricType = itk::ImpactImageToImageMetricv4<ImageType, ImageType>;
  auto                                 metric = MetricType::New();
  std::vector<itk::ModelConfiguration> models{ config };
  metric->SetModelsConfiguration(models);
  metric->SetDistance({ "NCC" });      // per-layer loss: L1, L2, NCC, Cosine, Dice, ...
  metric->SetLayersWeight({ 1.0f });
  metric->SetSubsetFeatures({ 4 });    // random channel subset for speed (0 = all)
  metric->SetPCA({ 0 });
  metric->SetMode("Static");           // "Static" (precomputed features) or "Jacobian"
  metric->SetDevice(device);

  using TransformType = itk::TranslationTransform<double, Dimension>;
  auto transform = TransformType::New();
  transform->SetIdentity();

  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  auto optimizer = OptimizerType::New();
  optimizer->SetNumberOfIterations(200);
  optimizer->SetLearningRate(2.0);
  optimizer->SetMinimumStepLength(1e-4);

  using RegistrationType = itk::ImageRegistrationMethodv4<ImageType, ImageType, TransformType>;
  auto registration = RegistrationType::New();
  registration->SetFixedImage(fixed);
  registration->SetMovingImage(moving);
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetInitialTransform(transform);

  // Single resolution level (no shrink, no smoothing).
  RegistrationType::ShrinkFactorsArrayType   shrinkFactors(1);
  RegistrationType::SmoothingSigmasArrayType smoothingSigmas(1);
  shrinkFactors[0] = 1;
  smoothingSigmas[0] = 0;
  registration->SetNumberOfLevels(1);
  registration->SetShrinkFactorsPerLevel(shrinkFactors);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmas);

  registration->Update();
  std::cout << "recovered parameters: " << registration->GetTransform()->GetParameters() << std::endl;

  // --- 3. resample the moving image with the recovered transform --------------------
  auto resample = itk::ResampleImageFilter<ImageType, ImageType>::New();
  resample->SetInput(moving);
  resample->SetTransform(registration->GetTransform());
  resample->SetUseReferenceImage(true);
  resample->SetReferenceImage(fixed);
  resample->SetDefaultPixelValue(0);
  itk::WriteImage(resample->GetOutput(), outputPath);

  return EXIT_SUCCESS;
}

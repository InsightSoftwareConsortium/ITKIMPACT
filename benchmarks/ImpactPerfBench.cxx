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

// The speed of the v4 metric on the lung CT pair and the 32-channel layer of M258, case by case:
// one warm-up evaluation, then the median of three, printed as `IMPACT_PERF <case> <ms>`.
//
// It is not part of the test suite: a time depends on the machine and on whatever else it runs.
// benchmarks/ab.sh appends this file to the GTest driver of each of two revisions, so that both are
// measured by the same code, and runs them alternately. It uses only the metric's public interface,
// which lets it measure revisions older than itself. IMPACT_PERF_ONLY keeps the cases whose name
// (mode-device-loss) contains it: online-cpu, online-cuda, static, ncc...

#include "itkAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkImageFileReader.h"
#include "itkImpactImageToImageMetricv4.h"
#include "itkMetaImageIO.h"
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

TEST(ImpactPerf, DISABLED_Evaluation)
{
  using PerfImageType = itk::Image<float, 3>;
  using MetricType = itk::ImpactImageToImageMetricv4<PerfImageType, PerfImageType, PerfImageType, double>;

  const auto data = [](const std::string & name) -> std::string {
    for (const std::string directory : { IMPACT_EXTERNAL_DATA_DIR, IMPACT_TEST_DATA_DIR })
    {
      if (std::ifstream(directory + "/" + name).good())
      {
        return directory + "/" + name;
      }
    }
    return {};
  };
  const std::string modelPath = data("M258_2_Layers.pt");
  const std::string fixedPath = data("3DCT_lung_baseline.mha");
  const std::string movingPath = data("3DCT_lung_followup.mha");
  if (modelPath.empty() || fixedPath.empty() || movingPath.empty())
  {
    GTEST_SKIP() << "real-CT test data not available";
  }
  const auto read = [](const std::string & path) {
    auto reader = itk::ImageFileReader<PerfImageType>::New();
    reader->SetFileName(path);
    reader->SetImageIO(itk::MetaImageIO::New());
    reader->Update();
    return PerfImageType::Pointer(reader->GetOutput());
  };
  const PerfImageType::Pointer fixed = read(fixedPath);
  const PerfImageType::Pointer moving = read(movingPath);

  // Points away from the border, drawn from a fixed seed: the same for every revision.
  const auto points = [&](unsigned int count) {
    auto         set = MetricType::FixedSampledPointSetType::New();
    std::mt19937 generator(1);
    const auto   size = fixed->GetLargestPossibleRegion().GetSize();
    for (unsigned int n = 0; n < count; ++n)
    {
      PerfImageType::IndexType index;
      for (unsigned int d = 0; d < 3; ++d)
      {
        index[d] = 8 + static_cast<long>(generator() % (size[d] - 16));
      }
      PerfImageType::PointType point;
      fixed->TransformIndexToPhysicalPoint(index, point);
      set->SetPoint(n, point);
    }
    return set;
  };

  struct Case
  {
    const char * name;
    const char * mode;
    const char * loss;
    unsigned int count;
    unsigned int workUnits; // 0: the default
    const char * device;
  };
  const char * only = std::getenv("IMPACT_PERF_ONLY");
  for (const Case c : { Case{ "static-cpu-l2-1thread", "Static", "L2", 20000, 1, "cpu" },
                        Case{ "static-cpu-l2", "Static", "L2", 20000, 0, "cpu" },
                        Case{ "online-cpu-l2", "Jacobian", "L2", 2000, 0, "cpu" },
                        Case{ "online-cpu-ncc", "Jacobian", "NCC", 2000, 0, "cpu" },
                        Case{ "online-cuda-l2", "Jacobian", "L2", 2000, 0, "cuda" },
                        Case{ "online-cuda-ncc", "Jacobian", "NCC", 2000, 0, "cuda" },
                        Case{ "online-cuda-l2-20000", "Jacobian", "L2", 20000, 0, "cuda" } })
  {
    if ((only != nullptr && std::string(c.name).find(only) == std::string::npos) ||
        (std::string(c.device) == "cuda" && !torch::cuda::is_available()))
    {
      continue;
    }
    // Online, a 5^3 patch at 3 mm around every point; Static, the whole image.
    const unsigned int                         patch = std::string(c.mode) == "Jacobian" ? 5 : 0;
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(modelPath,
                         3,
                         1,
                         std::vector<unsigned int>{ patch, patch, patch },
                         std::vector<float>{ 3.f, 3.f, 3.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, false },
                         false);
    auto metric = MetricType::New();
    metric->SetModelsConfiguration(configs);
    metric->SetDistance({ std::string(c.loss) });
    metric->SetLayersWeight({ 1.f });
    metric->SetSubsetFeatures({ 32 });
    metric->SetPCA({ 0 });
    metric->SetMode(c.mode);
    metric->SetSeed(1);
    metric->SetFeaturesMapUpdateInterval(-1);
    metric->SetDevice(c.device);
    auto affine = itk::AffineTransform<double, 3>::New();
    affine->SetIdentity();
    metric->SetFixedImage(fixed);
    metric->SetMovingImage(moving);
    metric->SetFixedTransform(itk::IdentityTransform<double, 3>::New());
    metric->SetMovingTransform(affine);
    metric->SetFixedSampledPointSet(points(c.count));
    metric->SetUseSampledPointSet(true);
    metric->SetVirtualDomainFromImage(fixed);
    if (c.workUnits > 0)
    {
      metric->SetMaximumNumberOfWorkUnits(c.workUnits);
    }
    metric->Initialize();

    MetricType::MeasureType    value;
    MetricType::DerivativeType derivative;
    metric->GetValueAndDerivative(value, derivative); // warm-up
    std::array<double, 3> times{};
    for (double & time : times)
    {
      const auto start = std::chrono::steady_clock::now();
      metric->GetValueAndDerivative(value, derivative);
      time = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(times.begin(), times.end());
    std::cout << "IMPACT_PERF " << c.name << " " << times[1] << std::endl;
  }
}

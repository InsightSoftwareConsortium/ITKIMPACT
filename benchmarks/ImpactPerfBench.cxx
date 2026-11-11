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
// (mode-device-loss) contains it: online-cpu, online-cuda, static, elastix-cpu, ncc...

#include "itkAffineTransform.h"
#include "itkIdentityTransform.h"
#include "itkImageFileReader.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImpactImageToImageMetricv4.h"
#include "itkImpactOnlineInference.h"
#include "itkMultiThreaderBase.h"
#include "itkMetaImageIO.h"
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
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

// The elastix metric's online path, on the same pair, model and points as above: every point handed to the shared
// Impact::EvaluateOnline, as the elastix metric does, through an Impact::OnlineHost that cuts the patches with
// elastix's interpolators, an order-3 B-spline for the fixed image and order 1 for the moving one. IMPACT_PERF_ONLY
// selects the cases as above: elastix-cpu, elastix-cuda... A revision older than Impact::OnlineHost measures none.
#ifdef ITK_IMPACT_HAS_ONLINE_HOST
TEST(ImpactPerf, DISABLED_ElastixOnline)
{
  using PerfImageType = itk::Image<float, 3>;
  using PointType = PerfImageType::PointType;
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
  auto fixedInterpolator = itk::BSplineInterpolateImageFunction<PerfImageType, double, double>::New();
  fixedInterpolator->SetSplineOrder(3);
  fixedInterpolator->SetInputImage(fixed);
  auto movingInterpolator = itk::BSplineInterpolateImageFunction<PerfImageType, double, double>::New();
  movingInterpolator->SetSplineOrder(1);
  movingInterpolator->SetInputImage(moving);
  const auto & direction = fixed->GetDirection();
  const auto   patchPoint = [&](const PointType & center, const std::vector<float> & offset) {
    PointType point(center);
    for (unsigned int j = 0; j < 3; ++j)
    {
      for (unsigned int r = 0; r < 3; ++r)
      {
        point[r] += direction[r][j] * offset[j];
      }
    }
    return point;
  };

  std::vector<PointType> points;
  {
    std::mt19937 generator(1);
    const auto   size = fixed->GetLargestPossibleRegion().GetSize();
    for (unsigned int n = 0; n < 2000; ++n)
    {
      PerfImageType::IndexType index;
      for (unsigned int d = 0; d < 3; ++d)
      {
        index[d] = 8 + static_cast<long>(generator() % (size[d] - 16));
      }
      PointType point;
      fixed->TransformIndexToPhysicalPoint(index, point);
      points.push_back(point);
    }
  }

  constexpr int64_t parameters = 12; // an affine transform's
  // d(x + A x + t)/d(A, t) at a point, [3, 12] row-major.
  const auto affineJacobian = [](const PointType & point, float * out) {
    std::fill(out, out + 3 * parameters, 0.0f);
    for (unsigned int d = 0; d < 3; ++d)
    {
      for (unsigned int c = 0; c < 3; ++c)
      {
        out[d * parameters + d * 3 + c] = static_cast<float>(point[c]);
      }
      out[d * parameters + 9 + d] = 1.0f;
    }
  };
  const unsigned int units = itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
  auto               threader = itk::MultiThreaderBase::New();
  threader->SetNumberOfWorkUnits(units);

  // The elastix metric as an Impact::OnlineHost: the points cut a row at a time, every patch voxel through the
  // interpolators.
  class Host : public itk::Impact::OnlineHost
  {
  public:
    std::function<void(int64_t, const std::function<void(int64_t)> &)> parallelize;
    std::function<void(int64_t row, bool derivative, bool fixed, itk::Impact::OnlineBatch & batch, int64_t at)> cut;
    int64_t voxels{ 0 };
    itk::Impact::OnlineBatch
    Sample(int64_t begin, int64_t end, bool parallel, bool derivative, bool fixed) const override
    {
      const int64_t            size = end - begin;
      itk::Impact::OnlineBatch batch;
      batch.fixedPatches.push_back(fixed ? torch::empty({ size, voxels }) : torch::Tensor());
      batch.movingPatches.push_back(torch::empty({ size, voxels }));
      batch.movingGradients.push_back(derivative ? torch::empty({ size, voxels, 3 }) : torch::Tensor());
      if (derivative)
      {
        batch.transformJacobian = torch::empty({ size, 3, parameters });
        batch.nonZeroJacobianIndices = torch::arange(parameters, torch::kInt64).repeat({ size, 1 });
      }
      auto rows = [&](int64_t first, int64_t last) {
        for (int64_t row = first; row < last; ++row)
        {
          cut(begin + row, derivative, fixed, batch, row);
        }
      };
      if (!parallel)
      {
        rows(0, size);
        return batch;
      }
      const int64_t parts = std::max<int64_t>(1, std::min<int64_t>(size, 64));
      parallelize(parts, [&](int64_t part) { rows(size * part / parts, size * (part + 1) / parts); });
      return batch;
    }
    void
    Parallelize(int64_t parts, const std::function<void(int64_t)> & body) const override
    {
      parallelize(parts, body);
    }
  };

  const char * only = std::getenv("IMPACT_PERF_ONLY");
  for (const auto & [name, loss, device] :
       { std::tuple<const char *, const char *, const char *>{ "elastix-cpu-l2", "L2", "cpu" },
         { "elastix-cpu-ncc", "NCC", "cpu" },
         { "elastix-cuda-l2", "L2", "cuda" },
         { "elastix-cuda-ncc", "NCC", "cuda" } })
  {
    if ((only != nullptr && std::string(name).find(only) == std::string::npos) ||
        (std::string(device) == "cuda" && !torch::cuda::is_available()))
    {
      continue;
    }
    const torch::Device                        torchDevice(device);
    std::vector<itk::ImpactModelConfiguration> configs;
    configs.emplace_back(modelPath,
                         3,
                         1,
                         std::vector<unsigned int>{ 5, 5, 5 },
                         std::vector<float>{ 3.f, 3.f, 3.f },
                         std::vector<unsigned int>{ 0, 0, 0 },
                         std::vector<bool>{ true, false },
                         false);
    itk::ModelTo(configs[0], torchDevice); // as elxImpactMetric does
    itk::Impact::GetModelOutputsExample(configs, "moving", torchDevice);
    const std::vector<torch::Tensor>        subsets{ torch::arange(32, torch::kLong).to(torchDevice) };
    const std::vector<std::vector<float>> & patch = configs[0].GetPatchIndex();

    Host host;
    host.voxels = static_cast<int64_t>(patch.size());
    host.parallelize = [&](int64_t parts, const std::function<void(int64_t)> & body) {
      threader->ParallelizeArray(0, parts, [&](itk::SizeValueType part) { body(static_cast<int64_t>(part)); }, nullptr);
    };
    host.cut = [&](int64_t point, bool derivative, bool fixedToo, itk::Impact::OnlineBatch & batch, int64_t row) {
      float * fixedOut = fixedToo ? batch.fixedPatches[0].data_ptr<float>() + row * host.voxels : nullptr;
      float * movingOut = batch.movingPatches[0].data_ptr<float>() + row * host.voxels;
      float * gradientOut = derivative ? batch.movingGradients[0].data_ptr<float>() + row * host.voxels * 3 : nullptr;
      for (size_t v = 0; v < patch.size(); ++v)
      {
        const PointType at = patchPoint(points[point], patch[v]);
        if (fixedOut != nullptr)
        {
          fixedOut[v] = static_cast<float>(fixedInterpolator->Evaluate(at));
        }
        if (gradientOut == nullptr)
        {
          movingOut[v] = static_cast<float>(movingInterpolator->Evaluate(at));
          continue;
        }
        double                                                                   value;
        itk::BSplineInterpolateImageFunction<PerfImageType>::CovariantVectorType gradient;
        movingInterpolator->EvaluateValueAndDerivative(at, value, gradient);
        movingOut[v] = static_cast<float>(value);
        for (unsigned int d = 0; d < 3; ++d)
        {
          gradientOut[v * 3 + d] = static_cast<float>(gradient[d]);
        }
      }
      if (derivative)
      {
        affineJacobian(points[point], batch.transformJacobian.data_ptr<float>() + row * 3 * parameters);
      }
    };
    auto evaluate = [&](double & value, torch::Tensor & derivative) {
      const itk::Impact::OneOpenMPThread                           oneThread(units > 1); // as the elastix metric
      std::vector<std::vector<std::unique_ptr<itk::Impact::Loss>>> losses(units);
      for (auto & part : losses)
      {
        part.push_back(itk::Impact::LossFactory::Instance().Create(loss));
        part.back()->SetNumberOfParameters(parameters);
      }
      const itk::Impact::OnlineSettings settings{ configs, configs, subsets, torchDevice, 0, units };
      itk::Impact::EvaluateOnline(
        settings,
        host,
        static_cast<int64_t>(points.size()),
        true,
        [&](int64_t part) -> std::vector<std::unique_ptr<itk::Impact::Loss>> & { return losses[part]; });
      for (unsigned int unit = 1; unit < units; ++unit)
      {
        *losses[0][0] += *losses[unit][0];
      }
      value = losses[0][0]->GetValue(static_cast<double>(points.size()));
      derivative = losses[0][0]->GetDerivative(static_cast<double>(points.size()));
    };
    double        value;
    torch::Tensor derivative;
    evaluate(value, derivative); // warm-up
    std::array<double, 3> times{};
    for (double & time : times)
    {
      const auto start = std::chrono::steady_clock::now();
      evaluate(value, derivative);
      time = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(times.begin(), times.end());
    std::cout << "IMPACT_PERF " << name << " " << times[1] << " value " << value << " |derivative| "
              << derivative.norm().item<double>() << std::endl;
  }
}
#endif

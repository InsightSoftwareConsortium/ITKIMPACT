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
#ifndef itkImpactBatchBudget_h
#define itkImpactBatchBudget_h

// How many patches one online (Jacobian-mode) forward + backward may hold on the device.
// Measured, not configured: the marginal memory of a patch from two probe passes, the free
// memory from the driver. ForEachBatch runs a sample set in batches of that size under one
// process-wide lock, and halves the batch when an allocation still fails.
// Pulls in LibTorch; never part of the castxml-parsed public surface.

#include "itkImpactModelConfigurationDetail.h"
#include "ImpactLoss.h"

#include <itkMacro.h>
#include <torch/torch.h>

#if __has_include(<c10/cuda/CUDACachingAllocator.h>) && __has_include(<cuda_runtime_api.h>)
#  include <c10/cuda/CUDACachingAllocator.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <cuda_runtime_api.h>
#  define ITK_IMPACT_HAS_CUDA_MEMORY_STATS 1
#else
#  define ITK_IMPACT_HAS_CUDA_MEMORY_STATS 0
#endif

#include <algorithm>
#include <cctype>
#include <mutex>
#include <string>
#include <vector>

namespace itk
{
namespace Impact
{

/** One lock per process: every work unit of a threaded metric would otherwise put a batch of
 * activations on the same device at once. Only taken for CUDA devices. */
inline std::mutex &
DeviceForwardMutex()
{
  static std::mutex mutex;
  return mutex;
}

/** The caching allocator throws c10::OutOfMemoryError; an allocation failing inside a scripted
 * forward is rethrown by the TorchScript interpreter as a std::runtime_error carrying the text. */
inline bool
IsDeviceOutOfMemory(const std::exception & error)
{
  if (dynamic_cast<const c10::OutOfMemoryError *>(&error) != nullptr)
  {
    return true;
  }
  std::string what = error.what();
  std::transform(what.begin(), what.end(), what.begin(), [](unsigned char c) { return std::tolower(c); });
  return what.find("out of memory") != std::string::npos || what.find("alloc_failed") != std::string::npos;
}

/** Patches of `configuration` that fit on `device`, forward and backward; 0 if not measurable
 * (CPU, CPU-only LibTorch, or no strictly positive patch size). Moves the model to the device. */
inline int64_t
MeasureBatchBudget(const ImpactModelConfiguration & configuration, const torch::Device & device, double safety = 0.8)
{
#if ITK_IMPACT_HAS_CUDA_MEMORY_STATS
  const std::vector<int64_t> & patchSize = configuration.GetPatchSize();
  if (!device.is_cuda() || patchSize.empty() ||
      std::any_of(patchSize.begin(), patchSize.end(), [](int64_t extent) { return extent <= 0; }))
  {
    return 0;
  }
  std::lock_guard<std::mutex> lock(DeviceForwardMutex());
  ModelTo(configuration, device);
  const c10::DeviceIndex index = device.has_index() ? device.index() : c10::DeviceIndex{ 0 };
  c10::cuda::CUDAGuard   guard(index);
  const auto             aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  std::vector<int64_t>   shape{ 1, static_cast<int64_t>(configuration.GetNumberOfChannels()) };
  shape.insert(shape.end(), patchSize.rbegin(), patchSize.rend());

  auto peakOf = [&](int64_t batch) {
    shape[0] = batch;
    c10::cuda::CUDACachingAllocator::resetPeakStats(index);
    {
      torch::Tensor patch =
        torch::rand(shape, torch::TensorOptions().dtype(GetModelDtype(configuration)).device(device))
          .set_requires_grad(true);
      torch::Tensor total;
      for (torch::jit::IValue & output : Forward(configuration, patch))
      {
        torch::Tensor term = output.toTensor().to(torch::kFloat32).sum();
        total = total.defined() ? total + term : term;
      }
      torch::autograd::grad({ total }, { patch });
    }
    return c10::cuda::CUDACachingAllocator::getDeviceStats(index).allocated_bytes[aggregate].peak;
  };
  const int64_t peakOne = peakOf(1);
  const int64_t perPatch = std::max<int64_t>(peakOf(2) - peakOne, std::max<int64_t>(peakOne / 2, 1));

  size_t freeBytes = 0, totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
  {
    return 0;
  }
  const auto    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(index);
  const int64_t cached = stats.reserved_bytes[aggregate].current - stats.allocated_bytes[aggregate].current;
  const double  usable = static_cast<double>(static_cast<int64_t>(freeBytes) + std::max<int64_t>(cached, 0)) * safety;
  return std::max<int64_t>(static_cast<int64_t>(usable / static_cast<double>(perPatch)), 1);
#else
  (void)configuration;
  (void)device;
  (void)safety;
  return 0;
#endif
}

/** Bound every configuration by min(requested, measured budget); 0 for either means "no bound". */
inline void
ConfigureBatchSize(std::vector<ImpactModelConfiguration> & configurations,
                   const torch::Device &                   device,
                   int64_t                                 requested = 0)
{
  for (ImpactModelConfiguration & configuration : configurations)
  {
    const int64_t budget = MeasureBatchBudget(configuration, device);
    const int64_t bound = requested > 0 && budget > 0 ? std::min(requested, budget) : std::max(requested, budget);
    SetBatchSize(configuration, bound);
  }
}

/** The tightest bound among `configurations`, 0 if none. */
inline int64_t
EffectiveBatchSize(const std::vector<ImpactModelConfiguration> & configurations)
{
  int64_t bound = 0;
  for (const ImpactModelConfiguration & configuration : configurations)
  {
    const int64_t own = GetBatchSize(configuration);
    bound = own > 0 && (bound == 0 || own < bound) ? own : bound;
  }
  return bound;
}

/** Drop the device memory a failed attempt left in the caching allocator. */
inline void
ReleaseCachedDeviceMemory(const torch::Device & device)
{
#if ITK_IMPACT_HAS_CUDA_MEMORY_STATS
  if (device.is_cuda())
  {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#else
  (void)device;
#endif
}

/** Halve the bound after a batch of `attempted` failed to allocate. */
inline void
ShrinkBatchSize(const ImpactModelConfiguration & configuration, int64_t attempted, const torch::Device & device)
{
  SetBatchSize(configuration, attempted / 2);
  ReleaseCachedDeviceMemory(device);
  itkGenericOutputMacro("IMPACT: a batch of " << attempted << " patches ran out of device memory; retrying with "
                                              << attempted / 2 << ".");
}

/** Run `body(begin, end)` over [0, count) in batches of at most the configuration's bound, on
 * the device under the lock. A batch that runs out of memory is replayed at half the size
 * after `rollback` (which must undo what `body` had accumulated); a batch of one that fails
 * propagates. */
template <typename TBody, typename TRollback>
inline void
ForEachBatch(const ImpactModelConfiguration & configuration,
             const torch::Device &            device,
             int64_t                          count,
             TBody &&                         body,
             TRollback &&                     rollback)
{
  for (int64_t begin = 0; begin < count;)
  {
    const int64_t bound = GetBatchSize(configuration);
    const int64_t end = bound > 0 ? std::min(begin + bound, count) : count;
    try
    {
      std::unique_lock<std::mutex> lock(DeviceForwardMutex(), std::defer_lock);
      if (device.is_cuda())
      {
        lock.lock();
      }
      body(begin, end);
      begin = end;
    }
    catch (const std::exception & error)
    {
      if (!IsDeviceOutOfMemory(error) || end - begin <= 1)
      {
        throw;
      }
      rollback();
      ShrinkBatchSize(configuration, end - begin, device);
    }
  }
}

} // namespace Impact
} // namespace itk

#endif // itkImpactBatchBudget_h

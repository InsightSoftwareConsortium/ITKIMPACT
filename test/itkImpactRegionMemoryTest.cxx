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

// What the two stages hold when they run by regions from mapped files does not grow with the image: each is run on
// two pairs streamed from disk, one of eight times the voxels of the other, a control cell a voxel, and the most
// bytes it held at once, on the host and on the device, must be the same to within a quarter of a byte a voxel
// added. A single array of floats over the cells of the greater pair is three times what a run holds at most, and
// would add three bytes a voxel; the lists of the boxes, which do grow, add a few hundredths.
//
// The host's bytes are counted where they are allocated, which a resident set size cannot tell from the pages of
// the mapped files the system keeps: operator new, replaced here, counts the C++ heap (the images, the vectors),
// and LibTorch's host tensors are put on that heap by a CPU allocator of the test's. Neither sees a mapped file.
// The device's are the CUDA allocator's own count, read when a device is there; on the CPU the device's tensors
// are the host's. A program of its own: the operators it replaces are the whole executable's.

#include "itkImpactCoarseRegistration.h"
#include "itkImpactFineRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMetaImageIOFactory.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <string>
#if defined(__APPLE__)
#  include <malloc/malloc.h>
#else
#  include <malloc.h>
#endif

namespace
{
// Signed: a block another module allocated may be freed here.
std::atomic<long long> hostLive{ 0 };
std::atomic<long long> hostPeak{ 0 };

long long
BlockBytes(void * block)
{
#if defined(_WIN32)
  return static_cast<long long>(_msize(block));
#elif defined(__APPLE__)
  return static_cast<long long>(malloc_size(block));
#else
  return static_cast<long long>(malloc_usable_size(block));
#endif
}
} // namespace

void *
operator new(std::size_t bytes)
{
  void * const block = std::malloc(bytes > 0 ? bytes : 1);
  if (block == nullptr)
  {
    throw std::bad_alloc();
  }
  const long long live = hostLive.fetch_add(BlockBytes(block)) + BlockBytes(block);
  for (long long peak = hostPeak.load(); live > peak && !hostPeak.compare_exchange_weak(peak, live);)
  {
  }
  return block;
}

void
operator delete(void * block) noexcept
{
  if (block != nullptr)
  {
    hostLive.fetch_sub(BlockBytes(block));
    std::free(block);
  }
}

namespace
{
// LibTorch's host tensors on the C++ heap, where they are counted.
class HeapAllocator final : public c10::Allocator
{
public:
  c10::DataPtr
  allocate(std::size_t bytes) override
  {
    void * const block = ::operator new(bytes);
    return { block, block, [](void * held) { ::operator delete(held); }, c10::Device(c10::DeviceType::CPU) };
  }
  void
  copy_data(void * to, const void * from, std::size_t count) const override
  {
    this->default_copy_data(to, from, count);
  }
};

using ImageType = itk::Image<float, 3>;
using CoarseType = itk::ImpactCoarseRegistration<ImageType>;
using FineType = itk::ImpactFineRegistration<ImageType>;
using ReaderType = itk::ImageFileReader<ImageType>;

// A smooth texture of `side` voxels a side, f(x + shift), written to `file`; nothing of it stays in memory.
void
WriteTexture(unsigned int side, double shift, const std::string & file)
{
  auto                  image = ImageType::New();
  ImageType::RegionType region;
  region.SetSize(ImageType::SizeType::Filled(side));
  image->SetRegions(region);
  image->Allocate();
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const auto   index = it.GetIndex();
    const double x = index[0] + shift, y = index[1] - shift, z = index[2] + 0.5 * shift;
    it.Set(static_cast<float>(std::sin(x / 2.3) * std::cos(y / 3.1) + 0.5 * std::sin((x + y + 2 * z) / 4.0) +
                              0.7 * std::cos(z / 2.7 + x / 5.0)));
  }
  itk::WriteImage(image, file);
}

// The most bytes held at once over a run, beyond those held when it starts.
struct Peaks
{
  double host{ 0.0 };
  double device{ 0.0 };
};

// Both stages by regions from mapped files on a pair of `side` voxels a side, on `device`: the coarse stage, then
// the fine stage started from the coarse stage's field, which it asks by regions. With `brickBytes`, the fine
// stage alone, its bricks of that many bytes.
bool
Run(unsigned int        side,
    const std::string & device,
    const std::string & temporary,
    itk::SizeValueType  brickBytes,
    Peaks &             coarsePeaks,
    Peaks &             finePeaks)
{
  const std::string fixedFile = temporary + "/itkImpactRegionMemoryTest_fixed.mha";
  const std::string movingFile = temporary + "/itkImpactRegionMemoryTest_moving.mha";
  WriteTexture(side, 0.0, fixedFile);
  WriteTexture(side, 1.0, movingFile);
  auto fixedReader = ReaderType::New();
  fixedReader->SetFileName(fixedFile);
  fixedReader->UseStreamingOn();
  auto movingReader = ReaderType::New();
  movingReader->SetFileName(movingFile);
  movingReader->UseStreamingOn();
  // What a reader keeps once it has opened its file is the reader's, not a stage's.
  fixedReader->UpdateOutputInformation();
  movingReader->UpdateOutputInformation();
  const std::string scratch = temporary + "/itkImpactRegionMemoryTest_scratch";
  std::filesystem::create_directories(scratch);

  const torch::Device torchDevice(device);
  auto                measured = [&](auto && run) {
    Peaks        peaks;
    const double held = static_cast<double>(hostLive.load());
    hostPeak = hostLive.load();
#if ITK_IMPACT_HAS_CUDA_MEMORY_STATS
    const auto             aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    const c10::DeviceIndex index = torchDevice.has_index() ? torchDevice.index() : c10::DeviceIndex{ 0 };
    int64_t                heldOnDevice = 0;
    if (torchDevice.is_cuda())
    {
      c10::cuda::CUDACachingAllocator::resetPeakStats(index);
      heldOnDevice = c10::cuda::CUDACachingAllocator::getDeviceStats(index).allocated_bytes[aggregate].current;
    }
#endif
    run();
    peaks.host = static_cast<double>(hostPeak.load()) - held;
    peaks.device = peaks.host;
#if ITK_IMPACT_HAS_CUDA_MEMORY_STATS
    if (torchDevice.is_cuda())
    {
      peaks.device = static_cast<double>(
        c10::cuda::CUDACachingAllocator::getDeviceStats(index).allocated_bytes[aggregate].peak - heldOnDevice);
    }
#endif
    return peaks;
  };

  auto coarse = CoarseType::New();
  coarse->SetFixedImage(fixedReader->GetOutput());
  coarse->SetMovingImage(movingReader->GetOutput());
  coarse->SetGridSpacing(1);
  coarse->SetDisplacementHalfWidth(1); // 27 candidates
  coarse->SetRegionSize(16);
  coarse->SetCostBoxBytes(27 * 8 * 20 * 20 * 20); // boxes of 16 cells with their margins
  coarse->SetResidentFeatureBudget(1);            // nothing fits: mapped files
  coarse->SetScratchDirectory(scratch);
  coarse->SetDevice(device);
  if (brickBytes == 0)
  {
    coarsePeaks = measured([&]() { coarse->GetCoarseGridTransform(); });
  }

  auto fine = FineType::New();
  fine->SetFixedImage(fixedReader->GetOutput());
  fine->SetMovingImage(movingReader->GetOutput());
  if (brickBytes == 0)
  {
    fine->SetInitialDisplacementField(coarse->GetOutput());
  }
  fine->SetGridShrinkFactor(1);
  fine->SetControlGridSmoothingIterations(1);
  fine->SetNumberOfIterations(2);
  fine->SetRegionSize(16);
  fine->SetResidentFeatureBudget(1);
  fine->SetBrickBytes(brickBytes > 0 ? brickBytes : 1); // 1: bricks of one box
  fine->SetScratchDirectory(scratch);
  fine->SetDevice(device);
  finePeaks = measured([&]() { fine->GetControlGridTransform(); });
  return !coarse->GetResidentFeatures() && !fine->GetResidentFeatures() &&
         fixedReader->GetOutput()->GetBufferedRegion() != fixedReader->GetOutput()->GetLargestPossibleRegion();
}
} // namespace

int
main(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <temporary directory> [<device>...]" << std::endl;
    return EXIT_FAILURE;
  }
  itk::MetaImageIOFactory::RegisterOneFactory(); // no test driver registers the factories of this program
  static HeapAllocator allocator;
  c10::SetAllocator(c10::DeviceType::CPU, &allocator);
  const std::string temporary = argv[1];

  // The devices named, or the CPU and the CUDA device when there is one.
  std::vector<std::string> devices(argv + 2, argv + argc);
  if (devices.empty())
  {
    devices.emplace_back("cpu");
    if (torch::cuda::is_available())
    {
      devices.emplace_back("cuda");
    }
  }
  int status = EXIT_SUCCESS;
  for (const std::string & device : devices)
  {
    // 48 and 96 voxels a side: 27 and 216 boxes of 16, of which the inner ones read all their margins. A first run
    // takes what the libraries allocate once for the process.
    constexpr unsigned int lesser = 48, greater = 96;
    const double           added = std::pow(greater, 3) - std::pow(lesser, 3);
    Peaks                  coarseLesser, fineLesser, coarseGreater, fineGreater;
    if (!Run(lesser, device, temporary, 0, coarseLesser, fineLesser) ||
        !Run(lesser, device, temporary, 0, coarseLesser, fineLesser) ||
        !Run(greater, device, temporary, 0, coarseGreater, fineGreater))
    {
      std::cerr << "The runs should read by regions and keep their maps in mapped files." << std::endl;
      return EXIT_FAILURE;
    }
    auto grows = [&](const char * stage, const char * where, double atLesser, double atGreater) {
      const double perVoxel = (atGreater - atLesser) / added;
      std::cout << device << ", " << stage << " stage, " << where << ": " << atLesser << " bytes at most on " << lesser
                << "^3 voxels, " << atGreater << " on " << greater << "^3: " << perVoxel << " byte a voxel added"
                << std::endl;
      return !(perVoxel < 0.25);
    };
    if (grows("coarse", "host", coarseLesser.host, coarseGreater.host) ||
        grows("coarse", "device", coarseLesser.device, coarseGreater.device) ||
        grows("fine", "host", fineLesser.host, fineGreater.host) ||
        grows("fine", "device", fineLesser.device, fineGreater.device))
    {
      std::cerr << "What a run by regions holds grows with the image." << std::endl;
      status = EXIT_FAILURE;
    }

    // BrickBytes bounds what a step of the fine stage holds on the device: bricks of several boxes hold more than
    // those of one, within it.
    constexpr itk::SizeValueType brickBytes = 6000000; // two boxes a side
    Peaks                        unused, fineBricks;
    Run(greater, device, temporary, brickBytes, unused, fineBricks);
    std::cout << device << ", fine stage, bricks within " << brickBytes << " bytes: " << fineBricks.device
              << " bytes at most on the device, " << fineGreater.device << " by bricks of one box" << std::endl;
    if (!(fineBricks.device > fineGreater.device && fineBricks.device <= static_cast<double>(brickBytes)))
    {
      std::cerr << "BrickBytes does not bound what a brick's step holds on the device." << std::endl;
      status = EXIT_FAILURE;
    }
  }
  return status;
}

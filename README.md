<p align="center">
  <img src="logo.png" alt="IMPACT-Reg logo" width="300"/>
</p>

# ITKImpact

[![Build Status](https://github.com/InsightSoftwareConsortium/ITKIMPACT/actions/workflows/build-test-package.yml/badge.svg)](https://github.com/InsightSoftwareConsortium/ITKIMPACT/actions/workflows/build-test-package.yml)
[![PyPI Version](https://img.shields.io/pypi/v/itk-impact.svg)](https://pypi.python.org/pypi/itk-impact)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/InsightSoftwareConsortium/ITKIMPACT/blob/main/LICENSE)

<p align="center"><em>Compare and register medical images in the deep-feature space of pretrained
models — semantic, model-agnostic, and native to ITK.</em></p>

**At a glance**

- **Semantic, model-agnostic comparison** — match anatomy in the internal features of *any*
  pretrained TorchScript model (TotalSegmentator, SAM 2.1, DINOv2, MIND, …), robust across
  modality, contrast, noise, and artifacts.
- **Drop-in ITK v4 metric** — `itk::ImpactImageToImageMetricv4` plugs straight into
  `itk::ImageRegistrationMethodv4` and any v4 optimizer; only the comparison changes.
- **Torch-backed dense registration** — a self-contained ConvexAdam-style coarse→fine pipeline,
  exposed as ITK filters, refined on GPU with `torch::optim::Adam`.
- **A deep-learning inference engine for ITK** — `itk::ImageToFeaturesMap` runs *any* TorchScript
  model patch-wise (segmentation, synthesis, denoising, …): image in, `itk::VectorImage` out.
- **Python & C++, CPU & CUDA** — one small wheel that reuses the LibTorch inside your installed
  `torch`; the same core also powers the elastix plugin.

## Overview

**ITKImpact** brings the **IMPACT** ecosystem into the heart of [ITK](https://itk.org): an
official ITK remote module that makes **anatomical comparison in the deep-feature space of
pretrained models** a first-class ITK capability, instead of comparing images at the
intensity level.

The same anatomy looks very different across modality (CT / MR / CBCT), contrast, noise, and
artifacts, so a pixel-wise intensity comparison is unreliable. The internal features of a
pretrained model — a segmentation backbone especially — make anatomical structures stand out
and attenuate artifacts, separating *anatomy* (shapes, structure, spatial organization) from
*appearance* (intensity, contrast, noise). Comparing images there stays stable when the
appearance changes — the common thread of the whole IMPACT ecosystem.

**IMPACT** (*Image Metric with Pretrained model-Agnostic Comparison for Transmodality
registration*) is **model-agnostic**: any model mapping an image to features works — a
`TotalSegmentator` / `nnU-Net` backbone, `SAM`, `DINOv2`, MIND descriptors, a self-supervised
encoder — once exported to TorchScript. IMPACT is already an official metric in
[elastix](https://github.com/SuperElastix/elastix); this module puts the *shared core* inside
ITK, so the ITK-native tools and the [**ImpactElastix**](https://github.com/vboussot/ImpactElastix)
plugin build on one implementation.

### The IMPACT ecosystem

Three layers, from the shared foundation up to the applications:

1. **Core — anatomy comparison.** The framework-neutral foundation: **load** a TorchScript
   model (`itk::ModelConfiguration`), **extract** dense feature maps (`itk::ImageToFeaturesMap`),
   and **compare** them with differentiable distances — L1, L2, NCC, cosine, L1-cosine,
   dot-product, Dice (`ImpactLoss.h`). Runs on LibTorch (CPU/CUDA) with ITK images as the
   boundary (`itk::ImageToTensorFilter` / `itk::TensorToImageFilter`). It depends only on ITK
   and LibTorch, and is the backend shared with **ImpactElastix**.

   > **More than features — a deep-learning inference engine for ITK.**
   > `itk::ImageToFeaturesMap` is really a **patch-based TorchScript inference engine**: it tiles
   > the image, runs *any* TorchScript model on each patch (CPU/CUDA, mixed precision), blends the
   > overlaps, and returns the result as an `itk::VectorImage`. Feature extraction for comparison
   > is just one use — the same class runs **segmentation, image synthesis / modality translation,
   > denoising, super-resolution**, any image→image or image→tensor model — so it doubles as a
   > general model-inference filter you can drop into ITK pipelines.

2. **ITK metric — IMPACT-Reg for the v4 framework.** `itk::ImpactImageToImageMetricv4`, a
   drop-in semantic similarity metric for `itk::ImageRegistrationMethodv4` and any
   `itk::*Optimizerv4`. The registration engine stays standard; only the comparison changes.

3. **Registration framework — Torch-backed dense registration.** A self-contained
   [ConvexAdam](https://github.com/multimodallearning/convexAdam)-style pipeline (coarse
   discrete initialization → GPU Adam refinement), exposed as ITK filters but driven by
   *anatomical features* rather than intensities, kept in a single autograd graph for speed.

### Pretrained models

You supply the TorchScript feature extractor. Ready-to-use models are on Hugging Face —
[**VBoussot/impact-torchscript-models**](https://huggingface.co/VBoussot/impact-torchscript-models):
**TotalSegmentator**, **MRSegmentator**, **SAM 2.1**, **DINOv2**, **Anatomix**, and a
TorchScript **MIND** descriptor. The best model / layer / distance is problem-dependent (see
the paper's ablations).

## Installation

### Python

```bash
pip install itk-impact torch
```

The wheel does **not** bundle LibTorch: it links the libtorch that ships inside the `torch`
package, so a single small wheel inherits **CPU or GPU from whichever torch you installed** —
install a CUDA build (`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`)
plus a matching NVIDIA driver for GPU execution, or the default CPU torch otherwise. The torch
version is ABI-coupled to the wheel; `pip` enforces the compatible range. Because the loader
resolves libtorch from the installed torch, `import torch` before the IMPACT filters if you do
not otherwise import it (KonfAI already does).

### C++ (from source)

`itk-impact` is a standard [ITK remote module](https://github.com/InsightSoftwareConsortium/ITKModuleTemplate),
with the one extra dependency of **LibTorch**. By default `find_package(Torch)` is auto-located
from the installed `torch` Python package (`torch.utils.cmake_prefix_path`) — just
`pip install torch` first. Build it against an existing ITK build:

```bash
git clone https://github.com/InsightSoftwareConsortium/ITKIMPACT
cmake -B ITKIMPACT-build -S ITKIMPACT \
  -DCMAKE_BUILD_TYPE=Release \
  -DITK_DIR=/path/to/ITK-build
cmake --build ITKIMPACT-build -j
```

To use a manually downloaded LibTorch C++ distribution instead of the pip package, pass
`-DCMAKE_PREFIX_PATH=/path/to/libtorch` (it takes precedence over the auto-detection). Build it
inside the ITK source tree by enabling `-DModule_Impact=ON`; with `-DITK_WRAP_PYTHON=ON` this
also produces the Python wrapping.

## The Torch registration pipeline

Layer 3 aligns images by their anatomical features in two stages — a coarse discrete
initializer, then a GPU Adam refinement:

```
fixed image ─┐
moving image ┼─▶ itk::ImpactCoarseRegistration ─▶ initial displacement field
             │     (coarse: discrete cost volume +
             │      coupled-convex global regularization)
             └─▶ itk::ImpactFineRegistration ──▶ refined displacement field
                   (fine: grid_sample warp + IMPACT
                    feature loss + diffusion reg, Adam)
```

- **`itk::ImpactCoarseRegistration`** — coarse stage. A discrete SSD cost volume over a dense
  displacement window on a pooled coarse grid, coupled-convex global regularization, then
  upsampling to a full-resolution field. Robust to large misalignments; runs on raw
  intensities or any IMPACT model's features; 2D and 3D.
- **`itk::ImpactFineRegistration`** — fine stage. Holds the field as a GPU leaf tensor, warps
  the moving image/features with `grid_sample`, and minimizes a similarity loss (intensity MSE
  or IMPACT feature loss) plus a diffusion regularizer with `torch::optim::Adam` — all on
  device, no per-iteration CPU↔GPU round trip. Optional low-resolution control grid
  (`GridShrinkFactor`) and PCA channel reduction.

Both output a geometry-correct `itk::DisplacementFieldTransform` (physical millimetres,
fixed→moving); the ITK↔Torch axis/units/direction conventions are handled internally.

## Quick start (Python)

```python
import itk

ImageType = itk.Image[itk.F, 3]
fixed  = itk.imread("fixed.mha",  itk.F)
moving = itk.imread("moving.mha", itk.F)

# Stage 1 — coarse initialization (intensity SSD cost volume)
coarse = itk.ImpactCoarseRegistration[ImageType, ImageType].New()
coarse.SetFixedImage(fixed)
coarse.SetMovingImage(moving)
coarse.SetGridSpacing(4)
coarse.SetDisplacementHalfWidth(5)
coarse.SetDevice("cuda:0")          # or "cpu"
coarse.Update()

# Stage 2 — fast Adam refinement on IMPACT features
cfg = itk.ModelConfiguration("features_model.pt", 3, 1,
                             [0, 0, 0], [1.0, 1.0, 1.0], 0, [True, False], False)
fine = itk.ImpactFineRegistration[ImageType, ImageType].New()
fine.SetFixedImage(fixed)
fine.SetMovingImage(moving)
fine.SetDevice("cuda:0")
fine.SetInitialDisplacementField(coarse.GetDisplacementField())  # warm start
fine.AddModelConfiguration(cfg)
fine.SetDistance(["L2"])            # per-layer loss: L1, L2, NCC, Cosine, Dice, ...
fine.SetNumberOfIterations(100)
fine.SetLearningRate(0.1)
fine.SetRegularizationWeight(1.0)
fine.Update()

field     = fine.GetDisplacementField()           # itk.Image[itk.Vector[itk.F,3],3]
transform = fine.GetDisplacementFieldTransform()   # ready for itk.ResampleImageFilter
warped    = fine.GetWarpedMovingImage()
```

Leaving the model configuration empty makes both filters use a raw-intensity (MSE/SSD)
similarity instead of features. See [`examples/`](examples/) for a metric-based registration
demo (C++ and Python).

## Quick start (C++)

```cpp
using ImageType = itk::Image<float, 3>;

auto coarse = itk::ImpactCoarseRegistration<ImageType>::New();
coarse->SetFixedImage(fixed);
coarse->SetMovingImage(moving);
coarse->SetGridSpacing(4);
coarse->SetDisplacementHalfWidth(5);
coarse->SetDevice("cuda:0");
coarse->Update();

auto fine = itk::ImpactFineRegistration<ImageType>::New();
fine->SetFixedImage(fixed);
fine->SetMovingImage(moving);
fine->SetDevice("cuda:0");
fine->SetInitialDisplacementField(coarse->GetDisplacementField());
itk::ModelConfiguration cfg("features_model.pt", 3, 1, {0,0,0}, {1,1,1}, 0, {true,false}, false);
fine->AddModelConfiguration(cfg);
fine->SetDistance({"L2"});
fine->SetNumberOfIterations(100);
fine->Update();

auto * transform = fine->GetDisplacementFieldTransform();
```

## Components

| Layer | Class / file | Role |
|---|---|---|
| Core | `itk::ModelConfiguration` | Configures and loads a TorchScript feature model. |
| Core | `itk::ImageToFeaturesMap` | Patch-based TorchScript inference engine (tiling, overlap blending, PCA): dense feature maps — or any model output (segmentation, synthesis, denoising…). |
| Core | `itk::ImageToTensorFilter` / `itk::TensorToImageFilter` | ITK image ↔ `torch::Tensor` bridge. |
| Core | `ImpactLoss.h` (`itk::Impact`) | Differentiable feature losses: L1, L2, NCC, Cosine, L1Cosine, DotProduct, Dice. |
| Metric | `itk::ImpactImageToImageMetricv4` | Semantic similarity metric for the ITK v4 framework. |
| Registration | `itk::ImpactCoarseRegistration` | ConvexAdam-style coarse discrete initializer (stage 1). |
| Registration | `itk::ImpactFineRegistration` | Torch-backed Adam dense registration (fine stage). |

## Dependencies

- **ITK** (the metric uses the v4 registration framework).
- **LibTorch** (the C++ PyTorch distribution), built with the matching CUDA toolkit for GPU
  support. Found via `find_package(Torch)`; every client links LibTorch.
- A **pretrained feature model exported to TorchScript** (`.pt`) for the feature modes (not
  needed for the raw-intensity modes) —
  [ready-to-use models](https://huggingface.co/VBoussot/impact-torchscript-models).

The public, Python-wrapped headers are intentionally free of any LibTorch include (torch state
lives behind opaque handles), so the Python bindings build without exposing `torch::*`.

## Notes

- Devices: `"cpu"`, `"cuda"`, `"cuda:0"`, … via `SetDevice`.
- Displacement fields are `itk::Image<itk::Vector<float, N>, N>`.
- The feature path runs each model once on the whole volume and warps the resulting feature
  maps; layers may be coarser than the input and are handled at their native (possibly
  downsampled) resolution, without ever upsampling the features.

## References

If you use IMPACT, please cite the paper ([arXiv:2503.24121](https://arxiv.org/abs/2503.24121)):

```bibtex
@article{boussot2025impact,
  title   = {IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration},
  author  = {Boussot, Valentin and H{\'e}mon, C{\'e}dric and Nunes, Jean-Claude and
             Dowling, Jason and Rouz{\'e}, Simon and Lafond, Caroline and
             Barateau, Ana{\"i}s and Dillenseger, Jean-Louis},
  journal = {arXiv preprint arXiv:2503.24121},
  year    = {2025}
}
```

The IMPACT ecosystem:

- **Paper:** [arXiv:2503.24121](https://arxiv.org/abs/2503.24121)
- **Reference implementation (PyTorch):** [vboussot/ImpactLoss](https://github.com/vboussot/ImpactLoss)
- **elastix integration:** [vboussot/ImpactElastix](https://github.com/vboussot/ImpactElastix)
- **Pretrained TorchScript models:** [VBoussot/impact-torchscript-models](https://huggingface.co/VBoussot/impact-torchscript-models)

Applications built on IMPACT:

- **3D Slicer extension** — [vboussot/SlicerImpactReg](https://github.com/vboussot/SlicerImpactReg): multimodal registration with the IMPACT metric, directly in 3D Slicer.
- **KonfAI CLI** — [vboussot/KonfAI](https://github.com/vboussot/KonfAI) (PyPI [`impact_reg_konfai`](https://pypi.org/project/impact_reg_konfai/)): ready-to-run IMPACT-Reg registration presets in the KonfAI framework.

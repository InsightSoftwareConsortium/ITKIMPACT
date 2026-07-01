#!/usr/bin/env python
#==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#==========================================================================

"""Smoke test of the IMPACT Python wrapping.

This script NEVER imports torch and uses only ITK/STL/POD types: it proves the
torch-free public facade is usable from Python end-to-end. It builds a
ModelConfiguration from a TorchScript model PATH (loading the model in C++),
wraps the core ImageToFeaturesMap, and runs the metric's Initialize()/GetValue()
(which executes the TorchScript inference and the loss) on two synthetic images.

    python itkImpactPythonTest.py <path-to-ImpactToyModel.pt>
"""

import math
import sys
import numpy as np
import itk

model_path = sys.argv[1] if len(sys.argv) > 1 else "Data/ImpactToyModel.pt"
ImageType = itk.Image[itk.F, 3]


def blob(n, cx):
    a = np.zeros((n, n, n), dtype=np.float32)
    c = n / 2.0
    for z in range(n):
        for y in range(n):
            for x in range(n):
                a[z, y, x] = math.exp(-(((x - (c + cx)) ** 2 + (y - c) ** 2 + (z - c) ** 2) / 8.0))
    return itk.image_from_array(a)


fixed, moving = blob(16, -2.0), blob(16, 2.0)

# 1) The torch-free ModelConfiguration loads a TorchScript model from a path string.
cfg = itk.ModelConfiguration(model_path, 3, 1, [0, 0, 0], [1.0, 1.0, 1.0], 2, [True, True], False)
assert cfg.GetModelPath() == model_path and cfg.GetDimension() == 3
print("1) ModelConfiguration:", cfg.GetModelPath(), "dim", cfg.GetDimension(),
      "channels", cfg.GetNumberOfChannels())

# 2) The core feature-extraction filter is wrapped (GetOutput is an itk.VectorImage).
InterpolatorType = itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.F]
features = itk.ImageToFeaturesMap[ImageType, InterpolatorType].New()
features.SetModelConfiguration(cfg)
features.SetDevice("cpu")
print("2) ImageToFeaturesMap:", features.GetNameOfClass(), "device", features.GetDevice())

# 3) The metric runs Initialize() + GetValue() from Python (TorchScript inference + loss).
metric = itk.ImpactImageToImageMetricv4[ImageType, ImageType].New()
metric.SetFixedImage(fixed)
metric.SetMovingImage(moving)
metric.AddModelConfiguration(cfg)  # appends to fixed and moving lists (no std::vector needed)
metric.SetDistance(["NCC"])
metric.SetLayersWeight([1.0])
metric.SetSubsetFeatures([4])
metric.SetPCA([0])
metric.SetMode("Static")
metric.SetDevice("cpu")
transform = itk.TranslationTransform[itk.D, 3].New()
transform.SetIdentity()
metric.SetMovingTransform(transform)
metric.SetFixedTransform(itk.IdentityTransform[itk.D, 3].New())
metric.Initialize()
value = metric.GetValue()
assert math.isfinite(value)
print("3) metric:", metric.GetNameOfClass(), "device", metric.GetDevice(),
      "distance", list(metric.GetDistance()), "| GetValue() =", value)

print("OK: the torch-free IMPACT facade works end-to-end from Python (no torch import, no torch types)")

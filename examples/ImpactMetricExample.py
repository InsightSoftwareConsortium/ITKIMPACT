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

"""IMPACT: semantic similarity registration from pretrained TorchScript features.

This example shows the two Python-facing pieces of the module -- note that no
``import torch`` is needed: all LibTorch work happens inside the C++ implementation,
and the public API speaks only ITK/STL types (images, a model-path string, a device
string).

  1. the core: ``itk.ImageToFeaturesMap`` extracts a feature map (an ``itk.VectorImage``)
     from an image with a TorchScript model;
  2. the metric: ``itk.ImpactImageToImageMetricv4`` plugs into
     ``itk.ImageRegistrationMethodv4`` to register two images by comparing those features.

Run with:  ./ImpactMetricExample.py model.pt fixed.mha moving.mha
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="IMPACT feature extraction + registration demo.")
parser.add_argument("model", help="TorchScript model (.pt) returning a list of feature maps")
parser.add_argument("fixed_image")
parser.add_argument("moving_image")
parser.add_argument("--device", default="cpu", help='"cpu", "cuda", "cuda:0", ...')
args = parser.parse_args()

Dimension = 3
PixelType = itk.F
ImageType = itk.Image[PixelType, Dimension]

fixed = itk.imread(args.fixed_image, PixelType)
moving = itk.imread(args.moving_image, PixelType)

# A TorchScript model configuration: (path, dimension, channels, patchSize, voxelSize,
# overlap, layersMask, mixedPrecision). All POD/STL -- no torch type crosses to Python.
config = itk.ModelConfiguration(
    args.model, Dimension, 1, [0, 0, 0], [1.0, 1.0, 1.0], 2, [True], False
)

# --- 1. core: extract a feature map -----------------------------------------------
InterpolatorType = itk.BSplineInterpolateImageFunction[ImageType, itk.D]
interpolator = InterpolatorType.New()
interpolator.SetSplineOrder(3)

features = itk.ImageToFeaturesMap[ImageType, InterpolatorType].New()
features.SetModelConfiguration(config)
features.SetInterpolator(interpolator)
features.SetDevice(args.device)
features.AddInput(fixed)
features.Update()
feature_map = features.GetOutput(0)  # itk.VectorImage[itk.F, 3]
print("feature map:", feature_map.GetLargestPossibleRegion().GetSize(),
      "channels:", feature_map.GetNumberOfComponentsPerPixel())

# --- 2. metric: register moving onto fixed ----------------------------------------
metric = itk.ImpactImageToImageMetricv4[ImageType, ImageType].New()
metric.SetFixedImage(fixed)
metric.SetMovingImage(moving)
metric.SetFixedModelsConfiguration([config])
metric.SetMovingModelsConfiguration([config])
metric.SetDistance(["NCC"])
metric.SetLayersWeight([1.0])
metric.SetSubsetFeatures([4])
metric.SetPCA([0])
metric.SetMode("Static")
metric.SetDevice(args.device)

transform = itk.TranslationTransform[itk.D, Dimension].New()
transform.SetIdentity()
metric.SetMovingTransform(transform)
metric.SetFixedTransform(itk.IdentityTransform[itk.D, Dimension].New())

optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
optimizer.SetNumberOfIterations(200)
optimizer.SetLearningRate(2.0)
optimizer.SetMinimumStepLength(1e-4)

registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
registration.SetFixedImage(fixed)
registration.SetMovingImage(moving)
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.SetInitialTransform(transform)
registration.SetNumberOfLevels(1)
registration.Update()

print("recovered transform parameters:", list(registration.GetTransform().GetParameters()))

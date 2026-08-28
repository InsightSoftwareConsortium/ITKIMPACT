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

"""IMPACT similarity metric: a feature-based ImageToImageMetricv4 for registration.

``itk.ImpactImageToImageMetricv4`` compares two images through TorchScript features
instead of raw intensities, so it plugs into ``itk.ImageRegistrationMethodv4`` and drives
any ITK transform/optimizer -- while working across modalities. Passing the modality
invariant MIND descriptor with a normalized-cross-correlation distance gives a smooth,
well-behaved metric for e.g. CT-to-MR alignment (no ``import torch`` needed).

This example recovers a rigid misalignment between a CT (fixed) and an MR (moving).

Run:  ./ImpactMetricExample.py fixed_ct.mha moving_mr.mha mind.pt \\
          --out moved.mha --device cuda
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="IMPACT metric v4 rigid registration demo.")
parser.add_argument("fixed_image")
parser.add_argument("moving_image")
parser.add_argument("models", nargs="+", help="TorchScript feature model(s), e.g. a MIND descriptor")
parser.add_argument("--out", default="moved.mha")
parser.add_argument("--iterations", type=int, default=60)
parser.add_argument("--voxel", type=float, default=2.0)
parser.add_argument("--mask", default=None,
                    help="body mask on the fixed image; strongly recommended (see below)")
parser.add_argument("--device", default="cuda", help='"cpu", "cuda", "cuda:0", ...')
args = parser.parse_args()

Dimension = 3
ImageType = itk.Image[itk.F, Dimension]

fixed = itk.imread(args.fixed_image, itk.F)
moving = itk.imread(args.moving_image, itk.F)

voxel = [args.voxel] * Dimension

metric = itk.ImpactImageToImageMetricv4[ImageType, ImageType].New()
# Add one configuration per model (applies to both fixed and moving images).
for path in args.models:
    metric.AddModelConfiguration(
        itk.ImpactModelConfiguration(path, Dimension, 1, [0, 0, 0], voxel, 0, [True], False)
    )
metric.SetDistance(["L2"] * len(args.models))      # the distance to reach for; NCC is an ablation
metric.SetLayersWeight([1.0] * len(args.models))
metric.SetSubsetFeatures([12] * len(args.models))  # channels compared per model
metric.SetPCA([0] * len(args.models))
metric.SetMode("Static")
metric.SetDevice(args.device)
if args.mask:
    # Air agrees between modalities everywhere, so a whole-image domain measures mostly
    # background: on an abdominal MR-CT pair the similarity varies by 1e-4 over a 2 cm offset,
    # which no optimizer can follow, against 5e-2 for the same offset inside the body.
    #
    # How tight the mask is decides the result more than any other setting here. A whole-body
    # mask is mostly wall and fat, which agree well between the two modalities and outvote the
    # organs: on the pair below it costs 0.14 of organ Dice against a mask around the organs
    # themselves, and no distance, model or optimizer setting recovers that. A mask of the
    # region you want aligned is worth more than a better similarity.
    mask = itk.ImageMaskSpatialObject[Dimension].New()
    mask.SetImage(itk.imread(args.mask, itk.UC))
    mask.Update()
    metric.SetFixedImageMask(mask)

# Rigid transform, centred on the fixed image, which is the domain the moving transform maps
# from. The centre goes through the direction cosines: computing it as origin + 0.5 * spacing *
# size ignores them, and on an image stored LPS rather than RAS that lands hundreds of
# millimetres outside the anatomy. Every degree of rotation then arrives with a lever arm, and
# the registration settles on a transform that translates the organs instead of turning them.
size = fixed.GetLargestPossibleRegion().GetSize()
center = list(fixed.TransformIndexToPhysicalPoint([size[0] // 2, size[1] // 2, size[2] // 2]))
transform = itk.Euler3DTransform[itk.D].New()
transform.SetIdentity()
transform.SetCenter(center)

optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
optimizer.SetLearningRate(2.0)
optimizer.SetMinimumStepLength(1e-4)
optimizer.SetNumberOfIterations(args.iterations)
optimizer.SetRelaxationFactor(0.8)
# Let ITK derive the parameter scales from the physical shift each parameter causes, rather
# than guessing a rotation-versus-translation ratio by hand: the right ratio depends on the
# metric's own gradient magnitude, and a wrong guess walks the transform off the image.
estimator = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
estimator.SetMetric(metric)
optimizer.SetScalesEstimator(estimator)

registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
registration.SetFixedImage(fixed)
registration.SetMovingImage(moving)
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.SetInitialTransform(transform)
registration.SetNumberOfLevels(1)
registration.SetSmoothingSigmasPerLevel([0])
registration.SetShrinkFactorsPerLevel([1])
registration.Update()

final = registration.GetTransform()
print("recovered parameters:", [round(v, 3) for v in final.GetParameters()])

moved = itk.resample_image_filter(
    moving, transform=final, use_reference_image=True, reference_image=fixed, default_pixel_value=0.0
)
itk.imwrite(moved, args.out)
print("wrote", args.out)

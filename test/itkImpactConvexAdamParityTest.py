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

"""ImpactCoarseRegistration then ImpactFineRegistration against the authors' ConvexAdam.

With the authors' settings the two stages are their algorithm: on the CPU in single precision the coarse fields are
equal, the fine ones equal after two Adam iterations, and after the eighty of the default they differ by the rounding
the iterations accumulate. The authors return the field their last iteration evaluates, before its step: their n
iterations are n - 1 steps of ImpactFineRegistration. The pair is two crops of one smooth random volume a whole number of voxels apart, of sizes
that are multiples of neither 6 nor 2.

    python itkImpactConvexAdamParityTest.py <path-to-MIND/R1D2_3D.pt> [<path holding the convexAdam package>]

Needs the convexAdam package (https://github.com/multimodallearning/convexAdam): exits 77 (skipped) without it.
"""

import sys

import itk
import numpy as np
import torch

if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
try:
    from convexAdam.convex_adam_MIND import convex_adam_pt
except ImportError as error:
    print(f"skipped: {error}")
    sys.exit(77)

torch.manual_seed(0)
volume = torch.randn(1, 1, 72, 80, 92)
for _ in range(3):
    volume = torch.nn.functional.avg_pool3d(volume, 5, stride=1, padding=2)
volume = (1000.0 * volume[0, 0] / volume.std()).numpy()
fixed = np.ascontiguousarray(volume[8:57, 9:70, 10:85])  # 49 x 61 x 75
moving = np.ascontiguousarray(volume[11:60, 7:68, 14:89])  # the same texture 3, -2 and 4 voxels further


def official(iterations: int, weight: float = 1.25) -> np.ndarray:
    """The authors' field, {a, b, c, 3}: voxels along the array's axes."""
    return convex_adam_pt(
        fixed, moving, lambda_weight=weight, selected_niter=iterations, dtype=torch.float32, device="cpu"
    )


ImageType = itk.Image[itk.F, 3]
model = itk.ImpactModelConfiguration(sys.argv[1], 3, 1, [0, 0, 0], [0.0, 0.0, 0.0], [0, 0, 0], [True], False)


def stage(filter_type):
    registration = filter_type[ImageType, ImageType].New()
    registration.SetFixedImage(itk.image_from_array(fixed))
    registration.SetMovingImage(itk.image_from_array(moving))
    registration.AddModelConfiguration(model)
    registration.SetDistance(["L2"])
    registration.SetLayersWeight([1.0])
    registration.SetNormalizeLosses(False)
    registration.SetDevice("cpu")
    return registration


def field_of(registration) -> np.ndarray:
    """An ITK field of a unit-spacing image as the authors return theirs: components along the array's axes."""
    return np.array(itk.array_view_from_image(registration.GetOutput()))[..., ::-1]


coarse = stage(itk.ImpactCoarseRegistration)
coarse.SetGridSpacing(6)
coarse.SetDisplacementHalfWidth(4)
coarse.Update()


def ours(iterations: int) -> np.ndarray:
    fine = stage(itk.ImpactFineRegistration)
    fine.SetInitialDisplacementField(coarse.GetOutput())
    fine.SetNumberOfIterations(iterations - 1)
    fine.SetLearningRate(2.0)
    fine.SetRegularizationWeight(1.25)
    fine.SetGridShrinkFactor(2)
    fine.SetControlGridSmoothingIterations(3)
    fine.Update()
    return field_of(fine)


failed = False
for name, reference, field, most, mean in (
    ("coarse stage", official(1, weight=0.0), field_of(coarse), 1e-3, 1e-4),
    ("two Adam iterations", official(2), ours(2), 1e-3, 1e-4),
    ("eighty Adam iterations", official(80), ours(80), None, 0.05),
):
    difference = np.linalg.norm(field - reference, axis=-1)
    print(f"{name}: |ours - authors'| mean {difference.mean():.2e}, max {difference.max():.2e} voxels")
    if difference.mean() > mean or (most is not None and difference.max() > most):
        print(f"  FAILED: more than {mean} on average" + (f" or {most} at most" if most is not None else ""))
        failed = True
sys.exit(1 if failed else 0)

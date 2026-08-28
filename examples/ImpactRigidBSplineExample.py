#!/usr/bin/env python
# ==========================================================================
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
# ==========================================================================

"""IMPACT with the standard ITK registration pipeline: rigid, then B-spline.

The same `itk.ImpactImageToImageMetricv4` drives both stages; only the transform and the
optimizer change. The layout follows ITK's own examples: Euler3D with a scales estimator for the
rigid stage, `BSplineTransform` with `LBFGSBOptimizerv4` for the deformable one, as in
`Examples/RegistrationITKv4/DeformableRegistration12.cxx`.

Three settings are worth knowing about, because each is easy to get wrong and each decides
whether the result is an alignment or a wander.

**Mask the region you want aligned**, not the whole body. Wall and fat agree well between
modalities and outvote the organs, so a body mask sends the registration somewhere else
entirely. No labels are needed: segment the fixed image with the model matching its modality and
dilate what it does not call background.

**The B-spline stage uses no scales estimator.** A B-spline coefficient is already a displacement
in millimetres, so every coefficient has the same physical meaning and there is nothing to
equalise. Asked anyway, the estimator returns about 6.7e-07 for each of them against 1 for a
translation parameter, and a gradient descent divides its update by that. LBFGSB is quasi-Newton
and builds its own conditioning, which is why ITK pairs it with the B-spline transform.

**One resolution level.** ITK's shrink factors never reach the model: `ImageToFeaturesMap`
resamples whatever it is handed to the `ImpactModelConfiguration` voxel size, so a shrunk image is
interpolated straight back up before the network sees it, leaving the smoothing and the lost
detail. Coarsening the model's voxel size is the real knob, and for a local descriptor like MIND
it describes different anatomy rather than the same anatomy more coarsely.

A finer B-spline mesh is not reached by chaining two registrations: ITK cannot compose a B-spline
as a moving initial transform (`ComputeJacobianWithRespectToPosition` is not implemented for it),
which is what `itk::BSplineTransformParametersAdaptor` exists for. That class is not currently
wrapped for Python, so a coarse-to-fine schedule needs C++.

Run:  ./ImpactRigidBSplineExample.py fixed.nii.gz moving.nii.gz mind.pt \\
          --mask organs.nii.gz --grid-spacing 40 --out moved.nii.gz --device cuda:0
"""

import argparse

import itk

parser = argparse.ArgumentParser(description="IMPACT rigid then B-spline registration.")
parser.add_argument("fixed_image")
parser.add_argument("moving_image")
parser.add_argument("models", nargs="+", help="one or more TorchScript feature models (.pt)")
parser.add_argument("--mask", default=None,
                    help="fixed-image mask, around the organs rather than the whole body; "
                         "see the docstring, it is the setting that matters most")
parser.add_argument("--out", default="moved.nii.gz", help="warped moving image")
parser.add_argument("--voxel", type=float, default=2.0, help="feature voxel size in mm")
parser.add_argument("--grid-spacing", type=float, default=20.0,
                    help="distance between B-spline control points, in mm. This is the knob that\n"
                         "decides what the deformation can express: too coarse and one control\n"
                         "point drags a whole region, which shows up as one structure improving\n"
                         "while its neighbour is pulled apart. elastix targets 10 mm for this\n"
                         "kind of abdominal case (FinalGridSpacingInPhysicalUnits)")
parser.add_argument("--sampling", type=float, default=0.10, help="fraction of voxels sampled")
parser.add_argument("--shrink-factors", type=int, nargs="+", default=[1],
                    help="one shrink factor per resolution level, coarsest first")
parser.add_argument("--smoothing-sigmas", type=float, nargs="+", default=[0.0],
                    help="Gaussian sigma per level, in millimetres, matching --shrink-factors")
parser.add_argument("--rigid-iterations", type=int, default=300)
parser.add_argument("--bspline-iterations", type=int, default=200,
                    help="also caps the LBFGSB function evaluations; the cost of the run is "
                         "roughly this times the sampled fraction of the volume")
parser.add_argument("--save-transform", default=None,
                    help="write the composed rigid+B-spline transform (.tfm/.h5), so the same\n"
                         "alignment can be applied to a segmentation or another sequence")
parser.add_argument("--device", default="cuda:0", help='"cpu", "cuda", "cuda:0", ...')
args = parser.parse_args()

if len(args.shrink_factors) != len(args.smoothing_sigmas):
    parser.error("--shrink-factors and --smoothing-sigmas carry one entry per level, "
                 f"got {len(args.shrink_factors)} and {len(args.smoothing_sigmas)}")

Dimension = 3
SplineOrder = 3
ImageType = itk.Image[itk.F, Dimension]

fixed = itk.imread(args.fixed_image, itk.F)
moving = itk.imread(args.moving_image, itk.F)

# --- the metric, shared by both stages ---------------------------------------------------
metric = itk.ImpactImageToImageMetricv4[ImageType, ImageType].New()
for path in args.models:
    metric.AddModelConfiguration(
        itk.ImpactModelConfiguration(path, Dimension, 1, [0, 0, 0], [args.voxel] * Dimension, 0, [True], False)
    )
metric.SetDistance(["L2"] * len(args.models))
metric.SetLayersWeight([1.0] * len(args.models))
metric.SetSubsetFeatures([12] * len(args.models))
metric.SetPCA([0] * len(args.models))
metric.SetMode("Static")
metric.SetDevice(args.device)

if args.mask:
    # Air agrees between modalities everywhere, so a whole-image domain measures mostly
    # background and the similarity barely varies with the transform.
    mask = itk.ImageMaskSpatialObject[Dimension].New()
    mask.SetImage(itk.imread(args.mask, itk.UC))
    mask.Update()
    metric.SetFixedImageMask(mask)

identity = itk.IdentityTransform[itk.D, Dimension].New()


def pyramid(registration, shrink_factors, smoothing_sigmas):
    """Resolution levels and sampling: see the module docstring."""
    registration.SetNumberOfLevels(len(shrink_factors))
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration.SetMetricSamplingStrategy(2)  # 0 NONE, 1 REGULAR, 2 RANDOM
    registration.SetMetricSamplingPercentage(args.sampling)


# --- 1. rigid ------------------------------------------------------------------------------
size = fixed.GetLargestPossibleRegion().GetSize()
centre = list(fixed.TransformIndexToPhysicalPoint([size[0] // 2, size[1] // 2, size[2] // 2]))

rigid = itk.Euler3DTransform[itk.D].New()
rigid.SetIdentity()
rigid.SetCenter(centre)

rigid_optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
rigid_optimizer.SetLearningRate(2.0)
rigid_optimizer.SetMinimumStepLength(1e-4)
rigid_optimizer.SetRelaxationFactor(0.8)
rigid_optimizer.SetNumberOfIterations(args.rigid_iterations)
# A radian and a millimetre are not comparable; let ITK derive the ratio from the physical
# shift each parameter causes, measured against this metric's own gradient.
rigid_scales = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
rigid_scales.SetMetric(metric)
rigid_optimizer.SetScalesEstimator(rigid_scales)

rigid_registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
rigid_registration.SetFixedImage(fixed)
rigid_registration.SetMovingImage(moving)
rigid_registration.SetMetric(metric)
rigid_registration.SetOptimizer(rigid_optimizer)
rigid_registration.SetInitialTransform(rigid)
pyramid(rigid_registration, args.shrink_factors, args.smoothing_sigmas)
rigid_registration.Update()
print("rigid :", rigid_optimizer.GetStopConditionDescription())

# --- 2. B-spline, on top of the rigid ------------------------------------------------------
# The transform domain covers the fixed image, as ITK's example sets it out.
physical_dimensions = [
    fixed.GetSpacing()[i] * (fixed.GetLargestPossibleRegion().GetSize()[i] - 1) for i in range(Dimension)
]
bspline = itk.BSplineTransform[itk.D, Dimension, SplineOrder].New()
bspline.SetTransformDomainOrigin(fixed.GetOrigin())
bspline.SetTransformDomainPhysicalDimensions(physical_dimensions)
bspline.SetTransformDomainDirection(fixed.GetDirection())
# Mesh size from the requested physical spacing, so the grid does not silently change
# meaning when the image geometry does.
mesh_size = [max(1, int(round(physical_dimensions[i] / args.grid_spacing))) for i in range(Dimension)]
bspline.SetTransformDomainMeshSize(mesh_size)
print(f"bspline grid: mesh {mesh_size}, "
      f"{[round(physical_dimensions[i] / mesh_size[i]) for i in range(Dimension)]} mm apart, "
      f"{bspline.GetNumberOfParameters()} parameters")

number_of_parameters = bspline.GetNumberOfParameters()
bspline_optimizer = itk.LBFGSBOptimizerv4.New()
# All bounds deselected, so the coefficients are free: LBFGSB is used here for its
# quasi-Newton conditioning, not for its bounds.
bspline_optimizer.SetBoundSelection(itk.Array[itk.SL]([0] * number_of_parameters))
bspline_optimizer.SetLowerBound(itk.Array[itk.D]([0.0] * number_of_parameters))
bspline_optimizer.SetUpperBound(itk.Array[itk.D]([0.0] * number_of_parameters))
bspline_optimizer.SetCostFunctionConvergenceFactor(1e7)
bspline_optimizer.SetGradientConvergenceTolerance(1e-35)
bspline_optimizer.SetNumberOfIterations(args.bspline_iterations)
bspline_optimizer.SetMaximumNumberOfFunctionEvaluations(args.bspline_iterations)
bspline_optimizer.SetMaximumNumberOfCorrections(7)
# Deliberately no scales estimator: see the module docstring.

bspline_registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
bspline_registration.SetFixedImage(fixed)
bspline_registration.SetMovingImage(moving)
bspline_registration.SetMetric(metric)
bspline_registration.SetOptimizer(bspline_optimizer)
bspline_registration.SetInitialTransform(bspline)
# The rigid result is held in front of the B-spline rather than folded into it, which is how
# ITKv4 composes a prior stage. A rigid transform can sit here; a B-spline cannot.
bspline_registration.SetMovingInitialTransform(rigid_registration.GetTransform())
# One level, unlike the rigid stage. A pyramid only helps a deformable stage if the control
# grid is coarsened along with the image; held at its final spacing while the image shrinks by
# four, the grid is fine relative to the blurred content and fits it, which the finer levels
# then inherit. Coarsening it is what BSplineTransformParametersAdaptor does, and that class is
# not wrapped for Python.
pyramid(bspline_registration, [1], [0.0])
bspline_registration.Update()
print("bspline :", bspline_optimizer.GetStopConditionDescription())

# --- 3. resample through both ---------------------------------------------------------------
# CompositeTransform applies the last added first, so this is rigid(bspline(point)) -- the
# same composition ITKv4 made internally during the second stage.
composite = itk.CompositeTransform[itk.D, Dimension].New()
composite.AddTransform(rigid_registration.GetTransform())
composite.AddTransform(bspline)

itk.imwrite(
    itk.resample_image_filter(
        moving, transform=composite, use_reference_image=True, reference_image=fixed,
        default_pixel_value=-1024.0,
    ),
    args.out,
)
if args.save_transform:
    itk.transformwrite([composite], args.save_transform)
    print("wrote", args.save_transform)
print("wrote", args.out)

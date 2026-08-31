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

"""IMPACT ConvexAdam: dense, multi-modal deformable registration.

``itk.ImpactCoarseRegistration`` runs a ConvexAdam-style Adam optimisation over a dense
displacement field, driven by the semantic similarity of TorchScript features (no
``import torch`` needed). ``itk.ImpactFineRegistration`` refines that field. Because the
similarity is computed on learned + hand-crafted (MIND) features rather than raw
intensities, it aligns *different modalities* -- e.g. CT to MR.

Each model is an ``itk.ImpactModelConfiguration``; passing several (here a deep segmentation
model and the modality-invariant MIND descriptor) combines their features. The output is
an ``itk.DisplacementField`` (== GetOutput()) on the fixed grid, in millimetres, plus a
ready-to-use ``itk.DisplacementFieldTransform``.

Run:  ./ImpactConvexAdamExample.py fixed.mha moving.mha model0.pt [model1.pt ...] \\
          --out moved.mha --device cuda
"""

import argparse
import itk

parser = argparse.ArgumentParser(description="IMPACT ConvexAdam deformable registration demo.")
parser.add_argument("fixed_image")
parser.add_argument("moving_image")
parser.add_argument("models", nargs="+", help="one or more TorchScript feature models (.pt)")
parser.add_argument("--out", default="moved.mha", help="warped moving image output")
parser.add_argument("--voxel", type=float, default=2.0, help="feature voxel size in mm")
parser.add_argument("--device", default="cuda", help='"cpu", "cuda", "cuda:0", ...')
args = parser.parse_args()

Dimension = 3
ImageType = itk.Image[itk.F, Dimension]

fixed = itk.imread(args.fixed_image, itk.F)
moving = itk.imread(args.moving_image, itk.F)

# One configuration per model. The layers mask selects which model outputs to use as
# features; here each model exposes a single kept layer ([True]). For a multi-layer
# network, give a mask of its layer count, e.g. [False]*7 + [True] for the last layer.
voxel = [args.voxel] * Dimension
def add_models(reg):
    for path in args.models:
        reg.AddModelConfiguration(
            itk.ImpactModelConfiguration(path, Dimension, 1, [0, 0, 0], voxel, [0, 0, 0], [True], False)
        )

# --- coarse field -----------------------------------------------------------------
coarse = itk.ImpactCoarseRegistration[ImageType, ImageType].New()
coarse.SetFixedImage(fixed)
coarse.SetMovingImage(moving)
add_models(coarse)                       # both fixed & moving use these models
coarse.SetDevice(args.device)
coarse.Update()

# --- fine refinement --------------------------------------------------------------
fine = itk.ImpactFineRegistration[ImageType, ImageType].New()
fine.SetFixedImage(fixed)
fine.SetMovingImage(moving)
add_models(fine)
fine.SetInitialDisplacementField(coarse.GetOutput())
fine.SetDevice(args.device)
fine.Update()

displacement_field = fine.GetOutput()  # itk.Image[itk.Vector[itk.F, 3], 3], millimetres

# --- warp the moving image onto the fixed grid ------------------------------------
warp = itk.WarpImageFilter[ImageType, ImageType, type(displacement_field)].New()
warp.SetInput(moving)
warp.SetDisplacementField(displacement_field)
warp.SetOutputParametersFromImage(fixed)
warp.Update()

itk.imwrite(warp.GetOutput(), args.out)
print("wrote", args.out)

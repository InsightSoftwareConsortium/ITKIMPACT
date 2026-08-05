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

"""IMPACT inference engine: a TorchScript model as an ITK feature extractor.

``itk.ImageToFeaturesMap`` runs any patch-based TorchScript model on an ITK image and
returns each requested layer as an ``itk.VectorImage[itk.F, dim]`` -- no ``import torch``
needed. A single forward pass exposes the whole hierarchy: from low-level features to,
for a segmentation network, the final label logits. Which layers you keep is chosen by
the boolean ``layersMask`` in the ``itk.ModelConfiguration``.

This example runs a TotalSegmentator-style model on a CT and keeps two layers at once:
a mid-level feature layer and the final segmentation layer.

Run:  ./ImpactInferenceExample.py model.pt ct.mha \\
          --feature-out features.mha --seg-out segmentation.mha --device cuda
"""

import argparse
import itk
import numpy as np

parser = argparse.ArgumentParser(description="IMPACT feature/segmentation extraction demo.")
parser.add_argument("model", help="TorchScript model (.pt) returning a list of layer tensors")
parser.add_argument("image", help="input image (e.g. a CT)")
parser.add_argument("--feature-layer", type=int, default=2, help="0-based index of the layer kept as features")
parser.add_argument("--seg-layer", type=int, default=7, help="0-based index of the layer kept as segmentation logits")
parser.add_argument("--num-layers", type=int, default=8, help="number of layers the model returns")
parser.add_argument("--voxel", type=float, default=1.5, help="model voxel size in mm")
parser.add_argument("--patch", type=int, default=0,
                    help="patch size in voxels; 0 (default) runs the whole image in one pass")
parser.add_argument("--overlap", type=int, default=32,
                    help="patch overlap in voxels, blended on assembly")
parser.add_argument("--combine", default="cosinus",
                    help="blend window: mean, cosinus, gaussian (nnU-Net's, as TotalSegmentator "
                         "uses) or trim (select instead of average, for a label map)")
parser.add_argument("--feature-out", default="features.mha")
parser.add_argument("--seg-out", default="segmentation.mha")
parser.add_argument("--device", default="cuda", help='"cpu", "cuda", "cuda:0", ...')
args = parser.parse_args()

Dimension = 3
ImageType = itk.Image[itk.F, Dimension]

image = itk.imread(args.image, itk.F)

# Keep the feature layer and the segmentation layer from one forward pass.
mask = [False] * args.num_layers
mask[args.feature_layer] = True
mask[args.seg_layer] = True
# A patch size of 0 runs the whole image through the model in one pass. Patch tiling bounds
# memory on volumes that do not fit on the device, and gives the model the field of view it was
# trained on; overlapping patches are blended into a partition of unity, so the assembled map
# lies on the resampled grid whatever the overlap. A larger overlap costs patches and buys
# accuracy near the patch borders, where the model has least context.
voxel = [args.voxel] * Dimension
patch = [args.patch] * Dimension
config = itk.ModelConfiguration(args.model, Dimension, 1, patch, voxel, args.overlap, mask, False)
config.SetPatchCombine(args.combine)
# The overlap can also be set per axis, which is what an anisotropic patch needs:
#   config.SetOverlaps([16, 16, 8])

InterpolatorType = itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.F]

features = itk.ImageToFeaturesMap[ImageType, InterpolatorType].New()
features.SetModelConfiguration(config)
features.SetDevice(args.device)   # the cubic B-spline interpolator is the default
features.AddInput(image)
features.Update()

# Kept layers come out in order: output 0 = feature layer, output 1 = segmentation layer.
feature_map = features.GetOutput(0)      # itk.VectorImage[itk.F, 3], C feature channels
seg_logits = features.GetOutput(1)       # itk.VectorImage[itk.F, 3], one channel per class
print("feature map:", feature_map.GetLargestPossibleRegion().GetSize(),
      "channels:", feature_map.GetNumberOfComponentsPerPixel())
print("segmentation logits:", seg_logits.GetLargestPossibleRegion().GetSize(),
      "classes:", seg_logits.GetNumberOfComponentsPerPixel())

itk.imwrite(feature_map, args.feature_out)

# Collapse the segmentation logits to a label map (argmax over channels).
logits = itk.array_view_from_image(seg_logits)          # z, y, x, classes
labels = logits.argmax(axis=-1).astype(np.uint8)        # z, y, x
label_image = itk.image_from_array(labels)
label_image.SetSpacing(seg_logits.GetSpacing())
label_image.SetOrigin(seg_logits.GetOrigin())
label_image.SetDirection(seg_logits.GetDirection())
itk.imwrite(label_image, args.seg_out)
print("wrote", args.feature_out, "and", args.seg_out)

# ITKImpact examples

Five runnable scripts, each self-contained. Everything below runs from Python with no
`import torch` — the LibTorch inside your installed `torch` is reused by the wheel.

```bash
pip install itk-impact
```

You also need a TorchScript model. The ones used here come from the
[impact-torchscript-models](https://huggingface.co/VBoussot/impact-torchscript-models) collection:

```python
from huggingface_hub import snapshot_download
root = snapshot_download("VBoussot/impact-torchscript-models")
# root/TS/M291.pt        TotalSegmentator organs, 3D, 1 channel, 8 layers
# root/MRSeg/MRSeg.pt    MRSegmentator, 3D, 1 channel
# root/MIND/R1D2_3D.pt   MIND descriptor, 3D — no training, useful across modalities
# root/SAM2.1/…, root/Dino/…, root/VGG/…   2D encoders, 3 channels
```

Give the models **raw intensities** — Hounsfield units for CT, raw signal for MR. The
normalisation is baked into the TorchScript export; doing it yourself a second time gives an
empty result.

---

## 1. Inference — one pass, every layer

[`ImpactInferenceExample.py`](ImpactInferenceExample.py)

`itk.ImageToFeaturesMap` turns any patch-based TorchScript model into an ITK filter: an image
goes in, an `itk.VectorImage` comes out per layer you asked for. A single forward pass exposes
the whole hierarchy — mid-level features and, for a segmentation network, the final class
logits — because the `layersMask` selects which layers are kept.

![CT, feature map, segmentation](images/example-inference.png)

```python
import itk

image = itk.imread("ct.nii.gz", itk.F)

mask = [False] * 8
mask[2] = True          # a mid-level feature layer
mask[7] = True          # the final logits

config = itk.ImpactModelConfiguration(
    "TS/M291.pt",       # model
    3,                  # dimension the model expects
    1,                  # input channels
    [128, 128, 128],    # patch size
    [1.5, 1.5, 1.5],    # voxel size the model was trained at, per IMAGE axis
    32,                 # overlap between patches, in voxels
    mask,
    False,              # mixed precision
)
config.SetPatchCombine("gaussian")   # nnU-Net's blend window

ImageType = itk.Image[itk.F, 3]
features = itk.ImageToFeaturesMap[
    ImageType, itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.F]
].New()
features.SetModelConfiguration(config)
features.SetDevice("cuda:0")     # or "cpu"
features.AddInput(image)
features.Update()

feature_map = features.GetOutput(0)   # 128 channels at 6 mm
logits      = features.GetOutput(1)   # 25 channels at 1.5 mm
```

Kept layers come out **in model order**, whatever their index: `GetOutput(0)` is the first
`True` in the mask.

### Patch tiling

The volume is resampled to the model's voxel size and cut into overlapping patches; each patch
is weighted by a blend window and accumulated, so the assembled map lies on the resampled grid
whatever the overlap. Four windows are available through `SetPatchCombine`:

| window | what it does | use it for |
|---|---|---|
| `cosinus` *(default)* | raised-cosine taper, an exact partition of unity | feature maps |
| `gaussian` | nnU-Net's importance weighting | reproducing TotalSegmentator / MRSegmentator |
| `mean` | plain average of the patches covering a voxel | a baseline |
| `trim` | keeps each patch's central band instead of averaging | a label map, where averaging would invent classes between classes |

A larger overlap costs patches and buys accuracy near the patch borders, where the model has
least context. Against the official TotalSegmentator on an abdominal CT, macro Dice goes
0.823 → 0.877 → 0.889 → 0.892 for overlaps 0 / 16 / 32 / 64. The overlap can also be set per
axis, which is what an anisotropic patch needs:

```python
config.SetOverlaps([32, 32, 16])
```

A patch size of `0` on every axis runs the whole image in one pass, which is exact and fine for
small volumes.

The same tiling serves the registration filters: `ImpactCoarseRegistration` and
`ImpactFineRegistration` read the patch size and overlap of the configurations you give them, so
a model can be run over a volume that does not fit whole, with its trained field of view. The one
exception is the fine stage's differentiable-feature mode, which re-runs the network inside the
Adam loop and keeps the whole volume so the autograd graph stays a single piece.

---

## 2. Segmentation — the last layer, arg-maxed

[`MakeExampleImages.py`](MakeExampleImages.py) regenerates the figures on this page, and doubles
as a worked example: the last layer of a segmentation network is a stack of class logits, so
`numpy.argmax` over the channel axis is the label map.

![Axial, coronal and sagittal views of 20 segmented organs](images/example-segmentation.png)

```python
import numpy as np

logits = features.GetOutput(1)
labels = np.argmax(itk.array_view_from_image(logits), axis=-1).astype(np.uint8)

label_image = itk.image_from_array(labels)
label_image.CopyInformation(logits)     # same grid, spacing, origin and direction
itk.imwrite(label_image, "segmentation.nii.gz")
```

The map lands on the model's resampled grid, not the input's — resample it back if you need the
original sampling. Run it yourself with:

```bash
./MakeExampleImages.py TS/M291.pt ct.nii.gz --outdir images
```

### A 2D model on a 3D volume

`SAM2.1`, `DINOv2` and `VGG` are 2D networks. Applied to a volume they are swept over it, one
slice at a time along the last image axis; the swept axis keeps its extent and its spacing.
Declare the patch size on the model's axes and the voxel size on the image's:

```python
config = itk.ImpactModelConfiguration(
    "SAM2.1/SAM2.1_Small.pt", 2, 3, [128, 128], [1.5, 1.5, 3.0], 32, [True], False
)
```

---

## 3. Similarity metric — registration in feature space

[`ImpactMetricExample.py`](ImpactMetricExample.py) · [`ImpactMetricExample.cxx`](ImpactMetricExample.cxx)

`itk.ImpactImageToImageMetricv4` compares images through features rather than intensities and
plugs straight into `itk.ImageRegistrationMethodv4` — only the comparison changes. Driven by the
modality-invariant MIND descriptor with an L2 distance, it aligns an abdominal **CT to an MR**.

![MR, CT, before and after registration](images/example-metricv4.png)

Three things decide whether this works at all, and all three are easy to get wrong:

**Mask the region you want aligned.** Air agrees between modalities everywhere, so a whole-image
domain measures mostly background and the similarity barely moves with the transform. A body mask
is little better: wall and fat agree well between modalities and outvote the organs. Mask what
you want aligned.

```python
mask = itk.ImageMaskSpatialObject[3].New()
mask.SetImage(itk.imread("organs.nii.gz", itk.UC))
mask.Update()
metric.SetFixedImageMask(mask)
```

No labels needed to build one: segment the fixed image with the model matching its modality and
dilate what it does not call background. `organ_mask()` in
[`MakeRegistrationImages.py`](MakeRegistrationImages.py) does exactly that.

**Let ITK estimate the optimizer scales.** The rotation-versus-translation ratio depends on the
metric's own gradient magnitude; guessing it by hand walks the transform off the image.

```python
estimator = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
estimator.SetMetric(metric)
optimizer.SetScalesEstimator(estimator)
```

**Centre the rigid transform through the direction cosines**, on the fixed image, which is the
domain the moving transform maps from. `origin + 0.5 * spacing * size` ignores them: on an image
stored LPS rather than RAS it names a point hundreds of millimetres outside the anatomy, and
every degree of rotation then arrives with that much lever arm.

```python
size = fixed.GetLargestPossibleRegion().GetSize()
centre = list(fixed.TransformIndexToPhysicalPoint([size[0] // 2, size[1] // 2, size[2] // 2]))
```

Two modes:

- **`Static`** — feature maps are computed once per resolution level and then interpolated.
  Fast, and what you want for a large model.
- **`Jacobian`** — a patch is extracted around every sampled point at each iteration and the
  gradient is backpropagated through the model. Slower, but nothing is precomputed, so the model
  sees the moving image as the transform currently places it.

---

## 4. ConvexAdam deformable — dense registration on the GPU

[`ImpactConvexAdamExample.py`](ImpactConvexAdamExample.py)

`itk.ImpactCoarseRegistration` builds a coarse displacement field from a cost volume, then
`itk.ImpactFineRegistration` refines it with Adam on the same features — a dense, multi-modal
deformable registration entirely on the GPU.

![MR, before, after, displacement field](images/example-convexadam.png)

The two filters refine; they are not meant to walk out of a centimetre-scale rigid offset on
their own, so they are given the rigid result and the panel answers what the deformable step
*adds*: the field takes up the deformation between the two acquisitions, which no rigid transform
can express.

Each panel lays the MR organs (teal) over the registered CT ones (orange), in axial and coronal.
One plane is not enough to judge an alignment: a cranio-caudal offset barely disturbs an axial
slice while moving every organ, which is why the coronal panels are there. The last panel is the
Jacobian determinant of the field: below 1 the tissue was compressed, above 1 expanded, and at or
below 0 the field folded, which the caption reports as a percentage.

Leaving the model configuration empty makes both filters fall back to a raw-intensity (MSE)
similarity, which is a useful baseline to compare against.

Both figures are regenerated by [`MakeRegistrationImages.py`](MakeRegistrationImages.py), which
also prints the per-organ Dice and builds the organ mask for you:

```bash
./MakeRegistrationImages.py fixed_MR.nii.gz moving_CT.nii.gz MIND/R1D2_3D.pt \
    --fixed-labels f.nii.gz --moving-labels m.nii.gz --mask-model MRSeg/MRSeg.pt
```

---

## 5. Rigid then B-spline — the metric in ITK's own pipeline

[`ImpactRigidBSplineExample.py`](ImpactRigidBSplineExample.py)

The same metric drives both stages; only the transform and the optimizer change. Euler3D with a
scales estimator, then `BSplineTransform` with `LBFGSBOptimizerv4`, as ITK lays it out in
`Examples/RegistrationITKv4/DeformableRegistration12.cxx`. The composed transform can be written
out, so the alignment carries to a segmentation or another sequence.

Use it when the pipeline has to be an ITKv4 one. When what matters is the alignment itself, the
dedicated filters of section 4 go further and take seconds rather than minutes: ITK accumulates
the metric derivative densely over every transform parameter for every sampled point, which caps
how many evaluations of a control grid are affordable.

```bash
./ImpactRigidBSplineExample.py fixed_MR.nii.gz moving_CT.nii.gz MIND/R1D2_3D.pt \
    --mask organs.nii.gz --grid-spacing 40 --sampling 0.03 --bspline-iterations 60
```

The script's docstring covers the two settings that decide the outcome: the mask, again, and a
single resolution level, because ITK's shrink factors never reach the model. `ImageToFeaturesMap`
resamples whatever it is handed to the `ImpactModelConfiguration` voxel size, so a shrunk image is
interpolated straight back up before the network sees it.

---

## Data used on this page

The CT in sections 1 and 2 is case `pair_0001_0004` of
[AMOS22](https://amos22.grand-challenge.org/). The CT–MR pair every registration number on this
page is measured on is case 1 of the AbdomenMRCT task of
[Learn2Reg](https://learn2reg.grand-challenge.org/), which ships the organ labels the Dice is
computed against. Any CT works for sections 1 and 2 — pass your own to the scripts.

Figures are produced with matplotlib, which the module itself does not depend on:

```bash
pip install matplotlib
```

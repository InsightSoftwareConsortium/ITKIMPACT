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

"""Regenerate the registration figures, with the accuracy they actually reach.

Both figures are measured, not just drawn: the organ labels that ship with the case are pushed
through the recovered transform, and the Dice before and after goes in the caption. Needs
matplotlib, which the module itself does not depend on.

Run:  ./MakeRegistrationImages.py fixed.nii.gz moving.nii.gz mind.pt \\
          --fixed-labels f.nii.gz --moving-labels m.nii.gz --outdir images
"""

import argparse
import os
import time

import itk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUND = "#151b22"
FRAME = "#4bbec7"
LABEL = "#e6edf2"
EDGE = "#f0a52a"
REFERENCE = "#4bbec7"

I3, UC3 = itk.Image[itk.F, 3], itk.Image[itk.UC, 3]


def parse():
    p = argparse.ArgumentParser(description="Regenerate the registration README figures.")
    p.add_argument("fixed_image")
    p.add_argument("moving_image")
    p.add_argument("models", nargs="+")
    p.add_argument("--fixed-labels", required=True)
    p.add_argument("--moving-labels", required=True)
    p.add_argument("--fixed-mask", default=None,
                   help="body mask; without one the metric measures mostly air")
    p.add_argument("--outdir", default="images")
    p.add_argument("--voxel", type=float, default=2.0)
    p.add_argument("--device", default="cuda:0")
    # The pair ships rigidly aligned, so the rigid demo has to create something to recover.
    p.add_argument("--misalign", type=float, nargs=6, default=[0.06, -0.04, 0.05, 12.0, -8.0, 6.0],
                   help="rx ry rz (rad) tx ty tz (mm) applied to the moving image before registering")
    p.add_argument("--sampling", type=float, default=0.10,
                   help="fraction of voxels the metric samples per iteration")
    return p.parse_args()


def configs(args):
    for path in args.models:
        yield itk.ModelConfiguration(path, 3, 1, [0, 0, 0], [args.voxel] * 3, 0, [True], False)


def dice(reference, candidate):
    """Per-label Dice, over the labels present in BOTH images -- a label the moving image does
    not contain has no counterpart to align, and averaging its zero would hide the result."""
    scores = {}
    for c in np.unique(reference):
        if c == 0:
            continue
        r, p = reference == c, candidate == c
        # Either side empty means the label has no counterpart, so there is nothing to align
        # and its zero says nothing about the registration. The test used to be "and", which
        # kept exactly the case the docstring excludes: on this pair label 4 is absent from the
        # moving image entirely, scored a flat zero at every stage, and pulled every reported
        # mean down by a quarter.
        if p.sum() == 0 or r.sum() == 0:
            continue
        scores[int(c)] = 2 * np.logical_and(p, r).sum() / (p.sum() + r.sum())
    return scores


def edges(volume, quantile=0.93):
    """A thin edge map, to lay one modality over the other."""
    gradient = np.sqrt(sum(np.gradient(volume.astype(np.float32), axis=a) ** 2 for a in range(2)))
    return gradient > np.quantile(gradient, quantile)


def organ_contour(labels):
    """The outline of the labelled organs. An intensity edge map is dominated by the body/air
    boundary and by bone, so it draws the skin rather than the structures Dice is computed on:
    the caption then reports one thing and the picture shows another."""
    body = labels > 0
    inner = body.copy()
    for axis in (0, 1):
        for shift in (1, -1):
            inner &= np.roll(body, shift, axis=axis)
    return body & ~inner


def checkerboard(reference, candidate, tiles=8):
    """Alternate tiles of the two images. Structures that cross a tile boundary without a step
    are aligned there; a visible break is a residual offset, read directly off the anatomy
    rather than off a contour someone had to choose a threshold for."""
    rows, columns = reference.shape
    size = max(rows, columns) // tiles or 1
    y, x = np.ogrid[:rows, :columns]
    take_reference = ((y // size) + (x // size)) % 2 == 0
    return np.where(take_reference, reference, candidate)


def normalise(volume):
    low, high = np.percentile(volume, (1, 99))
    return np.clip((volume - low) / max(high - low, 1e-6), 0, 1)


def plate(ax, background, edge=None, title="", detail="", reference=None):
    ax.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    # The reference outline goes underneath, so that where the two agree the moving one covers
    # it: alignment reads as a single contour, misalignment as two.
    for mask, colour in ((reference, REFERENCE), (edge, EDGE)):
        if mask is not None:
            ax.imshow(np.ma.masked_where(~mask, mask.astype(float)),
                      cmap=matplotlib.colors.ListedColormap([colour]),
                      interpolation="nearest", alpha=0.9)
    ax.set_facecolor(GROUND)
    for spine in ax.spines.values():
        spine.set_edgecolor(FRAME)
        spine.set_linewidth(1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"$\\bf{{{title}}}$\n{detail}", color=LABEL, fontsize=9, labelpad=8, linespacing=1.7)
    ax.xaxis.label.set_horizontalalignment("left")
    ax.xaxis.set_label_coords(0, -0.04)


def row(views, width, name, outdir, suptitle):
    ratios = [v[0].shape[1] / v[0].shape[0] for v in views]
    fig = plt.figure(figsize=(width, width / sum(ratios) * 1.30), facecolor=GROUND)
    grid = fig.add_gridspec(1, len(views), width_ratios=ratios, wspace=0.025)
    for i, (background, edge, title, detail, reference) in enumerate(views):
        plate(fig.add_subplot(grid[0, i]), background, edge, title, detail, reference)
    fig.text(0.008, 0.985, suptitle, color=FRAME, fontsize=9.5, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.008, right=0.992, top=0.945, bottom=0.14)
    fig.savefig(os.path.join(outdir, name), dpi=140, facecolor=GROUND)
    plt.close(fig)
    print("wrote", name)


def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    fixed = itk.imread(args.fixed_image, itk.F)
    moving = itk.imread(args.moving_image, itk.F)
    fixed_labels = itk.imread(args.fixed_labels, itk.UC)
    moving_labels = itk.imread(args.moving_labels, itk.UC)
    nearest = itk.NearestNeighborInterpolateImageFunction[UC3, itk.D].New()

    reference = itk.array_from_image(fixed_labels)

    # The pair ships rigidly aligned, which is the ceiling any rigid step can reach; a known
    # misalignment is applied so that there is something to recover, and the recovered Dice can
    # be read against that ceiling rather than against nothing.
    centre = [o + 0.5 * s * n for o, s, n in zip(
        moving.GetOrigin(), moving.GetSpacing(), moving.GetLargestPossibleRegion().GetSize())]
    misalignment = itk.Euler3DTransform[itk.D].New()
    misalignment.SetIdentity()
    misalignment.SetCenter(centre)
    misalignment.SetParameters(itk.OptimizerParameters[itk.D](list(args.misalign)))

    aligned = dice(reference, itk.array_from_image(itk.resample_image_filter(
        moving_labels, use_reference_image=True, reference_image=fixed_labels, interpolator=nearest)))
    moving = itk.resample_image_filter(
        moving, transform=misalignment, use_reference_image=True, reference_image=moving,
        default_pixel_value=-1024.0)
    moving_labels = itk.resample_image_filter(
        moving_labels, transform=misalignment, use_reference_image=True, reference_image=moving_labels,
        interpolator=nearest)

    on_fixed = itk.resample_image_filter(
        moving, use_reference_image=True, reference_image=fixed, default_pixel_value=-1024.0
    )
    before_labels = itk.array_from_image(
        itk.resample_image_filter(moving_labels, use_reference_image=True,
                                  reference_image=fixed_labels, interpolator=nearest)
    )
    before = dice(reference, before_labels)

    # --- rigid, through the feature metric ------------------------------------------------
    metric = itk.ImpactImageToImageMetricv4[I3, I3].New()
    models = list(configs(args))
    for c in models:
        metric.AddModelConfiguration(c)
    metric.SetDistance(["NCC"] * len(models))
    metric.SetLayersWeight([1.0] * len(models))
    metric.SetSubsetFeatures([12] * len(models))
    metric.SetPCA([0] * len(models))
    metric.SetMode("Static")
    metric.SetDevice(args.device)
    if args.fixed_mask:
        # Air agrees between modalities everywhere, so a whole-image domain measures mostly
        # background: the similarity varies by 1e-4 over a two-centimetre offset, which no
        # optimiser can follow. Restricted to the body it varies by 5e-2 over the same range.
        mask = itk.ImageMaskSpatialObject[3].New()
        mask.SetImage(itk.imread(args.fixed_mask, itk.UC))
        mask.Update()
        metric.SetFixedImageMask(mask)

    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetIdentity()
    transform.SetCenter(centre)
    optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
    optimizer.SetLearningRate(2.0)
    optimizer.SetMinimumStepLength(1e-4)
    optimizer.SetNumberOfIterations(300)
    optimizer.SetRelaxationFactor(0.8)
    # Let ITK derive the parameter scales from the physical shift each parameter causes. Hand-set
    # scales have to be guessed against the metric's own gradient magnitude, and a wrong guess
    # sends a rigid transform off the image rather than onto the anatomy.
    estimator = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
    estimator.SetMetric(metric)
    optimizer.SetScalesEstimator(estimator)
    registration = itk.ImageRegistrationMethodv4[I3, I3].New()
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(transform)
    # A single level. The usual argument for a pyramid is that a local optimiser cannot walk out
    # of a centimetre-scale offset without first seeing the images blurred and shrunk, and the
    # comment here used to claim three levels while the code set one. Two and three levels were
    # tried: the recovered Dice came out 0.479 at one level, 0.446 at three and 0.349 at two --
    # non-monotonic, so a single run per setting says nothing, the sampler draws 10% of the
    # voxels at random and is not seeded. Left at one level, which is what was measured for the
    # figure; anyone tuning this should repeat each setting before believing a difference.
    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetMetricSamplingStrategy(2)  # 0 NONE, 1 REGULAR, 2 RANDOM
    registration.SetMetricSamplingPercentage(args.sampling)
    start = time.time()
    registration.Update()
    rigid_time = time.time() - start
    final = registration.GetTransform()
    rigid_moving = itk.array_from_image(itk.resample_image_filter(
        moving, transform=final, use_reference_image=True, reference_image=fixed,
        default_pixel_value=-1024.0))
    rigid_dice = dice(reference, itk.array_from_image(itk.resample_image_filter(
        moving_labels, transform=final, use_reference_image=True, reference_image=fixed_labels,
        interpolator=nearest)))



    # --- deformable, on top of the rigid ----------------------------------------------------
    # The deformable filters refine; they are not meant to walk out of a centimetre-scale rigid
    # offset on their own, and asking them to makes the field absorb a translation instead of
    # the anatomy. They are handed the rigidly resampled moving image, which is how the pair
    # would actually be registered, so the panel answers "what does the deformable step add on
    # top of the rigid one" rather than "can a dense field imitate a rigid transform".
    rigid_image = itk.resample_image_filter(
        moving, transform=final, use_reference_image=True, reference_image=fixed,
        default_pixel_value=-1024.0)
    rigid_labels = itk.resample_image_filter(
        moving_labels, transform=final, use_reference_image=True, reference_image=fixed_labels,
        interpolator=nearest)

    coarse = itk.ImpactCoarseRegistration[I3, I3].New()
    coarse.SetFixedImage(fixed)
    coarse.SetMovingImage(rigid_image)
    for c in configs(args):
        coarse.AddModelConfiguration(c)
    coarse.SetDevice(args.device)
    start = time.time()
    coarse.Update()

    fine = itk.ImpactFineRegistration[I3, I3].New()
    fine.SetFixedImage(fixed)
    fine.SetMovingImage(rigid_image)
    for c in configs(args):
        fine.AddModelConfiguration(c)
    fine.SetInitialDisplacementField(coarse.GetOutput())
    fine.SetDevice(args.device)
    fine.Update()
    deformable_time = time.time() - start
    field = fine.GetOutput()

    def warp(image, ImageT, interpolator=None):
        w = itk.WarpImageFilter[ImageT, ImageT, type(field)].New()
        w.SetInput(image)
        w.SetDisplacementField(field)
        w.SetOutputParametersFromImage(fixed)
        if interpolator is not None:
            w.SetInterpolator(interpolator)
        w.Update()
        return itk.array_from_image(w.GetOutput())

    deformable_moving = warp(rigid_image, I3)
    deformable_labels = warp(rigid_labels, UC3, nearest)
    deformable_dice = dice(reference, deformable_labels)

    # --- plates ----------------------------------------------------------------------------
    grey_fixed = normalise(itk.array_from_image(fixed))
    grey_before = normalise(itk.array_from_image(on_fixed))
    grey_rigid = normalise(rigid_moving)
    grey_deformable = normalise(deformable_moving)
    rigid_labels_array = itk.array_from_image(rigid_labels)
    z = int(np.argmax([(reference[k] > 0).sum() for k in range(reference.shape[0])]))

    def mean(scores):
        return float(np.mean(list(scores.values()))) if scores else float("nan")

    reference_contour = organ_contour(reference[z])
    before_labels_slice = before_labels[z]
    row(
        [
            (grey_fixed[z], None, "MR\\ (fixed)", "axial", None),
            (grey_before[z], None, "CT\\ (moving)", "resampled on the MR grid", None),
            (checkerboard(grey_fixed[z], grey_before[z]), organ_contour(before_labels_slice),
             "Before", f"misaligned \u00b7 Dice {mean(before):.3f}", reference_contour),
            (checkerboard(grey_fixed[z], grey_rigid[z]), organ_contour(rigid_labels_array[z]),
             "After\\ rigid", f"Dice {mean(rigid_dice):.3f} of {mean(aligned):.3f} \u00b7 {rigid_time:.0f} s",
             reference_contour),
        ],
        width=15.0,
        name="example-metricv4.png",
        outdir=args.outdir,
        suptitle="itk.ImpactImageToImageMetricv4 · MIND features, NCC distance · "
                 "teal: MR organs · orange: CT organs",
    )

    # A magnitude map only says how far things moved, which is mostly a restatement of the
    # misalignment. The Jacobian determinant of the transform says what the field DID to the
    # tissue: below 1 it compressed, above 1 it expanded, and at or below 0 it folded the space
    # onto itself, which is the one thing a deformation field must never do. Two thirds of this
    # volume is air, where nothing constrains the field, so it is read inside the body.
    jacobian = itk.DisplacementFieldJacobianDeterminantFilter[type(field), itk.F].New()
    jacobian.SetInput(field)
    jacobian.Update()
    determinant = itk.array_from_image(jacobian.GetOutput())
    # The body region is derived from the fixed image rather than taken from --fixed-mask: the
    # mask that ships with this case is a coarse region whose straight posterior edge cuts across
    # real anatomy, and reading the field through it would draw that edge rather than the patient.
    body = normalise(itk.array_from_image(fixed)) > 0.08
    inside = determinant[body]
    folded = float((inside <= 0).mean())
    displacement_detail = (f"{np.percentile(inside, 5):.2f}\u2013{np.percentile(inside, 95):.2f} "
                           f"(5\u201395th pct) \u00b7 {folded:.2%} folded")
    shown = np.where(body[z], determinant[z], np.nan)
    fig = plt.figure(figsize=(15.0, 4.6), facecolor=GROUND)
    ratios = [grey_fixed[z].shape[1] / grey_fixed[z].shape[0]] * 4
    grid = fig.add_gridspec(1, 4, width_ratios=ratios, wspace=0.025)
    reference_contour = organ_contour(reference[z])
    plate(fig.add_subplot(grid[0, 0]), grey_fixed[z], None, "MR\\ (fixed)", "axial")
    plate(fig.add_subplot(grid[0, 1]), checkerboard(grey_fixed[z], grey_rigid[z]),
          organ_contour(rigid_labels_array[z]), "After\\ rigid", f"Dice {mean(rigid_dice):.3f}",
          reference=reference_contour)
    plate(fig.add_subplot(grid[0, 2]), checkerboard(grey_fixed[z], grey_deformable[z]),
          organ_contour(deformable_labels[z]), "After\\ deformable",
          f"Dice {mean(deformable_dice):.3f} of {mean(aligned):.3f} · {deformable_time:.0f} s",
          reference=reference_contour)
    ax = fig.add_subplot(grid[0, 3])
    # Diverging around 1, so "unchanged volume" is the neutral colour and compression and
    # expansion read as opposite directions rather than as two shades of the same ramp.
    span = max(0.6, float(np.nanpercentile(np.abs(shown - 1.0), 98)))
    image = ax.imshow(shown, cmap="RdBu_r", vmin=1 - span, vmax=1 + span, interpolation="bilinear")
    ax.set_facecolor(GROUND)
    for spine in ax.spines.values():
        spine.set_edgecolor(FRAME)
        spine.set_linewidth(1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"$\\bf{{Jacobian}}$\n{displacement_detail}", color=LABEL, fontsize=9,
                  labelpad=8, linespacing=1.7)
    ax.xaxis.label.set_horizontalalignment("left")
    ax.xaxis.set_label_coords(0, -0.04)
    fig.text(0.008, 0.985,
             "itk.ImpactCoarseRegistration + itk.ImpactFineRegistration, on top of the rigid result · "
             "teal: MR organs · orange: CT organs",
             color=FRAME, fontsize=9.5, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.008, right=0.992, top=0.945, bottom=0.14)
    fig.savefig(os.path.join(args.outdir, "example-convexadam.png"), dpi=140, facecolor=GROUND)
    plt.close(fig)
    print("wrote example-convexadam.png")

    print(f"DICE aligned(ceiling) {mean(aligned):.4f} · misaligned {mean(before):.4f} · "
      f"rigid {mean(rigid_dice):.4f} · deformable {mean(deformable_dice):.4f}")
    for c in sorted(before):
        print(f"  label {c}: {before[c]:.3f} -> rigid {rigid_dice.get(c, float('nan')):.3f} "
              f"-> deformable {deformable_dice.get(c, float('nan')):.3f}")


if __name__ == "__main__":
    main()

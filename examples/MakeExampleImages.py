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

"""Regenerate the figures used by the examples README.

One forward pass of a TotalSegmentator-style model gives both pictures: a mid-level feature
layer, projected to RGB by PCA, and the final label logits. Needs matplotlib, which the module
itself does not depend on.

Run:  ./MakeExampleImages.py model.pt ct.nii.gz --outdir images
"""

import argparse
import os

import itk
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# The house style of the README figures: a dark plate, cyan rules and captions.
GROUND = "#151b22"
FRAME = "#4bbec7"
LABEL = "#e6edf2"
MUTED = "#93a6b4"

# The 24 organs of the TotalSegmentator "organs" model, in the order the logits come in, with a
# colour close to the convention radiology viewers use for each.
ORGANS = [
    ("spleen", "#8c3b3b"), ("kidney right", "#b5651d"), ("kidney left", "#d98c3f"),
    ("gallbladder", "#3f7d4f"), ("liver", "#a0522d"), ("stomach", "#c98f6b"),
    ("pancreas", "#d4a017"), ("adrenal right", "#7fb3d5"), ("adrenal left", "#5499c7"),
    ("lung upper left", "#7fd4c1"), ("lung lower left", "#48a999"), ("lung upper right", "#a3e4d7"),
    ("lung middle right", "#76d7c4"), ("lung lower right", "#45b39d"), ("oesophagus", "#e59866"),
    ("trachea", "#aeb6bf"), ("thyroid", "#f5b7b1"), ("small bowel", "#c39bd3"),
    ("duodenum", "#bb8fce"), ("colon", "#9b59b6"), ("bladder", "#f4d03f"),
    ("prostate", "#e6b0aa"), ("kidney cyst left", "#85929e"), ("kidney cyst right", "#5d6d7e"),
]


def parse():
    p = argparse.ArgumentParser(description="Regenerate the examples README figures.")
    p.add_argument("model", help="TorchScript model (.pt), e.g. TotalSegmentator's organs export")
    p.add_argument("image", help="input CT")
    p.add_argument("--outdir", default="images")
    p.add_argument("--feature-layer", type=int, default=2)
    p.add_argument("--seg-layer", type=int, default=7)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--voxel", type=float, default=1.5)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def extract(args):
    """One forward pass, two layers out."""
    image = itk.imread(args.image, itk.F)
    mask = [False] * args.num_layers
    mask[args.feature_layer] = True
    mask[args.seg_layer] = True

    config = itk.ImpactModelConfiguration(
        args.model, 3, 1, [args.patch] * 3, [args.voxel] * 3, args.overlap, mask, False
    )
    # nnU-Net's window, so the reassembly matches what TotalSegmentator does itself.
    config.SetPatchCombine("gaussian")

    ImageType = itk.Image[itk.F, 3]
    features = itk.ImageToFeaturesMap[
        ImageType, itk.BSplineInterpolateImageFunction[ImageType, itk.D, itk.F]
    ].New()
    features.SetModelConfiguration(config)
    features.SetDevice(args.device)
    features.AddInput(image)
    features.Update()

    # Kept layers come out in order, whatever their index in the model.
    feature_map = np.array(itk.array_view_from_image(features.GetOutput(0)), copy=True)
    logits = np.array(itk.array_view_from_image(features.GetOutput(1)), copy=True)
    grid = np.array(itk.array_view_from_image(features.GetOutput(1))).shape[:3]
    resampled = itk.array_from_image(
        itk.resample_image_filter(
            image,
            size=[int(grid[2]), int(grid[1]), int(grid[0])],
            output_spacing=list(features.GetOutput(1).GetSpacing()),
            output_origin=list(features.GetOutput(1).GetOrigin()),
            output_direction=features.GetOutput(1).GetDirection(),
            interpolator=itk.LinearInterpolateImageFunction.New(image),
        )
    )
    return resampled, feature_map, logits, features.GetOutput(0), features.GetOutput(1)


def pca_rgb(feature_map, chroma=0.34):
    """The feature layer projected onto its three principal components, as an RGB image.

    The leading component drives the luminance and the next two only tint it. Giving the three
    components an equal share of R, G and B instead produces a rainbow in which no anatomy is
    legible -- the point of the picture is that the features follow the anatomy."""
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    sample = flat[:: max(1, flat.shape[0] // 200000)]
    mean = sample.mean(0)
    _, _, basis = np.linalg.svd(sample - mean, full_matrices=False)
    components = ((flat - mean) @ basis[:3].T).reshape(*feature_map.shape[:3], 3)

    def stretch(values, lo, hi):
        low, high = np.percentile(values, (2, 98))
        return np.clip((values - low) / max(high - low, 1e-6), 0, 1) * (hi - lo) + lo

    luminance = stretch(components[..., 0], 0.12, 1.0)
    a = stretch(components[..., 1], -1.0, 1.0)
    b = stretch(components[..., 2], -1.0, 1.0)
    rgb = np.stack(
        [
            luminance * (1.0 + chroma * a),
            luminance * (1.0 - chroma * 0.5 * (a - b)),
            luminance * (1.0 - chroma * b),
        ],
        axis=-1,
    )
    return np.clip(rgb, 0, 1)


def window(ct, centre=40.0, width=400.0):
    """A soft-tissue window, so the CT reads like a CT."""
    return np.clip((ct - (centre - width / 2)) / width, 0, 1)


def colour_labels(labels):
    lut = np.zeros((len(ORGANS) + 1, 4))
    for i, (_, hexa) in enumerate(ORGANS, 1):
        lut[i] = matplotlib.colors.to_rgba(hexa)
    return lut[np.clip(labels, 0, len(ORGANS))]


def roi_fractions(labels, margin=0.16):
    """The box the segmented organs occupy, as fractions of the volume.

    Taken from the labels rather than from the intensities: a threshold on a CT also catches the
    examination table. Fractions, so the same anatomy can be cut out of a coarser feature layer."""
    box = []
    for axis in range(3):
        hit = np.where((labels > 0).any(axis=tuple(a for a in range(3) if a != axis)))[0]
        if hit.size == 0:
            box.append((0.0, 1.0))
            continue
        extent = labels.shape[axis]
        pad = margin * (hit[-1] - hit[0] + 1)
        box.append((max(0.0, (hit[0] - pad) / extent), min(1.0, (hit[-1] + 1 + pad) / extent)))
    return box


def cut(volume, box, plane_axis, plane_fraction):
    """One plane of `volume`, cropped to `box`; both are given as fractions of the volume."""
    limits = [(int(round(lo * n)), max(int(round(hi * n)), int(round(lo * n)) + 1))
              for (lo, hi), n in zip(box, volume.shape[:3])]
    index = min(int(round(plane_fraction * volume.shape[plane_axis])), volume.shape[plane_axis] - 1)
    limits[plane_axis] = (index, index + 1)
    sliced = volume[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
    return np.squeeze(sliced, axis=plane_axis)[::-1]


def plate(ax, background, overlay=None, alpha=0.62, rgb=False):
    if rgb:
        # Bilinear, so a coarse feature layer reads as a smooth field and not as confetti.
        ax.imshow(background, interpolation="bilinear")
    else:
        ax.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    if overlay is not None:
        ax.imshow(overlay, interpolation="nearest", alpha=alpha * (overlay[..., 3] > 0))
    ax.set_facecolor(GROUND)
    for spine in ax.spines.values():
        spine.set_edgecolor(FRAME)
        spine.set_linewidth(1.1)
    ax.set_xticks([])
    ax.set_yticks([])


def caption(ax, title, detail):
    ax.set_xlabel(f"$\\bf{{{title}}}$\n{detail}", color=LABEL, fontsize=9, labelpad=8, linespacing=1.7)
    ax.xaxis.label.set_horizontalalignment("left")
    ax.xaxis.set_label_coords(0, -0.04)


def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    ct, feature_map, logits, feature_image, seg_image = extract(args)
    labels = np.argmax(logits, axis=-1).astype(np.uint8)
    grey = window(ct)
    rgb = pca_rgb(feature_map)
    box = roi_fractions(labels)

    # The plane of each orientation carrying the most structures reads best.
    def busiest(axis):
        counts = [len(np.unique(np.take(labels, i, axis=axis))) for i in range(labels.shape[axis])]
        return int(np.argmax(counts)) / labels.shape[axis]

    fz, fy, fx = busiest(0), busiest(1), busiest(2)

    def row(views, width, legend=None, name="plate"):
        """One row of panels, sized so that every panel ends up the same height."""
        ratios = [v[0].shape[1] / v[0].shape[0] for v in views] + ([0.55] if legend else [])
        fig = plt.figure(figsize=(width, width / sum(ratios) * 1.34), facecolor=GROUND)
        grid = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.025)
        axes = []
        for i, (background, overlay, title, detail, is_rgb) in enumerate(views):
            ax = fig.add_subplot(grid[0, i])
            plate(ax, background, overlay, rgb=is_rgb)
            caption(ax, title, detail)
            axes.append(ax)
        return fig, grid

    # ---------------------------------------------------------------- inference plate
    coronal_ct = cut(grey, box, 1, fy)
    coronal_labels = cut(labels, box, 1, fy)
    fig, _ = row(
        [
            (coronal_ct, None, "CT", f"input · resampled to {args.voxel} mm", False),
            (cut(rgb, box, 1, fy), None, "Feature\\ map",
             f"layer {args.feature_layer} · {feature_map.shape[-1]} ch @ "
             f"{feature_image.GetSpacing()[0]:.0f} mm · PCA→RGB", True),
            (coronal_ct, colour_labels(coronal_labels), "Segmentation",
             f"layer {args.seg_layer} · {len(np.unique(labels)) - 1} structures", False),
        ],
        width=13.5,
    )
    fig.subplots_adjust(left=0.008, right=0.992, top=0.985, bottom=0.115)
    fig.savefig(os.path.join(args.outdir, "example-inference.png"), dpi=140, facecolor=GROUND)
    plt.close(fig)
    print("wrote example-inference.png")

    # ------------------------------------------------------------- segmentation plate
    views = []
    for axis, fraction, name in [(0, fz, "Axial"), (1, fy, "Coronal"), (2, fx, "Sagittal")]:
        background = cut(grey, box, axis, fraction)
        views.append(
            (background, colour_labels(cut(labels, box, axis, fraction)), name,
             f"{int(round(fraction * labels.shape[axis]))} of {labels.shape[axis]}", False)
        )
    fig, grid = row(views, width=14.0, legend=True)
    legend_ax = fig.add_subplot(grid[0, 3])
    legend_ax.axis("off")
    present = [c for c in np.unique(labels) if c > 0]
    handles = [
        mpatches.Patch(facecolor=ORGANS[c - 1][1], edgecolor="none", label=ORGANS[c - 1][0])
        for c in present
    ]
    legend = legend_ax.legend(
        handles=handles, loc="center left", frameon=False, ncol=1, fontsize=8.5,
        labelcolor=LABEL, handlelength=1.1, handleheight=1.1, borderpad=0, labelspacing=0.62,
    )
    legend.set_title(f"{len(present)} structures", prop={"size": 9, "weight": "bold"})
    legend.get_title().set_color(FRAME)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1)
    fig.savefig(os.path.join(args.outdir, "example-segmentation.png"), dpi=140, facecolor=GROUND)
    plt.close(fig)
    print("wrote example-segmentation.png")


if __name__ == "__main__":
    main()

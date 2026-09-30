#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio


EVAL_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "evaluation/sardinia_surtr_v1"
)

DATA_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/patches_v1"
)

CASES_JSON = EVAL_ROOT / "qualitative_cases.json"
PRED_ROOT = EVAL_ROOT / "predictions"

OUT_PNG = EVAL_ROOT / "surtr_v1_qualitative_test.png"
OUT_PDF = EVAL_ROOT / "surtr_v1_qualitative_test.pdf"


def percentile_rgb(arr):
    """
    Input: [bands,H,W]
    S2 order:
    BLUE, GREEN, RED, NIR, SWIR1, SWIR2
    """

    rgb = arr[[2, 1, 0]].astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)

    out = np.zeros_like(rgb)

    for c in range(3):
        x = rgb[..., c]

        good = np.isfinite(x)

        if not good.any():
            continue

        lo, hi = np.percentile(
            x[good],
            [2, 98],
        )

        if hi <= lo:
            hi = lo + 1e-6

        out[..., c] = np.clip(
            (x - lo) / (hi - lo),
            0,
            1,
        )

    return out


def read_image(path):
    with rasterio.open(path) as src:
        return src.read()


def find_patch_file(split, folder, patch_id):
    candidates = list(
        (DATA_ROOT / split / folder).glob(
            f"{patch_id}.*"
        )
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one file for "
            f"{patch_id} in {folder}, "
            f"found {len(candidates)}"
        )

    return candidates[0]


def decode(x):
    """
    Stored encoding:
      0 invalid
      1 valid unburned
      2 valid burned

    Return:
      nan invalid
      0 unburned
      1 burned
    """
    out = np.full(
        x.shape,
        np.nan,
        dtype=np.float32,
    )

    out[x == 1] = 0
    out[x == 2] = 1

    return out


def error_map(pred, gt):
    """
    0 correct
    1 false positive
    2 false negative
    nan invalid
    """

    out = np.full(
        gt.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(gt)
        & np.isfinite(pred)
    )

    out[valid] = 0

    out[
        valid
        & (pred == 1)
        & (gt == 0)
    ] = 1

    out[
        valid
        & (pred == 0)
        & (gt == 1)
    ] = 2

    return out


def main():

    cases = json.loads(
        CASES_JSON.read_text()
    )

    ordered = [
        (
            "Best improvement",
            cases["best_improvement"],
        ),
        (
            "Similar performance",
            cases["most_similar"],
        ),
        (
            "Worst degradation",
            cases["worst_degradation"],
        ),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=7,
        figsize=(19, 8.5),
    )

    column_titles = [
        "Pre-fire",
        "Post-fire",
        "Ground truth",
        "POST-only",
        "SURTR",
        "POST-only errors",
        "SURTR errors",
    ]

    for j, title in enumerate(column_titles):
        axes[0, j].set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

    for row, (label, case) in enumerate(ordered):

        patch_id = case["patch_id"]

        pre_path = find_patch_file(
            "test",
            "pre",
            patch_id,
        )

        post_path = find_patch_file(
            "test",
            "post",
            patch_id,
        )

        pred_path = (
            PRED_ROOT
            / f"{patch_id}.npz"
        )

        if not pred_path.exists():
            raise FileNotFoundError(
                pred_path
            )

        pre = read_image(pre_path)
        post = read_image(post_path)

        pack = np.load(
            pred_path,
            allow_pickle=True,
        )

        gt = decode(pack["target"])
        baseline = decode(pack["baseline"])
        surtr = decode(pack["surtr"])

        err_base = error_map(
            baseline,
            gt,
        )

        err_surtr = error_map(
            surtr,
            gt,
        )

        images = [
            percentile_rgb(pre),
            percentile_rgb(post),
            gt,
            baseline,
            surtr,
            err_base,
            err_surtr,
        ]

        for col, image in enumerate(images):

            ax = axes[row, col]

            if col < 2:
                ax.imshow(image)

            elif col <= 4:
                ax.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )

            else:
                ax.imshow(
                    image,
                    cmap="viridis",
                    vmin=0,
                    vmax=2,
                )

            ax.set_xticks([])
            ax.set_yticks([])

        delta = case["delta_iou"]

        axes[row, 0].set_ylabel(
            f"{label}\n"
            f"ΔIoU = {delta:+.3f}",
            fontsize=11,
            fontweight="bold",
        )

        # Add compact metrics under predictions.
        axes[row, 3].set_xlabel(
            f"IoU = "
            f"{case['baseline_iou']:.3f}",
            fontsize=10,
        )

        axes[row, 4].set_xlabel(
            f"IoU = "
            f"{case['surtr_iou']:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        "Sardinia independent test — "
        "POST-only baseline vs. SURTR bi-temporal model",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.01,
        "Error maps: 0 = correct, "
        "1 = false positive, "
        "2 = false negative; "
        "invalid pixels are transparent.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[0, 0.035, 1, 0.965]
    )

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 72)
    print("QUALITATIVE FIGURE COMPLETE")
    print("=" * 72)

    for label, case in ordered:
        print(
            f"{label:20s} | "
            f"{case['patch_id']} | "
            f"A0={case['baseline_iou']:.4f} | "
            f"A1={case['surtr_iou']:.4f} | "
            f"Δ={case['delta_iou']:+.4f}"
        )

    print()
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()

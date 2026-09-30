#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch


def read_raster(path):
    with rasterio.open(path) as ds:
        arr = ds.read()
        bounds = ds.bounds
        transform = ds.transform
        crs = ds.crs
    return arr, bounds, transform, crs


def percentile_stretch(rgb, p_low=2, p_high=98):
    """
    Stretch RGB image to [0,1] using percentiles.
    Expects array shape (H, W, 3).
    """
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[:, :, i].astype(np.float32)
        lo = np.nanpercentile(band, p_low)
        hi = np.nanpercentile(band, p_high)

        if hi <= lo:
            out[:, :, i] = np.clip(band, 0, 1)
        else:
            out[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)

    return out


def make_rgb_from_s2(arr):
    """
    Input arr shape: (6, H, W)
    band order:
      0 BLUE
      1 GREEN
      2 RED
      3 NIR_NARROW
      4 SWIR_1
      5 SWIR_2
    Returns RGB as (H, W, 3)
    """
    rgb = np.stack(
        [arr[2], arr[1], arr[0]],
        axis=-1,
    )
    return percentile_stretch(rgb)


def extent_from_bounds(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def add_scalebar(ax, bounds, length_km=2.0, text_y_pad=0.015):
    """
    Simple scale bar in map coordinates.
    Assumes projected CRS in meters.
    """
    xmin, xmax = bounds.left, bounds.right
    ymin, ymax = bounds.bottom, bounds.top

    width = xmax - xmin
    height = ymax - ymin

    bar_len_m = length_km * 1000.0
    x0 = xmin + 0.05 * width
    y0 = ymin + 0.06 * height

    ax.plot([x0, x0 + bar_len_m], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.01 * height, y0 + 0.01 * height], color="white", linewidth=2)
    ax.plot([x0 + bar_len_m, x0 + bar_len_m], [y0 - 0.01 * height, y0 + 0.01 * height], color="white", linewidth=2)

    ax.text(
        x0 + bar_len_m / 2,
        y0 + text_y_pad * height,
        f"{int(length_km)} km",
        color="white",
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
    )


def add_north_arrow(ax, bounds):
    xmin, xmax = bounds.left, bounds.right
    ymin, ymax = bounds.bottom, bounds.top

    width = xmax - xmin
    height = ymax - ymin

    x = xmax - 0.06 * width
    y = ymax - 0.12 * height

    ax.annotate(
        "N",
        xy=(x, y + 0.07 * height),
        xytext=(x, y),
        color="white",
        fontsize=12,
        fontweight="bold",
        ha="center",
        arrowprops=dict(
            facecolor="white",
            edgecolor="white",
            width=2,
            headwidth=10,
            headlength=10,
        ),
        bbox=dict(facecolor="black", alpha=0.25, pad=1, edgecolor="none"),
    )


def show_map(ax, img, bounds, title):
    ax.imshow(
        img,
        extent=extent_from_bounds(bounds),
        origin="upper",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar(ax, bounds, length_km=2)
    add_north_arrow(ax, bounds)


def overlay_mask_fill(ax, base_img, mask, bounds, color=(1.0, 0.2, 0.2), alpha=0.45, title=""):
    ax.imshow(
        base_img,
        extent=extent_from_bounds(bounds),
        origin="upper",
    )

    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[mask == 1, 0] = color[0]
    overlay[mask == 1, 1] = color[1]
    overlay[mask == 1, 2] = color[2]
    overlay[mask == 1, 3] = alpha

    ax.imshow(
        overlay,
        extent=extent_from_bounds(bounds),
        origin="upper",
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar(ax, bounds, length_km=2)
    add_north_arrow(ax, bounds)


def overlay_contours(ax, base_img, ref_mask, pred_mask, bounds, title=""):
    ax.imshow(
        base_img,
        extent=extent_from_bounds(bounds),
        origin="upper",
    )

    xs = np.linspace(bounds.left, bounds.right, ref_mask.shape[1])
    ys = np.linspace(bounds.top, bounds.bottom, ref_mask.shape[0])

    # Reference outline
    ax.contour(
        xs,
        ys,
        (ref_mask == 1).astype(np.uint8),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.8,
    )

    # Prediction outline
    ax.contour(
        xs,
        ys,
        (pred_mask == 1).astype(np.uint8),
        levels=[0.5],
        colors=["magenta"],
        linewidths=1.8,
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scalebar(ax, bounds, length_km=2)
    add_north_arrow(ax, bounds)

    legend_handles = [
        Patch(facecolor="none", edgecolor="cyan", label="Reference perimeter"),
        Patch(facecolor="none", edgecolor="magenta", label="SURTR perimeter"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.85)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--event-dir", required=True, help="Directory containing pre.tif, post.tif, mask.tif, metadata.json")
    parser.add_argument("--demo-dir", required=True, help="Directory containing surtr outputs and report")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for PNG/PDF figure")

    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    demo_dir = Path(args.demo_dir)
    out_prefix = Path(args.out_prefix)

    pre_path = event_dir / "pre.tif"
    post_path = event_dir / "post.tif"
    ref_path = event_dir / "mask.tif"
    metadata_path = event_dir / "metadata.json"

    prob_path = demo_dir / "surtr_burn_probability.tif"
    pred_path = demo_dir / "surtr_burn_mask.tif"
    report_path = demo_dir / "surtr_area_report.json"

    pre_arr, bounds, _, _ = read_raster(pre_path)
    post_arr, _, _, _ = read_raster(post_path)
    ref_arr, _, _, _ = read_raster(ref_path)
    prob_arr, _, _, _ = read_raster(prob_path)
    pred_arr, _, _, _ = read_raster(pred_path)

    metadata = json.loads(metadata_path.read_text())
    report = json.loads(report_path.read_text())

    pre_rgb = make_rgb_from_s2(pre_arr)
    post_rgb = make_rgb_from_s2(post_arr)

    ref_mask = ref_arr[0].astype(np.uint8)
    pred_mask = pred_arr[0].astype(np.uint8)

    prob = prob_arr[0].astype(np.float32)
    prob_masked = np.ma.masked_where(prob < -9990, prob)

    municipality = metadata.get("municipality", "Unknown")
    fire_date = metadata.get("fire_date", "Unknown")
    pre_date = metadata.get("pre_date", "Unknown")
    post_date = metadata.get("post_date", "Unknown")
    event_id = metadata.get("event_id", "Unknown")
    official_area = metadata.get("official_geometry_area_ha", np.nan)
    raster_area = metadata.get("burned_area_raster_ha", np.nan)

    refcomp = report.get("reference_comparison", {})
    pred_area = refcomp.get("predicted_burned_area_ha", np.nan)
    ref_area = refcomp.get("reference_burned_area_ha", np.nan)
    area_diff = refcomp.get("difference_ha", np.nan)
    area_err = refcomp.get("absolute_relative_error_percent", np.nan)
    iou = refcomp.get("iou", np.nan)
    f1 = refcomp.get("f1", np.nan)
    precision = refcomp.get("precision", np.nan)
    recall = refcomp.get("recall", np.nan)
    eval_area = refcomp.get("evaluation_area_ha", np.nan)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.68], hspace=0.18, wspace=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax_txt = fig.add_subplot(gs[2, :])

    show_map(ax1, pre_rgb, bounds, f"PRE Sentinel-2 RGB\n{pre_date}")
    show_map(ax2, post_rgb, bounds, f"POST Sentinel-2 RGB\n{post_date}")

    im = ax3.imshow(
        prob_masked,
        cmap="inferno",
        extent=extent_from_bounds(bounds),
        origin="upper",
        vmin=0,
        vmax=1,
    )
    ax3.set_title("SURTR burn probability", fontsize=11, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])
    add_scalebar(ax3, bounds, length_km=2)
    add_north_arrow(ax3, bounds)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
    cbar.set_label("Burn probability", fontsize=9)

    overlay_mask_fill(
        ax4,
        post_rgb,
        (ref_mask == 1).astype(np.uint8),
        bounds,
        color=(0.10, 0.90, 1.00),
        alpha=0.42,
        title="Official raster reference on POST",
    )

    overlay_mask_fill(
        ax5,
        post_rgb,
        (pred_mask == 1).astype(np.uint8),
        bounds,
        color=(1.00, 0.15, 0.55),
        alpha=0.42,
        title="SURTR prediction on POST",
    )

    overlay_contours(
        ax6,
        post_rgb,
        ref_mask,
        pred_mask,
        bounds,
        title="Reference vs SURTR contours",
    )

    ax_txt.axis("off")

    summary = (
        f"Event: {event_id}\n"
        f"Municipality: {municipality}\n"
        f"Fire date: {fire_date}\n"
        f"Official CFVA geometry area: {official_area:.2f} ha\n"
        f"Raster reference burned area: {raster_area:.2f} ha\n"
        f"SURTR predicted burned area: {pred_area:.2f} ha\n"
        f"Difference vs reference: {area_diff:+.2f} ha\n"
        f"Absolute relative area error: {area_err:.2f}%\n"
        f"Evaluation area: {eval_area:.2f} ha"
    )

    metrics = (
        f"IoU       : {iou:.4f}\n"
        f"F1        : {f1:.4f}\n"
        f"Precision : {precision:.4f}\n"
        f"Recall    : {recall:.4f}"
    )

    interpretation = (
        "Interpretation:\n"
        "SURTR reproduces the official burned scar with high spatial agreement.\n"
        "The model is conservative: very high precision and slight area underestimation."
    )

    ax_txt.text(
        0.01, 0.92,
        "SURTR large-area inference on a real Sardinian wildfire",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )

    ax_txt.text(
        0.01, 0.72,
        summary,
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(facecolor="#f4f4f4", edgecolor="#cccccc", boxstyle="round,pad=0.5"),
    )

    ax_txt.text(
        0.43, 0.72,
        metrics,
        fontsize=12,
        family="monospace",
        ha="left",
        va="top",
        bbox=dict(facecolor="#eef6ff", edgecolor="#88aadd", boxstyle="round,pad=0.5"),
    )

    ax_txt.text(
        0.67, 0.72,
        interpretation,
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(facecolor="#eef9ef", edgecolor="#88bb88", boxstyle="round,pad=0.5"),
    )

    fig.suptitle(
        f"{event_id} — {municipality}, Sardinia",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("MICHELE FIGURE COMPLETE")
    print("=" * 72)
    print("PNG:", png_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()


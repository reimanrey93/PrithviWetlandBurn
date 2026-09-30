#!/usr/bin/env python3
"""
inspect_clean_labels.py

Genera previews claros de máscaras limpias:
  negro = no quemado (0)
  rojo   = quemado candidato (1)
  gris   = ignorar / incierto / agua / no-data (255)

Lee:
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/pre
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/post
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/aux
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/labels_clean

Escribe:
  /DATA/ralfonso/prithvi_wetlandburn/outputs/preview_clean_v2
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path("/DATA/ralfonso/prithvi_wetlandburn")
EVENT_IDS = ["SF_001", "SF_002", "SF_003", "SF_004", "SF_005"]

PRE_DIR = ROOT / "data/raw/pre"
POST_DIR = ROOT / "data/raw/post"
AUX_DIR = ROOT / "data/raw/aux"
LABEL_DIR = ROOT / "data/raw/labels_clean"
OUT_DIR = ROOT / "outputs/preview_clean_v2"

# Band order in exported pre/post: B2, B3, B4, B8, B11, B12.
RGB_BANDS = (2, 1, 0)  # B4, B3, B2, zero-based raster indexes.


def read_band(path, band_index=1):
    with rasterio.open(path) as src:
        return src.read(band_index)


def read_stack(path):
    with rasterio.open(path) as src:
        return src.read()


def robust_rgb(stack):
    rgb = np.moveaxis(stack[list(RGB_BANDS)], 0, -1).astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for channel in range(3):
        values = rgb[..., channel]
        valid = np.isfinite(values) & (values > 0)
        if not np.any(valid):
            continue
        lo, hi = np.percentile(values[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[..., channel] = np.clip((values - lo) / (hi - lo), 0, 1)

    return out


def find_aux_band(path, target_name):
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        for index, description in enumerate(descriptions, start=1):
            if description and description.lower() == target_name.lower():
                return src.read(index)
        # Fallback: expected rNBR fourth band in this pipeline.
        if target_name.lower() == "rnbr":
            return src.read(4)
        raise RuntimeError(f"No se encontró la banda {target_name} en {path}")


def label_for_plot(label):
    # Remap: 0 -> 0 (black); 1 -> 1 (red); 255 -> 2 (gray).
    out = np.full(label.shape, 2, dtype=np.uint8)
    out[label == 0] = 0
    out[label == 1] = 1
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_cmap = ListedColormap(["black", "red", "lightgray"])
    label_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], label_cmap.N)

    for event_id in EVENT_IDS:
        pre_path = PRE_DIR / f"{event_id}_pre.tif"
        post_path = POST_DIR / f"{event_id}_post.tif"
        aux_path = AUX_DIR / f"{event_id}_aux.tif"
        label_path = LABEL_DIR / f"{event_id}_label_clean.tif"

        paths = [pre_path, post_path, aux_path, label_path]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            print(f"[SKIP] {event_id}: faltan archivos")
            for path in missing:
                print("  ", path)
            continue

        pre = read_stack(pre_path)
        post = read_stack(post_path)
        rnbr = find_aux_band(aux_path, "rNBR")
        label = read_band(label_path)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        fig.suptitle(event_id, fontsize=16)

        axes[0].imshow(robust_rgb(pre))
        axes[0].set_title("RGB pre-evento")

        axes[1].imshow(robust_rgb(post))
        axes[1].set_title("RGB post-evento")

        im = axes[2].imshow(rnbr, cmap="RdYlGn_r", vmin=-0.25, vmax=0.75)
        axes[2].set_title("rNBR")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(label_for_plot(label), cmap=label_cmap, norm=label_norm)
        axes[3].set_title("Máscara limpia\nrojo = quemado | negro = no quemado | gris = ignorar")

        for ax in axes:
            ax.axis("off")

        burned = 100.0 * np.mean(label == 1)
        nonburn = 100.0 * np.mean(label == 0)
        ignore = 100.0 * np.mean(label == 255)
        fig.text(
            0.5,
            0.01,
            f"Quemado: {burned:.2f}%   |   No quemado: {nonburn:.2f}%   |   Ignorar: {ignore:.2f}%",
            ha="center",
            fontsize=10,
        )

        out_path = OUT_DIR / f"{event_id}_clean_preview.png"
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

        print(
            f"{event_id} | quemado={burned:.2f}% | "
            f"no_quemado={nonburn:.2f}% | ignorar={ignore:.2f}% | {out_path}"
        )

    print(f"\nPreviews guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

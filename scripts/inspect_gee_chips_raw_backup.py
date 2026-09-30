#!/usr/bin/env python3
"""
Visualiza chips descargados desde Earth Engine para revisión manual.
Genera, por evento:
  - RGB pre-evento
  - RGB post-evento
  - rNBR
  - máscara semilla
y exporta PNGs a /DATA/ralfonso/prithvi_wetlandburn/outputs/preview_raw/
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def robust_scale(arr, lo=2, hi=98):
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    a, b = np.percentile(arr[valid], [lo, hi])
    if b <= a:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - a) / (b - a)
    return np.clip(out, 0, 1)


def read_rgb(path):
    with rasterio.open(path) as ds:
        if ds.count < 3:
            raise ValueError(f"{path} tiene menos de 3 bandas.")
        # Se asume orden B2, B3, B4, B8, B11, B12.
        blue = ds.read(1).astype(np.float32)
        green = ds.read(2).astype(np.float32)
        red = ds.read(3).astype(np.float32)
        rgb = np.dstack([
            robust_scale(red),
            robust_scale(green),
            robust_scale(blue),
        ])
        meta = {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "crs": str(ds.crs),
            "transform": str(ds.transform),
            "descriptions": ds.descriptions,
        }
    return rgb, meta


def read_aux_rnbr(path):
    with rasterio.open(path) as ds:
        desc = [d.lower() if d else "" for d in ds.descriptions]
        rnbr_idx = None
        for i, d in enumerate(desc):
            if "rnbr" in d:
                rnbr_idx = i + 1
                break
        # Fallback: el downloader deja rNBR como cuarta banda auxiliar.
        if rnbr_idx is None:
            rnbr_idx = 4 if ds.count >= 4 else 1
        arr = ds.read(rnbr_idx).astype(np.float32)
        return arr, rnbr_idx, ds.descriptions


def read_label(path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32)


def show_label(ax, label):
    # 0 = no quemado candidato; 1 = quemado candidato; 255 = incierto/nodata.
    view = np.full(label.shape, np.nan, dtype=np.float32)
    view[label == 0] = 0
    view[label == 1] = 1
    ax.imshow(view, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Máscara semilla\nblanco = quemado", fontsize=10)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/DATA/ralfonso/prithvi_wetlandburn",
        help="Raíz de datos del proyecto.",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=["SF_001", "SF_002", "SF_003", "SF_004", "SF_005"],
        help="IDs de eventos a visualizar.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    pre_dir = root / "data/raw/pre"
    post_dir = root / "data/raw/post"
    aux_dir = root / "data/raw/aux"
    label_dir = root / "data/raw/labels_clean"
    out_dir = root / "outputs/preview_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for event_id in args.events:
        pre_path = pre_dir / f"{event_id}_pre.tif"
        post_path = post_dir / f"{event_id}_post.tif"
        aux_path = aux_dir / f"{event_id}_aux.tif"
        label_path = label_dir / f"{event_id}_label_clean.tif"

        missing = [str(p) for p in (pre_path, post_path, aux_path, label_path) if not p.exists()]
        if missing:
            print(f"[SKIP] {event_id}. Faltan archivos:")
            for p in missing:
                print("  ", p)
            continue

        rgb_pre, pre_meta = read_rgb(pre_path)
        rgb_post, post_meta = read_rgb(post_path)
        rnbr, rnbr_band, aux_desc = read_aux_rnbr(aux_path)
        label = read_label(label_path)

        print(f"\n{event_id}")
        print(f"  PRE:  {pre_meta['width']} x {pre_meta['height']} | bandas={pre_meta['count']} | CRS={pre_meta['crs']}")
        print(f"  POST: {post_meta['width']} x {post_meta['height']} | bandas={post_meta['count']} | CRS={post_meta['crs']}")
        print(f"  rNBR: banda auxiliar {rnbr_band} | min={np.nanmin(rnbr):.3f} max={np.nanmax(rnbr):.3f}")
        print(f"  Label: quemado={np.mean(label == 1)*100:.2f}% | incierto={np.mean(label == 255)*100:.2f}%")

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(rgb_pre)
        axes[0].set_title("RGB pre-evento", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(rgb_post)
        axes[1].set_title("RGB post-evento", fontsize=10)
        axes[1].axis("off")

        im = axes[2].imshow(rnbr, cmap="RdYlGn_r", vmin=-0.25, vmax=0.75)
        axes[2].set_title("rNBR", fontsize=10)
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        show_label(axes[3], label)

        fig.suptitle(event_id, fontsize=13)
        fig.tight_layout()
        out_png = out_dir / f"{event_id}_preview.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("  PNG:", out_png)

    print(f"\nPreviews guardados en: {out_dir}")


if __name__ == "__main__":
    main()

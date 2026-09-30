#!/usr/bin/env python3
"""
compute_train_stats.py

Compute per-band mean/std only over the TerraTorch TRAIN split.
Uses six-band float32 TIFFs in [0, 1] and ignores zeros as no-data only when
all bands are zero at a pixel. This preserves legitimate low reflectance values.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/DATA/ralfonso/prithvi_wetlandburn/data/terratorch_burnscar_512",
    )
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "images"
    split_path = root / "splits" / "train.txt"

    event_ids = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
    count = np.zeros(6, dtype=np.float64)
    mean = np.zeros(6, dtype=np.float64)
    m2 = np.zeros(6, dtype=np.float64)

    for event_id in event_ids:
        path = image_dir / f"{event_id}.tif"
        with rasterio.open(path) as src:
            x = src.read().astype(np.float64)  # C,H,W

        # Keep pixels where at least one spectral band carries data.
        valid = np.isfinite(x).all(axis=0) & (np.abs(x).sum(axis=0) > 0)

        for band in range(6):
            values = x[band][valid]
            if values.size == 0:
                continue
            n_b = values.size
            mean_b = values.mean()
            m2_b = ((values - mean_b) ** 2).sum()

            delta = mean_b - mean[band]
            n_total = count[band] + n_b
            mean[band] += delta * n_b / n_total
            m2[band] += m2_b + (delta**2) * count[band] * n_b / n_total
            count[band] = n_total

        print(f"[OK] {event_id}")

    std = np.sqrt(m2 / np.maximum(count - 1, 1))

    names = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    print("\n=== TRAIN STATISTICS ===")
    print("bands:", names)
    print("means:")
    for value in mean:
        print(f"  - {value:.12f}")
    print("stds:")
    for value in std:
        print(f"  - {value:.12f}")
    print("pixels_per_band:", int(count[0]))


if __name__ == "__main__":
    main()

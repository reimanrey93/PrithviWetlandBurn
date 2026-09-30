#!/usr/bin/env python3
"""
build_terratorch_burnscar_dataset.py

Creates a TerraTorch-ready 512x512 GeoTIFF dataset from the downloaded
Sentinel-2 post-event chips and cleaned pseudo-labels.

Input:
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/post/SF_xxx_post.tif
  /DATA/ralfonso/prithvi_wetlandburn/data/raw/labels_clean/SF_xxx_label_clean.tif
  /DATA/ralfonso/prithvi_wetlandburn/data/prithvi_512/manifests/split_manifest.csv

Output:
  /DATA/ralfonso/prithvi_wetlandburn/data/terratorch_burnscar_512/
    images/SF_xxx.tif
    labels/SF_xxx.tif
    splits/train.txt, val.txt, test.txt
    manifests/build_summary.json

Image bands: B2,B3,B4,B8,B11,B12 (six bands)
Label values: 0=no burn, 1=burn candidate, -1=ignore/nodata
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

TARGET_SIZE = 512
IGNORE_SOURCE = 255
IGNORE_TARGET = -1


def center_crop_window(height: int, width: int, size: int):
    if height < size or width < size:
        raise ValueError(f"Chip too small: {height}x{width}; expected >= {size}.")
    row_off = (height - size) // 2
    col_off = (width - size) // 2
    return row_off, col_off


def write_image(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        if src.count != 6:
            raise ValueError(f"{src_path.name}: expected 6 bands, got {src.count}.")
        row_off, col_off = center_crop_window(src.height, src.width, TARGET_SIZE)
        window = rasterio.windows.Window(col_off, row_off, TARGET_SIZE, TARGET_SIZE)
        image = src.read(window=window).astype(np.float32)
        transform = src.window_transform(window)
        profile = src.profile.copy()

    image[~np.isfinite(image)] = 0.0
    image = np.clip(image, 0.0, 1.0)

    profile.update(
        height=TARGET_SIZE,
        width=TARGET_SIZE,
        count=6,
        dtype="float32",
        transform=transform,
        nodata=0.0,
        compress="deflate",
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(image)
        descriptions = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
        for band, name in enumerate(descriptions, start=1):
            dst.set_band_description(band, name)


def write_label(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        row_off, col_off = center_crop_window(src.height, src.width, TARGET_SIZE)
        window = rasterio.windows.Window(col_off, row_off, TARGET_SIZE, TARGET_SIZE)
        label = src.read(1, window=window).astype(np.int16)
        transform = src.window_transform(window)
        profile = src.profile.copy()

    unexpected = np.setdiff1d(np.unique(label), [0, 1, IGNORE_SOURCE])
    if unexpected.size:
        raise ValueError(f"{src_path.name}: unexpected label values: {unexpected.tolist()}")

    label[label == IGNORE_SOURCE] = IGNORE_TARGET
    profile.update(
        height=TARGET_SIZE,
        width=TARGET_SIZE,
        count=1,
        dtype="int16",
        transform=transform,
        nodata=IGNORE_TARGET,
        compress="deflate",
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(label, 1)
        dst.set_band_description(1, "burn_scar_label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/DATA/ralfonso/prithvi_wetlandburn")
    parser.add_argument(
        "--split-manifest",
        default="data/prithvi_512/manifests/split_manifest.csv",
    )
    parser.add_argument("--output", default="data/terratorch_burnscar_512")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    manifest_path = data_root / args.split_manifest
    out_root = data_root / args.output
    image_out = out_root / "images"
    label_out = out_root / "labels"
    split_out = out_root / "splits"
    manifest_out = out_root / "manifests"

    with open(manifest_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    split_ids = {"train": [], "val": [], "test": []}
    stats = {"total": 0, "skipped": 0, "by_split": {}}

    for row in rows:
        event_id = row["event_id"]
        split = row["split"]
        if split not in split_ids:
            raise ValueError(f"Unexpected split: {split}")

        src_image = data_root / "data/raw/post" / f"{event_id}_post.tif"
        src_label = data_root / "data/raw/labels_clean" / f"{event_id}_label_clean.tif"
        dst_image = image_out / f"{event_id}.tif"
        dst_label = label_out / f"{event_id}.tif"

        if not src_image.exists() or not src_label.exists():
            print(f"[SKIP] {event_id}: source missing")
            stats["skipped"] += 1
            continue

        try:
            if args.overwrite or not dst_image.exists():
                write_image(src_image, dst_image)
            if args.overwrite or not dst_label.exists():
                write_label(src_label, dst_label)
        except Exception as error:
            print(f"[ERROR] {event_id}: {error}")
            stats["skipped"] += 1
            continue

        split_ids[split].append(event_id)
        stats["total"] += 1
        stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
        print(f"[OK] {event_id} -> {split}")

    split_out.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        (split_out / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    manifest_out.mkdir(parents=True, exist_ok=True)
    summary_path = manifest_out / "build_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(stats, indent=2))
    print(f"Dataset: {out_root}")


if __name__ == "__main__":
    main()

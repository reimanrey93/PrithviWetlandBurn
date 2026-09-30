#!/usr/bin/env python3
"""
Local spatial cleanup of downloaded seed labels.

Why locally:
Earth Engine's connectedPixelCount has restrictive limits and exporting complex
focal expressions can be brittle. This script runs after download, uses scipy
and rasterio, and is easier to inspect and reproduce.

Input label:
  0 = no-burn candidate
  1 = burn candidate
  255 = ignore / ambiguous / water / nodata

Output:
  0 = no-burn candidate
  1 = spatially cleaned burn candidate
  255 = ignore

Cleanup:
  - keep only burn candidates within a radius around each FIRMS/event point
  - binary opening removes isolated speckles
  - connected-component minimum area filter
  - does not convert ignore pixels into class 0 or 1
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from scipy import ndimage


FOCUS_BUFFER_M = 1200
OPENING_RADIUS_PIXELS = 1
MIN_COMPONENT_PIXELS = 100
IGNORE_VALUE = 255


def circular_structure(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def load_events(path):
    with open(path, encoding="utf-8") as handle:
        return {row["event_id"]: row for row in csv.DictReader(handle)}


def event_point_in_raster_crs(event, raster_crs):
    lon = float(event["lon"])
    lat = float(event["lat"])
    x, y = transform("EPSG:4326", raster_crs, [lon], [lat])
    return x[0], y[0]


def focus_mask(height, width, transform_affine, x, y, radius_m):
    rows, cols = np.indices((height, width))
    xs = transform_affine.c + (cols + 0.5) * transform_affine.a
    ys = transform_affine.f + (rows + 0.5) * transform_affine.e
    distance = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
    return distance <= radius_m


def process_one(label_path, event, out_dir):
    with rasterio.open(label_path) as src:
        label = src.read(1)
        profile = src.profile.copy()
        focus_x, focus_y = event_point_in_raster_crs(event, src.crs)
        near_focus = focus_mask(
            src.height, src.width, src.transform, focus_x, focus_y, FOCUS_BUFFER_M
        )

    original_burn = label == 1
    candidate = original_burn & near_focus

    # Remove isolated fragments, then retain components with at least 1 ha at 10 m.
    structure = circular_structure(OPENING_RADIUS_PIXELS)
    opened = ndimage.binary_opening(candidate, structure=structure)

    components, count = ndimage.label(opened, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(components.ravel())
    keep_ids = np.where(component_sizes >= MIN_COMPONENT_PIXELS)[0]
    keep_ids = keep_ids[keep_ids != 0]
    cleaned_burn = np.isin(components, keep_ids)

    # Preserve original class 0 and 255 exactly.
    output = label.copy()
    output[original_burn] = IGNORE_VALUE
    output[cleaned_burn] = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / label_path.name.replace("_label_seed.tif", "_label_clean.tif")

    profile.update(count=1, dtype="uint8", nodata=IGNORE_VALUE, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(output.astype(np.uint8), 1)

    total = output.size
    summary = {
        "event_id": event["event_id"],
        "source": str(label_path),
        "output": str(out_path),
        "burn_original_pct": float(100 * np.mean(label == 1)),
        "burn_clean_pct": float(100 * np.mean(output == 1)),
        "ignore_pct": float(100 * np.mean(output == IGNORE_VALUE)),
        "n_components_kept": int(len(keep_ids)),
        "total_pixels": int(total),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="events/events_santafe_dedup.csv")
    parser.add_argument("--root", default="/DATA/ralfonso/prithvi_wetlandburn")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    events = load_events(args.events)
    label_dir = root / "data/raw/labels"
    output_dir = root / "data/raw/labels_clean"

    paths = sorted(label_dir.glob("*_label_seed.tif"))
    if args.limit != 0:
        paths = paths[args.start:args.start + args.limit]
    else:
        paths = paths[args.start:]

    if not paths:
        raise RuntimeError(f"No se encontraron labels en {label_dir}")

    for path in paths:
        event_id = path.name.replace("_label_seed.tif", "")
        if event_id not in events:
            print(f"[SKIP] {path.name}: no existe {event_id} en events CSV")
            continue
        summary = process_one(path, events[event_id], output_dir)
        print(
            f"{event_id} | original={summary['burn_original_pct']:.2f}% | "
            f"clean={summary['burn_clean_pct']:.2f}% | "
            f"ignore={summary['ignore_pct']:.2f}% | "
            f"components={summary['n_components_kept']}"
        )


if __name__ == "__main__":
    main()

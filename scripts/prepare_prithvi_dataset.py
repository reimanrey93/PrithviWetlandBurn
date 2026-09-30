#!/usr/bin/env python3
"""
prepare_prithvi_dataset.py

Builds a first event-level split for Prithvi burn-scar fine-tuning.

Input:
  data/raw/post/<event>_post.tif        6 Sentinel-2 bands: B2,B3,B4,B8,B11,B12
  data/raw/labels_clean/<event>_label_clean.tif
  events/events_santafe_dedup.csv

Output:
  data/prithvi_512/
    train/images/*.npz, train/masks/*.npy
    val/images/*.npz,   val/masks/*.npy
    test/images/*.npz,  test/masks/*.npy
    manifests/{split_manifest.csv,dataset_summary.json}

Important:
  - Split is by EVENT, never by patches.
  - Each source chip is center-cropped to 512 x 512.
  - Images keep six channels in the order B2,B3,B4,B8,B11,B12.
  - Labels: 0 no-burn, 1 burn candidate, -1 ignored.
  - No event is removed here; this is only a reproducible first smoke-test dataset.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

TARGET_SIZE = 512
IGNORE_SOURCE = 255
IGNORE_TARGET = -1


def center_crop_2d(array, size):
    height, width = array.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"Chip demasiado pequeño: {height}x{width}, se esperaban >= {size}")
    top = (height - size) // 2
    left = (width - size) // 2
    return array[..., top:top + size, left:left + size]


def load_event_ids(events_csv):
    with open(events_csv, encoding="utf-8") as handle:
        return [row["event_id"] for row in csv.DictReader(handle)]


def event_splits(event_ids, seed, train_fraction, val_fraction):
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction debe ser menor que 1.")
    rng = np.random.default_rng(seed)
    shuffled = np.array(event_ids, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = min(max(int(round(n * train_fraction)), 1), n - 2)
    n_val = min(max(int(round(n * val_fraction)), 1), n - n_train - 1)
    return {
        "train": list(shuffled[:n_train]),
        "val": list(shuffled[n_train:n_train + n_val]),
        "test": list(shuffled[n_train + n_val:]),
    }


def ensure_dirs(root):
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)


def read_post_image(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
    if image.shape[0] != 6:
        raise ValueError(f"{path.name}: se esperaban 6 bandas, llegaron {image.shape[0]}")
    image = center_crop_2d(image, TARGET_SIZE)
    image[~np.isfinite(image)] = 0.0
    return np.clip(image, 0.0, 1.0)


def read_label(path):
    with rasterio.open(path) as src:
        label = src.read(1)
    label = center_crop_2d(label, TARGET_SIZE).astype(np.int16)
    valid_values = np.isin(label, [0, 1, IGNORE_SOURCE])
    if not np.all(valid_values):
        unknown = np.unique(label[~valid_values])
        raise ValueError(f"{path.name}: valores inesperados: {unknown[:10]}")
    label[label == IGNORE_SOURCE] = IGNORE_TARGET
    return label


def write_sample(out_root, split, event_id, image, label):
    np.savez_compressed(
        out_root / split / "images" / f"{event_id}.npz",
        image=image.astype(np.float32),
    )
    np.save(out_root / split / "masks" / f"{event_id}.npy", label.astype(np.int16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/DATA/ralfonso/prithvi_wetlandburn")
    parser.add_argument("--events", default="events/events_santafe_dedup.csv")
    parser.add_argument("--output", default="data/prithvi_512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    events_csv = (project_root / args.events).resolve()
    out_root = (data_root / args.output).resolve()
    post_dir = data_root / "data/raw/post"
    label_dir = data_root / "data/raw/labels_clean"

    event_ids = load_event_ids(events_csv)
    splits = event_splits(event_ids, args.seed, args.train_fraction, args.val_fraction)
    lookup = {event_id: split for split, ids in splits.items() for event_id in ids}
    ensure_dirs(out_root)

    rows = []
    summary = {
        "seed": args.seed,
        "target_size": TARGET_SIZE,
        "channels": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "label_mapping": {"0": "no_burn", "1": "burn_candidate", "-1": "ignore"},
        "splits": {},
        "global": {"events_written": 0, "burn_pixels": 0, "valid_pixels": 0},
    }

    for event_id in event_ids:
        split = lookup[event_id]
        image_path = post_dir / f"{event_id}_post.tif"
        label_path = label_dir / f"{event_id}_label_clean.tif"
        if not image_path.exists() or not label_path.exists():
            print(f"[SKIP] {event_id}: faltan archivos")
            continue
        try:
            image = read_post_image(image_path)
            label = read_label(label_path)
        except Exception as error:
            print(f"[ERROR] {event_id}: {error}")
            continue

        write_sample(out_root, split, event_id, image, label)
        burn_pixels = int((label == 1).sum())
        valid_pixels = int((label >= 0).sum())
        ignore_pixels = int((label < 0).sum())
        burn_pct_chip = 100 * burn_pixels / label.size
        burn_pct_valid = 100 * burn_pixels / valid_pixels if valid_pixels else 0.0

        rows.append({
            "event_id": event_id,
            "split": split,
            "burn_pixels": burn_pixels,
            "valid_pixels": valid_pixels,
            "ignore_pixels": ignore_pixels,
            "burn_pct_chip": f"{burn_pct_chip:.6f}",
            "burn_pct_valid": f"{burn_pct_valid:.6f}",
        })

        s = summary["splits"].setdefault(
            split, {"events": 0, "burn_pixels": 0, "valid_pixels": 0}
        )
        s["events"] += 1
        s["burn_pixels"] += burn_pixels
        s["valid_pixels"] += valid_pixels
        summary["global"]["events_written"] += 1
        summary["global"]["burn_pixels"] += burn_pixels
        summary["global"]["valid_pixels"] += valid_pixels
        print(f"[OK] {event_id} -> {split} | burn={burn_pct_chip:.3f}% | valid={100*valid_pixels/label.size:.2f}%")

    for s in summary["splits"].values():
        s["burn_pct_valid"] = 100 * s["burn_pixels"] / s["valid_pixels"] if s["valid_pixels"] else 0.0
    g = summary["global"]
    g["burn_pct_valid"] = 100 * g["burn_pixels"] / g["valid_pixels"] if g["valid_pixels"] else 0.0

    manifest = out_root / "manifests" / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        fields = ["event_id", "split", "burn_pixels", "valid_pixels", "ignore_pixels", "burn_pct_chip", "burn_pct_valid"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_root / "manifests" / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== RESUMEN ===")
    for split in ("train", "val", "test"):
        s = summary["splits"].get(split, {})
        print(f"{split}: {s.get('events', 0)} eventos | burn(valid)={s.get('burn_pct_valid', 0):.4f}%")
    print(f"Escrito: {g['events_written']} eventos")
    print(f"Manifest: {manifest}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

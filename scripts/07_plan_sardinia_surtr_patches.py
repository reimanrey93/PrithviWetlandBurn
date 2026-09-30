#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
import rasterio


EVENT_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/events"
)

SPLIT_CSV = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/splits/"
    "surtr_sardinia_v1_event_split.csv"
)

OUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/patches_v1/manifests"
)

OUT_CSV = OUT_DIR / "patch_plan.csv"

PATCH = 256
STRIDE = 128


def positions(length, patch, stride):
    if length < patch:
        raise ValueError(
            f"Dimension {length} < patch {patch}"
        )

    pos = list(
        range(0, length - patch + 1, stride)
    )

    last = length - patch

    if not pos or pos[-1] != last:
        pos.append(last)

    return sorted(set(pos))


def load_splits():
    with open(SPLIT_CSV, encoding="utf-8") as f:
        return {
            row["event_id"]: row["split"]
            for row in csv.DictReader(f)
        }


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = load_splits()

    rows = []

    print("=" * 78)
    print("SURTR SARDINIA V1 PATCH PLAN")
    print("=" * 78)

    for event_id in sorted(splits):

        split = splits[event_id]
        event_dir = EVENT_ROOT / event_id

        mask_path = event_dir / "mask.tif"

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        h, w = mask.shape

        ys = positions(h, PATCH, STRIDE)
        xs = positions(w, PATCH, STRIDE)

        event_rows = []

        for row in ys:
            for col in xs:

                patch = mask[
                    row:row + PATCH,
                    col:col + PATCH,
                ]

                # Official burn mask is expected to be binary.
                burn_pixels = int(
                    np.count_nonzero(patch == 1)
                )

                total_pixels = int(patch.size)

                burn_fraction = (
                    burn_pixels / total_pixels
                )

                patch_id = (
                    f"{event_id}"
                    f"__r{row:04d}"
                    f"_c{col:04d}"
                )

                record = {
                    "patch_id": patch_id,
                    "event_id": event_id,
                    "split": split,
                    "row": row,
                    "col": col,
                    "height": PATCH,
                    "width": PATCH,
                    "burn_pixels": burn_pixels,
                    "burn_fraction": burn_fraction,
                }

                rows.append(record)
                event_rows.append(record)

        burn_fracs = np.array(
            [r["burn_fraction"] for r in event_rows],
            dtype=float,
        )

        print(
            f"{event_id:28s} "
            f"{split:5s} "
            f"{w:4d}x{h:<4d} "
            f"patches={len(event_rows):3d} "
            f"positive={(burn_fracs > 0).sum():3d} "
            f"mean_burn={100*burn_fracs.mean():6.2f}% "
            f"max={100*burn_fracs.max():6.2f}%"
        )

    fields = [
        "patch_id",
        "event_id",
        "split",
        "row",
        "col",
        "height",
        "width",
        "burn_pixels",
        "burn_fraction",
    ]

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for split in ("train", "val", "test"):

        rr = [
            r for r in rows
            if r["split"] == split
        ]

        events = {
            r["event_id"]
            for r in rr
        }

        positive = sum(
            r["burn_pixels"] > 0
            for r in rr
        )

        print(
            f"{split:5s}: "
            f"events={len(events):2d} "
            f"patches={len(rr):4d} "
            f"positive={positive:4d} "
            f"zero_burn={len(rr)-positive:4d}"
        )

    print()
    print("TOTAL PATCHES:", len(rows))
    print("Manifest:", OUT_CSV)
    print("=" * 78)


if __name__ == "__main__":
    main()

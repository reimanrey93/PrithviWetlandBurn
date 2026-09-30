#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import rasterio


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/patches_v1"
)

SOURCE = ROOT / "manifests" / "patch_manifest.csv"

OUT = ROOT / "manifests" / "surtr_sardinia_v1_FROZEN.csv"

SUMMARY = ROOT / "manifests" / "surtr_sardinia_v1_FROZEN.json"

MIN_VALID_FRACTION = 0.25


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():

    with open(SOURCE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    excluded = []

    for row in rows:

        mask_path = Path(row["mask_path"])
        valid_path = Path(row["valid_path"])

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        with rasterio.open(valid_path) as src:
            valid = src.read(1)

        v = valid == 1
        b = (mask == 1) & v

        valid_pixels = int(v.sum())
        burn_valid_pixels = int(b.sum())

        valid_fraction = (
            valid_pixels / valid.size
        )

        if valid_pixels > 0:
            burn_fraction_valid = (
                burn_valid_pixels /
                valid_pixels
            )
        else:
            burn_fraction_valid = 0.0

        out = dict(row)

        out["valid_pixels_final"] = (
            valid_pixels
        )

        out["valid_fraction_final"] = (
            f"{valid_fraction:.8f}"
        )

        out["burn_valid_pixels"] = (
            burn_valid_pixels
        )

        out["burn_fraction_valid"] = (
            f"{burn_fraction_valid:.8f}"
        )

        if valid_fraction >= MIN_VALID_FRACTION:
            kept.append(out)
        else:
            excluded.append(out)

    fields = list(kept[0].keys())

    with open(
        OUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(kept)

    counts = {}

    for split in ("train", "val", "test"):

        rr = [
            r for r in kept
            if r["split"] == split
        ]

        counts[split] = {
            "patches": len(rr),
            "events": len({
                r["event_id"]
                for r in rr
            }),
        }

    summary = {
        "dataset": "SURTR Sardinia V1",
        "patch_size": 256,
        "resolution_m": 20,
        "min_valid_fraction": (
            MIN_VALID_FRACTION
        ),
        "materialized_patches": len(rows),
        "included_patches": len(kept),
        "excluded_patches": len(excluded),
        "counts": counts,
        "excluded_patch_ids": [
            r["patch_id"]
            for r in excluded
        ],
    }

    with open(
        SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    digest = sha256(OUT)

    print("=" * 72)
    print("SURTR SARDINIA V1 — FROZEN")
    print("=" * 72)

    print(
        "Materialized:",
        len(rows)
    )

    print(
        "Included    :",
        len(kept)
    )

    print(
        "Excluded    :",
        len(excluded)
    )

    print()

    for split, x in counts.items():
        print(
            f"{split:5s}: "
            f"{x['patches']:3d} patches | "
            f"{x['events']:2d} events"
        )

    print()

    print("Excluded patches:")

    for r in excluded:
        print(
            " ",
            r["patch_id"],
            "valid=",
            r["valid_fraction_final"],
            "burn_valid=",
            r["burn_fraction_valid"],
        )

    print()
    print("Manifest:", OUT)
    print("SHA256  :", digest)
    print("Summary :", SUMMARY)
    print("=" * 72)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
import rasterio


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/patches_v1"
)

MANIFEST = ROOT / "manifests" / "patch_manifest.csv"

OUT = ROOT / "manifests" / "patch_dataset_qc.csv"

EXPECTED_SHAPE = (6, 256, 256)


def same_transform(a, b):
    return np.allclose(
        tuple(a),
        tuple(b),
        rtol=0,
        atol=1e-10,
    )


def main():

    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    failures = []

    for i, row in enumerate(rows, 1):

        patch_id = row["patch_id"]

        paths = {
            "pre": Path(row["pre_path"]),
            "post": Path(row["post_path"]),
            "mask": Path(row["mask_path"]),
            "valid": Path(row["valid_path"]),
        }

        errors = []

        # ----------------------------------------------------------
        # Files exist
        # ----------------------------------------------------------

        for name, path in paths.items():
            if not path.exists():
                errors.append(
                    f"missing_{name}"
                )

        if errors:
            failures.append(
                (patch_id, ";".join(errors))
            )
            continue

        # ----------------------------------------------------------
        # Read metadata + arrays
        # ----------------------------------------------------------

        with rasterio.open(paths["pre"]) as src:
            pre = src.read()
            pre_crs = src.crs
            pre_transform = src.transform

        with rasterio.open(paths["post"]) as src:
            post = src.read()
            post_crs = src.crs
            post_transform = src.transform

        with rasterio.open(paths["mask"]) as src:
            mask = src.read(1)
            mask_crs = src.crs
            mask_transform = src.transform

        with rasterio.open(paths["valid"]) as src:
            valid = src.read(1)
            valid_crs = src.crs
            valid_transform = src.transform

        # ----------------------------------------------------------
        # Shapes
        # ----------------------------------------------------------

        if pre.shape != EXPECTED_SHAPE:
            errors.append(
                f"pre_shape={pre.shape}"
            )

        if post.shape != EXPECTED_SHAPE:
            errors.append(
                f"post_shape={post.shape}"
            )

        if mask.shape != (256, 256):
            errors.append(
                f"mask_shape={mask.shape}"
            )

        if valid.shape != (256, 256):
            errors.append(
                f"valid_shape={valid.shape}"
            )

        # ----------------------------------------------------------
        # CRS
        # ----------------------------------------------------------

        if not (
            pre_crs
            == post_crs
            == mask_crs
            == valid_crs
        ):
            errors.append("crs_mismatch")

        # ----------------------------------------------------------
        # Transform
        # ----------------------------------------------------------

        if not (
            same_transform(
                pre_transform,
                post_transform,
            )
            and same_transform(
                pre_transform,
                mask_transform,
            )
            and same_transform(
                pre_transform,
                valid_transform,
            )
        ):
            errors.append("transform_mismatch")

        # ----------------------------------------------------------
        # Finite reflectance
        # ----------------------------------------------------------

        pre_finite = np.isfinite(pre)
        post_finite = np.isfinite(post)

        if not pre_finite.all():
            errors.append("pre_nonfinite")

        if not post_finite.all():
            errors.append("post_nonfinite")

        # ----------------------------------------------------------
        # Binary masks
        # ----------------------------------------------------------

        mask_values = set(
            np.unique(mask).tolist()
        )

        valid_values = set(
            np.unique(valid).tolist()
        )

        if not mask_values.issubset({0, 1}):
            errors.append(
                f"mask_values={mask_values}"
            )

        if not valid_values.issubset({0, 1}):
            errors.append(
                f"valid_values={valid_values}"
            )

        # ----------------------------------------------------------
        # Compare manifest statistics
        # ----------------------------------------------------------

        burn_pixels = int(
            np.count_nonzero(mask == 1)
        )

        valid_pixels = int(
            np.count_nonzero(valid == 1)
        )

        expected_burn = int(
            row["burn_pixels"]
        )

        if burn_pixels != expected_burn:
            errors.append(
                "burn_count_mismatch"
            )

        valid_fraction = (
            valid_pixels / valid.size
        )


        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        result = {
            "patch_id": patch_id,
            "event_id": row["event_id"],
            "split": row["split"],
            "burn_pixels": burn_pixels,
            "burn_fraction": (
                burn_pixels / mask.size
            ),
            "valid_fraction": valid_fraction,
            "pre_min": float(pre.min()),
            "pre_max": float(pre.max()),
            "post_min": float(post.min()),
            "post_max": float(post.max()),
            "status": (
                "PASS"
                if not errors
                else "FAIL"
            ),
            "errors": ";".join(errors),
        }

        results.append(result)

        if errors:
            failures.append(
                (patch_id, ";".join(errors))
            )

        if (
            i % 50 == 0
            or i == len(rows)
        ):
            print(
                f"[{i:3d}/{len(rows)}] "
                f"{patch_id}"
            )

    # --------------------------------------------------------------
    # Write QC table
    # --------------------------------------------------------------

    fields = list(results[0].keys())

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
        writer.writerows(results)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 76)
    print("SURTR SARDINIA PATCH DATASET QC")
    print("=" * 76)

    for split in ("train", "val", "test"):

        rr = [
            r for r in results
            if r["split"] == split
        ]

        passed = sum(
            r["status"] == "PASS"
            for r in rr
        )

        print(
            f"{split:5s}: "
            f"{passed:3d}/{len(rr):3d} PASS"
        )

    print()
    print(
        "TOTAL:",
        len(results),
        "checked"
    )

    print(
        "FAILURES:",
        len(failures)
    )

    if failures:
        print()
        print("FAILURE DETAILS")

        for patch_id, error in failures[:20]:
            print(
                patch_id,
                "->",
                error,
            )

    # Dataset-wide ranges
    if results:

        print()
        print(
            "PRE range :",
            min(r["pre_min"] for r in results),
            "to",
            max(r["pre_max"] for r in results),
        )

        print(
            "POST range:",
            min(r["post_min"] for r in results),
            "to",
            max(r["post_max"] for r in results),
        )

        print(
            "VALID min :",
            min(
                r["valid_fraction"]
                for r in results
            ),
        )

    print()
    print("QC table:", OUT)
    print("=" * 76)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

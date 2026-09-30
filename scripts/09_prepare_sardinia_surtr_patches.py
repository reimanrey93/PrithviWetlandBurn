#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/events"
)

MANIFEST_IN = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/patch_plan_qc.csv"
)

OUT_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1"
)

MANIFEST_OUT = (
    OUT_ROOT / "manifests" / "patch_manifest.csv"
)

VALID_SCL = {2, 4, 5, 6, 7}


def write_raster(
    path,
    array,
    profile,
    transform,
    dtype,
    nodata=None,
):

    p = profile.copy()

    if array.ndim == 2:
        count = 1
    else:
        count = array.shape[0]

    p.update(
        driver="GTiff",
        height=array.shape[-2],
        width=array.shape[-1],
        count=count,
        dtype=dtype,
        transform=transform,
        compress="deflate",
        tiled=True,
        nodata=nodata,
    )

    with rasterio.open(path, "w", **p) as dst:

        if array.ndim == 2:
            dst.write(
                array.astype(dtype),
                1,
            )
        else:
            dst.write(
                array.astype(dtype)
            )


def main():

    with open(
        MANIFEST_IN,
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    # Clean only generated split folders.
    for split in ("train", "val", "test"):

        split_dir = OUT_ROOT / split

        if split_dir.exists():
            shutil.rmtree(split_dir)

        for sub in (
            "pre",
            "post",
            "masks",
            "valid",
        ):
            (
                split_dir / sub
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    output_rows = []

    current_event = None
    datasets = {}

    try:

        for i, row in enumerate(rows, 1):

            event = row["event_id"]
            split = row["split"]
            patch_id = row["patch_id"]

            if event != current_event:

                for ds in datasets.values():
                    ds.close()

                d = ROOT / event

                datasets = {
                    "pre": rasterio.open(
                        d / "pre.tif"
                    ),
                    "post": rasterio.open(
                        d / "post.tif"
                    ),
                    "mask": rasterio.open(
                        d / "mask.tif"
                    ),
                    "scl_pre": rasterio.open(
                        d / "scl_pre.tif"
                    ),
                    "scl_post": rasterio.open(
                        d / "scl_post.tif"
                    ),
                }

                current_event = event

            y = int(row["row"])
            x = int(row["col"])
            h = int(row["height"])
            w = int(row["width"])

            win = Window(
                x,
                y,
                w,
                h,
            )

            pre = datasets["pre"].read(
                window=win
            )

            post = datasets["post"].read(
                window=win
            )

            mask = datasets["mask"].read(
                1,
                window=win,
            )

            scl_pre = datasets[
                "scl_pre"
            ].read(
                1,
                window=win,
            )

            scl_post = datasets[
                "scl_post"
            ].read(
                1,
                window=win,
            )

            if (
                pre.shape[-2:] != (h, w)
                or post.shape[-2:] != (h, w)
                or mask.shape != (h, w)
            ):
                raise RuntimeError(
                    f"Bad shape: {patch_id}"
                )

            # --------------------------------------------------
            # Authoritative pixel validity:
            #   - acceptable PRE SCL
            #   - acceptable POST SCL
            #   - official GT exists (mask != 255)
            #   - finite / non-nodata PRE and POST radiometry
            # --------------------------------------------------

            valid_scl = (
                np.isin(
                    scl_pre,
                    list(VALID_SCL),
                )
                &
                np.isin(
                    scl_post,
                    list(VALID_SCL),
                )
            )

            valid_gt = (
                mask != 255
            )

            valid_pre_rad = (
                np.all(np.isfinite(pre), axis=0)
                &
                np.all(pre != -9999.0, axis=0)
            )

            valid_post_rad = (
                np.all(np.isfinite(post), axis=0)
                &
                np.all(post != -9999.0, axis=0)
            )

            valid = (
                valid_scl
                & valid_gt
                & valid_pre_rad
                & valid_post_rad
            ).astype(np.uint8)

            # --------------------------------------------------
            # Binary official fire mask.
            # 0   = unburned
            # 1   = burned
            # 255 = nodata in source mask
            # --------------------------------------------------

            mask = (
                mask == 1
            ).astype(np.uint8)

            # --------------------------------------------------
            # Never expose invalid/nodata radiometry to Prithvi.
            #
            # Pixels excluded by the joint PRE/POST SCL validity
            # mask are filled with a numerically neutral value.
            # The valid raster remains the authoritative mask for
            # loss and evaluation.
            # --------------------------------------------------

            pre = pre.astype(
                np.float32,
                copy=False,
            )

            post = post.astype(
                np.float32,
                copy=False,
            )

            invalid = valid == 0

            pre[:, invalid] = 0.0
            post[:, invalid] = 0.0

            # Additional safety for source nodata/non-finite values.
            pre[
                (~np.isfinite(pre))
                | (pre == -9999.0)
            ] = 0.0

            post[
                (~np.isfinite(post))
                | (post == -9999.0)
            ] = 0.0

            transform = window_transform(
                win,
                datasets["pre"].transform,
            )

            pre_path = (
                OUT_ROOT / split / "pre" /
                f"{patch_id}.tif"
            )

            post_path = (
                OUT_ROOT / split / "post" /
                f"{patch_id}.tif"
            )

            mask_path = (
                OUT_ROOT / split / "masks" /
                f"{patch_id}.tif"
            )

            valid_path = (
                OUT_ROOT / split / "valid" /
                f"{patch_id}.tif"
            )

            write_raster(
                pre_path,
                pre,
                datasets["pre"].profile,
                transform,
                "float32",
            )

            write_raster(
                post_path,
                post,
                datasets["post"].profile,
                transform,
                "float32",
            )

            write_raster(
                mask_path,
                mask,
                datasets["mask"].profile,
                transform,
                "uint8",
                nodata=255,
            )

            write_raster(
                valid_path,
                valid,
                datasets["mask"].profile,
                transform,
                "uint8",
                nodata=255,
            )

            out = dict(row)

            out.update({
                "pre_path": str(pre_path),
                "post_path": str(post_path),
                "mask_path": str(mask_path),
                "valid_path": str(valid_path),
            })

            output_rows.append(out)

            if (
                i % 25 == 0
                or i == len(rows)
            ):
                print(
                    f"[{i:3d}/{len(rows)}] "
                    f"{event} "
                    f"{patch_id}"
                )

    finally:

        for ds in datasets.values():
            ds.close()

    fields = list(
        output_rows[0].keys()
    )

    with open(
        MANIFEST_OUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("=" * 72)
    print("SURTR PATCH DATASET CREATED")
    print("=" * 72)

    for split in (
        "train",
        "val",
        "test",
    ):

        rr = [
            r for r in output_rows
            if r["split"] == split
        ]

        print(
            f"{split:5s}: "
            f"{len(rr):3d} patches"
        )

    print(
        "TOTAL:",
        len(output_rows),
    )

    print(
        "Manifest:",
        MANIFEST_OUT,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()

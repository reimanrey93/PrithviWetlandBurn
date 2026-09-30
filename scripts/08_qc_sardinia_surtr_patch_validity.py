#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
import rasterio


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/events"
)

PLAN = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/patch_plan.csv"
)

OUT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/patch_plan_qc.csv"
)

VALID_SCL = {2, 4, 5, 6, 7}


def main():

    with open(PLAN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cache = {}

    for row in rows:

        event = row["event_id"]

        if event not in cache:

            d = ROOT / event

            with rasterio.open(
                d / "scl_pre.tif"
            ) as src:
                pre = src.read(1)

            with rasterio.open(
                d / "scl_post.tif"
            ) as src:
                post = src.read(1)

            cache[event] = (pre, post)

        pre, post = cache[event]

        y = int(row["row"])
        x = int(row["col"])
        h = int(row["height"])
        w = int(row["width"])

        pre_p = pre[y:y+h, x:x+w]
        post_p = post[y:y+h, x:x+w]

        valid_pre = np.isin(
            pre_p,
            list(VALID_SCL),
        )

        valid_post = np.isin(
            post_p,
            list(VALID_SCL),
        )

        valid_joint = (
            valid_pre & valid_post
        )

        row["valid_pre_fraction"] = (
            f"{valid_pre.mean():.8f}"
        )

        row["valid_post_fraction"] = (
            f"{valid_post.mean():.8f}"
        )

        row["valid_joint_fraction"] = (
            f"{valid_joint.mean():.8f}"
        )

    fields = list(rows[0].keys())

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
        writer.writerows(rows)

    print("=" * 76)
    print("SURTR PATCH VALIDITY QC")
    print("=" * 76)

    for split in (
        "train",
        "val",
        "test",
    ):

        rr = [
            r for r in rows
            if r["split"] == split
        ]

        joint = np.array(
            [
                float(
                    r["valid_joint_fraction"]
                )
                for r in rr
            ]
        )

        print()
        print(
            split.upper(),
            f"n={len(rr)}"
        )

        print(
            " joint valid "
            f"min={100*joint.min():6.2f}% "
            f"p10={100*np.percentile(joint,10):6.2f}% "
            f"median={100*np.median(joint):6.2f}% "
            f"mean={100*joint.mean():6.2f}% "
            f"max={100*joint.max():6.2f}%"
        )

        for threshold in (
            0.70,
            0.80,
            0.90,
            0.95,
        ):

            n = int(
                (joint >= threshold).sum()
            )

            print(
                f" >= {threshold:.2f}: "
                f"{n:3d}/{len(rr):3d}"
            )

    print()
    print("Output:", OUT)
    print("=" * 76)


if __name__ == "__main__":
    main()

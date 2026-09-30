from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/events"
)

OUT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/qc"
)
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "pre.tif",
    "post.tif",
    "scl_pre.tif",
    "scl_post.tif",
    "dnbr.tif",
    "mask.tif",
    "metadata.json",
    "_SUCCESS",
]

RASTERS = [
    "pre.tif",
    "post.tif",
    "scl_pre.tif",
    "scl_post.tif",
    "dnbr.tif",
    "mask.tif",
]


def raster_info(path):
    with rasterio.open(path) as ds:
        # Important: cast BEFORE filling NaN.
        a = ds.read(masked=True).astype(np.float32)
        vals = a.filled(np.nan)

        finite = np.isfinite(vals)

        return {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "crs": str(ds.crs),
            "res_x": float(ds.res[0]),
            "res_y": float(ds.res[1]),
            "transform": tuple(ds.transform),
            "finite_fraction": (
                float(finite.mean())
                if finite.size
                else np.nan
            ),
        }


rows = []

event_dirs = sorted(
    p for p in ROOT.iterdir()
    if p.is_dir()
)

print("=" * 80)
print("SARDINIA SURTR EVENT QC")
print("=" * 80)
print("Events:", len(event_dirs))


for i, event_dir in enumerate(event_dirs, 1):

    event_id = event_dir.name

    print(
        f"[{i:02d}/{len(event_dirs):02d}] "
        f"{event_id}"
    )

    row = {
        "event_id": event_id,
    }

    missing = [
        f for f in REQUIRED
        if not (event_dir / f).exists()
    ]

    row["files_complete"] = len(missing) == 0
    row["missing_files"] = ";".join(missing)

    try:
        with open(
            event_dir / "metadata.json",
            "r",
        ) as f:
            meta = json.load(f)

        fire_date = pd.to_datetime(
            meta["fire_date"]
        )
        pre_date = pd.to_datetime(
            meta["pre_date"]
        )
        post_date = pd.to_datetime(
            meta["post_date"]
        )

        row.update({
            "year": meta.get("year"),
            "municipality": meta.get(
                "municipality"
            ),
            "fire_date": fire_date.date(),
            "pre_date": pre_date.date(),
            "post_date": post_date.date(),

            "days_pre_to_fire": int(
                (fire_date - pre_date).days
            ),
            "days_fire_to_post": int(
                (post_date - fire_date).days
            ),

            "official_area_ha": float(
                meta["official_geometry_area_ha"]
            ),
            "raster_area_ha": float(
                meta["burned_area_raster_ha"]
            ),
            "valid_area_ha": float(
                meta["valid_area_ha"]
            ),
            "burned_pixels_meta": int(
                meta["burned_pixels"]
            ),
            "valid_pixels_meta": int(
                meta["valid_pixels"]
            ),

            "pre_item_id": meta.get(
                "pre_item_id"
            ),
            "post_item_id": meta.get(
                "post_item_id"
            ),

            "dnbr_min_meta": meta.get(
                "dnbr_min"
            ),
            "dnbr_mean_meta": meta.get(
                "dnbr_mean"
            ),
            "dnbr_max_meta": meta.get(
                "dnbr_max"
            ),
        })

        official = row["official_area_ha"]
        raster = row["raster_area_ha"]

        row["area_error_ha"] = (
            raster - official
        )

        row["area_abs_error_ha"] = abs(
            raster - official
        )

        row["area_error_pct"] = (
            100.0
            * (raster - official)
            / official
            if official > 0
            else np.nan
        )

        # --------------------------------------------------
        # Raster alignment
        # --------------------------------------------------

        infos = {}

        for name in RASTERS:
            infos[name] = raster_info(
                event_dir / name
            )

        reference = infos["pre.tif"]

        aligned = True

        for name, info in infos.items():

            same = (
                info["width"]
                == reference["width"]
                and info["height"]
                == reference["height"]
                and info["crs"]
                == reference["crs"]
                and np.allclose(
                    info["transform"],
                    reference["transform"],
                )
            )

            if not same:
                aligned = False

        row["rasters_aligned"] = aligned

        row["width"] = reference["width"]
        row["height"] = reference["height"]
        row["crs"] = reference["crs"]
        row["resolution_m"] = (
            reference["res_x"]
        )

        row["pre_bands"] = (
            infos["pre.tif"]["count"]
        )
        row["post_bands"] = (
            infos["post.tif"]["count"]
        )

        row["pre_finite_fraction"] = (
            infos["pre.tif"][
                "finite_fraction"
            ]
        )
        row["post_finite_fraction"] = (
            infos["post.tif"][
                "finite_fraction"
            ]
        )
        row["dnbr_finite_fraction"] = (
            infos["dnbr.tif"][
                "finite_fraction"
            ]
        )

        # --------------------------------------------------
        # Mask QC
        # --------------------------------------------------

        with rasterio.open(
            event_dir / "mask.tif"
        ) as ds:

            mask = ds.read(
                1,
                masked=True,
            ).astype(np.float32)

            mask = mask.filled(np.nan)

        finite_mask = mask[
            np.isfinite(mask)
        ]

        unique = np.unique(
            finite_mask
        )

        row["mask_values"] = ";".join(
            str(float(x))
            for x in unique
        )

        row["mask_binary"] = bool(
            set(unique.tolist())
            <= {0.0, 1.0}
        )

        burned_pixels_raster = int(
            np.sum(mask == 1)
        )

        row[
            "burned_pixels_raster"
        ] = burned_pixels_raster

        pixel_area_ha = (
            reference["res_x"]
            * reference["res_y"]
            / 10000.0
        )

        area_from_mask = (
            burned_pixels_raster
            * pixel_area_ha
        )

        row[
            "area_from_mask_ha"
        ] = area_from_mask

        row[
            "mask_vs_meta_area_diff_ha"
        ] = abs(
            area_from_mask
            - row["raster_area_ha"]
        )

        # --------------------------------------------------
        # Temporal logic
        # --------------------------------------------------

        row["temporal_order_ok"] = bool(
            pre_date < fire_date < post_date
        )

        # --------------------------------------------------
        # Overall QC
        # --------------------------------------------------

        row["qc_pass"] = bool(
            row["files_complete"]
            and row["rasters_aligned"]
            and row["mask_binary"]
            and row["temporal_order_ok"]
            and row["pre_bands"] == 6
            and row["post_bands"] == 6
            and abs(
                row[
                    "mask_vs_meta_area_diff_ha"
                ]
            ) < 0.1
        )

        row["error"] = ""

    except Exception as exc:

        row["qc_pass"] = False
        row["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    rows.append(row)


df = pd.DataFrame(rows)

csv_path = (
    OUT
    / "sardinia_surtr_event_qc.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

failures = df[
    ~df["qc_pass"].fillna(False)
].copy()

fail_path = (
    OUT
    / "sardinia_surtr_qc_failures.csv"
)

failures.to_csv(
    fail_path,
    index=False,
)


print()
print("=" * 80)
print("QC SUMMARY")
print("=" * 80)

print("Events:", len(df))
print(
    "PASS:",
    int(df["qc_pass"].sum())
)
print(
    "FAIL:",
    int((~df["qc_pass"]).sum())
)

if "area_abs_error_ha" in df:
    print()
    print("Raster vs official area:")
    print(
        "  mean abs error [ha]:",
        round(
            df[
                "area_abs_error_ha"
            ].mean(),
            4,
        ),
    )
    print(
        "  median abs error [ha]:",
        round(
            df[
                "area_abs_error_ha"
            ].median(),
            4,
        ),
    )
    print(
        "  max abs error [ha]:",
        round(
            df[
                "area_abs_error_ha"
            ].max(),
            4,
        ),
    )

    print(
        "  mean abs error [%]:",
        round(
            df[
                "area_error_pct"
            ].abs().mean(),
            4,
        ),
    )
    print(
        "  max abs error [%]:",
        round(
            df[
                "area_error_pct"
            ].abs().max(),
            4,
        ),
    )

print()
print("Temporal spacing:")

print(
    "  PRE -> fire median [days]:",
    df[
        "days_pre_to_fire"
    ].median(),
)

print(
    "  fire -> POST median [days]:",
    df[
        "days_fire_to_post"
    ].median(),
)

print()
print(
    "QC table:",
    csv_path,
)

if len(failures):
    print(
        "Failures:",
        fail_path,
    )

print("=" * 80)

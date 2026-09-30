#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import transform


STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

PAIR_CSV = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog/sardinia_s2_good_pairs.csv"
)

FIRE_GPKG = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog/sardinia_candidates_ge100ha.gpkg"
)

OUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_QC = OUT_DIR / "sardinia_s2_pairs_scl_qc.csv"
OUT_SELECTED = OUT_DIR / "sardinia_selected_28.csv"
OUT_SELECTED_GPKG = OUT_DIR / "sardinia_selected_28.gpkg"

BUFFER_M = 2000

MAX_LOCAL_BAD_PCT = 5.0
MAX_LOCAL_INVALID_PCT = 5.0

MAX_PER_YEAR = 4

METRIC_CRS = "EPSG:32632"

# Sen2Cor SCL
BAD_SCL = {3, 8, 9, 10, 11}
INVALID_SCL = {0, 1}


def get_item(client, item_id):
    search = client.search(
        collections=[COLLECTION],
        ids=[item_id],
        max_items=1,
    )

    items = list(search.items())

    if not items:
        raise RuntimeError(
            f"STAC item not found: {item_id}"
        )

    return items[0]


def find_scl_asset(item):
    # Earth Search normally uses "scl".
    for key in ("scl", "SCL"):
        if key in item.assets:
            return item.assets[key]

    # Fallback based on metadata.
    for key, asset in item.assets.items():
        title = (asset.title or "").lower()

        if (
            "scene classification" in title
            or key.lower() == "scl"
        ):
            return asset

    raise KeyError(
        f"No SCL asset in {item.id}. "
        f"Available: {list(item.assets)}"
    )


def inspect_scl(
    href,
    aoi_metric,
):
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ):
        with rasterio.open(href) as src:

            if src.crs is None:
                raise RuntimeError(
                    f"SCL has no CRS: {href}"
                )

            transformer = Transformer.from_crs(
                METRIC_CRS,
                src.crs,
                always_xy=True,
            )

            aoi_src = transform(
                transformer.transform,
                aoi_metric,
            )

            arr, _ = mask(
                src,
                [mapping(aoi_src)],
                crop=True,
                filled=False,
                indexes=1,
            )

    values = np.asarray(arr)
    inside = ~np.ma.getmaskarray(arr)

    if inside.sum() == 0:
        raise RuntimeError(
            "AOI contains no SCL pixels."
        )

    values = values[inside]

    total = values.size

    bad = np.isin(
        values,
        list(BAD_SCL),
    ).sum()

    invalid = np.isin(
        values,
        list(INVALID_SCL),
    ).sum()

    bad_pct = 100.0 * bad / total
    invalid_pct = 100.0 * invalid / total

    usable_pct = (
        100.0
        - bad_pct
        - invalid_pct
    )

    return {
        "local_bad_pct": bad_pct,
        "local_invalid_pct": invalid_pct,
        "local_usable_pct": usable_pct,
        "scl_pixels": total,
    }


pairs = pd.read_csv(PAIR_CSV)

fires = gpd.read_file(
    FIRE_GPKG
).to_crs(METRIC_CRS)

fires = fires.set_index(
    "event_id"
)

print("Pairs:", len(pairs))
print("Opening STAC...")

client = Client.open(STAC_URL)

item_cache = {}
rows = []


for n, row in pairs.iterrows():

    event_id = row["event_id"]

    print()
    print(
        f"[{n + 1:03d}/{len(pairs):03d}] "
        f"{event_id}"
    )

    if event_id not in fires.index:
        print("  ERROR: geometry not found")
        continue

    fire = fires.loc[event_id]

    aoi = fire.geometry.buffer(
        BUFFER_M
    )

    result = row.to_dict()

    try:
        for phase in ("pre", "post"):

            item_id = row[
                f"{phase}_item_id"
            ]

            if item_id not in item_cache:
                item_cache[item_id] = get_item(
                    client,
                    item_id,
                )

            item = item_cache[item_id]

            scl_asset = find_scl_asset(
                item
            )

            qc = inspect_scl(
                scl_asset.href,
                aoi,
            )

            for key, value in qc.items():
                result[
                    f"{phase}_{key}"
                ] = value

            print(
                f"  {phase.upper():4s} "
                f"bad={qc['local_bad_pct']:.2f}% | "
                f"invalid={qc['local_invalid_pct']:.2f}% | "
                f"usable={qc['local_usable_pct']:.2f}%"
            )

        result["local_qc_ok"] = (
            result["pre_local_bad_pct"]
            <= MAX_LOCAL_BAD_PCT
            and result["post_local_bad_pct"]
            <= MAX_LOCAL_BAD_PCT
            and result["pre_local_invalid_pct"]
            <= MAX_LOCAL_INVALID_PCT
            and result["post_local_invalid_pct"]
            <= MAX_LOCAL_INVALID_PCT
        )

        # Higher is better.
        result["final_quality_score"] = (
            float(result["pair_score"])
            - 0.50 * (
                result["pre_local_bad_pct"]
                + result["post_local_bad_pct"]
            )
            - 0.25 * (
                result["pre_local_invalid_pct"]
                + result["post_local_invalid_pct"]
            )
        )

    except Exception as exc:

        print(
            "  QC ERROR:",
            exc,
        )

        result["local_qc_ok"] = False
        result["qc_error"] = str(exc)
        result["final_quality_score"] = np.nan

    rows.append(result)


qc = pd.DataFrame(rows)

qc.to_csv(
    OUT_QC,
    index=False,
)


# ============================================================
# FINAL PILOT SELECTION
# ============================================================

good = qc[
    qc["local_qc_ok"] == True
].copy()

good = good.sort_values(
    [
        "final_quality_score",
        "area_geom_ha",
    ],
    ascending=[
        False,
        False,
    ],
)


selected_parts = []

for year in sorted(
    good["year"].unique()
):

    year_rows = (
        good[
            good["year"] == year
        ]
        .head(MAX_PER_YEAR)
        .copy()
    )

    selected_parts.append(
        year_rows
    )


selected = pd.concat(
    selected_parts,
    ignore_index=True,
)

selected = selected.sort_values(
    [
        "year",
        "final_quality_score",
    ],
    ascending=[
        True,
        False,
    ],
)

selected.to_csv(
    OUT_SELECTED,
    index=False,
)


# ============================================================
# WRITE SELECTED GEOMETRIES
# ============================================================

selected_ids = set(
    selected["event_id"]
)

selected_geo = fires.loc[
    fires.index.isin(
        selected_ids
    )
].reset_index()

selected_geo = selected_geo.merge(
    selected[
        [
            "event_id",
            "pre_item_id",
            "post_item_id",
            "pre_date",
            "post_date",
            "pre_local_bad_pct",
            "post_local_bad_pct",
            "final_quality_score",
        ]
    ],
    on="event_id",
    how="left",
)

selected_geo.to_file(
    OUT_SELECTED_GPKG,
    layer="selected_fires",
    driver="GPKG",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("LOCAL SCL QC SUMMARY")
print("==========================================")

print(
    "Pairs inspected:",
    len(qc),
)

print(
    "Passing local QC:",
    len(good),
)

print(
    "Selected for pilot:",
    len(selected),
)

print()
print("Passing by year:")

print(
    good.groupby("year").size()
)

print()
print("Selected by year:")

print(
    selected.groupby("year").size()
)

print()
print("SELECTED EVENTS:")

cols = [
    "event_id",
    "year",
    "municipality",
    "area_geom_ha",
    "pre_date",
    "pre_local_bad_pct",
    "post_date",
    "post_local_bad_pct",
    "final_quality_score",
]

print(
    selected[cols]
    .to_string(
        index=False
    )
)

print()
print("Outputs:")
print(OUT_QC)
print(OUT_SELECTED)
print(OUT_SELECTED_GPKG)

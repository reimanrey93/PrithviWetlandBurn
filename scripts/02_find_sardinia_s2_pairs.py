#!/usr/bin/env python3

from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pystac_client import Client
from shapely.geometry import mapping, shape
from shapely.ops import transform


STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

CATALOG = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog/sardinia_candidates_ge100ha.gpkg"
)

OUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_DIR / "sardinia_s2_pair_search.csv"
OUT_GOOD = OUT_DIR / "sardinia_s2_good_pairs.csv"


# ------------------------------------------------------------
# Search configuration
# ------------------------------------------------------------

PRE_MIN = 5
PRE_MAX = 45

POST_MIN = 3
POST_MAX = 45

BUFFER_M = 2000

MAX_TILE_CLOUD = 80.0
MIN_COVERAGE = 0.95

METRIC_CRS = "EPSG:32632"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_cloud(item):
    value = item.properties.get("eo:cloud_cover", 100.0)

    try:
        return float(value)
    except Exception:
        return 100.0


def evaluate_item(
    item,
    aoi_metric,
    fire_date,
    to_metric,
):
    if item.datetime is None:
        return None

    if item.geometry is None:
        return None

    geom_wgs84 = shape(item.geometry)

    geom_metric = transform(
        to_metric.transform,
        geom_wgs84,
    )

    intersection = geom_metric.intersection(
        aoi_metric
    )

    coverage = (
        intersection.area
        / aoi_metric.area
    )

    coverage = min(
        max(coverage, 0.0),
        1.0,
    )

    acquisition_date = pd.Timestamp(
        item.datetime.date()
    )

    days = (
        acquisition_date
        - fire_date.normalize()
    ).days

    cloud = get_cloud(item)

    score = (
        100.0 * coverage
        - 0.50 * cloud
        - 0.20 * abs(days)
    )

    return {
        "item_id": item.id,
        "date": acquisition_date,
        "days": days,
        "coverage": coverage,
        "cloud": cloud,
        "score": score,
    }


def choose_best(records, phase):

    if phase == "pre":

        subset = [
            r for r in records
            if -PRE_MAX <= r["days"] <= -PRE_MIN
        ]

    elif phase == "post":

        subset = [
            r for r in records
            if POST_MIN <= r["days"] <= POST_MAX
        ]

    else:
        raise ValueError(phase)

    if not subset:
        return None

    subset.sort(
        key=lambda r: (
            r["coverage"] >= MIN_COVERAGE,
            r["score"],
        ),
        reverse=True,
    )

    return subset[0]


# ------------------------------------------------------------
# Load candidates
# ------------------------------------------------------------

fires = gpd.read_file(
    CATALOG
).to_crs(
    METRIC_CRS
)

fires["fire_date"] = pd.to_datetime(
    fires["fire_date"],
    errors="coerce",
)

fires = fires[
    fires["fire_date"].notna()
].copy()

print("Candidate fires:", len(fires))


# ------------------------------------------------------------
# Coordinate transformations
# ------------------------------------------------------------

to_wgs84 = Transformer.from_crs(
    METRIC_CRS,
    "EPSG:4326",
    always_xy=True,
)

to_metric = Transformer.from_crs(
    "EPSG:4326",
    METRIC_CRS,
    always_xy=True,
)


# ------------------------------------------------------------
# STAC
# ------------------------------------------------------------

print("Opening STAC...")
catalog = Client.open(STAC_URL)

rows = []


for n, (_, fire) in enumerate(
    fires.iterrows(),
    start=1,
):

    event_id = fire["event_id"]
    fire_date = fire["fire_date"]

    print()
    print(
        f"[{n:03d}/{len(fires):03d}] "
        f"{event_id} | "
        f"{fire_date.date()} | "
        f"{fire['area_geom_ha']:.1f} ha"
    )

    # Include surrounding unburned landscape.
    aoi_metric = fire.geometry.buffer(
        BUFFER_M
    )

    aoi_wgs84 = transform(
        to_wgs84.transform,
        aoi_metric,
    )

    start = (
        fire_date
        - timedelta(days=PRE_MAX)
    )

    end = (
        fire_date
        + timedelta(days=POST_MAX)
    )

    interval = (
        f"{start.date()}T00:00:00Z/"
        f"{end.date()}T23:59:59Z"
    )

    try:

        search = catalog.search(
            collections=[COLLECTION],
            intersects=mapping(aoi_wgs84),
            datetime=interval,
            query={
                "eo:cloud_cover": {
                    "lt": MAX_TILE_CLOUD
                }
            },
            max_items=200,
        )

        items = list(search.items())

    except Exception as exc:

        print(" STAC ERROR:", exc)

        rows.append(
            {
                "event_id": event_id,
                "status": "stac_error",
                "error": str(exc),
            }
        )

        continue

    print(" items:", len(items))

    records = []

    for item in items:

        result = evaluate_item(
            item,
            aoi_metric,
            fire_date,
            to_metric,
        )

        if result is not None:
            records.append(result)

    pre = choose_best(
        records,
        "pre",
    )

    post = choose_best(
        records,
        "post",
    )

    if pre is not None:

        print(
            f" PRE : {pre['date'].date()} "
            f"({pre['days']:+d} d) | "
            f"coverage={100*pre['coverage']:.1f}% | "
            f"cloud={pre['cloud']:.1f}%"
        )

    else:
        print(" PRE : NONE")

    if post is not None:

        print(
            f" POST: {post['date'].date()} "
            f"({post['days']:+d} d) | "
            f"coverage={100*post['coverage']:.1f}% | "
            f"cloud={post['cloud']:.1f}%"
        )

    else:
        print(" POST: NONE")

    pair_ok = (
        pre is not None
        and post is not None
        and pre["coverage"] >= MIN_COVERAGE
        and post["coverage"] >= MIN_COVERAGE
    )

    row = {
        "event_id": event_id,
        "year": int(fire["year"]),
        "fire_date": fire_date.date(),
        "municipality": fire["municipality"],
        "toponym": fire["toponym"],
        "area_geom_ha": fire["area_geom_ha"],
        "longitude": fire["longitude"],
        "latitude": fire["latitude"],
        "status": (
            "candidate_pair"
            if pair_ok
            else "insufficient_pair"
        ),
        "items_found": len(items),
    }

    if pre is not None:

        row.update(
            {
                "pre_item_id": pre["item_id"],
                "pre_date": pre["date"].date(),
                "pre_days": pre["days"],
                "pre_coverage": pre["coverage"],
                "pre_scene_cloud": pre["cloud"],
                "pre_score": pre["score"],
            }
        )

    if post is not None:

        row.update(
            {
                "post_item_id": post["item_id"],
                "post_date": post["date"].date(),
                "post_days": post["days"],
                "post_coverage": post["coverage"],
                "post_scene_cloud": post["cloud"],
                "post_score": post["score"],
            }
        )

    if pair_ok:

        row["pair_score"] = (
            pre["score"]
            + post["score"]
        ) / 2.0

    else:

        row["pair_score"] = np.nan

    rows.append(row)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results = pd.DataFrame(rows)

results.to_csv(
    OUT_ALL,
    index=False,
)

good = results[
    results["status"] == "candidate_pair"
].copy()

if len(good):

    good = good.sort_values(
        [
            "pair_score",
            "area_geom_ha",
        ],
        ascending=[
            False,
            False,
        ],
    )

good.to_csv(
    OUT_GOOD,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("==========================================")
print("SENTINEL-2 PRE/POST SEARCH SUMMARY")
print("==========================================")

print(
    "Events searched:",
    len(results),
)

print(
    "Usable metadata pairs:",
    len(good),
)

if len(good):

    print()
    print("Pairs by year:")

    print(
        good.groupby("year").size()
    )

    print()
    print("Top 30 preliminary pairs:")

    cols = [
        "event_id",
        "fire_date",
        "municipality",
        "area_geom_ha",
        "pre_date",
        "pre_days",
        "pre_scene_cloud",
        "post_date",
        "post_days",
        "post_scene_cloud",
        "pair_score",
    ]

    print(
        good[cols]
        .head(30)
        .to_string(index=False)
    )

print()
print("Saved:")
print(OUT_ALL)
print(OUT_GOOD)

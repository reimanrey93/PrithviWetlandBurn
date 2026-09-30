#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT


# ============================================================
# CONFIGURATION
# ============================================================

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

SELECTED_GPKG = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog/"
    "sardinia_selected_28_diverse.gpkg"
)

OUT_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/events"
)

TARGET_CRS = "EPSG:32632"
RESOLUTION = 20.0
BUFFER_M = 2000.0

REFLECTANCE_NODATA = -9999.0
DNBR_NODATA = -9999.0
MASK_NODATA = 255

BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]

# Earth Search asset names.
ASSET_CANDIDATES = {
    "BLUE": ["blue", "B02"],
    "GREEN": ["green", "B03"],
    "RED": ["red", "B04"],
    "NIR_NARROW": ["nir08", "B8A"],
    "SWIR_1": ["swir16", "B11"],
    "SWIR_2": ["swir22", "B12"],
}

SCL_CANDIDATES = [
    "scl",
    "SCL",
]

# Sen2Cor classes that we do not want to train on.
BAD_SCL = {
    0,   # No data
    1,   # Saturated / defective
    3,   # Cloud shadow
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
    11,  # Snow / ice
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="Process only one event_id.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N selected events.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# STAC HELPERS
# ============================================================

def get_item(client, item_id):

    search = client.search(
        collections=[COLLECTION],
        ids=[item_id],
        max_items=1,
    )

    items = list(
        search.items()
    )

    if not items:
        raise RuntimeError(
            f"STAC item not found: {item_id}"
        )

    return items[0]


def find_asset(
    item,
    candidates,
):

    for key in candidates:
        if key in item.assets:
            return key, item.assets[key]

    lower_lookup = {
        key.lower(): key
        for key in item.assets
    }

    for candidate in candidates:

        key = lower_lookup.get(
            candidate.lower()
        )

        if key is not None:
            return key, item.assets[key]

    raise KeyError(
        f"Could not find asset {candidates} "
        f"in item {item.id}. "
        f"Available assets: "
        f"{list(item.assets.keys())}"
    )


def get_scale_offset(asset):
    """
    Read scale/offset from STAC raster extension.

    Sentinel-2 processing baseline changes make this preferable
    to assuming DN / 10000 for all dates.
    """

    bands = asset.extra_fields.get(
        "raster:bands",
        []
    )

    if not bands:
        raise RuntimeError(
            "Asset has no raster:bands metadata. "
            f"Cannot safely determine scale/offset: "
            f"{asset.href}"
        )

    meta = bands[0]

    scale = meta.get(
        "scale",
        None,
    )

    offset = meta.get(
        "offset",
        0.0,
    )

    if scale is None:
        raise RuntimeError(
            "Asset has raster:bands but no scale. "
            f"Cannot safely normalize reflectance: "
            f"{asset.href}"
        )

    return (
        float(scale),
        float(offset),
    )


# ============================================================
# GRID
# ============================================================

def build_grid(aoi_geometry):

    minx, miny, maxx, maxy = (
        aoi_geometry.bounds
    )

    minx = (
        math.floor(
            minx / RESOLUTION
        )
        * RESOLUTION
    )

    miny = (
        math.floor(
            miny / RESOLUTION
        )
        * RESOLUTION
    )

    maxx = (
        math.ceil(
            maxx / RESOLUTION
        )
        * RESOLUTION
    )

    maxy = (
        math.ceil(
            maxy / RESOLUTION
        )
        * RESOLUTION
    )

    width = int(
        round(
            (maxx - minx)
            / RESOLUTION
        )
    )

    height = int(
        round(
            (maxy - miny)
            / RESOLUTION
        )
    )

    transform = from_origin(
        minx,
        maxy,
        RESOLUTION,
        RESOLUTION,
    )

    return (
        transform,
        width,
        height,
    )


# ============================================================
# REMOTE RASTER READERS
# ============================================================

def read_reflectance(
    asset,
    dst_transform,
    dst_width,
    dst_height,
):

    scale, offset = (
        get_scale_offset(asset)
    )

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    ):

        with rasterio.open(
            asset.href
        ) as src:

            with WarpedVRT(
                src,
                crs=TARGET_CRS,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
            ) as vrt:

                raw = vrt.read(
                    1,
                    masked=True,
                )

    mask = np.ma.getmaskarray(
        raw
    )

    values = np.asarray(
        raw.filled(0),
        dtype=np.float32,
    )

    values = (
        values * scale
        + offset
    )

    # Match the existing Prithvi pipeline.
    values = np.clip(
        values,
        0.0,
        1.0,
    )

    valid = ~mask

    values[
        ~valid
    ] = REFLECTANCE_NODATA

    return (
        values,
        valid,
        scale,
        offset,
    )


def read_scl(
    asset,
    dst_transform,
    dst_width,
    dst_height,
):

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    ):

        with rasterio.open(
            asset.href
        ) as src:

            with WarpedVRT(
                src,
                crs=TARGET_CRS,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
            ) as vrt:

                arr = vrt.read(
                    1,
                    masked=True,
                )

    mask = np.ma.getmaskarray(
        arr
    )

    values = np.asarray(
        arr.filled(0),
        dtype=np.uint8,
    )

    valid = ~mask

    return (
        values,
        valid,
    )


# ============================================================
# WRITE HELPERS
# ============================================================

def write_stack(
    path,
    stack,
    transform,
):

    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": REFLECTANCE_NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            stack.astype(
                np.float32
            )
        )

        for i, name in enumerate(
            BAND_NAMES,
            start=1,
        ):
            dst.set_band_description(
                i,
                name,
            )


def write_uint8(
    path,
    array,
    transform,
    nodata=None,
    description=None,
):

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            array.astype(
                np.uint8
            ),
            1,
        )

        if description:
            dst.set_band_description(
                1,
                description,
            )


def write_float(
    path,
    array,
    transform,
    description,
):

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": DNBR_NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            array.astype(
                np.float32
            ),
            1,
        )

        dst.set_band_description(
            1,
            description,
        )


# ============================================================
# PHASE READER
# ============================================================

def read_phase(
    item,
    transform,
    width,
    height,
):

    channels = []
    valid_all = np.ones(
        (height, width),
        dtype=bool,
    )

    asset_metadata = {}

    for band_name in BAND_NAMES:

        key, asset = find_asset(
            item,
            ASSET_CANDIDATES[
                band_name
            ],
        )

        print(
            f"      {band_name:11s} "
            f"<- {key}"
        )

        (
            values,
            valid,
            scale,
            offset,
        ) = read_reflectance(
            asset,
            transform,
            width,
            height,
        )

        channels.append(
            values
        )

        valid_all &= valid

        asset_metadata[
            band_name
        ] = {
            "asset_key": key,
            "href": asset.href,
            "scale": scale,
            "offset": offset,
        }

    scl_key, scl_asset = find_asset(
        item,
        SCL_CANDIDATES,
    )

    print(
        f"      {'SCL':11s} "
        f"<- {scl_key}"
    )

    scl, scl_valid = read_scl(
        scl_asset,
        transform,
        width,
        height,
    )

    valid_all &= scl_valid

    stack = np.stack(
        channels,
        axis=0,
    )

    return (
        stack,
        scl,
        valid_all,
        asset_metadata,
        {
            "asset_key": scl_key,
            "href": scl_asset.href,
        },
    )


# ============================================================
# EVENT
# ============================================================

def process_event(
    row,
    client,
    overwrite=False,
):

    event_id = row["event_id"]

    event_dir = (
        OUT_ROOT
        / event_id
    )

    event_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    done_file = (
        event_dir
        / "_SUCCESS"
    )

    if (
        done_file.exists()
        and not overwrite
    ):
        print(
            f"SKIP {event_id}: already complete"
        )
        return

    print()
    print(
        "=========================================="
    )
    print(
        event_id
    )
    print(
        "=========================================="
    )

    fire_geom = row.geometry

    aoi_geom = fire_geom.buffer(
        BUFFER_M
    )

    (
        transform,
        width,
        height,
    ) = build_grid(
        aoi_geom
    )

    print(
        f"Grid: {width} x {height} "
        f"@ {RESOLUTION:.0f} m"
    )

    print(
        f"Approx area raster: "
        f"{width * RESOLUTION / 1000:.1f} x "
        f"{height * RESOLUTION / 1000:.1f} km"
    )

    # --------------------------------------------------------
    # STAC ITEMS
    # --------------------------------------------------------

    pre_item_id = str(
        row["pre_item_id"]
    )

    post_item_id = str(
        row["post_item_id"]
    )

    print(
        "PRE item:",
        pre_item_id,
    )

    pre_item = get_item(
        client,
        pre_item_id,
    )

    print(
        "POST item:",
        post_item_id,
    )

    post_item = get_item(
        client,
        post_item_id,
    )

    # --------------------------------------------------------
    # PRE
    # --------------------------------------------------------

    print("  Reading PRE...")

    (
        pre,
        scl_pre,
        valid_pre,
        pre_assets,
        scl_pre_asset,
    ) = read_phase(
        pre_item,
        transform,
        width,
        height,
    )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    print("  Reading POST...")

    (
        post,
        scl_post,
        valid_post,
        post_assets,
        scl_post_asset,
    ) = read_phase(
        post_item,
        transform,
        width,
        height,
    )

    # --------------------------------------------------------
    # AOI + FIRE MASK
    # --------------------------------------------------------

    aoi_mask = rasterize(
        [
            (
                aoi_geom,
                1,
            )
        ],
        out_shape=(
            height,
            width,
        ),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    burn = rasterize(
        [
            (
                fire_geom,
                1,
            )
        ],
        out_shape=(
            height,
            width,
        ),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # CLOUD / INVALID
    # --------------------------------------------------------

    bad_pre = np.isin(
        scl_pre,
        list(BAD_SCL),
    )

    bad_post = np.isin(
        scl_post,
        list(BAD_SCL),
    )

    valid = (
        aoi_mask
        & valid_pre
        & valid_post
        & ~bad_pre
        & ~bad_post
    )

    label = burn.astype(
        np.uint8
    )

    label[
        ~valid
    ] = MASK_NODATA

    # --------------------------------------------------------
    # dNBR
    # --------------------------------------------------------

    nir_pre = pre[3]
    swir2_pre = pre[5]

    nir_post = post[3]
    swir2_post = post[5]

    eps = 1e-6

    nbr_pre = (
        nir_pre - swir2_pre
    ) / (
        nir_pre + swir2_pre
        + eps
    )

    nbr_post = (
        nir_post - swir2_post
    ) / (
        nir_post + swir2_post
        + eps
    )

    dnbr = (
        nbr_pre
        - nbr_post
    ).astype(
        np.float32
    )

    dnbr[
        ~valid
    ] = DNBR_NODATA

    # --------------------------------------------------------
    # Force invalid image pixels to NODATA
    # --------------------------------------------------------

    pre[
        :,
        ~valid_pre
    ] = REFLECTANCE_NODATA

    post[
        :,
        ~valid_post
    ] = REFLECTANCE_NODATA

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    write_stack(
        event_dir / "pre.tif",
        pre,
        transform,
    )

    write_stack(
        event_dir / "post.tif",
        post,
        transform,
    )

    write_uint8(
        event_dir / "scl_pre.tif",
        scl_pre,
        transform,
        nodata=0,
        description="SCL_PRE",
    )

    write_uint8(
        event_dir / "scl_post.tif",
        scl_post,
        transform,
        nodata=0,
        description="SCL_POST",
    )

    write_uint8(
        event_dir / "mask.tif",
        label,
        transform,
        nodata=MASK_NODATA,
        description="BURN_MASK",
    )

    write_float(
        event_dir / "dnbr.tif",
        dnbr,
        transform,
        "dNBR",
    )

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    valid_pixels = int(
        valid.sum()
    )

    burned_pixels = int(
        (
            (label == 1)
            & valid
        ).sum()
    )

    valid_area_ha = (
        valid_pixels
        * RESOLUTION
        * RESOLUTION
        / 10000.0
    )

    burned_area_raster_ha = (
        burned_pixels
        * RESOLUTION
        * RESOLUTION
        / 10000.0
    )

    valid_dnbr = (
        dnbr[
            dnbr != DNBR_NODATA
        ]
    )

    metadata = {
        "event_id": event_id,
        "year": int(row["year"]),
        "municipality": str(
            row["municipality"]
        ),
        "fire_date": str(
            row["fire_date"]
        ),
        "official_geometry_area_ha": float(
            row["area_geom_ha"]
        ),
        "pre_item_id": pre_item_id,
        "post_item_id": post_item_id,
        "pre_date": str(
            row["pre_date"]
        ),
        "post_date": str(
            row["post_date"]
        ),
        "crs": TARGET_CRS,
        "resolution_m": RESOLUTION,
        "buffer_m": BUFFER_M,
        "width": width,
        "height": height,
        "bands": BAND_NAMES,
        "valid_pixels": valid_pixels,
        "valid_area_ha": valid_area_ha,
        "burned_pixels": burned_pixels,
        "burned_area_raster_ha": (
            burned_area_raster_ha
        ),
        "pre_assets": pre_assets,
        "post_assets": post_assets,
        "scl_pre_asset": scl_pre_asset,
        "scl_post_asset": scl_post_asset,
    }

    if valid_dnbr.size:
        metadata[
            "dnbr_min"
        ] = float(
            np.nanmin(
                valid_dnbr
            )
        )

        metadata[
            "dnbr_mean"
        ] = float(
            np.nanmean(
                valid_dnbr
            )
        )

        metadata[
            "dnbr_max"
        ] = float(
            np.nanmax(
                valid_dnbr
            )
        )

    with open(
        event_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    done_file.write_text(
        "OK\n",
        encoding="utf-8",
    )

    print()
    print(
        f"  Valid area: "
        f"{valid_area_ha:.1f} ha"
    )

    print(
        f"  Burned area raster: "
        f"{burned_area_raster_ha:.1f} ha"
    )

    print(
        f"  Official geometry: "
        f"{float(row['area_geom_ha']):.1f} ha"
    )

    print(
        f"  Output: {event_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = gpd.read_file(
        SELECTED_GPKG
    ).to_crs(
        TARGET_CRS
    )

    if args.event:

        selected = selected[
            selected[
                "event_id"
            ]
            == args.event
        ].copy()

        if len(selected) == 0:
            raise RuntimeError(
                f"Event not found: "
                f"{args.event}"
            )

    if args.limit is not None:

        selected = (
            selected
            .head(
                args.limit
            )
            .copy()
        )

    print(
        "Events to process:",
        len(selected)
    )

    print(
        "Opening STAC..."
    )

    client = Client.open(
        STAC_URL
    )

    failures = []

    for _, row in selected.iterrows():

        try:

            process_event(
                row,
                client,
                overwrite=args.overwrite,
            )

        except Exception as exc:

            event_id = row[
                "event_id"
            ]

            print()
            print(
                f"ERROR {event_id}: "
                f"{exc}"
            )

            failures.append(
                (
                    event_id,
                    str(exc),
                )
            )

    print()
    print(
        "=========================================="
    )
    print(
        "BUILD SUMMARY"
    )
    print(
        "=========================================="
    )

    print(
        "Requested:",
        len(selected)
    )

    print(
        "Failures:",
        len(failures)
    )

    if failures:

        for event_id, error in failures:
            print(
                event_id,
                "->",
                error,
            )


if __name__ == "__main__":
    main()

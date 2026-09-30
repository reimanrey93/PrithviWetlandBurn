#!/usr/bin/env python3
"""
Stable Earth Engine downloader for wetland-fire candidate chips.

Produces:
  pre:   Sentinel-2 composite (B2,B3,B4,B8,B11,B12)
  post:  Sentinel-2 composite (B2,B3,B4,B8,B11,B12)
  aux:   NBR_pre, NBR_post, dNBR, rNBR, NDVI_pre, NDVI_post,
         NDMI_pre, NDMI_post, MNDWI_pre, MNDWI_post, open_water, valid
  label: 0=no-burn candidate, 1=burn candidate, 255=ignore/ambiguous/water/nodata

This file deliberately does NOT run connected-component or morphology in Earth
Engine. That spatial cleanup is safer and reproducible locally with
clean_seed_labels_local.py after downloading.
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import ee
import requests

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

SCALE = 10
CRS = "EPSG:32721"
HALF_SIDE_M = 2560

PRE_START_DAYS = 75
PRE_END_DAYS = 15
POST_START_DAYS = 5
POST_END_DAYS = 50

# Conservative spectral candidates for wetlands.
DNBR_BURN_MIN = 0.25
RNBR_BURN_MIN = 0.35
NDVI_PRE_BURN_MIN = 0.35

DNBR_NONBURN_MAX = 0.03
RNBR_NONBURN_MAX = 0.05

MNDWI_WATER_MIN = 0.15
NDVI_WATER_MAX = 0.20

RNBR_DENOM_MIN = 0.10
RNBR_CLIP_MIN = -1.0
RNBR_CLIP_MAX = 2.0


def mask_s2(image):
    scl = image.select("SCL")
    clear = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return (
        image.updateMask(clear)
        .select(BANDS)
        .multiply(0.0001)
        .copyProperties(image, ["system:time_start"])
    )


def composite(roi, start_date, end_date):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_s2)
    )
    n_images = collection.size().getInfo()
    if n_images == 0:
        raise RuntimeError(
            f"No hay imágenes Sentinel-2 para la ventana {start_date} a {end_date}"
        )
    return collection.median().clip(roi), n_images


def get_indices(image):
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return nbr, ndvi, ndmi, mndwi


def download_geotiff(image, roi, file_name, output_path):
    params = {
        "name": file_name,
        "region": roi,
        "scale": SCALE,
        "crs": CRS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    url = image.getDownloadURL(params)
    response = requests.get(url, timeout=600)

    if not response.ok:
        body = response.text[:4000]
        raise RuntimeError(
            f"Earth Engine HTTP {response.status_code} al descargar {file_name}.\n{body}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Earth Engine may provide a ZIP or a direct TIFF.
    if response.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            tif_names = [
                name for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]
            if not tif_names:
                raise RuntimeError(
                    f"No se encontró un TIFF dentro del ZIP: {archive.namelist()}"
                )
            with archive.open(tif_names[0]) as source, open(output_path, "wb") as destination:
                destination.write(source.read())
        return

    if response.content[:4] in (b"II*\x00", b"MM\x00*"):
        with open(output_path, "wb") as destination:
            destination.write(response.content)
        return

    preview = response.content[:500].decode("utf-8", errors="replace")
    raise RuntimeError(
        f"Formato de respuesta inesperado para {file_name}. Inicio: {preview}"
    )


def build_products(pre, post):
    nbr_pre, ndvi_pre, ndmi_pre, mndwi_pre = get_indices(pre)
    nbr_post, ndvi_post, ndmi_post, mndwi_post = get_indices(post)

    dnbr = nbr_pre.subtract(nbr_post).rename("dNBR")
    denominator = nbr_pre.abs().sqrt().max(ee.Image.constant(RNBR_DENOM_MIN))
    rnbr = (
        dnbr.divide(denominator)
        .clamp(RNBR_CLIP_MIN, RNBR_CLIP_MAX)
        .rename("rNBR")
    )

    valid = (
        pre.mask().reduce(ee.Reducer.min())
        .And(post.mask().reduce(ee.Reducer.min()))
        .rename("valid")
    )

    open_water_pre = (
        mndwi_pre.gte(MNDWI_WATER_MIN)
        .And(ndvi_pre.lte(NDVI_WATER_MAX))
    )
    open_water_post = (
        mndwi_post.gte(MNDWI_WATER_MIN)
        .And(ndvi_post.lte(NDVI_WATER_MAX))
    )
    open_water = open_water_pre.Or(open_water_post).rename("open_water")

    burn = (
        dnbr.gte(DNBR_BURN_MIN)
        .And(rnbr.gte(RNBR_BURN_MIN))
        .And(ndvi_pre.gte(NDVI_PRE_BURN_MIN))
        .And(open_water.Not())
    )

    no_burn = (
        dnbr.lte(DNBR_NONBURN_MAX)
        .And(rnbr.abs().lte(RNBR_NONBURN_MAX))
        .And(open_water.Not())
    )

    seed = (
        ee.Image(255)
        .where(no_burn, 0)
        .where(burn, 1)
        .where(open_water, 255)
        .updateMask(valid)
        .unmask(255)
        .rename("label_seed")
        .toUint8()
    )

    auxiliary = (
        nbr_pre.rename("NBR_pre")
        .addBands(nbr_post.rename("NBR_post"))
        .addBands(dnbr)
        .addBands(rnbr)
        .addBands(ndvi_pre.rename("NDVI_pre"))
        .addBands(ndvi_post.rename("NDVI_post"))
        .addBands(ndmi_pre.rename("NDMI_pre"))
        .addBands(ndmi_post.rename("NDMI_post"))
        .addBands(mndwi_pre.rename("MNDWI_pre"))
        .addBands(mndwi_post.rename("MNDWI_post"))
        .addBands(open_water.toUint8().rename("open_water"))
        .addBands(valid.toUint8().rename("valid"))
    )

    return seed, auxiliary


def process_event(event, root, overwrite):
    event_id = event["event_id"]
    event_date = datetime.fromisoformat(event["event_date"])

    point = ee.Geometry.Point([float(event["lon"]), float(event["lat"])])
    roi = point.transform(CRS, 1).buffer(HALF_SIDE_M, 1).bounds(1, CRS)

    pre_start = (event_date - timedelta(days=PRE_START_DAYS)).date().isoformat()
    pre_end = (event_date - timedelta(days=PRE_END_DAYS)).date().isoformat()
    post_start = (event_date + timedelta(days=POST_START_DAYS)).date().isoformat()
    post_end = (event_date + timedelta(days=POST_END_DAYS)).date().isoformat()

    pre, n_pre = composite(roi, pre_start, pre_end)
    post, n_post = composite(roi, post_start, post_end)
    seed, auxiliary = build_products(pre, post)

    targets = [
        (pre.select(BANDS), root / "data/raw/pre" / f"{event_id}_pre.tif"),
        (post.select(BANDS), root / "data/raw/post" / f"{event_id}_post.tif"),
        (auxiliary, root / "data/raw/aux" / f"{event_id}_aux.tif"),
        (seed, root / "data/raw/labels" / f"{event_id}_label_seed.tif"),
    ]

    for image, path in targets:
        if overwrite or not path.exists():
            print(f"    descargando: {path.name}", flush=True)
            download_geotiff(image, roi, path.stem, path)

    return {
        "event_id": event_id,
        "event_date": event["event_date"],
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
        "n_pre": n_pre,
        "n_post": n_post,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--events", default="events/events_santafe_dedup.csv")
    parser.add_argument("--root", default="/DATA/ralfonso/prithvi_wetlandburn")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10, help="0 = todos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    root = Path(args.root).resolve()

    with open(args.events, encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))

    selected = events[args.start:] if args.limit == 0 else events[
        args.start:args.start + args.limit
    ]

    batch_manifest = []
    for index, event in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {event['event_id']} {event['event_date']}", flush=True)
        try:
            record = process_event(event, root, args.overwrite)
            print(f"  OK {record['n_pre']} {record['n_post']}", flush=True)
        except Exception as error:
            record = {
                "event_id": event["event_id"],
                "event_date": event["event_date"],
                "status": "error",
                "error": str(error),
            }
            print(f"  ERROR {error}", file=sys.stderr, flush=True)
        batch_manifest.append(record)

    manifest_path = root / "data/manifests/download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    previous = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = []

    manifest_path.write_text(
        json.dumps(previous + batch_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Descarga chips Sentinel-2 pre/post y derivados para cicatrices de incendio en humedales.

Cambios principales respecto a la versión inicial:
  - rNBR con denominador protegido y limitado para evitar valores extremos.
  - etiqueta semilla más conservadora:
      quemado = dNBR alto + rNBR alto + vegetación pre-evento suficiente
      no quemado = cambios espectrales muy bajos
      incierto = resto
  - cuerpos de agua abiertos se excluyen de la etiqueta (valor 255).
  - responde correctamente si Earth Engine devuelve ZIP o GeoTIFF directo.

Etiqueta:
  0 = no quemado candidato
  1 = quemado candidato
  255 = incierto, agua abierta o no-data
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
CRS = "EPSG:32721"       # UTM 21S
HALF_SIDE_M = 2560       # chip ~5.12 km x 5.12 km

PRE_START_DAYS = 75
PRE_END_DAYS = 15
POST_START_DAYS = 5
POST_END_DAYS = 50

# ------------------------------------------------------------
# UMBRALES INICIALES PARA HUMEDALES
# ------------------------------------------------------------
# La etiqueta se genera de manera conservadora: las áreas dudosas
# quedan en 255 y no deben usarse como ground truth automático.
DNBR_BURN_MIN = 0.15
RNBR_BURN_MIN = 0.20
NDVI_PRE_BURN_MIN = 0.25

DNBR_NONBURN_MAX = 0.05
RNBR_NONBURN_MAX = 0.08

# Agua abierta / inundación dominante: se excluye del label.
# Se usa MNDWI alto junto a NDVI bajo para evitar borrar
# vegetación de humedal solamente húmeda.
MNDWI_WATER_MIN = 0.15
NDVI_WATER_MAX = 0.20

RNBR_DENOM_MIN = 0.10
RNBR_CLIP_MIN = -1.0
RNBR_CLIP_MAX = 2.0


def mask_s2(img):
    """Enmascara nubes, sombras, cirrus, nieve y no-data Sentinel-2."""
    scl = img.select("SCL")
    good = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return (
        img.updateMask(good)
        .select(BANDS)
        .multiply(0.0001)
        .copyProperties(img, ["system:time_start"])
    )


def composite(roi, start, end):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_s2)
    )

    n = collection.size().getInfo()
    if n == 0:
        raise RuntimeError(f"Sin Sentinel-2 para {start} a {end}")

    return collection.median().clip(roi), n


def indices(img):
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return nbr, ndvi, ndmi, mndwi


def download_geotiff(image, roi, name, out_tif):
    """Descarga ZIP o GeoTIFF directo desde Earth Engine."""
    params = {
        "name": name,
        "region": roi,
        "scale": SCALE,
        "crs": CRS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    url = image.getDownloadURL(params)
    response = requests.get(url, timeout=600)
    response.raise_for_status()

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # ZIP clásico.
    if response.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            tifs = [
                n for n in zf.namelist()
                if n.lower().endswith((".tif", ".tiff"))
            ]
            if not tifs:
                raise RuntimeError(
                    f"No se encontró GeoTIFF dentro del ZIP: {zf.namelist()}"
                )
            with zf.open(tifs[0]) as src, open(out_tif, "wb") as dst:
                dst.write(src.read())

    # GeoTIFF directo.
    elif response.content[:4] in (b"II*\x00", b"MM\x00*"):
        with open(out_tif, "wb") as dst:
            dst.write(response.content)

    else:
        preview = response.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            "Earth Engine devolvió un formato inesperado. "
            f"Inicio de respuesta: {preview}"
        )


def process_event(row, root, overwrite=False):
    event_id = row["event_id"]
    event_date = datetime.fromisoformat(row["event_date"])

    point = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
    roi = point.transform(CRS, 1).buffer(HALF_SIDE_M, 1).bounds(1, CRS)

    pre_start = (event_date - timedelta(days=PRE_START_DAYS)).date().isoformat()
    pre_end = (event_date - timedelta(days=PRE_END_DAYS)).date().isoformat()
    post_start = (event_date + timedelta(days=POST_START_DAYS)).date().isoformat()
    post_end = (event_date + timedelta(days=POST_END_DAYS)).date().isoformat()

    pre, n_pre = composite(roi, pre_start, pre_end)
    post, n_post = composite(roi, post_start, post_end)

    nbr_pre, ndvi_pre, ndmi_pre, mndwi_pre = indices(pre)
    nbr_post, ndvi_post, ndmi_post, mndwi_post = indices(post)

    dnbr = nbr_pre.subtract(nbr_post).rename("dNBR")

    # rNBR robusto: evita divisor cercano a cero y limita extremos
    # que son frecuentes en ambientes con agua/sedimentos/suelo expuesto.
    rnbr_denom = nbr_pre.abs().sqrt().max(ee.Image.constant(RNBR_DENOM_MIN))
    rnbr = (
        dnbr.divide(rnbr_denom)
        .clamp(RNBR_CLIP_MIN, RNBR_CLIP_MAX)
        .rename("rNBR")
    )

    # Píxeles válidos en ambos compuestos.
    valid = (
        pre.mask().reduce(ee.Reducer.min())
        .And(post.mask().reduce(ee.Reducer.min()))
        .rename("valid")
    )

    # Agua abierta: excluimos ríos, lagunas y sectores claramente inundados.
    # Se evalúa en pre O post para evitar confundir cambios hidrológicos con fuego.
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

    nonburn = (
        dnbr.lte(DNBR_NONBURN_MAX)
        .And(rnbr.lte(RNBR_NONBURN_MAX))
        .And(open_water.Not())
    )

    # 0/1 solo donde la evidencia es clara. El resto queda en 255.
    seed = (
        ee.Image(255)
        .where(nonburn, 0)
        .where(burn, 1)
        .where(open_water, 255)
        .updateMask(valid)
        .unmask(255)
        .rename("label_seed")
        .toUint8()
    )

    aux = (
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
        .addBands(open_water.toUint8())
        .addBands(valid.toUint8())
        .clip(roi)
    )

    outputs = [
        (pre.select(BANDS), root / "data/raw/pre" / f"{event_id}_pre.tif"),
        (post.select(BANDS), root / "data/raw/post" / f"{event_id}_post.tif"),
        (aux, root / "data/raw/aux" / f"{event_id}_aux.tif"),
        (seed, root / "data/raw/labels" / f"{event_id}_label_seed.tif"),
    ]

    for image, path in outputs:
        if overwrite or not path.exists():
            download_geotiff(image, roi, path.stem, path)

    return {
        "event_id": event_id,
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
    parser.add_argument("--events", default="events/events_santafe_dedup.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project registrado para Earth Engine",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Cantidad de eventos; use 0 para todos",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ee.Initialize(project=args.project)

    with open(args.events, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.limit == 0:
        selected = rows[args.start:]
    else:
        selected = rows[args.start: args.start + args.limit]

    manifest = []

    for i, row in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {row['event_id']} {row['event_date']}")
        try:
            record = process_event(row, root, args.overwrite)
            print("  OK", record["n_pre"], record["n_post"])
        except Exception as exc:
            record = {
                "event_id": row["event_id"],
                "status": "error",
                "error": str(exc),
            }
            print("  ERROR", exc, file=sys.stderr)

        manifest.append(record)

    manifest_path = root / "data/manifests/download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            pass

    manifest_path.write_text(json.dumps(existing + manifest, indent=2))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

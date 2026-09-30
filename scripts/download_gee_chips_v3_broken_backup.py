#!/usr/bin/env python3
"""
download_gee_chips_wetland_v3.py

Descarga chips Sentinel-2 pre/post y crea una máscara semilla CONSERVADORA
para cicatrices potenciales de incendio en humedales.

Etiqueta:
  0   = no quemado candidato
  1   = quemado candidato espacialmente coherente y cercano al foco
  255 = ignorar / incierto / agua / no-data

Mejoras frente a v2:
  - umbrales más estrictos
  - rNBR con denominador protegido
  - exclusión de agua abierta
  - apertura/cierre morfológico
  - eliminación de componentes pequeños (< 1 ha)
  - solo se acepta quemado dentro de un buffer alrededor del foco FIRMS
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

# ============================================================
# CONFIGURACIÓN ESPACIAL
# ============================================================

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

SCALE = 10
CRS = "EPSG:32721"       # UTM 21S
HALF_SIDE_M = 2560       # chip ~5.12 km × 5.12 km

PRE_START_DAYS = 75
PRE_END_DAYS = 15
POST_START_DAYS = 5
POST_END_DAYS = 50

# ============================================================
# UMBRALES CONSERVADORES
# ============================================================

DNBR_BURN_MIN = 0.25
RNBR_BURN_MIN = 0.35
NDVI_PRE_BURN_MIN = 0.35

DNBR_NONBURN_MAX = 0.03
RNBR_NONBURN_MAX = 0.05

# agua abierta / inundación fuerte
MNDWI_WATER_MIN = 0.15
NDVI_WATER_MAX = 0.20

# estabilidad numérica de rNBR
RNBR_DENOM_MIN = 0.10
RNBR_CLIP_MIN = -1.0
RNBR_CLIP_MAX = 2.0

# ============================================================
# FILTRO ESPACIAL DE CANDIDATOS
# ============================================================

# Radio de búsqueda alrededor del hotspot: no etiquetamos como quemado
# cualquier cambio lejano dentro del chip.
FOCUS_BUFFER_M = 2200

# limpieza morfológica a 10 m:
# 20 m = dos píxeles de radio aproximadamente.
MORPH_RADIUS_M = 20

# mínimo de 1 ha a 10 m:
# 1 ha = 10,000 m2 = 100 píxeles de 10 × 10 m.
MIN_COMPONENT_PIXELS = 100

# Conectividad de 8 vecinos.
CONNECTED_MAX_SIZE = 1024


# ============================================================
	# SENTINEL-2
# ============================================================

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


# ============================================================
# DESCARGA
# ============================================================

def download_geotiff(image, roi, name, out_tif):
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
    
    if not response.ok:
    detail = response.text[:3000]
    raise RuntimeError(
        f"Earth Engine HTTP {response.status_code} al descargar {name}\n"
        f"{detail}"
    )

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    if response.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            tifs = [n for n in zf.namelist()
                    if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                raise RuntimeError(
                    f"No se encontró GeoTIFF dentro del ZIP: {zf.namelist()}"
                )
            with zf.open(tifs[0]) as src, open(out_tif, "wb") as dst:
                dst.write(src.read())

    elif response.content[:4] in (b"II*\x00", b"MM\x00*"):
        with open(out_tif, "wb") as dst:
            dst.write(response.content)

    else:
        preview = response.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            "Earth Engine devolvió un formato inesperado. "
            f"Inicio: {preview}"
        )


# ============================================================
# ETIQUETA SEMILLA
# ============================================================

def clean_burn_candidates(burn_raw, open_water, focus_geom):
    """
    Devuelve candidatos quemados limpios:
      1. excluir agua
      2. abrir/cerrar para quitar ruido y compactar formas
      3. eliminar componentes < MIN_COMPONENT_PIXELS
      4. restringir a cercanía del foco
    """
    # binario puro
    x = burn_raw.And(open_water.Not()).toByte()

    # apertura: erosión + dilatación -> elimina motas pequeñas
    x = (
        x.focalMin(
            radius=MORPH_RADIUS_M,
            kernelType="circle",
            units="meters"
        )
        .focalMax(
            radius=MORPH_RADIUS_M,
            kernelType="circle",
            units="meters"
        )
        .gt(0)
        .toByte()
    )

    # cierre: dilatación + erosión -> rellena pequeñas discontinuidades
    x = (
        x.focalMax(
            radius=MORPH_RADIUS_M,
            kernelType="circle",
            units="meters"
        )
        .focalMin(
            radius=MORPH_RADIUS_M,
            kernelType="circle",
            units="meters"
        )
        .gt(0)
        .toByte()
    )

    # elimina objetos conectados pequeños.
    count = x.selfMask().connectedPixelCount(
        maxSize=CONNECTED_MAX_SIZE,
        eightConnected=True
    )
    x = x.updateMask(count.gte(MIN_COMPONENT_PIXELS)).unmask(0).toByte()

    # solo conservamos candidatos cercanos al hotspot/evento.
    near_focus = ee.Image.constant(1).clip(focus_geom).unmask(0).eq(1)
    x = x.And(near_focus).toByte()

    return x.rename("burn_clean")


def build_seed(pre, post, point, roi):
    nbr_pre, ndvi_pre, ndmi_pre, mndwi_pre = indices(pre)
    nbr_post, ndvi_post, ndmi_post, mndwi_post = indices(post)

    dnbr = nbr_pre.subtract(nbr_post).rename("dNBR")

    rnbr_denom = nbr_pre.abs().sqrt().max(ee.Image.constant(RNBR_DENOM_MIN))
    rnbr = (
        dnbr.divide(rnbr_denom)
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

    # evidencia espectral inicial, todavía sin limpieza espacial
    burn_raw = (
        dnbr.gte(DNBR_BURN_MIN)
        .And(rnbr.gte(RNBR_BURN_MIN))
        .And(ndvi_pre.gte(NDVI_PRE_BURN_MIN))
    )

    # no quemado: criterio muy seguro.
    nonburn = (
        dnbr.lte(DNBR_NONBURN_MAX)
        .And(rnbr.abs().lte(RNBR_NONBURN_MAX))
        .And(open_water.Not())
    )

    focus_geom = point.buffer(FOCUS_BUFFER_M, 1).intersection(roi, 1)
    burn_clean = clean_burn_candidates(burn_raw, open_water, focus_geom)

    # 255 se usa para todo lo ambiguo, agua o no-data.
    seed = (
        ee.Image(255)
        .where(nonburn, 0)
        .where(burn_clean.eq(1), 1)
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
        .addBands(open_water.toUint8().rename("open_water"))
        .addBands(burn_raw.toUint8().rename("burn_raw"))
        .addBands(burn_clean.toUint8().rename("burn_clean"))
        .addBands(valid.toUint8())
        .clip(roi)
    )

    return seed, aux


# ============================================================
# EVENTOS
# ============================================================

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

    seed, aux = build_seed(pre, post, point, roi)

    outputs = [
        (pre.select(BANDS), root / "data/raw/pre" / f"{event_id}_pre.tif"),
        (post.select(BANDS), root / "data/raw/post" / f"{event_id}_post.tif"),
        (aux, root / "data/raw/aux" / f"{event_id}_aux.tif"),
        (seed, root / "data/raw/labels" / f"{event_id}_label_seed.tif"),
    ]

    for image, path in outputs:
    if overwrite or not path.exists():
        print(f"    descargando: {path.name}")
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
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Cantidad de eventos. Use 0 para todos."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ee.Initialize(project=args.project)

    with open(args.events, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = rows[args.start:] if args.limit == 0 else rows[
        args.start:args.start + args.limit
    ]

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
            existing = []

    manifest_path.write_text(json.dumps(existing + manifest, indent=2))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/perimeters/raw"
)

OUT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "sardinia/catalog"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32632"


SCHEMA = {
    2019: {
        "id": "ID_INCE",
        "date": "DT_INCE",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPO_IGM",
        "type": "TIPOLOGIE",
        "official_area": "Ha_Totali",
        "forest_m2": "Mq_Bosco",
        "pasture_m2": "Mq_Pascolo",
        "other_m2": "Mq_Altro",
    },

    2020: {
        "id": "Id_Ince",
        "date": "Data",
        "province": "PROV",
        "municipality": "Comune",
        "toponym": "Toponimo",
        "type": "Tipologie",
        "official_area": "Ha_Totale",
        "forest_m2": "Mq_Bosco",
        "pasture_m2": "Mq_Pascolo",
        "other_m2": "Mq_Altro",
    },

    2021: {
        "id": "ID_INCE",
        "date": "DATA",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPONIMO",
        "type": "TIPOLOGIE",
        "official_area": "HA_TOTALE",
        "forest_m2": "MQ_BOSCO",
        "pasture_m2": "MQ_PASCOLO",
        "other_m2": "MQ_ALTRO",
    },

    2022: {
        "id": "ID_INCE",
        "date": "DT_INCE",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPO_IGM",
        "type": "TIPOLOGIE",
        "official_area": "TOTALE_HA",
        "forest_m2": "BOSCO_MQ",
        "pasture_m2": "PASCOLO_MQ",
        "other_m2": "ALTRO_MQ",
    },

    2023: {
        "id": "ID_INCE",
        "date": "DATA",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPONIMO",
        "type": "TIPOLOGIE",
        "official_area": "TOTALE_HA",
        "forest_m2": "BOSCO_MQ",
        "pasture_m2": "PASCOLO_MQ",
        "other_m2": "ALTRO_MQ",
    },

    2024: {
        "id": "COD_INCEND",
        "date": "DATA",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPONIMO",
        "type": "TIPOLOGIA",
        "official_area": "TOTALE_HA",
        "forest_m2": "BOSCO_MQ",
        "pasture_m2": "PASCOLO_MQ",
        "other_m2": "ALTRO_MQ",
    },

    2025: {
        "id": "COD_INCEND",
        "date": "DATA",
        "province": "PROV",
        "municipality": "COMUNE",
        "toponym": "TOPONIMO",
        "type": "TIPOLOGIA",
        "official_area": "TOTALE_HA",
        "forest_m2": "BOSCO_MQ",
        "pasture_m2": "PASCOLO_MQ",
        "other_m2": "ALTRO_MQ",
    },
}


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def sanitize(value):
    value = safe_text(value)

    value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    return value.strip("_")


rows = []


for year in sorted(SCHEMA):

    year_dir = ROOT / str(year)

    shapefiles = list(
        year_dir.glob("*.shp")
    )

    if len(shapefiles) != 1:
        raise RuntimeError(
            f"{year}: expected one shapefile, "
            f"found {len(shapefiles)}"
        )

    path = shapefiles[0]

    print(
        f"\n===== {year} ====="
    )

    print("Reading:", path.name)

    gdf = gpd.read_file(path)

    print(
        "Raw polygons:",
        len(gdf),
        "| CRS:",
        gdf.crs,
    )

    schema = SCHEMA[year]

    # --------------------------------------------------------
    # Geometry cleanup
    # --------------------------------------------------------

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    invalid = ~gdf.geometry.is_valid

    if invalid.any():

        print(
            "Repairing invalid geometries:",
            int(invalid.sum()),
        )

        gdf.loc[
            invalid,
            "geometry"
        ] = gdf.loc[
            invalid,
            "geometry"
        ].buffer(0)

    # Everything in one metric CRS.
    gdf = gdf.to_crs(
        TARGET_CRS
    )

    # --------------------------------------------------------
    # Normalized attributes
    # --------------------------------------------------------

    out = gpd.GeoDataFrame(
        geometry=gdf.geometry.copy(),
        crs=TARGET_CRS,
    )

    out["year"] = year

    out["fire_code"] = (
        gdf[schema["id"]]
        .apply(safe_text)
    )

    out["fire_date"] = pd.to_datetime(
        gdf[schema["date"]],
        errors="coerce",
        dayfirst=True,
    )

    out["province"] = (
        gdf[schema["province"]]
        .apply(safe_text)
    )

    out["municipality"] = (
        gdf[schema["municipality"]]
        .apply(safe_text)
    )

    out["toponym"] = (
        gdf[schema["toponym"]]
        .apply(safe_text)
    )

    out["fire_type"] = (
        gdf[schema["type"]]
        .apply(safe_text)
    )

    out["area_official_ha"] = safe_numeric(
        gdf[schema["official_area"]]
    )

    out["forest_ha"] = (
        safe_numeric(
            gdf[schema["forest_m2"]]
        )
        / 10000.0
    )

    out["pasture_ha"] = (
        safe_numeric(
            gdf[schema["pasture_m2"]]
        )
        / 10000.0
    )

    out["other_ha"] = (
        safe_numeric(
            gdf[schema["other_m2"]]
        )
        / 10000.0
    )

    # Independent area calculation from geometry.
    out["area_geom_ha"] = (
        out.geometry.area
        / 10000.0
    )

    out["month"] = (
        out["fire_date"]
        .dt.month
    )

    out["day_of_year"] = (
        out["fire_date"]
        .dt.dayofyear
    )

    # Stable unique identifier.
    ids = []

    for idx, fire_code in enumerate(
        out["fire_code"]
    ):

        code = sanitize(
            fire_code
        )

        if not code:
            code = f"row_{idx:05d}"

        ids.append(
            f"SA_{year}_{code}"
        )

    out["event_id"] = ids

    rows.append(out)


catalog = pd.concat(
    rows,
    ignore_index=True,
)

catalog = gpd.GeoDataFrame(
    catalog,
    geometry="geometry",
    crs=TARGET_CRS,
)


# ============================================================
# CENTROIDS
# ============================================================

centroids = catalog.geometry.centroid

centroid_gdf = gpd.GeoSeries(
    centroids,
    crs=TARGET_CRS,
).to_crs("EPSG:4326")

catalog["longitude"] = (
    centroid_gdf.x.values
)

catalog["latitude"] = (
    centroid_gdf.y.values
)


# ============================================================
# QUALITY FLAGS
# ============================================================

catalog["has_date"] = (
    catalog["fire_date"]
    .notna()
)

catalog["summer_fire"] = (
    catalog["month"]
    .between(5, 10)
)

catalog["area_ge_50ha"] = (
    catalog["area_geom_ha"]
    >= 50
)

catalog["area_ge_100ha"] = (
    catalog["area_geom_ha"]
    >= 100
)

catalog["area_ge_200ha"] = (
    catalog["area_geom_ha"]
    >= 200
)


# Difference official vs geometry area.
catalog["area_difference_pct"] = (
    100.0
    * (
        catalog["area_geom_ha"]
        - catalog["area_official_ha"]
    ).abs()
    / catalog["area_official_ha"]
)

catalog.loc[
    ~np.isfinite(
        catalog["area_difference_pct"]
    ),
    "area_difference_pct",
] = np.nan


# ============================================================
# DUPLICATE EVENT CODES
# ============================================================

dup_counts = (
    catalog.groupby(
        ["year", "fire_code"]
    )
    .size()
    .rename("n_geometries_same_code")
)

catalog = catalog.merge(
    dup_counts,
    on=["year", "fire_code"],
    how="left",
)


# ============================================================
# OUTPUT FULL CATALOG
# ============================================================

gpkg_path = (
    OUT
    / "sardinia_fire_catalog_2019_2025.gpkg"
)

csv_path = (
    OUT
    / "sardinia_fire_catalog_2019_2025.csv"
)

catalog.to_file(
    gpkg_path,
    layer="fires",
    driver="GPKG",
)

csv_df = (
    catalog
    .drop(columns="geometry")
    .copy()
)

csv_df.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# PILOT CANDIDATES
# ============================================================

candidates = catalog[
    catalog["has_date"]
    & catalog["summer_fire"]
    & catalog["area_ge_100ha"]
].copy()

candidates = candidates.sort_values(
    [
        "year",
        "area_geom_ha",
    ],
    ascending=[
        True,
        False,
    ],
)

candidate_gpkg = (
    OUT
    / "sardinia_candidates_ge100ha.gpkg"
)

candidate_csv = (
    OUT
    / "sardinia_candidates_ge100ha.csv"
)

candidates.to_file(
    candidate_gpkg,
    layer="candidates",
    driver="GPKG",
)

candidates.drop(
    columns="geometry"
).to_csv(
    candidate_csv,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("==========================================")
print("SARDINIA FIRE CATALOG SUMMARY")
print("==========================================")

print(
    "Total polygons:",
    len(catalog),
)

print(
    "With valid date:",
    int(catalog["has_date"].sum()),
)

print(
    ">= 50 ha:",
    int(catalog["area_ge_50ha"].sum()),
)

print(
    ">= 100 ha:",
    int(catalog["area_ge_100ha"].sum()),
)

print(
    ">= 200 ha:",
    int(catalog["area_ge_200ha"].sum()),
)

print(
    "Pilot candidates "
    "(>=100 ha, May-Oct):",
    len(candidates),
)

print("\nCandidates by year:")

print(
    candidates
    .groupby("year")
    .size()
)

print("\nLargest 30 candidate fires:")

show = candidates[
    [
        "event_id",
        "fire_date",
        "municipality",
        "toponym",
        "area_geom_ha",
        "area_official_ha",
        "forest_ha",
        "pasture_ha",
    ]
].sort_values(
    "area_geom_ha",
    ascending=False,
).head(30)

print(
    show.to_string(
        index=False
    )
)

print("\nDuplicate fire codes:")

dups = catalog[
    catalog["n_geometries_same_code"] > 1
]

print(
    "Rows belonging to duplicated codes:",
    len(dups),
)

if len(dups):

    print(
        dups[
            [
                "year",
                "fire_code",
                "municipality",
                "area_geom_ha",
                "n_geometries_same_code",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


print("\nOutputs:")
print(gpkg_path)
print(csv_path)
print(candidate_gpkg)
print(candidate_csv)

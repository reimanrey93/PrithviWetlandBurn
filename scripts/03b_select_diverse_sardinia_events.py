#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/catalog"
)

QC_CSV = ROOT / "sardinia_s2_pairs_scl_qc.csv"

FIRE_GPKG = (
    ROOT / "sardinia_candidates_ge100ha.gpkg"
)

OUT_CSV = (
    ROOT / "sardinia_selected_28_diverse.csv"
)

OUT_GPKG = (
    ROOT / "sardinia_selected_28_diverse.gpkg"
)

N_PER_YEAR = 4

CRS = "EPSG:32632"


# ============================================================
# Load
# ============================================================

qc = pd.read_csv(QC_CSV)

qc = qc[
    qc["local_qc_ok"] == True
].copy()

fires = gpd.read_file(
    FIRE_GPKG
).to_crs(CRS)

gdf = fires.merge(
    qc,
    on="event_id",
    how="inner",
    suffixes=("", "_qc"),
)

print(
    "QC-passing events:",
    len(gdf)
)


# ============================================================
# Helpers
# ============================================================

def normalize(series):

    x = series.astype(float)

    xmin = x.min()
    xmax = x.max()

    if xmax <= xmin:
        return pd.Series(
            np.ones(len(x)),
            index=x.index,
        )

    return (
        (x - xmin)
        / (xmax - xmin)
    )


def pair_key(row):
    return (
        str(row["pre_item_id"]),
        str(row["post_item_id"]),
    )


def min_distance_score(
    candidate_geom,
    selected_geoms,
):
    """
    0–1 score.

    Distances >= 100 km saturate at 1.
    """

    if not selected_geoms:
        return 1.0

    centroid = candidate_geom.centroid

    distances = [
        centroid.distance(
            geom.centroid
        )
        for geom in selected_geoms
    ]

    min_dist_m = min(distances)

    return min(
        min_dist_m / 100000.0,
        1.0,
    )


# ============================================================
# Selection
# ============================================================

selected_all = []


for year in sorted(
    gdf["year"].unique()
):

    subset = (
        gdf[
            gdf["year"] == year
        ]
        .copy()
        .reset_index(drop=True)
    )

    print()
    print(
        "====================================="
    )
    print(
        f"YEAR {year} | candidates={len(subset)}"
    )
    print(
        "====================================="
    )

    subset["quality_norm"] = normalize(
        subset["final_quality_score"]
    )

    subset["area_log"] = np.log1p(
        subset["area_geom_ha"]
    )

    subset["area_norm"] = normalize(
        subset["area_log"]
    )

    chosen = []
    used_pairs = set()


    # --------------------------------------------------------
    # 1. Largest event
    # --------------------------------------------------------

    largest_idx = (
        subset["area_geom_ha"]
        .idxmax()
    )

    largest = subset.loc[
        largest_idx
    ]

    chosen.append(
        largest
    )

    used_pairs.add(
        pair_key(largest)
    )

    print(
        "Largest:",
        largest["event_id"],
        f"{largest['area_geom_ha']:.1f} ha"
    )


    # --------------------------------------------------------
    # 2. Highest quality with a different acquisition pair
    # --------------------------------------------------------

    remaining = subset.drop(
        index=[
            x.name
            for x in chosen
        ],
        errors="ignore",
    )

    remaining = remaining[
        remaining.apply(
            lambda r: pair_key(r)
            not in used_pairs,
            axis=1,
        )
    ]

    if len(remaining):

        quality_idx = (
            remaining[
                "final_quality_score"
            ]
            .idxmax()
        )

        best = subset.loc[
            quality_idx
        ]

        chosen.append(best)

        used_pairs.add(
            pair_key(best)
        )

        print(
            "Quality:",
            best["event_id"],
            f"score={best['final_quality_score']:.3f}"
        )


    # --------------------------------------------------------
    # 3–4. Greedy diversity selection
    # --------------------------------------------------------

    while (
        len(chosen) < N_PER_YEAR
    ):

        chosen_indices = {
            x.name
            for x in chosen
        }

        candidates = subset[
            ~subset.index.isin(
                chosen_indices
            )
        ].copy()

        # Strong preference for a different
        # PRE/POST acquisition pair.
        unique_pair = candidates[
            candidates.apply(
                lambda r: pair_key(r)
                not in used_pairs,
                axis=1,
            )
        ].copy()

        if len(unique_pair):
            candidates = unique_pair

        selected_geoms = [
            x.geometry
            for x in chosen
        ]

        spatial_scores = []

        for _, row in candidates.iterrows():

            spatial_scores.append(
                min_distance_score(
                    row.geometry,
                    selected_geoms,
                )
            )

        candidates[
            "spatial_norm"
        ] = spatial_scores

        # Composite:
        #
        # 45% spatial diversity
        # 30% image quality
        # 25% event size
        #
        candidates[
            "selection_score"
        ] = (
            0.45
            * candidates[
                "spatial_norm"
            ]
            + 0.30
            * candidates[
                "quality_norm"
            ]
            + 0.25
            * candidates[
                "area_norm"
            ]
        )

        idx = candidates[
            "selection_score"
        ].idxmax()

        candidate = subset.loc[
            idx
        ].copy()

        candidate[
            "selection_score"
        ] = candidates.loc[
            idx,
            "selection_score",
        ]

        candidate[
            "spatial_norm"
        ] = candidates.loc[
            idx,
            "spatial_norm",
        ]

        chosen.append(
            candidate
        )

        used_pairs.add(
            pair_key(candidate)
        )

        print(
            "Diverse:",
            candidate["event_id"],
            f"{candidate['area_geom_ha']:.1f} ha",
            f"spatial={candidate['spatial_norm']:.3f}",
        )


    selected_all.extend(
        chosen
    )


# ============================================================
# Final GeoDataFrame
# ============================================================

selected = gpd.GeoDataFrame(
    selected_all,
    geometry="geometry",
    crs=CRS,
)

selected = (
    selected
    .sort_values(
        [
            "year",
            "area_geom_ha",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)


# ============================================================
# Checks
# ============================================================

selected[
    "acquisition_pair"
] = (
    selected[
        "pre_item_id"
    ].astype(str)
    + "___"
    + selected[
        "post_item_id"
    ].astype(str)
)

duplicates = selected[
    selected[
        "acquisition_pair"
    ].duplicated(
        keep=False
    )
]


# ============================================================
# Save
# ============================================================

selected.to_file(
    OUT_GPKG,
    layer="selected_fires",
    driver="GPKG",
)

selected.drop(
    columns="geometry"
).to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Report
# ============================================================

print()
print(
    "=========================================="
)
print(
    "FINAL DIVERSE PILOT SELECTION"
)
print(
    "=========================================="
)

print(
    "Selected:",
    len(selected)
)

print()
print(
    "By year:"
)

print(
    selected
    .groupby("year")
    .size()
)

print()
print(
    "Area statistics:"
)

print(
    selected[
        "area_geom_ha"
    ].describe()
)

print()
print(
    "Selected events:"
)

cols = [
    "event_id",
    "year",
    "municipality",
    "area_geom_ha",
    "pre_date",
    "post_date",
    "pre_local_bad_pct",
    "post_local_bad_pct",
    "final_quality_score",
]

print(
    selected[
        cols
    ].to_string(
        index=False
    )
)

print()
print(
    "Repeated acquisition pairs:",
    len(duplicates)
)

if len(duplicates):

    print(
        duplicates[
            [
                "event_id",
                "year",
                "municipality",
                "acquisition_pair",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("Outputs:")
print(OUT_CSV)
print(OUT_GPKG)

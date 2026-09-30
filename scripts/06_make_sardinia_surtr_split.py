#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations
import csv
import json

import numpy as np


EVENT_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/events"
)

OUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/splits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "surtr_sardinia_v1_event_split.csv"
)

OUT_JSON = (
    OUT_DIR
    / "surtr_sardinia_v1_split_summary.json"
)

N_VAL = 4
N_TEST = 4

# Very large event: valuable for training, but would dominate
# pixel-level validation/test metrics.
FORCE_TRAIN = {
    "SA_2021_2021_OR_2142",
}


# ------------------------------------------------------------
# Load events
# ------------------------------------------------------------

events = []

for d in sorted(EVENT_ROOT.iterdir()):

    if not d.is_dir():
        continue

    meta_path = d / "metadata.json"

    if not meta_path.exists():
        continue

    m = json.loads(
        meta_path.read_text()
    )

    events.append({
        "event_id": d.name,
        "year": int(m["year"]),
        "municipality": m["municipality"],
        "area_ha": float(
            m["official_geometry_area_ha"]
        ),
        "width": int(m["width"]),
        "height": int(m["height"]),
    })


if len(events) != 28:
    raise RuntimeError(
        f"Expected 28 events, found {len(events)}"
    )


# ------------------------------------------------------------
# Feature normalization
# ------------------------------------------------------------

areas = np.array(
    [np.log1p(e["area_ha"]) for e in events],
    dtype=float,
)

sizes = np.array(
    [
        np.log1p(e["width"] * e["height"])
        for e in events
    ],
    dtype=float,
)


def zscore(x):
    return (
        (x - x.mean())
        / (x.std() + 1e-12)
    )


area_z = zscore(areas)
size_z = zscore(sizes)

for e, a, s in zip(
    events,
    area_z,
    size_z,
):
    e["area_z"] = float(a)
    e["size_z"] = float(s)


eligible = [
    e for e in events
    if e["event_id"] not in FORCE_TRAIN
]


# ------------------------------------------------------------
# Subset score
# ------------------------------------------------------------

global_area_mean = np.mean(
    [e["area_z"] for e in events]
)

global_size_mean = np.mean(
    [e["size_z"] for e in events]
)


def subset_score(subset):

    years = [e["year"] for e in subset]

    # Reward temporal diversity.
    year_diversity_penalty = (
        len(subset)
        - len(set(years))
    )

    area_mean = np.mean(
        [e["area_z"] for e in subset]
    )

    size_mean = np.mean(
        [e["size_z"] for e in subset]
    )

    representation_penalty = (
        abs(area_mean - global_area_mean)
        + abs(size_mean - global_size_mean)
    )

    return (
        3.0 * year_diversity_penalty
        + representation_penalty
    )


# ------------------------------------------------------------
# Choose TEST
# ------------------------------------------------------------

test_candidates = list(
    combinations(eligible, N_TEST)
)

test = min(
    test_candidates,
    key=subset_score,
)

test_ids = {
    e["event_id"]
    for e in test
}


# ------------------------------------------------------------
# Choose VAL from remaining events
# ------------------------------------------------------------

remaining = [
    e for e in eligible
    if e["event_id"] not in test_ids
]


def val_score(subset):

    base = subset_score(subset)

    val_years = {
        e["year"]
        for e in subset
    }

    test_years = {
        e["year"]
        for e in test
    }

    # Prefer complementary years where possible.
    overlap = len(
        val_years & test_years
    )

    return (
        base
        + 0.25 * overlap
    )


val = min(
    combinations(remaining, N_VAL),
    key=val_score,
)

val_ids = {
    e["event_id"]
    for e in val
}


# ------------------------------------------------------------
# Assign
# ------------------------------------------------------------

rows = []

for e in events:

    event_id = e["event_id"]

    if event_id in test_ids:
        split = "test"

    elif event_id in val_ids:
        split = "val"

    else:
        split = "train"

    rows.append({
        "event_id": event_id,
        "split": split,
        "year": e["year"],
        "municipality": e["municipality"],
        "area_ha": e["area_ha"],
        "width": e["width"],
        "height": e["height"],
    })


# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------

fields = [
    "event_id",
    "split",
    "year",
    "municipality",
    "area_ha",
    "width",
    "height",
]

with open(
    OUT_CSV,
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


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = {}

for split in [
    "train",
    "val",
    "test",
]:

    rr = [
        r for r in rows
        if r["split"] == split
    ]

    summary[split] = {
        "events": len(rr),
        "years": sorted(
            set(r["year"] for r in rr)
        ),
        "area_ha": float(
            sum(r["area_ha"] for r in rr)
        ),
        "event_ids": [
            r["event_id"]
            for r in rr
        ],
    }


OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("=" * 72)
print("SURTR SARDINIA V1 EVENT SPLIT")
print("=" * 72)

for split in [
    "train",
    "val",
    "test",
]:

    s = summary[split]

    print()
    print(
        split.upper(),
        f"({s['events']} events)"
    )

    print(
        "years:",
        s["years"]
    )

    print(
        "area:",
        f"{s['area_ha']:.1f} ha"
    )

    for eid in s["event_ids"]:
        print(" ", eid)

print()
print("CSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print("=" * 72)

#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "evaluation/sardinia_surtr_v1"
)

PATCH_CSV = ROOT / "test_patch_metrics.csv"
EVENT_CSV = ROOT / "test_event_metrics.csv"

OUT_CSV = ROOT / "test_patch_comparison.csv"
OUT_JSON = ROOT / "qualitative_cases.json"


def main():

    df = pd.read_csv(PATCH_CSV)

    required = {
        "event_id",
        "patch_id",
        "model",
        "iou",
        "f1",
        "precision",
        "recall",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Baseline vs SURTR in same row
    # ---------------------------------------------------------

    base = (
        df[df["model"] == "baseline"]
        .copy()
        .set_index(["event_id", "patch_id"])
    )

    surtr = (
        df[df["model"] == "surtr"]
        .copy()
        .set_index(["event_id", "patch_id"])
    )

    common = base.index.intersection(
        surtr.index
    )

    rows = []

    for event_id, patch_id in common:

        b = base.loc[(event_id, patch_id)]
        s = surtr.loc[(event_id, patch_id)]

        rows.append({
            "event_id": event_id,
            "patch_id": patch_id,

            "baseline_iou": b["iou"],
            "surtr_iou": s["iou"],
            "delta_iou": (
                s["iou"] - b["iou"]
            ),

            "baseline_f1": b["f1"],
            "surtr_f1": s["f1"],
            "delta_f1": (
                s["f1"] - b["f1"]
            ),

            "baseline_precision":
                b["precision"],
            "surtr_precision":
                s["precision"],

            "baseline_recall":
                b["recall"],
            "surtr_recall":
                s["recall"],

            "delta_recall": (
                s["recall"]
                - b["recall"]
            ),

            "valid_pixels":
                int(b["valid_pixels"]),
        })

    comp = pd.DataFrame(rows)

    comp = comp.sort_values(
        "delta_iou",
        ascending=False,
    ).reset_index(drop=True)

    comp.to_csv(
        OUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Objective qualitative selection
    # ---------------------------------------------------------

    best = comp.iloc[0]

    worst = comp.iloc[-1]

    same_idx = (
        comp["delta_iou"]
        .abs()
        .idxmin()
    )

    same = comp.loc[same_idx]

    cases = {
        "best_improvement": {
            "event_id":
                best["event_id"],
            "patch_id":
                best["patch_id"],
            "baseline_iou":
                float(best["baseline_iou"]),
            "surtr_iou":
                float(best["surtr_iou"]),
            "delta_iou":
                float(best["delta_iou"]),
            "delta_recall":
                float(best["delta_recall"]),
        },

        "most_similar": {
            "event_id":
                same["event_id"],
            "patch_id":
                same["patch_id"],
            "baseline_iou":
                float(same["baseline_iou"]),
            "surtr_iou":
                float(same["surtr_iou"]),
            "delta_iou":
                float(same["delta_iou"]),
            "delta_recall":
                float(same["delta_recall"]),
        },

        "worst_degradation": {
            "event_id":
                worst["event_id"],
            "patch_id":
                worst["patch_id"],
            "baseline_iou":
                float(worst["baseline_iou"]),
            "surtr_iou":
                float(worst["surtr_iou"]),
            "delta_iou":
                float(worst["delta_iou"]),
            "delta_recall":
                float(worst["delta_recall"]),
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            cases,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("=" * 78)
    print("SURTR V1 — PATCH-LEVEL TEST ANALYSIS")
    print("=" * 78)

    print(f"Patches: {len(comp)}")

    print()
    print("Delta IoU distribution")
    print("----------------------")

    print(
        comp["delta_iou"]
        .describe()
        .to_string()
    )

    n_better = int(
        (comp["delta_iou"] > 0).sum()
    )

    n_equal = int(
        (
            comp["delta_iou"].abs()
            < 1e-6
        ).sum()
    )

    n_worse = int(
        (comp["delta_iou"] < 0).sum()
    )

    print()
    print(
        f"SURTR better: {n_better}/{len(comp)}"
    )

    print(
        f"Equal:        {n_equal}/{len(comp)}"
    )

    print(
        f"SURTR worse:  {n_worse}/{len(comp)}"
    )

    print()
    print("=" * 78)
    print("TOP 5 IMPROVEMENTS")
    print("=" * 78)

    cols = [
        "event_id",
        "patch_id",
        "baseline_iou",
        "surtr_iou",
        "delta_iou",
        "delta_recall",
    ]

    print(
        comp[cols]
        .head(5)
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("TOP 5 DEGRADATIONS")
    print("=" * 78)

    print(
        comp[cols]
        .tail(5)
        .sort_values("delta_iou")
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("SELECTED QUALITATIVE CASES")
    print("=" * 78)

    for name, case in cases.items():

        print()
        print(name)

        print(
            f"  event: {case['event_id']}"
        )

        print(
            f"  patch: {case['patch_id']}"
        )

        print(
            f"  A0 IoU: "
            f"{case['baseline_iou']:.4f}"
        )

        print(
            f"  A1 IoU: "
            f"{case['surtr_iou']:.4f}"
        )

        print(
            f"  delta: "
            f"{case['delta_iou']:+.4f}"
        )

        print(
            f"  delta recall: "
            f"{case['delta_recall']:+.4f}"
        )

    print()
    print("Saved:")
    print(OUT_CSV)
    print(OUT_JSON)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def pixel_area_m2(ds):
    if ds.crs is None:
        raise RuntimeError("Raster without CRS.")

    if ds.crs.is_geographic:
        raise RuntimeError(
            "Raster is in degrees. Reproject to metric CRS first."
        )

    t = ds.transform
    return abs(t.a * t.e - t.b * t.d)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-json", required=True)

    args = parser.parse_args()

    pred_path = Path(args.prediction)
    ref_path = Path(args.reference)
    out_json = Path(args.out_json)

    with rasterio.open(pred_path) as pred_ds, rasterio.open(ref_path) as ref_ds:
        if pred_ds.width != ref_ds.width or pred_ds.height != ref_ds.height:
            raise RuntimeError("Prediction/reference shape mismatch.")

        if pred_ds.transform != ref_ds.transform:
            raise RuntimeError("Prediction/reference transform mismatch.")

        if pred_ds.crs != ref_ds.crs:
            raise RuntimeError("Prediction/reference CRS mismatch.")

        pred = pred_ds.read(1)
        ref = ref_ds.read(1)

        pixel_m2 = pixel_area_m2(pred_ds)

        # Prediction format:
        # 0 = unburned
        # 1 = burned
        # 255 = invalid
        pred_burned = pred == 1

        # Reference mask:
        # assume burned == 1
        ref_burned = ref == 1

        pred_pixels = int(pred_burned.sum())
        ref_pixels = int(ref_burned.sum())

        pred_ha = pred_pixels * pixel_m2 / 10000.0
        ref_ha = ref_pixels * pixel_m2 / 10000.0

        diff_ha = pred_ha - ref_ha
        rel_err = abs(diff_ha) / ref_ha * 100.0 if ref_ha > 0 else float("nan")

        report = {
            "prediction": str(pred_path),
            "reference": str(ref_path),
            "pixel_area_m2": pixel_m2,
            "predicted_burned_pixels": pred_pixels,
            "reference_burned_pixels": ref_pixels,
            "predicted_burned_area_ha": pred_ha,
            "reference_burned_area_ha": ref_ha,
            "difference_ha": diff_ha,
            "absolute_relative_error_percent": rel_err,
        }

        out_json.write_text(json.dumps(report, indent=2))

        print("=" * 72)
        print("BURNED AREA COMPARISON")
        print("=" * 72)
        print(f"Predicted area : {pred_ha:,.2f} ha")
        print(f"Reference area : {ref_ha:,.2f} ha")
        print(f"Difference     : {diff_ha:+,.2f} ha")
        print(f"Abs. rel. err. : {rel_err:.2f}%")
        print()
        print("Saved:", out_json)


if __name__ == "__main__":
    main()

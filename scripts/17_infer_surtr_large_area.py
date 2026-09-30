#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.surtr_bitemporal import SURTRBiTemporal
from terratorch.cli_tools import LightningInferenceModel


CONFIG_ORIGINAL = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/configs/"
    "finetune_prithvi_burnscar_v2.yaml"
)

PRITHVI_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/models/"
    "prithvi_burnscars/"
    "Prithvi_EO_V2_300M_BurnScars.pt"
)

SURTR_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "checkpoints/sardinia_surtr_v1/"
    "surtr/best.pt"
)

CONFIG_TMP = Path("/tmp/surtr_large_area.yaml")


MEANS = torch.tensor([
    0.052376806868,
    0.070576639747,
    0.081104305106,
    0.177237257070,
    0.225423962965,
    0.174360119430,
], dtype=torch.float32).view(1, 6, 1, 1)

STDS = torch.tensor([
    0.021721302751,
    0.026402486035,
    0.034470597693,
    0.091198580301,
    0.103103430325,
    0.088007767199,
], dtype=torch.float32).view(1, 6, 1, 1)


def load_base_model():
    cfg = yaml.safe_load(CONFIG_ORIGINAL.read_text())
    cfg.pop("lr_scheduler", None)

    CONFIG_TMP.write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )

    old_argv = sys.argv[:]

    try:
        sys.argv = [sys.argv[0]]

        loaded = LightningInferenceModel.from_config(
            str(CONFIG_TMP),
            str(PRITHVI_CHECKPOINT),
        )

    finally:
        sys.argv = old_argv

    return loaded.model.model


def load_surtr(device):
    base_model = load_base_model()

    model = SURTRBiTemporal(
        pixel_model=base_model,
        freeze_encoder=True,
        freeze_neck=True,
        train_decoder=True,
    )

    checkpoint = torch.load(
        SURTR_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        f"SURTR checkpoint: epoch "
        f"{checkpoint['epoch']} | "
        f"val_loss={checkpoint['val_loss']:.6f}"
    )

    return model


def normalize(x):
    means = MEANS.to(x.device, x.dtype)
    stds = STDS.to(x.device, x.dtype)

    return (x - means) / stds


def check_alignment(pre, post):
    if pre.width != post.width:
        raise RuntimeError("PRE/POST width mismatch.")

    if pre.height != post.height:
        raise RuntimeError("PRE/POST height mismatch.")

    if pre.transform != post.transform:
        raise RuntimeError("PRE/POST transform mismatch.")

    if pre.crs != post.crs:
        raise RuntimeError("PRE/POST CRS mismatch.")

    if pre.count != 6 or post.count != 6:
        raise RuntimeError(
            "Expected exactly 6 bands: "
            "B2,B3,B4,B8,B11,B12."
        )


def blending_window(size):
    h = np.hanning(size)

    weight = np.outer(h, h)

    if weight.max() > 0:
        weight /= weight.max()

    weight = 0.05 + 0.95 * weight

    return weight.astype(np.float32)


def determine_scale(arr, valid):
    values = arr[:, valid]

    if values.size == 0:
        return 1.0

    q99 = np.nanpercentile(
        values,
        99,
    )

    if q99 > 2.0:
        return 10000.0

    return 1.0


def pixel_area_m2(dataset):
    if dataset.crs is None:
        raise RuntimeError(
            "Raster has no CRS."
        )

    if dataset.crs.is_geographic:
        raise RuntimeError(
            "Raster is in degrees. "
            "Reproject to a metric CRS first."
        )

    t = dataset.transform

    return abs(
        t.a * t.e - t.b * t.d
    )


def compute_reference_metrics(
    prediction,
    reference,
    pixel_m2,
):
    common_valid = (
        (prediction != 255)
        & (reference != 255)
    )

    pred_burned = (
        common_valid
        & (prediction == 1)
    )

    ref_burned = (
        common_valid
        & (reference == 1)
    )

    tp = int(
        (pred_burned & ref_burned).sum()
    )

    fp = int(
        (
            pred_burned
            & (~ref_burned)
            & common_valid
        ).sum()
    )

    fn = int(
        (
            (~pred_burned)
            & ref_burned
            & common_valid
        ).sum()
    )

    eps = 1e-12

    iou = tp / (tp + fp + fn + eps)

    precision = tp / (tp + fp + eps)

    recall = tp / (tp + fn + eps)

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall + eps)
    )

    common_valid_pixels = int(
        common_valid.sum()
    )

    pred_pixels = int(
        pred_burned.sum()
    )

    ref_pixels = int(
        ref_burned.sum()
    )

    common_valid_ha = (
        common_valid_pixels
        * pixel_m2
        / 10000.0
    )

    pred_ha = (
        pred_pixels
        * pixel_m2
        / 10000.0
    )

    ref_ha = (
        ref_pixels
        * pixel_m2
        / 10000.0
    )

    diff_ha = pred_ha - ref_ha

    rel_err = (
        abs(diff_ha)
        / ref_ha
        * 100.0
        if ref_ha > 0
        else float("nan")
    )

    return {
        "evaluation_area_ha": common_valid_ha,
        "predicted_burned_area_ha": pred_ha,
        "reference_burned_area_ha": ref_ha,
        "difference_ha": diff_ha,
        "absolute_relative_error_percent": rel_err,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pre",
        required=True,
    )

    parser.add_argument(
        "--post",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Optional aligned reference mask "
            "with 0=unburned, 1=burned, 255=ignore."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--scale",
        default="auto",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_size = args.tile_size
    overlap = args.overlap

    if overlap >= tile_size:
        raise ValueError(
            "Overlap must be smaller than tile size."
        )

    stride = tile_size - overlap

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    torch.set_float32_matmul_precision(
        "high"
    )

    model = load_surtr(device)

    weight_tile = blending_window(
        tile_size
    )

    with rasterio.open(args.pre) as pre_ds, \
         rasterio.open(args.post) as post_ds:

        check_alignment(
            pre_ds,
            post_ds,
        )

        width = pre_ds.width
        height = pre_ds.height

        print("=" * 72)
        print("SURTR LARGE-AREA INFERENCE")
        print("=" * 72)

        print("PRE :", args.pre)
        print("POST:", args.post)

        print(
            "Shape:",
            height,
            "x",
            width,
        )

        print(
            "CRS:",
            pre_ds.crs,
        )

        print(
            "Resolution:",
            pre_ds.res,
        )

        print(
            "Tile:",
            tile_size,
        )

        print(
            "Overlap:",
            overlap,
        )

        print(
            "Stride:",
            stride,
        )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="surtr_large_"
            )
        )

        prob_sum = np.memmap(
            temp_dir / "prob_sum.dat",
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )

        weight_sum = np.memmap(
            temp_dir / "weight_sum.dat",
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )

        valid_seen = np.memmap(
            temp_dir / "valid_seen.dat",
            mode="w+",
            dtype=np.uint8,
            shape=(height, width),
        )

        prob_sum[:] = 0
        weight_sum[:] = 0
        valid_seen[:] = 0

        scale = None
        tile_counter = 0

        for row in range(
            0,
            height,
            stride,
        ):

            for col in range(
                0,
                width,
                stride,
            ):

                window = Window(
                    col,
                    row,
                    tile_size,
                    tile_size,
                )

                # ==================================================
                # IMPORTANT:
                # Read zeros as VALID values.
                # Only explicit nodata / NaN / Inf are invalid.
                # ==================================================

                pre = pre_ds.read(
                    indexes=range(1, 7),
                    window=window,
                    boundless=True,
                    masked=False,
                    fill_value=pre_ds.nodata,
                ).astype(np.float32)

                post = post_ds.read(
                    indexes=range(1, 7),
                    window=window,
                    boundless=True,
                    masked=False,
                    fill_value=post_ds.nodata,
                ).astype(np.float32)

                valid = np.ones(
                    pre.shape[-2:],
                    dtype=bool,
                )

                if pre_ds.nodata is not None:
                    valid &= ~np.any(
                        pre == pre_ds.nodata,
                        axis=0,
                    )

                if post_ds.nodata is not None:
                    valid &= ~np.any(
                        post == post_ds.nodata,
                        axis=0,
                    )

                valid &= np.isfinite(
                    pre
                ).all(axis=0)

                valid &= np.isfinite(
                    post
                ).all(axis=0)

                pre[:, ~valid] = 0.0
                post[:, ~valid] = 0.0

                if scale is None:

                    if args.scale == "auto":
                        scale = determine_scale(
                            post,
                            valid,
                        )
                    else:
                        scale = float(
                            args.scale
                        )

                    print(
                        "Reflectance scale:",
                        scale,
                    )

                pre /= scale
                post /= scale

                if not valid.any():
                    continue

                pre_t = torch.from_numpy(
                    pre
                ).unsqueeze(0).to(
                    device
                )

                post_t = torch.from_numpy(
                    post
                ).unsqueeze(0).to(
                    device
                )

                pre_t = normalize(pre_t)
                post_t = normalize(post_t)

                with torch.inference_mode():

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):

                        logits = model(
                            pre=pre_t,
                            post=post_t,
                        )

                        probability = (
                            torch.softmax(
                                logits.float(),
                                dim=1,
                            )[:, 1]
                        )

                probability = (
                    probability[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                rh = min(
                    tile_size,
                    height - row,
                )

                cw = min(
                    tile_size,
                    width - col,
                )

                probability = probability[
                    :rh,
                    :cw,
                ]

                local_valid = valid[
                    :rh,
                    :cw,
                ]

                local_weight = (
                    weight_tile[
                        :rh,
                        :cw,
                    ]
                    * local_valid
                )

                rs = slice(
                    row,
                    row + rh,
                )

                cs = slice(
                    col,
                    col + cw,
                )

                prob_sum[rs, cs] += (
                    probability
                    * local_weight
                )

                weight_sum[rs, cs] += (
                    local_weight
                )

                valid_seen[rs, cs] |= (
                    local_valid.astype(
                        np.uint8
                    )
                )

                tile_counter += 1

                if tile_counter % 25 == 0:
                    print(
                        f"Processed "
                        f"{tile_counter} tiles"
                    )

        probability = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        good = weight_sum > 0

        probability[good] = (
            prob_sum[good]
            / weight_sum[good]
        )

        valid_final = (
            good
            & (valid_seen > 0)
        )

        burned = (
            (probability >= args.threshold)
            & valid_final
        )

        pixel_m2 = pixel_area_m2(
            pre_ds
        )

        burned_pixels = int(
            burned.sum()
        )

        valid_pixels = int(
            valid_final.sum()
        )

        burned_area_ha = (
            burned_pixels
            * pixel_m2
            / 10000.0
        )

        valid_area_ha = (
            valid_pixels
            * pixel_m2
            / 10000.0
        )

        burned_fraction = (
            burned_area_ha
            / valid_area_ha
            if valid_area_ha > 0
            else float("nan")
        )

        profile = pre_ds.profile.copy()

        profile.update(
            count=1,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        probability_path = (
            out_dir
            / "surtr_burn_probability.tif"
        )

        prob_profile = profile.copy()

        prob_profile.update(
            dtype="float32",
            nodata=-9999.0,
        )

        probability_out = (
            probability.copy()
        )

        probability_out[
            ~valid_final
        ] = -9999.0

        with rasterio.open(
            probability_path,
            "w",
            **prob_profile,
        ) as dst:
            dst.write(
                probability_out,
                1,
            )

        mask_path = (
            out_dir
            / "surtr_burn_mask.tif"
        )

        mask_profile = profile.copy()

        mask_profile.update(
            dtype="uint8",
            nodata=255,
        )

        mask_out = np.full(
            (height, width),
            255,
            dtype=np.uint8,
        )

        mask_out[
            valid_final & ~burned
        ] = 0

        mask_out[
            burned
        ] = 1

        with rasterio.open(
            mask_path,
            "w",
            **mask_profile,
        ) as dst:
            dst.write(
                mask_out,
                1,
            )

        report = {
            "pre": str(args.pre),
            "post": str(args.post),
            "checkpoint": str(
                SURTR_CHECKPOINT
            ),
            "crs": str(pre_ds.crs),
            "width": width,
            "height": height,
            "pixel_area_m2": pixel_m2,
            "threshold": args.threshold,
            "tile_size": tile_size,
            "overlap": overlap,
            "tiles_processed": tile_counter,
            "valid_pixels": valid_pixels,
            "burned_pixels": burned_pixels,
            "valid_area_ha": valid_area_ha,
            "burned_area_ha": burned_area_ha,
            "burned_fraction": burned_fraction,
        }

        # ====================================================
        # Optional comparison with reference mask
        # ====================================================

        if args.reference:

            with rasterio.open(
                args.reference
            ) as ref_ds:

                if (
                    ref_ds.width != width
                    or ref_ds.height != height
                    or ref_ds.transform
                    != pre_ds.transform
                    or ref_ds.crs
                    != pre_ds.crs
                ):
                    raise RuntimeError(
                        "Reference mask is not aligned "
                        "with PRE/POST."
                    )

                reference = (
                    ref_ds.read(1)
                    .astype(np.uint8)
                )

            reference_metrics = (
                compute_reference_metrics(
                    mask_out,
                    reference,
                    pixel_m2,
                )
            )

            report[
                "reference_comparison"
            ] = reference_metrics

        report_path = (
            out_dir
            / "surtr_area_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
        )

        print()
        print("=" * 72)
        print("SURTR LARGE-AREA RESULT")
        print("=" * 72)

        print(
            f"Valid area : "
            f"{valid_area_ha:,.2f} ha"
        )

        print(
            f"Burned area: "
            f"{burned_area_ha:,.2f} ha"
        )

        print(
            f"Burned frac: "
            f"{100.0 * burned_fraction:.2f}%"
        )

        if args.reference:

            m = report[
                "reference_comparison"
            ]

            print()
            print("=" * 72)
            print("REFERENCE COMPARISON")
            print("=" * 72)

            print(
                f"Evaluation area : "
                f"{m['evaluation_area_ha']:,.2f} ha"
            )

            print(
                f"Predicted burned: "
                f"{m['predicted_burned_area_ha']:,.2f} ha"
            )

            print(
                f"Reference burned: "
                f"{m['reference_burned_area_ha']:,.2f} ha"
            )

            print(
                f"Difference      : "
                f"{m['difference_ha']:+,.2f} ha"
            )

            print(
                f"Abs. rel. error : "
                f"{m['absolute_relative_error_percent']:.2f}%"
            )

            print()
            print(
                f"IoU       : "
                f"{m['iou']:.4f}"
            )

            print(
                f"F1        : "
                f"{m['f1']:.4f}"
            )

            print(
                f"Precision : "
                f"{m['precision']:.4f}"
            )

            print(
                f"Recall    : "
                f"{m['recall']:.4f}"
            )

        print()
        print(
            "Probability:",
            probability_path,
        )

        print(
            "Mask:",
            mask_path,
        )

        print(
            "Report:",
            report_path,
        )


if __name__ == "__main__":
    main()

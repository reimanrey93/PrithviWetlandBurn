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
from rasterio.features import shapes
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

CONFIG_TMP = Path("/tmp/surtr_negative_control.yaml")


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
            "B2,B3,B4,B8A,B11,B12."
        )


def blending_window(size):
    h = np.hanning(size)

    weight = np.outer(h, h)

    if weight.max() > 0:
        weight /= weight.max()

    weight = 0.05 + 0.95 * weight

    return weight.astype(np.float32)


def pixel_area_m2(dataset):
    if dataset.crs is None:
        raise RuntimeError("Raster has no CRS.")

    if dataset.crs.is_geographic:
        raise RuntimeError(
            "Raster is in degrees. Reproject first."
        )

    t = dataset.transform

    return abs(
        t.a * t.e - t.b * t.d
    )


def connected_components(mask):
    """
    Returns pixel sizes of connected components.

    Uses scipy if available.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return []

    labels, n = ndimage.label(mask)

    if n == 0:
        return []

    sizes = np.bincount(
        labels.ravel()
    )

    # remove background
    return sizes[1:].tolist()


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
        print("SURTR NEGATIVE CONTROL")
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
            "Resolution:",
            pre_ds.res,
        )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="surtr_negative_"
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

        probability = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        valid_final = (
            (weight_sum > 0)
            & (valid_seen > 0)
        )

        probability[
            valid_final
        ] = (
            prob_sum[valid_final]
            / weight_sum[valid_final]
        )

        burned = (
            valid_final
            & (
                probability
                >= args.threshold
            )
        )

        pixel_m2 = pixel_area_m2(
            pre_ds
        )

        valid_pixels = int(
            valid_final.sum()
        )

        burned_pixels = int(
            burned.sum()
        )

        valid_area_ha = (
            valid_pixels
            * pixel_m2
            / 10000.0
        )

        burned_area_ha = (
            burned_pixels
            * pixel_m2
            / 10000.0
        )

        burned_fraction = (
            burned_area_ha
            / valid_area_ha
            if valid_area_ha > 0
            else float("nan")
        )

        valid_probs = probability[
            valid_final
        ]

        mean_prob = float(
            valid_probs.mean()
        )

        median_prob = float(
            np.median(valid_probs)
        )

        p95_prob = float(
            np.percentile(
                valid_probs,
                95,
            )
        )

        p99_prob = float(
            np.percentile(
                valid_probs,
                99,
            )
        )

        components = connected_components(
            burned
        )

        if components:
            largest_component_pixels = int(
                max(components)
            )
        else:
            largest_component_pixels = 0

        largest_component_ha = (
            largest_component_pixels
            * pixel_m2
            / 10000.0
        )

        # --------------------------------------------
        # Save probability
        # --------------------------------------------

        profile = pre_ds.profile.copy()

        profile.update(
            count=1,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        probability_path = (
            out_dir
            / "negative_control_probability.tif"
        )

        prob_profile = profile.copy()

        prob_profile.update(
            dtype="float32",
            nodata=-9999.0,
        )

        prob_out = probability.copy()

        prob_out[
            ~valid_final
        ] = -9999.0

        with rasterio.open(
            probability_path,
            "w",
            **prob_profile,
        ) as dst:
            dst.write(
                prob_out,
                1,
            )

        # --------------------------------------------
        # Save mask
        # --------------------------------------------

        mask_path = (
            out_dir
            / "negative_control_mask.tif"
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
            "threshold": args.threshold,
            "tile_size": tile_size,
            "overlap": overlap,

            "valid_area_ha": valid_area_ha,
            "predicted_burned_area_ha":
                burned_area_ha,

            "predicted_burned_fraction":
                burned_fraction,

            "mean_burn_probability":
                mean_prob,

            "median_burn_probability":
                median_prob,

            "p95_burn_probability":
                p95_prob,

            "p99_burn_probability":
                p99_prob,

            "largest_component_ha":
                largest_component_ha,

            "tiles_processed":
                tile_counter,
        }

        report_path = (
            out_dir
            / "negative_control_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
        )

        print()
        print("=" * 72)
        print("NEGATIVE CONTROL RESULT")
        print("=" * 72)

        print(
            f"Valid area           : "
            f"{valid_area_ha:,.2f} ha"
        )

        print(
            f"Predicted burned     : "
            f"{burned_area_ha:,.2f} ha"
        )

        print(
            f"Burned fraction      : "
            f"{100.0 * burned_fraction:.4f}%"
        )

        print()
        print(
            f"Mean probability     : "
            f"{mean_prob:.4f}"
        )

        print(
            f"Median probability   : "
            f"{median_prob:.4f}"
        )

        print(
            f"P95 probability      : "
            f"{p95_prob:.4f}"
        )

        print(
            f"P99 probability      : "
            f"{p99_prob:.4f}"
        )

        print()
        print(
            f"Largest component    : "
            f"{largest_component_ha:,.2f} ha"
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

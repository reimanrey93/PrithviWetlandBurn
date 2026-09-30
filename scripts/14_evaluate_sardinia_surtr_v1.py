#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sardinia_surtr_dataset import SardiniaSURTRDataset
from src.surtr_bitemporal import SURTRBiTemporal
from terratorch.cli_tools import LightningInferenceModel


# ============================================================
# Frozen protocol
# ============================================================

MANIFEST = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/"
    "surtr_sardinia_v1_FROZEN.csv"
)

EXPECTED_SHA256 = (
    "5df424b2210b5db955275d795f13aeb81fd1b92eb9365baa39812612b7a9ec27"
)

CONFIG_ORIGINAL = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/configs/"
    "finetune_prithvi_burnscar_v2.yaml"
)

PRITHVI_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/models/"
    "prithvi_burnscars/"
    "Prithvi_EO_V2_300M_BurnScars.pt"
)

BASELINE_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "checkpoints/sardinia_surtr_v1/"
    "baseline/best.pt"
)

SURTR_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "checkpoints/sardinia_surtr_v1/"
    "surtr/best.pt"
)

CONFIG_TMP = Path(
    "/tmp/surtr_sardinia_eval.yaml"
)

OUTPUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "evaluation/sardinia_surtr_v1"
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"


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


# ============================================================
# Utilities
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def normalize(x):
    means = MEANS.to(
        device=x.device,
        dtype=x.dtype,
    )

    stds = STDS.to(
        device=x.device,
        dtype=x.dtype,
    )

    return (x - means) / stds


def load_base_model():
    cfg = yaml.safe_load(
        CONFIG_ORIGINAL.read_text()
    )

    # Compatibility with current Lightning/TerraTorch.
    cfg.pop("lr_scheduler", None)

    CONFIG_TMP.write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
        )
    )

    old_argv = sys.argv[:]

    try:
        sys.argv = [sys.argv[0]]

        loaded = (
            LightningInferenceModel.from_config(
                str(CONFIG_TMP),
                str(PRITHVI_CHECKPOINT),
            )
        )

    finally:
        sys.argv = old_argv

    return loaded.model.model


def load_checkpoint(path: Path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def confusion_from_prediction(
    prediction,
    target,
    valid,
):
    p = prediction[valid]
    t = target[valid]

    tp = int(
        ((p == 1) & (t == 1)).sum()
    )

    fp = int(
        ((p == 1) & (t == 0)).sum()
    )

    fn = int(
        ((p == 0) & (t == 1)).sum()
    )

    tn = int(
        ((p == 0) & (t == 0)).sum()
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def add_confusion(a, b):
    for key in (
        "tp",
        "fp",
        "fn",
        "tn",
    ):
        a[key] += b[key]


def metrics(c):
    tp = c["tp"]
    fp = c["fp"]
    fn = c["fn"]
    tn = c["tn"]

    eps = 1e-12

    precision = (
        tp / (tp + fp + eps)
    )

    recall = (
        tp / (tp + fn + eps)
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall + eps)
    )

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    accuracy = (
        (tp + tn)
        / (
            tp
            + fp
            + fn
            + tn
            + eps
        )
    )

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def row_from_confusion(
    model_name,
    c,
    **extra,
):
    m = metrics(c)

    return {
        **extra,
        "model": model_name,
        "tp": c["tp"],
        "fp": c["fp"],
        "fn": c["fn"],
        "tn": c["tn"],
        **m,
    }


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Model construction
# ============================================================

def build_baseline(device):
    model = load_base_model()

    checkpoint = load_checkpoint(
        BASELINE_CHECKPOINT
    )

    if (
        checkpoint.get("mode")
        != "baseline"
    ):
        raise RuntimeError(
            "Baseline checkpoint mode mismatch."
        )

    if (
        checkpoint.get(
            "manifest_sha256"
        )
        != EXPECTED_SHA256
    ):
        raise RuntimeError(
            "Baseline manifest SHA mismatch."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def build_surtr(device):
    base_model = load_base_model()

    model = SURTRBiTemporal(
        pixel_model=base_model,
        freeze_encoder=True,
        freeze_neck=True,
        train_decoder=True,
    )

    checkpoint = load_checkpoint(
        SURTR_CHECKPOINT
    )

    if (
        checkpoint.get("mode")
        != "surtr"
    ):
        raise RuntimeError(
            "SURTR checkpoint mode mismatch."
        )

    if (
        checkpoint.get(
            "manifest_sha256"
        )
        != EXPECTED_SHA256
    ):
        raise RuntimeError(
            "SURTR manifest SHA mismatch."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# Forward
# ============================================================

def forward_baseline(
    model,
    post,
):
    output = model(post)

    if hasattr(
        output,
        "output",
    ):
        output = output.output

    return output


def forward_surtr(
    model,
    pre,
    post,
):
    return model(
        pre=pre,
        post=post,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    digest = sha256(MANIFEST)

    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {digest}"
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device("cuda")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TEST only
    # --------------------------------------------------------

    test_ds = SardiniaSURTRDataset(
        MANIFEST,
        "test",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    baseline, baseline_ckpt = (
        build_baseline(device)
    )

    surtr, surtr_ckpt = (
        build_surtr(device)
    )

    print("=" * 72)
    print("SARDINIA SURTR V1 — FINAL TEST")
    print("=" * 72)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Manifest SHA256:",
        digest,
    )

    print(
        "TEST patches:",
        len(test_ds),
    )

    print(
        "Baseline checkpoint epoch:",
        baseline_ckpt["epoch"],
    )

    print(
        "Baseline checkpoint val loss:",
        baseline_ckpt["val_loss"],
    )

    print(
        "SURTR checkpoint epoch:",
        surtr_ckpt["epoch"],
    )

    print(
        "SURTR checkpoint val loss:",
        surtr_ckpt["val_loss"],
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    pooled = {
        "baseline": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        },
        "surtr": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        },
    }

    events = {
        "baseline": defaultdict(
            lambda: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
        ),
        "surtr": defaultdict(
            lambda: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
        ),
    }

    patch_rows = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for batch in test_loader:

            pre = batch["pre"].to(
                device,
                non_blocking=True,
            )

            post = batch["post"].to(
                device,
                non_blocking=True,
            )

            target = batch["mask"].to(
                device,
                non_blocking=True,
            )

            valid = batch["valid"].to(
                device,
                non_blocking=True,
            )

            pre = normalize(pre)
            post = normalize(post)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                baseline_logits = (
                    forward_baseline(
                        baseline,
                        post,
                    )
                )

                surtr_logits = (
                    forward_surtr(
                        surtr,
                        pre,
                        post,
                    )
                )

            baseline_pred = (
                baseline_logits
                .argmax(dim=1)
            )

            surtr_pred = (
                surtr_logits
                .argmax(dim=1)
            )

            batch_size = target.shape[0]

            for i in range(
                batch_size
            ):
                event_id = (
                    batch["event_id"][i]
                )

                patch_id = (
                    batch["patch_id"][i]
                )

                t = (
                    target[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                v = (
                    valid[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                bp = (
                    baseline_pred[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                sp = (
                    surtr_pred[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                c_base = (
                    confusion_from_prediction(
                        bp,
                        t,
                        v,
                    )
                )

                c_surtr = (
                    confusion_from_prediction(
                        sp,
                        t,
                        v,
                    )
                )

                add_confusion(
                    pooled["baseline"],
                    c_base,
                )

                add_confusion(
                    pooled["surtr"],
                    c_surtr,
                )

                add_confusion(
                    events["baseline"][
                        event_id
                    ],
                    c_base,
                )

                add_confusion(
                    events["surtr"][
                        event_id
                    ],
                    c_surtr,
                )

                base_row = (
                    row_from_confusion(
                        "baseline",
                        c_base,
                        event_id=event_id,
                        patch_id=patch_id,
                        valid_pixels=int(
                            v.sum()
                        ),
                    )
                )

                surtr_row = (
                    row_from_confusion(
                        "surtr",
                        c_surtr,
                        event_id=event_id,
                        patch_id=patch_id,
                        valid_pixels=int(
                            v.sum()
                        ),
                    )
                )

                patch_rows.extend([
                    base_row,
                    surtr_row,
                ])

                # Compact prediction package.
                # 0 = invalid
                # 1 = valid unburned
                # 2 = valid burned
                def encode(arr):
                    out = np.zeros(
                        arr.shape,
                        dtype=np.uint8,
                    )

                    out[
                        v & (arr == 0)
                    ] = 1

                    out[
                        v & (arr == 1)
                    ] = 2

                    return out

                np.savez_compressed(
                    PREDICTION_DIR
                    / f"{patch_id}.npz",
                    target=encode(t),
                    baseline=encode(bp),
                    surtr=encode(sp),
                    valid=v.astype(
                        np.uint8
                    ),
                    event_id=np.array(
                        event_id
                    ),
                    patch_id=np.array(
                        patch_id
                    ),
                )

    # ========================================================
    # Pooled metrics
    # ========================================================

    pooled_rows = []

    for model_name in (
        "baseline",
        "surtr",
    ):
        pooled_rows.append(
            row_from_confusion(
                model_name,
                pooled[model_name],
                scope="pixel_pooled",
            )
        )

    write_csv(
        OUTPUT_DIR
        / "test_pooled_metrics.csv",
        pooled_rows,
    )

    # ========================================================
    # Event metrics
    # ========================================================

    event_rows = []

    event_ids = sorted(
        set(
            events["baseline"].keys()
        )
        | set(
            events["surtr"].keys()
        )
    )

    for event_id in event_ids:

        for model_name in (
            "baseline",
            "surtr",
        ):
            event_rows.append(
                row_from_confusion(
                    model_name,
                    events[model_name][
                        event_id
                    ],
                    event_id=event_id,
                )
            )

    write_csv(
        OUTPUT_DIR
        / "test_event_metrics.csv",
        event_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "test_patch_metrics.csv",
        patch_rows,
    )

    # ========================================================
    # Event-macro metrics
    # ========================================================

    macro = {}

    for model_name in (
        "baseline",
        "surtr",
    ):
        model_event_metrics = [
            metrics(
                events[model_name][eid]
            )
            for eid in event_ids
        ]

        macro[model_name] = {
            key: float(
                np.mean([
                    m[key]
                    for m
                    in model_event_metrics
                ])
            )
            for key in (
                "iou",
                "f1",
                "precision",
                "recall",
                "accuracy",
            )
        }

    # ========================================================
    # Console report
    # ========================================================

    base = metrics(
        pooled["baseline"]
    )

    st = metrics(
        pooled["surtr"]
    )

    print()
    print("=" * 72)
    print("PIXEL-POOLED TEST")
    print("=" * 72)

    print(
        f"{'MODEL':12s} "
        f"{'IoU':>8s} "
        f"{'F1':>8s} "
        f"{'PREC':>8s} "
        f"{'RECALL':>8s}"
    )

    print(
        f"{'Baseline':12s} "
        f"{base['iou']:8.4f} "
        f"{base['f1']:8.4f} "
        f"{base['precision']:8.4f} "
        f"{base['recall']:8.4f}"
    )

    print(
        f"{'SURTR':12s} "
        f"{st['iou']:8.4f} "
        f"{st['f1']:8.4f} "
        f"{st['precision']:8.4f} "
        f"{st['recall']:8.4f}"
    )

    print(
        f"{'Delta':12s} "
        f"{st['iou']-base['iou']:+8.4f} "
        f"{st['f1']-base['f1']:+8.4f} "
        f"{st['precision']-base['precision']:+8.4f} "
        f"{st['recall']-base['recall']:+8.4f}"
    )

    print()
    print("=" * 72)
    print("EVENT-MACRO TEST")
    print("=" * 72)

    for model_name in (
        "baseline",
        "surtr",
    ):
        m = macro[model_name]

        print(
            f"{model_name:12s} "
            f"IoU={m['iou']:.4f} "
            f"F1={m['f1']:.4f} "
            f"P={m['precision']:.4f} "
            f"R={m['recall']:.4f}"
        )

    print()
    print("=" * 72)
    print("PER EVENT")
    print("=" * 72)

    for eid in event_ids:
        bm = metrics(
            events["baseline"][eid]
        )

        sm = metrics(
            events["surtr"][eid]
        )

        print(eid)

        print(
            f"  A0  IoU={bm['iou']:.4f} "
            f"F1={bm['f1']:.4f}"
        )

        print(
            f"  A1  IoU={sm['iou']:.4f} "
            f"F1={sm['f1']:.4f}"
        )

        print(
            f"  Δ   IoU="
            f"{sm['iou']-bm['iou']:+.4f} "
            f"F1="
            f"{sm['f1']-bm['f1']:+.4f}"
        )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {
        "protocol": {
            "manifest": str(
                MANIFEST
            ),
            "manifest_sha256": digest,
            "test_patches": len(
                test_ds
            ),
            "test_events": len(
                event_ids
            ),
        },
        "checkpoints": {
            "baseline": {
                "path": str(
                    BASELINE_CHECKPOINT
                ),
                "epoch": int(
                    baseline_ckpt[
                        "epoch"
                    ]
                ),
                "val_loss": float(
                    baseline_ckpt[
                        "val_loss"
                    ]
                ),
            },
            "surtr": {
                "path": str(
                    SURTR_CHECKPOINT
                ),
                "epoch": int(
                    surtr_ckpt[
                        "epoch"
                    ]
                ),
                "val_loss": float(
                    surtr_ckpt[
                        "val_loss"
                    ]
                ),
            },
        },
        "pixel_pooled": {
            "baseline": base,
            "surtr": st,
            "delta_surtr_minus_baseline": {
                key: (
                    st[key]
                    - base[key]
                )
                for key in base
            },
        },
        "event_macro": macro,
    }

    (
        OUTPUT_DIR
        / "test_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("FINAL TEST COMPLETE")
    print("=" * 72)

    print(
        "Results:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    evaluate()

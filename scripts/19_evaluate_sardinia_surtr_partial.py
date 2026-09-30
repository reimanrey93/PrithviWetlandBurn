#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sardinia_surtr_dataset import SardiniaSURTRDataset
from src.surtr_bitemporal_adapt import SURTRBiTemporalAdapt
from terratorch.cli_tools import LightningInferenceModel


# ============================================================
# Frozen protocol: EXACTLY the same test manifest
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

LORA_CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "checkpoints/sardinia_surtr_adaptation/"
    "partial_last4/best.pt"
)

CONFIG_TMP = Path(
    "/tmp/surtr_sardinia_lora_eval.yaml"
)

OUTPUT_DIR = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "evaluation/sardinia_surtr_partial_last4"
)


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


def sha256(path):
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
        x.device,
        x.dtype,
    )

    stds = STDS.to(
        x.device,
        x.dtype,
    )

    return (x - means) / stds


def load_base_model():

    cfg = yaml.safe_load(
        CONFIG_ORIGINAL.read_text()
    )

    cfg.pop(
        "lr_scheduler",
        None,
    )

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


def metrics_from_confusion(
    tp,
    fp,
    fn,
    tn,
):
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


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    digest = sha256(MANIFEST)

    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen manifest SHA mismatch."
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device("cuda")

    # ========================================================
    # Same held-out test dataset
    # ========================================================

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

    # ========================================================
    # Build EXACT LoRA architecture used during training
    # ========================================================

    base_model = load_base_model()

    model = SURTRBiTemporalAdapt(
        pixel_model=base_model,
        adaptation="partial",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        partial_last_n_blocks=4,
        freeze_neck=True,
        train_decoder=True,
    )

    checkpoint = torch.load(
        LORA_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get(
            "manifest_sha256"
        )
        != EXPECTED_SHA256
    ):
        raise RuntimeError(
            "LoRA checkpoint manifest SHA mismatch."
        )

    if (
        checkpoint.get(
            "adaptation"
        )
        != "partial"
    ):
        raise RuntimeError(
            "Checkpoint is not a partial fine-tuning adaptation."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("=" * 72)
    print("SARDINIA SURTR PARTIAL FT — FINAL HELD-OUT TEST")
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
        "Checkpoint:",
        LORA_CHECKPOINT,
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint val loss:",
        checkpoint["val_loss"],
    )

    print(
        "Adaptation:",
        checkpoint.get(
            "adaptation_info"
        ),
    )

    # ========================================================
    # Pixel-pooled confusion
    # ========================================================

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    event_ids = set()

    with torch.inference_mode():

        for batch in test_loader:

            pre = normalize(
                batch["pre"].to(
                    device,
                    non_blocking=True,
                )
            )

            post = normalize(
                batch["post"].to(
                    device,
                    non_blocking=True,
                )
            )

            target = batch["mask"].to(
                device,
                non_blocking=True,
            )

            valid = batch["valid"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    pre=pre,
                    post=post,
                )

            pred = logits.argmax(
                dim=1
            )

            p = pred[valid]
            t = target[valid]

            tp += int(
                ((p == 1) & (t == 1))
                .sum()
                .item()
            )

            fp += int(
                ((p == 1) & (t == 0))
                .sum()
                .item()
            )

            fn += int(
                ((p == 0) & (t == 1))
                .sum()
                .item()
            )

            tn += int(
                ((p == 0) & (t == 0))
                .sum()
                .item()
            )

            for eid in batch["event_id"]:
                event_ids.add(
                    str(eid)
                )

    result = metrics_from_confusion(
        tp,
        fp,
        fn,
        tn,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model": "siamese_partial_last4",
        "protocol": {
            "manifest": str(MANIFEST),
            "manifest_sha256": digest,
            "test_patches": len(test_ds),
            "test_events": len(event_ids),
        },
        "checkpoint": {
            "path": str(LORA_CHECKPOINT),
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "adaptation_info": checkpoint.get(
                "adaptation_info"
            ),
        },
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "pixel_pooled": result,
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
    print("PIXEL-POOLED HELD-OUT TEST")
    print("=" * 72)

    print(
        f"{'MODEL':20s} "
        f"{'IoU':>8s} "
        f"{'F1':>8s} "
        f"{'PREC':>8s} "
        f"{'RECALL':>8s} "
        f"{'ACC':>8s}"
    )

    print(
        f"{'Siamese + Partial FT':20s} "
        f"{result['iou']:8.4f} "
        f"{result['f1']:8.4f} "
        f"{result['precision']:8.4f} "
        f"{result['recall']:8.4f} "
        f"{result['accuracy']:8.4f}"
    )

    print()
    print("Confusion:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print()
    print(
        "Saved:",
        OUTPUT_DIR / "test_summary.json"
    )


if __name__ == "__main__":
    main()

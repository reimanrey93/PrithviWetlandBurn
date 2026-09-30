#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sardinia_surtr_dataset import (
    SardiniaSURTRDataset,
    make_event_balanced_sampler,
)
from src.surtr_bitemporal import (
    SURTRBiTemporal,
    parameter_summary,
)

from terratorch.cli_tools import LightningInferenceModel


# ============================================================
# Fixed protocol
# ============================================================

MANIFEST = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/"
    "surtr_sardinia_v1_FROZEN.csv"
)

EXPECTED_MANIFEST_SHA256 = (
    "5df424b2210b5db955275d795f13aeb81fd1b92eb9365baa39812612b7a9ec27"
)

CONFIG_ORIGINAL = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/configs/"
    "finetune_prithvi_burnscar_v2.yaml"
)

CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/models/"
    "prithvi_burnscars/"
    "Prithvi_EO_V2_300M_BurnScars.pt"
)

CONFIG_TMP = Path(
    "/tmp/surtr_sardinia_train.yaml"
)

MEANS = torch.tensor(
    [
        0.052376806868,
        0.070576639747,
        0.081104305106,
        0.177237257070,
        0.225423962965,
        0.174360119430,
    ],
    dtype=torch.float32,
).view(1, 6, 1, 1)

STDS = torch.tensor(
    [
        0.021721302751,
        0.026402486035,
        0.034470597693,
        0.091198580301,
        0.103103430325,
        0.088007767199,
    ],
    dtype=torch.float32,
).view(1, 6, 1, 1)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_base_model():
    cfg = yaml.safe_load(
        CONFIG_ORIGINAL.read_text()
    )

    # Current Lightning/TerraTorch rejects the old
    # ReduceLROnPlateau specification.
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
                str(CHECKPOINT),
            )
        )

    finally:
        sys.argv = old_argv

    return loaded.model.model


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


def masked_ce(
    logits,
    target,
    valid,
):
    """
    Cross entropy only over supervised pixels.
    """

    loss = F.cross_entropy(
        logits,
        target,
        reduction="none",
    )

    valid = valid.float()

    denominator = valid.sum().clamp_min(1.0)

    return (
        loss * valid
    ).sum() / denominator


def make_loaders():

    train_ds = SardiniaSURTRDataset(
        MANIFEST,
        "train",
    )

    val_ds = SardiniaSURTRDataset(
        MANIFEST,
        "val",
    )

    sampler = make_event_balanced_sampler(
        train_ds,
        seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["baseline", "surtr"],
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One forward/backward/update only.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Frozen protocol guard
    # --------------------------------------------------------

    digest = sha256(MANIFEST)

    if digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual:   {digest}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device("cuda")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader = (
        make_loaders()
    )

    # --------------------------------------------------------
    # Official model
    # --------------------------------------------------------

    base_model = load_base_model()

    if args.mode == "baseline":

        model = base_model

        # Fair first comparison:
        # train decoder/head, freeze encoder/neck.
        for p in model.encoder.parameters():
            p.requires_grad = False

        for p in model.neck.parameters():
            p.requires_grad = False

        for p in model.decoder.parameters():
            p.requires_grad = True

        for p in model.head.parameters():
            p.requires_grad = True

    else:

        model = SURTRBiTemporal(
            pixel_model=base_model,
            freeze_encoder=True,
            freeze_neck=True,
            train_decoder=True,
        )

    model = model.to(device)

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-5,
        weight_decay=0.01,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("=" * 72)
    print("SURTR SARDINIA V1")
    print("=" * 72)

    print("Mode:", args.mode)
    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "Manifest SHA256:",
        digest,
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    print(
        "Val batches:",
        len(val_loader),
    )

    if args.mode == "surtr":

        print(
            "Parameters:",
            parameter_summary(model),
        )

    else:

        total = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_n = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            "Total parameters:",
            f"{total:,}",
        )

        print(
            "Trainable parameters:",
            f"{trainable_n:,}",
        )

    # --------------------------------------------------------
    # Smoke forward/backward
    # --------------------------------------------------------

    batch = next(
        iter(train_loader)
    )

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

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):

        if args.mode == "baseline":

            output = model(post)

            logits = (
                output.output
                if hasattr(output, "output")
                else output
            )

        else:

            logits = model(
                pre=pre,
                post=post,
            )

        loss = masked_ce(
            logits,
            target,
            valid,
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss.item()}"
        )

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=1.0,
    )

    optimizer.step()

    print()
    print("SMOKE RESULT")
    print(
        "PRE:",
        tuple(pre.shape),
    )

    print(
        "POST:",
        tuple(post.shape),
    )

    print(
        "LOGITS:",
        tuple(logits.shape),
    )

    print(
        "TARGET:",
        tuple(target.shape),
    )

    print(
        "VALID FRACTION:",
        float(
            valid.float().mean()
        ),
    )

    print(
        "LOSS:",
        float(loss.detach()),
    )

    print(
        "GRAD NORM:",
        float(grad_norm),
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()

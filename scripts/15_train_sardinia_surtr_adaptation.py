#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
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
from src.surtr_bitemporal_adapt import (
    SURTRBiTemporalAdapt,
    parameter_summary,
    trainable_parameter_groups,
)
from terratorch.cli_tools import LightningInferenceModel


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

CHECKPOINT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/models/"
    "prithvi_burnscars/"
    "Prithvi_EO_V2_300M_BurnScars.pt"
)

CONFIG_TMP = Path("/tmp/surtr_sardinia_train.yaml")

OUTPUT_ROOT = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/"
    "checkpoints/sardinia_surtr_adaptation"
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
    with open(path, "rb") as f:
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
        loaded = LightningInferenceModel.from_config(
            str(CONFIG_TMP),
            str(CHECKPOINT),
        )
    finally:
        sys.argv = old_argv

    return loaded.model.model


def normalize(x):
    means = MEANS.to(x.device, x.dtype)
    stds = STDS.to(x.device, x.dtype)
    return (x - means) / stds


def masked_ce(logits, target, valid):
    loss = F.cross_entropy(
        logits,
        target,
        reduction="none",
    )

    v = valid.float()

    return (
        (loss * v).sum()
        / v.sum().clamp_min(1.0)
    )


def confusion(logits, target, valid):
    pred = logits.argmax(dim=1)

    p = pred[valid]
    t = target[valid]

    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    tn = ((p == 0) & (t == 0)).sum().item()

    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = (
        2 * precision * recall
        / (precision + recall + eps)
    )

    iou = tp / (tp + fp + fn + eps)

    accuracy = (
        (tp + tn)
        / (tp + fp + fn + tn + eps)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def make_loaders():
    train_ds = SardiniaSURTRDataset(
        MANIFEST, "train"
    )

    val_ds = SardiniaSURTRDataset(
        MANIFEST, "val"
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


def forward_model(model, pre, post):
    return model(
        pre=pre,
        post=post,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    adaptation,
    adaptation_info,
):
    torch.save(
        {
            "epoch": epoch,
            "adaptation": adaptation,
            "adaptation_info": adaptation_info,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "val_loss": val_loss,
            "manifest_sha256": (
                EXPECTED_SHA256
            ),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adaptation",
        choices=["lora", "partial"],
        required=True,
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--unfreeze-last",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--task-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    digest = sha256(MANIFEST)

    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen manifest SHA256 mismatch."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")

    train_loader, val_loader = make_loaders()

    base_model = load_base_model()

    model = SURTRBiTemporalAdapt(
        pixel_model=base_model,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        partial_last_n_blocks=args.unfreeze_last,
        freeze_neck=True,
        train_decoder=True,
    )

    model = model.to(device)

    encoder_params, task_params = (
        trainable_parameter_groups(model)
    )

    trainable = (
        encoder_params
        + task_params
    )

    optimizer_groups = []

    if encoder_params:
        optimizer_groups.append(
            {
                "params": encoder_params,
                "lr": args.encoder_lr,
                "name": "encoder",
            }
        )

    if task_params:
        optimizer_groups.append(
            {
                "params": task_params,
                "lr": args.task_lr,
                "name": "task",
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=0.01,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    )

    if args.adaptation == "lora":
        run_name = (
            f"lora_r{args.lora_rank}"
            f"_a{int(args.lora_alpha)}"
        )
    else:
        run_name = (
            f"partial_last{args.unfreeze_last}"
        )

    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        out_dir / "history.csv"
    )

    print("=" * 72)
    print("SURTR SARDINIA BACKBONE ADAPTATION")
    print("=" * 72)
    print("Adaptation:", args.adaptation)
    print(
        "Adaptation info:",
        model.adaptation_info,
    )
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
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
    print(
        "Trainable:",
        f"{sum(p.numel() for p in trainable):,}",
    )

    print(
        "Parameters:",
        parameter_summary(model),
    )

    print(
        "Encoder trainable:",
        f"{sum(p.numel() for p in encoder_params):,}",
    )

    print(
        "Task trainable:",
        f"{sum(p.numel() for p in task_params):,}",
    )

    if args.smoke:
        max_epochs = 1
        max_train_batches = 1
        max_val_batches = 1
    else:
        max_epochs = args.epochs
        max_train_batches = None
        max_val_batches = None

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_accuracy",
        "encoder_lr",
        "task_lr",
        "seconds",
    ]

    with history_path.open(
        "w",
        newline="",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=fields,
        ).writeheader()

    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        start = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss_sum = 0.0
        train_valid_pixels = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for batch_idx, batch in enumerate(
            train_loader,
            start=1,
        ):
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
                logits = forward_model(
                    model,
                    pre,
                    post,
                )

                loss = masked_ce(
                    logits,
                    target,
                    valid,
                )

                # Effective batch 8:
                # physical batch 2 x accumulation 4
                scaled_loss = loss / 4.0

            scaled_loss.backward()

            do_step = (
                batch_idx % 4 == 0
                or batch_idx
                == len(train_loader)
            )

            if do_step:
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            n_valid = int(
                valid.sum().item()
            )

            train_loss_sum += (
                float(loss.detach())
                * n_valid
            )

            train_valid_pixels += n_valid

            if (
                max_train_batches is not None
                and batch_idx
                >= max_train_batches
            ):
                # Smoke must still perform update.
                if not do_step:
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad(
                        set_to_none=True
                    )
                break

        train_loss = (
            train_loss_sum
            / max(train_valid_pixels, 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_sum = 0.0
        val_valid_pixels = 0

        tp = fp = fn = tn = 0

        with torch.no_grad():

            for batch_idx, batch in enumerate(
                val_loader,
                start=1,
            ):
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
                    logits = forward_model(
                    model,
                    pre,
                    post,
                )

                    loss = masked_ce(
                        logits,
                        target,
                        valid,
                    )

                n_valid = int(
                    valid.sum().item()
                )

                val_loss_sum += (
                    float(loss)
                    * n_valid
                )

                val_valid_pixels += n_valid

                c = confusion(
                    logits,
                    target,
                    valid,
                )

                tp += c[0]
                fp += c[1]
                fn += c[2]
                tn += c[3]

                if (
                    max_val_batches is not None
                    and batch_idx
                    >= max_val_batches
                ):
                    break

        val_loss = (
            val_loss_sum
            / max(val_valid_pixels, 1)
        )

        metrics = metrics_from_confusion(
            tp,
            fp,
            fn,
            tn,
        )

        scheduler.step(val_loss)

        lr_map = {
            group.get(
                "name",
                f"group_{i}",
            ): group["lr"]
            for i, group in enumerate(
                optimizer.param_groups
            )
        }

        encoder_lr = lr_map.get(
            "encoder",
            float("nan"),
        )

        task_lr = lr_map.get(
            "task",
            float("nan"),
        )

        seconds = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": metrics["iou"],
            "val_f1": metrics["f1"],
            "val_precision": (
                metrics["precision"]
            ),
            "val_recall": (
                metrics["recall"]
            ),
            "val_accuracy": (
                metrics["accuracy"]
            ),
            "encoder_lr": encoder_lr,
            "task_lr": task_lr,
            "seconds": seconds,
        }

        with history_path.open(
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )
            writer.writerow(row)

        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            args.adaptation,

            model.adaptation_info,
        )

        improved = (
            val_loss < best_loss
        )

        if improved:
            best_loss = val_loss
            bad_epochs = 0

            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                args.adaptation,

                model.adaptation_info,
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"IoU {metrics['iou']:.4f} | "
            f"F1 {metrics['f1']:.4f} | "
            f"EncLR {encoder_lr:.2e} | "
            f"TaskLR {task_lr:.2e} | "
            f"{seconds:.1f}s"
        )

        if args.smoke:
            print("SMOKE PASS")
            break

        if bad_epochs >= args.patience:
            print(
                "Early stopping:",
                bad_epochs,
                "epochs without improvement.",
            )
            break

    summary = {
        "adaptation": args.adaptation,
        "adaptation_info": model.adaptation_info,
        "parameter_summary": parameter_summary(model),
        "best_val_loss": best_loss,
        "manifest_sha256": digest,
        "max_epochs": args.epochs,
        "patience": args.patience,
    }

    (
        out_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("Best val loss:", best_loss)
    print(
        "Best checkpoint:",
        out_dir / "best.pt",
    )
    print(
        "History:",
        history_path,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

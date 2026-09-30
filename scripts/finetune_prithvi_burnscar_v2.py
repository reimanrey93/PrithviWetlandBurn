#!/usr/bin/env python3
"""
Fine-tune Prithvi EO v2 300M BurnScars on Santa Fe.

Loads the official Lightning checkpoint through the official TerraTorch helper,
but starts a fresh Trainer (no original optimizer or epoch resume).
"""

import argparse
import sys
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from terratorch.cli_tools import LightningInferenceModel


def load_official_model(config_path: str, checkpoint_path: str):
    # LightningInferenceModel internally builds a LightningCLI. Hide this
    # script's flags so it sees only the config supplied programmatically.
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        loaded = LightningInferenceModel.from_config(config_path, checkpoint_path)
    finally:
        sys.argv = old_argv
    return loaded.model, loaded.datamodule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/DATA/ralfonso/prithvi_wetlandburn/configs/finetune_prithvi_burnscar_v2.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="/DATA/ralfonso/prithvi_wetlandburn/models/prithvi_burnscars/Prithvi_EO_V2_300M_BurnScars.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="/DATA/ralfonso/prithvi_wetlandburn/checkpoints/prithvi_burnscar_santafe_v1",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not config.exists():
        raise FileNotFoundError(config)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    torch.set_float32_matmul_precision("high")
    seed_everything(42, workers=True)
    model, datamodule = load_official_model(str(config), str(checkpoint))

    output_dir.mkdir(parents=True, exist_ok=True)
#    best = ModelCheckpoint(
#        dirpath=output_dir,
#        filename="prithvi-santafe-{epoch:02d}-{val_loss:.4f}",
#        monitor="val/loss",
#        mode="min",
#        save_top_k=1,
#        save_last=True,
#        auto_insert_metric_name=False,
#    )


    best = ModelCheckpoint(
        dirpath=output_dir,
        filename="prithvi-santafe-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    max_epochs = 1 if args.dry_run else args.epochs
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        callbacks=[
            best,
            EarlyStopping(monitor="val/loss", mode="min", patience=7),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print("=== PRITHVI SANTA FE FINE-TUNING ===")
    print("Base checkpoint:", checkpoint)
    print("Output:", output_dir)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", max_epochs)
    print("Batch: 2 | accumulation: 4 | effective batch: 8")
    trainer.fit(model=model, datamodule=datamodule)
    print("Best checkpoint:", best.best_model_path)
    print("Best val/loss:", best.best_model_score)


if __name__ == "__main__":
    main()

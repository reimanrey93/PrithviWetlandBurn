#!/usr/bin/env python3
"""
Fine-tune the official Prithvi BurnScars Lightning checkpoint without resuming
its original optimizer/epoch state.

It loads the pretrained model + datamodule through the official
LightningInferenceModel helper, then starts a NEW Lightning Trainer.
"""

import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from terratorch.cli_tools import LightningInferenceModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/DATA/ralfonso/prithvi_wetlandburn/configs/finetune_prithvi_burnscar.yaml",
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
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Run exactly one train epoch.")
    args = parser.parse_args()

    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not config.exists():
        raise FileNotFoundError(config)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en este entorno.")

    torch.set_float32_matmul_precision("high")
    seed_everything(42, workers=True)

    # Uses TerraTorch's official config/checkpoint loader, but does NOT pass
    # the checkpoint to Trainer.fit. Therefore epoch, optimizer and loop state
    # from the HLS training are not resumed.
    loaded = LightningInferenceModel.from_config(str(config), str(checkpoint))
    model = loaded.model
    datamodule = loaded.datamodule

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="prithvi-santafe-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        ckpt_callback,
        EarlyStopping(monitor="val/loss", mode="min", patience=7),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    max_epochs = 1 if args.dry_run else args.epochs
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
        deterministic=False,
    )

    print("=== PRITHVI SANTA FE FINE-TUNING ===")
    print("Config:", config)
    print("Checkpoint base:", checkpoint)
    print("Output:", output_dir)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", max_epochs)
    print("Batch (config): 2 | Gradient accumulation: 4 | Effective batch: 8")

    trainer.fit(model=model, datamodule=datamodule)

    print("\n=== FIN ===")
    print("Best checkpoint:", ckpt_callback.best_model_path)
    print("Best val/loss:", ckpt_callback.best_model_score)


if __name__ == "__main__":
    main()

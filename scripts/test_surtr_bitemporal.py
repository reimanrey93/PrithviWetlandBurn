#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.surtr_bitemporal import (
    SURTRBiTemporal,
    parameter_summary,
)

from terratorch.cli_tools import LightningInferenceModel


CONFIG_ORIGINAL = Path(
    "/DATA/ralfonso/prithvi_wetlandburn/configs/"
    "finetune_prithvi_burnscar_v2.yaml"
)

CONFIG_TMP = Path("/tmp/surtr_smoke.yaml")

CHECKPOINT = (
    "/DATA/ralfonso/prithvi_wetlandburn/models/"
    "prithvi_burnscars/"
    "Prithvi_EO_V2_300M_BurnScars.pt"
)


# ============================================================
# Temporary configuration
# ============================================================

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


# ============================================================
# Load official Prithvi BurnScars model
# ============================================================

old_argv = sys.argv[:]

try:
    sys.argv = [sys.argv[0]]

    loaded = LightningInferenceModel.from_config(
        str(CONFIG_TMP),
        CHECKPOINT,
    )

finally:
    sys.argv = old_argv


task = loaded.model
dm = loaded.datamodule

base_model = task.model


# ============================================================
# Real example from existing dataset
# ============================================================

dm.setup("fit")

batch = next(
    iter(dm.train_dataloader())
)

x = batch["image"][:1].cuda()


# ============================================================
# Official Prithvi prediction
# ============================================================

base_model = base_model.cuda().eval()

with torch.no_grad():
    base_output = base_model(x).output


# ============================================================
# SURTR model
# ============================================================

surtr = SURTRBiTemporal(
    pixel_model=base_model,
    freeze_encoder=True,
    freeze_neck=True,
    train_decoder=True,
).cuda()

surtr.eval()

with torch.no_grad():

    # PRE == POST intentionally:
    # Initial SURTR should reproduce POST-only prediction.
    surtr_output = surtr(
        pre=x,
        post=x,
    )


# ============================================================
# Diagnostics
# ============================================================

print("\n================ SURTR SMOKE TEST ================")

print(
    "Input:",
    tuple(x.shape),
)

print(
    "Base output:",
    tuple(base_output.shape),
)

print(
    "SURTR output:",
    tuple(surtr_output.shape),
)

difference = (
    base_output - surtr_output
).abs()

print(
    "Max abs difference:",
    difference.max().item(),
)

print(
    "Mean abs difference:",
    difference.mean().item(),
)


stats = parameter_summary(surtr)

print("\n================ PARAMETERS ================")

for key, value in stats.items():

    if key == "trainable_pct":
        print(
            f"{key:20s}: {value:.4f}%"
        )
    else:
        print(
            f"{key:20s}: {value:,}"
        )


adapter_params = sum(
    p.numel()
    for p in surtr.change_adapters.parameters()
)

print("\n================ CHANGE ADAPTERS ================")

print(
    "Adapter parameters:",
    f"{adapter_params:,}",
)

print("\nDONE")

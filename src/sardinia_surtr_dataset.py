from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class SardiniaSURTRDataset(Dataset):
    """
    Frozen SURTR Sardinia V1 dataset.

    Returns:
        pre:       float32 [6,H,W]
        post:      float32 [6,H,W]
        mask:      int64   [H,W], values {0,1}
        valid:     bool    [H,W]
        event_id:  str
        patch_id:  str
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str,
    ):
        self.manifest = Path(manifest)
        self.split = split

        with self.manifest.open(
            encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

        self.rows = [
            r for r in rows
            if r["split"] == split
        ]

        if not self.rows:
            raise RuntimeError(
                f"No samples for split={split}"
            )

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _read_multiband(path):
        with rasterio.open(path) as src:
            x = src.read()

        return np.asarray(
            x,
            dtype=np.float32,
        )

    @staticmethod
    def _read_single(path):
        with rasterio.open(path) as src:
            x = src.read(1)

        return x

    def __getitem__(self, idx):

        r = self.rows[idx]

        pre = self._read_multiband(
            r["pre_path"]
        )

        post = self._read_multiband(
            r["post_path"]
        )

        mask = self._read_single(
            r["mask_path"]
        ).astype(np.int64)

        valid = self._read_single(
            r["valid_path"]
        ).astype(bool)

        if pre.shape != (6, 256, 256):
            raise RuntimeError(
                f"Bad PRE shape: {r['patch_id']} "
                f"{pre.shape}"
            )

        if post.shape != (6, 256, 256):
            raise RuntimeError(
                f"Bad POST shape: {r['patch_id']} "
                f"{post.shape}"
            )

        if mask.shape != (256, 256):
            raise RuntimeError(
                f"Bad mask shape: {r['patch_id']}"
            )

        if valid.shape != (256, 256):
            raise RuntimeError(
                f"Bad valid shape: {r['patch_id']}"
            )

        return {
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "event_id": r["event_id"],
            "patch_id": r["patch_id"],
        }


def make_event_balanced_sampler(
    dataset: SardiniaSURTRDataset,
    seed: int = 42,
):
    """
    Each event receives equal total sampling probability.

    For event e with N_e patches:
        weight(patch in e) = 1 / N_e

    WeightedRandomSampler then samples len(dataset) items
    per epoch with replacement.
    """

    counts = Counter(
        r["event_id"]
        for r in dataset.rows
    )

    weights = [
        1.0 / counts[r["event_id"]]
        for r in dataset.rows
    ]

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    return sampler

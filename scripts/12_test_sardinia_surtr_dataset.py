#!/usr/bin/env python3

from collections import Counter

from torch.utils.data import DataLoader

from src.sardinia_surtr_dataset import (
    SardiniaSURTRDataset,
    make_event_balanced_sampler,
)


MANIFEST = (
    "/DATA/ralfonso/prithvi_wetlandburn/sardinia/"
    "patches_v1/manifests/"
    "surtr_sardinia_v1_FROZEN.csv"
)


def main():

    train = SardiniaSURTRDataset(
        MANIFEST,
        "train",
    )

    val = SardiniaSURTRDataset(
        MANIFEST,
        "val",
    )

    test = SardiniaSURTRDataset(
        MANIFEST,
        "test",
    )

    print("=" * 72)
    print("SURTR DATASET SMOKE TEST")
    print("=" * 72)

    print(
        "Lengths:",
        len(train),
        len(val),
        len(test),
    )

    x = train[0]

    for key in (
        "pre",
        "post",
        "mask",
        "valid",
    ):
        print(
            f"{key:6s}",
            tuple(x[key].shape),
            x[key].dtype,
        )

    print(
        "event:",
        x["event_id"],
    )

    print(
        "patch:",
        x["patch_id"],
    )

    print(
        "mask values:",
        x["mask"].unique().tolist(),
    )

    print(
        "valid fraction:",
        float(x["valid"].float().mean()),
    )

    # ----------------------------------------------------------
    # Actual DataLoader batch
    # ----------------------------------------------------------

    sampler = make_event_balanced_sampler(
        train,
        seed=42,
    )

    loader = DataLoader(
        train,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
    )

    batch = next(iter(loader))

    print()
    print("BATCH")

    print(
        "pre  :",
        tuple(batch["pre"].shape),
    )

    print(
        "post :",
        tuple(batch["post"].shape),
    )

    print(
        "mask :",
        tuple(batch["mask"].shape),
    )

    print(
        "valid:",
        tuple(batch["valid"].shape),
    )

    # ----------------------------------------------------------
    # Statistical sampler check.
    # Draw many independent sampler epochs.
    # ----------------------------------------------------------

    counts = Counter()

    for epoch in range(200):

        sampler = make_event_balanced_sampler(
            train,
            seed=42 + epoch,
        )

        for idx in sampler:
            counts[
                train.rows[idx]["event_id"]
            ] += 1

    values = list(counts.values())

    print()
    print("EVENT-BALANCED SAMPLER")
    print(
        "events:",
        len(counts),
    )
    print(
        "draws:",
        sum(values),
    )
    print(
        "min per event:",
        min(values),
    )
    print(
        "max per event:",
        max(values),
    )
    print(
        "mean:",
        sum(values) / len(values),
    )

    bonarcado = (
        "SA_2021_2021_OR_2142"
    )

    print(
        "Bonarcado draws:",
        counts[bonarcado],
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()

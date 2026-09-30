from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualChangeAdapter(nn.Module):
    """
    Multiscale residual change adapter.

    At initialization:
        output ~= post

    because the final convolution is initialized to zero.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden = max(channels // reduction, 32)

        self.adapter = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),
        )

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:

        diff = torch.abs(post - pre)

        change = torch.cat(
            [pre, post, diff],
            dim=1,
        )

        residual = self.adapter(change)

        return post + residual


class SURTRBiTemporal(nn.Module):
    """
    SURTR v0.1

    Shared Prithvi encoder and neck.
    Multiscale PRE/POST change fusion.
    Original BurnScars decoder + head.
    """

    FEATURE_CHANNELS = [256, 512, 1024, 1024]

    def __init__(
        self,
        pixel_model,
        freeze_encoder: bool = True,
        freeze_neck: bool = True,
        train_decoder: bool = True,
    ):
        super().__init__()

        self.encoder = pixel_model.encoder
        self.neck = pixel_model.neck
        self.decoder = pixel_model.decoder
        self.head = pixel_model.head

        self.freeze_encoder = freeze_encoder
        self.freeze_neck = freeze_neck

        self.change_adapters = nn.ModuleList(
            [
                ResidualChangeAdapter(c)
                for c in self.FEATURE_CHANNELS
            ]
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if freeze_neck:
            for p in self.neck.parameters():
                p.requires_grad = False

        if not train_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

            for p in self.head.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        if self.freeze_neck:
            self.neck.eval()

        return self

    def extract_pyramid(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:

        image_size = x.shape[-2:]

        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        if self.freeze_neck:
            with torch.no_grad():
                pyramid = self.neck(
                    features,
                    image_size=image_size,
                )
        else:
            pyramid = self.neck(
                features,
                image_size=image_size,
            )

        return pyramid

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:

        if pre.shape != post.shape:
            raise ValueError(
                f"PRE/POST mismatch: "
                f"{tuple(pre.shape)} vs {tuple(post.shape)}"
            )

        output_size = post.shape[-2:]

        pre_pyramid = self.extract_pyramid(pre)
        post_pyramid = self.extract_pyramid(post)

        if len(pre_pyramid) != 4 or len(post_pyramid) != 4:
            raise RuntimeError(
                "SURTR expected four feature pyramid levels."
            )

        fused = []

        for adapter, f_pre, f_post in zip(
            self.change_adapters,
            pre_pyramid,
            post_pyramid,
        ):
            fused.append(
                adapter(
                    f_pre,
                    f_post,
                )
            )

        decoded = self.decoder(
            [f.clone() for f in fused]
        )

        logits = self.head(decoded)

        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


def parameter_summary(model: nn.Module) -> dict:
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / total,
    }

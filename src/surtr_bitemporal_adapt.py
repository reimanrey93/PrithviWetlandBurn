from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.surtr_bitemporal import ResidualChangeAdapter


# ============================================================
# Utilities
# ============================================================

def freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def parameter_summary(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
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


def trainable_parameter_groups(model: nn.Module):
    encoder_params = []
    task_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            task_params.append(p)

    return encoder_params, task_params


# ============================================================
# LoRA
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper for an existing Linear layer.

    y = W(x) + (alpha/r) * B(A(x))

    The pretrained W remains frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base_layer = base_layer

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Linear(
            base_layer.in_features,
            rank,
            bias=False,
        )

        self.lora_B = nn.Linear(
            rank,
            base_layer.out_features,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        # Start exactly from the pretrained solution.
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.base_layer(x)

        delta = self.lora_B(
            self.lora_A(
                self.dropout(x)
            )
        )

        return base + self.scaling * delta


def _get_parent_module(
    model: nn.Module,
    module_name: str,
):
    parts = module_name.split(".")
    parent = model

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    return parent, parts[-1]


def inject_lora(
    encoder: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
):
    """
    Inject LoRA into ViT attention qkv/projection layers.
    """

    targets = []

    for name, module in encoder.named_modules():

        if not isinstance(module, nn.Linear):
            continue

        lname = name.lower()

        if (
            lname.endswith("attn.qkv")
            or lname.endswith("attn.proj")
            or lname.endswith("attention.qkv")
            or lname.endswith("attention.proj")
        ):
            targets.append(
                (name, module)
            )

    if not targets:
        print("\nAttention-like Linear modules found:")

        for name, module in encoder.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            lname = name.lower()

            if (
                "attn" in lname
                or "attention" in lname
                or "qkv" in lname
                or "query" in lname
                or "key" in lname
                or "value" in lname
            ):
                print(
                    f"{name:80s} "
                    f"{module.in_features} -> "
                    f"{module.out_features}"
                )

        raise RuntimeError(
            "Could not locate Prithvi attention "
            "QKV/projection layers automatically."
        )

    replaced = []

    for name, module in targets:

        parent, child_name = _get_parent_module(
            encoder,
            name,
        )

        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )

        replaced.append(name)

    return replaced


# ============================================================
# Partial fine-tuning
# ============================================================

def find_transformer_blocks(
    encoder: nn.Module,
):
    candidates = []

    for name, module in encoder.named_modules():

        if (
            name.split(".")[-1] == "blocks"
            and isinstance(
                module,
                (nn.ModuleList, nn.Sequential),
            )
        ):
            candidates.append(
                (name, module)
            )

    if not candidates:
        raise RuntimeError(
            "Could not locate Transformer blocks "
            "inside Prithvi encoder."
        )

    candidates.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    return candidates[0]


def enable_partial_finetuning(
    encoder: nn.Module,
    last_n_blocks: int = 4,
):
    freeze_all(encoder)

    block_name, blocks = find_transformer_blocks(
        encoder
    )

    n_blocks = len(blocks)

    if not 1 <= last_n_blocks <= n_blocks:
        raise ValueError(
            f"Requested {last_n_blocks} blocks, "
            f"but encoder contains {n_blocks}."
        )

    first_trainable = (
        n_blocks - last_n_blocks
    )

    for idx in range(
        first_trainable,
        n_blocks,
    ):
        for p in blocks[idx].parameters():
            p.requires_grad = True

    return {
        "block_container": block_name,
        "total_blocks": n_blocks,
        "first_trainable": first_trainable,
        "last_trainable": n_blocks - 1,
    }


# ============================================================
# Siamese model
# ============================================================

class SURTRBiTemporalAdapt(nn.Module):

    FEATURE_CHANNELS = [
        256,
        512,
        1024,
        1024,
    ]

    def __init__(
        self,
        pixel_model,
        adaptation: str,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        partial_last_n_blocks: int = 4,
        freeze_neck: bool = True,
        train_decoder: bool = True,
    ):
        super().__init__()

        self.encoder = pixel_model.encoder
        self.neck = pixel_model.neck
        self.decoder = pixel_model.decoder
        self.head = pixel_model.head

        self.adaptation = adaptation
        self.freeze_neck = freeze_neck

        self.change_adapters = nn.ModuleList(
            [
                ResidualChangeAdapter(c)
                for c in self.FEATURE_CHANNELS
            ]
        )

        if freeze_neck:
            freeze_all(self.neck)

        if not train_decoder:
            freeze_all(self.decoder)
            freeze_all(self.head)

        if adaptation == "lora":

            freeze_all(self.encoder)

            replaced = inject_lora(
                self.encoder,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

            self.adaptation_info = {
                "type": "lora",
                "rank": lora_rank,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "n_lora_layers": len(replaced),
                "layers": replaced,
            }

        elif adaptation == "partial":

            info = enable_partial_finetuning(
                self.encoder,
                last_n_blocks=partial_last_n_blocks,
            )

            self.adaptation_info = {
                "type": "partial",
                **info,
            }

        else:
            raise ValueError(
                "adaptation must be "
                "'lora' or 'partial'"
            )

    def train(self, mode: bool = True):
        super().train(mode)

        # Frozen weights, but gradients must still pass
        # through the neck toward the encoder adaptation.
        if self.freeze_neck:
            self.neck.eval()

        return self

    def extract_pyramid(
        self,
        x: torch.Tensor,
    ):
        image_size = x.shape[-2:]

        # IMPORTANT:
        # no torch.no_grad() here.
        # LoRA / partial-FT requires gradients.
        features = self.encoder(x)

        pyramid = self.neck(
            features,
            image_size=image_size,
        )

        return pyramid

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ):
        if pre.shape != post.shape:
            raise ValueError(
                f"PRE/POST mismatch: "
                f"{tuple(pre.shape)} vs "
                f"{tuple(post.shape)}"
            )

        output_size = post.shape[-2:]

        pre_pyramid = self.extract_pyramid(
            pre
        )

        post_pyramid = self.extract_pyramid(
            post
        )

        if (
            len(pre_pyramid) != 4
            or len(post_pyramid) != 4
        ):
            raise RuntimeError(
                "Expected four feature pyramid levels."
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

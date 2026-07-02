"""PEFT video *teacher*: a frozen image ViT adapted to video gestures.

BRIEF §3.1 design
-----------------
A pretrained (frozen) image ViT-B/16 -- with ViT-S / DINOv2-S as lighter
fallbacks -- is turned into a video model by:

  1. **Per-frame encoding.** The input clip ``[B, C, T, H, W]`` is reshaped to
     ``[B*T, C, H, W]`` and pushed through the *frozen* ViT once per frame to
     obtain a pooled/CLS token per frame -> ``[B, T, D]``.
  2. **Temporal head.** A small (1-2 layer) ``TransformerEncoder`` models the T
     per-frame tokens; its output is mean-pooled over time -> ``[B, D]``. This
     is the *trainable* temporal modeling.
  3. **Classifier.** A single Linear ``D -> num_classes``.

Parameter-efficient fine-tuning is applied to the **frozen backbone**:
  * ``lora``    -- low-rank adapters on attention qkv/proj (src.models.peft).
  * ``adapter`` -- AIM-style bottleneck adapters around each block.
  * ``prompt``  -- shallow learnable prompt tokens prepended to patch tokens.
  * ``full_ft`` -- unfreeze the whole backbone (reference upper bound).
  * ``none``    -- frozen backbone + trainable temporal head only (linear probe).

8 GB (RTX 4060) budget tactics
------------------------------
  * Backbone stays frozen for every method except ``full_ft`` -> no optimizer
    state / activation grads for the bulk of the params.
  * **Gradient checkpointing** on the ViT blocks (``grad_checkpointing``) trades
    compute for activation memory; enabled via timm's ``set_grad_checkpointing``
    when available, else a manual ``torch.utils.checkpoint`` fallback.
  * Frames are folded into the batch dim so a single backbone call covers all
    ``B*T`` frames; the external engine runs this under bf16 autocast.
  * ``dynamic_img_size=True`` lets non-native frame sizes (e.g. 64 or 172) flow
    through the ViT without a resize copy.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from src.models.peft import (
    Adapter,
    LoRALinear,
    PromptTokens,
    apply_lora,
    freeze_backbone,
    insert_adapters,
)
from src.utils.env import count_parameters
from src.utils.logging_utils import get_logger
from src.utils.registry import register

log = get_logger("models.peft_teacher")

# Map friendly / brief names to concrete timm model names.
_BACKBONE_ALIASES = {
    "vit_base_patch16_224": "vit_base_patch16_224",
    "vit_small_patch16_224": "vit_small_patch16_224",
    # DINOv2 small -- accept both the brief's spelling and the timm name.
    "vit_base_patch14_dinov2": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2": "vit_small_patch14_dinov2.lvd142m",
}


def _create_backbone(name: str) -> nn.Module:
    """Create a frozen-ready timm ViT, degrading gracefully on download failure.

    Uses ``num_classes=0`` (feature extractor) and ``dynamic_img_size=True`` so
    arbitrary frame sizes are accepted. If the pretrained-weight download fails
    (offline CI, no cache), we retry with ``pretrained=False`` and log a warning
    so construction still succeeds for unit tests.
    """
    import timm

    timm_name = _BACKBONE_ALIASES.get(name, name)
    common = dict(pretrained=True, num_classes=0)
    # dynamic_img_size is supported by timm ViTs; guard for backbones that lack it.
    try:
        return timm.create_model(timm_name, dynamic_img_size=True, **common)
    except TypeError:
        # Backbone does not accept dynamic_img_size -> create without it.
        try:
            return timm.create_model(timm_name, **common)
        except Exception as e:  # pragma: no cover - network/cache dependent
            log.warning("Pretrained load for %r failed (%s); "
                        "falling back to pretrained=False.", timm_name, e)
            return timm.create_model(timm_name, pretrained=False, num_classes=0)
    except Exception as e:  # pragma: no cover - network/cache dependent
        log.warning("Pretrained load for %r failed (%s); "
                    "falling back to pretrained=False.", timm_name, e)
        try:
            return timm.create_model(timm_name, pretrained=False, num_classes=0,
                                     dynamic_img_size=True)
        except TypeError:
            return timm.create_model(timm_name, pretrained=False, num_classes=0)


@register("model", "peft_teacher")
class PEFTVideoTeacher(nn.Module):
    """Frozen image-ViT + PEFT + lightweight temporal head for video gestures.

    Constructor kwargs (all config-driven; unknown kwargs are ignored):
        backbone:           timm model name / alias (see ``_BACKBONE_ALIASES``).
        peft_method:        one of {none, lora, adapter, prompt, full_ft}.
        lora_rank/alpha:    LoRA hyperparameters.
        lora_targets:       list like ['q','k','v','o'] (fused qkv handled).
        adapter_dim:        bottleneck width for adapter method.
        prompt_tokens:      number of learnable prompt tokens for prompt method.
        temporal_layers:    TransformerEncoder depth over the T frame tokens.
        temporal_heads:     attention heads in the temporal encoder.
        num_classes:        classifier output width.
        frame_size:         informational only (input H/W); backbone is dynamic.
        grad_checkpointing:  enable activation checkpointing on backbone blocks.
    """

    def __init__(
        self,
        num_classes: int = 27,
        backbone: str = "vit_base_patch16_224",
        peft_method: str = "lora",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: Optional[Sequence[str]] = None,
        adapter_dim: int = 64,
        prompt_tokens: int = 8,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        frame_size: int = 224,
        grad_checkpointing: bool = False,
        **_ignore,
    ):
        super().__init__()
        self.peft_method = (peft_method or "none").lower()
        self.frame_size = frame_size
        self.grad_checkpointing = bool(grad_checkpointing)
        # Trainable PEFT modules are stored so they are registered as submodules
        # (and thus their params show up under .parameters()).
        self._peft_modules = nn.ModuleList()
        self.prompt: Optional[PromptTokens] = None

        # -- 1. Backbone -----------------------------------------------------
        self.backbone = _create_backbone(backbone)
        # Pooled feature width of the ViT (embed_dim). num_features is the timm
        # attribute for the post-norm feature width; embed_dim is the token dim.
        self.embed_dim = int(
            getattr(self.backbone, "embed_dim", None)
            or getattr(self.backbone, "num_features")
        )
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # Fail fast with an actionable message if frame_size isn't compatible
        # with the backbone's patch size (timm's patch embed asserts divisibility
        # even with dynamic_img_size). e.g. 172 is invalid for patch16.
        patch = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", None)
        if isinstance(patch, (tuple, list)):
            patch = patch[0]
        if patch and frame_size % int(patch) != 0:
            raise ValueError(
                f"frame_size={frame_size} is not divisible by the backbone patch "
                f"size {int(patch)} (ViT requires this). Use a multiple of "
                f"{int(patch)} — e.g. {frame_size // int(patch) * int(patch)} or "
                f"{(frame_size // int(patch) + 1) * int(patch)} (224 is native)."
            )

        # -- 2. Apply PEFT to the (to-be) frozen backbone --------------------
        self._configure_peft(lora_rank, lora_alpha, lora_targets, adapter_dim,
                             prompt_tokens)

        # -- 3. Gradient checkpointing on backbone blocks --------------------
        if self.grad_checkpointing:
            self._enable_grad_checkpointing()

        # -- 4. Temporal head + classifier (always trainable) ----------------
        # Batch-first TransformerEncoder over the T per-frame tokens.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=max(1, temporal_heads),
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for a small head
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=max(1, temporal_layers))
        self.temporal_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Pre-classifier feature width, exposed for downstream distillation.
        self.feature_dim = self.embed_dim

        self._log_trainable_stats()

    # ------------------------------------------------------------------ PEFT
    def _configure_peft(self, lora_rank, lora_alpha, lora_targets, adapter_dim,
                        prompt_tokens) -> None:
        method = self.peft_method
        if method == "full_ft":
            # Reference upper bound: everything in the backbone trains.
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            log.info("PEFT=full_ft: backbone fully trainable (upper bound).")
            return

        # Every other method starts from a fully frozen backbone.
        freeze_backbone(self.backbone)

        if method == "none":
            log.info("PEFT=none: frozen backbone, only temporal head trainable "
                     "(linear-probe-style).")
            return

        if method == "lora":
            targets = list(lora_targets) if lora_targets else ["q", "k", "v", "o"]
            wrappers = apply_lora(self.backbone, targets, lora_rank, lora_alpha)
            self._peft_modules.extend(wrappers)
            return

        if method == "adapter":
            adapters = insert_adapters(self.backbone, adapter_dim)
            self._peft_modules.extend(adapters)
            return

        if method == "prompt":
            self.prompt = PromptTokens(prompt_tokens, self.embed_dim)
            self._peft_modules.append(self.prompt)
            log.info("PEFT=prompt: %d learnable prompt tokens (shallow VPT).",
                     prompt_tokens)
            return

        raise ValueError(
            f"Unknown peft_method {method!r}. Expected one of "
            "{none, lora, adapter, prompt, full_ft}."
        )

    def _enable_grad_checkpointing(self) -> None:
        """Turn on activation checkpointing over backbone blocks if possible."""
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if callable(setter):
            try:
                setter(True)
                log.info("Gradient checkpointing enabled via timm "
                         "set_grad_checkpointing.")
                return
            except Exception as e:  # pragma: no cover
                log.warning("set_grad_checkpointing failed (%s); using manual "
                            "checkpoint fallback.", e)
        # Manual fallback: mark a flag consumed in _encode_frames.
        self._manual_ckpt = True
        log.info("Gradient checkpointing enabled via manual torch.utils.checkpoint "
                 "fallback.")

    # --------------------------------------------------------------- forward
    def _forward_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the backbone on ``[N, C, H, W]`` -> pooled token ``[N, D]``.

        For the ``prompt`` method we need the token sequence (to prepend prompt
        tokens), so we reconstruct the standard ViT stem -> blocks -> norm path
        using timm's public helpers. For all other methods we can use the
        backbone's own ``forward`` (num_classes=0 => returns the pooled feature).
        """
        if self.prompt is not None:
            return self._forward_tokens_with_prompt(frames)
        # Standard path: timm ViT with num_classes=0 returns pooled [N, D].
        return self.backbone(frames)

    def _forward_tokens_with_prompt(self, frames: torch.Tensor) -> torch.Tensor:
        """Prompt-tuning path: prepend learnable tokens to the patch sequence.

        Rebuilds the ViT forward from public timm pieces so we can splice prompt
        tokens in *after* the prefix (CLS) tokens and *before* the blocks. Falls
        back to the plain backbone forward if the expected pieces are missing.
        """
        bk = self.backbone
        needed = ("patch_embed", "blocks", "norm")
        if not all(hasattr(bk, a) for a in needed):
            log.warning("Backbone lacks %s; prompt tokens skipped for this "
                        "backbone, using plain forward.", needed)
            return bk(frames)

        # Patch embed + prefix tokens + positional embedding (timm helper).
        x = bk.patch_embed(frames)
        if hasattr(bk, "_pos_embed"):
            x = bk._pos_embed(x)  # adds CLS/reg tokens + pos-embed + dropout
        # Insert prompt tokens right after the prefix (CLS/register) tokens.
        n_prefix = self.num_prefix_tokens
        prompts = self.prompt.expand(x.shape[0]).to(dtype=x.dtype, device=x.device)
        x = torch.cat([x[:, :n_prefix], prompts, x[:, n_prefix:]], dim=1)

        if hasattr(bk, "patch_drop"):
            x = bk.patch_drop(x)
        if hasattr(bk, "norm_pre"):
            x = bk.norm_pre(x)
        x = bk.blocks(x)
        x = bk.norm(x)
        # Pool: prefer the CLS token, else mean-pool tokens.
        if n_prefix > 0:
            return x[:, 0]
        return x.mean(dim=1)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``[N, C, H, W]`` frames -> ``[N, D]``, with optional manual
        gradient checkpointing when timm's own flag wasn't available."""
        if getattr(self, "_manual_ckpt", False) and frames.requires_grad and self.training:
            # use_reentrant=False is the modern, safer checkpoint variant.
            return cp.checkpoint(self._forward_tokens, frames, use_reentrant=False)
        return self._forward_tokens(frames)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x: ``[B, C, T, H, W]`` -> logits ``[B, num_classes]``.

        Optionally also returns the pre-classifier temporal feature ``[B, D]``
        (used by feature distillation).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B,C,T,H,W], got shape {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        # [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W] (fold frames into batch).
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        tokens = self._encode_frames(frames)          # [B*T, D]
        tokens = tokens.reshape(b, t, self.embed_dim)  # [B, T, D]

        # Temporal modeling over the T frame tokens, then mean-pool over time.
        temporal = self.temporal(tokens)               # [B, T, D]
        feat = self.temporal_norm(temporal.mean(dim=1))  # [B, D]

        logits = self.classifier(feat)                 # [B, num_classes]
        if return_features:
            return logits, feat
        return logits

    # ----------------------------------------------------------------- utils
    def trainable_modules(self) -> List[nn.Module]:
        """The PEFT modules created for the current method (may be empty)."""
        return list(self._peft_modules)

    def _log_trainable_stats(self) -> None:
        """Log overall and backbone-only trainable parameter percentages."""
        overall = count_parameters(self)
        bk = count_parameters(self.backbone)
        overall_pct = 100.0 * overall["trainable"] / max(overall["total"], 1)
        bk_pct = 100.0 * bk["trainable"] / max(bk["total"], 1)
        log.info(
            "PEFTVideoTeacher[method=%s]: trainable %s / %s params (%.3f%% total). "
            "Backbone trainable %.3f%% (%s / %s).",
            self.peft_method,
            f"{overall['trainable']:,}", f"{overall['total']:,}", overall_pct,
            bk_pct, f"{bk['trainable']:,}", f"{bk['total']:,}",
        )

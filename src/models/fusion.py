"""Genuine multimodal RGB-D+IR fusion for NVGesture (BRIEF §3.5, §6.5).

This is the *real* multimodal track that replaces the dropped simulated
modalities: RGB, depth, and IR are three physically distinct video streams from
the NVGesture sensor rig. The fusion model composes a per-modality (or shared)
3D-CNN encoder with one of three fusion strategies and supports the modality
ablation axis RGB -> RGB+D -> RGB+D+IR.

Two usage modes (both supported, and documented for the reproducer):

  1. Ablation-by-training (recommended, clean): train a *separate* model per
     modality set by passing `modalities=('rgb',)`, `('rgb','depth')`,
     `('rgb','depth','ir')`. Each model only builds encoders for its modalities.

  2. Ablation-by-masking (convenience): train one full RGB+D+IR model, then feed
     it a subset dict (e.g. just {'rgb': ...}) at eval time. `forward` fuses only
     over the modalities *present in the input*, so a full model still runs on an
     RGB-only dict. This mixes train/eval modality sets and is only a convenience
     probe, not the headline ablation number.

Fusion strategies
-----------------
* late_logit    : one independent encoder per modality; average their classifier
                  logits. No extra fusion parameters.
* late_feature  : one independent encoder per modality; concat pooled features ->
                  small fusion head (Linear -> GELU -> Linear) -> logits.
* shared_adapter: a SINGLE shared encoder backbone across all modalities, plus a
                  tiny per-modality Linear bottleneck adapter on the features.
                  The shared backbone is the efficiency angle — parameter count
                  barely grows as modalities are added (only a small adapter per
                  modality), unlike the independent-encoder variants.

Input: dict {modality: [B, C, T, H, W]} with C=3 (depth/IR grayscale expanded to
3 channels by the loader), or a single [B,C,T,H,W] tensor (treated as the first
declared modality).
Output: [B, num_classes], or (logits, fused_features) if return_features=True.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn

from src.models.compact3dcnn import Compact3DCNN
from src.utils.logging_utils import get_logger
from src.utils.registry import register

log = get_logger("models.fusion")

# Encoder backbones selectable via the `encoder` kwarg. Each must accept
# (num_classes, in_channels, **kwargs), expose `.feature_dim`, and implement
# forward(x, return_features=False) -> logits | (logits, features).
_ENCODERS = {
    "compact3dcnn": Compact3DCNN,
}


def select_modalities(
    batch_x_dict: Dict[str, torch.Tensor], modalities: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Return a filtered dict containing only `modalities` that are present.

    Drives the ablation: `select_modalities(batch, ('rgb',))` yields an RGB-only
    dict from a full RGB+D+IR batch. Order follows `modalities`.
    """
    if not isinstance(batch_x_dict, dict):
        raise TypeError(f"select_modalities expects a dict, got {type(batch_x_dict)}")
    return {m: batch_x_dict[m] for m in modalities if m in batch_x_dict}


def forward_multimodal(model: "MultiModalFusion", batch_x):
    """Thin forward helper an eval `forward_fn` can use: `model(batch_x)`.

    Device handling stays in the caller (per the engine contract); this only
    normalises the call so `forward_fn=forward_multimodal` works directly.
    `batch_x` may be a dict {modality: tensor} or a single tensor.
    """
    return model(batch_x)


@register("model", "multimodal_fusion")
class MultiModalFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 25,
        modalities: Sequence[str] = ("rgb", "depth", "ir"),
        fusion: str = "late_feature",
        encoder: str = "compact3dcnn",
        encoder_kwargs: dict | None = None,
        **_ignore,
    ):
        super().__init__()
        if fusion not in ("late_logit", "late_feature", "shared_adapter"):
            raise ValueError(
                f"Unknown fusion {fusion!r}; expected "
                "'late_logit' | 'late_feature' | 'shared_adapter'"
            )
        if encoder not in _ENCODERS:
            raise ValueError(
                f"Unknown encoder {encoder!r}; available: {sorted(_ENCODERS)}"
            )
        if not modalities:
            raise ValueError("`modalities` must be a non-empty sequence")

        self.num_classes = int(num_classes)
        self.modalities: Tuple[str, ...] = tuple(modalities)
        self.fusion = fusion
        self.encoder_name = encoder
        enc_kwargs = dict(encoder_kwargs or {})
        enc_kwargs.setdefault("in_channels", 3)
        EncCls = _ENCODERS[encoder]

        if fusion == "shared_adapter":
            # ONE shared backbone across every modality. We only need its
            # features, so num_classes on the shared encoder is irrelevant; we
            # attach our own head. Adapters are tiny per-modality bottlenecks.
            self.shared_encoder = EncCls(num_classes=self.num_classes, **enc_kwargs)
            enc_feat = self.shared_encoder.feature_dim
            adapter_dim = max(enc_feat // 2, 32)
            self.adapters = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(enc_feat, adapter_dim),
                        nn.GELU(),
                    )
                    for m in self.modalities
                }
            )
            # Sum adapted features across modalities (order/count invariant),
            # then classify. Summation keeps the head shape fixed regardless of
            # how many modalities are present at forward time.
            self.feature_dim = adapter_dim
            self.head = nn.Linear(adapter_dim, self.num_classes)
            self.encoders = None
            self.fusion_head = None
        else:
            # Independent encoder per modality (separate weights).
            self.encoders = nn.ModuleDict(
                {m: EncCls(num_classes=self.num_classes, **enc_kwargs) for m in self.modalities}
            )
            self.shared_encoder = None
            self.adapters = None
            enc_feat = next(iter(self.encoders.values())).feature_dim

            if fusion == "late_logit":
                # Average per-modality logits; no extra fusion params.
                self.feature_dim = enc_feat
                self.fusion_head = None
                self.head = None
            else:  # late_feature
                # Concat features across ALL declared modalities -> fusion head.
                # (When a subset is present at forward time we zero-pad the
                # missing slots so the concat width stays fixed.)
                concat_dim = enc_feat * len(self.modalities)
                hidden = max(concat_dim // 2, self.num_classes)
                self.feature_dim = concat_dim
                self.fusion_head = nn.Sequential(
                    nn.Linear(concat_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.num_classes),
                )
                self.head = None

        log.info(
            "MultiModalFusion: fusion=%s encoder=%s modalities=%s "
            "num_classes=%d feature_dim=%d",
            self.fusion, self.encoder_name, self.modalities,
            self.num_classes, self.feature_dim,
        )

    # ------------------------------------------------------------------ #
    def _normalize_input(self, x) -> Dict[str, torch.Tensor]:
        """Accept a dict (preferred) or a single tensor (first modality)."""
        if isinstance(x, dict):
            present = {m: x[m] for m in self.modalities if m in x}
            if not present:
                raise ValueError(
                    f"No known modality in input. Model modalities={self.modalities}, "
                    f"input keys={list(x.keys())}"
                )
            return present
        if torch.is_tensor(x):
            return {self.modalities[0]: x}
        raise TypeError(
            f"forward expects a dict or tensor, got {type(x)}"
        )

    def forward(
        self, x: Union[Dict[str, torch.Tensor], torch.Tensor], return_features: bool = False
    ):
        present = self._normalize_input(x)
        # Reference tensor for device/dtype/batch when zero-padding missing slots.
        ref = next(iter(present.values()))

        if self.fusion == "late_logit":
            logits_sum = None
            feats = []
            for m, t in present.items():
                logit, feat = self.encoders[m](t, return_features=True)
                logits_sum = logit if logits_sum is None else logits_sum + logit
                feats.append(feat)
            logits = logits_sum / float(len(present))
            fused = torch.stack(feats, dim=0).mean(dim=0)  # mean feature, for probe

        elif self.fusion == "late_feature":
            # Fixed-width concat over ALL declared modalities; zero missing ones.
            enc_feat = self.encoders[self.modalities[0]].feature_dim
            parts = []
            for m in self.modalities:
                if m in present:
                    _, feat = self.encoders[m](present[m], return_features=True)
                else:
                    feat = torch.zeros(
                        ref.shape[0], enc_feat, device=ref.device, dtype=ref.dtype
                    )
                parts.append(feat)
            fused = torch.cat(parts, dim=1)
            logits = self.fusion_head(fused)

        else:  # shared_adapter
            adapted_sum = None
            for m, t in present.items():
                _, feat = self.shared_encoder(t, return_features=True)
                a = self.adapters[m](feat)
                adapted_sum = a if adapted_sum is None else adapted_sum + a
            fused = adapted_sum
            logits = self.head(fused)

        if return_features:
            return logits, fused
        return logits

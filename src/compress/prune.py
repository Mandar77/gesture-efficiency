"""Structured channel pruning for the compact 3D-CNN student (BRIEF §3.4, §6.3).

L1-norm *structured* channel pruning on convolutional layers using
``torch.nn.utils.prune``. We prune whole output channels (``dim=0``) ranked by
their L1 weight norm, which is the standard structured criterion and the one
that can translate into real speedups once the model is re-densified.

Two-step usage (matches torch's reparametrization model):
    model = structured_channel_prune(model, ratio=0.5)  # applies masks
    model = remove_pruning_reparam(model)               # bakes masks in (permanent)

``structured_channel_prune`` logs per-layer channels pruned and the resulting
global weight sparsity, and returns the pruned model. It is defensive: layers
that can't be pruned (e.g. too few channels for the ratio) are skipped with a
logged warning rather than crashing the run.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.utils.logging_utils import get_logger

log = get_logger("compress.prune")

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _global_weight_sparsity(model: nn.Module) -> float:
    """Fraction of zero weight elements across all Conv layers (0..1)."""
    total = 0
    zeros = 0
    for mod in model.modules():
        if isinstance(mod, _CONV_TYPES):
            w = mod.weight
            total += w.nelement()
            zeros += int((w == 0).sum().item())
    return (zeros / total) if total > 0 else 0.0


def structured_channel_prune(model: nn.Module, ratio: float) -> nn.Module:
    """L1-norm structured channel pruning on all Conv layers (in place).

    Args:
        model: model to prune (mutated in place; also returned for convenience).
        ratio: fraction of output channels to prune per Conv layer, in [0, 1).

    Behaviour:
        * ratio <= 0: no-op (logged), returns model unchanged.
        * For each Conv layer, prune ``round(ratio * out_channels)`` channels by
          smallest L1 weight norm along ``dim=0`` via ``prune.ln_structured``.
        * Layers where pruning would remove all channels (or that have a single
          channel) are skipped with a warning.
        * Masks are applied as reparametrizations; call ``remove_pruning_reparam``
          to make them permanent. Sparsity is reported before/after.

    Returns the pruned model.
    """
    if ratio is None or ratio <= 0:
        log.info("structured_channel_prune: ratio=%s <= 0 -> no pruning applied.",
                 ratio)
        return model
    if ratio >= 1:
        log.warning("structured_channel_prune: ratio=%s >= 1 clamped to 0.99 to "
                    "avoid removing every channel.", ratio)
        ratio = 0.99

    sparsity_before = _global_weight_sparsity(model)
    pruned_layers = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, _CONV_TYPES):
            continue
        out_ch = mod.weight.shape[0]
        n_prune = int(round(ratio * out_ch))
        if out_ch <= 1 or n_prune <= 0 or n_prune >= out_ch:
            log.warning("prune: skip %s (out_channels=%d, would prune=%d) — "
                        "ratio too aggressive/small for this layer.",
                        name or "conv", out_ch, n_prune)
            continue
        try:
            prune.ln_structured(mod, name="weight", amount=n_prune, n=1, dim=0)
            # Count fully-zeroed output channels for an honest per-layer report.
            with torch.no_grad():
                ch_l1 = mod.weight.detach().abs().flatten(1).sum(dim=1)
                zeroed = int((ch_l1 == 0).sum().item())
            log.info("prune: %s Conv out_channels=%d -> zeroed %d channel(s) "
                     "(target %d, ratio=%.2f).",
                     name or "conv", out_ch, zeroed, n_prune, ratio)
            pruned_layers += 1
        except Exception as e:
            log.warning("prune: failed on %s (%s: %s); layer left dense.",
                        name or "conv", type(e).__name__, e)

    sparsity_after = _global_weight_sparsity(model)
    log.info("structured_channel_prune: pruned %d Conv layer(s); global Conv "
             "weight sparsity %.4f -> %.4f (ratio=%.2f).",
             pruned_layers, sparsity_before, sparsity_after, ratio)
    setattr(model, "_prune_note",
            f"structured L1 channel prune (dim=0, ratio={ratio}): "
            f"{pruned_layers} Conv layer(s) pruned; sparsity "
            f"{sparsity_before:.4f}->{sparsity_after:.4f}.")
    return model


def remove_pruning_reparam(model: nn.Module) -> nn.Module:
    """Make pruning permanent: fold each ``weight_mask`` into ``weight`` and drop
    the reparametrization hooks. Safe to call on an unpruned model (no-op)."""
    removed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, _CONV_TYPES):
            continue
        try:
            if prune.is_pruned(mod) and hasattr(mod, "weight_orig"):
                prune.remove(mod, "weight")
                removed += 1
        except Exception as e:
            log.warning("remove_pruning_reparam: could not remove on %s (%s: %s).",
                        name or "conv", type(e).__name__, e)
    log.info("remove_pruning_reparam: made pruning permanent on %d layer(s).",
             removed)
    return model


def prune_report(model_before: nn.Module, model_after: nn.Module) -> Dict[str, Any]:
    """Small helper: Conv-weight sparsity before/after for honest reporting."""
    return {
        "conv_sparsity_before": round(_global_weight_sparsity(model_before), 6),
        "conv_sparsity_after": round(_global_weight_sparsity(model_after), 6),
        "prune_note": getattr(model_after, "_prune_note", "no prune note recorded"),
    }

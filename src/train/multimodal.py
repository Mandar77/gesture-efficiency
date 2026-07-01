"""Training / eval glue for the genuine multimodal NVGesture track (BRIEF §3.5).

The standard engine's cross-entropy path assumes `x` is a single tensor it can
`.to(device)`. The multimodal loader instead yields `(x_dict, y)` where `x_dict`
is `{modality: [B,C,T,H,W]}`, so both training and eval need dict-aware movers.

Usage
-----
Training with the shared engine (`src.train.engine.train_model`):

    from src.train.multimodal import multimodal_loss_fn
    train_model(model, loaders, cfg, loss_fn=multimodal_loss_fn(cfg))

The engine's own per-epoch `evaluate()` call does `x = x.to(device)`, which
FAILS on a dict. So for multimodal validation/eval use the dict-aware
`evaluate_multimodal()` in this file instead of `src.eval.metrics.evaluate`
(e.g. drive it from an `on_epoch_end` callback, or call it directly for the
final reported number). `multimodal_forward_fn` is provided for the case where
you *do* route through a dict-aware evaluate that already moved the tensors.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _move_x(x, device: torch.device):
    """Move a modality dict (or single tensor) to `device`."""
    if isinstance(x, dict):
        return {m: t.to(device, non_blocking=True) for m, t in x.items()}
    return x.to(device, non_blocking=True)


def multimodal_loss_fn(cfg: Dict[str, Any]) -> Callable:
    """Factory -> `loss_fn(model, batch, device, amp_ctx) -> (loss, logits, y)`.

    Matches the engine's loss_fn contract. Moves each modality tensor and the
    labels to `device`, runs `model(x_dict)` under `amp_ctx`, and computes CE
    with label smoothing pulled from `cfg['train']`.
    """
    tcfg = cfg.get("train", {})
    ce = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))

    def loss_fn(model, batch, device, amp_ctx):
        x, y = batch
        x = _move_x(x, device)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
            loss = ce(logits, y)
        return loss, logits, y

    return loss_fn


def multimodal_forward_fn(model, x):
    """Eval forward: `x` is the dict (already on device); return logits."""
    return model(x)


@torch.no_grad()
def evaluate_multimodal(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    return_confusion: bool = False,
) -> Dict[str, Any]:
    """Dict-aware mirror of `src.eval.metrics.evaluate` for multimodal batches.

    Batches are `(x_dict, y)`; each modality tensor is moved to `device`.
    Returns {'top1', 'top5', 'per_class_acc', 'num_samples', ('confusion')}.
    Self-contained so it can be used wherever the standard evaluate is unusable.
    """
    model.eval()

    correct1 = correct5 = total = 0
    num_classes = None
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}
    conf = None

    for batch in loader:
        x, y = batch
        x = _move_x(x, device)
        y = y.to(device, non_blocking=True)
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
        with ctx:
            logits = model(x)
        logits = logits.float()

        if num_classes is None:
            num_classes = logits.shape[1]
            if return_confusion:
                conf = np.zeros((num_classes, num_classes), dtype=np.int64)

        k5 = min(5, logits.shape[1])
        _, top5 = logits.topk(k5, dim=1)
        pred1 = top5[:, 0]
        correct1 += (pred1 == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()

        for t, p in zip(y.tolist(), pred1.tolist()):
            per_class_total[t] = per_class_total.get(t, 0) + 1
            if t == p:
                per_class_correct[t] = per_class_correct.get(t, 0) + 1
            if conf is not None:
                conf[t, p] += 1

    top1 = 100.0 * correct1 / max(total, 1)
    top5 = 100.0 * correct5 / max(total, 1)
    per_class_acc = {
        int(c): round(100.0 * per_class_correct.get(c, 0) / n, 2)
        for c, n in sorted(per_class_total.items())
    }
    out: Dict[str, Any] = {
        "top1": round(top1, 3),
        "top5": round(top5, 3),
        "per_class_acc": per_class_acc,
        "num_samples": total,
    }
    if return_confusion and conf is not None:
        out["confusion"] = conf.tolist()
    return out

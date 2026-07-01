"""Accuracy, per-class accuracy, and confusion matrix on a DataLoader.

Used both mid-training (engine) and by the efficiency bench for the final
reported accuracy. AMP is optional and only affects speed, not the reported
number materially (eval is done in inference mode).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    forward_fn=None,
    return_confusion: bool = False,
) -> Dict[str, Any]:
    """Return {'top1', 'top5', 'per_class_acc', ('confusion')}.

    `forward_fn(model, x) -> logits` lets multimodal/streaming models override
    the call signature; defaults to `model(x)`.
    """
    model.eval()
    forward_fn = forward_fn or (lambda m, x: m(x))

    correct1 = correct5 = total = 0
    num_classes = None
    per_class_correct: Dict[int, int] = {}
    per_class_total: Dict[int, int] = {}
    conf = None

    for batch in loader:
        x, y = batch
        # Support both single-tensor and multimodal dict batches so the same
        # evaluate works for the fusion model (x = {modality: tensor}).
        if isinstance(x, dict):
            x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
        with ctx:
            logits = forward_fn(model, x)
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
    out: Dict[str, Any] = {"top1": round(top1, 3), "top5": round(top5, 3),
                           "per_class_acc": per_class_acc, "num_samples": total}
    if return_confusion and conf is not None:
        out["confusion"] = conf.tolist()
    return out

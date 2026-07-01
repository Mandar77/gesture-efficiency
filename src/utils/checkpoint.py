"""Checkpoint save/load with embedded config + env metadata.

A checkpoint bundles model weights, optimizer/scheduler state (optional), the
resolved config, epoch/step, best metric, and env metadata — so a checkpoint is
self-describing and a reviewer can trace exactly how it was produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.env import env_metadata
from src.utils.logging_utils import get_logger

log = get_logger("utils.checkpoint")


def save_checkpoint(
    path: str | Path,
    model: "Any",
    *,
    optimizer: Optional["Any"] = None,
    scheduler: Optional["Any"] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config,
        "env": env_metadata(seed=seed),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    log.info("Saved checkpoint -> %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    model: Optional["Any"] = None,
    *,
    optimizer: Optional["Any"] = None,
    scheduler: Optional["Any"] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    log.info("Loaded checkpoint <- %s (epoch=%s)", path, payload.get("epoch"))
    return payload

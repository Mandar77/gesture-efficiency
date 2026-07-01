"""Config-driven training engine shared by the baseline, PEFT-teacher, and
distillation entrypoints.

Responsibilities:
    - AMP (bf16 preferred, fp16 fallback) via torch.autocast + GradScaler.
    - Optimizer (AdamW) + cosine schedule with linear warmup.
    - Gradient clipping, label smoothing.
    - Peak-VRAM logging at train start (BRIEF §3.1) and per-epoch eval.
    - Deterministic, reproducible; seeds/versions handled by callers.

The engine is intentionally model-agnostic: it takes a model, loaders, and cfg.
Distillation supplies a custom `loss_fn` and `forward_fn`; the baseline uses the
defaults (cross-entropy on model(x)).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import evaluate
from src.utils.logging_utils import get_logger

log = get_logger("train.engine")


def _resolve_amp(cfg: Dict[str, Any], device: torch.device):
    """Return (enabled, dtype). bf16 on Ada if supported, else fp16."""
    if not cfg.get("amp", True) or device.type != "cuda":
        return False, torch.float32
    want = cfg.get("amp_dtype", "bf16")
    if want == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def build_optimizer(model: nn.Module, tcfg: Dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = tcfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.05)
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=tcfg["lr"], momentum=0.9,
            weight_decay=tcfg.get("weight_decay", 0.05), nesterov=True,
        )
    raise ValueError(f"Unknown optimizer {name!r}")


def build_scheduler(optimizer, tcfg, steps_per_epoch: int):
    epochs = tcfg["epochs"]
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return step / max(warmup_steps, 1)
        if tcfg.get("scheduler", "cosine") == "cosine":
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model: nn.Module,
    loaders: Dict[str, Optional[DataLoader]],
    cfg: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    loss_fn: Optional[Callable] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    resume_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Train `model` and return a summary dict with best val metrics.

    `loss_fn(model, batch, device, amp_ctx) -> (loss, logits, targets)` may be
    supplied to customise the objective (used by distillation). If None, the
    default supervised cross-entropy path is used.

    If `resume_ckpt` is given and exists, training resumes from the epoch after
    the one saved there (model + optimizer + scheduler state restored). A
    checkpoint is written to that path after every epoch so a long run can
    survive an interruption (e.g. a Windows DataLoader shared-memory crash).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]
    model.to(device)

    amp_enabled, amp_dtype = _resolve_amp(cfg, device)
    # GradScaler only needed for fp16; bf16 has enough range without it.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    train_loader = loaders["train"]
    val_loader = loaders.get("val") or loaders.get("test")
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg, max(len(train_loader), 1))

    # ---- optional resume from a per-epoch checkpoint ------------------------
    start_epoch = 0
    resume_best = {"val_acc": -1.0, "epoch": -1}
    resume_history: list = []
    if resume_ckpt and Path(resume_ckpt).exists():
        import torch as _torch

        payload = _torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        extra = payload.get("extra") or {}
        start_epoch = int(payload.get("epoch", -1)) + 1
        resume_history = extra.get("history", [])
        resume_best = extra.get("best", resume_best)
        log.info("Resumed from %s -> starting at epoch %d (best val %.3f so far)",
                 resume_ckpt, start_epoch, resume_best.get("val_acc", -1.0))

    ce = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))

    def default_loss(m, batch, dev, ctx):
        x, y = batch
        # Support dict (multimodal) batches too, so the shared engine's default
        # supervised path works for the fusion model even without a custom
        # loss_fn (train_multimodal.py still supplies one; this is a safety net).
        if isinstance(x, dict):
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
        else:
            x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        with ctx:
            logits = m(x)
            loss = ce(logits, y)
        return loss, logits, y

    loss_fn = loss_fn or default_loss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = list(resume_history)
    best = dict(resume_best)
    global_step = 0
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        running = 0.0
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
            loss, _, _ = loss_fn(model, batch, device, ctx)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip", 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running += float(loss.detach())
            if it % tcfg.get("log_every", 20) == 0:
                log.info(
                    "epoch %d it %d/%d loss %.4f lr %.2e",
                    epoch, it, len(train_loader), float(loss.detach()),
                    scheduler.get_last_lr()[0],
                )

        epoch_summary = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1),
                         "epoch_time_s": round(time.time() - t0, 2)}

        if val_loader is not None and (epoch % tcfg.get("eval_every", 1) == 0):
            metrics = evaluate(model, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            epoch_summary.update({f"val_{k}": v for k, v in metrics.items()})
            if metrics["top1"] > best["val_acc"]:
                best = {"val_acc": metrics["top1"], "epoch": epoch}

        history.append(epoch_summary)
        log.info("epoch %d summary: %s", epoch, epoch_summary)
        if on_epoch_end:
            on_epoch_end(epoch, epoch_summary)

        # Per-epoch checkpoint so a long run survives an interruption. Written
        # atomically (temp + replace) so a crash mid-write can't corrupt it.
        if resume_ckpt:
            ckpt_p = Path(resume_ckpt)
            ckpt_p.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_p.with_suffix(ckpt_p.suffix + ".tmp")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "extra": {"history": history, "best": best},
            }, tmp)
            tmp.replace(ckpt_p)

    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
        log.info("Peak train VRAM: %.1f MB", peak_vram_mb)

    return {
        "history": history,
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "peak_train_vram_mb": peak_vram_mb,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
    }

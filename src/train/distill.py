"""Knowledge distillation objective for the streaming student (BRIEF §3.3).

`distillation_loss_fn(teacher, cfg)` is a FACTORY that returns a `loss_fn`
compatible with the training engine
(``loss_fn(student, batch, device, amp_ctx) -> (loss, logits, targets)``).

The total loss is::

    L = alpha_ce * CE(student_logits, labels)
      + beta_kd  * T^2 * KL(softmax(student/T) || softmax(teacher/T))
      + gamma_feat * feat_loss(proj(student_feats), teacher_feats)

Ablation axes (§6.2) are controlled purely by the config coefficients:
    * ``beta_kd  == 0``  -> no KD term          (logit-only supervision / "no_kd")
    * ``gamma_feat == 0`` -> logit distillation only (no feature matching)
    * ``gamma_feat  > 0`` -> logit + feature distillation

Teacher runs in ``eval()`` under ``torch.no_grad()`` and is called with
``return_features=True`` so it returns ``(logits, feats)``. Teacher and student
consume the SAME clip tensor ``x``.

Feature projection
------------------
When ``gamma_feat > 0`` the student's features are projected to the teacher's
feature dim by a trainable ``nn.Linear``. Because the engine builds the optimizer
from ``student.parameters()``, the projector MUST be a submodule of the student so
its parameters are optimized. ``attach_feature_projector(student, teacher_dim)``
adds ``student.feat_proj`` (created lazily on the correct device/dtype the first
time the loss is computed) and returns it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_utils import get_logger

log = get_logger("train.distill")


def attach_feature_projector(student: nn.Module, teacher_dim: int) -> Optional[nn.Module]:
    """Attach a trainable ``student.feat_proj`` mapping student feats -> teacher_dim.

    Made a submodule of the *student* so the engine's optimizer (built from
    ``student.parameters()``) trains it. Returns the projector, or an Identity if
    the student's ``feature_dim`` already matches ``teacher_dim``. Idempotent:
    calling twice reuses the existing projector.
    """
    student_dim = getattr(student, "feature_dim", None)
    if student_dim is None:
        raise AttributeError(
            "student must expose `feature_dim` to attach a feature projector"
        )
    existing = getattr(student, "feat_proj", None)
    if existing is not None:
        return existing
    if int(student_dim) == int(teacher_dim):
        # No projection needed; register Identity so callers can rely on the attr.
        proj: nn.Module = nn.Identity()
    else:
        proj = nn.Linear(int(student_dim), int(teacher_dim))
    # Place on the same device/dtype as the student so it trains in-place.
    ref = next((p for p in student.parameters()), None)
    if ref is not None:
        proj = proj.to(device=ref.device, dtype=ref.dtype)
    student.add_module("feat_proj", proj)
    log.info(
        "Attached feature projector: student_dim=%d -> teacher_dim=%d (%s)",
        int(student_dim), int(teacher_dim), type(proj).__name__,
    )
    return proj


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-scaled KL divergence, gradient-scaled by T^2 (Hinton et al.)."""
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    # batchmean matches the standard KD formulation; * T^2 restores gradient scale.
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return kl * (T * T)


def _feat_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor, kind: str) -> torch.Tensor:
    """Feature-matching loss between (already projected) student and teacher feats."""
    if kind == "cosine":
        # 1 - cosine similarity, averaged over the batch (0 == perfectly aligned).
        return (1.0 - F.cosine_similarity(student_feats, teacher_feats, dim=1)).mean()
    # default: mean-squared error
    return F.mse_loss(student_feats, teacher_feats)


def distillation_loss_fn(teacher: nn.Module, cfg: Dict[str, Any]) -> Callable:
    """Build a distillation `loss_fn` compatible with `train_model`.

    Args:
        teacher: pretrained teacher model. Set to ``eval()`` and run under
            ``no_grad``; must accept ``return_features=True`` and return
            ``(logits, feats)`` on the same clip tensor ``x`` as the student.
        cfg: full config dict; reads ``cfg['distill']`` for temperature,
            alpha_ce, beta_kd, gamma_feat, and optional ``feat_loss`` ('mse' |
            'cosine').

    Returns:
        ``loss_fn(student, batch, device, amp_ctx) -> (loss, student_logits, targets)``.
    """
    dcfg: Dict[str, Any] = cfg.get("distill", {}) or {}
    T = float(dcfg.get("temperature", 4.0))
    alpha_ce = float(dcfg.get("alpha_ce", 1.0))
    beta_kd = float(dcfg.get("beta_kd", 1.0))
    gamma_feat = float(dcfg.get("gamma_feat", 0.0))
    feat_kind = str(dcfg.get("feat_loss", "mse")).lower()
    label_smoothing = float(cfg.get("train", {}).get("label_smoothing", 0.0))

    use_feat = gamma_feat > 0.0
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    log.info(
        "Distillation objective: alpha_ce=%.3g beta_kd=%.3g gamma_feat=%.3g T=%.3g feat_loss=%s (%s)",
        alpha_ce, beta_kd, gamma_feat, T, feat_kind,
        "logit+feature" if use_feat else ("logit-only" if beta_kd > 0 else "CE-only/no_kd"),
    )

    # Periodic component logging state (module-level via closure).
    state = {"step": 0, "log_every": int(cfg.get("train", {}).get("log_every", 20))}

    def loss_fn(
        student: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        amp_ctx,
    ):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Teacher forward: eval + no_grad, same clip x, returns (logits, feats).
        teacher_logits = None
        teacher_feats = None
        if teacher is not None and (beta_kd > 0.0 or use_feat):
            teacher.to(device)
            with torch.no_grad():
                with amp_ctx:
                    t_out = teacher(x, return_features=True)
            if isinstance(t_out, (tuple, list)):
                teacher_logits, teacher_feats = t_out[0], t_out[1]
            else:  # teacher returned logits only
                teacher_logits = t_out
            teacher_logits = teacher_logits.detach()
            if teacher_feats is not None:
                teacher_feats = teacher_feats.detach()

        with amp_ctx:
            # Student forward with features (needed for feature distillation and
            # cheap enough to always request; contract guarantees the tuple form).
            s_out = student(x, return_features=True)
            if isinstance(s_out, (tuple, list)):
                student_logits, student_feats = s_out[0], s_out[1]
            else:
                student_logits, student_feats = s_out, None

            # --- 1. Cross-entropy on ground-truth labels ---
            loss_ce = ce(student_logits, y)
            total = alpha_ce * loss_ce

            # --- 2. KD (logit) term ---
            loss_kd = student_logits.new_zeros(())
            if beta_kd > 0.0 and teacher_logits is not None:
                loss_kd = _kd_loss(student_logits, teacher_logits.to(student_logits.dtype), T)
                total = total + beta_kd * loss_kd

            # --- 3. Feature-matching term ---
            loss_feat = student_logits.new_zeros(())
            if use_feat and teacher_feats is not None and student_feats is not None:
                proj = getattr(student, "feat_proj", None)
                if proj is None:
                    # Lazily attach on first use so its params join model.parameters().
                    proj = attach_feature_projector(student, int(teacher_feats.shape[-1]))
                projected = proj(student_feats)
                loss_feat = _feat_loss(projected, teacher_feats.to(projected.dtype), feat_kind)
                total = total + gamma_feat * loss_feat

        # Periodic logging of the three components.
        if state["step"] % max(state["log_every"], 1) == 0:
            log.info(
                "distill step %d | total %.4f = %.3g*ce(%.4f) + %.3g*kd(%.4f) + %.3g*feat(%.4f)",
                state["step"], float(total.detach()),
                alpha_ce, float(loss_ce.detach()),
                beta_kd, float(loss_kd.detach()),
                gamma_feat, float(loss_feat.detach()),
            )
        state["step"] += 1

        return total, student_logits, y

    return loss_fn

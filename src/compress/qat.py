"""Quantization-Aware Training (QAT) for the compact 3D-CNN student (BRIEF §3.4, §6.3).

QAT inserts fake-quant / observer modules so the network *learns* to be robust
to INT8 rounding during a short fine-tune, then is converted to a real INT8
model. Per the BRIEF decision threshold, **QAT is the recommended path when the
plain PTQ top-1 drop exceeds ~2-3 points** — otherwise PTQ (see ``ptq.py``) is
cheaper and sufficient.

Workflow:
    model = build_student(...)                    # trained fp32
    qat_model = prepare_qat(model)                # observers inserted
    #  ... fine-tune qat_model for a few epochs with the normal train engine ...
    int8_model = convert_qat(qat_model)           # final CPU INT8 model
    report = report_compression(model, int8_model, loader, device, "qat")

Same hardware caveats as PTQ apply: eager-mode INT8 runs on **CPU**, and torch's
static support for ``Conv3d`` is limited. If ``prepare_qat``/``convert`` hit an
unsupported op we FALL BACK (QAT of Linear-only, or a plain fp32 pass-through)
and LOG exactly what happened — we never claim a fully-int8 model on fallback.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

log = get_logger("compress.qat")

# BRIEF §6.3 decision threshold: prefer QAT over PTQ when PTQ costs more than
# this many top-1 points.
QAT_PREFERENCE_THRESHOLD_PP = 3.0


def should_prefer_qat(ptq_top1_drop_pp: float,
                      threshold_pp: float = QAT_PREFERENCE_THRESHOLD_PP) -> bool:
    """Return True if the observed PTQ accuracy drop justifies QAT (BRIEF §6.3).

    ``ptq_top1_drop_pp`` is in percentage points (positive = accuracy lost),
    e.g. the ``top1_drop`` field from ``ptq.report_compression``.
    """
    prefer = ptq_top1_drop_pp is not None and ptq_top1_drop_pp > threshold_pp
    log.info("should_prefer_qat: ptq drop=%s pp, threshold=%s pp -> %s",
             ptq_top1_drop_pp, threshold_pp, "QAT" if prefer else "PTQ ok")
    return prefer


def _find_fusable_sequences(model: nn.Module):
    """See ptq._find_fusable_sequences — duplicated here to keep QAT self-contained."""
    named = dict(model.named_modules())
    fusable = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for parent_name, parent in named.items():
        if not isinstance(parent, nn.Sequential):
            continue
        children = list(parent.named_children())
        i = 0
        while i < len(children):
            grp = []
            if isinstance(children[i][1], conv_types):
                grp.append(children[i][0])
                if i + 1 < len(children) and isinstance(children[i + 1][1], bn_types):
                    grp.append(children[i + 1][0])
                    if i + 2 < len(children) and isinstance(children[i + 2][1], nn.ReLU):
                        grp.append(children[i + 2][0])
                if len(grp) >= 2:
                    prefix = f"{parent_name}." if parent_name else ""
                    fusable.append([prefix + g for g in grp])
                    i += len(grp)
                    continue
            i += 1
    return fusable


def prepare_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Insert fake-quant observers and return a QAT-ready model to fine-tune.

    The returned model is in ``.train()`` mode (QAT fine-tuning updates the
    observers' ranges). Fine-tune it for a few epochs with the normal engine,
    then call ``convert_qat`` to produce the final INT8 model.

    Defensive: if fusion or ``prepare_qat`` fails (e.g. Conv3d unsupported), we
    fall back to a QAT config that only touches ``nn.Linear`` and LOG it; if even
    that fails we return the original model untouched (so training can proceed as
    plain fp32) and record the situation on ``_qat_note``.

    NOTE: the model is deep-copied so the caller's fp32 model is preserved.
    """
    import torch.ao.quantization as tq

    if backend not in ("fbgemm", "qnnpack"):
        log.warning("prepare_qat: unknown backend %r; defaulting to fbgemm.", backend)
        backend = "fbgemm"
    torch.backends.quantized.engine = backend

    m = copy.deepcopy(model).train()
    try:
        m.qconfig = tq.get_default_qat_qconfig(backend)
        try:
            m = tq.fuse_modules(m, _find_fusable_sequences(m), inplace=False)
            m.train()  # fuse_modules may flip to eval; QAT prepare needs train
        except Exception as e:
            log.debug("prepare_qat: fusion skipped (%s).", e)
        prepared = tq.prepare_qat(m, inplace=False)
        note = (f"qat_full (backend={backend}): fake-quant observers on "
                f"Conv/Linear/activations; fine-tune then convert_qat -> CPU INT8.")
        log.info("prepare_qat: %s", note)
        setattr(prepared, "_qat_note", note)
        return prepared
    except Exception as e:
        log.warning("prepare_qat: full QAT prepare failed (%s: %s); trying "
                    "Linear-only QAT fallback.", type(e).__name__, e)

    # Fallback: qconfig only on Linear layers.
    try:
        m2 = copy.deepcopy(model).train()
        m2.qconfig = None  # disable everywhere by default
        n_linear = 0
        for mod in m2.modules():
            if isinstance(mod, nn.Linear):
                mod.qconfig = tq.get_default_qat_qconfig(backend)
                n_linear += 1
        prepared = tq.prepare_qat(m2, inplace=False)
        note = (f"qat_linear_only_FALLBACK: full QAT unsupported (Conv3d). "
                f"Fake-quant on {n_linear} nn.Linear layer(s) only; Conv/other "
                f"stay FP32. NOT fully int8 after convert.")
        log.info("prepare_qat: %s", note)
        setattr(prepared, "_qat_note", note)
        return prepared
    except Exception as e2:
        log.error("prepare_qat: Linear-only QAT also failed (%s: %s); returning "
                  "the original fp32 model unchanged (NO QAT).",
                  type(e2).__name__, e2)
        out = copy.deepcopy(model).train()
        setattr(out, "_qat_note",
                f"qat FAILED ({e2!r}); model unchanged, NO fake-quant inserted.")
        return out


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a fine-tuned QAT model into the final INT8 model (CPU).

    Defensive: on any conversion failure we log and return the (eval) input model
    so the run continues; the returned model carries ``_quantization_note``.
    """
    import torch.ao.quantization as tq

    prior_note = getattr(model, "_qat_note", "qat (no prepare note recorded)")
    m = copy.deepcopy(model).eval().to("cpu")
    try:
        converted = tq.convert(m, inplace=False)
        note = f"converted from QAT -> CPU INT8. [{prior_note}]"
        log.info("convert_qat: %s", note)
    except Exception as e:
        log.error("convert_qat: conversion failed (%s: %s); returning the QAT "
                  "model in eval mode WITHOUT int8 conversion.",
                  type(e).__name__, e)
        converted = m
        note = (f"convert_qat FAILED ({e!r}); returned un-converted eval model "
                f"(NO real int8). [{prior_note}]")
    setattr(converted, "_quantization_note", note)
    return converted

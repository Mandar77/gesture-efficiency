"""Post-training quantization (PTQ) for the compact 3D-CNN student (BRIEF §3.4, §6.3).

This module implements two PTQ paths and an honest reporting helper:

  * ``quantize_fp16``     — half-precision copy for **GPU** inference.
  * ``quantize_int8_ptq`` — eager-mode INT8 PTQ via ``torch.ao.quantization``.
  * ``report_compression`` — size + top-1 before/after, with the honest drop.

IMPORTANT hardware / backend notes (do not paper over these):

  * FP16 (``.half()``) is an *inference-on-GPU* optimisation. On an RTX 4060 it
    roughly halves the on-disk size and speeds up GPU inference. It is NOT a
    CPU quantization scheme.

  * INT8 eager-mode quantization in torch runs its quantized kernels on the
    **CPU** (fbgemm / qnnpack backends). There is no INT8 eager path on the
    4060's CUDA cores here, so a quantized model's forward MUST run on CPU.
    ``report_compression`` therefore evaluates the INT8 model on CPU.

  * torch's *static* INT8 support for ``Conv3d`` is limited/absent on the CPU
    backends. We attempt full static PTQ (fuse -> prepare -> calibrate ->
    convert); if any op is unsupported we FALL BACK to *dynamic* quantization
    of ``Linear`` layers only, and we LOG exactly what was quantized vs left in
    fp32. We never claim a fully-int8 model when a fallback occurred — the
    returned model carries a ``_quantization_note`` attribute and
    ``report_compression`` surfaces it in the report dict.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

log = get_logger("compress.ptq")


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------
def quantize_fp16(model: nn.Module) -> nn.Module:
    """Return an FP16 (half-precision) **copy** of ``model`` for GPU inference.

    This is a lossless-to-implement, defensive op: we deep-copy the model so the
    caller's fp32 model is untouched, then call ``.half()``. On-disk size is
    ~halved. Intended to be run on CUDA (fp16 CPU kernels are limited); the
    caller is responsible for ``.to('cuda')`` at inference time.
    """
    model_fp16 = copy.deepcopy(model).eval()
    try:
        model_fp16 = model_fp16.half()
        note = "fp16: all parameters/buffers cast to torch.float16 (GPU inference)."
    except Exception as e:  # pragma: no cover - .half() is very robust
        log.warning("quantize_fp16: .half() failed (%s); returning fp32 copy.", e)
        note = f"fp16 FAILED ({e!r}); returned fp32 copy (NO compression applied)."
    log.info("quantize_fp16: %s", note)
    setattr(model_fp16, "_quantization_note", note)
    return model_fp16


# ---------------------------------------------------------------------------
# INT8 PTQ (eager mode, CPU backend)
# ---------------------------------------------------------------------------
def _try_static_ptq(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    backend: str,
) -> nn.Module:
    """Attempt full static INT8 PTQ. Raises on any unsupported-op failure so the
    caller can fall back. Does NOT mutate the input model (works on a copy)."""
    import torch.ao.quantization as tq

    m = copy.deepcopy(model).eval().to("cpu")

    # BatchNorm running stats must be frozen for eager static quant fusion.
    torch.backends.quantized.engine = backend
    m.qconfig = tq.get_default_qconfig(backend)

    # Try to fuse Conv+BN(+ReLU) where the model exposes such sequences. The
    # compact student uses nn.Sequential(Conv3d, BatchNorm3d, ReLU) blocks; if
    # fusion patterns don't match we just skip fusion (prepare still works).
    try:
        m = tq.fuse_modules(m, _find_fusable_sequences(m), inplace=False)
    except Exception as e:
        log.debug("static PTQ: module fusion skipped (%s).", e)

    prepared = tq.prepare(m, inplace=False)

    # Calibrate: run a handful of batches through the observers on CPU.
    prepared.eval()
    n_batches = 0
    with torch.no_grad():
        for batch in calib_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            prepared(x.to("cpu"))
            n_batches += 1
            if n_batches >= 16:  # a tiny calib set is plenty for observers
                break
    if n_batches == 0:
        raise RuntimeError("empty calibration loader — cannot calibrate observers")

    converted = tq.convert(prepared, inplace=False)

    # tq.convert() does NOT fail on Conv3d — it happily produces a quantized
    # conv3d module. The failure is DEFERRED to the forward pass, because this
    # torch/CPU build has no 'quantized::conv3d' kernel. Validate with a real
    # forward here so an unsupported-op model raises NOW and the caller's
    # dynamic Linear-only fallback actually triggers (instead of returning a
    # model that crashes later at eval time and silently reports top1=None).
    with torch.no_grad():
        probe = next(iter(calib_loader))
        px = probe[0] if isinstance(probe, (list, tuple)) else probe
        converted(px[:1].to("cpu"))  # raises NotImplementedError on Conv3d

    log.info("static INT8 PTQ succeeded (backend=%s, calib_batches=%d).",
             backend, n_batches)
    return converted


def _find_fusable_sequences(model: nn.Module):
    """Best-effort discovery of [Conv, BN, (ReLU)] fusion groups by module name.

    Returns a list of name-lists suitable for ``fuse_modules``. Only groups whose
    layer *types* are fusable are returned; anything else is left alone.
    """
    named = dict(model.named_modules())
    fusable = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    # Group by parent Sequential container: look for consecutive numeric indices.
    for parent_name, parent in named.items():
        if not isinstance(parent, nn.Sequential):
            continue
        children = list(parent.named_children())
        i = 0
        while i < len(children):
            grp = []
            # conv
            if isinstance(children[i][1], conv_types):
                grp.append(children[i][0])
                # bn
                if i + 1 < len(children) and isinstance(children[i + 1][1], bn_types):
                    grp.append(children[i + 1][0])
                    # relu
                    if i + 2 < len(children) and isinstance(children[i + 2][1], nn.ReLU):
                        grp.append(children[i + 2][0])
                if len(grp) >= 2:
                    prefix = f"{parent_name}." if parent_name else ""
                    fusable.append([prefix + g for g in grp])
                    i += len(grp)
                    continue
            i += 1
    return fusable


def _dynamic_fallback(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of Linear layers only (CPU). Robust; supports
    any model regardless of Conv3d limitations."""
    import torch.ao.quantization as tq

    m = copy.deepcopy(model).eval().to("cpu")
    quantized = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    n_linear = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    log.info("dynamic INT8 fallback: quantized %d nn.Linear layer(s); all "
             "Conv/other layers remain fp32.", n_linear)
    return quantized


def quantize_int8_ptq(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    backend: str = "fbgemm",
) -> nn.Module:
    """Post-training INT8 quantization (eager mode) of the student model.

    Strategy (defensive, never crashes the run):
      1. Try **static** PTQ: fuse -> prepare -> calibrate on ``calib_loader``
         -> convert. This is the ideal path (conv + linear + activations int8).
      2. If ANY op is unsupported by the CPU INT8 backend (common for Conv3d),
         catch the error and FALL BACK to **dynamic** quantization of Linear
         layers, logging clearly what was and was not quantized.

    The returned model runs on **CPU** (torch eager INT8 has no CUDA kernels).
    It carries a ``_quantization_note`` attribute describing exactly what
    happened, which ``report_compression`` surfaces honestly.

    Args:
        model: fp32 student to quantize (not mutated; we deep-copy).
        calib_loader: iterable yielding (x, y) or x; used for static calibration.
        device: original device (informational; INT8 runs on CPU regardless).
        backend: 'fbgemm' (x86 server) or 'qnnpack' (ARM/mobile).
    """
    if backend not in ("fbgemm", "qnnpack"):
        log.warning("quantize_int8_ptq: unknown backend %r; defaulting to fbgemm.",
                    backend)
        backend = "fbgemm"

    note: str
    quantized: nn.Module
    try:
        quantized = _try_static_ptq(model, calib_loader, device, backend)
        note = (f"int8_static_ptq (backend={backend}): Conv/Linear/activations "
                f"quantized to INT8 via eager static PTQ; runs on CPU.")
    except Exception as e:
        log.warning(
            "static INT8 PTQ failed (%s: %s) — likely Conv3d/op unsupported on "
            "the CPU INT8 backend. Falling back to DYNAMIC Linear-only int8.",
            type(e).__name__, e,
        )
        try:
            quantized = _dynamic_fallback(model)
            note = (
                "int8_dynamic_FALLBACK: static PTQ unsupported (Conv3d not "
                "quantizable on CPU backend). ONLY nn.Linear layers are INT8 "
                "(dynamic); all Conv3d/BN/other layers remain FP32. This is NOT "
                "a fully-int8 model. Runs on CPU."
            )
        except Exception as e2:  # last-resort: never crash the run
            log.error("dynamic INT8 fallback also failed (%s: %s); returning an "
                      "fp32 CPU copy with NO quantization.", type(e2).__name__, e2)
            quantized = copy.deepcopy(model).eval().to("cpu")
            note = (f"int8 PTQ FAILED entirely ({e2!r}); returned FP32 CPU copy "
                    f"with NO quantization applied.")

    log.info("quantize_int8_ptq note: %s", note)
    setattr(quantized, "_quantization_note", note)
    return quantized


# ---------------------------------------------------------------------------
# Honest reporting
# ---------------------------------------------------------------------------
def report_compression(
    model_fp32: nn.Module,
    model_compressed: nn.Module,
    loader,
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    """Compare an fp32 model to its compressed variant. NEVER hides the drop.

    Returns a dict with on-disk size before/after (MB), the size-reduction
    factor, top-1 accuracy before/after (via ``src.eval.metrics.evaluate``), the
    honest accuracy drop (percentage points), and a ``quantization_note`` string
    describing exactly what was quantized.

    Notes:
      * INT8 (and any model whose note mentions CPU) is evaluated on CPU; fp32
        baseline and fp16 are evaluated on ``device``. fp16 inputs are cast to
        half so the loader's fp32 tensors match the model dtype.
    """
    from src.bench.efficiency_bench import measure_disk_size
    from src.eval.metrics import evaluate

    note = getattr(model_compressed, "_quantization_note",
                   "no quantization_note recorded on compressed model")

    size_before = measure_disk_size(model_fp32)
    size_after = measure_disk_size(model_compressed)
    reduction = round(size_before / size_after, 3) if size_after > 0 else None

    # --- choose eval devices honestly -------------------------------------
    note_l = note.lower()
    is_cpu_only = ("int8" in note_l) or ("cpu" in note_l and "fp16" not in note_l)
    is_fp16 = "fp16" in note_l and "no compression" not in note_l

    comp_device = torch.device("cpu") if is_cpu_only else device

    def _eval(m: nn.Module, dev: torch.device, half: bool) -> Optional[float]:
        try:
            m = m.to(dev).eval()
            forward_fn = None
            if half:
                # loader tensors are fp32; cast to match the half model.
                forward_fn = lambda mod, x: mod(x.half())
            res = evaluate(m, loader, dev, forward_fn=forward_fn)
            return float(res["top1"])
        except Exception as e:
            log.error("report_compression[%s]: eval failed on %s (%s: %s).",
                      label, dev, type(e).__name__, e)
            return None

    top1_before = _eval(model_fp32, device, half=False)
    top1_after = _eval(model_compressed, comp_device, half=is_fp16)

    drop = None
    if top1_before is not None and top1_after is not None:
        drop = round(top1_before - top1_after, 3)  # positive = accuracy lost

    report: Dict[str, Any] = {
        "label": label,
        "size_mb_before": size_before,
        "size_mb_after": size_after,
        "size_reduction_x": reduction,
        "top1_before": top1_before,
        "top1_after": top1_after,
        "top1_drop": drop,  # honest: NEVER suppressed, positive means worse
        "compressed_eval_device": str(comp_device),
        "quantization_note": note,
    }
    log.info(
        "report_compression[%s]: size %.3f MB -> %.3f MB (%.2fx); "
        "top1 %s -> %s (drop=%s pp) | %s",
        label, size_before, size_after,
        (reduction if reduction is not None else float("nan")),
        top1_before, top1_after, drop, note,
    )
    return report

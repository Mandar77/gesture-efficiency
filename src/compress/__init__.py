"""Compression for the compact 3D-CNN student (BRIEF §3.4, §6.3).

Public API:

  PTQ (post-training quantization) — ``ptq.py``:
    * ``quantize_fp16``        — fp16 copy for GPU inference.
    * ``quantize_int8_ptq``    — eager-mode INT8 PTQ (CPU), robust Conv3d fallback.
    * ``report_compression``   — honest size + top-1 before/after (never hides drop).

  QAT (quantization-aware training) — ``qat.py``:
    * ``prepare_qat``          — insert fake-quant observers; fine-tune, then...
    * ``convert_qat``          — ...convert to final CPU INT8 model.
    * ``should_prefer_qat``    — BRIEF §6.3 threshold (PTQ drop > ~3 pp -> QAT).

  Pruning — ``prune.py``:
    * ``structured_channel_prune`` — L1-norm structured channel pruning (dim=0).
    * ``remove_pruning_reparam``   — make pruning permanent.
    * ``prune_report``             — sparsity before/after.

All functions are defensive: risky torch quantization ops are wrapped in
try/except with clearly LOGGED fallbacks; returned dicts / models carry an
honest note describing exactly what was quantized (never a fabricated
"fully-int8" claim after a fallback).
"""

from src.compress.ptq import (
    quantize_fp16,
    quantize_int8_ptq,
    report_compression,
)
from src.compress.qat import (
    prepare_qat,
    convert_qat,
    should_prefer_qat,
    QAT_PREFERENCE_THRESHOLD_PP,
)
from src.compress.prune import (
    structured_channel_prune,
    remove_pruning_reparam,
    prune_report,
)

__all__ = [
    "quantize_fp16",
    "quantize_int8_ptq",
    "report_compression",
    "prepare_qat",
    "convert_qat",
    "should_prefer_qat",
    "QAT_PREFERENCE_THRESHOLD_PP",
    "structured_channel_prune",
    "remove_pruning_reparam",
    "prune_report",
]

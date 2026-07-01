"""Environment metadata capture + parameter counting.

Every results artifact must record GPU name, CUDA version, torch version, and
seed (BRIEF section 11). `env_metadata` returns a JSON-serialisable dict that
the ResultsWriter stamps onto every row. `Date.now()`-style timestamps are
produced here (allowed in normal runtime — only workflow scripts forbid it).
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def env_metadata(seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect reproducibility metadata for a results artifact."""
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "torch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_total_mem_mb": None,
        "device": "cpu",
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            meta["device"] = "cuda"
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            meta["gpu_total_mem_mb"] = round(props.total_memory / (1024**2), 1)
            try:
                meta["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                pass
    except ImportError:
        pass

    if extra:
        meta.update(extra)
    return meta


def count_parameters(model: "Any") -> Dict[str, int]:
    """Return {'total', 'trainable', 'frozen'} parameter counts for a module."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}

"""efficiency_bench.py — the measurement harness (BRIEF §5).

For any model + input spec, measure and log:
    - Top-1 / top-5 accuracy (+ per-class, confusion) on a loader (optional).
    - Trainable params and total params.
    - MACs/FLOPs per clip via fvcore (primary; cross-checked with ptflops/thop).
    - Measured FPS + end-to-end latency on the GPU: warm up, time many runs,
      report mean ± std; single-clip latency (bs=1) separate from batched
      throughput; preprocessing time available for the streaming number.
    - Peak VRAM at inference (torch.cuda.max_memory_allocated).
    - On-disk model size (state_dict serialized size).

All numbers come from real runs; nothing is fabricated. Every artifact records
GPU/CUDA/torch/seed/timestamp via ResultsWriter.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import evaluate
from src.utils.env import count_parameters
from src.utils.logging_utils import get_logger

log = get_logger("bench")


# ---------------------------------------------------------------------------
# FLOPs / MACs
# ---------------------------------------------------------------------------
def measure_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, Optional[float]]:
    """Return MACs/FLOPs per single clip via fvcore, cross-checked.

    input_shape is per-sample (no batch dim), e.g. (3, 16, 172, 172).
    Convention: FLOPs = 2 * MACs. fvcore reports MACs ("flops" in its API are
    actually MACs); we report both and note the protocol in the paper.
    """
    model = model.eval().to(device)
    x = torch.randn(1, *input_shape, device=device)
    out: Dict[str, Optional[float]] = {
        "macs_g": None, "flops_g": None,
        "macs_g_ptflops": None, "macs_g_thop": None,
        "flops_backend": "fvcore",
    }

    # fvcore (primary, matches ConvMixFormer protocol)
    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs = fca.total()  # fvcore "flops" == MACs
        out["macs_g"] = round(macs / 1e9, 4)
        out["flops_g"] = round(2 * macs / 1e9, 4)
    except Exception as e:  # pragma: no cover
        log.warning("fvcore FLOPs failed: %s", e)

    # ptflops cross-check
    try:
        from ptflops import get_model_complexity_info

        macs_p, _ = get_model_complexity_info(
            model, tuple(input_shape), as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        out["macs_g_ptflops"] = round(macs_p / 1e9, 4)
    except Exception as e:
        log.debug("ptflops cross-check unavailable: %s", e)

    # thop cross-check
    try:
        from thop import profile

        macs_t, _ = profile(model, inputs=(x,), verbose=False)
        out["macs_g_thop"] = round(macs_t / 1e9, 4)
    except Exception as e:
        log.debug("thop cross-check unavailable: %s", e)

    return out


# ---------------------------------------------------------------------------
# Latency / throughput
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    *,
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 100,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    """Warm up, then time `timed_iters` forward passes. Returns per-run latency
    (ms) mean±std and derived throughput (clips/s = FPS)."""
    model = model.eval().to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)

    for _ in range(warmup_iters):
        with ctx:
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(timed_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with ctx:
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std())
    fps = batch_size * 1000.0 / mean_ms if mean_ms > 0 else float("nan")
    return {
        "batch_size": batch_size,
        "latency_ms_mean": round(mean_ms, 4),
        "latency_ms_std": round(std_ms, 4),
        "latency_ms_p50": round(float(np.percentile(arr, 50)), 4),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 4),
        "throughput_fps": round(fps, 2),
        "amp_enabled": amp_enabled,
    }


# ---------------------------------------------------------------------------
# VRAM / disk
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_peak_vram(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    batch_size: int = 1,
) -> Optional[float]:
    if device.type != "cuda":
        return None
    model = model.eval().to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    _ = model(x)
    torch.cuda.synchronize(device)
    return round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)


def measure_disk_size(model: nn.Module) -> float:
    """Serialized state_dict size in MB (proxy for on-disk model size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / (1024**2), 3)


# ---------------------------------------------------------------------------
# Full bench
# ---------------------------------------------------------------------------
def bench_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    *,
    loader=None,
    forward_fn=None,
    batch_sizes: Sequence[int] = (1, 8),
    warmup_iters: int = 20,
    timed_iters: int = 100,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    measure_acc: bool = True,
) -> Dict[str, Any]:
    """Run the full efficiency profile. `input_shape` is per-sample."""
    result: Dict[str, Any] = {"input_shape": list(input_shape)}
    result["params"] = count_parameters(model)
    result["params"]["trainable_pct"] = (
        round(100.0 * result["params"]["trainable"] / max(result["params"]["total"], 1), 3)
    )
    result["disk_size_mb"] = measure_disk_size(model)
    result["flops"] = measure_flops(model, input_shape, device)

    latencies = {}
    for bs in batch_sizes:
        try:
            latencies[f"bs{bs}"] = measure_latency(
                model, input_shape, device, batch_size=bs,
                warmup_iters=warmup_iters, timed_iters=timed_iters,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            )
        except RuntimeError as e:  # e.g. OOM at large batch on 8GB
            log.warning("latency bs=%d failed (%s); recording OOM", bs, e)
            latencies[f"bs{bs}"] = {"batch_size": bs, "error": "oom_or_runtime"}
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result["latency"] = latencies
    # Convenience top-level: single-clip latency + batched throughput.
    if "bs1" in latencies and "latency_ms_mean" in latencies["bs1"]:
        result["single_clip_latency_ms"] = latencies["bs1"]["latency_ms_mean"]
        result["single_clip_fps"] = latencies["bs1"]["throughput_fps"]

    result["peak_infer_vram_mb"] = measure_peak_vram(
        model, input_shape, device, batch_size=max(batch_sizes)
    )

    if measure_acc and loader is not None:
        acc = evaluate(model, loader, device, amp_enabled=amp_enabled,
                       amp_dtype=amp_dtype, forward_fn=forward_fn,
                       return_confusion=True)
        result["accuracy"] = {"top1": acc["top1"], "top5": acc["top5"],
                              "num_samples": acc["num_samples"]}
        result["per_class_acc"] = acc["per_class_acc"]
        result["confusion"] = acc.get("confusion")
    else:
        # Not measured this run -> leave as TODO, never fabricate (BRIEF §11).
        result["accuracy"] = {"top1": None, "top5": None}

    return result

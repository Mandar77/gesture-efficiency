"""Single-session, back-to-back re-bench of ALL frontier models for mutually
comparable single-clip latency/FPS.

The per-model FPS numbers in the committed results were each measured at the end
of a SEPARATE training run, under different GPU thermal/clock states -- so they
are NOT a controlled like-for-like comparison (see SANITY.md bench caveat; e.g.
the two architecturally-identical students benched 110 vs 45 FPS). Accuracy and
FLOPs are deterministic and fine; only FPS/latency need re-benching.

This script loads each checkpoint, rebuilds the model from its EMBEDDED config
(so reconstruction is faithful and generic across student / peft_teacher /
compact3dcnn), and runs the identical latency protocol on all of them in ONE
process, one after another -- same warmup, same timed iters, bs=1 single-clip.
A short inter-model cooldown + a fixed warmup per model keeps the clock state
consistent. Writes a JSON table to experiments/rebench_frontier.json.

Usage: .venv/Scripts/python scripts/rebench_frontier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401  (register datasets - not used here but harmless)
import src.models  # noqa: F401  (register models)
from src.bench.efficiency_bench import measure_latency, measure_flops
from src.utils import build, get_logger
from src.utils.checkpoint import load_checkpoint
from src.utils.env import count_parameters

log = get_logger("scripts.rebench")

# The frontier models to re-bench, in a fixed order. (run_name, checkpoint path).
# Diagnostic/sanity/smoke checkpoints are intentionally excluded -- they are not
# frontier operating points.
MODELS = [
    ("jester_student_logit_feat_kd", "checkpoints/distill/jester_student_logit_feat_kd.pt"),
    ("jester_student_logit_kd",      "checkpoints/distill/jester_student_logit_kd.pt"),
    ("jester_student_no_kd",         "checkpoints/distill/jester_student_no_kd.pt"),
    ("jester_vitb_lora_r16_lr2e4_8f224", "checkpoints/peft/jester_vitb_lora_r16_lr2e4_8f224.pt"),
    ("jester_vits_lora_8f224",       "checkpoints/peft/jester_vits_lora_8f224.pt"),
    ("jester_vits_adapter_8f224",    "checkpoints/peft/jester_vits_adapter_8f224.pt"),
    ("jester_vits_full_ft_8f224",    "checkpoints/peft/jester_vits_full_ft_8f224.pt"),
    ("jester_vits_prompt_8f224",     "checkpoints/peft/jester_vits_prompt_8f224.pt"),
    ("jester_compact3dcnn_16f172_30ep", "checkpoints/baseline/jester_compact3dcnn_16f172_30ep.pt"),
    ("jester_compact3dcnn_8f224_30ep",  "checkpoints/baseline/jester_compact3dcnn_8f224_30ep.pt"),
]

# Identical protocol for every model. AMP on (bf16) to match how these models
# run on the 4060. GPU-clock locking needs admin (denied on this laptop), so we
# get REPRODUCIBILITY instead via: heavy warmup, many timed iters, a GPU
# clock-up burn BEFORE the first real model (so none runs cold), and REPEATS x3
# per model reporting the MEDIAN. bs=1 = on-device streaming latency (primary);
# bs=8 = batched throughput (supplementary, more compute-bound).
WARMUP_ITERS = 50
TIMED_ITERS = 500
REPEATS = 3
BATCH_SIZES = (1, 8)
AMP_ENABLED = True
AMP_DTYPE = torch.bfloat16


def _rebuild_from_ckpt(path: str, device):
    """Reconstruct a model from its embedded config (generic across model types)."""
    payload = load_checkpoint(path, map_location="cpu")
    cfg = payload.get("config") or {}
    mcfg = cfg.get("model", {}) or {}
    name = mcfg.get("name", "compact3dcnn")
    mkwargs = dict(mcfg.get("kwargs", {}) or {})
    dcfg = cfg.get("data", {}) or {}
    mkwargs["num_classes"] = dcfg.get("num_classes", 27)
    # PEFT teacher needs its PEFT hyperparams to match module shapes for a clean
    # state_dict load (same reconstruction distill_student.py::_build_teacher does).
    if name == "peft_teacher":
        pcfg = cfg.get("peft", {}) or {}
        mkwargs.update(
            peft_method=pcfg.get("method", "lora"),
            lora_rank=pcfg.get("lora_rank", 8),
            lora_alpha=pcfg.get("lora_alpha", 16),
            lora_targets=pcfg.get("lora_targets", ["q", "k", "v", "o"]),
            adapter_dim=pcfg.get("adapter_dim", 64),
            prompt_tokens=pcfg.get("prompt_tokens", 8),
            frame_size=dcfg.get("frame_size", 224),
        )
    model = build("model", name, **mkwargs)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        log.warning("%s: %d missing / %d unexpected keys on load",
                    Path(path).name, len(missing), len(unexpected))
    frames = dcfg.get("num_frames", 8)
    size = dcfg.get("frame_size", 224)
    return model.to(device).eval(), (3, frames, size, size)


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else round(0.5 * (xs[n // 2 - 1] + xs[n // 2]), 4)


def _bench_one(model, shape, device, bs):
    """Median-of-REPEATS latency at batch size `bs`."""
    fps_runs, ms_runs, p95_runs = [], [], []
    for _ in range(REPEATS):
        lat = measure_latency(model, shape, device, batch_size=bs,
                              warmup_iters=WARMUP_ITERS, timed_iters=TIMED_ITERS,
                              amp_enabled=AMP_ENABLED, amp_dtype=AMP_DTYPE)
        fps_runs.append(lat.get("throughput_fps"))
        ms_runs.append(lat.get("latency_ms_mean"))
        p95_runs.append(lat.get("latency_ms_p95"))
    return {"fps_median": _median(fps_runs), "ms_median": _median(ms_runs),
            "p95_ms_median": _median(p95_runs), "fps_runs": fps_runs}


def main():
    if not torch.cuda.is_available():
        log.error("CUDA not available; single-clip FPS must be measured on the GPU of record.")
        return
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    log.info("Rigorous re-bench: %d models | warmup=%d timed=%d repeats=%d bs=%s amp=bf16 | %s",
             len(MODELS), WARMUP_ITERS, TIMED_ITERS, REPEATS, BATCH_SIZES, gpu)

    # Clock-up burn BEFORE any real model so none runs cold (clock-lock unavailable
    # without admin). ~3s of dense matmul spins the GPU to steady clocks.
    log.info("GPU clock-up burn (~3s)...")
    burn = torch.randn(2048, 2048, device=device)
    t_end = time.time() + 3.0
    while time.time() < t_end:
        burn = burn @ burn
        burn = burn / (burn.norm() + 1e-6)
    torch.cuda.synchronize()
    log.info("clocks now: see nvidia-smi; starting measurements.")

    # Prepend a throwaway warm pass of the first model so the real first
    # measurement is not cold. Its result is discarded (not appended to rows).
    if MODELS and Path(MODELS[0][1]).exists():
        wm, wshape = _rebuild_from_ckpt(MODELS[0][1], device)
        log.info("warm-up pass (discarded): %s", MODELS[0][0])
        _bench_one(wm, wshape, device, 1)
        del wm
        torch.cuda.empty_cache()

    rows = []
    for run_name, ckpt in MODELS:
        if not Path(ckpt).exists():
            log.warning("SKIP %s: checkpoint not found (%s)", run_name, ckpt)
            continue
        model, shape = _rebuild_from_ckpt(ckpt, device)
        params = count_parameters(model)
        flops = measure_flops(model, shape, device)
        per_bs = {}
        for bs in BATCH_SIZES:
            try:
                per_bs[f"bs{bs}"] = _bench_one(model, shape, device, bs)
            except RuntimeError as e:
                log.warning("%s bs=%d failed (%s)", run_name, bs, e)
                per_bs[f"bs{bs}"] = {"error": "oom_or_runtime"}
                torch.cuda.empty_cache()
        b1 = per_bs.get("bs1", {})
        b8 = per_bs.get("bs8", {})
        rows.append({
            "run_name": run_name,
            "input_shape": list(shape),
            "params_total": params["total"],
            "flops_g": flops.get("flops_g"),
            "bs1_fps_median": b1.get("fps_median"),
            "bs1_latency_ms_median": b1.get("ms_median"),
            "bs1_fps_runs": b1.get("fps_runs"),
            "bs8_fps_median": b8.get("fps_median"),
            "bs8_latency_ms_median": b8.get("ms_median"),
        })
        log.info("%-38s FLOPs=%7.2fG | bs1 %6.2f FPS (%.2f ms) runs=%s | bs8 %7.2f FPS",
                 run_name, flops.get("flops_g") or float("nan"),
                 b1.get("fps_median") or float("nan"), b1.get("ms_median") or float("nan"),
                 [round(x, 1) for x in (b1.get("fps_runs") or []) if x is not None],
                 b8.get("fps_median") or float("nan"))
        del model
        torch.cuda.empty_cache()
        time.sleep(3)

    out = {
        "protocol": {"warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
                     "repeats": REPEATS, "batch_sizes": list(BATCH_SIZES),
                     "amp": "bf16", "gpu": gpu, "clock_lock": "unavailable (no admin)",
                     "reproducibility": "clock-up burn + throwaway warm pass + heavy warmup + median-of-3 (no clock lock)",
                     "note": "PyTorch eager; bs1=streaming latency (primary), bs8=batched throughput"},
        "rows": rows,
    }
    out_path = Path("experiments/rebench_frontier.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s (%d models)", out_path, len(rows))


if __name__ == "__main__":
    main()

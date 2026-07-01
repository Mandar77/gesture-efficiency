"""Compress a trained student and emit the FP32/FP16/INT8/pruned comparison (M6).

Loads a student checkpoint, then for each requested mode produces the compressed
model, runs `report_compression` (honest size + accuracy delta), and writes one
results row per (mode, prune_ratio). QAT additionally fine-tunes the fake-quant
model for a few epochs before converting. Accuracy drops are reported, never
hidden (BRIEF §6.3, §11).

Usage:
    python scripts/compress_student.py --config configs/distill_student.yaml \
        --ckpt checkpoints/distill/jester_student_logit_kd.pt \
        --modes fp32 fp16 int8_ptq int8_qat --prune-ratios 0.0 0.3 0.5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401
import src.models  # noqa: F401
from src.bench.efficiency_bench import measure_disk_size
from src.compress import (
    convert_qat, prepare_qat, prune_report, quantize_fp16,
    quantize_int8_ptq, remove_pruning_reparam, report_compression,
    structured_channel_prune,
)
from src.data.loaders import build_dataloaders
from src.train.engine import train_model
from src.utils import (
    ResultsWriter, build, get_logger, load_config, seed_everything,
)
from src.utils.checkpoint import load_checkpoint

log = get_logger("scripts.compress_student")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--modes", nargs="+",
                   default=["fp32", "fp16", "int8_ptq", "int8_qat"])
    p.add_argument("--prune-ratios", nargs="*", type=float, default=[0.0])
    p.add_argument("--qat-epochs", type=int, default=2)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    return p.parse_args()


def _load_student(cfg, ckpt, device):
    model = build("model", cfg["model"].get("name", "streaming_student"),
                  num_classes=cfg["data"]["num_classes"], **cfg["model"].get("kwargs", {}))
    payload = load_checkpoint(ckpt, model, map_location="cpu")
    model.to(device).eval()
    log.info("Loaded student from %s (epoch %s)", ckpt, payload.get("epoch"))
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    seed = seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    eval_loader = loaders.get("test") or loaders.get("val")
    calib_loader = loaders.get("train") or eval_loader
    if eval_loader is None:
        log.error("No eval loader; cannot report accuracy. Prepare data first.")
        return

    base = _load_student(cfg, args.ckpt, device)
    writer = ResultsWriter(root=cfg["output"]["root"], seed=seed)
    group = "compress"

    for ratio in args.prune_ratios:
        # Apply structured pruning first (ratio=0 => no-op), then quantize.
        pruned = copy.deepcopy(base)
        if ratio > 0:
            structured_channel_prune(pruned, ratio)
            remove_pruning_reparam(pruned)
        prune_info = prune_report(base, pruned) if ratio > 0 else {"conv_sparsity_after": 0.0}

        for mode in args.modes:
            tag = f"{mode}_prune{ratio}"
            log.info("=== compressing: %s ===", tag)
            if mode == "fp32":
                comp = copy.deepcopy(pruned)
                setattr(comp, "_quantization_note", "fp32 baseline (no quantization).")
            elif mode == "fp16":
                comp = quantize_fp16(pruned)
            elif mode == "int8_ptq":
                comp = quantize_int8_ptq(pruned, calib_loader, device)
            elif mode == "int8_qat":
                qat_model = prepare_qat(pruned).to(device)
                # Short QAT fine-tune with the standard supervised engine.
                qcfg = copy.deepcopy(cfg)
                qcfg["train"]["epochs"] = args.qat_epochs
                train_model(qat_model, loaders, qcfg, device=device)
                comp = convert_qat(qat_model)
            else:
                log.warning("Unknown mode %s; skipping.", mode)
                continue

            rep = report_compression(pruned, comp, eval_loader, device, tag)
            result = {
                "compress": {"mode": mode, "prune_ratio": ratio, **rep, **prune_info},
                "bench": {  # minimal bench block so viz can pick up size/accuracy
                    "params": {"total": sum(p.numel() for p in base.parameters()),
                               "trainable": 0, "frozen": 0, "trainable_pct": 0.0},
                    "disk_size_mb": rep["size_mb_after"],
                    "flops": {"macs_g": None, "flops_g": None, "flops_backend": "n/a"},
                    "accuracy": {"top1": rep.get("top1_after"), "top5": None},
                    "single_clip_fps": None, "single_clip_latency_ms": None,
                    "peak_infer_vram_mb": None,
                },
                "config_path": args.config,
            }
            writer.write(result, group=group, run_name=tag, seed=seed)

    log.info("DONE compression matrix -> experiments/%s/", group)


if __name__ == "__main__":
    main()

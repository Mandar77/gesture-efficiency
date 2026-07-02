"""Distill a streaming student from a trained PEFT teacher (M5).

Loads a teacher checkpoint, builds the streaming student, trains it with the
distillation objective (logit + optional feature KD) via the shared engine, then
runs the full efficiency bench on the student. The distillation ablation
(no-KD / logit-only / logit+feature) is controlled purely by the distill.* config
coefficients (beta_kd, gamma_feat).

Usage:
    python scripts/distill_student.py --config configs/distill_student.yaml \
        --set distill.teacher_ckpt=checkpoints/peft/jester_lora.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401
import src.models  # noqa: F401
from src.bench.efficiency_bench import bench_model
from src.data.loaders import build_dataloaders
from src.train.distill import distillation_loss_fn, attach_feature_projector
from src.train.engine import _resolve_amp, train_model
from src.utils import (
    ResultsWriter, build, get_logger, load_config, save_config,
    seed_everything, setup_file_logging,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint

log = get_logger("scripts.distill_student")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--no-bench", action="store_true")
    return p.parse_args()


def _build_teacher(cfg, device):
    """Reconstruct + load the teacher from its checkpoint (config embedded)."""
    ckpt_path = cfg["distill"].get("teacher_ckpt")
    if not ckpt_path or not Path(ckpt_path).exists():
        log.warning("No valid teacher_ckpt (%s). Distilling with NO teacher — "
                    "this reduces to plain CE training (documents the no-teacher "
                    "baseline). Set distill.teacher_ckpt to a real checkpoint.",
                    ckpt_path)
        return None
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    tcfg = payload.get("config") or {}
    mkwargs = dict((tcfg.get("model", {}) or {}).get("kwargs", {}))
    # Reconstruct the EXACT teacher architecture. The PEFT hyperparameters
    # (rank/alpha/targets/adapter_dim/prompt_tokens) change the module shapes,
    # so all of them must be passed — otherwise defaults (e.g. lora_rank=8) would
    # produce mismatched shapes and strict=False would silently drop the trained
    # LoRA/adapter weights, giving a randomly-initialised teacher.
    pcfg = tcfg.get("peft", {}) or {}
    dcfg_t = tcfg.get("data", {}) or {}
    mkwargs.update(
        num_classes=dcfg_t.get("num_classes", cfg["data"]["num_classes"]),
        peft_method=pcfg.get("method", "lora"),
        lora_rank=pcfg.get("lora_rank", 8),
        lora_alpha=pcfg.get("lora_alpha", 16),
        lora_targets=pcfg.get("lora_targets", ["q", "k", "v", "o"]),
        adapter_dim=pcfg.get("adapter_dim", 64),
        prompt_tokens=pcfg.get("prompt_tokens", 8),
        frame_size=dcfg_t.get("frame_size", cfg["data"]["frame_size"]),
    )
    teacher = build("model", (tcfg.get("model", {}) or {}).get("name", "peft_teacher"), **mkwargs)
    missing, unexpected = teacher.load_state_dict(payload["model_state"], strict=False)
    # A mismatch here means the reconstructed architecture differs from the saved
    # one — surface it loudly rather than silently distilling from random weights.
    if missing or unexpected:
        log.warning("Teacher load: %d missing, %d unexpected keys. If these are "
                    "PEFT/backbone weights the teacher is NOT faithfully restored.",
                    len(missing), len(unexpected))
    teacher.to(device).eval()
    log.info("Loaded teacher from %s", ckpt_path)
    return teacher


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    seed = seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))

    run_name = cfg["output"]["run_name"]
    run_dir = Path(cfg["output"]["root"]) / cfg["output"]["group"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(run_dir)
    save_config(cfg, run_dir / "config.resolved.yaml")

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(cfg)

    student = build("model", cfg["model"].get("name", "streaming_student"),
                    num_classes=cfg["data"]["num_classes"],
                    **cfg["model"].get("kwargs", {}))
    student.to(device)

    teacher = _build_teacher(cfg, device)
    # If feature distillation is on, attach the projector up-front so its params
    # are in the optimizer from step 0.
    if cfg["distill"].get("gamma_feat", 0.0) > 0 and teacher is not None:
        attach_feature_projector(student, teacher.feature_dim)

    loss_fn = distillation_loss_fn(teacher, cfg) if teacher is not None else None
    ckpt_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints")) / cfg["output"]["group"]
    resume_path = ckpt_dir / f"{run_name}.resume.pt"  # per-epoch; auto-resumes
    summary = train_model(student, loaders, cfg, device=device, loss_fn=loss_fn,
                          resume_ckpt=str(resume_path))
    save_checkpoint(ckpt_dir / f"{run_name}.pt", student,
                    epoch=cfg["train"]["epochs"], best_metric=summary.get("best_val_acc"),
                    config=cfg, seed=seed)

    result = {"train": summary,
              "distill": {k: cfg["distill"].get(k) for k in
                          ("temperature", "alpha_ce", "beta_kd", "gamma_feat")},
              "teacher_ckpt": cfg["distill"].get("teacher_ckpt"),
              "config_path": args.config}
    if not args.no_bench:
        amp_enabled, amp_dtype = _resolve_amp(cfg, device)
        dcfg = cfg["data"]
        shape = (3, dcfg["num_frames"], dcfg["frame_size"], dcfg["frame_size"])
        eval_loader = loaders.get("test") or loaders.get("val")
        result["bench"] = bench_model(
            student, shape, device, loader=eval_loader,
            batch_sizes=cfg["bench"]["batch_sizes"],
            warmup_iters=cfg["bench"]["warmup_iters"], timed_iters=cfg["bench"]["timed_iters"],
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, measure_acc=eval_loader is not None,
        )

    ResultsWriter(root=cfg["output"]["root"], seed=seed).write(
        result, group=cfg["output"]["group"], run_name=run_name, seed=seed)
    log.info("DONE distillation run: %s", run_name)


if __name__ == "__main__":
    main()

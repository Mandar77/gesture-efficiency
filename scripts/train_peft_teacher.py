"""Train the PEFT teacher (frozen ViT + LoRA/adapter/prompt/full-FT) — M4.

Reuses the shared engine (AMP + grad checkpointing) and the efficiency bench.
The PEFT method is selected via config (peft.method), enabling the PEFT sweep
(§6.1): run this with different `--set peft.method=...` to emit one row each.

Usage:
    python scripts/train_peft_teacher.py --config configs/peft_lora.yaml
    python scripts/train_peft_teacher.py --config configs/peft_lora.yaml --set peft.method=adapter
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
from src.train.engine import _resolve_amp, train_model
from src.utils import (
    ResultsWriter, build, get_logger, load_config, save_config,
    seed_everything, setup_file_logging,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.env import count_parameters

log = get_logger("scripts.train_peft_teacher")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--no-bench", action="store_true")
    return p.parse_args()


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

    mkwargs = dict(cfg["model"].get("kwargs", {}))
    mkwargs.update(
        num_classes=cfg["data"]["num_classes"],
        peft_method=cfg["peft"]["method"],
        lora_rank=cfg["peft"]["lora_rank"],
        lora_alpha=cfg["peft"]["lora_alpha"],
        lora_targets=cfg["peft"]["lora_targets"],
        adapter_dim=cfg["peft"]["adapter_dim"],
        prompt_tokens=cfg["peft"]["prompt_tokens"],
        frame_size=cfg["data"]["frame_size"],
        grad_checkpointing=cfg.get("grad_checkpointing", True),
    )
    model = build("model", cfg["model"].get("name", "peft_teacher"), **mkwargs)

    params = count_parameters(model)
    log.info("PEFT=%s | trainable %.3f%% (%d / %d)", cfg["peft"]["method"],
             params["trainable"] / max(params["total"], 1) * 100,
             params["trainable"], params["total"])

    ckpt_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints")) / cfg["output"]["group"]
    resume_path = ckpt_dir / f"{run_name}.resume.pt"  # per-epoch; auto-resumes
    summary = train_model(model, loaders, cfg, device=device,
                          resume_ckpt=str(resume_path))
    save_checkpoint(ckpt_dir / f"{run_name}.pt", model,
                    epoch=cfg["train"]["epochs"], best_metric=summary.get("best_val_acc"),
                    config=cfg, seed=seed)

    result = {"train": summary, "peft_method": cfg["peft"]["method"],
              "dataset": cfg["data"]["name"], "config_path": args.config}
    if not args.no_bench:
        amp_enabled, amp_dtype = _resolve_amp(cfg, device)
        dcfg = cfg["data"]
        shape = (3, dcfg["num_frames"], dcfg["frame_size"], dcfg["frame_size"])
        eval_loader = loaders.get("test") or loaders.get("val")
        result["bench"] = bench_model(
            model, shape, device, loader=eval_loader,
            batch_sizes=cfg["bench"]["batch_sizes"],
            warmup_iters=cfg["bench"]["warmup_iters"], timed_iters=cfg["bench"]["timed_iters"],
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, measure_acc=eval_loader is not None,
        )

    ResultsWriter(root=cfg["output"]["root"], seed=seed).write(
        result, group=cfg["output"]["group"], run_name=run_name, seed=seed)
    log.info("DONE PEFT teacher run: %s", run_name)


if __name__ == "__main__":
    main()

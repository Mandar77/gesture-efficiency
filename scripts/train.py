"""Generic train + bench entrypoint (thin CLI over src/*).

Usage:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/baseline_jester.yaml --set train.epochs=1

Loads a config, builds data + model, trains via the engine, runs the full
efficiency bench, and writes a results JSON/CSV with env metadata. This single
script drives the smoke test and the from-scratch baseline (M3). PEFT teacher
and distillation have their own entrypoints that reuse the same engine + bench.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401  registers datasets
import src.models  # noqa: F401  registers models
from src.bench.efficiency_bench import bench_model
from src.data.loaders import build_dataloaders
from src.train.engine import _resolve_amp, train_model
from src.utils import (
    ResultsWriter,
    build,
    get_logger,
    load_config,
    save_config,
    seed_everything,
    setup_file_logging,
)
from src.utils.checkpoint import save_checkpoint

log = get_logger("scripts.train")


def parse_args():
    p = argparse.ArgumentParser(description="Train + bench a gesture model.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="Dotted config overrides, e.g. train.epochs=1")
    p.add_argument("--no-bench", action="store_true", help="Skip efficiency bench.")
    p.add_argument("--no-train", action="store_true", help="Bench only (needs --ckpt).")
    p.add_argument("--ckpt", default=None, help="Optional checkpoint to load.")
    return p.parse_args()


def build_model(cfg):
    mcfg = cfg["model"]
    kwargs = dict(mcfg.get("kwargs", {}))
    kwargs.setdefault("num_classes", cfg["data"]["num_classes"])
    return build("model", mcfg["name"], **kwargs)


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    seed = seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))

    out_root = Path(cfg["output"]["root"]) / cfg["output"]["group"]
    run_name = cfg["output"]["run_name"]
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(run_dir)
    save_config(cfg, run_dir / "config.resolved.yaml")

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    log.info("Device: %s | seed: %d", device, seed)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg)
    log.info("Model %s built.", cfg["model"]["name"])

    if args.ckpt:
        from src.utils.checkpoint import load_checkpoint
        load_checkpoint(args.ckpt, model, map_location=str(device))

    train_summary = {}
    if not args.no_train:
        ckpt_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints")) / cfg["output"]["group"]
        # Per-epoch resume checkpoint (auto-resumes if it already exists, e.g.
        # after a crash). Distinct from the final self-describing checkpoint.
        resume_path = ckpt_dir / f"{run_name}.resume.pt"
        train_summary = train_model(model, loaders, cfg, device=device,
                                    resume_ckpt=str(resume_path))
        save_checkpoint(
            ckpt_dir / f"{run_name}.pt", model, epoch=cfg["train"]["epochs"],
            best_metric=train_summary.get("best_val_acc"), config=cfg, seed=seed,
        )

    result = {"train": train_summary, "config_path": args.config}

    if not args.no_bench:
        amp_enabled, amp_dtype = _resolve_amp(cfg, device)
        dcfg = cfg["data"]
        in_ch = dcfg.get("channels", 3)
        input_shape = (in_ch, dcfg["num_frames"], dcfg["frame_size"], dcfg["frame_size"])
        eval_loader = loaders.get("test") or loaders.get("val")
        bench = bench_model(
            model, input_shape, device,
            loader=eval_loader,
            batch_sizes=cfg["bench"]["batch_sizes"],
            warmup_iters=cfg["bench"]["warmup_iters"],
            timed_iters=cfg["bench"]["timed_iters"],
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            measure_acc=eval_loader is not None,
        )
        result["bench"] = bench
        log.info("Bench: params=%s FLOPs(G)=%s single-clip FPS=%s acc=%s",
                 bench["params"]["total"], bench["flops"]["flops_g"],
                 bench.get("single_clip_fps"), bench["accuracy"].get("top1"))

    writer = ResultsWriter(root=cfg["output"]["root"], seed=seed)
    writer.write(result, group=cfg["output"]["group"], run_name=run_name, seed=seed)
    log.info("DONE. Artifacts in %s", run_dir)


if __name__ == "__main__":
    main()

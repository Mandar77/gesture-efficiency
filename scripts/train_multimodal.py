"""Train the multimodal fusion model on NVGesture (M7) with modality ablation.

The multimodal loader yields ``(modality_dict, label)`` batches, so this uses the
dict-aware ``multimodal_loss_fn`` and ``evaluate_multimodal``. The modality set is
controlled by ``data.modalities`` + ``model.kwargs.modalities`` (RGB / RGB+D /
RGB+D+IR), giving the §6.5 ablation.

Usage:
    python scripts/train_multimodal.py --config configs/multimodal_nvgesture.yaml \
        --set data.modalities='[rgb,depth]' model.kwargs.modalities='[rgb,depth]' \
              output.run_name=nvgesture_rgbd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401
import src.models  # noqa: F401
from src.data.loaders import build_dataloaders
from src.eval.metrics import evaluate  # noqa: F401 (kept for parity/imports)
from src.train.engine import _resolve_amp, train_model
from src.train.multimodal import evaluate_multimodal, multimodal_loss_fn
from src.utils import (
    ResultsWriter, build, get_logger, load_config, save_config,
    seed_everything, setup_file_logging,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.env import count_parameters

log = get_logger("scripts.train_multimodal")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
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

    model = build("model", cfg["model"].get("name", "multimodal_fusion"),
                  num_classes=cfg["data"]["num_classes"], **cfg["model"].get("kwargs", {}))
    model.to(device)
    params = count_parameters(model)
    log.info("MultiModalFusion modalities=%s params=%d",
             cfg["model"]["kwargs"].get("modalities"), params["total"])

    # Dict-aware training. Per-epoch eval uses evaluate_multimodal via callback.
    amp_enabled, amp_dtype = _resolve_amp(cfg, device)
    eval_loader = loaders.get("test") or loaders.get("val")

    def on_epoch_end(epoch, summary):
        if eval_loader is not None:
            m = evaluate_multimodal(model, eval_loader, device,
                                    amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            log.info("epoch %d multimodal val top1=%.3f top5=%.3f",
                     epoch, m["top1"], m["top5"])
            summary["val_top1"] = m["top1"]
            summary["val_top5"] = m["top5"]

    summary = train_model(model, loaders, cfg, device=device,
                          loss_fn=multimodal_loss_fn(cfg), on_epoch_end=on_epoch_end)

    ckpt_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints")) / cfg["output"]["group"]
    save_checkpoint(ckpt_dir / f"{run_name}.pt", model, epoch=cfg["train"]["epochs"],
                    best_metric=summary.get("best_val_acc"), config=cfg, seed=seed)

    # Final measured accuracy (dict-aware) + params/disk. FLOPs/latency for the
    # dict input are model-specific; we record the accuracy + size row here and
    # note that per-modality FLOPs can be measured with efficiency_bench on a
    # single-modality tensor if needed.
    final = evaluate_multimodal(model, eval_loader, device,
                                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                                return_confusion=True) if eval_loader else {"top1": None, "top5": None}
    from src.bench.efficiency_bench import measure_disk_size
    result = {
        "train": summary,
        "modalities": cfg["model"]["kwargs"].get("modalities"),
        "fusion": cfg["model"]["kwargs"].get("fusion"),
        "bench": {
            "params": {"total": params["total"], "trainable": params["trainable"],
                       "frozen": params["frozen"],
                       "trainable_pct": round(100.0 * params["trainable"] / max(params["total"], 1), 3)},
            "disk_size_mb": measure_disk_size(model),
            "flops": {"macs_g": None, "flops_g": None, "flops_backend": "n/a (multimodal dict)"},
            "accuracy": {"top1": final.get("top1"), "top5": final.get("top5")},
            "single_clip_fps": None, "single_clip_latency_ms": None, "peak_infer_vram_mb": None,
        },
        "per_class_acc": final.get("per_class_acc"),
        "config_path": args.config,
    }
    ResultsWriter(root=cfg["output"]["root"], seed=seed).write(
        result, group=cfg["output"]["group"], run_name=run_name, seed=seed)
    log.info("DONE multimodal run: %s (top1=%s)", run_name, final.get("top1"))


if __name__ == "__main__":
    main()

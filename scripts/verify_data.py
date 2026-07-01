"""Verify a prepared dataset: fetch one batch, sanity-check shapes/ranges, save
a montage image, and print the class histogram + official-split counts (M2
acceptance).

Usage:
    python scripts/verify_data.py --config configs/baseline_jester.yaml
    python scripts/verify_data.py --dataset jester --root data/jester
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import src.data  # noqa: F401
from src.data.loaders import build_dataloaders
from src.utils import get_logger, load_config

log = get_logger("scripts.verify_data")

# Official split sizes for cross-checking counts (BRIEF §2).
OFFICIAL = {
    "jester": {"train": 118562, "val": 14787, "test": 14743, "num_classes": 27},
    "nvgesture": {"train": 1050, "test": 482, "num_classes": 25},
    "shrec": {"num_classes": 14},
}


def save_montage(clip: torch.Tensor, out: Path):
    """clip: [C,T,H,W] -> save a horizontal strip of frames as PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never require a display / Tk
    import matplotlib.pyplot as plt

    c, t, h, w = clip.shape
    frames = clip.permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,C]
    # de-normalize (ImageNet) for display
    frames = frames * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    frames = np.clip(frames, 0, 1)
    n = min(t, 8)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(frames[i])
        axes[i].axis("off")
        axes[i].set_title(f"t={i}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    log.info("Saved batch montage -> %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config("configs/base.yaml", overrides=[
            f"data.name={args.dataset}", f"data.root={args.root}"])
    name = cfg["data"]["name"]
    root = Path(cfg["data"]["root"]) if cfg["data"].get("root") else None

    # Report integrity from prepared meta if present.
    if root and (root / "index_meta.json").exists():
        meta = json.loads((root / "index_meta.json").read_text(encoding="utf-8"))
        log.info("Prepared meta: %s", {k: meta.get(k) for k in ("num_classes", "splits")})
        for split, stats in meta.get("integrity", {}).items():
            hist = stats.get("class_histogram", {})
            log.info("  %s: %d classes present, %d clips", split, len(hist),
                     sum(hist.values()))
            off = OFFICIAL.get(name, {}).get(split)
            actual = meta["splits"].get(split, {}).get("num_clips")
            if off and actual:
                status = "OK" if abs(off - actual) <= max(0.02 * off, 5) else "MISMATCH"
                log.info("  %s count: prepared=%d official=%d [%s]",
                         split, actual, off, status)

    loaders = build_dataloaders(cfg, splits=("train",))
    train = loaders.get("train")
    if train is None:
        log.error("No train loader (data not prepared?). See download_data.py.")
        return
    batch = next(iter(train))
    x, y = batch
    if isinstance(x, dict):  # multimodal
        for mod, t in x.items():
            log.info("modality %s: shape=%s range=[%.3f,%.3f]", mod, tuple(t.shape),
                     float(t.min()), float(t.max()))
        first = next(iter(x.values()))[0]
    else:
        log.info("clip batch: shape=%s dtype=%s range=[%.3f,%.3f]",
                 tuple(x.shape), x.dtype, float(x.min()), float(x.max()))
        first = x[0]
    log.info("labels: %s", y[:8].tolist())
    save_montage(first, Path("experiments") / f"verify_{name}_batch.png")
    log.info("Data verification OK for %s.", name)


if __name__ == "__main__":
    main()

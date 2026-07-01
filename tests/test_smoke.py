"""Smoke + unit tests for the scaffold (M1).

Fast, CPU-friendly where possible. The end-to-end test uses the tiny synthetic
dataset and a tiny model so it runs in a couple of seconds and proves the
data -> model -> train -> bench -> results contract holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import src.data  # noqa: F401  registers datasets
import src.models  # noqa: F401  registers models
from src.bench.efficiency_bench import bench_model, measure_flops
from src.data.base import sample_frame_indices
from src.data.loaders import build_dataloaders
from src.train.engine import train_model
from src.utils import build, load_config, seed_everything
from src.utils.registry import REGISTRY


def test_frame_sampling_shapes_and_bounds():
    for mode in ("uniform", "random_uniform", "segment"):
        idx = sample_frame_indices(30, 16, mode=mode, training=False)
        assert len(idx) == 16
        assert all(0 <= i < 30 for i in idx)
    # Short clip: pad by repeating last index, still exactly T frames.
    idx = sample_frame_indices(5, 16, mode="uniform")
    assert len(idx) == 16 and max(idx) <= 4


def test_registry_has_core_components():
    assert "synthetic" in REGISTRY.available("dataset")
    assert "compact3dcnn" in REGISTRY.available("model")
    assert "dummy" in REGISTRY.available("model")


def test_config_inheritance_and_override():
    cfg = load_config("configs/smoke.yaml", overrides=["train.epochs=2"])
    assert cfg["train"]["epochs"] == 2
    assert cfg["data"]["name"] == "synthetic"
    # inherited from base.yaml
    assert cfg["bench"]["flops_backend"] == "fvcore"


def test_model_forward_and_flops():
    model = build("model", "compact3dcnn", num_classes=10, width=8, depth=2)
    x = torch.randn(2, 3, 8, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 10)
    logits, feats = model(x, return_features=True)
    assert feats.shape[0] == 2
    flops = measure_flops(model, (3, 8, 64, 64), torch.device("cpu"))
    assert flops["macs_g"] is not None and flops["macs_g"] > 0


def test_end_to_end_pipeline():
    seed_everything(0, deterministic_algorithms=False)
    cfg = load_config("configs/smoke.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(cfg)
    model = build("model", "compact3dcnn", num_classes=cfg["data"]["num_classes"],
                  **cfg["model"]["kwargs"])
    summary = train_model(model, loaders, cfg, device=device)
    assert "best_val_acc" in summary
    dcfg = cfg["data"]
    shape = (3, dcfg["num_frames"], dcfg["frame_size"], dcfg["frame_size"])
    result = bench_model(model, shape, device, loader=loaders["test"],
                         batch_sizes=[1], warmup_iters=2, timed_iters=5)
    assert result["params"]["total"] > 0
    assert result["single_clip_latency_ms"] > 0
    assert result["accuracy"]["top1"] is not None

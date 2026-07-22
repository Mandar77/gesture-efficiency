"""Build train/val/test DataLoaders from a resolved config dict.

Datasets are constructed via the registry with a `split=` kwarg. Every dataset
accepts the common data-config keys (num_frames, frame_size, frame_sampling,
num_classes) plus `root` for real datasets. Reproducible worker seeding is
wired in via `worker_init_fn`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.utils.logging_utils import get_logger
from src.utils.registry import build
from src.utils.seeding import worker_init_fn

log = get_logger("data.loaders")


def _resolve_norm(cfg: Dict[str, Any]):
    """Resolve the (mean, std) the model's pretrained backbone expects.

    Pretrained image backbones are trained with a specific input normalization;
    feeding the wrong stats shifts inputs out of distribution and caps accuracy.
    timm's AugReg ViTs (vit_small/base_patch16_224) want (0.5,0.5,0.5) — NOT
    ImageNet. We read the stats from the loaded backbone's ``pretrained_cfg`` so
    every backbone (ViT-S, ViT-B, DINOv2, future) gets ITS correct values, with
    no hardcoding of either ImageNet or (0.5,0.5,0.5).

    Returns (mean, std) tuples, or (None, None) when there is no timm backbone
    (e.g. the compact3dcnn baseline / multimodal), so the dataset falls back to
    its own default.
    """
    mcfg = cfg.get("model", {}) or {}
    backbone = (mcfg.get("kwargs", {}) or {}).get("backbone")
    # Explicit override in the data config always wins.
    dcfg = cfg.get("data", {}) or {}
    if dcfg.get("norm_mean") and dcfg.get("norm_std"):
        return tuple(dcfg["norm_mean"]), tuple(dcfg["norm_std"])
    if not backbone:
        return None, None
    try:
        import timm
        pcfg = timm.get_pretrained_cfg(backbone)
        mean = getattr(pcfg, "mean", None)
        std = getattr(pcfg, "std", None)
        if mean and std:
            log.info("Resolved normalization for backbone %s: mean=%s std=%s",
                     backbone, tuple(mean), tuple(std))
            return tuple(mean), tuple(std)
    except Exception as e:  # timm missing / unknown backbone -> dataset default
        log.warning("Could not resolve norm for backbone %s (%s); using dataset "
                    "default.", backbone, e)
    return None, None


def _make_dataset(cfg: Dict[str, Any], split: str):
    dcfg = cfg["data"]
    mean, std = _resolve_norm(cfg)
    kwargs = dict(
        split=split,
        num_frames=dcfg["num_frames"],
        frame_size=dcfg["frame_size"],
        frame_sampling=dcfg.get("frame_sampling", "segment"),
        num_classes=dcfg["num_classes"],
        root=dcfg.get("root"),
        seed=cfg.get("seed", 42),
    )
    if mean is not None and std is not None:
        kwargs["mean"] = mean
        kwargs["std"] = std
    # Pass through any smoke/limit knobs some datasets accept.
    for k in ("num_samples", "max_clips", "modalities"):
        if k in dcfg:
            kwargs[k] = dcfg[k]
    return build("dataset", dcfg["name"], **kwargs)


def build_dataloaders(
    cfg: Dict[str, Any],
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Optional[DataLoader]]:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    loaders: Dict[str, Optional[DataLoader]] = {}
    for split in splits:
        try:
            ds = _make_dataset(cfg, split)
        except FileNotFoundError as e:
            log.warning("Split %s unavailable: %s", split, e)
            loaders[split] = None
            continue
        is_train = split == "train"
        n_workers = dcfg.get("num_workers", 4)
        loader_kwargs = dict(
            batch_size=tcfg["batch_size"],
            shuffle=is_train,
            num_workers=n_workers,
            pin_memory=dcfg.get("pin_memory", True) and torch.cuda.is_available(),
            drop_last=is_train,
            worker_init_fn=worker_init_fn,
            generator=g,
            persistent_workers=n_workers > 0,
        )
        # prefetch_factor is only valid when num_workers > 0. Lowering it caps
        # the shared-memory each worker buffers ahead — important on Windows,
        # where too many workers x high prefetch can exhaust the commit limit
        # (RuntimeError "Couldn't open shared file mapping ... 1455").
        if n_workers > 0:
            loader_kwargs["prefetch_factor"] = dcfg.get("prefetch_factor", 2)
        loaders[split] = DataLoader(ds, **loader_kwargs)
        log.info("Built %s loader: %d clips", split, len(ds))
    return loaders

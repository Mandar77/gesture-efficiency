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


def _make_dataset(cfg: Dict[str, Any], split: str):
    dcfg = cfg["data"]
    kwargs = dict(
        split=split,
        num_frames=dcfg["num_frames"],
        frame_size=dcfg["frame_size"],
        frame_sampling=dcfg.get("frame_sampling", "segment"),
        num_classes=dcfg["num_classes"],
        root=dcfg.get("root"),
        seed=cfg.get("seed", 42),
    )
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
        loaders[split] = DataLoader(
            ds,
            batch_size=tcfg["batch_size"],
            shuffle=is_train,
            num_workers=dcfg.get("num_workers", 4),
            pin_memory=dcfg.get("pin_memory", True) and torch.cuda.is_available(),
            drop_last=is_train,
            worker_init_fn=worker_init_fn,
            generator=g,
            persistent_workers=dcfg.get("num_workers", 4) > 0,
        )
        log.info("Built %s loader: %d clips", split, len(ds))
    return loaders

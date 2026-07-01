"""SHREC'17 skeleton loader — BACKUP track (BRIEF §2.3), efficiency-framed only.

Loads 3D hand-skeleton sequences (22 joints x 3 coords) and resamples to N
frames, returning ([T, 22*3] or [T,22,3], label). This reuses the MediaPipe-
style landmark representation from the base project. Skeleton SOTA is saturated
(~97.7% SHREC 14G), so any results here are reported for *efficiency*, never as
an accuracy contribution.

Expected prepared index: <root>/index_<split>.csv with columns
    seq_path,label_idx,num_frames
where seq_path points to a whitespace-separated skeleton .txt (one frame/line).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.base import sample_frame_indices
from src.utils.logging_utils import get_logger
from src.utils.registry import register

log = get_logger("data.shrec")

NUM_JOINTS = 22


@register("dataset", "shrec")
class ShrecSkeletonDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 32,
        frame_sampling: str = "uniform",
        num_classes: int = 14,
        flatten: bool = True,
        seed: int = 42,
        max_clips: int | None = None,
        **_ignore,
    ):
        self.root = Path(root)
        idx = self.root / f"index_{split}.csv"
        if not idx.exists():
            raise FileNotFoundError(f"{idx} not found. Run prepare_shrec.py first.")
        self.entries: List[Tuple[str, int, int]] = []
        with open(idx, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.entries.append((row["seq_path"], int(row["label_idx"]),
                                     int(row["num_frames"])))
        if max_clips:
            self.entries = self.entries[:max_clips]
        self.training = split == "train"
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.num_classes = num_classes
        self.flatten = flatten
        self._seed = seed
        log.info("SHREC %s: %d sequences (%d-class)", split, len(self.entries), num_classes)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        seq_path, label, _ = self.entries[idx]
        data = np.loadtxt(seq_path, dtype=np.float32)  # [T, 22*3]
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[0]
        rng = np.random.default_rng(self._seed * 3299 + idx)
        sel = sample_frame_indices(n, self.num_frames, mode=self.frame_sampling,
                                   training=self.training, rng=rng)
        seq = data[np.clip(sel, 0, n - 1)]  # [T, 66]
        # Normalize by centering on the wrist (joint 0) per frame for scale/trans
        seq = seq.reshape(self.num_frames, NUM_JOINTS, 3)
        seq = seq - seq[:, :1, :]
        if not self.flatten:
            out = seq.astype(np.float32)                    # [T,22,3]
        else:
            out = seq.reshape(self.num_frames, -1).astype(np.float32)  # [T,66]
        return torch.from_numpy(out), int(label)

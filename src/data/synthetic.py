"""Synthetic video-clip dataset for smoke tests and CI.

Generates deterministic random clips of shape [C, T, H, W] with class-correlated
signal so a tiny model can reach >chance accuracy in one epoch — proving the
full train/eval/bench pipeline works on the 8GB GPU *before* the 23GB Jester
download. This is NOT a results dataset; it is never used for reported numbers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.registry import register


@register("dataset", "synthetic")
class SyntheticGestureClips(Dataset):
    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 27,
        num_frames: int = 16,
        frame_size: int = 172,
        channels: int = 3,
        split: str = "train",
        seed: int = 42,
        **_ignore,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.channels = channels
        # Distinct seed per split so train/val/test don't overlap exactly.
        self._seed = seed + {"train": 0, "val": 1, "test": 2}.get(split, 0)
        self.labels = np.random.default_rng(self._seed).integers(
            0, num_classes, size=num_samples
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        label = int(self.labels[idx])
        rng = np.random.default_rng(self._seed * 100003 + idx)
        clip = rng.standard_normal(
            (self.channels, self.num_frames, self.frame_size, self.frame_size)
        ).astype(np.float32)
        # Inject a class-correlated bias so the model can actually learn.
        clip += (label / self.num_classes) * 0.5
        return torch.from_numpy(clip), label

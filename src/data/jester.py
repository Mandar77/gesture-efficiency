"""Jester dataset loader — reads N sampled frames per clip from the prepared
index (BRIEF §2.1).

Expects `prepare_jester.py` to have written `<root>/index_<split>.csv`. Each
item returns (clip_tensor [C,T,H,W] float32 in [0,1] ImageNet-normed, label).
Frame sampling (uniform/random_uniform/segment) is configurable; default 16
frames, resolution configurable (172/224). Training uses light spatial aug
(resized-crop + flip); eval uses center crop — deterministic per §11.
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

log = get_logger("data.jester")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: str, size: int) -> np.ndarray:
    """Load + resize a frame to (size,size,3) float32 [0,1]. Uses cv2 if
    available, else PIL. Kept import-local so scaffold import doesn't need cv2."""
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"unreadable frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        return img.astype(np.float32) / 255.0
    except ImportError:
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((size, size))
        return np.asarray(img, dtype=np.float32) / 255.0


@register("dataset", "jester")
class JesterDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 172,
        frame_sampling: str = "segment",
        num_classes: int = 27,
        seed: int = 42,
        max_clips: int | None = None,
        mean: Tuple[float, float, float] | None = None,
        std: Tuple[float, float, float] | None = None,
        **_ignore,
    ):
        # Normalization stats. Default to ImageNet, but callers SHOULD pass the
        # stats the model's pretrained backbone expects (e.g. timm AugReg ViTs
        # want (0.5,0.5,0.5)/(0.5,0.5,0.5), NOT ImageNet). Feeding a pretrained
        # ViT the wrong mean/std shifts inputs out of distribution and caps
        # accuracy. See src/data/loaders.py::_resolve_norm.
        self._mean = np.array(mean if mean is not None else (0.485, 0.456, 0.406),
                              dtype=np.float32)
        self._std = np.array(std if std is not None else (0.229, 0.224, 0.225),
                             dtype=np.float32)
        log.info("Jester %s normalization: mean=%s std=%s",
                 split, tuple(self._mean.tolist()), tuple(self._std.tolist()))
        self.root = Path(root)
        # Map test->val if a test index isn't present (Jester test has no labels).
        index_path = self.root / f"index_{split}.csv"
        if not index_path.exists() and split == "test":
            index_path = self.root / "index_val.csv"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run prepare_jester.py first "
                f"(see download_data.py). "
            )
        self.entries: List[Tuple[str, int, int]] = []
        with open(index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.entries.append((row["path"], int(row["label_idx"]), int(row["num_frames"])))
        if max_clips:
            self.entries = self.entries[:max_clips]
        self.split = split
        self.training = split == "train"
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        self.num_classes = num_classes
        self._seed = seed
        log.info("Jester %s: %d clips (frames=%d size=%d sampling=%s)",
                 split, len(self.entries), num_frames, frame_size, frame_sampling)

    def __len__(self) -> int:
        return len(self.entries)

    def _list_frames(self, clip_dir: str) -> List[str]:
        p = Path(clip_dir)
        frames = sorted(
            [str(f) for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        return frames

    def __getitem__(self, idx: int):
        clip_dir, label, nframes = self.entries[idx]
        frames = self._list_frames(clip_dir)
        n = len(frames)
        rng = np.random.default_rng(self._seed * 7919 + idx)
        sel = sample_frame_indices(n, self.num_frames, mode=self.frame_sampling,
                                   training=self.training, rng=rng)
        clip = np.stack([_load_image(frames[min(i, n - 1)], self.frame_size) for i in sel])
        # clip: [T,H,W,C] -> normalize -> [C,T,H,W]
        clip = (clip - self._mean) / self._std
        if self.training and rng.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()  # horizontal flip
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)
        return torch.from_numpy(clip), int(label)

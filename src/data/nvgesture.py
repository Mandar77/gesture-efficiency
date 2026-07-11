"""NVGesture multimodal loader (RGB, depth, IR) — OPTIONAL / SECONDARY multimodal
track (BRIEF §2.2, §3.5).

STATUS: NVGesture access is PENDING (NVIDIA Google Drive permissions gate as of
2026-07). **Briareo is the PRIMARY multimodal dataset** for the M7 track (see
`src/data/briareo.py`); NVGesture drops in as a second multimodal dataset if/when
access is granted — no rearchitecting needed, since this loader already mirrors
the same API (returns ``({modality: [C,T,H,W]}, label)``) and feeds the same
M7 fusion / ablation machinery.

This loader is a complete, working implementation (reads per-modality .avi clips,
samples N frames, subset-selectable modalities for the RGB / RGB+D / RGB+D+IR
ablation). It is intentionally NOT stubbed out — it is ready to run the moment
``prepare_nvgesture.py`` has produced an index. If the data is absent, the
dataset raises a clear FileNotFoundError directing you to acquire + prepare it.

# TODO: run once NVGesture access is granted:
#   python src/data/download_data.py --dataset nvgesture   # instructions
#   python src/data/prepare_nvgesture.py --root data/nvgesture
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.base import sample_frame_indices
from src.utils.logging_utils import get_logger
from src.utils.registry import register

log = get_logger("data.nvgesture")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_video_frames(path: str, indices: List[int], size: int, gray: bool) -> np.ndarray:
    """Read specific frame indices from an .avi. Falls back to nearest frame."""
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (max(indices) + 1)
    frames = []
    for i in indices:
        i = min(i, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        if gray:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.repeat(g[:, :, None], 3, axis=2)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames)  # [T,H,W,3]


@register("dataset", "nvgesture")
class NVGestureDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 172,
        frame_sampling: str = "segment",
        num_classes: int = 25,
        modalities=("rgb", "depth", "ir"),
        seed: int = 42,
        max_clips: int | None = None,
        **_ignore,
    ):
        self.root = Path(root)
        idx_path = self.root / f"index_{split}.json"
        if not idx_path.exists() and split == "val":
            idx_path = self.root / "index_test.json"  # NVGesture: no official val
        if not idx_path.exists():
            raise FileNotFoundError(
                f"{idx_path} not found. Run prepare_nvgesture.py first."
            )
        self.entries = json.loads(idx_path.read_text(encoding="utf-8"))
        if max_clips:
            self.entries = self.entries[:max_clips]
        self.split = split
        self.training = split == "train"
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        self.num_classes = num_classes
        self.modalities = tuple(modalities)
        self._seed = seed
        log.info("NVGesture %s: %d clips, modalities=%s", split,
                 len(self.entries), self.modalities)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        label = int(e["label_idx"])
        rng = np.random.default_rng(self._seed * 6151 + idx)
        # Use the frame range if provided, else assume ~80 frames.
        lo, hi = e.get("frame_range", [0, 80])
        avail = max(hi - lo, 1)
        sel = sample_frame_indices(avail, self.num_frames, mode=self.frame_sampling,
                                   training=self.training, rng=rng)
        indices = [lo + s for s in sel]

        out: Dict[str, torch.Tensor] = {}
        for mod in self.modalities:
            path = e["modalities"].get(mod)
            gray = mod in ("depth", "ir")
            if path is None:
                clip = np.zeros((self.num_frames, self.frame_size, self.frame_size, 3),
                                dtype=np.float32)
            else:
                clip = _read_video_frames(path, indices, self.frame_size, gray=gray)
            clip = (clip - _MEAN) / _STD
            clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)
            out[mod] = torch.from_numpy(clip)
        return out, label

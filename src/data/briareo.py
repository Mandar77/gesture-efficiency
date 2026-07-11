"""Briareo multimodal loader (RGB + ToF depth + IR) — PRIMARY multimodal track
(BRIEF M7). Mirrors the NVGesture loader API exactly, so the same M7 fusion /
ablation machinery drives it.

Briareo (Manganaro et al., ICIAP 2019): 12 gestures, 40 subjects, 3 repetitions
each, car-cockpit capture. Sequences are >=40 frames (typically ~51). We reuse
the shared configurable frame sampling (default 16, allow 8), identical to the
Jester loader.

Returns ({modality: [C, T, H, W]}, label), C=3. depth/IR are single-channel
expanded to 3 channels for shared-backbone encoders (matches NVGesture loader).

Modalities:
  - rgb   : NNN_rgb.png  (3-channel PNG)
  - depth : NNN_z.npz    (compressed float ToF depth; decompressed + normalized)
  - ir    : NNN_ir.png   (single-channel IR PNG)
  - leap  : optional 3D hand joints (tracking_data) — exposed but NOT default,
            surfaced as a flat [T, D] tensor when requested via modalities.

Requires `prepare_briareo.py` to have written index_<split>.json.
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

log = get_logger("data.briareo")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _list_frames(dir_path: str, exts=(".png", ".jpg", ".jpeg")) -> List[str]:
    p = Path(dir_path)
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)


def _load_rgb(path: str, size: int) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def _load_ir(path: str, size: int) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # IR PNGs may be 16-bit; normalize to [0,1] by max.
    img = img.astype(np.float32)
    img = img / (img.max() + 1e-6)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.repeat(img[:, :, None], 3, axis=2)


def _load_depth_npz(path: str, size: int) -> np.ndarray:
    """Load a compressed ToF depth frame (NNN_z.npz) -> [size,size,3] float [0,1].

    Depth is stored as a float array (metric depth). We take the first array in
    the archive, robustly normalize by its 99th percentile to tame ToF outliers,
    resize, and expand to 3 channels.
    """
    import cv2

    try:
        with np.load(path) as z:
            key = list(z.keys())[0]
            depth = np.asarray(z[key], dtype=np.float32)
    except Exception:
        return np.zeros((size, size, 3), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    hi = np.percentile(depth, 99) if depth.size else 1.0
    depth = np.clip(depth / (hi + 1e-6), 0.0, 1.0)
    depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.repeat(depth[:, :, None], 3, axis=2)


@register("dataset", "briareo")
class BriareoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 172,
        frame_sampling: str = "segment",
        num_classes: int = 12,
        modalities=("rgb", "depth", "ir"),
        seed: int = 42,
        max_clips: int | None = None,
        **_ignore,
    ):
        self.root = Path(root)
        idx = self.root / f"index_{split}.json"
        # No official 'val' name mismatch: prepare writes index_val.json for the
        # 'validation' split dir; test uses index_test.json.
        if not idx.exists():
            raise FileNotFoundError(
                f"{idx} not found. Run prepare_briareo.py first "
                "(see download_data.py --dataset briareo)."
            )
        self.entries = json.loads(idx.read_text(encoding="utf-8"))
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
        log.info("Briareo %s: %d clips, modalities=%s (frames=%d size=%d)",
                 split, len(self.entries), self.modalities, num_frames, frame_size)

    def __len__(self) -> int:
        return len(self.entries)

    def _frame_tensor(self, mod: str, mdir: str, indices, n_avail: int) -> torch.Tensor:
        if mod == "rgb":
            files = _list_frames(mdir)
            loader = _load_rgb
        elif mod == "ir":
            files = _list_frames(mdir)
            loader = _load_ir
        elif mod == "depth":
            files = sorted(str(f) for f in Path(mdir).iterdir()
                           if f.suffix.lower() == ".npz") if Path(mdir).is_dir() else []
            loader = _load_depth_npz
        else:
            raise ValueError(f"unknown image modality {mod}")

        if not files:
            clip = np.zeros((self.num_frames, self.frame_size, self.frame_size, 3),
                            dtype=np.float32)
        else:
            nf = len(files)
            clip = np.stack([loader(files[min(i, nf - 1)], self.frame_size)
                             for i in indices])
        if mod == "rgb":
            clip = (clip - _MEAN) / _STD  # ImageNet norm for RGB backbone
        clip = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)  # [C,T,H,W]
        return torch.from_numpy(clip)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        label = int(e["label_idx"])
        n_avail = int(e["num_frames"])
        rng = np.random.default_rng(self._seed * 5779 + idx)
        indices = sample_frame_indices(n_avail, self.num_frames, mode=self.frame_sampling,
                                       training=self.training, rng=rng)

        out: Dict[str, torch.Tensor] = {}
        for mod in self.modalities:
            if mod == "leap":
                out["leap"] = self._load_leap(e["modalities"].get("leap"), indices)
                continue
            mdir = e["modalities"].get(mod)
            out[mod] = self._frame_tensor(mod, mdir, indices, n_avail) if mdir else \
                torch.zeros((3, self.num_frames, self.frame_size, self.frame_size))
        return out, label

    def _load_leap(self, leap_dir, indices) -> torch.Tensor:
        """Optional Leap 3D-joint modality -> [T, D] flat per-frame joint vector.

        Best-effort: reads per-frame tracking files if present; otherwise returns
        zeros. Not depended on by the default RGB+D+IR track.
        """
        T = self.num_frames
        if not leap_dir or not Path(leap_dir).is_dir():
            return torch.zeros((T, 63))  # 21 joints x 3, placeholder width
        files = sorted(str(f) for f in Path(leap_dir).iterdir()
                       if f.suffix.lower() in {".txt", ".json", ".npy"})
        if not files:
            return torch.zeros((T, 63))
        feats = []
        for i in indices:
            fp = files[min(i, len(files) - 1)]
            try:
                if fp.endswith(".npy"):
                    v = np.load(fp).astype(np.float32).ravel()
                else:
                    v = np.loadtxt(fp, dtype=np.float32).ravel()
            except Exception:
                v = np.zeros(63, dtype=np.float32)
            feats.append(v)
        # pad/truncate to a common width
        w = max(len(v) for v in feats) if feats else 63
        arr = np.zeros((T, w), dtype=np.float32)
        for t, v in enumerate(feats):
            arr[t, :len(v)] = v
        return torch.from_numpy(arr)

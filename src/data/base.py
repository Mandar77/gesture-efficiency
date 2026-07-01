"""Common frame-sampling utilities and a video-clip dataset base class.

A "clip" is a directory (or entry) with N ordered frames. Datasets build an
on-disk index of (path, label, num_frames); this base handles the shared
concern of *which* frames to sample and how to stack them into a tensor of
shape [C, T, H, W] (channels-first, time-major) for 3D-CNN / video-ViT inputs.

Frame sampling modes (BRIEF §2 data-handling):
    - uniform:        evenly spaced indices, deterministic (used for val/test).
    - random_uniform: uniform grid + per-clip random jitter (train aug).
    - segment:        TSN-style — split into T segments, pick one frame per
                      segment (random in train, center in eval).
"""

from __future__ import annotations

from typing import List

import numpy as np


def sample_frame_indices(
    num_frames_available: int,
    num_frames_wanted: int,
    mode: str = "segment",
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> List[int]:
    """Return `num_frames_wanted` frame indices into a clip of length
    `num_frames_available`. Short clips are handled by repeating the last frame
    (padding), which is logged as a corrupt/short-clip case at prepare time.
    """
    rng = rng or np.random.default_rng()
    n = max(int(num_frames_available), 1)
    t = int(num_frames_wanted)

    if n <= t:
        # Too few frames: take all, then pad by repeating the last index.
        idx = list(range(n)) + [n - 1] * (t - n)
        return idx[:t]

    if mode == "uniform":
        idx = np.linspace(0, n - 1, t).round().astype(int)
        return idx.tolist()

    if mode == "random_uniform":
        base = np.linspace(0, n - 1, t)
        if training:
            step = (n - 1) / max(t, 1)
            jitter = rng.uniform(-step / 2, step / 2, size=t)
            base = np.clip(base + jitter, 0, n - 1)
        return base.round().astype(int).tolist()

    if mode == "segment":
        # TSN: divide the clip into t equal segments, pick one frame per segment.
        bounds = np.linspace(0, n, t + 1).astype(int)
        idx = []
        for i in range(t):
            lo, hi = bounds[i], max(bounds[i + 1], bounds[i] + 1)
            if training:
                idx.append(int(rng.integers(lo, hi)))
            else:
                idx.append(int((lo + hi - 1) // 2))
        return idx

    raise ValueError(f"Unknown frame_sampling mode: {mode!r}")

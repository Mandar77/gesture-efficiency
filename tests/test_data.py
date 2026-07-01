"""Data-layer tests: build tiny fake Jester/SHREC dirs on disk and verify the
prepare scripts + loaders produce correct shapes and honour official-style
splits. Proves the real-data code path without the multi-GB download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import src.data  # noqa: F401


def _make_fake_jester(root: Path, n_clips=6, n_frames=12, classes=("Swiping Left", "Swiping Right", "No gesture")):
    from PIL import Image

    frame_root = root / "20bn-jester-v1"
    frame_root.mkdir(parents=True)
    # labels file
    (root / "jester-v1-labels.csv").write_text("\n".join(classes), encoding="utf-8")
    train_lines, val_lines = [], []
    for c in range(n_clips):
        clip_id = str(1000 + c)
        cdir = frame_root / clip_id
        cdir.mkdir()
        for f in range(n_frames):
            img = Image.fromarray((np.random.rand(30, 40, 3) * 255).astype("uint8"))
            img.save(cdir / f"{f:05d}.jpg")
        label = classes[c % len(classes)]
        (train_lines if c % 2 == 0 else val_lines).append(f"{clip_id};{label}")
    (root / "jester-v1-train.csv").write_text("\n".join(train_lines), encoding="utf-8")
    (root / "jester-v1-validation.csv").write_text("\n".join(val_lines), encoding="utf-8")


def test_prepare_and_load_jester(tmp_path):
    pytest.importorskip("PIL")
    root = tmp_path / "jester"
    _make_fake_jester(root)
    from src.data.prepare_jester import prepare

    meta = prepare(root, min_frames=8)
    assert meta["num_classes"] == 3
    assert meta["splits"]["train"]["num_clips"] == 3
    assert meta["splits"]["val"]["num_clips"] == 3

    from src.utils.registry import build

    ds = build("dataset", "jester", root=str(root), split="train",
               num_frames=8, frame_size=32, num_classes=3)
    clip, label = ds[0]
    assert tuple(clip.shape) == (3, 8, 32, 32)
    assert 0 <= label < 3


def test_prepare_and_load_shrec(tmp_path):
    import csv

    root = tmp_path / "shrec"
    root.mkdir()
    seqs = []
    for i in range(4):
        p = root / f"seq_{i}.txt"
        np.savetxt(p, np.random.rand(20, 66).astype("float32"))
        seqs.append((str(p), i % 2, 20))
    with open(root / "index_train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_path", "label_idx", "num_frames"])
        w.writerows(seqs)

    from src.utils.registry import build

    ds = build("dataset", "shrec", root=str(root), split="train",
               num_frames=16, num_classes=2)
    seq, label = ds[0]
    assert tuple(seq.shape) == (16, 66)
    assert label in (0, 1)

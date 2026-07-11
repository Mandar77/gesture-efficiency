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


def _make_fake_briareo(rgb_root, tof_root, split_dir="train", sessions=("000", "001"),
                       gestures=3, reps=2, n_frames=10):
    """Build a tiny Briareo-format tree: <mod>/<split>/<sess>/gNN/<rep>/... ."""
    from PIL import Image

    for sess in sessions:
        for g in range(gestures):
            for r in range(reps):
                rep = f"{r:02d}"
                gd = f"g{g:02d}"
                rgb_dir = rgb_root / split_dir / sess / gd / rep / "rgb"
                depth_dir = tof_root / split_dir / sess / gd / rep / "tof" / "depth"
                ir_dir = tof_root / split_dir / sess / gd / rep / "tof" / "ir"
                for d in (rgb_dir, depth_dir, ir_dir):
                    d.mkdir(parents=True)
                for f in range(n_frames):
                    Image.fromarray((np.random.rand(20, 24, 3) * 255).astype("uint8")).save(
                        rgb_dir / f"{f:03d}_rgb.png")
                    Image.fromarray((np.random.rand(20, 24) * 255).astype("uint8")).save(
                        ir_dir / f"{f:03d}_ir.png")
                    np.savez_compressed(depth_dir / f"{f:03d}_z.npz",
                                        z=np.random.rand(20, 24).astype("float32"))


def test_prepare_and_load_briareo(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("cv2")
    rgb_root = tmp_path / "rgb"
    tof_root = tmp_path / "tof"
    _make_fake_briareo(rgb_root, tof_root, split_dir="train")
    from src.data.prepare_briareo import prepare

    out = tmp_path / "briareo"
    meta = prepare(rgb_root, tof_root, None, out, min_frames=8)
    # 2 sessions x 3 gestures x 2 reps = 12 train clips
    assert meta["splits"]["train"]["num_clips"] == 12
    assert meta["num_classes"] == 12

    from src.utils.registry import build

    ds = build("dataset", "briareo", root=str(out), split="train",
               num_frames=8, frame_size=16, num_classes=12,
               modalities=("rgb", "depth", "ir"))
    x, label = ds[0]
    assert set(x.keys()) == {"rgb", "depth", "ir"}
    for m in ("rgb", "depth", "ir"):
        assert tuple(x[m].shape) == (3, 8, 16, 16)
    assert 0 <= label < 12
    # depth decoded from .npz should be non-trivial
    assert float((x["depth"] != 0).float().mean()) > 0.0

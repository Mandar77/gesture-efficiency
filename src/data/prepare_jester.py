"""Build an on-disk index for Jester + integrity report (BRIEF M2).

Reads the official label CSVs, resolves each clip's frame directory, counts
frames, detects missing/short/empty clips, logs the class histogram, and writes
a compact index (CSV + JSON meta) that the loader consumes. Uses the OFFICIAL
splits; never mixes them.

Output:
    <root>/index_train.csv, index_val.csv   ("clip_id,label_idx,num_frames,path")
    <root>/index_meta.json                  (class names, counts, integrity stats)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

log = get_logger("data.prepare_jester")

# Candidate frame-root subdirectories seen across Jester mirrors.
_FRAME_SUBDIRS = ["20bn-jester-v1", "jester-v1", "Train", "frames", "."]
_LABEL_FILES = ["jester-v1-labels.csv", "jester-v1-labels-quik.csv", "labels.csv"]
_SPLIT_FILES = {
    "train": ["jester-v1-train.csv", "train.csv"],
    "val": ["jester-v1-validation.csv", "validation.csv", "val.csv"],
}


def _first_existing(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _find_frame_root(root: Path) -> Path:
    for sub in _FRAME_SUBDIRS:
        cand = root / sub
        if cand.is_dir() and any(cand.iterdir()):
            # Heuristic: contains numbered clip subdirs.
            for child in cand.iterdir():
                if child.is_dir():
                    return cand
    raise FileNotFoundError(
        f"Could not locate Jester frame root under {root}. Expected one of "
        f"{_FRAME_SUBDIRS} containing per-clip frame folders."
    )


def _load_label_map(root: Path) -> Dict[str, int]:
    lf = _first_existing(root, _LABEL_FILES)
    if lf is None:
        raise FileNotFoundError(f"No labels file among {_LABEL_FILES} under {root}")
    labels = [ln.strip() for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {name: i for i, name in enumerate(labels)}


def _read_split_csv(path: Path) -> List[Tuple[str, str]]:
    """Jester CSV rows are 'clip_id;label text'. Returns (clip_id, label)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        clip_id = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else ""
        rows.append((clip_id, label))
    return rows


def _count_frames(clip_dir: Path) -> int:
    if not clip_dir.is_dir():
        return 0
    return sum(1 for p in clip_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def prepare(root: str | Path, min_frames: int = 8) -> Dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Jester root {root} does not exist. See "
                                "`python src/data/download_data.py --dataset jester`.")
    frame_root = _find_frame_root(root)
    label_map = _load_label_map(root)
    log.info("Frame root: %s | %d classes", frame_root, len(label_map))

    meta = {"dataset": "jester", "frame_root": str(frame_root),
            "num_classes": len(label_map), "classes": list(label_map.keys()),
            "splits": {}, "integrity": {}}

    for split, names in _SPLIT_FILES.items():
        sf = _first_existing(root, names)
        if sf is None:
            log.warning("Split %s: no CSV among %s — skipping", split, names)
            continue
        rows = _read_split_csv(sf)
        index_rows = []
        hist = Counter()
        short = missing = 0
        for clip_id, label in rows:
            if label not in label_map:
                # test split has no labels; store -1
                label_idx = -1
            else:
                label_idx = label_map[label]
            clip_dir = frame_root / clip_id
            nframes = _count_frames(clip_dir)
            if nframes == 0:
                missing += 1
                continue
            if nframes < min_frames:
                short += 1  # kept, but flagged (loader pads short clips)
            index_rows.append((clip_id, label_idx, nframes, str(clip_dir)))
            if label_idx >= 0:
                hist[label_idx] += 1

        out_csv = root / f"index_{split}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "label_idx", "num_frames", "path"])
            w.writerows(index_rows)
        log.info("Split %s: %d clips indexed (%d missing, %d short<%d) -> %s",
                 split, len(index_rows), missing, short, min_frames, out_csv)
        meta["splits"][split] = {"num_clips": len(index_rows), "index": str(out_csv)}
        meta["integrity"][split] = {"missing": missing, "short": short,
                                    "class_histogram": dict(sorted(hist.items()))}

    with open(root / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote index_meta.json. Integrity: %s",
             {k: v for k, v in meta["integrity"].items()})
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/jester")
    ap.add_argument("--min-frames", type=int, default=8)
    args = ap.parse_args()
    prepare(args.root, min_frames=args.min_frames)


if __name__ == "__main__":
    main()

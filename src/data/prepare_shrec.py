"""Build an on-disk index for SHREC'17 skeleton sequences (BACKUP track).

Parses the official train_gestures.txt / test_gestures.txt split files, which
list per-sequence: gesture_id finger_id subject_id essai_id [14-label] [28-label]
frames. Resolves each sequence's `skeletons_world.txt` (22 joints x 3 = 66
values per frame), counts frames, and writes index_<split>.csv columns
    seq_path,label_idx,num_frames

Use `--num-classes 14` (default) or 28. Efficiency-framed only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

from src.utils.logging_utils import get_logger

log = get_logger("data.prepare_shrec")

_SPLIT_FILES = {
    "train": ["train_gestures.txt", "train.txt"],
    "test": ["test_gestures.txt", "test.txt"],
}
_SEQ_FILES = ["skeletons_world.txt", "skeleton_world.txt", "skeletons.txt"]


def _first_existing(root: Path, names) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _seq_dir(root: Path, gid, fid, sid, eid) -> Path:
    # Official SHREC'17 layout.
    return (root / f"gesture_{gid}" / f"finger_{fid}"
            / f"subject_{sid}" / f"essai_{eid}")


def prepare(root: str | Path, num_classes: int = 14) -> dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SHREC root {root} missing. See download_data.py.")
    label_col = 4 if num_classes == 14 else 5  # 0-idx col after 4 ids
    meta = {"dataset": "shrec", "num_classes": num_classes, "splits": {}}
    for split, names in _SPLIT_FILES.items():
        sf = _first_existing(root, names)
        if sf is None:
            log.warning("Split %s: no file among %s — skipping", split, names)
            continue
        rows = []
        for line in sf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            gid, fid, sid, eid = parts[0], parts[1], parts[2], parts[3]
            label = int(parts[label_col]) - 1  # 1-indexed -> 0-indexed
            sdir = _seq_dir(root, gid, fid, sid, eid)
            seq = _first_existing(sdir, _SEQ_FILES)
            if seq is None:
                continue
            nframes = sum(1 for _ in seq.open(encoding="utf-8"))
            rows.append((str(seq), label, nframes))
        out = root / f"index_{split}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seq_path", "label_idx", "num_frames"])
            w.writerows(rows)
        log.info("Split %s: %d sequences -> %s", split, len(rows), out)
        meta["splits"][split] = {"num_seqs": len(rows), "index": str(out)}
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/shrec")
    ap.add_argument("--num-classes", type=int, default=14, choices=[14, 28])
    args = ap.parse_args()
    prepare(args.root, num_classes=args.num_classes)


if __name__ == "__main__":
    main()

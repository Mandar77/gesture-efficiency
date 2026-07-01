"""Build an on-disk index for NVGesture (RGB-D + IR) from the official .lst
split files (BRIEF M2, §2.2).

The NVGesture release ships train/test `.lst` files. Each entry references a
clip directory and per-modality video files (sk_color, sk_depth, and IR where
available) plus a label and frame range. We parse those into a JSON index the
multimodal loader consumes, and log the class histogram + integrity.

Because .lst formats vary slightly across mirrors, the parser is defensive:
it extracts `path:...`, `label:...`, and any `*_used:start,end` fields, and
falls back to scanning the clip directory for known modality filenames.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger("data.prepare_nvgesture")

_LST_FILES = {
    "train": ["nvgesture_train_correct_cvpr2016_v2.lst",
              "nvgesture_train_correct.lst", "train.lst"],
    "test": ["nvgesture_test_correct_cvpr2016_v2.lst",
             "nvgesture_test_correct.lst", "test.lst"],
}
# Modality key -> candidate filename stems in each clip dir.
_MODALITY_FILES = {
    "rgb": ["sk_color.avi", "color.avi", "sk_color_all.avi"],
    "depth": ["sk_depth.avi", "depth.avi", "sk_depth_all.avi"],
    "ir": ["sk_ir.avi", "ir.avi", "duo_left.avi"],
}


def _first_existing(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _parse_lst_line(line: str, root: Path) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None
    entry: Dict = {"modalities": {}}
    # path
    m = re.search(r"path:(\S+)", line)
    rel = m.group(1) if m else line.split()[0]
    rel = rel.replace("./", "").strip()
    clip_dir = (root / rel)
    entry["clip_dir"] = str(clip_dir)
    # label (1-indexed in NVGesture -> convert to 0-indexed)
    ml = re.search(r"label:(\d+)", line)
    if ml:
        entry["label_idx"] = int(ml.group(1)) - 1
    else:
        # fall back: try class_XX in path
        mc = re.search(r"class[_/](\d+)", rel)
        entry["label_idx"] = int(mc.group(1)) - 1 if mc else -1
    # frame range (color_used:start,end) — best-effort
    fr = re.search(r"color[_a-z]*:(\d+),(\d+)", line)
    if fr:
        entry["frame_range"] = [int(fr.group(1)), int(fr.group(2))]
    # resolve modality files on disk
    for mod, cands in _MODALITY_FILES.items():
        f = _first_existing(clip_dir, cands)
        if f is not None:
            entry["modalities"][mod] = str(f)
    return entry


def prepare(root: str | Path) -> Dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"NVGesture root {root} missing. See "
                                "`python src/data/download_data.py --dataset nvgesture`.")
    meta = {"dataset": "nvgesture", "num_classes": 25, "splits": {}, "integrity": {}}
    for split, names in _LST_FILES.items():
        lst = _first_existing(root, names)
        if lst is None:
            log.warning("Split %s: no .lst among %s under %s — skipping",
                        split, names, root)
            continue
        entries = []
        hist = Counter()
        missing_mod = Counter()
        for line in lst.read_text(encoding="utf-8").splitlines():
            e = _parse_lst_line(line, root)
            if e is None:
                continue
            if not e["modalities"]:
                missing_mod["no_modality_files"] += 1
                continue
            for mod in ("rgb", "depth", "ir"):
                if mod not in e["modalities"]:
                    missing_mod[mod] += 1
            entries.append(e)
            if e["label_idx"] >= 0:
                hist[e["label_idx"]] += 1
        out = root / f"index_{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
        log.info("Split %s: %d clips -> %s (missing modality files: %s)",
                 split, len(entries), out, dict(missing_mod))
        meta["splits"][split] = {"num_clips": len(entries), "index": str(out)}
        meta["integrity"][split] = {"class_histogram": dict(sorted(hist.items())),
                                    "missing_modalities": dict(missing_mod)}
    with open(root / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote NVGesture index_meta.json")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/nvgesture")
    args = ap.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()

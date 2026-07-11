"""Build an on-disk index for the Briareo dataset (BRIEF M7, multimodal primary).

Briareo (Manganaro et al., ICIAP 2019, "Hand Gestures for the Human-Car
Interaction: the Briareo dataset") — 12 dynamic gestures, 40 subjects, each
gesture performed 3 times, captured in a car cockpit with:
  - RGB      (Leap Motion-adjacent RGB sensor)   -> <rgb>/<split>/<sess>/gNN/<rep>/rgb/NNN_rgb.png
  - ToF depth (Pico Flexx)                         -> <tof>/<split>/<sess>/gNN/<rep>/tof/depth/NNN_z.npz
  - ToF IR   (Pico Flexx infrared)                 -> <tof>/<split>/<sess>/gNN/<rep>/tof/ir/NNN_ir.png
  - Leap Motion raw/rectified IR + 3D hand joints  -> <leap_motion>/... (optional extra modality)

Our multimodal track uses **RGB + depth + IR** to parallel the NVGesture-style
RGB-D+IR setup. Leap 3D joints are exposed as an optional modality but not
depended on.

Split: this build uses the **official train / validation / test session split
that ships with the distribution** (train=26, val=6, test=8 disjoint sessions =
40 subjects). This is subject-disjoint (no subject appears in more than one
split), so there is no cross-split subject leakage. We fix this split and state
it explicitly (Briareo has no single universal split beyond the shipped one).

On-disk depth note: depth frames are compressed float arrays (`NNN_z.npz`), not
images — the loader decompresses + normalizes them. IR/RGB are PNGs.

Output (under --root):
    index_train.json, index_val.json, index_test.json  (list of clip records)
    index_meta.json                                     (class hist + integrity)

Each clip record:
    {clip_id, label_idx, num_frames,
     modalities: {rgb: dir, depth: dir, ir: dir, leap: dir|null}}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logging_utils import get_logger

log = get_logger("data.prepare_briareo")

NUM_CLASSES = 12  # g00..g11 ; g12_test is a no-gesture/test folder, excluded
SPLIT_DIRNAMES = {"train": "train", "val": "validation", "test": "test"}
_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _count_rgb_frames(rep_dir: Path) -> int:
    d = rep_dir / "rgb"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _resolve_modalities(rgb_root: Path, tof_root: Path, leap_root: Optional[Path],
                        split_dir: str, sess: str, gesture: str, rep: str) -> Dict:
    rgb_dir = rgb_root / split_dir / sess / gesture / rep / "rgb"
    depth_dir = tof_root / split_dir / sess / gesture / rep / "tof" / "depth"
    ir_dir = tof_root / split_dir / sess / gesture / rep / "tof" / "ir"
    mods: Dict[str, Optional[str]] = {}
    if rgb_dir.is_dir():
        mods["rgb"] = str(rgb_dir)
    if depth_dir.is_dir():
        mods["depth"] = str(depth_dir)
    if ir_dir.is_dir():
        mods["ir"] = str(ir_dir)
    if leap_root is not None:
        leap_dir = leap_root / split_dir / sess / gesture / rep / "leap_motion" / "tracking_data"
        mods["leap"] = str(leap_dir) if leap_dir.is_dir() else None
    return mods


def prepare(rgb_root, tof_root, leap_root, out_root, min_frames: int = 8) -> Dict:
    rgb_root = Path(rgb_root)
    tof_root = Path(tof_root)
    leap_root = Path(leap_root) if leap_root else None
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not rgb_root.exists() or not tof_root.exists():
        raise FileNotFoundError(
            f"Briareo rgb/tof roots missing (rgb={rgb_root}, tof={tof_root}). "
            "See `python src/data/download_data.py --dataset briareo`."
        )

    meta = {"dataset": "briareo", "num_classes": NUM_CLASSES,
            "split_policy": "official shipped session split (subject-disjoint: "
                            "train=26 / val=6 / test=8 sessions = 40 subjects)",
            "modalities": ["rgb", "depth", "ir", "leap(optional)"],
            "splits": {}, "integrity": {}}

    for split, split_dir in SPLIT_DIRNAMES.items():
        sroot = rgb_root / split_dir
        if not sroot.is_dir():
            log.warning("Briareo split %s (%s) missing under %s — skipping.",
                        split, split_dir, rgb_root)
            continue
        records: List[Dict] = []
        hist = Counter()
        short = missing_mod = 0
        for sess_dir in sorted(p for p in sroot.iterdir() if p.is_dir()):
            sess = sess_dir.name
            for g in range(NUM_CLASSES):
                gesture = f"g{g:02d}"
                gdir = sess_dir / gesture
                if not gdir.is_dir():
                    continue
                for rep_dir in sorted(p for p in gdir.iterdir() if p.is_dir()):
                    rep = rep_dir.name
                    n = _count_rgb_frames(rep_dir)
                    if n == 0:
                        continue
                    mods = _resolve_modalities(rgb_root, tof_root, leap_root,
                                               split_dir, sess, gesture, rep)
                    if "rgb" not in mods or "depth" not in mods or "ir" not in mods:
                        missing_mod += 1
                    if n < min_frames:
                        short += 1  # kept; loader pads short sequences
                    records.append({
                        "clip_id": f"{sess}_{gesture}_{rep}",
                        "label_idx": g,
                        "num_frames": n,
                        "modalities": mods,
                    })
                    hist[g] += 1

        out_json = out_root / f"index_{split}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=1)
        log.info("Briareo %s: %d clips (%d missing-modality, %d short<%d) -> %s",
                 split, len(records), missing_mod, short, min_frames, out_json)
        meta["splits"][split] = {"num_clips": len(records), "index": str(out_json)}
        meta["integrity"][split] = {"missing_modality": missing_mod, "short": short,
                                    "class_histogram": dict(sorted(hist.items()))}

    with open(out_root / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote Briareo index_meta.json. Integrity: %s",
             {k: {kk: vv for kk, vv in v.items() if kk != 'class_histogram'}
              for k, v in meta["integrity"].items()})
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rgb-root", required=True, help="Briareo rgb/ dir (has train/validation/test).")
    ap.add_argument("--tof-root", required=True, help="Briareo tof/ dir (depth + ir).")
    ap.add_argument("--leap-root", default=None, help="Optional leap_motion/ dir.")
    ap.add_argument("--out-root", default="data/briareo", help="Where to write the index.")
    ap.add_argument("--min-frames", type=int, default=8)
    args = ap.parse_args()
    prepare(args.rgb_root, args.tof_root, args.leap_root, args.out_root,
            min_frames=args.min_frames)


if __name__ == "__main__":
    main()

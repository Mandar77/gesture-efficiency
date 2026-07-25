"""Load committed result JSONs + reported baselines into one normalized DataFrame.

Reads every ``experiments/<group>/*.json`` record (skipping the shared
``all_results.csv`` and any non-record JSON), flattens the nested ``bench`` dict
into flat columns, and merges in the reported external baselines
(``source='reported'``) from :mod:`src.viz.reported_baselines`.

Missing measurements stay ``NaN`` so downstream tables can render them as
"TODO" and plots can skip them — we never fabricate a value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.viz.reported_baselines import reported_baselines

log = get_logger("viz.loader")

# Normalized column order for the returned DataFrame.
COLUMNS: List[str] = [
    "run_name",
    "group",
    "source",
    "dataset",
    "params_total",
    "params_trainable_pct",
    "flops_g",
    "macs_g",
    "top1",
    "single_clip_fps",
    "single_clip_latency_ms",
    "peak_infer_vram_mb",
    "disk_size_mb",
    "gpu_name",
    "notes",
]


def _get(d: Optional[Dict[str, Any]], *path: str) -> Any:
    """Safely walk a nested dict; return NaN if any key is missing or None."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return np.nan
        cur = cur[key]
    return np.nan if cur is None else cur


def _frontier_note(record: Dict[str, Any]) -> Any:
    """Flag compression cells that are NOT real frontier operating points, so the
    Pareto plot can exclude them while the honest data stays logged.

    Two measured negative results (see SANITY.md, M6 compression framing) must
    never appear as tradeoff points on the frontier:
      * structured pruning WITHOUT fine-tune (prune_ratio > 0): collapses this
        3.1M-param student to near-random (30%->9.9%, 50%->3.6%) with no on-disk
        size reduction — broken, not a tradeoff.
      * INT8 fallback: no quantized::conv3d kernel, so only the final Linear
        quantizes (Conv3d stays fp32) — ~0 size/latency benefit, not a real
        low-precision operating point.
    Also excludes smoke/pipeline-test runs (synthetic data, tiny model) — e.g.
    `smoke_compact3dcnn` (~6 % on synthetic data) — which are validation
    artifacts, not measured operating points on Jester.

    Returns a short reason string when the row is NOT frontier-eligible, else NaN.
    """
    run_name = str(record.get("run_name", ""))
    rn = run_name.lower()
    if "smoke" in rn:
        return "NOT_FRONTIER: smoke/pipeline-test run (synthetic data, not a real point)"
    if "sanity" in rn:
        return "NOT_FRONTIER: 1-epoch sanity/diagnostic run (not a converged model)"
    comp = record.get("compress")
    if not isinstance(comp, dict):
        return np.nan  # not a compression record; normal run -> eligible
    ratio = comp.get("prune_ratio") or 0.0
    mode = str(comp.get("mode", ""))
    if ratio and float(ratio) > 0:
        return f"NOT_FRONTIER: pruned (ratio={ratio}, no fine-tune) — cratered/near-random"
    if mode.startswith("int8"):
        return "NOT_FRONTIER: int8 fallback (Conv3d unsupported; ~0 benefit)"
    return np.nan


def _record_to_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one committed result JSON record into a normalized row."""
    bench = record.get("bench") or {}
    env = record.get("env") or {}
    return {
        "run_name": record.get("run_name", np.nan),
        "group": record.get("group", np.nan),
        "source": "ours",
        "dataset": record.get("dataset", np.nan),
        "params_total": _get(bench, "params", "total"),
        "params_trainable_pct": _get(bench, "params", "trainable_pct"),
        "flops_g": _get(bench, "flops", "flops_g"),
        "macs_g": _get(bench, "flops", "macs_g"),
        "top1": _get(bench, "accuracy", "top1"),
        "single_clip_fps": _get(bench, "single_clip_fps"),
        "single_clip_latency_ms": _get(bench, "single_clip_latency_ms"),
        "peak_infer_vram_mb": _get(bench, "peak_infer_vram_mb"),
        "disk_size_mb": _get(bench, "disk_size_mb"),
        "gpu_name": env.get("gpu_name", np.nan),
        "notes": _frontier_note(record),
    }


def _reported_to_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Map a reported-baseline dict onto the normalized schema."""
    return {
        "run_name": entry.get("run_name", np.nan),
        "group": "reported",
        "source": "reported",
        "dataset": entry.get("dataset", np.nan),
        "params_total": entry.get("params_total", np.nan),
        "params_trainable_pct": np.nan,
        "flops_g": entry.get("flops_g", np.nan),
        "macs_g": entry.get("macs_g", np.nan),
        "top1": entry.get("top1", np.nan),
        "single_clip_fps": np.nan,
        "single_clip_latency_ms": np.nan,
        "peak_infer_vram_mb": np.nan,
        "disk_size_mb": np.nan,
        "gpu_name": np.nan,
        # `notes` carries the citation + not-comparable caveat for reported rows.
        "notes": entry.get("citation", ""),
    }


def _looks_like_record(obj: Any) -> bool:
    """A committed result record is a dict with our canonical top-level keys."""
    return isinstance(obj, dict) and "run_name" in obj and "bench" in obj


def load_results(results_dir: str | Path = "experiments") -> pd.DataFrame:
    """Load all committed result JSONs + reported baselines into one DataFrame.

    Parameters
    ----------
    results_dir : path to the ``experiments/`` root containing ``<group>/*.json``.

    Returns
    -------
    pandas.DataFrame with the columns in :data:`COLUMNS`. Always includes the
    reported baselines, even when there are zero measured ("ours") runs.
    """
    results_dir = Path(results_dir)
    rows: List[Dict[str, Any]] = []

    if results_dir.exists():
        # Recurse so nested per-group dirs are picked up; skip the shared CSV
        # and any file that is not a canonical record.
        for json_path in sorted(results_dir.rglob("*.json")):
            if json_path.name == "all_results.csv":  # defensive; not a .json anyway
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping unreadable JSON %s: %s", json_path, exc)
                continue
            if not _looks_like_record(obj):
                log.debug("Skipping non-record JSON %s", json_path)
                continue
            rows.append(_record_to_row(obj))
    else:
        log.warning("Results dir does not exist: %s", results_dir)

    n_ours = len(rows)

    # Override per-run single-clip FPS/latency with the AUTHORITATIVE single-session
    # re-bench when available (see SANITY.md "AUTHORITATIVE latency/FPS"). Per-run
    # numbers are cross-run thermal artifacts and are NOT mutually comparable; the
    # re-bench measured all models back-to-back, warm, median-of-3. This affects
    # only the frontier figure's latency axis — the individual result JSONs are
    # left untouched (they honestly record what each run measured).
    rebench_path = results_dir / "rebench_frontier.json"
    if rebench_path.exists():
        try:
            rb = json.loads(rebench_path.read_text(encoding="utf-8"))
            by_name = {r["run_name"]: r for r in rb.get("rows", [])}
            # The KD / logit-KD / no-KD students are architecturally IDENTICAL at
            # inference (the distinction is training-only). Report ONE student
            # latency for all three — any per-run FPS spread (134/159/135) is pure
            # measurement noise, not distinct operating points. Use the
            # median-of-medians across the three student runs.
            stu = [r for r in rb.get("rows", []) if "student" in r["run_name"]
                   and r.get("bs1_fps_median") is not None]
            stu_fps = stu_ms = None
            if stu:
                fps_sorted = sorted(r["bs1_fps_median"] for r in stu)
                ms_sorted = sorted(r["bs1_latency_ms_median"] for r in stu
                                   if r.get("bs1_latency_ms_median") is not None)
                stu_fps = fps_sorted[len(fps_sorted) // 2]
                stu_ms = ms_sorted[len(ms_sorted) // 2] if ms_sorted else None
            n_over = 0
            for row in rows:
                name = row.get("run_name", "")
                if stu_fps is not None and "student" in name:
                    row["single_clip_fps"] = stu_fps       # one shared number
                    row["single_clip_latency_ms"] = stu_ms
                    n_over += 1
                    continue
                rbr = by_name.get(name)
                if rbr and rbr.get("bs1_fps_median") is not None:
                    row["single_clip_fps"] = rbr["bs1_fps_median"]
                    row["single_clip_latency_ms"] = rbr.get("bs1_latency_ms_median")
                    n_over += 1
            log.info("Applied re-bench FPS/latency override to %d/%d runs "
                     "(students collapsed to one latency = %s FPS).", n_over, n_ours, stu_fps)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            log.warning("Could not apply re-bench override (%s); using per-run FPS.", exc)

    for entry in reported_baselines():
        rows.append(_reported_to_row(entry))

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Coerce numeric columns so NaN (not None/strings) represents "missing".
    numeric_cols = [
        "params_total",
        "params_trainable_pct",
        "flops_g",
        "macs_g",
        "top1",
        "single_clip_fps",
        "single_clip_latency_ms",
        "peak_infer_vram_mb",
        "disk_size_mb",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(
        "Loaded %d measured run(s) + %d reported baseline(s) from %s",
        n_ours,
        len(rows) - n_ours,
        results_dir,
    )
    return df

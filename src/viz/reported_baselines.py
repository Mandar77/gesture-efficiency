"""Curated external baselines that are *reported* (from published papers), NOT
re-run in this repo (BRIEF section 4).

These numbers come from different datasets and evaluation protocols and are
therefore **not directly comparable** to our measured runs. Every consumer of
this list (tables, plots) must surface that caveat. Each entry carries
``reported: True`` plus ``dataset`` / ``metric_note`` fields so the caveat can
be rendered next to the number.

Fields (a superset of the normalized loader columns):
    run_name          display name of the method
    reported          always True
    dataset           dataset + protocol the reported number is on
    params_total      total parameters (absolute count, not millions)
    flops_g           GFLOPs per clip if the paper reports FLOPs, else NaN
    macs_g            GMACs per clip if the paper reports MACs, else NaN
    top1              reported Top-1 accuracy (%)
    citation          short author/venue citation
    metric_note       free-text caveat (dataset / protocol / modality)

Absolute param counts are used (e.g. 3.1e6) so the loader's params_total column
is homogeneous with our measured runs; tables convert to millions for display.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

NAN = math.nan

# Each dict is intentionally shaped like a normalized loader row so it can be
# merged directly into the results DataFrame.
REPORTED_BASELINES: List[Dict[str, Any]] = [
    {
        "run_name": "MoViNet-A0",
        "reported": True,
        "dataset": "Kinetics-600",
        "params_total": 3.1e6,
        "flops_g": 2.71,
        "macs_g": NAN,
        "top1": 71.5,
        "citation": "Kondratyuk et al., CVPR 2021",
        "metric_note": "Kinetics-600 action recognition; not a gesture dataset.",
    },
    {
        "run_name": "MoViNet-A1",
        "reported": True,
        "dataset": "Kinetics-600",
        "params_total": 4.6e6,
        "flops_g": 6.02,
        "macs_g": NAN,
        "top1": 76.0,
        "citation": "Kondratyuk et al., CVPR 2021",
        "metric_note": "Kinetics-600 action recognition; not a gesture dataset.",
    },
    {
        "run_name": "ConvMixFormer",
        "reported": True,
        "dataset": "NVGesture (RGB)",
        "params_total": 13.57e6,
        "flops_g": NAN,
        "macs_g": 59.98,
        "top1": 76.04,
        "citation": "Garg et al., WACV 2025",
        "metric_note": "NVGesture RGB (depth 80.83%); MACs reported, not FLOPs.",
    },
    {
        "run_name": "GestFormer",
        "reported": True,
        "dataset": "NVGesture (5-modality)",
        "params_total": 24.08e6,
        "flops_g": NAN,
        "macs_g": 60.40,
        "top1": 85.85,
        "citation": "Garg et al., CVPR 2024 WiCV",
        "metric_note": "NVGesture 5-modality fusion; MACs reported, not FLOPs.",
    },
    {
        "run_name": "DSTSA-GCN",
        "reported": True,
        "dataset": "SHREC'17 (14G, skeleton)",
        "params_total": 1.99e6,
        "flops_g": 1.79,
        "macs_g": NAN,
        "top1": 97.74,
        "citation": "Cui et al., Neurocomputing 2025",
        "metric_note": "SHREC'17 14-gesture skeleton; FLOPs are per-stream.",
    },
]


def reported_baselines() -> List[Dict[str, Any]]:
    """Return a fresh copy of the reported-baseline rows (never mutate the module constant)."""
    return [dict(row) for row in REPORTED_BASELINES]

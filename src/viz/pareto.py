"""Accuracy-vs-efficiency Pareto scatter plots (BRIEF section 8, flagship figures).

Ours (measured on the fixed RTX 4060) are drawn as filled, labeled markers.
Reported external baselines are drawn as hollow markers and carry a footnote
that they come from different datasets/protocols and are **not directly
comparable**. Points with NaN on either axis are skipped, but the axes and the
caveat are always rendered so `make repro-main` never crashes on an empty repo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never require a display for repro
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("viz.pareto")

_CAVEAT = (
    "Hollow markers = reported (published) numbers on different "
    "datasets/protocols — NOT directly comparable to our measured runs."
)
_NO_RUNS_NOTE = "No measured runs yet — showing reported baselines only."


def _finite(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows where both axes are present and finite."""
    if df.empty or x not in df.columns or y not in df.columns:
        return df.iloc[0:0]
    sub = df.dropna(subset=[x, y])
    return sub[pd.to_numeric(sub[x], errors="coerce").notna()
               & pd.to_numeric(sub[y], errors="coerce").notna()]


def plot_pareto(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: str | Path,
    title: str,
    annotate_reported_caveat: bool = True,
) -> Path:
    """Generic Pareto scatter of ``y`` vs ``x``, split by source.

    Ours -> filled markers with per-point labels; reported -> hollow markers.
    NaN-on-either-axis points are skipped; axes + caveat still render.
    Returns the written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    if df is None or df.empty:
        ours = reported = df.iloc[0:0] if isinstance(df, pd.DataFrame) else pd.DataFrame()
    else:
        source = df.get("source", pd.Series(["ours"] * len(df), index=df.index))
        # Exclude measured NEGATIVE-result cells (pruned-no-finetune, int8
        # fallback) from the frontier: they are flagged in `notes` with a
        # NOT_FRONTIER prefix by the loader. They stay in the dataframe (tables
        # report them honestly) but must NOT plot as real operating points.
        notes = df.get("notes", pd.Series([None] * len(df), index=df.index))
        not_frontier = notes.astype(str).str.startswith("NOT_FRONTIER")
        ours = _finite(df[(source == "ours") & (~not_frontier)], x, y)
        reported = _finite(df[source == "reported"], x, y)

    if not ours.empty:
        ax.scatter(
            ours[x], ours[y],
            marker="o", s=90, c="#1f77b4",
            edgecolors="black", linewidths=0.6, zorder=3, label="Ours (measured)",
        )
        for _, row in ours.iterrows():
            ax.annotate(
                str(row.get("run_name", "")),
                (row[x], row[y]),
                textcoords="offset points", xytext=(6, 4), fontsize=8, zorder=4,
            )

    if not reported.empty:
        ax.scatter(
            reported[x], reported[y],
            marker="s", s=90, facecolors="none",
            edgecolors="#d62728", linewidths=1.4, zorder=3, label="Reported (not re-run)",
        )
        for _, row in reported.iterrows():
            ax.annotate(
                str(row.get("run_name", "")),
                (row[x], row[y]),
                textcoords="offset points", xytext=(6, -10),
                fontsize=8, style="italic", color="#7f1d1d", zorder=4,
            )

    ax.set_xlabel(_axis_label(x))
    ax.set_ylabel(_axis_label(y))
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    if not ours.empty or not reported.empty:
        ax.legend(loc="best", fontsize=8)

    footnotes = []
    if ours.empty:
        footnotes.append(_NO_RUNS_NOTE)
    if annotate_reported_caveat:
        footnotes.append(_CAVEAT)
    if footnotes:
        fig.text(0.01, 0.005, "  ".join(footnotes), fontsize=7, style="italic",
                 wrap=True, ha="left", va="bottom", color="#444444")
        fig.subplots_adjust(bottom=0.18)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote Pareto figure -> %s (ours=%d, reported=%d)",
             out_path, len(ours), len(reported))
    return out_path


_AXIS_LABELS = {
    "flops_g": "FLOPs per clip (G)",
    "macs_g": "MACs per clip (G)",
    "top1": "Top-1 accuracy (%)",
    "single_clip_latency_ms": "Single-clip latency (ms)",
    "single_clip_fps": "Throughput (FPS)",
    "params_total": "Parameters",
    "peak_infer_vram_mb": "Peak inference VRAM (MB)",
    "disk_size_mb": "On-disk size (MB)",
}


def _axis_label(col: str) -> str:
    return _AXIS_LABELS.get(col, col)


def plot_accuracy_vs_flops(df: pd.DataFrame, out: str | Path) -> Path:
    """Convenience wrapper: Top-1 vs FLOPs per clip."""
    return plot_pareto(
        df, x="flops_g", y="top1", out_path=out,
        title="Accuracy vs FLOPs (Pareto)", annotate_reported_caveat=True,
    )


def plot_accuracy_vs_latency(df: pd.DataFrame, out: str | Path) -> Path:
    """Convenience wrapper: Top-1 vs single-clip latency."""
    return plot_pareto(
        df, x="single_clip_latency_ms", y="top1", out_path=out,
        title="Accuracy vs Latency (Pareto)", annotate_reported_caveat=True,
    )

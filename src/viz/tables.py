"""Comparison-table generation (LaTeX + Markdown) from the results DataFrame.

Columns: Method, Source, Dataset, Params(M), FLOPs(G), Top-1(%), FPS,
Latency(ms), PeakVRAM(MB), Disk(MB). NULL/NaN cells render as "TODO" (never a
fabricated number). Reported rows show their citation in the Source column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd

from src.utils.logging_utils import get_logger
from src.viz.loader import load_results

log = get_logger("viz.tables")

TODO = "TODO"

# (header, dataframe column, formatter). params are stored as absolute counts.
_COLUMNS = [
    ("Method", "run_name", lambda v: _s(v)),
    ("Source", "_source_disp", lambda v: _s(v)),
    ("Dataset", "dataset", lambda v: _s(v)),
    ("Params(M)", "params_total", lambda v: _fmt_num(v / 1e6, 2) if _present(v) else TODO),
    ("FLOPs(G)", "flops_g", lambda v: _fmt_num(v, 2)),
    ("Top-1(%)", "top1", lambda v: _fmt_num(v, 2)),
    ("FPS", "single_clip_fps", lambda v: _fmt_num(v, 1)),
    ("Latency(ms)", "single_clip_latency_ms", lambda v: _fmt_num(v, 2)),
    ("PeakVRAM(MB)", "peak_infer_vram_mb", lambda v: _fmt_num(v, 1)),
    ("Disk(MB)", "disk_size_mb", lambda v: _fmt_num(v, 3)),
]


def results_to_dataframe(results_dir: str | Path = "experiments") -> pd.DataFrame:
    """Load the normalized results DataFrame (thin reuse of the loader)."""
    return load_results(results_dir)


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v)) and not pd.isna(v)


def _s(v: Any) -> str:
    return TODO if not _present(v) else str(v)


def _fmt_num(v: Any, ndigits: int) -> str:
    if not _present(v):
        return TODO
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return TODO


def _source_display(row: pd.Series) -> str:
    """'ours' or 'reported (<citation>)'."""
    if row.get("source") == "reported":
        cite = row.get("notes")
        cite = str(cite) if _present(cite) else "reported"
        return f"reported ({cite})"
    return "ours"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["_source_disp"] = []
    else:
        df["_source_disp"] = df.apply(_source_display, axis=1)
    return df


def _rows(df: pd.DataFrame) -> List[List[str]]:
    df = _prepare(df)
    out: List[List[str]] = []
    for _, row in df.iterrows():
        out.append([fmt(row.get(col)) for _, col, fmt in _COLUMNS])
    return out


# --- LaTeX ------------------------------------------------------------------

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    # Handle backslash first so we don't double-escape the replacements.
    out = s.replace("\\", _LATEX_SPECIALS["\\"])
    for ch, rep in _LATEX_SPECIALS.items():
        if ch == "\\":
            continue
        out = out.replace(ch, rep)
    return out


def make_comparison_table_latex(df: pd.DataFrame) -> str:
    """Render the comparison table as a LaTeX ``tabular`` string."""
    headers = [h for h, _, _ in _COLUMNS]
    col_spec = "l" * len(headers)
    lines: List[str] = []
    lines.append("% Reported rows are published numbers on different datasets/protocols;")
    lines.append("% they are NOT directly comparable to our measured runs.")
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\hline")
    lines.append(" & ".join(_latex_escape(h) for h in headers) + " \\\\")
    lines.append("\\hline")
    for row in _rows(df):
        lines.append(" & ".join(_latex_escape(c) for c in row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


# --- Markdown ---------------------------------------------------------------

def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def make_comparison_table_markdown(df: pd.DataFrame) -> str:
    """Render the comparison table as a GitHub-flavored Markdown string."""
    headers = [h for h, _, _ in _COLUMNS]
    lines: List[str] = []
    lines.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in _rows(df):
        lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    lines.append("")
    lines.append(
        "> Reported rows are published numbers on different datasets/protocols "
        "and are **not directly comparable** to our measured runs."
    )
    return "\n".join(lines) + "\n"

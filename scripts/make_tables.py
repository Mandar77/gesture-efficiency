#!/usr/bin/env python
"""Generate the headline comparison table (LaTeX + Markdown) from committed results.

Called by ``make repro-main``. Writes the LaTeX ``tabular`` to --out and a
sibling ``.md`` alongside it. NULL cells render as "TODO"; reported rows carry
their citation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import get_logger  # noqa: E402
from src.viz.loader import load_results  # noqa: E402
from src.viz.tables import (  # noqa: E402
    make_comparison_table_latex,
    make_comparison_table_markdown,
)

log = get_logger("scripts.make_tables")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparison tables from results.")
    p.add_argument("--results", default="experiments", help="Results root dir (default: experiments)")
    p.add_argument("--out", default="paper/tables.tex", help="Output .tex path (default: paper/tables.tex)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_tex = Path(args.out)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_tex.with_suffix(".md")

    df = load_results(args.results)

    latex = make_comparison_table_latex(df)
    markdown = make_comparison_table_markdown(df)

    out_tex.write_text(latex, encoding="utf-8")
    log.info("Wrote LaTeX table -> %s", out_tex)

    out_md.write_text(markdown, encoding="utf-8")
    log.info("Wrote Markdown table -> %s", out_md)

    n_rows = 0 if df.empty else len(df)
    log.info("Done: %d row(s) rendered.", n_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

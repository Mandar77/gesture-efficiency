#!/usr/bin/env python
"""Generate the flagship Pareto figures from committed results (BRIEF section 8, M8).

Called by ``make repro-main``. Produces:
    pareto_accuracy_vs_flops.png
    pareto_accuracy_vs_latency.png
into --out. If there are zero measured ("ours") rows with accuracy, the figures
are still emitted with just the reported baselines + a "no measured runs yet"
note, so reproduction never crashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import get_logger  # noqa: E402
from src.viz.loader import load_results  # noqa: E402
from src.viz.pareto import (  # noqa: E402
    plot_accuracy_vs_flops, plot_accuracy_vs_latency, plot_accuracy_vs_throughput_bs8,
)

log = get_logger("scripts.make_figures")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate flagship Pareto figures from results.")
    p.add_argument("--results", default="experiments", help="Results root dir (default: experiments)")
    p.add_argument("--out", default="paper/figures", help="Output dir for figures (default: paper/figures)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)

    n_ours = int((df["source"] == "ours").sum()) if not df.empty else 0
    n_ours_acc = (
        int(((df["source"] == "ours") & df["top1"].notna()).sum()) if not df.empty else 0
    )
    if n_ours_acc == 0:
        log.warning(
            "No measured 'ours' runs with accuracy found — emitting figures with "
            "reported baselines only (annotated)."
        )

    written = [
        plot_accuracy_vs_flops(df, out_dir / "pareto_accuracy_vs_flops.png"),
        plot_accuracy_vs_throughput_bs8(df, out_dir / "pareto_accuracy_vs_throughput_bs8.png"),
        plot_accuracy_vs_latency(df, out_dir / "pareto_accuracy_vs_latency.png"),
    ]
    for path in written:
        log.info("Figure written: %s", path)
    log.info(
        "Done: %d figure(s) in %s (%d ours run(s), %d with accuracy).",
        len(written), out_dir, n_ours, n_ours_acc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

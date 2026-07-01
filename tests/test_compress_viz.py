"""Tests for compression (fp16/int8-ptq/prune) and viz (loader/pareto/tables).

Compression tests are honest: they assert the *reported* behaviour (size drop,
sparsity increase, non-crashing PTQ with a fallback note) rather than a specific
accuracy, since accuracy on random weights is meaningless.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset

import src.models  # noqa: F401
from src.bench.efficiency_bench import measure_disk_size
from src.compress import (
    prune_report,
    quantize_fp16,
    quantize_int8_ptq,
    report_compression,
    structured_channel_prune,
    should_prefer_qat,
)
from src.utils.registry import build


def _tiny_loader(n=8, num_classes=5):
    x = torch.randn(n, 3, 6, 32, 32)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_fp16_halves_size():
    m = build("model", "compact3dcnn", num_classes=5, width=16, depth=3)
    before = measure_disk_size(m)
    mh = quantize_fp16(m)
    after = measure_disk_size(mh)
    assert after < before * 0.75, f"fp16 did not shrink size: {before}->{after}"
    assert hasattr(mh, "_quantization_note")


def test_int8_ptq_runs_and_reports_honestly():
    m = build("model", "compact3dcnn", num_classes=5, width=16, depth=3).eval()
    loader = _tiny_loader(num_classes=5)
    q = quantize_int8_ptq(m, loader, torch.device("cpu"))
    assert hasattr(q, "_quantization_note")  # honest note present
    rep = report_compression(m, q, loader, torch.device("cpu"), "int8_ptq")
    # top1 fields present (may be equal on random weights); drop is reported.
    assert "top1_drop" in rep and "quantization_note" in rep
    assert rep["size_mb_before"] > 0


def test_structured_prune_increases_sparsity():
    m = build("model", "compact3dcnn", num_classes=5, width=16, depth=3)
    before = prune_report(m, m)["conv_sparsity_after"]
    structured_channel_prune(m, ratio=0.5)
    after = prune_report(m, m)["conv_sparsity_after"]
    assert after > before, f"pruning did not increase sparsity: {before}->{after}"


def test_qat_threshold_logic():
    assert should_prefer_qat(5.0) is True    # >3 pp drop -> prefer QAT
    assert should_prefer_qat(1.0) is False   # small drop -> PTQ ok
    assert should_prefer_qat(None) is False


# --------------------------- viz -------------------------------------------
def test_viz_loader_includes_reported_baselines(tmp_path):
    from src.viz.loader import load_results

    df = load_results(tmp_path)  # empty results dir
    assert (df["source"] == "reported").sum() == 5  # the 5 curated baselines
    assert "MoViNet-A0" in df["run_name"].values


def test_pareto_writes_png_with_nans(tmp_path):
    from src.viz.loader import load_results
    from src.viz.pareto import plot_accuracy_vs_flops, plot_accuracy_vs_latency

    df = load_results("experiments")  # has the smoke run + reported baselines
    p1 = plot_accuracy_vs_flops(df, tmp_path / "flops.png")
    p2 = plot_accuracy_vs_latency(df, tmp_path / "lat.png")
    assert p1.exists() and p2.exists()


def test_tables_render_todo_and_citations(tmp_path):
    from src.viz.loader import load_results
    from src.viz.tables import make_comparison_table_latex, make_comparison_table_markdown

    df = load_results("experiments")
    tex = make_comparison_table_latex(df)
    md = make_comparison_table_markdown(df)
    assert "tabular" in tex and "TODO" in md  # null cells -> TODO
    assert "Kondratyuk" in md  # reported citation present
    assert "not directly comparable" in md.lower()

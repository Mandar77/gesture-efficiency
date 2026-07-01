"""Live training-progress monitor for any gesture-efficiency run.

Parses a run's `run.log` (the per-run log written by setup_file_logging) and
prints a clean one-screen status: current epoch/iter, ETA, loss trend, and the
best validation top-1 so far. Reusable across runs — point it at a run name, a
log path, or let it auto-pick the most recently modified run log.

Usage:
    # auto-pick the most recently active run under experiments/
    python scripts/watch_progress.py

    # a specific run by name (its dir under experiments/<group>/<run_name>/)
    python scripts/watch_progress.py --run jester_compact3dcnn_16f172_30ep

    # an explicit log file
    python scripts/watch_progress.py --log path/to/run.log

    # auto-refresh every 30s until the run finishes (Ctrl-C to stop)
    python scripts/watch_progress.py --watch
    python scripts/watch_progress.py --watch --interval 15

Read-only: never touches training. Safe to run anytime, including while a run
is active or after it has finished (it will report the final result).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
RE_IT = re.compile(_TS + r".*epoch (\d+) it (\d+)/(\d+) loss ([\d.]+) lr ([\d.eE+-]+)")
# Summaries are pretty-printed and may wrap across lines, so match the marker
# and pull val_top1 / epoch with targeted sub-patterns rather than eval-ing the
# whole (possibly multi-line) dict.
RE_SUMMARY = re.compile(_TS + r".*epoch (\d+) summary:")
RE_VAL_TOP1 = re.compile(r"'val_top1':\s*([\d.]+)")
RE_PEAK = re.compile(r"Peak train VRAM: ([\d.]+) MB")
RE_DONE = re.compile(r"DONE\.|DONE (PEFT|distillation|multimodal)")
RE_TOTAL_EPOCHS = None  # inferred from config if available


def _find_latest_log() -> Path | None:
    exp = ROOT / "experiments"
    logs = list(exp.rglob("run.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _resolve_log(args) -> Path | None:
    if args.log:
        return Path(args.log)
    if args.run:
        hits = list((ROOT / "experiments").rglob(f"{args.run}/run.log"))
        if hits:
            return hits[0]
        # maybe they passed the group/run dir directly
        cand = ROOT / "experiments" / args.run / "run.log"
        return cand if cand.exists() else None
    return _find_latest_log()


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _total_epochs_from_config(log_path: Path) -> int | None:
    cfg = log_path.parent / "config.resolved.yaml"
    if not cfg.exists():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        m = re.search(r"^\s*epochs:\s*(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def summarize(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    first_ts = None
    for ln in lines:
        m = re.match(_TS, ln)
        if m:
            first_ts = _parse_ts(m.group(1))
            break

    it_matches = [RE_IT.search(ln) for ln in lines]
    it_matches = [m for m in it_matches if m]
    # An epoch summary marker and its val_top1 can be on the same or adjacent
    # lines (the dict pretty-prints and wraps). Pair each summary's epoch with
    # the next val_top1 found within a small window.
    summaries = []  # list of (epoch:int, val_top1:float|None)
    for i, ln in enumerate(lines):
        sm = RE_SUMMARY.search(ln)
        if not sm:
            continue
        ep = int(sm.group(2))
        window = " ".join(lines[i:i + 4])
        vt = RE_VAL_TOP1.search(window)
        summaries.append((ep, float(vt.group(1)) if vt else None))
    done = any(RE_DONE.search(ln) for ln in lines)
    peak = None
    for ln in lines:
        pm = RE_PEAK.search(ln)
        if pm:
            peak = float(pm.group(1))

    total_epochs = _total_epochs_from_config(log_path)

    out = []
    out.append(f"Run     : {log_path.parent.name}")
    out.append(f"Log     : {log_path}")

    # Best val top-1 across epoch summaries.
    best_acc, best_ep = None, None
    for ep, acc in summaries:
        if acc is not None and (best_acc is None or acc > best_acc):
            best_acc, best_ep = acc, ep

    if not it_matches and not summaries:
        out.append("Status  : starting up (no iterations logged yet)")
        return "\n".join(out)

    last_it = it_matches[-1] if it_matches else None
    if last_it:
        cur_ep = int(last_it.group(2))
        cur_it = int(last_it.group(3))
        tot_it = int(last_it.group(4))
        loss = float(last_it.group(5))
        last_ts = _parse_ts(last_it.group(1))

        # ETA: elapsed / fraction-done.
        frac = None
        if total_epochs:
            done_units = cur_ep + (cur_it / max(tot_it, 1))
            frac = done_units / total_epochs
        if first_ts and last_ts and frac and frac > 0:
            elapsed = (last_ts - first_ts).total_seconds()
            remaining = elapsed * (1 - frac) / frac
            eta_h = remaining / 3600
            eta_str = f"{eta_h:.1f} h" if eta_h >= 1 else f"{remaining/60:.0f} min"
        else:
            eta_str = "n/a"

        ep_disp = f"{cur_ep}/{total_epochs}" if total_epochs else str(cur_ep)
        out.append(f"Status  : {'FINISHED' if done else 'TRAINING'}")
        out.append(f"Epoch   : {ep_disp}   (iter {cur_it}/{tot_it})")
        out.append(f"Loss    : {loss:.4f}  (last logged iter)")
        if not done:
            out.append(f"ETA     : ~{eta_str} remaining")

    # loss trend over the last several iters
    if len(it_matches) >= 6:
        losses = [float(m.group(5)) for m in it_matches[-6:]]
        trend = " -> ".join(f"{l:.2f}" for l in losses)
        out.append(f"Trend   : {trend}")

    if best_acc is not None:
        out.append(f"Best val: {best_acc:.3f}% top-1 (epoch {best_ep})")
    elif summaries:
        out.append("Best val: (no val_top1 in summaries yet)")

    if peak is not None:
        out.append(f"Peak VRAM (train): {peak:.0f} MB")

    if done:
        # Try to surface the final bench row from the results JSON.
        rj = list((ROOT / "experiments").rglob(f"{log_path.parent.name}.json"))
        if rj:
            out.append(f"Result JSON: {rj[0]}")
        out.append("Status  : DONE — run complete.")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="Run name (dir under experiments/).")
    ap.add_argument("--log", default=None, help="Explicit run.log path.")
    ap.add_argument("--watch", action="store_true", help="Auto-refresh until done.")
    ap.add_argument("--interval", type=int, default=30, help="Refresh seconds (--watch).")
    args = ap.parse_args()

    while True:
        log = _resolve_log(args)
        if log is None or not log.exists():
            print("No run.log found under experiments/. Is a run active? "
                  "(pass --log or --run.)")
            if not args.watch:
                return
            time.sleep(args.interval)
            continue
        report = summarize(log)
        if args.watch:
            print("\033[2J\033[H", end="")  # clear screen
        print("=" * 60)
        print(f"gesture-efficiency training status  @ {datetime.now():%H:%M:%S}")
        print("=" * 60)
        print(report)
        print("=" * 60)
        if not args.watch or "DONE — run complete" in report:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

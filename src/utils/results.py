"""Results writer: append run rows to JSON (one file per run) and a shared CSV.

Every experiment emits exactly one canonical JSON artifact under
`experiments/<group>/<run_name>.json` and appends a flattened row to
`experiments/all_results.csv`. Env metadata (GPU/CUDA/torch/seed/timestamp) is
stamped automatically so no artifact can omit it (BRIEF section 11).

Never fabricate: callers pass real measured values; missing measurements should
be left as `None` (rendered as the string "TODO" by the table generator), never
filled with a plausible guess.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.env import env_metadata
from src.utils.logging_utils import get_logger

log = get_logger("utils.results")

DEFAULT_ROOT = Path("experiments")


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


class ResultsWriter:
    """Writes a per-run JSON and appends to a shared CSV of all runs."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, seed: Optional[int] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.csv_path = self.root / "all_results.csv"

    def write(
        self,
        result: Dict[str, Any],
        *,
        group: str,
        run_name: str,
        seed: Optional[int] = None,
    ) -> Path:
        """Write one run's result. `result` should already contain measured
        fields (accuracy, params, flops, latency, vram, ...). Env metadata is
        merged in automatically.
        """
        seed = seed if seed is not None else self.seed
        record = {
            "group": group,
            "run_name": run_name,
            **result,
            "env": env_metadata(seed=seed),
        }
        group_dir = self.root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        json_path = group_dir / f"{run_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        log.info("Wrote result JSON -> %s", json_path)

        self._append_csv(_flatten(record))
        return json_path

    def _append_csv(self, flat_row: Dict[str, Any]) -> None:
        existing_rows = []
        fieldnames: list[str] = []
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                fieldnames = reader.fieldnames or []
        # Union of columns so schema can grow across runs.
        all_fields = list(dict.fromkeys([*fieldnames, *flat_row.keys()]))
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(flat_row)
        log.info("Appended row -> %s", self.csv_path)

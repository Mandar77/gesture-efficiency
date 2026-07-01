#!/usr/bin/env bash
# 1-2 minute end-to-end pipeline check on the 8GB RTX 4060.
# Proves data -> model -> train (1 epoch) -> efficiency bench -> results JSON.
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
echo "[smoke] using interpreter: $PY"
"$PY" scripts/train.py --config configs/smoke.yaml
echo "[smoke] OK — see experiments/smoke/ for artifacts"

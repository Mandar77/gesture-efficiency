# Experiments Log

Human-readable log of every run: the command, the config, and the headline
result. Machine-readable artifacts live in `experiments/<group>/<run>.json` and
the flattened `experiments/all_results.csv`. Every artifact records GPU / CUDA /
torch / seed / timestamp (BRIEF §11).

**Hardware of record:** NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB) · Intel
i9 13th-gen · Windows 11 · torch 2.6.0+cu124 · CUDA 12.4 · cuDNN 9.1.0 ·
Python 3.12.6.

Legend: **[run]** = our measured run · **[reported]** = external number cited
from a paper (never presented as ours) · `TODO` = not yet run (never fabricate).

---

## M1 — Scaffold + smoke

### smoke_compact3dcnn  **[run]**
- **Command:** `python scripts/train.py --config configs/smoke.yaml`
- **Purpose:** prove the full pipeline (synthetic data → tiny 3D-CNN → 1 epoch
  train → efficiency bench → results JSON/CSV) runs on the 8GB RTX 4060.
- **Config:** `configs/smoke.yaml` (synthetic, 10 classes, 8 frames @64px,
  compact3dcnn width=8 depth=2, 1 epoch, seed=0).
- **Result:** pipeline OK. Params 4,322; MACs 0.009 G (fvcore) — cross-checked
  ptflops/thop 0.0092 G; single-clip latency 0.79 ms (bs=1, bf16 AMP) →
  ~1268 FPS; **peak train VRAM 69.8 MB**. Accuracy is meaningless on synthetic
  data by design (not a results row).
- **Artifact:** `experiments/smoke/smoke_compact3dcnn.json`.

---

## M2 — Data

Data layer complete: `download_data.py` (prints per-dataset acquisition steps,
no credentials hardcoded), `prepare_jester.py` / `prepare_nvgesture.py` /
`prepare_shrec.py` (build on-disk indices, integrity checks, class histograms),
and registry-backed loaders (`jester`, `nvgesture` multimodal RGB-D+IR, `shrec`
skeleton). Frame sampling: uniform / random_uniform / segment (TSN), default 16
frames, resolution 172/224. `scripts/verify_data.py` fetches a batch, checks
shapes/ranges, saves a montage, and cross-checks counts against official splits.

- **Verification [run]:** `pytest tests/test_data.py` builds fake Jester + SHREC
  dirs on disk, runs prepare → loader, asserts correct tensor shapes
  ([3,16,32,32] Jester clip, [16,66] SHREC skeleton) and split counts. **Passes.**
  This proves the real-data path without the 23GB download.
- **Real data:** obtain per `DATA_LICENSES.md`, then
  `python src/data/prepare_jester.py --root data/jester` and
  `python scripts/verify_data.py --config configs/baseline_jester.yaml`.

## M3 — Pipeline-validation baseline (from-scratch 3D-CNN on Jester)

_TODO: fill after first real Jester training run + bench row._

## M4 — PEFT teacher (frozen ViT + LoRA/adapter/prompt/full-FT)

_TODO._

## M5 — Distillation (streaming causal student)

_TODO._

## M6 — Compression (INT8 PTQ/QAT + pruning)

_TODO._

## M7 — Multimodal (NVGesture RGB / RGB+D / RGB+D+IR)

_TODO._

## M8 — Frontier + demo + paper assets

_TODO._

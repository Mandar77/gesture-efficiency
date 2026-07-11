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

### Jester acquired + prepared **[run]** (2026-06-30)
- Frames: user-downloaded Qualcomm v1 parts (21.36 GB), concatenated
  (`cat parts | tar zx`) and extracted to `data/jester/20bn-jester-v1/`
  (148,092 clip dirs, tar exit 0). Labels: fetched from the udacity mirror and
  verified (see DATA_LICENSES "Provenance").
- **`prepare_jester.py` integrity report:** train **118,562** clips / val
  **14,787** clips, **0 missing, 0 short (<8 frames)** — both counts match the
  official splits exactly (`verify_data.py` cross-check: `[OK]` / `[OK]`).
  27-class histogram logged in `data/jester/index_meta.json` (class 26 "Doing
  other things" largest at 9,592 train, consistent with the official 12,416).
- **Loader verified:** batch shape `[16, 3, 16, 172, 172]`, ImageNet-normalised
  range `[-2.12, 2.64]`, montage at `experiments/verify_jester_batch.png`.
- **Measured training throughput (RTX 4060, compact3dcnn, 16f/172px, bs16,
  bf16 AMP):** 33.7 clips/s (I/O-bound on JPEG decode), peak train VRAM
  **1476 MB**; ~59 min/epoch on the full 118,562-clip train split.

## M3 — Pipeline-validation baseline (from-scratch 3D-CNN on Jester)

### jester_compact3dcnn_16f172_30ep  **[run — COMPLETE]**
- **Command:** `python scripts/train.py --config configs/baseline_jester.yaml
  --set data.num_workers=4 data.prefetch_factor=2
  output.run_name=jester_compact3dcnn_16f172_30ep`
- **Config:** compact3dcnn (width=32, depth=4), 16 frames @172px, segment
  sampling, bs=16, AdamW lr 5e-4, cosine + 2-epoch warmup, label smoothing 0.1,
  bf16 AMP, seed 42. **Full official splits** (118,562 train / 14,787 val).
- **Result (measured, RTX 4060):**
  - **Val top-1 78.91%** (best 79.21% @ epoch 27), **top-5 96.82%** — the first
    real, non-synthetic headline number.
  - **1.17M params**, **4.84 GFLOPs** / 2.42 GMACs per clip (fvcore).
  - **459.7 FPS** single-clip, **2.18 ms** latency (bs=1, bf16).
  - **Peak train VRAM 654 MB**, **peak infer VRAM 686 MB** (both far under 8 GB).
  - **On-disk 4.48 MB**.
- **Artifact:** `experiments/baseline/jester_compact3dcnn_16f172_30ep.json`;
  now the first "ours" point on the Pareto frontier
  (`paper/figures/pareto_accuracy_vs_*.png`, `paper/tables.md`).
- **Run note:** the first launch (8 workers) crashed at epoch ~10 on a Windows
  DataLoader shared-memory limit (err 1455); relaunched with 4 workers +
  prefetch 2 and **auto-resumed from the per-epoch checkpoint at epoch 10**
  (no lost epochs). This is exactly the resume path added in the engine.

## M4 — PEFT teacher (frozen ViT + LoRA/adapter/prompt/full-FT)

Implemented: `src/models/peft.py` (self-contained LoRA / AIM-style adapters /
shallow prompt tuning, no external `peft` dep) + `src/models/peft_teacher.py`
(frozen timm ViT-B/16 or ViT-S/DINOv2-S, per-frame encode → TransformerEncoder
temporal head → classifier; grad checkpointing + bf16 AMP). Entrypoint
`scripts/train_peft_teacher.py`; config `configs/peft_lora.yaml` (sweep via
`--set peft.method={lora,adapter,prompt,full_ft,none}`).

- **GPU integration [run]** (synthetic data, ViT-S, LoRA, 8f/64px, seed 0):
  trains on the RTX 4060; **backbone trainable 1.011%** (221,184 / 21.9M) —
  well under the 5% target; total-model trainable 8.4% (incl. temporal head).
  **Peak train VRAM 200.9 MB.** Bench: FLOPs fvcore 2.98 GMACs, cross-checked
  ptflops 2.97 / thop 5.82; teacher single-clip 53 FPS, peak infer VRAM 360 MB.
  (Synthetic-data run — pipeline validation only, not a results row; artifacts
  were removed after verifying to avoid polluting the frontier.)
### jester_vits_lora_8f224  **[run — COMPLETE]**
- **Command:** `python scripts/train_peft_teacher.py --config configs/peft_lora.yaml
  --set data.num_workers=4 data.prefetch_factor=2 output.run_name=jester_vits_lora_8f224`
- **Config:** frozen ViT-S/16 (timm) + LoRA (rank 8, α16, q/k/v/o) + 2-layer
  temporal TransformerEncoder head; 8 frames @224px, bs8, grad checkpointing,
  bf16 AMP, 20 epochs, seed 42. Full official splits.
- **Result (measured, RTX 4060):**
  - **Val top-1 86.49%** (best 86.80% @ ep14), **top-5 97.36%** — **+7.6 pts
    over the from-scratch baseline (78.9%)** while training only **1.01% of the
    backbone** (LoRA); total trainable 14.9% incl. the temporal head.
  - 25.4M params, **68.8 GFLOPs**, **56 FPS** / 17.8 ms single-clip, peak
    train/infer VRAM **725/720 MB**, 97 MB on disk.
- **Interpretation:** validates the PEFT hypothesis — a frozen foundation model
  adapted with ~1% trainable backbone params clearly beats full from-scratch
  training. But it is ~14x the baseline's FLOPs and ~8x slower, which is exactly
  why the next step (distillation into the streaming student) matters.
- **Note:** paused for a laptop shutdown after epoch 0 and **auto-resumed from
  the per-epoch checkpoint at epoch 1** — clean, no lost epochs.

**PEFT sweep (adapter / prompt / full_ft):** running next for the
accuracy-vs-trainable-params ablation (§6.1).

- Unit tests: `tests/test_models.py` verifies all methods construct + forward on
  CPU and trainable < total. (Caught + fixed a real bug: LoRA injection mutated
  the module tree while iterating `model.modules()`, causing unbounded recursion;
  now snapshots targets before swapping.)

## M5 — Distillation (streaming causal student)

Implemented: `src/models/student.py` — `StreamingStudent`, a MoViNet-A0/A1-class
causal 3D-CNN (~3.1M params at defaults) with `CausalConv3d` + per-layer stream
buffers for **constant-memory online inference** (`reset_stream()` /
`forward_step(frame)`); streaming last-step matches whole-clip `forward()` in
eval (bit-identical, verified). `src/train/distill.py` — logit KD (KL·T²) +
optional feature KD with a lazily-attached student-side projector so its params
join the optimizer; ablation via `beta_kd`/`gamma_feat`. Entrypoint
`scripts/distill_student.py`; config `configs/distill_student.yaml`.

- **GPU integration [run]** (synthetic, student←ViT-S-LoRA teacher, logit+feature,
  seed 0): all three loss terms computed & logged (CE 1.75 + KD 1.10 +
  0.5·feat 1.05); feature projector auto-attached (student 64 → teacher 384).
  **Peak train VRAM 185.5 MB.** (Synthetic pipeline validation; artifacts removed.)
- **Real Jester distillation:** pending teacher checkpoint + data.

## M6 — Compression (INT8 PTQ/QAT + pruning)

Implemented `src/compress/`: `ptq.py` (fp16 GPU copy; eager INT8 static PTQ with
a **defensive dynamic-Linear fallback** for Conv3d, honest `report_compression`
that never hides the accuracy drop), `qat.py` (`prepare_qat`/`convert_qat` +
`should_prefer_qat` implementing the §6.3 >3 pp threshold), `prune.py` (L1
structured channel pruning).

- **GPU integration [run]** (streaming student, random weights): fp16 halves
  on-disk size; INT8 static PTQ succeeded on CPU (0.088→0.059 MB); structured
  prune @0.3 pruned 12 Conv layers. INT8 eval-accuracy delta reported as `None`
  when the quantized-op eval falls back — surfaced honestly, never fabricated.
- Unit tests: `tests/test_compress_viz.py` (fp16 size drop, PTQ non-crash +
  honest note, prune sparsity increase, QAT threshold logic).

## M7 — Multimodal (Briareo RGB / RGB+D / RGB+D+IR) — **PRIMARY = Briareo**

**Primary multimodal dataset is now Briareo** (NVGesture demoted to optional/
pending access — see DATA_LICENSES). `src/models/fusion.py` — `MultiModalFusion`
with three strategies (late-logit / late-feature / shared-adapter; shared-adapter
is the efficiency angle — one backbone + tiny per-modality adapters).
`src/train/multimodal.py` — dict-aware `multimodal_loss_fn` +
`evaluate_multimodal`. Config `configs/multimodal_briareo.yaml`; ablation via
`--set data.modalities=... model.kwargs.modalities=...`.

- **Briareo data prepared [run] (2026-07-11):** `prepare_briareo.py` built the
  index from the shipped subject-disjoint split — **936 train / 216 val / 288
  test = 1,440 clips** (= 40 subjects × 12 gestures × 3 reps, exactly the
  documented size), **0 missing modalities, 0 short sequences**. RGB (PNG) +
  ToF depth (float `.npz`, decompressed + normalized, ~65% nonzero) + IR (PNG)
  all load; verified all three fusion modes forward on a real Briareo batch on
  the RTX 4060.
- **Split policy (fixed, stated):** official shipped session split, subject-
  disjoint (train=26 / val=6 / test=8 sessions); no subject leakage. g00–g11 are
  the 12 classes; g12_test (no-gesture/test folder) is excluded.
- Unit tests: `tests/test_models.py` (all 3 fusion modes, RGB-only subset,
  shared-adapter cheaper) + `tests/test_data.py::test_prepare_and_load_briareo`
  (fake Briareo tree incl. `.npz` depth → prepare → loader shapes). All pass.
- **NVGesture:** loader kept fully implemented (same API) but marked optional/
  pending; drops into this same machinery when access is granted.
- **Real Briareo modality-ablation runs (RGB / RGB+D / RGB+D+IR):** queued after
  the Jester PEFT sweep + M5 distillation finish on the GPU.

## M8 — Frontier + demo + paper assets

Implemented `src/viz/` (loader → normalized DataFrame incl. 5 reported
baselines; `pareto.py` accuracy-vs-FLOPs / accuracy-vs-latency plots with
reported markers hollow + not-comparable caveat; `tables.py` LaTeX+Markdown with
`TODO` for unmeasured cells) and `src/demo/webcam_demo.py` (real-time streaming
student with on-screen FPS/latency, reusing the base project's rolling-average
FPS pattern + optional MediaPipe hand overlay). Scripts `make_figures.py` /
`make_tables.py`; `make repro-main` wired.

- **[run] `make repro-main`** regenerates `paper/figures/pareto_accuracy_vs_flops.png`,
  `pareto_accuracy_vs_latency.png`, `paper/tables.tex`, `paper/tables.md` from
  committed results (the M1 smoke row + 5 reported baselines). Verified.
- **[run]** Webcam demo streaming path (`build_student` + `forward_step`) runs
  headless; needs a camera + trained checkpoint for the live demo.

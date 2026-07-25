# gesture-efficiency

**From Static Gestures to Efficient Dynamic Recognition:** parameter-efficient
adaptation, knowledge distillation, and quantization for on-device hand-gesture
**video** recognition — with a rigorous, reproducible **accuracy–efficiency
Pareto frontier measured on a single consumer GPU**.

This repo is the empirical study behind a workshop paper. The product is not the
highest accuracy number — it is a **clean, honestly-measured efficiency
frontier** that reports the on-device numbers most prior gesture papers omit
(measured FPS, end-to-end latency, peak VRAM) alongside accuracy / params /
FLOPs. See [BRIEF.md](BRIEF.md) for the full research framing.

## Hardware of record

> **Everything trains and runs inference within 8 GB VRAM.**
> NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB) · Intel i9 13th-gen · ~1 TB SSD ·
> Windows 11 · torch 2.6.0+cu124 · CUDA 12.4 · Python 3.12.

Mixed precision (bf16 AMP) and gradient checkpointing are on by default for any
transformer training. If a design does not fit, we shrink it (clip length,
resolution, batch size, ViT-S instead of ViT-B) — we never assume more memory.

## Research question

Can a frozen image foundation model be adapted to dynamic gesture *video* with
**<5% of its parameters trainable**, then **distilled and quantized** into a
streaming student that runs in **real time within 8 GB VRAM**, while retaining
competitive accuracy on **Jester** (RGB temporal) and **Briareo** (genuine
multimodal RGB-D+IR)?

## Datasets

- **Jester (20BN)** — PRIMARY RGB-temporal: 27 classes, ~148K clips (118,562
  train / 14,787 val). Qualcomm Research-Use license. Drives the PEFT teacher,
  distillation, quantization, and the full accuracy–efficiency frontier.
- **Briareo** — PRIMARY multimodal (RGB + ToF depth + IR): 12 classes, 40
  subjects, 1,440 samples (936 / 216 / 288 official subject-disjoint split).
  Research/educational use only. Drives the M7 modality ablation
  (RGB → RGB+D → RGB+D+IR). Small — fits 8 GB comfortably at 16f.
- **NVGesture** — OPTIONAL second multimodal dataset, **access pending**; loader
  is implemented against the same API and drops in when granted.
- **SHREC'17 / DHG** — optional efficiency-only skeleton backup (mirror-sourced).

See `DATA_LICENSES.md` for provenance and terms of every dataset.

## Quickstart

```bash
# 1. Create the environment (Python 3.12) and install CUDA torch + deps
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt

# 2. Prove the whole pipeline works on your GPU (1-2 min, synthetic data, tiny model)
make smoke          # or: .venv/Scripts/python scripts/train.py --config configs/smoke.yaml

# 3. Get the data (prints per-dataset instructions; no credentials hardcoded)
make data-jester
make data-nvgesture

# 4. Regenerate the headline tables + Pareto figures from committed results
make repro-main
```

On Windows without `make`, call the interpreter directly, e.g.
`.venv/Scripts/python scripts/train.py --config configs/smoke.yaml`.

## Repository layout

```
configs/        YAML configs per experiment (inherit via _base_)
src/
  data/         download / prepare / loaders  (jester, nvgesture, shrec, synthetic)
  models/       compact3dcnn baseline, PEFT teacher, streaming student, fusion
  train/        engine + entrypoints (peft_teacher, distill_student, qat)
  compress/     PTQ, QAT, structured pruning
  eval/         accuracy, per-class, confusion matrix
  bench/        efficiency_bench.py  (FLOPs / latency / FPS / peak VRAM / disk)
  viz/          Pareto plots, results -> tables
  demo/         real-time webcam inference (streaming student)
  utils/        seeding, logging, config, registry, checkpoint IO, results writer
scripts/        thin CLI wrappers over src/*
experiments/    logged JSON/CSV results (committed)
paper/          tables.tex + figures (regenerated programmatically)
tests/          unit + smoke tests
EXPERIMENTS.md  human-readable log of every run + command + result
DATA_LICENSES.md
```

## Reproducibility & honesty

Every results artifact records seed, GPU name, CUDA + torch versions, and a
timestamp. Configs are committed; the exact command lives in `EXPERIMENTS.md`.
Official dataset splits are used with no test leakage. FLOPs are measured with
`fvcore` (protocol noted, cross-checked with `ptflops`/`thop`); latency is
warmed up and reported mean ± std. External/reported numbers are always labeled
as such. **Any number that hasn't been measured is left as `TODO` — never
filled with a plausible guess.**

## Running the experiments

`scripts/run_experiments.sh` documents the full experiment matrix (BRIEF §6):
PEFT sweep, distillation ablation, compression (FP32/FP16/INT8-PTQ/QAT/pruning),
clip-length/resolution sensitivity, and the Briareo modality ablation. Run
selectively — the full matrix is many GPU-hours. Individual entrypoints:

```bash
.venv/Scripts/python scripts/train.py             --config configs/baseline_jester.yaml   # M3
.venv/Scripts/python scripts/train_peft_teacher.py --config configs/peft_lora.yaml         # M4
.venv/Scripts/python scripts/distill_student.py    --config configs/distill_student.yaml   # M5
.venv/Scripts/python scripts/compress_student.py   --config configs/distill_student.yaml --ckpt <student.pt>  # M6
.venv/Scripts/python scripts/train_multimodal.py   --config configs/multimodal_briareo.yaml                 # M7 (Briareo primary)
```

## Real-time webcam demo

```bash
.venv/Scripts/python src/demo/webcam_demo.py --ckpt checkpoints/distill/<student>.pt \
    --labels data/jester/jester-v1-labels.csv --frame-size 172 [--show-hand]
```

Drives the streaming student's constant-memory `forward_step` API and overlays
measured FPS + per-frame latency. Runs untrained (clearly labelled) if no
checkpoint is given, so the pipeline can be demoed before training.

## Environment pinning

`requirements.txt` has loose ranges; `requirements.lock.txt` pins the exact
versions this study was measured on (torch 2.6.0+cu124, CUDA 12.4, Python 3.12).

## Results

All numbers below are **measured** on the hardware of record (RTX 4060 Laptop,
8 GB), on the **official Jester validation split (14,787 clips)** unless noted.
Figures/tables regenerate via `make repro-main`
(`paper/figures/pareto_accuracy_vs_flops.png`, `paper/tables.md`). Full audit
trail (every number traceable, negative results included) is in
[SANITY.md](SANITY.md); per-run commands in [EXPERIMENTS.md](EXPERIMENTS.md).

### Headline finding — the efficiency inversion

A **3.11 M-param streaming 3D-CNN beats every PEFT-adapted frozen ViT** on this
gesture-video task, at a fraction of the parameters and compute:

| Model | val top-1 | params | GFLOPs | bs=1 FPS | bs=8 FPS |
|---|---|---|---|---|---|
| **Streaming 3D-CNN (ours, distilled)** | **93.5 %** | **3.11 M** | **6.2** | ~135 | 211 |
| ViT-B/16 LoRA (properly tuned) | 87.7 % | 100.9 M | 272.9 | 50 | 53 |
| ViT-S/16 LoRA | 86.5 % | 25.4 M | 68.8 | 80 | 146 |

The efficiency claim rests on **parameters and FLOPs** (deterministic, mutually
comparable): the purpose-built motion model beats a **fairly-tuned 86 M ViT-B** by
~5.8 pts at **8× fewer params and ~44× less compute**, and beats ViT-S at ~8×
fewer params / ~11× less compute. For dynamic gesture *video*, a small
motion-native architecture dominates a large per-frame image foundation model —
the larger model is not merely less efficient, it is **less accurate**.

**On measured latency, read carefully.** FPS above is a single-session,
warm, median-of-3 re-bench (bs=1 = on-device streaming; bs=8 = batched
throughput), so rows are mutually comparable — unlike raw per-run numbers, which
are cross-run thermal artifacts (an earlier draft reported the student at "110
FPS"; its true bs=1 is ~135). The student leads on bs=1 latency and pulls
further ahead at bs=8, but note **bs=1 FPS does not cleanly track FLOPs**: ViT-S
full-FT and prompt bench ~120 FPS despite ~11× the student's compute, because
bs=1 inference is kernel-launch / memory-bound rather than compute-bound (the
student's SE + depthwise-separable design especially). We therefore do **not**
claim a large latency *speedup* over ViT-S — the ViT-S win is params/FLOPs; the
ViT-B win is all axes. This FLOPs-vs-measured-latency divergence is itself a core
finding: **FLOPs alone mislead, and on-device cost must be measured and
reported** — precisely the gap this study fills. (Numbers are PyTorch-eager; a
deployment stack such as TensorRT would be faster, most of all for the
launch-bound student.)

### Contributions (all measured)

1. **Efficiency inversion** (above) — the headline.
2. **LoRA > full fine-tuning.** On the ViT-S PEFT sweep (identical 8f/224 regime),
   LoRA (**86.5 %**, ~1 % of backbone trainable) beats a genuine 100 %-trainable
   full fine-tune (**83.2 %**) by **+3.3 pts**; prompt/VPT trails at 75.0 %.
   Parameter-efficient adaptation is not just cheaper here — it's more accurate.
3. **FP16 is the deployable compression lever** — 12.0 → 6.1 MB (2×), 0.01 pp drop.

### Honest negative results (reported, not buried)

- **Knowledge distillation adds almost nothing here.** A no-KD control (same
  student, pure CE, 8f/224) hits **93.3 %** vs the distilled **93.5 %** — a
  **+0.2 pp** effect. The 3D-CNN's **+24 pts over the from-scratch 8f/224 baseline
  (69.2 %)** is *architecture*, not distillation. We do **not** headline "KD → +24".
- **INT8 quantization** yields no benefit for this conv-dominated model: no
  `quantized::conv3d` CPU kernel, so only the final Linear quantizes (12.0 → 12.0
  MB, 0 pp). Reported as a deployment limitation.
- **Structured pruning without fine-tuning** collapses the compact student to
  near-random (30 % → 9.9 %, 50 % → 3.6 %). Excluded from the frontier as broken
  operating points, not tradeoffs.

### Method note — per-backbone LR tuning

ViT backbones are tuned **per-backbone** (ViT-S at lr 5e-4, ViT-B at lr 2e-4):
the ViT-S-tuned 5e-4 is too hot for the 4×-wider ViT-B and destabilizes it at
warmup-end, so each backbone gets an appropriate LR (stated explicitly, not one
blind shared recipe). The ViT-S 86.5 % is treated as a **lower bound**, not a
claimed ceiling; the inversion does not depend on either ViT being at its exact
optimum. Input normalization is read per-backbone from each model's timm
`pretrained_cfg` (both ViTs are AugReg checkpoints expecting (0.5,0.5,0.5)).

Reported baselines (MoViNet / ConvMixFormer / GestFormer / DSTSA-GCN) show `TODO`
for FPS/latency/VRAM because those papers don't report them — precisely the
on-device gap this study fills. **Any unmeasured cell shows `TODO`, never a
fabricated number.**

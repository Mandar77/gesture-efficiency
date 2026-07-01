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
competitive accuracy on **Jester** and **NVGesture**?

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
clip-length/resolution sensitivity, and the NVGesture modality ablation. Run
selectively — the full matrix is many GPU-hours. Individual entrypoints:

```bash
.venv/Scripts/python scripts/train.py             --config configs/baseline_jester.yaml   # M3
.venv/Scripts/python scripts/train_peft_teacher.py --config configs/peft_lora.yaml         # M4
.venv/Scripts/python scripts/distill_student.py    --config configs/distill_student.yaml   # M5
.venv/Scripts/python scripts/compress_student.py   --config configs/distill_student.yaml --ckpt <student.pt>  # M6
.venv/Scripts/python scripts/train_multimodal.py   --config configs/multimodal_nvgesture.yaml               # M7
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

## Results teaser

Populated by `make repro-main` from committed runs. Headline figures:
`paper/figures/pareto_accuracy_vs_flops.png` and
`paper/figures/pareto_accuracy_vs_latency.png`; the comparison table is
`paper/tables.md` (LaTeX in `paper/tables.tex`), with our measured runs alongside
reported baselines (MoViNet / ConvMixFormer / GestFormer / DSTSA-GCN) clearly
marked as reported-not-rerun. _(Real Jester/NVGesture rows fill in as the runs in
`scripts/run_experiments.sh` complete on prepared data; see EXPERIMENTS.md. Any
unmeasured cell shows `TODO` — never a fabricated number.)_

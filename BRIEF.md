# Project Brief

**Working title:** *From Static Gestures to Efficient Dynamic Recognition:
Parameter-Efficient Adaptation, Distillation, and Quantization for On-Device
Hand-Gesture Video Recognition.*

## Research question
Can a frozen image foundation model be adapted to dynamic gesture *video* with
**<5% of its parameters trainable**, then **distilled and quantized** into a
streaming student that runs in **real time within 8 GB VRAM**, while retaining
competitive accuracy on **Jester** and **NVGesture**?

## Central contribution
An **accuracy–efficiency Pareto frontier on one fixed consumer GPU** (RTX 4060,
8 GB), reporting the on-device numbers most prior gesture papers omit —
**measured FPS, end-to-end latency, peak VRAM** — alongside accuracy / params /
FLOPs.

## Explicitly NOT doing
- No static single-image classification as the headline task.
- No simulated/synthetic modalities (no landmark-derived "depth", no simulated
  "EMG"). **Multimodal means *real* RGB-D+IR from NVGesture.**

## Datasets
- **Jester (20BN)** — PRIMARY, RGB temporal, 27 classes, ~148K clips. License
  Qualcomm "Data License Agreement – Research Use" (research-only, no
  redistribution — NOT CC BY-NC-ND; see `DATA_LICENSES.md`).
- **NVGesture** — PRIMARY multimodal, RGB-D+IR, 25 classes, 1,532 clips.
- **SHREC'17 / DHG-14/28** — BACKUP skeleton track (efficiency-framed only;
  skeleton SOTA is saturated ~97.7% on SHREC 14G).

## Method
1. **Teacher:** frozen CLIP ViT-B/16 or DINOv2 ViT-B (ViT-S/DINOv2-S fallback)
   + PEFT (LoRA / AIM-style adapters / prompt tuning / full-FT switch) + a
   lightweight temporal head. Only PEFT params + temporal head + classifier are
   trained. Grad checkpointing + AMP; 8–16 frames @172–224px; small batch.
2. **Student:** streaming causal 3D-CNN (~3–5M params, MoViNet-A0/A1-class) with
   a **stream buffer for constant-memory online inference** — this is what makes
   the real-time claim credible.
3. **Distillation:** logit KD (KL + temperature) + optional feature KD.
   `L = α·CE + β·KD + γ·feat`.
4. **Compression:** INT8 PTQ + QAT + optional structured pruning on the student.
5. **Multimodal (NVGesture):** per-modality encoders + late fusion / shared
   adapter; RGB → RGB+D → RGB+D+IR ablation.

## Baselines
- **[run]** from-scratch compact 3D-CNN / X3D-XS / TSM-tiny on Jester.
- **[run]** full fine-tune of the chosen ViT.
- **[run]** our PEFT teacher, distilled student, quantized student.
- **[reported]** MoViNet-A0 (3.1M, 2.71 GFLOPs, 71.5% K600) / A1 (4.6M, 6.02
  GFLOPs, 76.0%); ConvMixFormer (13.57M, 59.98 GMACs; NVGesture RGB 76.04% /
  depth 80.83%); GestFormer (24.08M, 60.40 GMACs; NVGesture 5-modality 85.85%);
  DSTSA-GCN (~1.99M, 1.79 GFLOPs/stream; SHREC'17 14G 97.74%).

## Measurement (as important as the models) — report ALL
Top-1 (+per-class, confusion) · trainable & total params · MACs/FLOPs per clip
(fvcore, cross-checked ptflops/thop) · measured FPS + end-to-end latency
(warmed up, mean±std, single-clip vs batched) · peak VRAM (train + inference) ·
on-disk size (pre/post quant). Flagship: **accuracy-vs-FLOPs** and
**accuracy-vs-latency** Pareto plots, generated programmatically.

## Build order (milestones)
- **M1** Scaffold + smoke ✅
- **M2** Data (Jester + NVGesture prepare/loaders)
- **M3** Pipeline-validation baseline (from-scratch 3D-CNN on Jester)
- **M4** PEFT teacher (ViT + LoRA/adapter/prompt/full-FT sweep)
- **M5** Distillation (streaming causal student)
- **M6** Compression (INT8 PTQ/QAT + pruning)
- **M7** Multimodal (NVGesture modality ablation)
- **M8** Frontier + demo + paper assets (`make repro-main`)

## Decision thresholds
- ViT-B PEFT won't fit 8 GB even at 8f/172px + checkpointing → ViT-S/DINOv2-S;
  still infeasible → skeleton backup (efficiency-framed).
- PEFT teacher fails to beat from-scratch student → reposition as a
  distillation+quantization efficiency study (still publishable), say so.
- INT8 PTQ accuracy drop > ~2–3 pts → run QAT, report both, frame the trade-off.
- Long run threatens timeline → shrink clip/res or subsample for *ablations*,
  but report **final headline numbers on the full official splits**.

## Reproducibility & honesty (enforced every results commit)
Seed + GPU + CUDA + torch versions logged · config committed + command in
`EXPERIMENTS.md` · official splits, no test leakage · fvcore FLOPs (protocol
noted), warmed-up latency · reported numbers labeled as reported · empty `TODO`
cells until a real run fills them · accuracy drops reported, never hidden.

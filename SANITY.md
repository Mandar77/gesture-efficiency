# Sanity Checks & Audit Log

Auditable record of verification checks so every headline number is traceable
when writing the paper. Reproduce with `scripts/verify_peft_sweep.py` and the
`experiments/**/*.json` artifacts. Hardware of record: single RTX 4060 Laptop
(8 GB), CUDA 12.4, torch 2.6.0+cu124, Python 3.12, seed 42.

---

## PEFT sweep verification (pre-M5 gate, 2026-07-11)

Command: `python scripts/verify_peft_sweep.py --config configs/peft_lora.yaml`

**Backbone is ViT-S/16** (`vit_small_patch16_224`, ~22M backbone params) — *not*
ViT-B. This is why full-FT peak VRAM is modest; it is correct, not a bug.

**Teacher-feature cache: NOT used.** There is no precompute-cache code path; the
teacher does a live ViT forward/backward in every arm, so all VRAM/latency
numbers are directly comparable.

Per-arm (all arms built from the SAME config; only what-is-trainable varies):

| arm      | frames | res | bs | trainable % (overall) | backbone trainable % | train-step peak VRAM |
|----------|:------:|:---:|:--:|:---------------------:|:--------------------:|:--------------------:|
| none     |   8    | 224 | 8  | 14.11 %               | 0.000 % (frozen probe) | 325.9 MB           |
| lora     |   8    | 224 | 8  | 14.86 %               | **1.011 %**          | 585.6 MB             |
| adapter  |   8    | 224 | 8  | 17.98 %               | 5.208 %              | 611.2 MB             |
| prompt   |   8    | 224 | 8  | 14.12 %               | 0.000 %              | 1717.0 MB †          |
| full_ft  |   8    | 224 | 8  | **100.000 %**         | **100.000 %**        | 623.1 MB             |

- **Identical clip/res/batch across all arms: YES** — `(frames=8, res=224, bs=8)`
  for every arm. The PEFT-vs-full-FT comparison is apples-to-apples.
- **full_ft is a genuine full fine-tune: YES** — backbone trainable 100.000 %
  (the earlier "~5%" concern was the *adapter* arm, not full_ft).
- Peak VRAM measured with `torch.cuda.max_memory_allocated()` **after**
  forward+backward+optimizer-step, with `reset_peak_memory_stats()` at step
  start (see `scripts/verify_peft_sweep.py::measure_arm`).
- The earlier "861 MB" full_ft figure came from a separate ad-hoc probe with
  different transient state; the authoritative post-backward figure is 623 MB.

† **Caveat — `prompt` VRAM not checkpointing-comparable.** The prompt arm
rebuilds the ViT forward from timm internals (patch_embed → blocks → norm) to
splice prompt tokens, which bypasses timm's `set_grad_checkpointing`, so its
activations are NOT checkpointed. Its **accuracy is valid**; only its VRAM is
higher than the other arms for that reason. It still fits 8 GB. (Fixable by
enabling checkpointing on the manual path if we want VRAM parity; not required
for the accuracy ablation.)

---

## Accuracy & FPS reconciliation — LoRA teacher (`jester_vits_lora_8f224`)

- **Accuracy dataset/split: official Jester VALIDATION split.** The bench logged
  `accuracy.num_samples = 14787`, which is exactly the official Jester val count
  (118,562 train / 14,787 val). The 86.49 % top-1 is Jester-val, not train, not a
  subset. (Jester's public test split has no labels, so val is the reported
  split — standard for this dataset.)
- **FPS reported both ways** (both already in the bench JSON `latency`):
  - **Single-clip streaming (bs=1): 56.09 FPS, 17.83 ms** — the real-time /
    on-device number the paper's streaming claim uses.
  - **Batched throughput (bs=8): 136.87 FPS, 58.45 ms/batch.**
  - Note: bench latency is **model forward only** (no preprocessing). End-to-end
    latency *including* frame decode/resize/normalize is measured separately by
    the webcam demo (`src/demo/webcam_demo.py`), which is the honest number for a
    live streaming claim. State which one a given table cell reports.

---

## Baseline (`jester_compact3dcnn_16f172_30ep`) — for reference

- Jester-val (14,787): 78.91 % top-1 / 96.82 % top-5.
- Single-clip (bs=1): 459.72 FPS / 2.18 ms. Batched (bs=16): logged in JSON.
- 1.17M params, 4.84 GFLOPs, peak train/infer VRAM 654/686 MB.

Interpretation: LoRA teacher = **+7.6 pts** over the from-scratch baseline while
training ~1 % of the backbone, but at ~14× FLOPs and ~8× slower single-clip —
this is the accuracy/efficiency gap that M5 distillation must close.

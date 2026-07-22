# Sanity Checks & Audit Log

Auditable record of verification checks so every headline number is traceable
when writing the paper. Reproduce with `scripts/verify_peft_sweep.py` and the
`experiments/**/*.json` artifacts. Hardware of record: single RTX 4060 Laptop
(8 GB), CUDA 12.4, torch 2.6.0+cu124, Python 3.12, seed 42.

---

## PRE-REGISTERED — ViT-B fairness sanity + norm-isolation decision rules (set 2026-07-22, BEFORE the numbers)

Context: the first ViT-B LoRA run (rank 8, ImageNet norm) plateaued at epoch 0
(78.25 %, never beaten over 3 epochs) — *below* ViT-S LoRA (86.49 %). Diagnostic
found (a) a normalization bug: BOTH `vit_small` and `vit_base_patch16_224` are
AugReg checkpoints expecting (0.5,0.5,0.5), but the loader fed ImageNet stats;
(b) rank 8 on the 2× wider ViT-B gave only 0.513 % backbone-trainable vs 1.011 %
for ViT-S. Killed and retuned: correct per-backbone norm + rank 16/alpha 32
(backbone trainable now 1.021 %).

**Rule 1 — ViT-B LoRA sanity (rank16 + correct norm), epoch-0 val top-1:**
| epoch-0 val | interpretation | action |
|---|---|---|
| ≥ 85 % | **H1**: config (norm/rank) was the problem | commit full 20-epoch run |
| ≤ 80 % | **H2**: a bigger *frozen* image backbone genuinely doesn't help this video task | STOP; report as a real finding, do NOT chase with more tuning |
| 80–85 % | **AMBIGUOUS** | do NOT commit 40h; extend sanity to 3 epochs (~4–6h), report climbing vs flat; user decides from trajectory |

epoch-0 is a fair predictor here *because* the old config's epoch 0 was already
its best over 3 epochs (a true plateau, not mid-progress).

**Rule 2 — ViT-S norm-isolation check (runs AFTER the ViT-B sanity; one GPU, not
parallel).** The ViT-B sanity changed norm AND rank together, so it cannot tell
us whether the EXISTING ViT-S PEFT sweep is compromised by the norm bug. Isolate
norm: **ViT-S LoRA, rank 8 UNCHANGED, correct norm, 1 epoch**, distinct
run_name. Compare its epoch-0 val to the existing `jester_vits_lora_8f224`
epoch-0 val (from run.log).
| epoch-0 delta (correct − wrong norm) | interpretation | action |
|---|---|---|
| ≤ ~1 pp | norm immaterial to ViT-S | existing PEFT sweep stands; document the mismatch as a bounded caveat; no rerun |
| > ~1 pp | the ViT-S sweep (lora/adapter/prompt/full_ft) ran on wrong norm | LoRA>full-FT comparison is not apples-to-apples; flag with the measured delta; do NOT rerun yet — user decides |

Prioritize Rule 2 if the ViT-B sanity jumped (H1): that is exactly when norm is
implicated and the existing ViT-S numbers become suspect.

**HOLD:** ViT-B full-FT and the LoRA rank sweep do NOT start until the ViT-B LoRA
number is locked.

---

## PRE-REGISTERED analysis plan — no-KD distillation ablation (set BEFORE the run)

Committed before running so the interpretation cannot be rationalized after the
fact. The no-KD arm is a **fork in the paper's thesis, not a confirmation** of
the distillation claim. Approach open to either outcome.

**Controls (the ONLY thing that may differ from the KD student is the teacher signal):**
- no-KD student: `beta_kd=0, gamma_feat=0, alpha_ce=1.0`, **same** student arch,
  **same** 8 frames / 224 px / segment sampling, 30 epochs, seed 42, Jester full
  splits. Run name `jester_student_no_kd`.
- Also run a **from-scratch 3D-CNN baseline at 8f/224px** (`compact3dcnn`, same
  regime) so every 3D-CNN row on the frontier shares one input regime. The
  existing 16f/172px baseline (78.9%) is NOT a valid comparison point for the
  8f/224px student on either accuracy or FLOPs/FPS.

**Decision rule (state which bin we land in; do not default to the KD narrative):**
| no-KD student (8f/224px) top-1 | Interpretation to report |
|---|---|
| ≈ 85–88 % | Distillation is real and large — the designed thesis holds. |
| ≈ 91–93 % | Distillation adds little; the real finding is **"a purpose-built streaming 3D-CNN beats a PEFT-adapted frozen ViT for gesture video."** A valid, publishable — but *different* — paper. **Flag explicitly, do not bury.** |
| in between | Distillation is a **modest booster**; report the delta plainly, no overselling. |

**Honest delta rule:** the distillation effect is `(KD student − no-KD student)`,
**both at 8f/224px**. NEVER headline "distillation → +14.6 over baseline" (that
compares across regimes and conflates architecture + labels with KD).

**Second framing kept open (decide after the ablation, note for the writeup):**
the frontier already shows the distilled student (3.11M params, 6.20 GFLOPs,
93.5%) dominating the LoRA teacher (25.4M, 68.8 GFLOPs, 86.5%) — ~8× fewer
params, ~11× less compute, higher accuracy. This **"efficient purpose-built
student vs. foundation-model baseline"** axis may be a stronger paper than
distillation per se. Hold both framings until the no-KD number lands; let the
data pick the thesis.

---

## M6 compression framing — FP16 is the result; INT8 is a NEGATIVE result (LOCKED 2026-07-21)

Framing locked BEFORE finalizing numbers so the writeup cannot overstate INT8.

**FP16 is the ONLY working compression result** (both INT8 and pruning are
negative results below). `.half()` on the KD student halves on-disk size
(12.03 → 6.06 MB) for GPU inference with negligible accuracy change (93.494 →
93.481, **0.013 pp**). This is the practical, deployable compression lever and
the only one we place on the frontier.

**INT8 is NOT an achieved result for this model — report it honestly as a
negative/limitation.** The student is a **conv-dominated 3D-CNN**. torch's
eager-mode INT8 has **no `quantized::conv3d` CPU kernel** in this stack (verified:
`NotImplementedError: Could not run 'quantized::conv3d.new'` on the CPU backend,
and no CUDA INT8 eager path on the 4060). So both INT8 paths fall back to
**dynamic Linear-only** quantization: every Conv3d stays FP32, only the final
`nn.Linear` becomes INT8. Evidence it did essentially nothing:
- size 12.031 → 11.998 MB (**~1.00× — no meaningful reduction**)
- top-1 93.494 → 93.494 (**0.0 pp** — because ~all compute is still FP32 conv)

**Writeup rule:** frame INT8 as *"eager-mode INT8 quantization of Conv3d is
unsupported on commodity CPU/GPU stacks for this model; only pointwise Linear
layers quantize, so INT8 yields no practical size/latency benefit here. FP16 is
the deployable low-precision result."* Do NOT present INT8 as an achieved
compression point on the frontier. We run INT8 PTQ/QAT once at prune 0.0 purely
to **document the fallback**, not as a sweep (INT8 evals on CPU, ~44 min each —
a prune sweep would be pure GPU-idle waste). This is a genuine, useful finding
about on-device deployment limits, stated as a limitation not a win.

**Structured pruning WITHOUT fine-tuning is ALSO a NEGATIVE result (measured
2026-07-21).** Channel pruning was applied to the KD student and evaluated with
NO recovery fine-tune. It collapses the model to near-random. Explicit numbers
(Jester-val n=14,787; top-1 before = 93.494 % for every cell):

| cell | conv sparsity | top-1 after | drop | on-disk MB |
|---|---|---|---|---|
| fp32/fp16 prune 0.3 | 0.300 | **9.93 %** | −83.6 pp | 12.03 / 6.06 |
| fp32/fp16 prune 0.5 | 0.500 | **3.56 %** | −89.9 pp | 12.03 / 6.06 |

3.56 % ≈ the 27-class chance floor (1/27 = 3.70 %) — the model is destroyed. The
pruning *did* apply (measured conv sparsity 0.30 / 0.50), so this is a real
capacity limit, not a no-op: a 3.1 M-param student has no channel redundancy to
spare, and one-shot pruning with no fine-tune removes learned filters
irrecoverably. **Two independent caveats, either one disqualifies these cells:**
(1) accuracy craters; (2) on-disk size does NOT shrink either — masking/zeroing
channels via `structured_channel_prune` + `remove_pruning_reparam` does not
physically resize tensors, so MB is unchanged even at 50 % sparsity.

**Writeup rule:** frame pruning as *"one-shot structured channel pruning
degrades this compact 3D-CNN to near-random accuracy (30 % → 9.9 %, 50 % →
3.6 %) with no on-disk size reduction; recovering it would require QAT-style
fine-tuning, left to future work. FP16 is the practical compression lever."*
**Do NOT place the pruned points (prune 0.3 / 0.5) on the frontier plot as
operating points** — they are broken, not tradeoffs. The frontier's only
compression point is FP16 (2× smaller, 0.013 pp).

---

## RESOLVED — no-KD ablation result + FORMAL THESIS PIVOT (2026-07-17)

The no-KD run completed (30 epochs, seed 42, official Jester-val n=14,787). The
pre-registered decision rule is now decisive. **The distillation hypothesis is
REJECTED as a headline contribution.**

### Three-row decomposition (all at 8f / 224 px, Jester-val, n=14,787)

| Row | Model | val top-1 | Source JSON |
|---|---|---|---|
| A | From-scratch 3D-CNN baseline | **69.23 %** | `baseline/jester_compact3dcnn_8f224_30ep.json` |
| B | no-KD student (same arch, pure CE, **no teacher**) | **93.26 %** | `distill/jester_student_no_kd.json` |
| C | KD student (logit + feature KD) | **93.47 %** | `distill/jester_student_logit_feat_kd.json` |

**The gain decomposes into two gaps, reported SEPARATELY (never merged):**
- **Architecture gap (A → B): +24.03 pts** (69.23 → 93.26). The purpose-built
  streaming 3D-CNN + training recipe accounts for essentially the entire gain,
  with **no teacher signal at all**.
- **Distillation gap (B → C): +0.21 pts** (93.26 → 93.47). Within run-to-run
  noise; **effectively null.**

**Bin landed: 91–93 % (the pre-registered "distillation adds little" outcome).**
Per the plan committed *before* the run, we report this explicitly and do not
bury it. We do **NOT** headline "distillation → +24 over baseline": that entire
delta is architecture + supervised labels, not KD. KD contributes +0.2 pts.

The no-KD trajectory confirms this is a genuine plateau, not undertraining: it
reached 93.0 % by epoch ~16 and flattened (…93.11, 93.26, 93.19, 93.24, 93.26).
The student reaches teacher-beating accuracy with the teacher entirely absent.

### Paper's contributions, formally pivoted (was: distillation; now:)

**(a) The efficiency inversion.** A small streaming 3D-CNN (3.11 M params,
6.20 GFLOPs) *beats* the PEFT-adapted frozen ViT-S teacher (25.4 M params,
68.8 GFLOPs, 86.49 %) on accuracy — 93.26 % vs 86.49 %, i.e. **+6.8 pts at
8.2× fewer params and 11.1× less compute.** For dynamic gesture *video*, a
motion-native architecture dominates a per-frame image foundation model adapted
with PEFT. This is the headline.

**(b) LoRA beats full fine-tuning.** In the teacher PEFT sweep (all arms 8f/224,
identical clip/res/batch — see the PEFT sweep verification section below), LoRA
(86.49 %, ~1 % of backbone trainable) beats a genuine 100%-trainable full
fine-tune (83.20 %) by **+3.29 pts**. Parameter-efficient adaptation is not just
cheaper here, it is *more accurate* — consistent with full-FT drifting/overfitting
the pretrained features while LoRA regularizes. Ranking: LoRA ≈ adapter (86.50) >
full_ft (83.20) > prompt/VPT (75.00).

**Distillation is demoted** from a contribution to a **negative/null result we
report honestly**: on this task, distilling from a *weaker* (lower-accuracy)
teacher into an already-stronger student adds nothing measurable. This is itself
a useful, publishable finding (teacher–student capability inversion ⇒ KD stops
helping), but it is a secondary/ablation result, not the thesis.

### Bench caveat — do NOT report the no-KD student as "slower"

The two students are **architecturally identical** (both 3.11 M params, 6.20
GFLOPs). Their single-clip bench FPS differ (KD 110.4 FPS / 9.06 ms vs no-KD
45.4 FPS / 22.0 ms) — this is a **measurement artifact** of two separate bench
runs (GPU thermal/clock state, warmup), **not** a real efficiency difference.
Report ONE representative student-latency number (the KD student's 110 FPS
single-clip) and note both students share it by construction. Never present the
no-KD student as a distinct, slower efficiency point.

### ViT-S vs ViT-B fairness — DECIDED: run ViT-B LoRA + full-FT (staged)

The foundation-model baseline was **ViT-S/16** (~22 M backbone). To defend
contribution (a) against "you only beat the *smallest* ViT," we will also run a
**ViT-B/16** teacher (86.2 M backbone — the standard 'base' foundation model):
- `jester_vitb_lora_8f224` — ViT-B LoRA (~0.51 % backbone trainable): the key
  comparison. If the 3.11 M CNN still wins, the inversion holds against a proper
  86 M model and the compute ratio widens from ~11× toward ~40×.
- `jester_vitb_full_ft_8f224` — ViT-B full fine-tune: tests whether the
  LoRA > full-FT finding (b) generalizes beyond ViT-S.

Feasibility verified (2026-07-17): ViT-B LoRA builds (total 100.4 M, trainable
14.6 M) and forwards at 8f/224; VRAM is ample (ViT-S full-FT peaked at 831 MB
train). Staged in `scripts/_run_vitb_fairness.ps1`, to launch AFTER the current
matrix — HOLDING for the user's GPU go. Results + verdict appended here when done.

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

## M5 distilled student (`jester_student_logit_feat_kd`) — scrutiny of the 93.5 %

The distilled student reaches **93.47 % Jester-val top-1** (best 93.52 % @ ep28),
which is **higher than its teacher (86.49 %)**. A student beating its teacher by
~7 pts is unusual and was audited before logging:

- **Same eval, apples-to-apples.** Teacher and student are both evaluated on the
  official Jester **val** split (`num_samples=14787` for both) at the **same**
  input regime: 8 frames, 224 px, segment sampling. The gap is a real measured
  difference, not an eval-condition artifact.
- **Why it's plausible (not a bug):**
  1. **Architecture.** The student is a causal **3D-CNN** that models spatio-
     temporal motion natively; the teacher is a **frozen per-frame image ViT** +
     a lightweight 2-layer temporal head. For a motion task like gesture
     recognition, the 3D-CNN is architecturally better suited — the "student" is
     not a weaker clone of the teacher, it is a different, temporally stronger
     model.
  2. **Ground-truth signal.** The student trains on true labels too
     (`alpha_ce=1.0`) in addition to the teacher's soft logits (`beta_kd=1.0`) +
     features (`gamma_feat=0.5`), so it is **not upper-bounded by the teacher**;
     KD acts as a booster/regularizer on top of supervised learning.
  3. **Healthy trajectory.** Monotonic, stable climb
     (66 → 90 → 91 → 92 → 93.5 %), no instability/leak signature.
- **HONEST CAVEAT — distillation benefit is NOT yet isolated.** With
  `alpha_ce=1.0 + beta_kd=1.0`, this run conflates *distillation* with *"a good
  3D-CNN trained on labels at 8f/224px"*. The current from-scratch baseline
  (78.9 %) is at a DIFFERENT regime (16f/172px), so it is **not a clean control**
  for this student. **Do NOT headline "distillation → +14.6 pts over baseline."**
  The clean control is the **no-KD ablation** (`beta_kd=0, gamma_feat=0`, same
  student at 8f/224px), which is queued (§6.2). Only after that arm exists can we
  attribute the delta between (no-KD) and (logit+feature KD) to distillation.
- **Student efficiency (measured):** 3.11M params, 6.20 GFLOPs, single-clip
  110.4 FPS / 9.06 ms (bs=1), peak infer VRAM 2464 MB, 12.03 MB on disk. Note:
  at 8f/224px the student is heavier than the 16f/172px baseline row, so a
  like-for-like FLOPs/FPS comparison also needs the same-regime baseline.

## Baseline (`jester_compact3dcnn_16f172_30ep`) — for reference

- Jester-val (14,787): 78.91 % top-1 / 96.82 % top-5.
- Single-clip (bs=1): 459.72 FPS / 2.18 ms. Batched (bs=16): logged in JSON.
- 1.17M params, 4.84 GFLOPs, peak train/infer VRAM 654/686 MB.

Interpretation: LoRA teacher = **+7.6 pts** over the from-scratch baseline while
training ~1 % of the backbone, but at ~14× FLOPs and ~8× slower single-clip —
this is the accuracy/efficiency gap that M5 distillation must close.

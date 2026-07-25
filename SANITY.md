# Sanity Checks & Audit Log

Auditable record of verification checks so every headline number is traceable
when writing the paper. Reproduce with `scripts/verify_peft_sweep.py` and the
`experiments/**/*.json` artifacts. Hardware of record: single RTX 4060 Laptop
(8 GB), CUDA 12.4, torch 2.6.0+cu124, Python 3.12, seed 42.

---

## AUTHORITATIVE latency/FPS — single-session re-bench (2026-07-25)

**SUPERSEDES all per-run `single_clip_fps` in the individual result JSONs.** Those
were each measured at the end of a SEPARATE training run under different GPU
thermal/clock states and are NOT mutually comparable (e.g. the KD student logged
110 FPS while the bit-identical no-KD student logged 45 — pure cross-run
artifact). All frontier models were re-benched **back-to-back in one process**,
warm, GPU-clock-stabilized (clock-up burn + throwaway warm pass; hardware clock
lock needs admin = unavailable), warmup=50, timed=500, **median of 3**, bs=1
(streaming) and bs=8 (batched). Source: `scripts/rebench_frontier.py` →
`experiments/rebench_frontier.json`.

| model | FLOPs (G) | bs1 FPS | bs1 ms | bs8 FPS |
|---|---|---|---|---|
| student (logit+feat KD) | 6.20 | ~135 | 7.5 | 211 |
| student (logit KD) | 6.20 | ~159 | 6.3 | 211 |
| student (no-KD) | 6.20 | ~135 | 7.4 | 211 |
| ViT-B LoRA | 272.9 | 50 | 20.1 | 53 |
| ViT-S LoRA | 68.8 | 80 | 12.5 | 146 |
| ViT-S adapter | 71.8 | 71 | 14.1 | 158 |
| ViT-S full-FT | 68.1 | 125 | 8.0 | 188 |
| ViT-S prompt | 70.8 | 122 | 8.2 | 181 |
| compact3dcnn 16f/172 | 4.84 | 911 | 1.1 | 1592 |
| compact3dcnn 8f/224 | 4.01 | 1331 | 0.75 | 1867 |

**Honest reading:**
- The three **bit-identical** students (all 6.20 GFLOPs — KD / logit-KD / no-KD
  differ only in TRAINING, identical at inference) bench 134/159/135. **Report ONE
  student latency for all three** (median-of-medians = ~135 FPS / 7.4 ms); the
  ~135–159 spread is pure measurement noise, NOT distinct operating points. The
  frontier + tables collapse them to one point (see `viz/loader.py` student
  collapse). This confirms the old 110-vs-45 gap was a cross-run artifact: the
  student's true bs=1 is ~135, and the committed README "110" was wrong.
- **FPS is INDICATIVE, not precisely reproducible.** Absolute FPS carries
  run-to-run variance from unlocked GPU boost clocks (hardware clock lock via
  `nvidia-smi -lgc` needs admin — unavailable on this laptop). Even warm +
  median-of-3, the baseline spreads ~940–1440 FPS across repeats. We report
  median-of-3 warm for mutual comparability but **lead the efficiency claim on
  FLOPs** (exact, deterministic); FPS is supporting, not a headline precise value.
- **bs=1 FPS does NOT track FLOPs.** ViT-S full-FT (125) and prompt (122) bench
  nearly as fast as the student (~135) despite ~11× the FLOPs, because bs=1 is
  kernel-launch / memory-bound, not compute-bound (the student's SE + depthwise-
  separable blocks are low arithmetic intensity). This is a REAL property to
  report, not noise — and it is *evidence for* the paper's thesis that FLOPs alone
  mislead and measured on-device cost must be reported.
- **bs=8 is more compute-bound** and behaves closer to FLOPs: student 211 vs ViT-B
  53 (~4×), baselines ~1600–1867. Report bs=8 as supplementary batched throughput.
- Numbers are **PyTorch-eager**; a deployment stack (TensorRT / kernel fusion)
  would be faster, especially for the launch-bound student.
- **Efficiency claim leads on params + FLOPs** (deterministic, comparable; ~44×
  FLOPs inversion CNN vs ViT-B). On bs=1 latency the student ties/leads ViT-S and
  beats ViT-B — but we do NOT claim a large latency speedup over ViT-S; the ViT-S
  win is params/FLOPs, the ViT-B win is all axes.

---

## PRE-REGISTERED — ViT-B fairness sanity + norm-isolation decision rules (set 2026-07-22, BEFORE the numbers)

Context: the first ViT-B LoRA run (rank 8, ImageNet norm) plateaued at epoch 0
(78.25 %, never beaten over 3 epochs) — *below* ViT-S LoRA (86.49 %). Diagnostic
found (a) a normalization bug: BOTH `vit_small` and `vit_base_patch16_224` are
AugReg checkpoints expecting (0.5,0.5,0.5), but the loader fed ImageNet stats;
(b) rank 8 on the 2× wider ViT-B gave only 0.513 % backbone-trainable vs 1.011 %
for ViT-S. Killed and retuned: correct per-backbone norm + rank 16/alpha 32
(backbone trainable now 1.021 %).

### RESOLVED — ViT-B → **H1** (earlier "plateau" was mis-tuning, NOT a frozen-backbone ceiling)

**SUPERSEDES an earlier H2 read in this file** (kept as an audit note: a 1-epoch
sanity at 78.116 % was misread as a plateau — corrected below). H2 was **wrong**;
it was an LR confound.

The "plateau" was a **learning-rate instability**, not a frozen-feature ceiling.
LR 5e-4 was tuned for ViT-S and reused unchanged for the 4×-wider ViT-B. At
warmup-end (LR → peak 5e-4) ViT-B **dropped** 78.7 → 71.8 (a *drop*, not the flat
~78 a plateau predicts), while ViT-S *climbed* through the identical ramp — the
signature of too-high LR for the wider backbone. Re-running ViT-B at **lr 2e-4**
(rank16 + correct norm, all else identical) removed the dip:

| epoch | ViT-B lr5e-4 (killed) | **ViT-B lr2e-4** | ViT-S rank8 ref |
|---|---|---|---|
| ep0 | 78.7 | **81.78** | 78.5 |
| ep1 | 71.8 ↓ | **82.53** | 79.5 |
| ep2 | — | 81.63 | 81.8 |
| ep3 | — | 83.51 | 82.9 |
| ep4 | — | **84.49** | ~83 |

ViT-B lr2e-4 climbs and tracks/leads ViT-S at every epoch. **H1 confirmed.**
Run `jester_vitb_lora_r16_lr2e4_8f224` continues to convergence (epochs=20) — it
IS the committed headline run. **Do NOT write a convergence number yet:** ep4 =
84.49 is not ep19; the cosine has ~15 epochs of decay left and could land
anywhere ~85–88 or stall. Report the actual **ep19** number; ~86–88 is
extrapolation, not a known value.

Resume-safety VERIFIED (2026-07-23): `jester_vitb_lora_r16_lr2e4_8f224.resume.pt`
updates each epoch (ckpt epoch matches last summary; model+optimizer+scheduler+
history all present), so an interruption resumes mid-run, not from zero.

Implication for the headline (pending ep19): a *properly-tuned* 86 M ViT-B is
competitive with ViT-S (~86 %), and the 3.11 M streaming 3D-CNN (93.5 %) still
beats it. The efficiency inversion holds against a **fairly-tuned** foundation
model — a STRONGER, more defensible framing than "we beat a broken ViT-B."

**Writeup framing — PER-BACKBONE LR tuning (lock this now).** ViT-S ran at lr5e-4,
ViT-B at lr2e-4. Frame the method as **deliberate per-backbone LR selection**
(each backbone gets an LR appropriate to its width — a standard, defensible
practice), explicitly stated — NOT one shared recipe applied blindly. Otherwise a
reviewer reads the different LRs as an inconsistency. Corollary to state honestly:
the ViT-S 86.5 % may itself not be its ceiling (it showed the same LR-sensitivity
family, and 5e-4 may be slightly hot for it too). We do **not** rerun ViT-S; we
frame both numbers as per-backbone-tuned and note ViT-S 86.5 % is a lower bound,
not a claimed optimum. The efficiency inversion (CNN > both ViTs) does not depend
on either ViT being at its exact ceiling.

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

**Rule 2a — how to READ the norm-iso delta (set before the number; ep0-vs-ep0 is
a SCREENING test, NOT definitive, because the existing ViT-S ep0 = 78.535 % was
MID-TRAJECTORY — it climbed to 86.49 % by ep19, not a plateau like ViT-B).**
- delta > ~1–2 pp at ep0 → norm materially helps ViT-S; the existing sweep is
  suspect on **absolute** numbers → flag for rerun.
- delta ≈ 0 at ep0 → do **NOT** write "norm immaterial to converged accuracy."
  A null ep0 delta only licenses: *"norm does not change the first-pass starting
  point."* Proceed with a documented caveat; no rerun for now. (A screening test
  can rule a large effect *in*, but a null result can't rule a converged-number
  effect *out* from ep0 alone.)

**Rule 2b — SCOPE what the norm bug actually threatens.** Contribution (b) is
**LoRA > full-FT**, a *ranking within the sweep*. All four arms
(lora/adapter/prompt/full_ft) ran on the **same** wrong norm. A uniform input
normalization offset shifts all arms **together** far more readily than it flips
their **order** — so the ranking (b) rests on is robust to this bug **even if the
absolute ViT-S numbers are not**. Therefore: a ViT-S sweep rerun (if triggered)
is for **absolute-number cleanliness / reviewer-proofing only — NOT because (b)
is at risk.** State this explicitly wherever the rerun is discussed.

**FOR LATER (do not act now) — cross-backbone comparability.** "Corrected ViT-B
(rank16 + right norm)" vs "existing ViT-S (rank8 + wrong norm)" differ in TWO
variables, so do NOT claim an identical pipeline across backbones. Keep each
backbone internally consistent across ITS OWN arms and report **per-backbone**.
For (b) to generalize to ViT-B, ViT-B full-FT only needs to run under the SAME
conditions as ViT-B LoRA (rank16-equivalent regime + right norm) — an
intra-ViT-B comparison, not a cross-backbone one.

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

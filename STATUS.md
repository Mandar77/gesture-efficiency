# STATUS — unattended window (2026-07-28 → user back eve of Aug 3)

Single source of truth for the unmonitored run. Entries are timestamped. User
approved a SHORT self-gating plan: finish the one open run (5e-5 ViT-B full-FT),
apply pre-registered gates automatically, STOP+HOLD on any anomaly, then go IDLE.
**No new runs to fill time. Rank sweep held. Nothing pushed** (user pushes on
return).

## Plan (ordered)
1. Resume 5e-5 ViT-B full-FT from the **ep3 checkpoint** → epoch 20 (~16 ep, ~32h).
   Must log "Resumed ... epoch 4" (NOT fresh ep0). Healthy → bench + update
   SANITY/README/frontier + **commit (do NOT push)**. Then idle.
2. LoRA rank sweep: **HELD — never auto-start.** At most a recommendation below.

## Self-gate thresholds (5e-5 full-FT; checked at each epoch summary)
Healthy ref through ep3: 78.59 → 77.39 → 81.28 → 82.41 (best 82.41 @ ep3).
- **COLLAPSE → kill (ckpt-safe) + STOP + report, do NOT relaunch at another LR:**
  val ≤ 10 % any epoch; OR drops > 8 pp below running best and doesn't recover to
  within 3 pp of best within 2 epochs; OR NaN/inf loss / hard crash.
- **STALL → finish to ep20 but FLAG (never push):** no new best for 5 consecutive
  epochs → complete (number still valid) but mark "possibly plateaued below LoRA;
  user decides if it supports (b) or needs rerun." Reaching ep20 ≠ clean result.
- **HEALTHY → continue** otherwise. Normal ±1–2 pp wobble is NOT a trigger.

## Reboot / crash-loop handling
- Clean power-cycle → resume same run from last epoch checkpoint (confirm "Resumed
  at ep N"). Never restart from scratch, never start something new.
- **Circuit breaker:** if the SAME run resumes and dies again at ~the same point
  **>2 times** (crash loop, not clean power-cycles), STOP, write it here, go idle.
  Do not relaunch into a crash loop for days.

## Log
- **2026-07-28 (setup):** STATUS.md created. About to resume 5e-5 full-FT from ep3
  ckpt (verified epoch=3, model+opt+sched present, trajectory 78.59/77.39/81.28/82.41).
  Git at ab3da4f (all prior work pushed). GPU idle. Rank sweep NOT queued.
- **2026-07-28 (resume OK):** Launched resume (PID 23180). VERIFIED **"epoch 4 it
  0/14820"** — resumed at ep4, NOT fresh ep0 (warm loss 0.88 not ~3.4; LR 4.85e-5 =
  correct ep4 cosine position). Training to ep20 under the self-gates above. Agent
  watching run.log per epoch; will log each epoch's number + gate decision here.
  Nothing pushed. Rank sweep held.
- **2026-07-29 05:13 (Windows-update reboot #1):** machine rebooted for OS updates
  mid-epoch-6 (it 11640/14820). Trajectory through ep5: 78.59 / 77.39 / 81.28 /
  82.41 / 82.17 / 82.14 — HEALTHY (climbed then plateauing ~82%, no gate trip;
  tracking ~5pp below ViT-B LoRA 87.7, consistent with (b) holding). ep5 checkpoint
  intact (opt+sched present); only in-progress ep6 lost (~1.6h). NOT a crash loop
  (first reboot of this run) → circuit breaker does not apply.
- **2026-07-29 14:30 (resume #2 OK):** relaunched (PID 32564). VERIFIED log
  **"Resumed ... starting at epoch 6 (best val 82.410 so far)"** — clean resume at
  ep6, best-so-far restored, no fresh ep0. Continuing to ep20 under the same gates.
  User disabling OS updates for the rest of the session. Nothing pushed.
- **2026-07-31 10:36 (TRAINING COMPLETE, gate HEALTHY):** full-FT reached ep19.
  Full trajectory: 78.59 / 77.39 / 81.28 / 82.41 / 82.17 / 82.14 / 82.56 / 82.55 /
  83.91 / 83.60 / 84.02 / 84.45 / 84.33 / 84.89 / 84.97 / 85.22 / 85.14 / **85.41
  (best @ep17)** / 85.37 / **85.41 (ep19)**. Gentle climb to convergence, never 5
  epochs without a new best → **HEALTHY, not STALLED**. Final model checkpoint
  saved (`jester_vitb_full_ft_lr5e5_8f224.pt`). **ViT-B full-FT converged = 85.41 %
  — vs ViT-B LoRA 87.69 %: (b) LoRA > full-FT HOLDS on ViT-B (+2.28 pp).**
- **2026-07-31 → 08-03 (ANOMALY — bench hung):** run.log froze immediately after
  "Saved checkpoint"; the post-training efficiency bench never produced its result
  JSON. ~15 python processes stayed resident ~3 days holding ~3.6 GB GPU (orphaned
  DataLoader worker pools — the Windows DataLoader deadlock seen before, hit during
  the accuracy eval). No reboot (boot still 07-29). Per the STOP+HOLD-on-anomaly
  rule: did NOT kill or re-run unattended. Training result is SAFE (ckpt + val
  numbers above); only the params/FLOPs/FPS JSON is missing. HOLDING for user
  (back 08-03) to approve killing the stale processes + re-benching (short,
  deterministic; training already done). Nothing pushed.

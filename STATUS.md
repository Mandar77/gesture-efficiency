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

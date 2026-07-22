# Fixed-ViT-B PLATEAU CONFIRMATION under the COMMITTED schedule. Runs AFTER
# norm-iso -- one GPU. rank16/alpha32 + correct norm.
#
# WHY epochs=20 (not a short 4-epoch run):
#   * ep0->ep2 are IDENTICAL between a 4-epoch and 20-epoch schedule (same 2-epoch
#     warmup), so the early trend is clean either way.
#   * But ep3 under a 4-epoch cosine has already decayed toward 0 (~2.5e-4->0),
#     while ViT-S's 20-epoch schedule was at ~4.96e-4 (near peak) at ep3. A short
#     run's ep3 therefore conflates a REAL plateau with truncation-induced
#     undertraining -- ambiguous exactly where H2 (now a headline-supporting
#     claim) needs a clean signal.
#   * Under epochs=20, ep3 lands at near-peak LR (where it SHOULD move if it's
#     going to), AND if it climbs the SAME run continues to convergence -- no
#     wasted epochs, no schedule discontinuity from a restart.
#
# DECISION at the ep3 summary (anchored to ViT-S early climb 78.5->79.5->81.8->82.9):
#   climbing (>= ~81-82 by ep3)      -> H2 WRONG. Do NOT kill; this run's epochs
#                                       ARE the committed 20-epoch run. Report; user
#                                       confirms the commit; it just keeps going.
#   flat ~76-79 no trend through ep3 -> H2 CONFIRMED (at near-peak LR). Kill.
#   in between                       -> no action; keeps running; watch ep5-6.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224",
    "peft.method=lora", "peft.lora_rank=16", "peft.lora_alpha=32",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_lora_r16_8f224"
)

Write-Output "================ ViT-B LoRA plateau/committed run (rank16, correct norm, epochs=20) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ vitb committed run exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "VITB RUN COMPLETE" } else { Write-Output "VITB RUN ENDED (exit $code)" }

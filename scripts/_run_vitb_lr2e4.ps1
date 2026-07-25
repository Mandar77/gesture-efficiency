# ViT-B LoRA at a LOWER LR (2e-4, down from the ViT-S-tuned 5e-4). The 5e-4 run
# showed a 7pp DROP (78.7->71.8) exactly at warmup-end / peak-LR -- an instability
# signature, NOT the flat plateau H2 predicts. Both ViT-B runs dip through the LR
# ramp while ViT-S CLIMBS through the identical ramp (bigger adapter capacity ->
# bigger dip), i.e. 5e-4 is too hot for the 4x-wider ViT-B. LR was never re-tuned
# for ViT-B, so "bigger backbone doesn't help" (H2) is untested vs "mis-tuned".
# Prior work (AIM: frozen ViT-L > ViT-B on K400) says a fair ViT-B should adapt.
#
# Everything else UNCHANGED from the fixed config: rank16 / alpha32 / correct
# per-backbone norm / epochs=20 / warmup=2 / cosine. Only train.lr 5e-4 -> 2e-4.
# Distinct run_name preserves the LR5e-4 trajectory on disk.
#
# DIAGNOSTIC (the warmup dip), watch ep0->ep2:
#   dip DISAPPEARS, climbs like ViT-S (~79-81 by ep2) -> plateau was mis-tuning
#     (H1). Let it continue to convergence (this IS the committed run); report.
#   STILL dips / stuck ~78 -> try lr=1e-4 once; if that also plateaus, H2 is
#     finally credible (fairly tested). Decide at ep3, but only after LR is sane.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224",
    "peft.method=lora", "peft.lora_rank=16", "peft.lora_alpha=32",
    "train.lr=2e-4",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_lora_r16_lr2e4_8f224"
)

Write-Output "================ ViT-B LoRA lr=2e-4 (rank16, correct norm, epochs=20) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ vitb lr2e4 exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "VITB LR2E4 COMPLETE" } else { Write-Output "VITB LR2E4 ENDED (exit $code)" }

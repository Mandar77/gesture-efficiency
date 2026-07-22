# ViT-B LoRA 1-EPOCH SANITY run (before committing ~40h). Tests BOTH fixes:
#   - correct normalization: loader now reads (0.5,0.5,0.5) from the ViT-B
#     pretrained_cfg instead of hardcoded ImageNet (both ViTs were mismatched).
#   - scaled LoRA: rank 16 / alpha 32 -> backbone trainable back to 1.021%
#     (was 0.513% at rank 8; ViT-S is 1.011%).
#
# Decision rule (user):
#   epoch-0 val ~85%+  -> config was the problem (H1); commit full 20-epoch run.
#   epoch-0 val ~78-80% -> bigger frozen backbone genuinely doesn't help this
#                          video task (H2); STOP, report as a real finding.
#
# 1 epoch only via train.epochs=1. Distinct run_name so it can't clobber the
# (killed) rank-8 run's artifacts.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224",
    "peft.method=lora", "peft.lora_rank=16", "peft.lora_alpha=32",
    "train.epochs=1",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_lora_r16_sanity1ep"
)

Write-Output "================ ViT-B LoRA sanity (rank16, correct norm, 1 epoch) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ sanity exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "SANITY COMPLETE" } else { Write-Output "SANITY FAILED" }

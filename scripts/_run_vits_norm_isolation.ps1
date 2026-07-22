# ViT-S norm-ISOLATION check (Rule 2). Runs AFTER the ViT-B sanity — ONE GPU,
# never in parallel. Isolates the normalization variable ALONE: ViT-S LoRA with
# rank 8 UNCHANGED (same as the existing jester_vits_lora_8f224) but with the
# CORRECT (0.5,0.5,0.5) norm now resolved from timm pretrained_cfg. 1 epoch.
#
# Compare epoch-0 val to the existing jester_vits_lora_8f224 epoch-0 val (78.535%,
# from run.log — note that run was mid-trajectory at ep0, not plateaued).
#   delta <= ~1 pp -> norm immaterial to ViT-S; existing sweep stands (bounded caveat).
#   delta  > ~1 pp -> ViT-S sweep ran on wrong norm; flag for possible rerun.
#
# rank 8 / alpha 16 UNCHANGED so the ONLY difference vs the existing run is norm.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_small_patch16_224",
    "peft.method=lora", "peft.lora_rank=8", "peft.lora_alpha=16",
    "train.epochs=1",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_lora_normfix_sanity1ep"
)

Write-Output "================ ViT-S LoRA norm-isolation (rank8 UNCHANGED, correct norm, 1 epoch) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ norm-isolation exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "NORMISO COMPLETE" } else { Write-Output "NORMISO FAILED" }

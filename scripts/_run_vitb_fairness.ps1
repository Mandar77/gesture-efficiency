# ViT-B fairness check (defends the efficiency-inversion headline against the
# "you only beat the smallest ViT" critique). Runs AFTER the main matrix.
# Two runs, both at 8f/224, backbone=vit_base_patch16_224 (~86M backbone):
#   1. ViT-B LoRA teacher (~1% trainable) -- the key comparison vs the 3.1M CNN.
#   2. ViT-B full fine-tune -- tests whether LoRA>full-FT generalizes past ViT-S.
# VRAM is ample (ViT-S full-FT peaked at 831 MB train). bf16 + grad-checkpointing
# already on in peft_lora.yaml. num_workers=4 prefetch_factor=1 to stay under the
# Windows commit limit. Fail-fast + per-epoch resume, same as the matrix driver.

Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

function Run-Stage {
    param([string]$Name, [string[]]$PyArgs)
    Write-Output ("================ START: " + $Name + " ================")
    & $py @PyArgs 2>&1 | Select-String -NotMatch "platform independent"
    $code = $LASTEXITCODE
    Write-Output ("================ END: " + $Name + " (exit " + $code + ") ================")
    if ($code -ne 0) {
        Write-Output ("!!!! STAGE FAILED: " + $Name + " (exit " + $code + ") -- STOPPING !!!!")
        exit $code
    }
}

# 1. ViT-B LoRA teacher (the key fairness comparison)
Run-Stage "vitb_lora" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224", "peft.method=lora",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_lora_8f224")

# 2. ViT-B full fine-tune (does LoRA>full-FT hold for ViT-B too?)
Run-Stage "vitb_full_ft" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224", "peft.method=full_ft",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_full_ft_8f224")

# regenerate tables/figures so the ViT-B rows land on the frontier
Run-Stage "make_tables"  @("scripts\make_tables.py",  "--out", "paper\tables.tex")
Run-Stage "make_figures" @("scripts\make_figures.py", "--out", "paper\figures")

Write-Output "VITB FAIRNESS COMPLETE"

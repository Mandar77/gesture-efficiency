# RESUME the 5e-5 ViT-B full-FT run from its ep3 checkpoint -> epoch 20.
# UNATTENDED-WINDOW run (user away ~6 days). This is a SINGLE train call: no
# chain, no supervisor, nothing that can advance to the rank sweep. Same config
# as the launch (lr=5e-5, full_ft, vit_base, correct norm, epochs=20). The engine
# auto-resumes from checkpoints/peft/jester_vitb_full_ft_lr5e5_8f224.resume.pt
# (must log "Resumed ... epoch 4", NOT fresh ep0 -- verified by the caller).
#
# Gates are applied by the AGENT watching run.log (see STATUS.md), not by this
# script. This script just trains; the agent kills on a gate failure.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224",
    "peft.method=full_ft",
    "train.lr=5e-5",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_full_ft_lr5e5_8f224"
)

Write-Output "================ RESUME ViT-B full-FT lr=5e-5 from ep3 -> ep20 ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ vitb full-ft 5e5 exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "VITB FULLFT5E5 COMPLETE" } else { Write-Output "VITB FULLFT5E5 ENDED (exit $code)" }

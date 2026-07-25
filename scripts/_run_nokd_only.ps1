# Relaunch ONLY the no-KD ablation (the baseline already completed: 69.23%).
# The first attempt crashed in epoch 0 with a Windows commit-limit MemoryError
# (a 4.6 MiB alloc failed despite 6.8 GB free RAM) because the no-KD process
# still loaded the ~25M-param ViT teacher onto the GPU even though KD weights are
# 0. Fix: give teacher_ckpt a non-existent path (`none`) -> distill_student.py
# takes its documented no-teacher path (Path('none').exists() is False ->
# teacher=None, loss_fn=None -> engine default CE with label_smoothing=0.1). This
# is IDENTICAL to the KD run's CE term (alpha_ce=1.0) and is a strictly cleaner
# control (the teacher provably plays zero role). Also drop prefetch_factor 2->1
# for extra commit headroom.
#
# Args passed as an array (NO backtick line-continuations, NO empty-string arg)
# to avoid the PowerShell tokenizer error the previous version hit.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\distill_student.py",
    "--config", "configs\distill_student.yaml",
    "--set",
    "distill.beta_kd=0",
    "distill.gamma_feat=0",
    "distill.teacher_ckpt=none",
    "data.prefetch_factor=1",
    "output.run_name=jester_student_no_kd"
)

Write-Output "================ no-KD ablation (jester_student_no_kd) -- teacher SKIPPED ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ no-KD exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "NOKD COMPLETE" } else { Write-Output "NOKD FAILED" }

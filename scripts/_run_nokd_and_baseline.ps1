# Sequential: (1) no-KD ablation, then (2) same-regime 8f/224 baseline.
# One at a time to avoid GPU contention. Both resumable via per-epoch ckpts.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"

Write-Output "================ RUN 1: no-KD ablation (jester_student_no_kd) ================"
& $py scripts\distill_student.py --config configs\distill_student.yaml `
    --set distill.beta_kd=0 distill.gamma_feat=0 output.run_name=jester_student_no_kd 2>&1 |
    Select-String -NotMatch "platform independent"
Write-Output "================ done: no-KD ================"

Write-Output "================ RUN 2: same-regime baseline (jester_compact3dcnn_8f224_30ep) ================"
& $py scripts\train.py --config configs\baseline_jester_8f224.yaml 2>&1 |
    Select-String -NotMatch "platform independent"
Write-Output "================ done: baseline_8f224 ================"
Write-Output "NOKD+BASELINE COMPLETE"

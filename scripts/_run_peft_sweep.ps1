# Sequential PEFT sweep: adapter -> prompt -> full_ft. Each resumable via its
# per-epoch checkpoint. Runs one at a time to avoid GPU contention on the 4060.
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$methods = @("adapter","prompt","full_ft")
foreach ($m in $methods) {
    Write-Output "================ PEFT method: $m ================"
    & $py scripts\train_peft_teacher.py --config configs\peft_lora.yaml `
        --set peft.method=$m data.num_workers=4 data.prefetch_factor=2 `
              output.run_name="jester_vits_${m}_8f224" 2>&1 |
        Select-String -NotMatch "platform independent"
    Write-Output "================ done: $m ================"
}
Write-Output "PEFT SWEEP COMPLETE"

# Supervisor: wait for the ViT-B LoRA sanity process to exit, THEN run the ViT-S
# norm-isolation check (Rule 2). One GPU -> strictly sequential, never parallel.
# The norm-isolation run is queued by the user to run AFTER the sanity regardless
# of the sanity outcome (it isolates a different variable). It does NOT commit any
# full 40h run -- that decision stays with the user.
param([int]$SanityPid)

Set-Location "D:\KhouryGithub\gesture-efficiency"

Write-Output ("Chain: waiting for ViT-B sanity PID " + $SanityPid + " to exit...")
try { Wait-Process -Id $SanityPid -ErrorAction Stop }
catch { Write-Output ("sanity PID " + $SanityPid + " not found (already exited); proceeding.") }
Write-Output "ViT-B sanity finished. Starting ViT-S norm-isolation in 20s..."
Start-Sleep -Seconds 20

& "D:\KhouryGithub\gesture-efficiency\scripts\_run_vits_norm_isolation.ps1"
Write-Output "NORMISO CHAIN COMPLETE"

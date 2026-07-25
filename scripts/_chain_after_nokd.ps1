# Supervisor: wait for the running no-KD process to finish, THEN run the full
# matrix driver. Avoids GPU contention (only one training on the 8GB card at a
# time). The no-KD run was launched separately (PID passed in as arg).
param([int]$NokdPid)

Set-Location "D:\KhouryGithub\gesture-efficiency"

Write-Output ("Supervisor: waiting for no-KD PID " + $NokdPid + " to exit...")
try {
    Wait-Process -Id $NokdPid -ErrorAction Stop
} catch {
    # Process already gone (finished before we attached) -- proceed.
    Write-Output ("no-KD PID " + $NokdPid + " not found (already exited); proceeding.")
}
Write-Output "no-KD finished. Starting full matrix in 20s..."
Start-Sleep -Seconds 20   # let GPU memory fully release before the next run

& "D:\KhouryGithub\gesture-efficiency\scripts\_run_full_matrix.ps1"
Write-Output "CHAIN COMPLETE"

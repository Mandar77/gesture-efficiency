# Supervisor: wait for the norm-iso run to fully complete (its driver prints
# NORMISO COMPLETE / NORMISO FAILED to _normiso.out after the python exits,
# including its bench step), THEN run the fixed-ViT-B 4-epoch plateau check.
# One GPU -> strictly sequential. Waits on the completion MARKER (robust to the
# bench step) rather than a raw PID.
$ErrorActionPreference = "Continue"
Set-Location "D:\KhouryGithub\gesture-efficiency"
$marker = "D:\KhouryGithub\gesture-efficiency\experiments\_normiso.out"

Write-Output "Chain: waiting for norm-iso completion marker..."
while ($true) {
    if (Test-Path $marker) {
        $txt = Get-Content $marker -Raw -ErrorAction SilentlyContinue
        if ($txt -match "NORMISO COMPLETE" -or $txt -match "NORMISO FAILED") { break }
    }
    Start-Sleep -Seconds 30
}
Write-Output "norm-iso finished. Starting fixed-ViT-B plateau check in 20s..."
Start-Sleep -Seconds 20

& "D:\KhouryGithub\gesture-efficiency\scripts\_run_vitb_plateau_check.ps1"
Write-Output "PLATEAU CHAIN COMPLETE"

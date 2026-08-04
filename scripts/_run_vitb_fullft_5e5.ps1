# ViT-B FULL fine-tune at lr=5e-5 (full-FT convention, ~4x lower than the 2e-4 that
# was too hot). Primary full-FT run for the two-part (b) [LoRA > full-FT] on ViT-B,
# to preempt the "undertuned full-FT" objection. Committed epochs=20 schedule so a
# healthy run continues seamlessly.
#
# GATE (full-FT-specific -- DIFFERENT from the LoRA gate). Full-FT dips at warmup
# even when HEALTHY (ViT-S full-FT dipped 61.9->51.3 at 5e-4 and still reached 83.2).
# So the criterion is COLLAPSE vs RECOVERABLE-DIP, decided at ep1->ep2 (one extra
# epoch vs the LoRA gate, since full-FT recovers slowly):
#   * 5e-5 dips SHALLOWER than the 2e-4 run's -8.5pp (68.7->60.2), OR dips then turns
#     UP by ep2 -> HEALTHY, continue to 20 (this is the committed full-FT run).
#   * 5e-5 dips HARDER than 2e-4, or heads toward ~3.7% chance floor with no recovery
#     -> genuinely broken; stop, report; (b) then rests on ViT-S with LR-fragility noted.
#
# ep0 context: ViT-B full-FT@2e-4 started 68.7 (above ViT-S full-FT 61.9) but ~13pp
# BELOW ViT-B LoRA's ep0 81.8 -> full-FT looks recoverable but starts far behind LoRA,
# consistent with (b) holding. 5e-5 is to preempt the objection, not because the
# outcome is in doubt.
#
# method=full_ft (100% trainable), backbone=vit_base, correct norm, lr=5e-5, epochs=20.
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

Write-Output "================ ViT-B full-FT lr=5e-5 (100% trainable, epochs=20) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ vitb full-ft 5e5 exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "VITB FULLFT5E5 COMPLETE" } else { Write-Output "VITB FULLFT5E5 ENDED (exit $code)" }

# ViT-B FULL fine-tune at lr=2e-4 -- the MATCHED-LR half of a two-part (b)
# [LoRA > full-FT] on ViT-B. Runs on the COMMITTED epochs=20 schedule so, if it
# survives the warmup transition, the SAME run continues seamlessly to convergence
# (same trick as the LoRA plateau check -- no restart, no waste).
#
# WHY sanity-gated: full-FT is LR-fragile. ViT-S full-FT at lr=5e-4 already dipped
# 61.9 -> 51.3 ep0->ep1 (too hot even for 22M); full fine-tuning 86M ViT-B params
# at 5e-4 would very likely diverge. We use lr=2e-4 (the per-backbone LR tuned for
# ViT-B LoRA) so the intra-ViT-B LoRA-vs-full-FT comparison shares one LR.
#
# DECISION at the ep0->ep1 warmup transition (warmup=2ep, so ep1 end is near-peak LR):
#   holds / climbs (no collapse through ep1) -> continue to 20; matched-LR full-FT run.
#   dips hard like ViT-S full-FT@5e-4        -> 2e-4 still too hot for 86M full-FT;
#                                               STOP, report, drop to ~5e-5 primary.
#
# method=full_ft (100% trainable), backbone=vit_base, correct per-backbone norm
# (auto from timm cfg), lr=2e-4, epochs=20. Committed run_name (a survivor becomes
# the final artifact).
Set-Location "D:\KhouryGithub\gesture-efficiency"
$py = ".venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"

$argsList = @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224",
    "peft.method=full_ft",
    "train.lr=2e-4",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_full_ft_lr2e4_8f224"
)

Write-Output "================ ViT-B full-FT lr=2e-4 (100% trainable, epochs=20) ================"
& $py @argsList 2>&1 | Select-String -NotMatch "platform independent"
$code = $LASTEXITCODE
Write-Output ("================ vitb full-ft exit code: " + $code + " ================")
if ($code -eq 0) { Write-Output "VITB FULLFT COMPLETE" } else { Write-Output "VITB FULLFT ENDED (exit $code)" }

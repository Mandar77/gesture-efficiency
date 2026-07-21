# Resume driver: everything the interrupted matrix did NOT finish, plus the
# approved ViT-B fairness runs, in one unattended fail-fast chain.
#
# Already DONE (skipped here): baselines, full PEFT sweep (lora/adapter/prompt/
# full_ft), no-KD, logit-only, logit+feat KD.
#
# Ordering:
#   1. M6 compression   (RERUN -- int8 bugs fixed in commit 98def4e)
#   2. Briareo M7        (rgb -> rgb+depth -> rgb+depth+ir)
#   3. LoRA rank sweep   (r=4 / r=16; r=8 already done)
#   4. ViT-B fairness    (ViT-B LoRA + full-FT -- defends efficiency inversion)
#   5. regenerate tables + figures
#
# Fail-fast (stop on first non-zero exit); per-epoch resume checkpoints; PEFT/ViT
# arms pinned to num_workers=4 prefetch_factor=1 (Windows commit-limit guard).

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

# 1a. M6 compression -- FP32/FP16 across the full pruning curve (GPU-fast; this
#     IS the real compression story for this conv-dominated model).
Run-Stage "compress_fp" @(
    "scripts\compress_student.py", "--config", "configs\distill_student.yaml",
    "--ckpt", "checkpoints\distill\jester_student_logit_feat_kd.pt",
    "--modes", "fp32", "fp16",
    "--prune-ratios", "0.0", "0.3", "0.5")

# 1b. M6 compression -- INT8 PTQ/QAT at prune 0.0 ONLY. INT8 is NOT a working
#     result for this model: torch has no quantized::conv3d CPU kernel, so the
#     dynamic fallback leaves ALL Conv3d in FP32 (only the final Linear
#     quantizes) -> ~0.0 pp drop, 12.03->12.00 MB. We run it once at prune 0.0
#     purely to DOCUMENT the fallback honestly, not as a sweep. int8 evals on CPU
#     (~44 min each) so a prune sweep here would be pure GPU-idle waste.
Run-Stage "compress_int8_doc" @(
    "scripts\compress_student.py", "--config", "configs\distill_student.yaml",
    "--ckpt", "checkpoints\distill\jester_student_logit_feat_kd.pt",
    "--modes", "int8_ptq", "int8_qat",
    "--prune-ratios", "0.0")

# 2. Briareo M7 modality ablation (quoted-literal lists for ast.literal_eval)
Run-Stage "briareo_rgb" @(
    "scripts\train_multimodal.py", "--config", "configs\multimodal_briareo.yaml", "--set",
    "data.modalities=['rgb']", "model.kwargs.modalities=['rgb']",
    "output.run_name=briareo_rgb")
Run-Stage "briareo_rgbd" @(
    "scripts\train_multimodal.py", "--config", "configs\multimodal_briareo.yaml", "--set",
    "data.modalities=['rgb','depth']", "model.kwargs.modalities=['rgb','depth']",
    "output.run_name=briareo_rgbd")
Run-Stage "briareo_rgbdir" @(
    "scripts\train_multimodal.py", "--config", "configs\multimodal_briareo.yaml", "--set",
    "data.modalities=['rgb','depth','ir']", "model.kwargs.modalities=['rgb','depth','ir']",
    "output.run_name=briareo_rgbdir")

# 3. ViT-B fairness FIRST: ViT-B/16 (86.2M backbone) LoRA + full-FT. These defend
#    the efficiency-inversion headline and MUST be in hand if the window is
#    interrupted again -- so they run BEFORE the (droppable) rank sweep.
Run-Stage "vitb_lora" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224", "peft.method=lora",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_lora_8f224")
Run-Stage "vitb_full_ft" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "model.kwargs.backbone=vit_base_patch16_224", "peft.method=full_ft",
    "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vitb_full_ft_8f224")

# 4. LoRA rank sweep LAST (r=8 already done) -- the droppable slack stage. If the
#    window ends, everything above (incl. ViT-B) is already secured.
Run-Stage "lora_r4" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=lora", "peft.lora_rank=4", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_lora_r4_8f224")
Run-Stage "lora_r16" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=lora", "peft.lora_rank=16", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_lora_r16_8f224")

# 5. regenerate frontier tables + figures from all committed results
Run-Stage "make_tables"  @("scripts\make_tables.py",  "--out", "paper\tables.tex")
Run-Stage "make_figures" @("scripts\make_figures.py", "--out", "paper\figures")

Write-Output "RESUME MATRIX COMPLETE"

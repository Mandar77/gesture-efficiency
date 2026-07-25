# Sequential driver for the remaining GPU matrix (~25-30h), chained back-to-back.
# User is available ~35h and wants runs queued one after another (no holding).
#
# Ordering (priority; later = first to drop if we run long):
#   [already running separately] no-KD ablation (jester_student_no_kd)
#   1. PEFT prompt arm      (resumes @ep2)
#   2. PEFT full_ft arm     (from scratch; genuine 100%-trainable full FT)
#   3. distill logit-only   (beta_kd=1, gamma_feat=0 -> isolates feature-KD)
#   4. M6 compression       (fp16 / int8_ptq / int8_qat / pruning on KD student)
#   5. Briareo M7 ablation  (rgb -> rgb+depth -> rgb+depth+ir)
#   6. LoRA rank sweep      (r=4 / r=8 already done / r=16)  [slack; last]
#   final: regenerate frontier tables + figures
#
# FAIL-FAST: each stage runs only if the previous succeeded. Exit codes checked
# via $LASTEXITCODE (NOT swallowed by the pipeline this time). Memory-hungry PEFT
# arms get num_workers=4 prefetch_factor=1 to avoid the Windows commit-limit crash
# that has bitten twice. All training uses per-epoch resume checkpoints, so a
# crash mid-run resumes rather than restarts.

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
        Write-Output ("!!!! STAGE FAILED: " + $Name + " (exit " + $code + ") -- STOPPING MATRIX !!!!")
        exit $code
    }
}

# 1. PEFT prompt arm (resumes from epoch 2 via checkpoints/peft/*.resume.pt)
Run-Stage "peft_prompt" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=prompt", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_prompt_8f224")

# 2. PEFT full_ft arm (from scratch; 100% trainable)
Run-Stage "peft_full_ft" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=full_ft", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_full_ft_8f224")

# 3. distillation logit-only (isolates feature-KD's contribution vs logit+feature)
Run-Stage "distill_logit_only" @(
    "scripts\distill_student.py", "--config", "configs\distill_student.yaml", "--set",
    "distill.beta_kd=1", "distill.gamma_feat=0", "data.prefetch_factor=1",
    "output.run_name=jester_student_logit_kd")

# 4. M6 compression on the KD student (fp16 / int8 ptq / int8 qat / structured pruning)
Run-Stage "compress_student" @(
    "scripts\compress_student.py", "--config", "configs\distill_student.yaml",
    "--ckpt", "checkpoints\distill\jester_student_logit_feat_kd.pt",
    "--modes", "fp32", "fp16", "int8_ptq", "int8_qat",
    "--prune-ratios", "0.0", "0.3", "0.5")

# 5. Briareo M7 modality ablation: RGB -> RGB+D -> RGB+D+IR.
# List values MUST be valid Python literals (ast.literal_eval): quoted strings.
# '[rgb,depth]' fails literal_eval and silently stays a raw string, which breaks
# the loader. Single-quoted PowerShell literals pass the inner quotes verbatim.
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

# 6. LoRA rank sweep [SLACK -- last, first to drop if we run long]. r=8 already done.
Run-Stage "lora_r4" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=lora", "peft.lora_rank=4", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_lora_r4_8f224")
Run-Stage "lora_r16" @(
    "scripts\train_peft_teacher.py", "--config", "configs\peft_lora.yaml", "--set",
    "peft.method=lora", "peft.lora_rank=16", "data.num_workers=4", "data.prefetch_factor=1",
    "output.run_name=jester_vits_lora_r16_8f224")

# final: regenerate the frontier tables + figures from all committed results
Run-Stage "make_tables" @("scripts\make_tables.py", "--out", "paper\tables.tex")
Run-Stage "make_figures" @("scripts\make_figures.py", "--out", "paper\figures")

Write-Output "MATRIX COMPLETE"

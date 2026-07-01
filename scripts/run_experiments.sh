#!/usr/bin/env bash
# The full experiment matrix (BRIEF §6). Each command emits one results row/JSON
# under experiments/<group>/ and a bench profile. Run after preparing the data
# (see DATA_LICENSES.md + src/data/download_data.py). These are the *real*
# headline runs on the full official splits.
#
# This script documents the exact commands; run them selectively — the full
# matrix is many GPU-hours. `make repro-main` then regenerates the frontier.
set -euo pipefail
PY="${PY:-.venv/Scripts/python.exe}"

echo "=== M3: from-scratch baseline (pipeline validation) ==="
"$PY" scripts/train.py --config configs/baseline_jester.yaml

echo "=== §6.1 PEFT sweep: LoRA / adapter / prompt / full-FT / none ==="
for method in lora adapter prompt full_ft none; do
  "$PY" scripts/train_peft_teacher.py --config configs/peft_lora.yaml \
      --set peft.method="$method" output.run_name="jester_vits_${method}_8f172"
done
# LoRA rank sweep (accuracy vs trainable params)
for r in 4 8 16; do
  "$PY" scripts/train_peft_teacher.py --config configs/peft_lora.yaml \
      --set peft.method=lora peft.lora_rank="$r" output.run_name="jester_vits_lora_r${r}"
done

echo "=== §6.2 distillation ablation: no-KD / logit-only / logit+feature ==="
TEACHER=checkpoints/peft/jester_vits_lora_8f172.pt
"$PY" scripts/distill_student.py --config configs/distill_student.yaml \
    --set distill.teacher_ckpt="$TEACHER" distill.beta_kd=0 distill.gamma_feat=0 \
          output.run_name=jester_student_no_kd
"$PY" scripts/distill_student.py --config configs/distill_student.yaml \
    --set distill.teacher_ckpt="$TEACHER" distill.beta_kd=1 distill.gamma_feat=0 \
          output.run_name=jester_student_logit_kd
"$PY" scripts/distill_student.py --config configs/distill_student.yaml \
    --set distill.teacher_ckpt="$TEACHER" distill.beta_kd=1 distill.gamma_feat=0.5 \
          output.run_name=jester_student_logit_feat_kd

echo "=== §6.3 compression: FP32 / FP16 / INT8-PTQ / INT8-QAT / pruning ==="
STUDENT=checkpoints/distill/jester_student_logit_kd.pt
"$PY" scripts/compress_student.py --config configs/distill_student.yaml \
    --ckpt "$STUDENT" --modes fp32 fp16 int8_ptq int8_qat --prune-ratios 0.0 0.3 0.5

echo "=== §6.4 clip-length / resolution sensitivity (8 vs 16 frames; 172 vs 224) ==="
for nf in 8 16; do for sz in 172 224; do
  "$PY" scripts/train.py --config configs/baseline_jester.yaml \
      --set data.num_frames="$nf" data.frame_size="$sz" \
            output.run_name="jester_baseline_${nf}f${sz}"
done; done

echo "=== §6.5 multimodal ablation: RGB / RGB+D / RGB+D+IR (NVGesture) ==="
"$PY" scripts/train_multimodal.py --config configs/multimodal_nvgesture.yaml \
    --set data.modalities='[rgb]' model.kwargs.modalities='[rgb]' \
          output.run_name=nvgesture_rgb
"$PY" scripts/train_multimodal.py --config configs/multimodal_nvgesture.yaml \
    --set data.modalities='[rgb,depth]' model.kwargs.modalities='[rgb,depth]' \
          output.run_name=nvgesture_rgbd
"$PY" scripts/train_multimodal.py --config configs/multimodal_nvgesture.yaml \
    --set data.modalities='[rgb,depth,ir]' model.kwargs.modalities='[rgb,depth,ir]' \
          output.run_name=nvgesture_rgbdir

echo "=== regenerate the frontier + tables ==="
"$PY" scripts/make_figures.py --out paper/figures
"$PY" scripts/make_tables.py --out paper/tables.tex
echo "DONE. See paper/figures and paper/tables.{tex,md}."

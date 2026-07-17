#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/sp/bin/python}"
SPLIT_ROOT="${1:-external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10}"
TASK_ID="${TASK_ID:-0}"
AGENT_ID="${AGENT_ID:-0}"
GPU="${GPU:-0}"
SPLIT_ABS="$(realpath "${SPLIT_ROOT}")"

: "${MASC_STUDENT_CKPT_DIR:?Set MASC_STUDENT_CKPT_DIR to a directory containing checkpoint.pth}"
: "${MASC_TEACHER_CKPT_DIR:?Set MASC_TEACHER_CKPT_DIR to a directory containing checkpoint.pth}"

cd external_baselines/repos/MASC_SL/MASC_SL_inference_CIFAR100_raw_4_25
"${PYTHON_BIN}" test_multi_agent_cifar100_ts.py \
  --batch_size 10000 \
  --epochs 1 \
  --num_works_multi_task "${TASK_ID}" \
  --num_works_multi_agent "${AGENT_ID}" \
  --path "${SPLIT_ABS}/masc" \
  --model_path_s "${MASC_STUDENT_CKPT_DIR}" \
  --model_path_t "${MASC_TEACHER_CKPT_DIR}" \
  --gpu_num "${GPU}"

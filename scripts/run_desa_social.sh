#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/sp/bin/python}"
SPLIT_ROOT="${1:-external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10}"
IPC="${IPC:-10}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
"${PYTHON_BIN}" -m baseline_adapters.run_desa_social \
  --metadata "${SPLIT_ROOT}/metadata/split_metadata.json" \
  --data-dir ./data \
  --ipc "${IPC}" \
  --gpu 0 \
  --output "external_baselines/outputs/desa_social/ipc${IPC}_results.csv" \
  --anchor-output "external_baselines/outputs/desa_social/ipc${IPC}_anchors.pt"

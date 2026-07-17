#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/sp/bin/python}"
SPLIT_ROOT="${1:-external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-100}"

"${PYTHON_BIN}" -m baseline_adapters.run_fedproto_social \
  --metadata "${SPLIT_ROOT}/metadata/split_metadata.json" \
  --data-dir ./data \
  --rounds "${ROUNDS}" \
  --gpu "${GPU}" \
  --output "external_baselines/outputs/fedproto_social/$(basename "$(dirname "${SPLIT_ROOT}")")_$(basename "${SPLIT_ROOT}").csv"

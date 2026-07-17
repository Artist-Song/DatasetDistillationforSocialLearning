#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/sp/bin/python}"
SPLIT_ROOT="${1:-external_baselines/outputs/splits/main_cifar100_hetero6arch_ipc10_seed0/ipc10}"
DEVICE_ID="${DEVICE_ID:-0}"
ROUNDS="${ROUNDS:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_FOLDER_NAME="${SAVE_FOLDER_NAME:-${ROOT_DIR}/external_baselines/outputs/fedre_social/model_store}"

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
"${PYTHON_BIN}" -m baseline_adapters.prepare_fedre_dataset \
  --exported-cifar100 "${SPLIT_ROOT}/fedre/Cifar100" \
  --replace-symlink

cd external_baselines/repos/FedRE/HtFLlib/system
"${PYTHON_BIN}" main.py \
  -data Cifar100 \
  -nb 100 \
  -nc 4 \
  -algo FedRE \
  -gr "${ROUNDS}" \
  -dev cuda \
  -did 0 \
  -jr 1.0 \
  -lbs 32 \
  -lr 0.06 \
  -sfn "${SAVE_FOLDER_NAME}" \
  -go social_cifar100_4agent_25cls

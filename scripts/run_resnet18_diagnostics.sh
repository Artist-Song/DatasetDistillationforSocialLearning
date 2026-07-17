#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/sp/bin/python}"

cd "$ROOT"
"$PY" scripts/prepare_resnet18_diagnostics.py

STD_CFG="configs/diagnostic_resnet18_standard_seed0_ipc50_fast.yaml"
CMP_CFG="configs/diagnostic_resnet18_compact_seed0_ipc50_fast.yaml"

"$PY" run_social_pipeline.py --config "$STD_CFG" --stage train_experts --only-agent 2

for cfg in "$CMP_CFG" "$STD_CFG"; do
  "$PY" run_social_pipeline.py --config "$cfg" --stage build_communication --packet-method fast
  "$PY" validate_packets.py --config "$cfg" --packet-method fast
  "$PY" run_social_pipeline.py \
    --config "$cfg" \
    --stage train_receivers \
    --packet-method fast \
    --only-receiver 2 \
    --init-mode expert
done

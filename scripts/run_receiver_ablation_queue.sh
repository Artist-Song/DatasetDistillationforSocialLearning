#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/sp/bin/python
mkdir -p logs/receiver_ablation
CONFIGS=(
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr010_kd050_ep500.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr020_kd050_ep500.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr050_kd050_ep500.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr010_kd025_ep500.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr020_kd025_ep500.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr010_kd050_ep200.yaml
  configs/receiver_ablation/main_cifar100_conv_family_ablation_fr020_kd050_ep200.yaml
)
MAX_JOBS=2
pids=()

run_one() {
  local cfg="$1"
  local name
  name=$(basename "$cfg" .yaml)
  local log="logs/receiver_ablation/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%F_%T)] START ${name} log=${log}"
  set +e
  "$PY" scripts/run_receiver_ablation.py --config "$cfg" --packet-method dsdm --init-mode expert > "$log" 2>&1
  local status=$?
  set -e
  echo "[$(date +%F_%T)] DONE ${name} status=${status}"
  return "$status"
}

compact_pids() {
  local new=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new+=("$pid")
    else
      wait "$pid" || true
    fi
  done
  pids=("${new[@]}")
}

for cfg in "${CONFIGS[@]}"; do
  run_one "$cfg" &
  pids+=("$!")
  while [ "${#pids[@]}" -ge "$MAX_JOBS" ]; do
    compact_pids
    if [ "${#pids[@]}" -ge "$MAX_JOBS" ]; then
      sleep 20
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done
"$PY" scripts/update_experiment_registry.py
echo "[$(date +%F_%T)] ALL RECEIVER ABLATIONS DONE"

#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/sp/bin/python
mkdir -p logs/receiver_tuning
CONFIGS=(
  configs/receiver_tuning/main_cifar100_conv_family_tuning_A_ep100_fr020_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_B_ep150_fr020_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_C_ep250_fr020_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_D_ep300_fr020_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_E_ep200_fr015_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_F_ep200_fr025_kd050.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_G_ep200_fr020_kd060.yaml
  configs/receiver_tuning/main_cifar100_conv_family_tuning_H_ep200_fr020_kd075.yaml
)
MAX_JOBS=2
pids=()

run_one() {
  local cfg="$1"
  local name
  name=$(basename "$cfg" .yaml)
  local log="logs/receiver_tuning/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%F_%T)] START ${name} log=${log}"
  set +e
  "$PY" scripts/run_receiver_ablation.py --config "$cfg" --packet-method dsdm --init-mode expert > "$log" 2>&1
  local status=$?
  set -e
  echo "[$(date +%F_%T)] DONE ${name} status=${status}"
  return "$status"
}

compact() {
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
    compact
    if [ "${#pids[@]}" -ge "$MAX_JOBS" ]; then
      sleep 20
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done
"$PY" scripts/update_experiment_registry.py
echo "[$(date +%F_%T)] ALL RECEIVER TUNING DONE"

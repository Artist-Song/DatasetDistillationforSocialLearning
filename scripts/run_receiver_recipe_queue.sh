#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/sp/bin/python
mkdir -p logs/receiver_recipe
TASK_CONFIGS=(
  configs/receiver_recipe/main_cifar100_conv_family_recipe_dsdm_logit_fr020_ep200.yaml
  configs/receiver_recipe/main_cifar100_conv_family_recipe_dsdm_image_fr020_ep200.yaml
  configs/receiver_recipe/main_cifar100_conv_family_recipe_heuristic_image_fr020_ep200.yaml
)
TASK_METHODS=(
  dsdm
  dsdm
  heuristic
)
MAX_JOBS=2
pids=()

run_one() {
  local cfg="$1"
  local method="$2"
  local name
  name=$(basename "$cfg" .yaml)
  local log="logs/receiver_recipe/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%F_%T)] START ${name} method=${method} log=${log}"
  set +e
  "$PY" scripts/run_receiver_ablation.py --config "$cfg" --packet-method "$method" --init-mode expert > "$log" 2>&1
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

for idx in "${!TASK_CONFIGS[@]}"; do
  run_one "${TASK_CONFIGS[$idx]}" "${TASK_METHODS[$idx]}" &
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
echo "[$(date +%F_%T)] ALL RECEIVER RECIPE DONE"

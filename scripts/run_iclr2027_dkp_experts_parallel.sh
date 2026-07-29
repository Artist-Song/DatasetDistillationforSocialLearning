#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
RUN_NAME="cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
LOG_ROOT="logs/iclr2027_dkp_first_round_seed0_v1/experts"

cd "$ROOT"

if [[ -d "outputs/$RUN_NAME" ]]; then
  echo "Refusing to reuse existing expert run: outputs/$RUN_NAME" >&2
  exit 1
fi

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Refusing to launch while GPU compute processes are active: ${gpu_pids[*]}" >&2
  exit 1
fi

if [[ -e "$LOG_ROOT/status.tsv" ]]; then
  echo "Refusing to overwrite existing launcher status: $LOG_ROOT/status.tsv" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf 'agent\tpid\texit_code\n' > "$LOG_ROOT/status.tsv"

pids=()
for agent_id in 0 1 2 3 4; do
  log_path="$LOG_ROOT/agent_${agent_id}.log"
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
    "$PYTHON_BIN" scripts/train_pool_communication_experts.py \
      --config "$CONFIG" --only-agent "$agent_id" > "$log_path" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf '%s\t%s\n' "$agent_id" "$pid" >> "$LOG_ROOT/pids.tsv"
  echo "[experts] launched agent=$agent_id pid=$pid log=$log_path"
done

failed=0
for agent_id in 0 1 2 3 4; do
  pid="${pids[$agent_id]}"
  if wait "$pid"; then
    exit_code=0
  else
    exit_code=$?
    failed=1
  fi
  printf '%s\t%s\t%s\n' "$agent_id" "$pid" "$exit_code" >> "$LOG_ROOT/status.tsv"
  echo "[experts] completed agent=$agent_id pid=$pid exit=$exit_code"
done

if (( failed != 0 )); then
  echo "One or more expert jobs failed; communication remains gated." >&2
  exit 1
fi

echo "[experts] all five agents completed"

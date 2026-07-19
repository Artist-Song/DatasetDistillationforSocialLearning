#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
PY="/root/miniconda3/envs/sp/bin/python"
LOG_DIR="${BASE_DIR}/logs/tinyimagenet_backbone_validation"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SEED="${SEED:-0}"
RUN_PREFIX="${RUN_PREFIX:-tinyimagenet_backbone_validation}"
MODEL_LIST="${MODELS:-convnet4 resnet18 alexnet mobilenetv2}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -n "$WAIT_FOR_PID" ]]; then
  if [[ ! "$WAIT_FOR_PID" =~ ^[0-9]+$ ]]; then
    echo "WAIT_FOR_PID must be numeric: ${WAIT_FOR_PID}" >&2
    exit 2
  fi
  echo "$(date -Is) queued models=${MODEL_LIST} waiting_for_pid=${WAIT_FOR_PID}"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 15
  done
  echo "$(date -Is) prerequisite_finished pid=${WAIT_FOR_PID}"
fi

pids=()
read -r -a models <<< "$MODEL_LIST"
if [[ "${#models[@]}" -eq 0 ]]; then
  echo "MODELS must contain at least one model" >&2
  exit 2
fi
for model in "${models[@]}"; do
  run_name="${RUN_PREFIX}_${model}_seed${SEED}"
  cmd=(
    "$PY" scripts/run_tiny_backbone_validation.py
    --model "$model"
    --run-name "$run_name"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --workers "$WORKERS"
    --seed "$SEED"
    --resume
  )
  if [[ "$MAX_TRAIN_BATCHES" -gt 0 ]]; then
    cmd+=(--max-train-batches "$MAX_TRAIN_BATCHES")
  fi
  if [[ "$MAX_VAL_BATCHES" -gt 0 ]]; then
    cmd+=(--max-val-batches "$MAX_VAL_BATCHES")
  fi
  "${cmd[@]}" >"${LOG_DIR}/${run_name}.log" 2>&1 &
  pids+=("$!")
  echo "$(date -Is) started model=${model} pid=${pids[-1]} run=${run_name}"
done

printf '%s\n' "${pids[@]}" >"${LOG_DIR}/active_pids.txt"
status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "$(date -Is) completed model=${models[$index]} pid=${pids[$index]}"
  else
    code=$?
    echo "$(date -Is) failed model=${models[$index]} pid=${pids[$index]} exit=${code}"
    status=$code
  fi
done
rm -f "${LOG_DIR}/active_pids.txt"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_linear_experts_seed0_v1.yaml"
CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1.yaml"
FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
EXPERT_RUN="cifar100_5agent20cls_dkp_linear_experts_seed0_v1"
CE_RUN="cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1"
FULL_RUN="cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1"
LOG_ROOT="logs/iclr2027_dkp_linear_head_seed0_v1"
MIN_FREE_KIB=$((2 * 1024 * 1024))

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[linear-head] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[linear-head] done  $name"
}

run_receiver_group() {
  local variant="$1"
  local config="$2"
  shift 2
  local receiver_ids=("$@")
  local pids=()
  local receiver_id
  echo "[linear-head] launch ${variant} receivers: ${receiver_ids[*]}"
  for receiver_id in "${receiver_ids[@]}"; do
    local log_path="$LOG_ROOT/${variant}_receiver_${receiver_id}.log"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" run_social_pipeline.py \
        --config "$config" --stage train_receivers --packet-method dsdm \
        --init-mode expert --only-receiver "$receiver_id" > "$log_path" 2>&1 &
    local pid=$!
    pids+=("$pid")
    printf '%s\t%s\t%s\n' "$variant" "$receiver_id" "$pid" >> "$LOG_ROOT/receiver_pids.tsv"
  done

  local failed=0
  local index
  for index in "${!receiver_ids[@]}"; do
    receiver_id="${receiver_ids[$index]}"
    local pid="${pids[$index]}"
    local exit_code
    if wait "$pid"; then
      exit_code=0
    else
      exit_code=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\n' "$variant" "$receiver_id" "$pid" "$exit_code" \
      >> "$LOG_ROOT/receiver_status.tsv"
    echo "[linear-head] receiver variant=$variant id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "Matched-linear receiver group failed; downstream stages remain gated." >&2
    return 1
  fi
}

available_kib="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free disk space for $ROOT" >&2
  exit 1
fi
if (( available_kib < MIN_FREE_KIB )); then
  echo "Matched-linear launch requires at least 2 GiB free; available=${available_kib} KiB" >&2
  exit 1
fi
echo "[linear-head] disk gate passed: available=${available_kib} KiB"

env OMP_NUM_THREADS=1 "$PYTHON_BIN" scripts/prepare_iclr2027_linear_head_ablation.py

for run_name in "$EXPERT_RUN" "$CE_RUN" "$FULL_RUN"; do
  if [[ -e "outputs/$run_name" ]]; then
    echo "Refusing to reuse matched-linear output: outputs/$run_name" >&2
    exit 1
  fi
done
if [[ -e "$LOG_ROOT" ]]; then
  echo "Refusing to reuse matched-linear log root: $LOG_ROOT" >&2
  exit 1
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Matched-linear launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf 'agent\tpid\texit_code\n' > "$LOG_ROOT/expert_status.tsv"
printf 'variant\treceiver\tpid\n' > "$LOG_ROOT/receiver_pids.tsv"
printf 'variant\treceiver\tpid\texit_code\n' > "$LOG_ROOT/receiver_status.tsv"

run_logged py_compile env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m py_compile \
  run_social_pipeline.py agent_data.py config_adapter.py agent_trainer.py \
  packet_integrity.py fullclass_pool_packets.py packet_logits.py packet_consumer.py \
  social_trainer.py social_metrics.py social_output_manager.py validate_packets.py \
  scripts/prepare_iclr2027_linear_head_ablation.py \
  scripts/validate_iclr2027_linear_experts.py \
  scripts/validate_iclr2027_linear_communication.py \
  scripts/summarize_iclr2027_linear_head_ablation.py \
  scripts/audit_iclr2027_linear_final_receivers.py
run_logged tests env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m unittest \
  tests.test_cosine_expert_protocol tests.test_dkp_receiver_protocol \
  tests.test_iclr2027_linear_head_ablation \
  tests.test_iclr2027_linear_final_receiver_audit tests.test_dkp_packet_integrity \
  tests.test_social_output_concurrency -v
run_logged docs "$PYTHON_BIN" scripts/validate_project_docs.py

expert_pids=()
for agent_id in 0 1 2 3 4; do
  log_path="$LOG_ROOT/expert_agent_${agent_id}.log"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
    "$PYTHON_BIN" scripts/train_pool_communication_experts.py \
      --config "$EXPERT_CONFIG" --only-agent "$agent_id" > "$log_path" 2>&1 &
  pid=$!
  expert_pids+=("$pid")
  printf '%s\t%s\n' "$agent_id" "$pid" >> "$LOG_ROOT/expert_pids.tsv"
  echo "[linear-head] launched expert agent=$agent_id pid=$pid"
done

expert_failed=0
for agent_id in 0 1 2 3 4; do
  pid="${expert_pids[$agent_id]}"
  if wait "$pid"; then
    exit_code=0
  else
    exit_code=$?
    expert_failed=1
  fi
  printf '%s\t%s\t%s\n' "$agent_id" "$pid" "$exit_code" >> "$LOG_ROOT/expert_status.tsv"
done
if (( expert_failed != 0 )); then
  echo "One or more matched-linear experts failed; communication remains gated." >&2
  exit 1
fi

run_logged expert_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_linear_experts.py \
    --config "$EXPERT_CONFIG" --device cuda --min-local-test-accuracy 20 \
    --report-json "outputs/$EXPERT_RUN/metrics/linear_expert_preflight.json"

run_logged reuse_ce "$PYTHON_BIN" scripts/reuse_iclr2027_experts.py \
  --source-config "$EXPERT_CONFIG" --target-config "$CE_CONFIG"
run_logged reuse_full "$PYTHON_BIN" scripts/reuse_iclr2027_experts.py \
  --source-config "$EXPERT_CONFIG" --target-config "$FULL_CONFIG"

run_logged materialize_ce env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  scripts/materialize_fullclass_pool_packets.py --config "$CE_CONFIG" --pool-catalog "$CATALOG"
run_logged build_ce env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  run_social_pipeline.py --config "$CE_CONFIG" --stage build_communication --packet-method dsdm
run_logged validate_ce env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  validate_packets.py --config "$CE_CONFIG" --packet-method dsdm

run_logged materialize_full env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  scripts/materialize_fullclass_pool_packets.py --config "$FULL_CONFIG" --pool-catalog "$CATALOG"
run_logged attach_linear_logits env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" run_social_pipeline.py --config "$FULL_CONFIG" --stage attach_logits --packet-method dsdm
run_logged build_full env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  run_social_pipeline.py --config "$FULL_CONFIG" --stage build_communication --packet-method dsdm
run_logged validate_full env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  validate_packets.py --config "$FULL_CONFIG" --packet-method dsdm

run_logged communication_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_linear_communication.py \
    --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
    --device cuda --report-json "outputs/$FULL_RUN/metrics/linear_communication_preflight.json"

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "Matched-linear receivers blocked by active GPU PIDs: ${receiver_gate_pids[*]}" >&2
  exit 1
fi

run_receiver_group ce_only "$CE_CONFIG" 0 1 2
run_receiver_group ce_only "$CE_CONFIG" 3 4
run_receiver_group full "$FULL_CONFIG" 0 1 2
run_receiver_group full "$FULL_CONFIG" 3
run_receiver_group full "$FULL_CONFIG" 4

run_logged summarize env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_linear_head_ablation.py \
  --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
  --output-json "outputs/$FULL_RUN/metrics/linear_head_seed0_summary.json"
run_logged final_receiver_audit env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/audit_iclr2027_linear_final_receivers.py \
    --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
    --summary-json "outputs/$FULL_RUN/metrics/linear_head_seed0_summary.json" \
    --output-json "outputs/$FULL_RUN/metrics/linear_final_receiver_checkpoint_audit.json"

echo "[linear-head] complete: five linear experts, CE-only and full DKP-SL, five receivers each"

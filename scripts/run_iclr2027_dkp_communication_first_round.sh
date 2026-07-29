#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
EXPERT_RUN="cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
CE_RUN="cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2"
FULL_RUN="cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2"
LOG_ROOT="logs/iclr2027_dkp_first_round_seed0_v1/communication"
EXPERT_STATUS="logs/iclr2027_dkp_first_round_seed0_v1/experts/status.tsv"

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[first-round] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[first-round] done  $name"
}

run_receiver_group() {
  local variant="$1"
  local config="$2"
  shift 2
  local receiver_ids=("$@")
  local pids=()
  local receiver_id
  echo "[first-round] launch ${variant} receivers: ${receiver_ids[*]}"
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
    echo "[first-round] receiver variant=$variant id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "Receiver group failed; downstream stages remain gated." >&2
    return 1
  fi
}

if [[ ! -f "$EXPERT_STATUS" ]]; then
  echo "Expert launcher status is missing: $EXPERT_STATUS" >&2
  exit 1
fi
completed_experts="$(awk 'NR > 1 && $3 == 0 {count++} END {print count + 0}' "$EXPERT_STATUS")"
if [[ "$completed_experts" != "5" ]]; then
  echo "Expert gate requires five zero exits, got $completed_experts" >&2
  exit 1
fi
for run_name in "$CE_RUN" "$FULL_RUN"; do
  if [[ -e "outputs/$run_name" ]]; then
    echo "Refusing to reuse an existing derived run: outputs/$run_name" >&2
    exit 1
  fi
done
if [[ ! -d "outputs/$EXPERT_RUN" ]]; then
  echo "Expert run is missing: outputs/$EXPERT_RUN" >&2
  exit 1
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Communication launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi
if [[ -e "$LOG_ROOT/receiver_status.tsv" ]]; then
  echo "Refusing to overwrite prior communication launcher state: $LOG_ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf 'variant\treceiver\tpid\n' > "$LOG_ROOT/receiver_pids.tsv"
printf 'variant\treceiver\tpid\texit_code\n' > "$LOG_ROOT/receiver_status.tsv"

run_logged py_compile env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m py_compile \
  run_social_pipeline.py agent_data.py config_adapter.py agent_trainer.py \
  packet_integrity.py fullclass_pool_packets.py packet_logits.py packet_consumer.py \
  social_trainer.py social_metrics.py social_output_manager.py validate_packets.py \
  scripts/validate_iclr2027_cosine_experts.py \
  scripts/validate_iclr2027_dkp_communication.py \
  scripts/summarize_iclr2027_dkp_first_round.py
run_logged tests_core env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m unittest \
  tests.test_dsdm_evaluation_schedule tests.test_resnet_variants \
  tests.test_atomic_artifact_writes tests.test_fullclass_pool_packets \
  tests.test_pat_class_split_configs tests.test_teacher_quality_protocol \
  tests.test_packet_validation_active_classes -v
run_logged tests_cosine env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m unittest \
  tests.test_cosine_expert_protocol tests.test_iclr2027_cosine_expert_preflight -v
run_logged tests_packet env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m unittest \
  tests.test_dkp_packet_integrity -v
run_logged tests_receiver env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" -m unittest \
  tests.test_dkp_receiver_protocol tests.test_iclr2027_dkp_communication_preflight \
  tests.test_iclr2027_first_round_summary -v
run_logged docs "$PYTHON_BIN" scripts/validate_project_docs.py

run_logged expert_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  scripts/validate_iclr2027_cosine_experts.py \
  --config "$EXPERT_CONFIG" --device cuda --min-local-test-accuracy 20 \
  --report-json "outputs/$EXPERT_RUN/metrics/cosine_expert_preflight.json"

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
run_logged attach_full_logits env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" run_social_pipeline.py --config "$FULL_CONFIG" --stage attach_logits --packet-method dsdm
run_logged build_full env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  run_social_pipeline.py --config "$FULL_CONFIG" --stage build_communication --packet-method dsdm
run_logged validate_full env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  validate_packets.py --config "$FULL_CONFIG" --packet-method dsdm

run_logged communication_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_communication.py \
  --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
  --device cuda --report-json "outputs/$FULL_RUN/metrics/communication_preflight.json"

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "Receiver launch blocked by active GPU PIDs after preflight: ${receiver_gate_pids[*]}" >&2
  exit 1
fi

run_receiver_group ce_only "$CE_CONFIG" 0 1 2
run_receiver_group ce_only "$CE_CONFIG" 3 4
run_receiver_group full "$FULL_CONFIG" 0 1 2
run_receiver_group full "$FULL_CONFIG" 3
run_receiver_group full "$FULL_CONFIG" 4

run_logged summarize "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_first_round.py \
  --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
  --output-json "outputs/$FULL_RUN/metrics/first_round_seed0_summary.json"

echo "[first-round] complete: experts + CE-only + full DKP-SL, five receivers each"

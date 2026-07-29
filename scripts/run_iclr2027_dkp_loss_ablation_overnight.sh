#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
EXPERT_RUN="cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
LOG_ROOT="logs/iclr2027_dkp_loss_ablation_seed0_v1"
SUMMARY_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_v1.json"
AUDIT_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_final_checkpoint_audit_v1.json"

CONDITIONS=(
  fr0_kd0_sc1
  fr1_kd0_sc0
  fr0_kd1_sc0
  fr1_kd0_sc1
  fr1_kd1_sc0
  fr0_kd1_sc1
)
CONFIGS=(
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd0_sc1_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc0_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc0_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd1_sc0_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc1_ipc10_seed0_v1.yaml
)
RUNS=(
  cifar100_5agent20cls_dkp_ablation_fr0_kd0_sc1_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc0_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc0_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_ablation_fr1_kd1_sc0_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc1_ipc10_seed0_v1
)
KD_SWITCHES=(0 0 1 0 1 1)

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[loss-ablation] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[loss-ablation] done  $name"
}

run_receiver_group() {
  local condition="$1"
  local config="$2"
  shift 2
  local receiver_ids=("$@")
  local pids=()
  local receiver_id
  echo "[loss-ablation] launch condition=$condition receivers=${receiver_ids[*]}"
  for receiver_id in "${receiver_ids[@]}"; do
    local log_path="$LOG_ROOT/${condition}_receiver_${receiver_id}.log"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" run_social_pipeline.py \
        --config "$config" --stage train_receivers --packet-method dsdm \
        --init-mode expert --only-receiver "$receiver_id" > "$log_path" 2>&1 &
    local pid=$!
    pids+=("$pid")
    printf '%s\t%s\t%s\n' "$condition" "$receiver_id" "$pid" >> "$LOG_ROOT/receiver_pids.tsv"
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
    printf '%s\t%s\t%s\t%s\n' "$condition" "$receiver_id" "$pid" "$exit_code" \
      >> "$LOG_ROOT/receiver_status.tsv"
    echo "[loss-ablation] receiver condition=$condition id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "Receiver group failed; downstream conditions remain gated." >&2
    return 1
  fi
}

if (( ${#CONDITIONS[@]} != 6 || ${#CONFIGS[@]} != 6 || ${#RUNS[@]} != 6 || ${#KD_SWITCHES[@]} != 6 )); then
  echo "Loss-ablation launcher matrix is internally inconsistent." >&2
  exit 1
fi
if [[ ! -d "outputs/$EXPERT_RUN" ]]; then
  echo "Cosine expert run is missing: outputs/$EXPERT_RUN" >&2
  exit 1
fi
for source_config in "$EXPERT_CONFIG" "$CE_CONFIG" "$FULL_CONFIG" "$CATALOG"; do
  if [[ ! -f "$source_config" ]]; then
    echo "Required immutable source is missing: $source_config" >&2
    exit 1
  fi
done
for run_name in "${RUNS[@]}"; do
  if [[ -e "outputs/$run_name" ]]; then
    echo "Refusing to reuse an existing loss-ablation output: outputs/$run_name" >&2
    exit 1
  fi
done
if [[ -e "$SUMMARY_OUTPUT" ]]; then
  echo "Refusing to overwrite an existing diagnostic summary: $SUMMARY_OUTPUT" >&2
  exit 1
fi
if [[ -e "$AUDIT_OUTPUT" ]]; then
  echo "Refusing to overwrite an existing final-checkpoint audit: $AUDIT_OUTPUT" >&2
  exit 1
fi
if [[ -e "$LOG_ROOT" ]]; then
  echo "Refusing to overwrite prior loss-ablation launcher state: $LOG_ROOT" >&2
  exit 1
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Loss-ablation launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi
available_kb="$(df --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')"
if (( available_kb < 2097152 )); then
  echo "Loss-ablation launch requires at least 2 GiB free, found ${available_kb} KiB." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf 'condition\treceiver\tpid\n' > "$LOG_ROOT/receiver_pids.tsv"
printf 'condition\treceiver\tpid\texit_code\n' > "$LOG_ROOT/receiver_status.tsv"

run_logged config_check env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/prepare_iclr2027_dkp_loss_ablation.py --check
run_logged py_compile env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m py_compile \
  run_social_pipeline.py config_adapter.py social_trainer.py social_output_manager.py \
  scripts/prepare_iclr2027_dkp_loss_ablation.py \
  scripts/summarize_iclr2027_dkp_loss_ablation.py \
  scripts/audit_iclr2027_dkp_loss_ablation_final_receivers.py
run_logged tests env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest \
  tests.test_cosine_expert_protocol \
  tests.test_dkp_receiver_protocol \
  tests.test_dkp_packet_integrity \
  tests.test_iclr2027_dkp_loss_ablation \
  tests.test_iclr2027_dkp_loss_ablation_final_receiver_audit -v
run_logged expert_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/validate_iclr2027_cosine_experts.py \
  --config "$EXPERT_CONFIG" --device cuda --min-local-test-accuracy 20 \
  --report-json "$LOG_ROOT/cosine_expert_preflight.json"
run_logged source_communication_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_communication.py \
  --expert-config "$EXPERT_CONFIG" --ce-config "$CE_CONFIG" --full-config "$FULL_CONFIG" \
  --device cuda --report-json "$LOG_ROOT/source_communication_preflight.json"

for index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$index]}"
  config="${CONFIGS[$index]}"
  kd="${KD_SWITCHES[$index]}"
  run_logged "${condition}_reuse_experts" "$PYTHON_BIN" scripts/reuse_iclr2027_experts.py \
    --source-config "$EXPERT_CONFIG" --target-config "$config"
  run_logged "${condition}_materialize" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" scripts/materialize_fullclass_pool_packets.py \
    --config "$config" --pool-catalog "$CATALOG"
  if [[ "$kd" == "1" ]]; then
    run_logged "${condition}_attach_logits" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" run_social_pipeline.py \
      --config "$config" --stage attach_logits --packet-method dsdm
  fi
  run_logged "${condition}_build_communication" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" run_social_pipeline.py \
    --config "$config" --stage build_communication --packet-method dsdm
  run_logged "${condition}_validate_packets" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" validate_packets.py --config "$config" --packet-method dsdm
done

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "Receiver launch blocked by active GPU PIDs after packet preparation: ${receiver_gate_pids[*]}" >&2
  exit 1
fi

for index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$index]}"
  config="${CONFIGS[$index]}"
  run_receiver_group "$condition" "$config" 0 1 2
  run_receiver_group "$condition" "$config" 3 4
done

run_logged summarize env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_loss_ablation.py \
  --expert-config "$EXPERT_CONFIG" --config-dir configs/iclr2027 \
  --output-json "$SUMMARY_OUTPUT"
run_logged final_checkpoint_audit env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/audit_iclr2027_dkp_loss_ablation_final_receivers.py \
  --expert-config "$EXPERT_CONFIG" --config-dir configs/iclr2027 \
  --summary-json "$SUMMARY_OUTPUT" --output-json "$AUDIT_OUTPUT"

echo "[loss-ablation] complete: six missing conditions x five receivers; summary and 30-checkpoint audit passed"

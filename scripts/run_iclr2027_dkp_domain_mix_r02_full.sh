#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
SOURCE_CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
SOURCE_FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
RUN_NAME="cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2"
LOG_ROOT="logs/iclr2027_dkp_domain_mix_r02_full_seed0_v2"
PREFLIGHT_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_preflight_v2.json"
SUMMARY_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_v2.json"
AUDIT_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_final_checkpoint_audit_v2.json"
CHECK_ONLY=0

if [[ ${1:-} == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[domain-mix-r02-full] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[domain-mix-r02-full] done  $name"
}

run_receivers() {
  local phase="$1"
  shift
  local receiver_ids=("$@")
  local pids=()
  local receiver_id pid exit_code
  echo "[domain-mix-r02-full] launch phase=$phase receivers=${receiver_ids[*]}"
  for receiver_id in "${receiver_ids[@]}"; do
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON_BIN" run_social_pipeline.py \
        --config "$CONFIG" --stage train_receivers --packet-method dsdm \
        --init-mode expert --only-receiver "$receiver_id" \
        > "$LOG_ROOT/receiver_${receiver_id}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    printf '%s\t%s\t%s\n' "$phase" "$receiver_id" "$pid" >> "$LOG_ROOT/receiver_pids.tsv"
  done
  local failed=0 index
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    receiver_id="${receiver_ids[$index]}"
    if wait "$pid"; then
      exit_code=0
    else
      exit_code=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\n' "$phase" "$receiver_id" "$pid" "$exit_code" >> "$LOG_ROOT/receiver_status.tsv"
    echo "[domain-mix-r02-full] receiver phase=$phase id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "r02 full receiver phase failed; downstream phases remain gated." >&2
    return 1
  fi
}

for source in "$EXPERT_CONFIG" "$SOURCE_CE_CONFIG" "$SOURCE_FULL_CONFIG" "$CONFIG" "$CATALOG"; do
  if [[ ! -f "$source" ]]; then
    echo "Required source is missing: $source" >&2
    exit 1
  fi
done
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/prepare_iclr2027_dkp_domain_mix_r02_full.py --check
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m py_compile \
    config_adapter.py packet_consumer.py packet_logits.py social_trainer.py social_output_manager.py \
    scripts/prepare_iclr2027_dkp_domain_mix_r02_full.py \
    scripts/validate_iclr2027_dkp_domain_mix_r02_full.py \
    scripts/summarize_iclr2027_dkp_domain_mix_r02_full.py \
    scripts/audit_iclr2027_dkp_domain_receivers.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest \
    tests.test_cosine_expert_protocol \
    tests.test_dkp_receiver_config \
    tests.test_dkp_receiver_protocol \
    tests.test_dkp_packet_integrity \
    tests.test_iclr2027_dkp_domain_mix_r02_full \
    tests.test_social_output_concurrency -v

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "r02 full launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi
available_kb="$(df --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')"
if (( available_kb < 1258291 )); then
  echo "r02 full launch requires at least 1.2 GiB free, found ${available_kb} KiB." >&2
  exit 1
fi
if [[ -e "outputs/$RUN_NAME" ]]; then
  echo "Refusing to reuse or overwrite output: outputs/$RUN_NAME" >&2
  exit 1
fi
for target in "$LOG_ROOT" "$PREFLIGHT_OUTPUT" "$SUMMARY_OUTPUT" "$AUDIT_OUTPUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite launcher artifact: $target" >&2
    exit 1
  fi
done
if (( CHECK_ONLY == 1 )); then
  echo "[domain-mix-r02-full] check-only passed; no artifact or experiment process was started"
  exit 0
fi

mkdir -p "$LOG_ROOT"
printf 'phase\treceiver\tpid\n' > "$LOG_ROOT/receiver_pids.tsv"
printf 'phase\treceiver\tpid\texit_code\n' > "$LOG_ROOT/receiver_status.tsv"

run_logged expert_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_cosine_experts.py \
    --config "$EXPERT_CONFIG" --device cuda --min-local-test-accuracy 20 \
    --report-json "$LOG_ROOT/cosine_expert_preflight.json"
run_logged source_communication_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_communication.py \
    --expert-config "$EXPERT_CONFIG" --ce-config "$SOURCE_CE_CONFIG" --full-config "$SOURCE_FULL_CONFIG" \
    --device cuda --report-json "$LOG_ROOT/source_communication_preflight.json"
run_logged reuse_experts "$PYTHON_BIN" scripts/reuse_iclr2027_experts.py \
  --source-config "$EXPERT_CONFIG" --target-config "$CONFIG"
run_logged materialize env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/materialize_fullclass_pool_packets.py --config "$CONFIG" --pool-catalog "$CATALOG"
run_logged attach_logits env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" run_social_pipeline.py --config "$CONFIG" --stage attach_logits --packet-method dsdm
run_logged build_communication env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" run_social_pipeline.py --config "$CONFIG" --stage build_communication --packet-method dsdm
run_logged validate_packets env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" validate_packets.py --config "$CONFIG" --packet-method dsdm
run_logged target_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_domain_mix_r02_full.py \
    --device cuda --report-json "$PREFLIGHT_OUTPUT"

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "r02 full receiver launch blocked by active GPU PIDs: ${receiver_gate_pids[*]}" >&2
  exit 1
fi
run_receivers conv_alex 0 1 2
run_receivers standard_resnet 3 4
run_logged final_summary env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_domain_mix_r02_full.py --output-json "$SUMMARY_OUTPUT"
run_logged final_checkpoint_audit env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/audit_iclr2027_dkp_domain_receivers.py \
    --summary-json "$SUMMARY_OUTPUT" --report-json "$AUDIT_OUTPUT"
echo "[domain-mix-r02-full] complete: KD-on r02 x five receivers, strict summary and audit passed"

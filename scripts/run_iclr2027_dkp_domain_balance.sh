#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
LOG_ROOT="logs/iclr2027_dkp_domain_balance_seed0_v1"
PREFLIGHT_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_balance_preflight_v1.json"
SUMMARY_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_balance_v1.json"
CHECK_ONLY=0

if [[ ${1:-} == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

CONDITIONS=(s_real u_packet h_real_packet)
CONFIGS=(
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_s_real_steps3780_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_u_packet_steps3780_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_h_real_packet_steps3780_ipc10_seed0_v1.yaml
)
RUNS=(
  cifar100_5agent20cls_dkp_domain_s_real_steps3780_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_domain_u_packet_steps3780_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_domain_h_real_packet_steps3780_ipc10_seed0_v1
)

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[domain-balance] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[domain-balance] done  $name"
}

run_receiver_matrix() {
  local phase="$1"
  shift
  local receiver_ids=("$@")
  local pids=()
  local labels=()
  local condition config receiver_id index pid exit_code
  echo "[domain-balance] launch phase=$phase receivers=${receiver_ids[*]}"
  for index in "${!CONDITIONS[@]}"; do
    condition="${CONDITIONS[$index]}"
    config="${CONFIGS[$index]}"
    for receiver_id in "${receiver_ids[@]}"; do
      env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
        "$PYTHON_BIN" run_social_pipeline.py \
          --config "$config" --stage train_receivers --packet-method dsdm \
          --init-mode expert --only-receiver "$receiver_id" \
          > "$LOG_ROOT/${condition}_receiver_${receiver_id}.log" 2>&1 &
      pid=$!
      pids+=("$pid")
      labels+=("${condition}:${receiver_id}")
      printf '%s\t%s\t%s\t%s\n' "$phase" "$condition" "$receiver_id" "$pid" \
        >> "$LOG_ROOT/receiver_pids.tsv"
    done
  done

  local failed=0
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    IFS=: read -r condition receiver_id <<< "${labels[$index]}"
    if wait "$pid"; then
      exit_code=0
    else
      exit_code=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$phase" "$condition" "$receiver_id" "$pid" "$exit_code" \
      >> "$LOG_ROOT/receiver_status.tsv"
    echo "[domain-balance] receiver phase=$phase condition=$condition id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "Receiver phase failed; all downstream phases remain gated." >&2
    return 1
  fi
}

if (( ${#CONDITIONS[@]} != 3 || ${#CONFIGS[@]} != 3 || ${#RUNS[@]} != 3 )); then
  echo "Domain-balance launcher matrix is internally inconsistent." >&2
  exit 1
fi
for source in "$EXPERT_CONFIG" "$CE_CONFIG" "$FULL_CONFIG" "$CATALOG"; do
  if [[ ! -f "$source" ]]; then
    echo "Required immutable source is missing: $source" >&2
    exit 1
  fi
done
for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Generated diagnostic config is missing: $config" >&2
    exit 1
  fi
done

env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/prepare_iclr2027_dkp_domain_balance.py --check
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m py_compile \
    run_social_pipeline.py config_adapter.py packet_consumer.py social_trainer.py \
    social_output_manager.py scripts/prepare_iclr2027_dkp_domain_balance.py \
    scripts/validate_iclr2027_dkp_domain_balance.py \
    scripts/summarize_iclr2027_dkp_domain_balance.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest \
    tests.test_cosine_expert_protocol \
    tests.test_dkp_receiver_config \
    tests.test_dkp_receiver_protocol \
    tests.test_dkp_packet_integrity \
    tests.test_iclr2027_dkp_communication_preflight \
    tests.test_social_output_concurrency -v
for config in "${CONFIGS[@]}"; do
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" scripts/materialize_fullclass_pool_packets.py \
      --config "$config" --pool-catalog "$CATALOG" --dry-run > /dev/null
done

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Domain-balance launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi
available_kb="$(df --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')"
if (( available_kb < 2097152 )); then
  echo "Domain-balance launch requires at least 2 GiB free, found ${available_kb} KiB." >&2
  exit 1
fi

if (( CHECK_ONLY == 1 )); then
  echo "[domain-balance] check-only passed; no artifact or experiment process was started"
  exit 0
fi

for index in "${!RUNS[@]}"; do
  run_name="${RUNS[$index]}"
  config="${CONFIGS[$index]}"
  run_dir="outputs/$run_name"
  if [[ ! -e "$run_dir" ]]; then
    continue
  fi
  if [[ ! -f "$run_dir/config/main.yaml" ]] || ! cmp -s "$config" "$run_dir/config/main.yaml"; then
    echo "Refusing to reuse existing output with a different config: $run_dir" >&2
    exit 1
  fi
  unexpected_file="$(find "$run_dir" -type f \
    ! -path "$run_dir/config/main.yaml" \
    ! -path "$run_dir/config/social_resolved_args.json" \
    ! -path "$run_dir/config/.snapshot.lock" -print -quit)"
  if [[ -n "$unexpected_file" ]]; then
    echo "Refusing to reuse non-pristine output; found: $unexpected_file" >&2
    exit 1
  fi
  echo "[domain-balance] accepting identical dry-run skeleton: $run_dir"
done
for target in "$LOG_ROOT" "$PREFLIGHT_OUTPUT" "$SUMMARY_OUTPUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite launcher artifact: $target" >&2
    exit 1
  fi
done

mkdir -p "$LOG_ROOT"
printf 'phase\tcondition\treceiver\tpid\n' > "$LOG_ROOT/receiver_pids.tsv"
printf 'phase\tcondition\treceiver\tpid\texit_code\n' > "$LOG_ROOT/receiver_status.tsv"

run_logged expert_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
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
  run_logged "${condition}_reuse_experts" \
    "$PYTHON_BIN" scripts/reuse_iclr2027_experts.py \
      --source-config "$EXPERT_CONFIG" --target-config "$config"
  run_logged "${condition}_materialize" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" scripts/materialize_fullclass_pool_packets.py \
      --config "$config" --pool-catalog "$CATALOG"
  run_logged "${condition}_build_communication" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" run_social_pipeline.py \
      --config "$config" --stage build_communication --packet-method dsdm
  run_logged "${condition}_validate_packets" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" validate_packets.py --config "$config" --packet-method dsdm
done

run_logged target_preflight env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_domain_balance.py \
    --config-dir configs/iclr2027 --device cuda --report-json "$PREFLIGHT_OUTPUT"

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "Receiver launch blocked by active GPU PIDs after preflight: ${receiver_gate_pids[*]}" >&2
  exit 1
fi

run_receiver_matrix conv_gate 0 1
run_logged conv_gate_summary env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_domain_balance.py \
    --config-dir configs/iclr2027 --receiver-ids 0,1 \
    --output-json "$LOG_ROOT/conv_gate_summary.json"

# All three conditions continue, so the Conv test metrics do not select or stop a method.
run_receiver_matrix alexnet_confirmation 2
run_receiver_matrix standard_resnet_confirmation 3 4

run_logged final_summary env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_domain_balance.py \
    --config-dir configs/iclr2027 --receiver-ids 0,1,2,3,4 \
    --output-json "$SUMMARY_OUTPUT"

echo "[domain-balance] complete: S/U/H x five receivers and strict summary passed"

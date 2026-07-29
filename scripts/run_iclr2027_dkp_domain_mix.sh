#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/sp/bin/python"
EXPERT_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
CE_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
FULL_CONFIG="configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
CATALOG="configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
LOG_ROOT="logs/iclr2027_dkp_domain_mix_seed0_v1"
PREFLIGHT_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_preflight_v1.json"
SUMMARY_OUTPUT="outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_v1.json"
CHECK_ONLY=0

if [[ ${1:-} == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

CONDITIONS=(r05 r10 r20 r30)
CONFIGS=(
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r05_steps3780_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r10_steps3780_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r20_steps3780_ipc10_seed0_v1.yaml
  configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r30_steps3780_ipc10_seed0_v1.yaml
)
RUNS=(
  cifar100_5agent20cls_dkp_domain_mix_r05_steps3780_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_domain_mix_r10_steps3780_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_domain_mix_r20_steps3780_ipc10_seed0_v1
  cifar100_5agent20cls_dkp_domain_mix_r30_steps3780_ipc10_seed0_v1
)

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "[domain-mix] start $name -> $log_path"
  "$@" > "$log_path" 2>&1
  echo "[domain-mix] done  $name"
}

run_condition_receivers() {
  local phase="$1"
  shift
  local condition_indexes="$1"
  shift
  local receiver_ids=("$@")
  local pids=()
  local labels=()
  local index condition config receiver_id pid exit_code
  echo "[domain-mix] launch phase=$phase condition_indexes=$condition_indexes receivers=${receiver_ids[*]}"
  for index in $condition_indexes; do
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
    echo "[domain-mix] receiver phase=$phase condition=$condition id=$receiver_id exit=$exit_code"
  done
  if (( failed != 0 )); then
    echo "Receiver phase failed; all downstream phases remain gated." >&2
    return 1
  fi
}

if (( ${#CONDITIONS[@]} != 4 || ${#CONFIGS[@]} != 4 || ${#RUNS[@]} != 4 )); then
  echo "Domain-mix launcher matrix is internally inconsistent." >&2
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
  "$PYTHON_BIN" scripts/prepare_iclr2027_dkp_domain_mix.py --check
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m py_compile \
    run_social_pipeline.py config_adapter.py packet_consumer.py social_trainer.py \
    social_output_manager.py scripts/prepare_iclr2027_dkp_domain_mix.py \
    scripts/validate_iclr2027_dkp_domain_mix.py \
    scripts/summarize_iclr2027_dkp_domain_mix.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest \
    tests.test_cosine_expert_protocol \
    tests.test_dkp_receiver_config \
    tests.test_dkp_receiver_protocol \
    tests.test_dkp_packet_integrity \
    tests.test_iclr2027_dkp_communication_preflight \
    tests.test_iclr2027_dkp_domain_mix \
    tests.test_social_output_concurrency -v

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "Domain-mix launch blocked by active GPU PIDs: ${gpu_pids[*]}" >&2
  exit 1
fi
available_kb="$(df --output=avail "$ROOT" | tail -n 1 | tr -d '[:space:]')"
if (( available_kb < 2097152 )); then
  echo "Domain-mix launch requires at least 2 GiB free, found ${available_kb} KiB." >&2
  exit 1
fi
for run_name in "${RUNS[@]}"; do
  if [[ -e "outputs/$run_name" ]]; then
    echo "Refusing to reuse or overwrite output: outputs/$run_name" >&2
    exit 1
  fi
done
for target in "$LOG_ROOT" "$PREFLIGHT_OUTPUT" "$SUMMARY_OUTPUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite launcher artifact: $target" >&2
    exit 1
  fi
done

if (( CHECK_ONLY == 1 )); then
  echo "[domain-mix] check-only passed; no artifact or experiment process was started"
  exit 0
fi

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
  "$PYTHON_BIN" scripts/validate_iclr2027_dkp_domain_mix.py \
    --config-dir configs/iclr2027 --device cuda --report-json "$PREFLIGHT_OUTPUT"

mapfile -t receiver_gate_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#receiver_gate_pids[@]} > 0 )); then
  echo "Receiver launch blocked by active GPU PIDs after preflight: ${receiver_gate_pids[*]}" >&2
  exit 1
fi

# At most four receiver processes run together. All conditions continue regardless of test metrics.
run_condition_receivers conv_low "0 1" 0 1
run_condition_receivers conv_high "2 3" 0 1
run_logged conv_integrity_summary env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_domain_mix.py \
    --config-dir configs/iclr2027 --receiver-ids 0,1 \
    --output-json "$LOG_ROOT/conv_integrity_summary.json"
run_condition_receivers alexnet_all "0 1 2 3" 2
run_condition_receivers resnet_low "0 1" 3 4
run_condition_receivers resnet_high "2 3" 3 4

run_logged final_summary env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" scripts/summarize_iclr2027_dkp_domain_mix.py \
    --config-dir configs/iclr2027 --receiver-ids 0,1,2,3,4 \
    --output-json "$SUMMARY_OUTPUT"

echo "[domain-mix] complete: four fractions x five receivers and strict summary passed"

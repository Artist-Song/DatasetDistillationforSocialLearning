#!/usr/bin/env bash
set -euo pipefail

# Run packet-only receiver fine-tuning ablations.
#
# This script assumes local_pretrain checkpoints and packet files already exist
# under outputs/260608-cifar10_direct. It does not rebuild packets.
#
# Usage:
#   bash scripts/run_packet_only_ablation.sh
#
# Useful overrides:
#   PYTHON=python3 bash scripts/run_packet_only_ablation.sh
#   FORCE=1 bash scripts/run_packet_only_ablation.sh

PYTHON_BIN="${PYTHON:-python}"
FORCE="${FORCE:-0}"
OUTPUT_ROOT="outputs/260608-cifar10_direct"
LOG_DIR="${OUTPUT_ROOT}/logs/packet_only_ablation"
mkdir -p "${LOG_DIR}"

BASE_RUN="cifar10_direct_conv"

now_text() {
  date "+%Y-%m-%d %H:%M:%S"
}

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02dh:%02dm:%02ds" "${hours}" "${minutes}" "${seconds}"
}

run_step() {
  local name="$1"
  local done_path="$2"
  shift 2

  local log_path="${LOG_DIR}/${name}.log"
  if [[ "${FORCE}" != "1" && -e "${done_path}" ]]; then
    echo "[skip] $(now_text) ${name}: found ${done_path}"
    return
  fi

  local start_ts
  start_ts="$(date +%s)"
  echo "[run] $(now_text) ${name}"
  echo "      log: ${log_path}"
  if "$@" 2>&1 | tee "${log_path}"; then
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[done] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
  else
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[fail] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
    return 1
  fi
}

run_experiment() {
  local label="$1"
  local config="$2"
  local run_name="$3"

  local ckpt_dir="${OUTPUT_ROOT}/checkpoints/packet_only_train/${run_name}"
  local eval_report="${OUTPUT_ROOT}/reports/eval/packet_only_train_${run_name}_all.json"
  local compare_report="${OUTPUT_ROOT}/reports/compare/compare_packet_only_train_${run_name}_all.json"

  run_step \
    "${label}_train" \
    "${ckpt_dir}/agent_4_packet_only.pt" \
    "${PYTHON_BIN}" -m src.main.run_packet_only_train --config "${config}"

  run_step \
    "${label}_eval" \
    "${eval_report}" \
    "${PYTHON_BIN}" -m src.main.run_eval --config "${config}" --checkpoint-stage packet_only_train

  run_step \
    "${label}_compare" \
    "${compare_report}" \
    "${PYTHON_BIN}" -m src.main.run_compare --config "${config}" --target-stage packet_only_train
}

echo "=== packet-only ablation ==="
echo "started_at: $(now_text)"
echo "python: ${PYTHON_BIN}"
echo "force: ${FORCE}"
echo "output_root: ${OUTPUT_ROOT}"
echo "log_dir: ${LOG_DIR}"

SCRIPT_START_TS="$(date +%s)"

run_experiment \
  "dsdm_plain" \
  "configs/exp/260608-cifar10_direct_dsdm.yaml" \
  "${BASE_RUN}_src-dsdm_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-full"

run_experiment \
  "dsdm_retain" \
  "configs/exp/260608-cifar10_direct_dsdm_packet_retain.yaml" \
  "${BASE_RUN}_src-dsdm_po-kd-sender_subset_po-lkd-1p0_po-retain-1p0_po-full"

run_experiment \
  "dsdm_head" \
  "configs/exp/260608-cifar10_direct_dsdm_packet_head.yaml" \
  "${BASE_RUN}_src-dsdm_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-head"

run_experiment \
  "dsdm_head_retain" \
  "configs/exp/260608-cifar10_direct_dsdm_packet_head_retain.yaml" \
  "${BASE_RUN}_src-dsdm_po-kd-sender_subset_po-lkd-1p0_po-retain-1p0_po-head"

run_experiment \
  "raw_plain" \
  "configs/exp/260608-cifar10_direct_raw.yaml" \
  "${BASE_RUN}_src-raw_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-full"

run_experiment \
  "raw_retain" \
  "configs/exp/260608-cifar10_direct_raw_packet_retain.yaml" \
  "${BASE_RUN}_src-raw_po-kd-sender_subset_po-lkd-1p0_po-retain-1p0_po-full"

run_experiment \
  "raw_head" \
  "configs/exp/260608-cifar10_direct_raw_packet_head.yaml" \
  "${BASE_RUN}_src-raw_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-head"

run_experiment \
  "raw_head_retain" \
  "configs/exp/260608-cifar10_direct_raw_packet_head_retain.yaml" \
  "${BASE_RUN}_src-raw_po-kd-sender_subset_po-lkd-1p0_po-retain-1p0_po-head"

run_step \
  "summarize_packet_only_ablation" \
  "${OUTPUT_ROOT}/reports/summary/.always_run_packet_only_ablation" \
  "${PYTHON_BIN}" -m src.main.run_summarize_reports --config "configs/exp/260608-cifar10_direct_dsdm.yaml"

TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START_TS ))
echo "finished_at: $(now_text)"
echo "total_elapsed: $(format_seconds "${TOTAL_ELAPSED}")"
echo "summary: ${OUTPUT_ROOT}/reports/summary/compare_summary.md"

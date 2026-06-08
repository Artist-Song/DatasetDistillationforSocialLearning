#!/usr/bin/env bash
set -euo pipefail

# Run the CIFAR10 direct social-packet experiment end to end.
#
# Usage:
#   bash scripts/run_cifar10_direct_real.sh
#
# Useful overrides:
#   PYTHON=python3 bash scripts/run_cifar10_direct_real.sh
#   FORCE=1 bash scripts/run_cifar10_direct_real.sh
#
# AutoDL background run:
#   mkdir -p outputs/260608-cifar10_direct/logs
#   nohup bash scripts/run_cifar10_direct_real.sh > outputs/260608-cifar10_direct/logs/nohup.log 2>&1 &

PYTHON_BIN="${PYTHON:-python}"
FORCE="${FORCE:-0}"

DSDM_CONFIG="configs/exp/260608-cifar10_direct_dsdm.yaml"
RAW_CONFIG="configs/exp/260608-cifar10_direct_raw.yaml"

BASE_RUN="cifar10_direct_conv"
DSDM_RUN="${BASE_RUN}_src-dsdm_kd-sender_subset_retain-1p0_packet-1p0"
RAW_RUN="${BASE_RUN}_src-raw_kd-sender_subset_retain-1p0_packet-1p0"
DSDM_PACKET_ONLY_RUN="${BASE_RUN}_src-dsdm_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-full"
RAW_PACKET_ONLY_RUN="${BASE_RUN}_src-raw_po-kd-sender_subset_po-lkd-1p0_po-retain-0p0_po-full"
OUTPUT_ROOT="outputs/260608-cifar10_direct"

LOCAL_CKPT_DIR="${OUTPUT_ROOT}/checkpoints/local_pretrain/${BASE_RUN}"
DSDM_PACKET_DIR="${OUTPUT_ROOT}/packets/${BASE_RUN}/dsdm"
RAW_PACKET_DIR="${OUTPUT_ROOT}/packets/${BASE_RUN}/raw"
DSDM_SOCIAL_DIR="${OUTPUT_ROOT}/checkpoints/social_train/${DSDM_RUN}"
RAW_SOCIAL_DIR="${OUTPUT_ROOT}/checkpoints/social_train/${RAW_RUN}"
DSDM_PACKET_ONLY_DIR="${OUTPUT_ROOT}/checkpoints/packet_only_train/${DSDM_PACKET_ONLY_RUN}"
RAW_PACKET_ONLY_DIR="${OUTPUT_ROOT}/checkpoints/packet_only_train/${RAW_PACKET_ONLY_RUN}"

LOCAL_EVAL_REPORT="${OUTPUT_ROOT}/reports/eval/local_pretrain_${BASE_RUN}_all.json"
DSDM_EVAL_REPORT="${OUTPUT_ROOT}/reports/eval/social_train_${DSDM_RUN}_all.json"
RAW_EVAL_REPORT="${OUTPUT_ROOT}/reports/eval/social_train_${RAW_RUN}_all.json"
DSDM_PACKET_ONLY_EVAL_REPORT="${OUTPUT_ROOT}/reports/eval/packet_only_train_${DSDM_PACKET_ONLY_RUN}_all.json"
RAW_PACKET_ONLY_EVAL_REPORT="${OUTPUT_ROOT}/reports/eval/packet_only_train_${RAW_PACKET_ONLY_RUN}_all.json"
DSDM_COMPARE_REPORT="${OUTPUT_ROOT}/reports/compare/compare_${DSDM_RUN}_all.json"
RAW_COMPARE_REPORT="${OUTPUT_ROOT}/reports/compare/compare_${RAW_RUN}_all.json"
DSDM_PACKET_ONLY_COMPARE_REPORT="${OUTPUT_ROOT}/reports/compare/compare_packet_only_train_${DSDM_PACKET_ONLY_RUN}_all.json"
RAW_PACKET_ONLY_COMPARE_REPORT="${OUTPUT_ROOT}/reports/compare/compare_packet_only_train_${RAW_PACKET_ONLY_RUN}_all.json"
SUMMARY_REPORT="${OUTPUT_ROOT}/reports/summary/compare_summary.md"
DSDM_INSPECT_REPORT="${OUTPUT_ROOT}/reports/packet_inspect/packet_inspection_${BASE_RUN}_dsdm_all.md"
RAW_INSPECT_REPORT="${OUTPUT_ROOT}/reports/packet_inspect/packet_inspection_${BASE_RUN}_raw_all.md"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
SCRIPT_START_TS="$(date +%s)"

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
  echo "      tqdm output inside this step will show live ETA where available."
  if "$@" 2>&1 | tee "${log_path}"; then
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[done] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
  else
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[fail] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
    return 1
  fi
}

expect_agent_files() {
  local dir="$1"
  local pattern="$2"
  local expected="$3"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi

  local count
  count="$(find "${dir}" -maxdepth 1 -name "${pattern}" | wc -l | tr -d ' ')"
  [[ "${count}" == "${expected}" ]]
}

run_agent_step() {
  local name="$1"
  local dir="$2"
  local pattern="$3"
  local expected="$4"
  shift 4

  local log_path="${LOG_DIR}/${name}.log"
  if [[ "${FORCE}" != "1" ]] && expect_agent_files "${dir}" "${pattern}" "${expected}"; then
    echo "[skip] $(now_text) ${name}: found ${expected} files in ${dir}"
    return
  fi

  local start_ts
  start_ts="$(date +%s)"
  echo "[run] $(now_text) ${name}"
  echo "      log: ${log_path}"
  echo "      tqdm output inside this step will show live ETA where available."
  if "$@" 2>&1 | tee "${log_path}"; then
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[done] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
  else
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[fail] $(now_text) ${name} elapsed=$(format_seconds "${elapsed}")"
    return 1
  fi
}

echo "=== CIFAR10 direct real experiment ==="
echo "started_at: $(now_text)"
echo "python: ${PYTHON_BIN}"
echo "force: ${FORCE}"
echo "dsdm_config: ${DSDM_CONFIG}"
echo "raw_config: ${RAW_CONFIG}"
echo "output_root: ${OUTPUT_ROOT}"
echo "log_dir: ${LOG_DIR}"

run_agent_step \
  "01_local_pretrain" \
  "${LOCAL_CKPT_DIR}" \
  "agent_*_anchor.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_local_pretrain --config "${DSDM_CONFIG}"

run_step \
  "02_eval_local" \
  "${LOCAL_EVAL_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_eval --config "${DSDM_CONFIG}" --checkpoint-stage local_pretrain

run_agent_step \
  "03_build_packets_dsdm" \
  "${DSDM_PACKET_DIR}" \
  "agent_*_packet.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_build_packets --config "${DSDM_CONFIG}"

run_step \
  "04_inspect_packets_dsdm" \
  "${DSDM_INSPECT_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_inspect_packets --config "${DSDM_CONFIG}"

run_agent_step \
  "05_social_train_dsdm" \
  "${DSDM_SOCIAL_DIR}" \
  "agent_*_social.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_social_train --config "${DSDM_CONFIG}"

run_step \
  "06_eval_social_dsdm" \
  "${DSDM_EVAL_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_eval --config "${DSDM_CONFIG}" --checkpoint-stage social_train

run_step \
  "07_compare_dsdm" \
  "${DSDM_COMPARE_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_compare --config "${DSDM_CONFIG}"

run_agent_step \
  "08_packet_only_train_dsdm" \
  "${DSDM_PACKET_ONLY_DIR}" \
  "agent_*_packet_only.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_packet_only_train --config "${DSDM_CONFIG}"

run_step \
  "09_eval_packet_only_dsdm" \
  "${DSDM_PACKET_ONLY_EVAL_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_eval --config "${DSDM_CONFIG}" --checkpoint-stage packet_only_train

run_step \
  "10_compare_packet_only_dsdm" \
  "${DSDM_PACKET_ONLY_COMPARE_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_compare --config "${DSDM_CONFIG}" --target-stage packet_only_train

run_agent_step \
  "11_build_packets_raw" \
  "${RAW_PACKET_DIR}" \
  "agent_*_packet.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_build_packets --config "${RAW_CONFIG}"

run_step \
  "12_inspect_packets_raw" \
  "${RAW_INSPECT_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_inspect_packets --config "${RAW_CONFIG}"

run_agent_step \
  "13_social_train_raw" \
  "${RAW_SOCIAL_DIR}" \
  "agent_*_social.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_social_train --config "${RAW_CONFIG}"

run_step \
  "14_eval_social_raw" \
  "${RAW_EVAL_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_eval --config "${RAW_CONFIG}" --checkpoint-stage social_train

run_step \
  "15_compare_raw" \
  "${RAW_COMPARE_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_compare --config "${RAW_CONFIG}"

run_agent_step \
  "16_packet_only_train_raw" \
  "${RAW_PACKET_ONLY_DIR}" \
  "agent_*_packet_only.pt" \
  "5" \
  "${PYTHON_BIN}" -m src.main.run_packet_only_train --config "${RAW_CONFIG}"

run_step \
  "17_eval_packet_only_raw" \
  "${RAW_PACKET_ONLY_EVAL_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_eval --config "${RAW_CONFIG}" --checkpoint-stage packet_only_train

run_step \
  "18_compare_packet_only_raw" \
  "${RAW_PACKET_ONLY_COMPARE_REPORT}" \
  "${PYTHON_BIN}" -m src.main.run_compare --config "${RAW_CONFIG}" --target-stage packet_only_train

run_step \
  "19_summarize_reports" \
  "${OUTPUT_ROOT}/reports/summary/.always_run_summarize" \
  "${PYTHON_BIN}" -m src.main.run_summarize_reports --config "${DSDM_CONFIG}"

run_step \
  "20_plot_results" \
  "${OUTPUT_ROOT}/reports/figures/.always_run_plot_results" \
  "${PYTHON_BIN}" -m src.main.run_plot_results --config "${DSDM_CONFIG}"

echo "=== done ==="
echo "DSDM compare: ${DSDM_COMPARE_REPORT}"
echo "RAW compare:  ${RAW_COMPARE_REPORT}"
echo "DSDM packet-only compare: ${DSDM_PACKET_ONLY_COMPARE_REPORT}"
echo "RAW packet-only compare:  ${RAW_PACKET_ONLY_COMPARE_REPORT}"
echo "Summary:      ${SUMMARY_REPORT}"
echo "Overview figure: ${OUTPUT_ROOT}/reports/figures/experiment_overview.png"
echo "Tradeoff figure: ${OUTPUT_ROOT}/reports/figures/expert_general_tradeoff.png"
echo "DSDM inspect: ${DSDM_INSPECT_REPORT}"
echo "RAW inspect:  ${RAW_INSPECT_REPORT}"
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START_TS ))
echo "finished_at: $(now_text)"
echo "total_elapsed: $(format_seconds "${TOTAL_ELAPSED}")"

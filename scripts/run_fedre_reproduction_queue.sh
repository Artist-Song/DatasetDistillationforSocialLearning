#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/DatasetDistillationforSocialLearning"
PYTHON="/root/miniconda3/envs/sp/bin/python"
OUTPUT_ROOT="${ROOT}/external_baselines/outputs/fedre_reproduction"
LOG_ROOT="${ROOT}/logs/fedre_reproduction"
PID_ROOT="${OUTPUT_ROOT}/pids"

mkdir -p "${LOG_ROOT}" "${PID_ROOT}"
cd "${ROOT}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_seed_pair() {
    local seed="$1"
    local task1_dir="${OUTPUT_ROOT}/task1_official_pat10/seed_${seed}"
    local task3_dir="${OUTPUT_ROOT}/task3_social_pat4/seed_${seed}"
    local task1_log="${LOG_ROOT}/task1_official_pat10_seed${seed}.log"
    local task3_log="${LOG_ROOT}/task3_social_pat4_seed${seed}.log"

    echo "running seed ${seed}" >"${PID_ROOT}/queue.status"

    "${PYTHON}" baseline_adapters/run_fedre_reproduction.py \
        --dataset Cifar100FedREPat10 \
        --num-clients 10 \
        --seed "${seed}" \
        --output-dir "${task1_dir}" \
        >"${task1_log}" 2>&1 &
    local task1_pid=$!

    "${PYTHON}" baseline_adapters/run_fedre_reproduction.py \
        --dataset Cifar100SocialPAT4 \
        --num-clients 4 \
        --seed "${seed}" \
        --output-dir "${task3_dir}" \
        >"${task3_log}" 2>&1 &
    local task3_pid=$!

    {
        echo "task1_pid=${task1_pid}"
        echo "task3_pid=${task3_pid}"
        echo "seed=${seed}"
    } >"${PID_ROOT}/seed_${seed}.pid"

    local task1_status=0
    local task3_status=0
    wait "${task1_pid}" || task1_status=$?
    wait "${task3_pid}" || task3_status=$?
    if [[ "${task1_status}" -ne 0 || "${task3_status}" -ne 0 ]]; then
        echo "seed ${seed} failed: task1=${task1_status}, task3=${task3_status}" >&2
        return 1
    fi

    "${PYTHON}" baseline_adapters/validate_fedre_reproduction.py \
        --run-dir "${task1_dir}" \
        --expected-clients 10 \
        --expected-paper-accuracy 62.56 \
        --paper-accuracy-tolerance 10.0 \
        >"${LOG_ROOT}/task1_official_pat10_seed${seed}_validation.json"
    "${PYTHON}" baseline_adapters/validate_fedre_reproduction.py \
        --run-dir "${task3_dir}" \
        --expected-clients 4 \
        --minimum-paper-accuracy 10.0 \
        >"${LOG_ROOT}/task3_social_pat4_seed${seed}_validation.json"
}

"${PYTHON}" baseline_adapters/prepare_fedre_reproduction.py

# Seed 0 is the correctness gate. Later seeds are launched only after both
# official-protocol runs pass structural and metric validation.
run_seed_pair 0
run_seed_pair 1
run_seed_pair 2

echo "complete" >"${PID_ROOT}/queue.status"

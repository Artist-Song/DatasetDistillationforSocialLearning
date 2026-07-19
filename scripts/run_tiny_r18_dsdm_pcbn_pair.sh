#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/autodl-tmp/DatasetDistillationforSocialLearning"
PY="/root/miniconda3/envs/sp/bin/python"
PURE_CONFIG="configs/tinyimagenet_r18_agent0_dsdm_ipc10_seed0.yaml"
PCBN_CONFIG="configs/tinyimagenet_r18_agent0_dsdm_pcbn_ipc10_seed0.yaml"
PURE_RUN="tinyimagenet_r18_agent0_dsdm_ipc10_seed0"
PCBN_RUN="tinyimagenet_r18_agent0_dsdm_pcbn_ipc10_seed0"
LOG_DIR="${BASE_DIR}/logs/tinyimagenet_r18_dsdm_pcbn_pair"

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" scripts/validate_tiny_r18_dsdm_pair.py --pure "$PURE_CONFIG" --pcbn "$PCBN_CONFIG" \
  >"${LOG_DIR}/preflight.log" 2>&1

"$PY" run_social_pipeline.py --config "$PURE_CONFIG" --stage train_experts --only-agent 0 --resume \
  >"${LOG_DIR}/train_shared_guides.log" 2>&1

"$PY" run_social_pipeline.py --config "$PCBN_CONFIG" --stage train_experts --only-agent 0 --dry-run \
  >"${LOG_DIR}/prepare_pcbn_output.log" 2>&1

PURE_CKPT="${BASE_DIR}/outputs/${PURE_RUN}/agents/agent_0/checkpoints"
PCBN_CKPT="${BASE_DIR}/outputs/${PCBN_RUN}/agents/agent_0/checkpoints"
mkdir -p "$PCBN_CKPT"
for index in $(seq 0 9); do
  "$PY" -c 'from output_manager import atomic_copyfile; import sys; atomic_copyfile(sys.argv[1], sys.argv[2])' \
    "${PURE_CKPT}/guide_model_${index}.pt" "${PCBN_CKPT}/guide_model_${index}.pt"
done
for name in expert_model.pt expert_selection.json; do
  "$PY" -c 'from output_manager import atomic_copyfile; import sys; atomic_copyfile(sys.argv[1], sys.argv[2])' \
    "${PURE_CKPT}/${name}" "${PCBN_CKPT}/${name}"
done

for index in $(seq 0 9); do
  cmp "${PURE_CKPT}/guide_model_${index}.pt" "${PCBN_CKPT}/guide_model_${index}.pt"
done

sha256sum "${PURE_CKPT}"/guide_model_*.pt "${PCBN_CKPT}"/guide_model_*.pt \
  >"${LOG_DIR}/shared_guide_sha256.txt"

pids=()
"$PY" run_social_pipeline.py --config "$PURE_CONFIG" --stage distill_packets --only-agent 0 --resume \
  >"${LOG_DIR}/distill_pure.log" 2>&1 &
pids+=("$!")
"$PY" run_social_pipeline.py --config "$PCBN_CONFIG" --stage distill_packets --only-agent 0 --resume \
  >"${LOG_DIR}/distill_pcbn.log" 2>&1 &
pids+=("$!")
printf '%s\n' "${pids[@]}" >"${LOG_DIR}/active_distill_pids.txt"
echo "$(date -Is) pure_pid=${pids[0]} pcbn_pid=${pids[1]}" >"${LOG_DIR}/distill_start.txt"

status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    :
  else
    code=$?
    status=$code
  fi
done
rm -f "${LOG_DIR}/active_distill_pids.txt"
echo "$(date -Is) exit=${status}" >"${LOG_DIR}/distill_complete.txt"
exit "$status"

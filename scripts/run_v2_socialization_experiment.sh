#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/v2/cifar10_5agent_dsdm.yaml}"
AGENT_IDS="${AGENT_IDS:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---no-download}"

echo "=== v2 socialization-only experiment ==="
echo "CONFIG=${CONFIG}"
echo "AGENT_IDS=${AGENT_IDS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "DOWNLOAD_FLAG=${DOWNLOAD_FLAG}"

for PACKET_SOURCE in raw strict_dsdm; do
  python -m src.main.run_socialize_agents_v2 \
    --config "${CONFIG}" \
    --packet-source "${PACKET_SOURCE}" \
    --adaptation-mode last_block_anchor \
    --agent-ids "${AGENT_IDS}" \
    --num-workers "${NUM_WORKERS}" \
    ${DOWNLOAD_FLAG}

  python -m src.main.run_eval_v2 \
    --config "${CONFIG}" \
    --checkpoint-stage socialized \
    --packet-source "${PACKET_SOURCE}" \
    --adaptation-mode last_block_anchor \
    --agent-ids "${AGENT_IDS}" \
    --num-workers "${NUM_WORKERS}" \
    ${DOWNLOAD_FLAG}

  python -m src.main.run_socialize_agents_v2 \
    --config "${CONFIG}" \
    --packet-source "${PACKET_SOURCE}" \
    --adaptation-mode full_finetune \
    --agent-ids "${AGENT_IDS}" \
    --num-workers "${NUM_WORKERS}" \
    ${DOWNLOAD_FLAG}

  python -m src.main.run_eval_v2 \
    --config "${CONFIG}" \
    --checkpoint-stage socialized \
    --packet-source "${PACKET_SOURCE}" \
    --adaptation-mode full_finetune \
    --agent-ids "${AGENT_IDS}" \
    --num-workers "${NUM_WORKERS}" \
    ${DOWNLOAD_FLAG}
done

python -m src.main.run_compare_v2 \
  --config "${CONFIG}"

echo "=== v2 socialization-only experiment done ==="

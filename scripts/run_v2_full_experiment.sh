#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/v2/cifar10_5agent_dsdm.yaml}"
AGENT_IDS="${AGENT_IDS:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:---no-download}"

echo "=== v2 full experiment ==="
echo "CONFIG=${CONFIG}"
echo "AGENT_IDS=${AGENT_IDS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "DOWNLOAD_FLAG=${DOWNLOAD_FLAG}"

python -m src.main.run_train_agents_v2 \
  --config "${CONFIG}" \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_pretrain_dsdm_guides \
  --config "${CONFIG}" \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_build_packets_v2 \
  --config "${CONFIG}" \
  --packet-source raw \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  --skip-existing \
  ${DOWNLOAD_FLAG}

python -m src.main.run_build_packets_v2 \
  --config "${CONFIG}" \
  --packet-source strict_dsdm \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  --skip-existing \
  ${DOWNLOAD_FLAG}

python -m src.main.run_socialize_agents_v2 \
  --config "${CONFIG}" \
  --packet-source raw \
  --adaptation-mode last_block_anchor \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_socialize_agents_v2 \
  --config "${CONFIG}" \
  --packet-source strict_dsdm \
  --adaptation-mode last_block_anchor \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_socialize_agents_v2 \
  --config "${CONFIG}" \
  --packet-source raw \
  --adaptation-mode full_finetune \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_socialize_agents_v2 \
  --config "${CONFIG}" \
  --packet-source strict_dsdm \
  --adaptation-mode full_finetune \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_eval_v2 \
  --config "${CONFIG}" \
  --checkpoint-stage expert \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_eval_v2 \
  --config "${CONFIG}" \
  --checkpoint-stage socialized \
  --packet-source raw \
  --adaptation-mode last_block_anchor \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_eval_v2 \
  --config "${CONFIG}" \
  --checkpoint-stage socialized \
  --packet-source strict_dsdm \
  --adaptation-mode last_block_anchor \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_eval_v2 \
  --config "${CONFIG}" \
  --checkpoint-stage socialized \
  --packet-source raw \
  --adaptation-mode full_finetune \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_eval_v2 \
  --config "${CONFIG}" \
  --checkpoint-stage socialized \
  --packet-source strict_dsdm \
  --adaptation-mode full_finetune \
  --agent-ids "${AGENT_IDS}" \
  --num-workers "${NUM_WORKERS}" \
  ${DOWNLOAD_FLAG}

python -m src.main.run_compare_v2 \
  --config "${CONFIG}"

echo "=== v2 full experiment done ==="

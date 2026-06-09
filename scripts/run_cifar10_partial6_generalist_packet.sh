#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp/2606-cifar10_partial6_global_raw_packet_smoke.yaml}"

python -m src.main.run_train_generalist --config "${CONFIG}"
python -m src.main.run_train_specialists --config "${CONFIG}" --agent-ids all
python -m src.main.run_build_generalist_packets --config "${CONFIG}"
python -m src.main.run_train_social_head --config "${CONFIG}" --agent-ids all
python -m src.main.run_eval_specialists --config "${CONFIG}" --checkpoint-stage local
python -m src.main.run_eval_specialists --config "${CONFIG}" --checkpoint-stage social_head
python -m src.main.run_compare_generalist --config "${CONFIG}"

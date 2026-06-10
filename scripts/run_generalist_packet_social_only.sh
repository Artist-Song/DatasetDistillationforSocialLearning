#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?please provide config path}"

python -m src.main.run_train_social_head --config "${CONFIG}" --agent-ids all
python -m src.main.run_eval_specialists --config "${CONFIG}" --checkpoint-stage local
python -m src.main.run_eval_specialists --config "${CONFIG}" --checkpoint-stage social_head
python -m src.main.run_compare_generalist --config "${CONFIG}"
python -m src.main.run_collect_experiment_results --output-root ./outputs

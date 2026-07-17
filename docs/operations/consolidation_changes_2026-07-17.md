# Consolidation Changes (2026-07-17)

## Canonicalized

- `AGENTS.md`: reduced to the current instruction matrix.
- `PROJECT_SPEC.md`: current research definition, protocol, hyperparameters, and interpretation.
- `RESULTS.md`: generated from `experiments/registry.yaml`.
- `EXPERIMENT_LOG.md`: compact chronological decision log.
- `README.md`: repository navigation and current entry points.

## Archived

- Previous `AGENTS.md`, `PROJECT_SPEC.md`, `WORKLOG.md`, baseline server notes, and paper context were moved under `docs/archive/` with dated names.
- Historical Conv-family, all-ConvNet, hetero4/6, no-VGG/ResNet-guide, recipe diagnostics, and centralized upper bounds remain registered evidence.

## Deleted Under Approved Policy 1A

- 47 non-best tuning configuration files; exact paths are recorded in `deleted_configs_2026-07-17.csv`.
- Two obsolete queue scripts that referenced only those removed configs:
  `scripts/run_receiver_ablation_queue.sh` and `scripts/run_receiver_tuning_queue.sh`.
- Stale Jupyter checkpoint copies of `AGENTS.md`, `PROJECT_SPEC.md`, and `WORKLOG.md`.
- Redundant `paper_tables/main_results_full_data_validation.json`, superseded by the generated provenance file.

## Awaiting Explicit Deletion Approval

- 48 non-best output directories containing checkpoints and/or packets.
- Estimated reclaim: 1.85 GB.
- Exact paths: `deletion_candidates_2026-07-17.csv`.

No current one-ResNet output, canonical Conv-family/all-ConvNet evidence, registered diagnostic, or active queue artifact was deleted.

# Cleanup Candidates (2026-07-17)

This is a review-only inventory. No output, checkpoint, or packet was deleted.

- Candidate output directories: `48`
- Candidate logical size: `1.85 GB`
- Candidate unique physical blocks: `1.85 GB`
- Estimated reclaim if every candidate is approved: `1.85 GB`
- Candidate blocks retained by hard links outside the list: `0.00 B`
- Config files attached to candidates: `0`
- Obsolete configs already removed under policy 1A: `47`
- Best tuning rows retained: `12`

Protected families include current one-ResNet, canonical Conv-family/all-ConvNet runs, hetero4/6, upper bounds, no-VGG and ResNet recipe diagnostics, cross-architecture diagnostics, registry sources, and each tuning family's selected best run. Non-best Conv-family tuning/ablation branches may still appear below; the canonical Conv-family IPC=10/50 evidence never does.

## Largest Candidates

| Run | Size | Reason |
|---|---:|---|
| `masc_official_smoke` | 424.08 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_hetero_receiver_tuning_ipc50_ep200_lr0010_fr005_kd050_t2` | 73.06 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_hetero_receiver_tuning_ipc50_ep300_lr0010_fr005_kd050_t2` | 73.05 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc50_ep275_fr005_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc50_ep250_fr006_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc50_ep250_fr004_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc50_ep500_fr0075_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_ipc50_ep350_fr005_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_ipc50_ep300_fr005_kd040` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc50_ep500_fr003_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_ipc50_ep300_fr005_kd060` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc50_ep700_fr005_kd050` | 67.34 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched_latesched` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched_self15` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_latesched` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_self20` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_self15` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep055_fr020_kd060_t2` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t1` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t3` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep065_fr020_kd060_t2` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd070_t3` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t4` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr025_kd060_t2` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep050_fr020_kd050` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr025_kd050` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr015_kd050` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr030_kd060_t2` | 20.60 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |

Detailed lists:

- `docs/operations/deletion_candidates_2026-07-17.csv`
- `docs/operations/config_candidates_2026-07-17.csv`
- `docs/operations/deleted_configs_2026-07-17.csv` (completed config cleanup)
- `experiments/generated/tuning_best_by_family.csv`

Deletion requires explicit user approval after checking hard-link/source dependencies.

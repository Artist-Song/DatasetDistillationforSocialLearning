# Cleanup Candidates (2026-07-27)

This is a review-only inventory. No output, checkpoint, or packet was deleted.

- Candidate output directories: `52`
- Candidate logical size: `9.43 MB`
- Candidate unique physical blocks: `10.12 MB`
- Estimated reclaim if every candidate is approved: `10.12 MB`
- Candidate blocks retained by hard links outside the list: `0.00 B`
- Config files attached to candidates: `2`
- Obsolete configs already removed under policy 1A: `0`
- Best tuning rows retained: `12`

Protected families include current one-ResNet, canonical Conv-family/all-ConvNet runs, hetero4/6, upper bounds, no-VGG and ResNet recipe diagnostics, cross-architecture diagnostics, registry sources, and each tuning family's selected best run. Non-best Conv-family tuning/ablation branches may still appear below; the canonical Conv-family IPC=10/50 evidence never does.

## Largest Candidates

| Run | Size | Reason |
|---|---:|---|
| `cifar100_tq_conv3_guidee0002_seed0_ipc10_smoke_v2` | 4.19 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_tq_conv3_guidee0002_seed0_ipc10_smoke` | 4.19 MB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched_latesched` | 23.46 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched_self15` | 23.30 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_latesched` | 23.25 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_sched` | 23.19 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_self20` | 23.10 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t2_self15` | 23.07 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t1` | 23.04 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd070_t3` | 23.03 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep065_fr020_kd060_t2` | 23.03 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t4` | 23.03 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr025_kd060_t2` | 23.03 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep055_fr020_kd060_t2` | 23.02 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr030_kd060_t2` | 23.02 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060_t3` | 23.02 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr025_kd050` | 23.00 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr020_kd060` | 23.00 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep050_fr020_kd050` | 22.99 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_followup_ipc10_ep060_fr015_kd050` | 22.97 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc10_ep100_fr020_kd040` | 22.88 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_refine_tuning_ipc10_ep090_fr020_kd050` | 22.88 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc10_ep125_fr020_kd050` | 22.88 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_joint_tuning_ipc10_ep100_fr020_kd060` | 22.86 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_tuning_E_ep200_fr015_kd050` | 22.40 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_ablation_fr010_kd050_ep200` | 22.39 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_ablation_fr010_kd025_ep500` | 22.38 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_ablation_fr010_kd050_ep500` | 22.38 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_ablation_fr020_kd050_ep500` | 22.38 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |
| `cifar100_4agent_25cls_conv_family_tuning_C_ep250_fr020_kd050` | 22.38 KB | non-best tuning/smoke/intermediate run; review before deleting artifacts |

Detailed lists:

- `docs/operations/deletion_candidates_2026-07-27.csv`
- `docs/operations/config_candidates_2026-07-27.csv`
- `docs/operations/deleted_configs_2026-07-27.csv` (completed config cleanup)
- `experiments/generated/tuning_best_by_family.csv`

Deletion requires explicit user approval after checking hard-link/source dependencies.

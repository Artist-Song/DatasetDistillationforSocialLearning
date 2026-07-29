# Main Results (2026-07-28)

Generated from `experiments/registry.yaml` on `2026-07-28`.
Do not edit numeric tables manually.

## Canonical IPC Results

| Method | Packet | Soft labels | n10 | IPC10 Global | New | Expert | Forgetting | n50 | IPC50 Global | New | Expert | Forgetting |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MASC-complete* | Real CC images | Teacher KD | 1 | 9.96 | 2.76 | 31.55 | 39.69 | 1 | 22.81 | 14.79 | 46.89 | 24.35 |
| DeSA-CIL* | Synthetic anchors | Owner KD | 1 | 19.54 | 4.24 | 65.41 | 0.53 | 1 | 15.02 | 0.96 | 57.20 | 8.74 |
| Heuristic | Random real images | No | 3 | 22.35 +/- 0.14 | 18.14 +/- 0.33 | 34.98 +/- 0.73 | 32.06 +/- 0.49 | 3 | 32.58 +/- 0.50 | 29.24 +/- 0.80 | 42.57 +/- 0.87 | 24.47 +/- 0.68 |
| FAST | Selected real images | No | 3 | 20.30 +/- 0.31 | 15.58 +/- 0.16 | 34.44 +/- 1.00 | 32.60 +/- 1.01 | 3 | 31.06 +/- 0.36 | 27.18 +/- 0.46 | **42.70 +/- 0.58** | **24.34 +/- 0.42** |
| Ours | Distilled images | Sender expert logits | 3 | **32.78 +/- 0.29** | **29.11 +/- 0.35** | **43.81 +/- 0.69** | **23.24 +/- 0.60** | 2 | **35.60 +/- 0.56** | **33.48 +/- 0.56** | 41.96 +/- 0.59 | 25.14 +/- 0.87 |

Notes:

- Ours is sender task-backbone-guided DSDM with sender expert-class logits.
- Heuristic and FAST use hard labels only. FAST follows official pixels/per-class/minmax.
- MASC-complete* and DeSA-CIL* are single-seed adapted baselines with protocol caveats.
- Bold values are computed only within the directly comparable one-ResNet image-budget group.
- Ours IPC=50 is the final two-seed result over seeds 0 and 2; Heuristic and FAST retain their three-seed aggregates.
- High expert accuracy with near-zero new accuracy is not successful socialization.

## Full-data Validation

| Method | Protocol | n | Global | New | Expert | Forgetting |
|---|---|---:|---:|---:|---:|---:|
| Full Real | one-ResNet heterogeneous receivers; 37,500 external images/receiver | 3 | 51.16 +/- 0.32 | 50.40 +/- 0.47 | 53.44 +/- 0.42 | 13.60 +/- 0.19 |
| MASC Full | homogeneous official Netwider; validation gate | 1 | 65.43 | 65.03 | 66.62 | 4.62 |

Full-data rows validate implementations and are not IPC-matched method rankings.

## Important Intermediate Evidence

| ID | Role | Status | Evidence retained |
|---|---|---|---|
| `conv_family_core` | historical_evidence | complete | Established low-budget DSDM behavior and receiver hyperparameter basis.<br>`outputs/cifar100_4agent_25cls_conv_family_ipc10/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_conv_family_ipc50/metrics/social_results.csv` |
| `all_conv_ipc50` | historical_evidence | complete | High-budget receiver and communication diagnostic; do not mark invalid.<br>`outputs/cifar100_4agent_25cls_ipc50_allconvnet/metrics/social_results.csv` |
| `hetero4arch_history` | diagnostic | complete | Stability and feature-index evidence; retain metrics/config/provenance.<br>`outputs/cifar100_4agent_25cls_hetero4arch_ipc10/metrics/social_results.csv` |
| `hetero6arch_history` | historical_evidence | complete | Cross-architecture ConvNet-family packet reuse and guide-mode comparisons.<br>`outputs/cifar100_4agent_25cls_hetero6arch_ipc10_seed0/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_hetero6arch_ipc10_seed1/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_hetero6arch_ipc10_seed2/metrics/social_results.csv` |
| `centralized_upper_bounds` | diagnostic | complete | Separates model capacity and training recipe from packet quality.<br>`outputs/cifar100_4agent_25cls_upper_bound/centralized_full/centralized_results_conv3in_dsdm_strict.csv`<br>`outputs/cifar100_4agent_25cls_upper_bound/centralized_full/centralized_results_resnet10_bn_aug_ms.csv` |
| `novgg_backbone_guidance` | diagnostic | complete | Retains self-guided ResNet packet provenance and penultimate-feature receiver comparisons.<br>`outputs/cifar100_4agent_25cls_novgg_r10_backbone_ipc10/metrics/novgg_r10_artifact_provenance.json`<br>`outputs/cifar100_4agent_25cls_novgg_r10_backbone_penult_ipc10/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_novgg_r18_backbone_ipc10/metrics/novgg_r18_artifact_provenance.json`<br>`outputs/cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10/metrics/social_results.csv` |
| `resnet_recipe_diagnostics` | diagnostic | complete | Separates compact-versus-standard model recipe behavior from selection packet behavior.<br>`outputs/cifar100_resnet18_compact_recipe_diag_seed0_ipc50_fast/metrics/packet_integrity_fast.csv`<br>`outputs/cifar100_resnet18_standard_recipe_diag_seed0_ipc50_fast/metrics/packet_integrity_fast.csv` |
| `tinyimagenet_backbone_validation` | diagnostic | complete | Seed0 clean-validation capacity gate for ConvNet-4, ResNet-18, AlexNet, and MobileNetV2.<br>`experiments/diagnostics/tinyimagenet_backbone_validation_seed0.json` |
| `tinyimagenet_r18_50class_dsdm_pcbn` | diagnostic | complete | IPC10 sender0 packet-quality comparison with byte-identical guide pools; PCBN best is 1.0408 points above pure DSDM.<br>`experiments/diagnostics/tinyimagenet_r18_50class_dsdm_pcbn_seed0.json`<br>`configs/tinyimagenet_r18_agent0_dsdm_ipc10_seed0.yaml`<br>`configs/tinyimagenet_r18_agent0_dsdm_pcbn_ipc10_seed0.yaml` |
| `tinyimagenet_r18_all200_dsdm_pcbn` | diagnostic | running | IPC10 all-200 scaling diagnostic; all-class guide pool training is active.<br>`configs/tinyimagenet_r18_all200_dsdm_ipc10_seed0.yaml`<br>`configs/tinyimagenet_r18_all200_dsdm_pcbn_ipc10_seed0.yaml` |
| `iclr2027_dkp_first_round_seed0_ipc10` | diagnostic | complete | Complete five-receiver cosine-expert, CE-only and full DKP-SL communication validation using backbone-specific full-class pool slices; diagnostic only, not sender-local DSDM evidence.<br>`configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml`<br>`outputs/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1/metrics/cosine_expert_preflight.json`<br>`outputs/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2/metrics/packet_integrity_dsdm.json`<br>`outputs/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/packet_integrity_dsdm.json`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/communication_preflight.json`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/first_round_seed0_summary.json`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/final_receiver_checkpoint_audit.json` |
| `iclr2027_dkp_matched_linear_head_seed0_ipc10` | diagnostic | complete | Complete five-receiver matched-linear expert, CE-only and full DKP-SL end-to-end classifier diagnostic; uses separately trained linear experts and norm-matched linear prototype imprinting, and is not a formal result.<br>`configs/iclr2027/cifar100_5agent20cls_dkp_linear_experts_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1.yaml`<br>`outputs/cifar100_5agent20cls_dkp_linear_experts_seed0_v1/metrics/linear_expert_preflight.json`<br>`outputs/cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_communication_preflight.json`<br>`outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_head_seed0_summary.json`<br>`outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_final_receiver_checkpoint_audit.json` |
| `iclr2027_dkp_loss_component_matrix_seed0_ipc10` | diagnostic | complete | Complete fixed FR/KD/SupCon binary matrix over eight conditions and five receivers; the six new conditions use final-only checkpoint retention, and the matrix is diagnostic rather than test-selected hyperparameter evidence.<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd0_sc1_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc0_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc1_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc0_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd1_sc0_ipc10_seed0_v1.yaml`<br>`configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml`<br>`outputs/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr0_kd0_sc1_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc0_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc1_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc0_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_ablation_fr1_kd1_sc0_ipc10_seed0_v1/metrics/social_results.csv`<br>`outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/social_results.csv`<br>`outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_v1.json`<br>`outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_final_checkpoint_audit_v1.json` |

## Provenance

| Experiment | Status | Seeds | Missing | Source files | Caveat |
|---|---|---:|---|---|---|
| `desa_cil_ipc10` | complete | 1 | - | `external_baselines/outputs/desa_cil/cifar100_4agent_25cls_one_resnet_seed0_ipc10/social_results.csv` | Single seed; iterative owner logits are additional communication. |
| `desa_cil_ipc50` | complete | 1 | - | `external_baselines/outputs/desa_cil/cifar100_4agent_25cls_one_resnet_seed0_ipc50/social_results.csv` | Single seed; iterative owner logits are additional communication. |
| `masc_complete_ipc10` | complete | 1 | - | `external_baselines/outputs/masc_complete/seed0_10/social_results.csv` | Single seed and about 452 MB model-parameter interaction. |
| `masc_complete_ipc50` | complete | 1 | - | `external_baselines/outputs/masc_complete/seed0_50/social_results.csv` | Single seed and about 452 MB model-parameter interaction. |
| `masc_full_validation` | complete | 1 | - | `external_baselines/outputs/masc_complete/seed0_full/social_results.csv` | Not comparable to one-ResNet Full Real as a method ranking. |
| `one_resnet_fast_ipc10` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc10_fast/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc10_fast/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc10_fast/metrics/social_results.csv` |  |
| `one_resnet_fast_ipc50` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc50_fast/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc50_fast/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc50_fast/metrics/social_results.csv` |  |
| `one_resnet_full_real` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc10_fullreal/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc10_fullreal/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc10_fullreal/metrics/social_results.csv` |  |
| `one_resnet_heuristic_ipc10` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc10_heuristic_hard/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc10_heuristic_hard/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc10_heuristic_hard/metrics/social_results.csv` |  |
| `one_resnet_heuristic_ipc50` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc50_heuristic_hard/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc50_heuristic_hard/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc50_heuristic_hard/metrics/social_results.csv` |  |
| `one_resnet_ours_ipc10` | complete | 3 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc10/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc10/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc10/metrics/social_results.csv` |  |
| `one_resnet_ours_ipc50` | complete | 2 | - | `outputs/cifar100_4agent_25cls_one_resnet_seed0_ipc50/metrics/social_results.csv`<br>`outputs/cifar100_4agent_25cls_one_resnet_seed2_ipc50/metrics/social_results.csv` | Final two-seed result over seeds 0 and 2; seed1 was stopped by user decision and is excluded. |

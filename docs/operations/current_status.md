# Current Status

更新时间：`2026-07-29 08:45 UTC`

## IPC10 多 seed / 多 agent 活动队列

主方法 `5/10/20 agents x seeds1/2/3` 九个单元已全部完成，105个receiver最终checkpoint均通过
strict-load、100维输出、finite state/output、结果行SHA和独立重评门禁。基于seed-level mean的
Global/New/Expert（population std）分别为：5-agent `39.1827+/-0.5228 / 33.7167+/-0.5334 /
61.0467+/-0.9058`，10-agent `34.0247+/-0.1653 / 31.3885+/-0.2033 /
57.7500+/-0.1768`，20-agent `29.6722+/-0.2126 / 28.2265+/-0.2209 /
57.1400+/-0.0779`。

5-agent的Heuristic、FAST和Full Real三seed，以及10-agent的Heuristic和FAST三seed均已完成并通过
完整审计。基于seed-level mean的Global/New/Expert分别为：5-agent Heuristic
`33.6047+/-0.5867 / 25.9058+/-0.7516 / 64.4000+/-0.0852`，FAST
`31.6693+/-0.2909 / 23.5400+/-0.4361 / 64.1867+/-0.3578`，Full Real
`58.0707+/-1.2810 / 54.4458+/-2.0711 / 72.5700+/-1.8820`；10-agent Heuristic
`26.7610+/-0.4469 / 23.9837+/-0.5145 / 51.7567+/-0.8858`，FAST
`24.1510+/-0.1236 / 21.1359+/-0.1487 / 51.2867+/-0.2175`。这些结果尚待registry登记，不能在
`RESULTS.md`中手工写入。

Full Real、Heuristic、FAST、DeSA-CIL、MASC-complete和FedRE的动态5/10/20-agent适配、通信统计、
provenance、resume门禁和完成审计已闭合；182项全量单测、py_compile、配置生成、48任务dry-run、
项目文档检查和`git diff --check`均通过。启动前RTX 4090为空闲`1 MiB / 24,564 MiB, 0%`，数据盘
可用约45 GiB；没有删除、移动或覆盖任何已有checkpoint、packet、pool source或provenance。

baseline统一队列：`scripts/run_iclr2027_baseline_matrix.py`；状态与日志位于
`logs/iclr2027_baseline_matrix/`。任一expert、packet、通信量、receiver coverage、checkpoint或
SHA provenance门禁失败都会停止队列，不会静默进入下一任务。

队列已于`2026-07-29T02:56:29Z`启动，master PID/PGID=`105288/105288`，最大并发为5。当前已完成
`16/48`个baseline单元，正在运行20-agent Heuristic seed2；前15个receiver已完成，最后5个
receiver于`08:44 UTC`后启动。最近采样为GPU utilization `99%`、显存`10,726 MiB`、功耗约
`244 W`，没有Traceback、OOM或non-finite日志；数据盘可用约36 GiB。状态文件：
`logs/iclr2027_baseline_matrix/queue_status.json`；master log：
`logs/iclr2027_baseline_matrix/master.log`。

扩容前已完成的最新通信诊断为：

```text
run_name: cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2
status: complete_diagnostic (formal_result=false, paper_eligible=false)
mean Global/New/Expert: 39.632 / 34.290 / 61.000
optimizer steps: 3780 per receiver
external raw images: 800 per receiver
sender-logit bytes: 128,000 per receiver
```

五个 receiver 的 `Global/New/Expert` 分别为：ConvNet-3
`38.910/35.625/52.050`，ConvNet-4 `40.880/35.075/64.100`，AlexNet
`39.510/35.3375/56.200`，standard R10 `40.550/33.8375/67.400`，standard R18
`38.310/31.575/65.250`。相对严格配对的 r02 KD-off，完整方法为
`Global +1.132 / New +1.8925 / Expert -1.910`。五个最终 checkpoint 均通过 strict-load、
finite state/output/feature、100维输出、cosine head 和 SHA-256 审计。证据：

```text
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_v2.json
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_full_final_checkpoint_audit_v2.json
configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2.yaml
```

用户已确认：蒸馏阶段使用 evaluator 从多个候选图片 snapshot 中选择知识价值最高的图片，属于本
项目接受的数据集蒸馏流程，不再作为通信实验阻断项。后续继续固定并复用现有五个 backbone-specific
全100类 IPC10 pool，不重新蒸馏；仍须完整保留 pool catalog、source packet、best snapshot 和
SHA-256 provenance。

当前队列顺序固定为：

1. Ours的5/10/20-agent三seed九个单元已完成并审计通过。
2. baseline先完成5-agent横向比较：Heuristic、FAST、Full Real三seed；其中Heuristic seed1已完成。
3. 再补齐10-agent和20-agent的Heuristic、FAST三seed；Full Real不扩到10/20-agent。
4. 外部baseline按DeSA-CIL、同构MASC-complete、FedRE分别完成5/10/20-agent三seed。
5. 当前不启动IPC20/IPC50；每个任务完成审计后才进入下一任务。

最终研究矩阵共57个method-scale-seed单元：Ours 9、Heuristic 9、FAST 9、Full Real 3、DeSA 9、
MASC 9、FedRE 9。baseline队列含48个单元，当前已完成16个、剩余32个（含正在运行的单元）。
静态配对审计已通过：相同split/model/seed、cosine expert复用、固定3780步、hard-label方法
零logits、FAST固定官方commit和补丁SHA、Full Real独立full-data group，以及外部方法各自独立的
通信口径均符合约定。

固定图片池 catalog：
`configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml`，catalog SHA-256 为
`1302f291efa7c57a4ec24487b64241d0672f3a3dc4d9b2ed88a1c84ebc848b28`。五种 backbone
继续严格使用 ConvNet-3、ConvNet-4、AlexNet、standard ResNet-10、standard ResNet-18；禁止把
compact ResNet 静默替换进当前协议。

## 活动队列

```text
Live process/GPU state was rechecked at 2026-07-28 08:19 UTC; archived PIDs were not treated as live.
The fixed-step seed0 DKP receiver domain-balance queue completed all 15 receiver jobs with exit 0. The
three non-overwriting diagnostic run names are:
cifar100_5agent20cls_dkp_domain_s_real_steps3780_ipc10_seed0_v1,
cifar100_5agent20cls_dkp_domain_u_packet_steps3780_ipc10_seed0_v1, and
cifar100_5agent20cls_dkp_domain_h_real_packet_steps3780_ipc10_seed0_v1.
Their complete diagnostic means are S=31.922/22.1075/71.180, U=38.708/34.2700/56.460 and
H=33.116/22.4100/75.940 for Global/New/Expert. The strict summary is:
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_balance_v1.json.

A second fixed seed0 internal curve completed under PID/PGID 421443. It fixed
all U settings and varies only the real-data loss weight in local CE at 0.05/0.10/0.20/0.30; all four
conditions are predeclared to cover all five receivers, remain paper_eligible=false, and use fresh run names
with `domain_mix_r05/r10/r20/r30`. This weight is not a sampling fraction. All 20 receiver jobs exited 0;
the means for r05/r10/r20/r30 are respectively 37.930/30.4725/67.760,
36.704/28.1325/70.990, 35.232/25.6875/73.410 and 34.186/23.9650/75.070 for
Global/New/Expert. All 55 launcher tests, expert/source preflights, packet validators, target provenance,
prototype, strict summary and 20-checkpoint strict-load audits passed. Evidence:
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_preflight_v1.json,
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_v1.json, and
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_final_checkpoint_audit_v1.json.
Master log: logs/iclr2027_dkp_domain_mix_seed0_v1_master.log.

The single post-curve r02 five-receiver confirmation completed under PID/PGID 429611 with the fresh
run name `cifar100_5agent20cls_dkp_domain_mix_r02_steps3780_ipc10_seed0_v1`. It is explicitly adaptive,
paper-ineligible and non-formal. All five receiver jobs exited 0. Its mean is
Global/New/Expert=38.500/32.3975/62.910; all 55 tests, expert/source checks, packet validator, combined
U/r02/r05/r10/r20/r30 packet/provenance/prototype preflight, strict summary, and 25-checkpoint combined
strict-load audit passed. Evidence:
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_r02_preflight_v1.json,
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_with_r02_v1.json, and
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_with_r02_final_checkpoint_audit_v1.json.
No receiver or external baseline process is live; the RTX 4090 was idle at 1 MiB / 24,564 MiB and 0%
utilization after completion. Disk free is about 1.4 GiB. Master log:
logs/iclr2027_dkp_domain_mix_r02_seed0_v1_master.log. Existing experts, packets, pool sources, outputs and
logs remain immutable; deletion still requires a cleanup manifest and user confirmation.

Live process/GPU state was rechecked at 2026-07-27 20:24 UTC; archived PIDs were not treated as live.
The matched-linear seed0 diagnostic completed five freshly trained experts and ten receivers under:
cifar100_5agent20cls_dkp_linear_experts_seed0_v1,
cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1, and
cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1.
Its five-receiver summary is complete_diagnostic and its 10-checkpoint audit is passed:
outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_head_seed0_summary.json
outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1/metrics/linear_final_receiver_checkpoint_audit.json
Logs: logs/iclr2027_dkp_linear_head_seed0_v1/.

The fixed seed0 FR/KD/SupCon loss-matrix launcher PID 385757 completed all six fresh conditions:
cifar100_5agent20cls_dkp_ablation_fr0_kd0_sc1_ipc10_seed0_v1,
cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc0_ipc10_seed0_v1,
cifar100_5agent20cls_dkp_ablation_fr0_kd1_sc1_ipc10_seed0_v1,
cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc0_ipc10_seed0_v1,
cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1, and
cifar100_5agent20cls_dkp_ablation_fr1_kd1_sc0_ipc10_seed0_v1.
All 30 receiver child jobs exited 0. The strict eight-condition summary is complete_diagnostic and the
30-checkpoint final audit is passed:
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_v1.json
outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_final_checkpoint_audit_v1.json
Logs: logs/iclr2027_dkp_loss_ablation_seed0_v1/.

No DKP receiver or external baseline process is live. The RTX 4090 was idle at 1 MiB / 24,564 MiB and
0% utilization; disk free was about 3.1 GiB. All current DKP outputs, packet sources, configs and logs
remain immutable; deletion still requires a cleanup manifest and user confirmation.

Live process/GPU state was rechecked at 2026-07-27 15:20 UTC; archived PIDs were not treated as live.
The ICLR 2027 DKP seed0 cosine-expert launcher PID 356797 and all five child jobs completed with exit 0.
Expert run: cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.
The strict communication launcher PID 364945 also completed with exit 0. Its fresh derived runs are
cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2 and
cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2. All ten receiver jobs (five per variant) exited 0;
the first-round summary is
outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/first_round_seed0_summary.json.
Logs: logs/iclr2027_dkp_first_round_seed0_v1/experts/ and
logs/iclr2027_dkp_first_round_seed0_v1/communication/.
All five selected full-class IPC10 pool sources passed the strict read-only catalog, packet SHA,
best-snapshot SHA, completed-iteration, class-count and exact-backbone dry-run gate before launch.
All five experts, CE/full packet manifests, sender-logit alignment, prototype initialization and focused
tests passed before receiver launch. The generated summary is diagnostic-only (`formal_result: false`,
`paper_eligible: false`) and contains all five unique receivers for both variants. No external baseline is
running. The empty directory skeleton
`cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v1` is not an experiment run and remains unused; it was
not overwritten or deleted after discovery.

AlexNet historical-recipe recovery completed 2026-07-26T16:51:37Z with best 46.05 @5500;
10,000 iterations, build_communication and packet validation all exited 0. It used f_idx=7,
lr_img=0.005, semantic/MSE, factor=2 and eval/frozen guides.
Run: cifar100_fullclass_dsdm_alexnet_historicale0020_ipc10_seed0.
Log: logs/fullclass_alexnet_historicale0020_seed0/alexnet.log.
The original model-specific e200 queue was interrupted. Its AlexNet branch remains diagnostic-only.
The healthy ResNet-10 branch was cleanly restarted under a new `_recovery` run because optimizer/
iteration state was unavailable. It completed with best 38.86 @ iteration 3000 and validator exit 0.
Its reused guides have exact matching state dicts; source and reserialized mapping hashes are both retained.
ResNet-18 completed its model-specific 10,000-iteration run with best 39.90 @7500; build communication
and packet validation exited 0.
Log: logs/fullclass_resnet_model_specific_recovery_seed0/.
A ResNet-10 PCBN control completed after the pure R10 run. Calibration selected weight 960 over
all 12 normalized BN layers, giving 7.504% of initial total loss. The pair reuses the same guide states,
seed, initialization, f_idx=5, lr_img=0.01 and sparse evaluation schedule. Launcher PID/PGID
301541/301541 exited 0; best is 39.22 @5000 and packet validation passed.
Config: configs/fullclass_dsdm/fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml.
Log: logs/cifar100_r10_pcbn_control_seed0/.
The controlled PCBN weight sweep launcher PID/PGID 327701/327701 is no longer live. The w1300 and
w2100 branches are incomplete (their logs stop near 47% of 10,000 iterations), so their interim values
are diagnostic-only and they must not be treated as completed pool candidates. They are not part of the
current DKP queue and must not be resumed implicitly. Logs: logs/cifar100_r10_pcbn_weight_sweep_seed0/.
Official DSDM source: commit cb12851831e39da6b0169da84598166ad7706e01.
ConvNet-4/AlexNet queue (launcher 194128) completed 2026-07-24T22:39:13Z; all four pure/trajectory
pipelines and packet validators exited 0. Best full-100 accuracies: Conv-4 48.57/46.38 and AlexNet
32.35/36.81 (pure/trajectory).
ConvNet-3 comparison completed before this queue: pure200 46.54 @ iter10000; trajectory 45.47 @ iter5000;
both packet validators passed. Those completed artifacts remain immutable comparison inputs.
All future new ResNet experiments use resnet10_standard/resnet18_standard (CIFAR stem, base width 64),
not the historical compact resnet10/resnet18 (base width 32). Standard ResNet uses only fully pretrained,
independent epoch-200 teacher checkpoints; no single-trajectory checkpoint-pool run is scheduled.
Standard ResNet successor queue (launcher 194963) completed 2026-07-25T18:42:44Z; R10=19.92 and
R18=20.26, both packet validators exited 0. These are standard-width CIFAR ResNets with independent
epoch-200 teachers; no trajectory runs were used.
ConvNet-3 full100-slice vs PAT5-local20 diagnostic (waiter 195996) completed 2026-07-24T13:40:38Z.
Fresh ConvNet-3 results: full100 pure slice 60.55, full100 trajectory slice 61.65, PAT5 local20
guide-20 57.50, PAT5 local20 pure-200 60.15. Controlled pure-200 difference is +0.40 points.
This diagnostic is not in the formal main table.
GPU was idle after the DKP first-round launcher exited; disk free was about 4.9 GiB at completion.
Retain all current DKP outputs, full-class pool sources and logs;
deletion requires a cleanup manifest and user confirmation.
The image-only full-class pool communication gate is materialized and validated at
outputs/cifar100_4agent_25cls_fullclass_pool_seed0_ipc10/. It contains four complete sender slices and
a complete communication manifest, but no sender logits yet; receiver training is intentionally unstarted.
PAT5 seed0: launcher exit 0 at 2026-07-22T11:22:03Z; 5 packets, logits, communication,
            and 5 receiver outputs are present.
PAT10 seed0: original queue stopped on the AlexNet numerical issue; recovery agent 3 completed
             DSDM but exited 1 on the old nested packet-path check. Main valid packets currently
             exist for agents 0, 1, and 3; agent 2 remains diagnostic-only; agents 4-9 are not complete.
Scope: CIFAR-100 original train/test; PAT-style class-disjoint allocation only.
PAT sparse evaluation: 100/500/1000/2000/3000/5000/7500/10000.
Configs: configs/pat_class_split/ and configs/teacher_quality/.
Logs: logs/pat_class_split_seed0/ and logs/teacher_quality_seed0/.
```

Tiny centralized backbone validation is complete; retained outputs are:

```text
outputs/tinyimagenet_backbone_validation_convnet4_seed0/
outputs/tinyimagenet_backbone_validation_resnet18_seed0/
outputs/tinyimagenet_backbone_validation_alexnet_seed0/
outputs/tinyimagenet_backbone_validation_mobilenetv2_seed0/
```

Teacher-quality is no longer active. Do not delete or move its configs, logs, packet sources, or
outputs without a cleanup manifest and user confirmation. Full-class DSDM configs are in
`configs/fullclass_dsdm/`, outputs are in `outputs/cifar100_fullclass_dsdm_*`, and logs/status are in
`logs/fullclass_dsdm_dsdmguidee0200_seed0/`. Do not delete or move PAT configs, logs,
packet sources, or partial outputs without the cleanup manifest and user confirmation. PAT10 downstream logits/communication/
receiver remain intentionally unstarted because the class-split run is incomplete. The earlier
Tiny-ImageNet all-200 queue is not currently running.

`one_resnet_ours_ipc50` 已按用户决定固定为 seed0/2 的 complete 两种子结果。seed1 没有形成
完整 receiver 结果，其 agent0 complete artifact 和 agent1 partial artifact 只作保留记录。

## 保留事项

未经单独确认，不要删除：

```text
configs/main_cifar100_one_resnet_seed1_ipc50.yaml
outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc50/
scripts/run_one_resnet_main_queue.sh
logs/one_resnet_main/
```

不要通过 `--resume` 或重新启动 shell queue 意外恢复 seed1。

## 检查命令

```bash
ps -eo pid,ppid,stat,etime,cmd \
  | rg 'run_one_resnet|run_social_pipeline|run_tiny_r18.*dsdm_pcbn_pair' \
  | rg -v 'rg '

tail -n 80 logs/tinyimagenet_r18_dsdm_pcbn_pair/train_shared_guides.log
tail -n 80 logs/tinyimagenet_r18_dsdm_pcbn_pair/distill_pure.log
tail -n 80 logs/tinyimagenet_r18_dsdm_pcbn_pair/distill_pcbn.log
tail -n 80 logs/tinyimagenet_r18_all200_dsdm_pcbn_pair/train_shared_guides.log
tail -n 80 logs/tinyimagenet_r18_all200_dsdm_pcbn_pair/distill_pure.log
tail -n 80 logs/tinyimagenet_r18_all200_dsdm_pcbn_pair/distill_pcbn.log
tail -n 80 logs/one_resnet_main/master_20260710_120023.log
tail -c 5000 logs/one_resnet_main/seed1_ipc50_distill_agent1.log | tr '\r' '\n' | tail -30
```

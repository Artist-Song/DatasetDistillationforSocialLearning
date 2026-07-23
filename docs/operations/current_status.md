# Current Status

更新时间：`2026-07-23 05:16 UTC`

## 活动队列

```text
Full-class DSDM queue active; ConvNet-3 and ConvNet-4 guide pools are training in parallel.
Launcher PID/PGID: 165235 / 165235. Teacher-quality seed0 calibration completed at 2026-07-23T01:22:44Z.
Scope completed: 5 teacher runs + 14 DSDM guide-maturity candidates; image/logit quality only.
Quality summary: outputs/teacher_quality_seed0_summary/summary.json (`passed=true`, complete=true).
Selected guide: epoch 200 for ConvNet-3/4, AlexNet, standard ResNet-10/18.
No communication or receiver stage was started from this queue.
Launcher: scripts/run_teacher_quality_seed0_parallel.sh.
Status/logs: logs/teacher_quality_seed0/.
Current GPU: full-class guide training active; current disk free: about 12 GiB. Do not move or
delete full-class configs, outputs, packet sources or logs while this queue is active.
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
`logs/fullclass_dsdm_seed0/`. Do not delete or move PAT configs, logs,
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

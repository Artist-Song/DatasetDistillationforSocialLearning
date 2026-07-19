# Current Status

更新时间：`2026-07-19 17:47 +08:00`

## 活动队列

```text
Queue: scripts/run_tiny_r18_all200_dsdm_pcbn_pair.sh
Queue PID/PGID: 17898 / 17898
Current stage: train all-200 ResNet-18 guide pool (10 models x 100 epochs)
Snapshot: guide models 0-2 complete; guide model 3 at about epoch 70/100
Next stage: all-200 pure DSDM and DSDM+PCBN distillation launch concurrently
Scope: Tiny-ImageNet all classes 0-199 packet-quality diagnostic only; no logits/receivers/social
Protocol: IPC10, factor2, niter10000, seed0, validation at 100/500/1000/2000/3000/5000/7500/10000
PCBN: all 20 BN layers, normalized, calibrated weight 10000 (not a completed hyperparameter search)
Data gate: outputs/tinyimagenet_data_validation_20260718/data_integrity.json (passed)
Raw/effective budget: 2,000 raw synthetic images / 8,000 factor-decoded training views
Guide pool: new all-200 ResNet-18 pool, 10 models x 100 epochs, shared byte-identically
Configs: configs/tinyimagenet_r18_all200_dsdm_ipc10_seed0.yaml
         configs/tinyimagenet_r18_all200_dsdm_pcbn_ipc10_seed0.yaml
Logs: logs/tinyimagenet_r18_all200_dsdm_pcbn_pair/
Completed predecessor: 50-class pair exit=0 on 2026-07-19 12:32 +08:00
Completed 50-class best: pure DSDM 32.5460 at iter2000; PCBN 33.5869 at iter5000
Completed result summary: experiments/diagnostics/tinyimagenet_r18_50class_dsdm_pcbn_seed0.json
Stopped queue: scripts/run_one_resnet_main_queue.sh
Stopped stage: seed1 IPC=50 distill agent1 (AlexNet, classes 25-49)
Stopped progress: about 17% / 10,000 iterations
```

Tiny centralized backbone validation is complete; retained outputs are:

```text
outputs/tinyimagenet_backbone_validation_convnet4_seed0/
outputs/tinyimagenet_backbone_validation_resnet18_seed0/
outputs/tinyimagenet_backbone_validation_alexnet_seed0/
outputs/tinyimagenet_backbone_validation_mobilenetv2_seed0/
```

Do not move the active configs, Tiny data root, integrity report, launcher, logs, guide source, or
output directories while PID 17898 is alive. ResNet-34/50 are not queued
because the standard ResNet-18 first-layer result is already sufficient for DSDM validation.

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

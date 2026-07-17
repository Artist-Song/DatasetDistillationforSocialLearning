# Current Status

更新时间：`2026-07-17 16:22 +08:00`

## 活动队列

```text
Queue: scripts/run_one_resnet_main_queue.sh
Current stage: seed1 IPC=50 distill agent1 (AlexNet, classes 25-49)
Progress at snapshot: about 16.0% / 10,000 iterations
Best packet self-evaluation at snapshot: 69.9
```

当前 seed1 IPC=50 尚未形成完整 receiver 结果，因此 `one_resnet_ours_ipc50` 在 registry
中保持 `interim`。运行 `python scripts/build_main_results_table.py` 会自动检测完整 seed，
但在实验结束并审计前不要手工修改 registry 状态。

## 禁止操作

在主队列完成前，不要移动或删除：

```text
configs/main_cifar100_one_resnet_seed1_ipc50.yaml
outputs/cifar100_4agent_25cls_one_resnet_seed1_ipc50/
scripts/run_one_resnet_main_queue.sh
logs/one_resnet_main/
```

不要启动新的重蒸馏任务与当前队列竞争 GPU。

## 检查命令

```bash
ps -eo pid,ppid,stat,etime,cmd \
  | rg 'run_one_resnet|run_social_pipeline' \
  | rg -v 'rg '

tail -n 80 logs/one_resnet_main/master_20260710_120023.log
tail -c 5000 logs/one_resnet_main/seed1_ipc50_distill_agent1.log | tr '\r' '\n' | tail -30
```

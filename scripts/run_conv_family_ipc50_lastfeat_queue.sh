#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DatasetDistillationforSocialLearning
PY=/root/miniconda3/envs/sp/bin/python
CFG=configs/main_cifar100_conv_family_ipc50_lastfeat.yaml
LOG_DIR=logs/conv_family_ipc50_lastfeat
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
{
  echo "[start] $(date '+%F %T %Z')"
  echo "[config] $CFG"
  echo "[verify feature indices]"
  $PY - <<'PY'
from config_adapter import load_config
from agent_data import build_agent_args, get_agent_model_split
cfg_path='configs/main_cifar100_conv_family_ipc50_lastfeat.yaml'
cfg=load_config(cfg_path)
for aid, model in sorted(get_agent_model_split(cfg).items()):
    a=build_agent_args(cfg,cfg_path,aid)
    print(f'agent{aid} model={model} depth={a.depth} width={a.width} f_idx={a.f_idx} idx_from={a.idx_from} idx_to={a.idx_to}')
PY
  echo "[stage] distill agent2 last feature"
  $PY run_social_pipeline.py --config "$CFG" --stage distill_packets --packet-method dsdm --only-agent 2
  echo "[stage] distill agent3 last feature"
  $PY run_social_pipeline.py --config "$CFG" --stage distill_packets --packet-method dsdm --only-agent 3
  echo "[stage] attach logits all agents"
  $PY run_social_pipeline.py --config "$CFG" --stage attach_logits --packet-method dsdm
  echo "[stage] build communication"
  $PY run_social_pipeline.py --config "$CFG" --stage build_communication --packet-method dsdm
  echo "[stage] validate packets"
  $PY validate_packets.py --config "$CFG" --packet-method dsdm
  echo "[stage] train receivers"
  $PY run_social_pipeline.py --config "$CFG" --stage train_receivers --packet-method dsdm --init-mode expert
  echo "[stage] update registry"
  $PY scripts/update_experiment_registry.py
  echo "[done] $(date '+%F %T %Z')"
} 2>&1 | tee "$LOG"

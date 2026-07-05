#!/usr/bin/env bash
set -u
ROOT="/root/autodl-tmp/DatasetDistillationforSocialLearning"
cd "$ROOT" || exit 1
OUT="logs/monitoring/conv_family_ipc50_lastfeat_hourly_$(date +%Y%m%d_%H%M%S).log"
LATEST="logs/monitoring/conv_family_ipc50_lastfeat_hourly_latest.log"
mkdir -p logs/monitoring
printf '%s\n' "$OUT" > "$LATEST"
while true; do
  {
    echo "========================================================================"
    echo "[monitor] $(date '+%F %T %Z')"
    echo "## processes"
    ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main_cifar100_conv_family_ipc50_lastfeat|run_conv_family_ipc50_lastfeat|run_social_pipeline.py|validate_packets.py|update_experiment_registry.py' | grep -v 'grep -E' || true
    echo
    echo "## gpu"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv || true
    echo
    echo "## packet files"
    for a in 0 1 2 3; do
      printf 'agent_%s packet: ' "$a"
      ls -lh "outputs/cifar100_4agent_25cls_conv_family_ipc50_lastfeat/agents/agent_${a}/packets/dsdm_packet.pt" 2>/dev/null || echo missing
    done
    echo
    echo "## parsed progress"
    /root/miniconda3/envs/sp/bin/python - <<'PY' || true
import re
from pathlib import Path
logs=sorted(Path('logs/conv_family_ipc50_lastfeat').glob('run_*.log'))
if not logs:
    print('missing log')
    raise SystemExit
p=logs[-1]
print(f'log={p}')
text=p.read_text(errors='ignore').replace('\r','\n')
agent_re=re.compile(r'\[distill_packets\] agent=(\d+) model=([^ ]+) classes=')
positions=[]
for m in agent_re.finditer(text):
    positions.append((m.start(), int(m.group(1)), m.group(2)))
positions.append((len(text), None, None))
for (start,a,model),(end,_,__) in zip(positions, positions[1:]):
    block=text[start:end]
    bests=[float(x) for x in re.findall(r'Best Result:\s*([0-9.]+)', block)]
    progress=re.findall(r'\[DSDM condense\].*?(\d+)/10000\s+([0-9.]+)%.*?eta ([0-9:]+).*?best=([0-9.\-]+)', block)
    last=progress[-1] if progress else None
    print(f'agent{a} model={model} best_max={max(bests) if bests else None} best_last={bests[-1] if bests else None} eval_count={len(bests)} last_progress={last}')
PY
    echo
    echo "## latest log tail"
    latest=$(ls -1t logs/conv_family_ipc50_lastfeat/run_*.log 2>/dev/null | head -1 || true)
    [ -n "$latest" ] && tail -n 45 "$latest" || true
  } >> "$OUT" 2>&1
  sleep 3600
done

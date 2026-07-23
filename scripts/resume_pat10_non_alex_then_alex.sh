#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG="configs/pat_class_split/main_cifar100_pat10agent_seed0_ipc10.yaml"
LOG_ROOT="$ROOT/logs/pat_class_split_seed0"
LOG_PATH="$LOG_ROOT/pat10agent_seed0_ipc10_numerical_fix.log"
STATUS_PATH="$LOG_ROOT/pat10agent_seed0_ipc10_numerical_fix_status.tsv"

mkdir -p "$LOG_ROOT"
cd "$ROOT"
export OMP_NUM_THREADS=4

printf 'phase\tagent\tstate\ttimestamp_utc\texit_code\n' > "$STATUS_PATH"

run_agent() {
  local phase="$1"
  local agent_id="$2"
  local started_at
  local finished_at
  local exit_code

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\tstarted\t%s\t-\n' "$phase" "$agent_id" "$started_at" >> "$STATUS_PATH"
  set +e
  "$PYTHON" -u run_social_pipeline.py \
    --config "$CONFIG" \
    --stage distill_packets \
    --packet-method dsdm \
    --only-agent "$agent_id" \
    >> "$LOG_PATH" 2>&1
  exit_code=$?
  set -e
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\tfinished\t%s\t%s\n' \
    "$phase" "$agent_id" "$finished_at" "$exit_code" >> "$STATUS_PATH"
  return "$exit_code"
}

verify_alexnet_quality() {
  local agent_id="$1"
  local minimum_acc="$2"
  "$PYTHON" - "$agent_id" "$minimum_acc" <<'PY'
import json
import sys
from pathlib import Path

agent_id = int(sys.argv[1])
minimum_acc = float(sys.argv[2])
run_root = Path("outputs/cifar100_pat10agent_10cls_ours_seed0_ipc10")
manifest_path = run_root / "agents" / f"agent_{agent_id}" / "synthetic" / "best_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
best_acc = float(manifest["best_acc"])
if best_acc < minimum_acc:
    raise SystemExit(
        f"AlexNet quality gate failed: agent={agent_id} best_acc={best_acc:.4f} < {minimum_acc:.4f}"
    )
print(f"[quality gate] AlexNet agent={agent_id} best_acc={best_acc:.4f} >= {minimum_acc:.4f}")
PY
}

printf '[queue] non-AlexNet agents first: 3 4 5 6 8 9\n' >> "$LOG_PATH"
for agent_id in 3 4 5 6 8 9; do
  run_agent non_alex "$agent_id"
done

printf '[queue] stabilized AlexNet agents last: 2 7\n' >> "$LOG_PATH"
run_agent alexnet 2
verify_alexnet_quality 2 64.9 | tee -a "$LOG_PATH"
run_agent alexnet 7

printf '[queue] packet distillation complete; downstream stages intentionally not started\n' >> "$LOG_PATH"

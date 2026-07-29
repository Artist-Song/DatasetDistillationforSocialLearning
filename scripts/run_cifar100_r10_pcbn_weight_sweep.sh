#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/miniconda3/envs/sp/bin/python"
CONFIG_ROOT="$ROOT/configs/fullclass_dsdm"
LOG_ROOT="$ROOT/logs/cifar100_r10_pcbn_weight_sweep_seed0"
STATUS_FILE="$LOG_ROOT/status.tsv"
DEPENDENCY_PIDS="${DEPENDENCY_PIDS:-301541 285660}"

cd "$ROOT"
mkdir -p "$LOG_ROOT"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=4; fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export MKL_NUM_THREADS=4; fi

printf 'phase\tweight\tstate\ttimestamp_utc\texit_code\tdetail\n' > "$STATUS_FILE"
record_status() {
  (
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" "$5" >&9
  ) 9>>"$STATUS_FILE"
}

"$PYTHON" scripts/prepare_cifar100_r10_pcbn_weight_sweep.py > "$LOG_ROOT/setup.log" 2>&1 || exit $?
for weight in 1300 2100 3400; do
  "$PYTHON" scripts/validate_cifar100_r10_pcbn_sweep.py \
    --config "configs/fullclass_dsdm/fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w${weight}.yaml" \
    --weight "$weight" > "$LOG_ROOT/preflight_w${weight}.json" 2>&1 || exit $?
done

for pid in $DEPENDENCY_PIDS; do
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
    record_status dependency "$pid" waiting - active_queue
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    record_status dependency "$pid" finished 0 exited
  fi
done

available_kib="$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')"
required_kib="$((2 * 1024 * 1024))"
if [[ "$available_kib" -lt "$required_kib" ]]; then
  record_status preflight disk failed 2 less_than_2_GiB_free
  exit 2
fi

"$PYTHON" - <<'PY'
from pathlib import Path
import torch

root = Path("outputs/cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w960")
packet = torch.load(root / "agents/agent_0/packets/dsdm_packet.pt", map_location="cpu", weights_only=False)
meta = packet.get("meta", {})
if not meta.get("condense_complete") or int(meta.get("completed_iterations", 0)) != 10000:
    raise RuntimeError("w960 reference is not complete")
PY

run_weight() {
  local weight="$1" config log stage code
  config="$CONFIG_ROOT/fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w${weight}.yaml"
  log="$LOG_ROOT/w${weight}.log"
  : > "$log"
  record_status pipeline "$weight" started - -
  for stage in distill_packets build_communication; do
    record_status "$stage" "$weight" started - -
    "$PYTHON" -u run_social_pipeline.py --config "$config" \
      --stage "$stage" --packet-method dsdm --only-agent 0 --resume >> "$log" 2>&1
    code=$?
    record_status "$stage" "$weight" finished "$code" -
    if [[ "$code" -ne 0 ]]; then
      record_status pipeline "$weight" finished "$code" -
      return "$code"
    fi
  done
  record_status validate "$weight" started - -
  "$PYTHON" -u validate_packets.py --config "$config" --packet-method dsdm >> "$log" 2>&1
  code=$?
  record_status validate "$weight" finished "$code" -
  record_status pipeline "$weight" finished "$code" -
  return "$code"
}

run_weight 1300 &
pid1300=$!
run_weight 2100 &
pid2100=$!
failures=0
wait "$pid1300" || failures=1
wait "$pid2100" || failures=1
if [[ "$failures" -ne 0 ]]; then exit 1; fi

decision="$($PYTHON - <<'PY'
import json
from pathlib import Path

root = Path("outputs")
prefix = "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w"
scores = {}
for weight in (960, 1300, 2100):
    path = root / f"{prefix}{weight}" / "agents/agent_0/synthetic/best_manifest.json"
    scores[weight] = float(json.loads(path.read_text(encoding="utf-8"))["best_acc"])
run_upper = scores[2100] >= max(scores[960], scores[1300]) + 0.10
print("run_w3400" if run_upper else "stop_at_w2100")
PY
)"
record_status boundary 3400 decision 0 "$decision"
if [[ "$decision" == "run_w3400" ]]; then
  run_weight 3400 || exit $?
fi

"$PYTHON" scripts/summarize_cifar100_r10_pcbn_sweep.py > "$LOG_ROOT/summary.log" 2>&1
record_status sweep all finished 0 "$decision"

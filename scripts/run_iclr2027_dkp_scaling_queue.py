#!/usr/bin/env python3
"""Run preregistered DKP scaling experiments on one fully utilized GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_receiver_dir  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_scaling import (  # noqa: E402
    DEFAULT_SEEDS,
    SUPPORTED_AGENT_COUNTS,
    config_path,
)


PYTHON = "/root/miniconda3/envs/sp/bin/python"
LOG_ROOT = ROOT / "logs/iclr2027_dkp_scaling"
STATUS_PATH = LOG_ROOT / "queue_status.json"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_dir(config_path_value):
    config = load_config(config_path_value)
    output_root = Path(config["project"].get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / config["project"]["run_name"]


def _receiver_checkpoint_path(full_config_path, receiver_id):
    config = load_config(full_config_path)
    args = build_dsdm_args_from_config(config, config_path=full_config_path)
    return (
        get_receiver_dir(args, receiver_id)
        / "checkpoints/dkp_sl_v1_full/after_social.pt"
    )


def _environment():
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _gpu_processes():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _write_status(status):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(status, STATUS_PATH)


def _run_logged(name, command, log_dir, status):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    status["current_stage"] = name
    status["updated_at_utc"] = _now()
    _write_status(status)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now()}] command={json.dumps(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.write(f"[{_now()}] exit={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"stage {name} failed with exit {result.returncode}; log={log_path}")
    return log_path


def _run_parallel(stage, jobs, max_parallel, log_dir, status):
    pending = deque(jobs)
    active = {}
    failures = []
    log_dir.mkdir(parents=True, exist_ok=True)
    status["current_stage"] = stage
    status["updated_at_utc"] = _now()
    _write_status(status)
    while pending or active:
        while pending and len(active) < int(max_parallel):
            name, command = pending.popleft()
            log_path = log_dir / f"{name}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"[{_now()}] command={json.dumps(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[process.pid] = (name, process, handle, log_path)
            print(f"[{_now()}] {stage}: launched {name} pid={process.pid}", flush=True)
        time.sleep(1)
        for pid, (name, process, handle, log_path) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.write(f"[{_now()}] exit={return_code}\n")
            handle.close()
            del active[pid]
            print(f"[{_now()}] {stage}: completed {name} pid={pid} exit={return_code}", flush=True)
            if return_code != 0:
                failures.append((name, return_code, log_path))
        if failures:
            for _pid, (_name, process, handle, _path) in list(active.items()):
                process.terminate()
                process.wait(timeout=30)
                handle.close()
            break
    if failures:
        details = ", ".join(f"{name}:exit={code}:log={path}" for name, code, path in failures)
        raise RuntimeError(f"parallel stage {stage} failed: {details}")


def _receiver_complete(full_config_path, receiver_id):
    run_dir = _run_dir(full_config_path)
    results_path = run_dir / "metrics/social_results.csv"
    checkpoint = _receiver_checkpoint_path(full_config_path, receiver_id)
    if not results_path.is_file() and not checkpoint.exists():
        return False
    if not results_path.is_file() or not checkpoint.is_file():
        raise RuntimeError(
            f"receiver {receiver_id} has a partial final artifact; refusing implicit overwrite"
        )
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row.get("receiver_agent", -1)) == int(receiver_id)
            and row.get("method") == "DKP_SL"
            and row.get("dkp_variant") == "full"
        ]
    if not rows:
        raise RuntimeError(
            f"receiver {receiver_id} checkpoint exists without a complete result row"
        )
    latest = rows[-1]
    if int(latest.get("optimizer_steps", -1)) != 3780:
        raise RuntimeError(f"receiver {receiver_id} completed with the wrong step count")
    if latest.get("after_social_checkpoint_sha256") != file_sha256(checkpoint):
        raise RuntimeError(f"receiver {receiver_id} checkpoint/result SHA mismatch")
    return True


def _task(seed, agent_count, max_parallel, resume, status):
    expert_config = config_path(seed, agent_count, "expert")
    full_config = config_path(seed, agent_count, "dkp_full")
    if not expert_config.is_file() or not full_config.is_file():
        raise FileNotFoundError("scaling configs are missing; run prepare_iclr2027_dkp_scaling.py --write")
    expert_run = _run_dir(expert_config)
    full_run = _run_dir(full_config)
    if not resume and (expert_run.exists() or full_run.exists()):
        raise FileExistsError(
            f"refusing to reuse scaling outputs without --resume: {expert_run} {full_run}"
        )
    task_name = f"{agent_count}agent_seed{seed}"
    log_dir = LOG_ROOT / task_name
    status.update(
        {
            "current_task": task_name,
            "seed": int(seed),
            "agent_count": int(agent_count),
            "updated_at_utc": _now(),
        }
    )
    _write_status(status)

    expert_jobs = []
    for agent_id in range(agent_count):
        command = [
            PYTHON,
            "scripts/train_pool_communication_experts.py",
            "--config",
            str(expert_config.relative_to(ROOT)),
            "--only-agent",
            str(agent_id),
        ]
        if resume:
            command.append("--resume")
        expert_jobs.append((f"expert_agent_{agent_id}", command))
    _run_parallel("experts", expert_jobs, max_parallel, log_dir / "experts", status)

    expert_report = expert_run / "metrics/scaling_cosine_expert_preflight.json"
    _run_logged(
        "expert_preflight",
        [
            PYTHON,
            "scripts/validate_iclr2027_cosine_experts.py",
            "--config",
            str(expert_config.relative_to(ROOT)),
            "--device",
            "cuda",
            "--report-json",
            str(expert_report.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )

    reuse_command = [
        PYTHON,
        "scripts/reuse_iclr2027_experts.py",
        "--source-config",
        str(expert_config.relative_to(ROOT)),
        "--target-config",
        str(full_config.relative_to(ROOT)),
    ]
    if resume:
        reuse_command.append("--resume")
    _run_logged("reuse_experts", reuse_command, log_dir, status)

    materialize_command = [
        PYTHON,
        "scripts/materialize_fullclass_pool_packets.py",
        "--config",
        str(full_config.relative_to(ROOT)),
        "--pool-catalog",
        "configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml",
    ]
    if resume:
        materialize_command.append("--resume")
    _run_logged("materialize_packets", materialize_command, log_dir, status)

    logit_jobs = []
    for agent_id in range(agent_count):
        command = [
            PYTHON,
            "run_social_pipeline.py",
            "--config",
            str(full_config.relative_to(ROOT)),
            "--stage",
            "attach_logits",
            "--packet-method",
            "dsdm",
            "--only-agent",
            str(agent_id),
        ]
        if resume:
            command.append("--resume")
        logit_jobs.append((f"logits_agent_{agent_id}", command))
    _run_parallel("attach_logits", logit_jobs, max_parallel, log_dir / "logits", status)

    _run_logged(
        "build_communication",
        [
            PYTHON,
            "run_social_pipeline.py",
            "--config",
            str(full_config.relative_to(ROOT)),
            "--stage",
            "build_communication",
            "--packet-method",
            "dsdm",
        ],
        log_dir,
        status,
    )
    _run_logged(
        "validate_packets",
        [
            PYTHON,
            "validate_packets.py",
            "--config",
            str(full_config.relative_to(ROOT)),
            "--packet-method",
            "dsdm",
        ],
        log_dir,
        status,
    )
    preflight_report = full_run / "metrics/scaling_communication_preflight.json"
    _run_logged(
        "communication_preflight",
        [
            PYTHON,
            "scripts/validate_iclr2027_dkp_scaling.py",
            "--expert-config",
            str(expert_config.relative_to(ROOT)),
            "--full-config",
            str(full_config.relative_to(ROOT)),
            "--device",
            "cuda",
            "--report-json",
            str(preflight_report.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )

    receiver_jobs = []
    for receiver_id in range(agent_count):
        if resume and _receiver_complete(full_config, receiver_id):
            print(f"[{_now()}] receivers: reuse validated receiver={receiver_id}", flush=True)
            continue
        receiver_jobs.append(
            (
                f"receiver_{receiver_id}",
                [
                    PYTHON,
                    "run_social_pipeline.py",
                    "--config",
                    str(full_config.relative_to(ROOT)),
                    "--stage",
                    "train_receivers",
                    "--packet-method",
                    "dsdm",
                    "--only-receiver",
                    str(receiver_id),
                ],
            )
        )
    _run_parallel("receivers", receiver_jobs, max_parallel, log_dir / "receivers", status)
    _run_logged(
        "strict_summary",
        [
            PYTHON,
            "scripts/summarize_iclr2027_dkp_scaling.py",
            "--config",
            str(full_config.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )
    status.setdefault("completed_tasks", []).append(task_name)
    status["updated_at_utc"] = _now()
    _write_status(status)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--agent-counts", nargs="+", type=int, default=list(SUPPORTED_AGENT_COUNTS))
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    if cli.max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    for agent_count in cli.agent_counts:
        if agent_count not in SUPPORTED_AGENT_COUNTS:
            raise ValueError(f"unsupported agent count: {agent_count}")
    active_gpu = _gpu_processes()
    if active_gpu:
        raise RuntimeError(f"refusing to launch while GPU compute processes are active: {active_gpu}")
    if STATUS_PATH.exists() and not cli.resume:
        raise FileExistsError(f"queue status already exists; use --resume after inspection: {STATUS_PATH}")
    status = {
        "status": "running",
        "started_at_utc": _now(),
        "updated_at_utc": _now(),
        "seeds": [int(value) for value in cli.seeds],
        "agent_counts": [int(value) for value in cli.agent_counts],
        "max_parallel": int(cli.max_parallel),
        "gpu": "NVIDIA GeForce RTX 4090",
        "completed_tasks": [],
    }
    if cli.resume and STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        status["started_at_utc"] = previous.get("started_at_utc", status["started_at_utc"])
        status["completed_tasks"] = list(previous.get("completed_tasks", []))
    _write_status(status)
    try:
        for agent_count in cli.agent_counts:
            for seed in cli.seeds:
                task_name = f"{agent_count}agent_seed{seed}"
                if task_name in status["completed_tasks"]:
                    continue
                _task(seed, agent_count, cli.max_parallel, cli.resume, status)
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "error": str(error),
                "updated_at_utc": _now(),
            }
        )
        _write_status(status)
        raise
    status.update(
        {
            "status": "complete",
            "current_task": None,
            "current_stage": None,
            "updated_at_utc": _now(),
        }
    )
    _write_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

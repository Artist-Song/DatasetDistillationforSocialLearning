#!/usr/bin/env python3
"""Run the preregistered internal and external CIFAR-100 baseline matrix on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split, get_receiver_dir  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_scaling import config_path  # noqa: E402


PYTHON = "/root/miniconda3/envs/sp/bin/python"
LOG_ROOT = ROOT / "logs/iclr2027_baseline_matrix"
STATUS_PATH = LOG_ROOT / "queue_status.json"
DESA_ROOT = ROOT / "external_baselines/outputs/desa_scaling"
MASC_ROOT = ROOT / "external_baselines/outputs/masc_scaling"
FEDRE_ROOT = ROOT / "external_baselines/outputs/fedre_scaling"
FEDRE_DATA_ROOT = ROOT / "external_baselines/outputs/fedre_scaling_datasets"
FEDRE_REPO_DATA_ROOT = ROOT / "external_baselines/repos/FedRE/HtFLlib/dataset"
MIN_FREE_BYTES = 8 * 1024**3
VARIANT = "dkp_sl_v1_ablation_fr1_kd0_sc1"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _environment() -> dict[str, str]:
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


def _write_status(status: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status["updated_at_utc"] = _now()
    atomic_write_json(status, STATUS_PATH)


def _gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _storage_gate() -> None:
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"storage gate failed: free={free / 1024**3:.2f} GiB < {MIN_FREE_BYTES / 1024**3:.0f} GiB"
        )


def _run_logged(name: str, command: list[str], log_dir: Path, status: dict) -> Path:
    _storage_gate()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    status["current_stage"] = name
    status["command"] = command
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


def _run_parallel(
    stage: str,
    jobs: list[tuple[str, list[str]]],
    max_parallel: int,
    log_dir: Path,
    status: dict,
) -> None:
    if not jobs:
        return
    _storage_gate()
    pending = deque(jobs)
    active: dict[int, tuple[str, subprocess.Popen, object, Path]] = {}
    failures = []
    log_dir.mkdir(parents=True, exist_ok=True)
    status["current_stage"] = stage
    status["parallel_jobs"] = [name for name, _command in jobs]
    _write_status(status)
    while pending or active:
        while pending and len(active) < int(max_parallel):
            name, command = pending.popleft()
            path = log_dir / f"{name}.log"
            handle = path.open("a", encoding="utf-8")
            handle.write(f"[{_now()}] command={json.dumps(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[process.pid] = (name, process, handle, path)
            print(f"[{_now()}] {stage}: launched {name} pid={process.pid}", flush=True)
        time.sleep(1)
        for pid, (name, process, handle, path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.write(f"[{_now()}] exit={code}\n")
            handle.close()
            del active[pid]
            print(f"[{_now()}] {stage}: completed {name} pid={pid} exit={code}", flush=True)
            if code != 0:
                failures.append((name, code, path))
        if failures:
            for _pid, (_name, process, handle, _path) in list(active.items()):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                handle.close()
            break
    if failures:
        details = ", ".join(f"{name}:exit={code}:log={path}" for name, code, path in failures)
        raise RuntimeError(f"parallel stage {stage} failed: {details}")


def _run_dir(config: dict) -> Path:
    output_root = Path(config["project"].get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / config["project"]["run_name"]


def _internal_receiver_complete(config_path_value: Path, method: str, receiver: int) -> bool:
    config = load_config(config_path_value)
    args = build_dsdm_args_from_config(config, config_path=config_path_value)
    checkpoint = get_receiver_dir(args, receiver) / "checkpoints" / VARIANT / "after_social.pt"
    results = _run_dir(config) / "metrics/social_results.csv"
    if not checkpoint.exists() and not results.exists():
        return False
    if not checkpoint.is_file() or not results.is_file():
        raise RuntimeError(f"internal receiver {receiver} has partial final artifacts")
    with results.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("packet_method") == method
            and int(row.get("receiver_agent", -1)) == receiver
            and row.get("dkp_variant") == "ablation_fr1_kd0_sc1"
        ]
    if not rows:
        raise RuntimeError(f"receiver {receiver} checkpoint exists without a matching result row")
    row = rows[-1]
    if int(row.get("optimizer_steps", -1)) != 3780:
        raise RuntimeError(f"receiver {receiver} has the wrong optimizer step count")
    if row.get("after_social_checkpoint_sha256") != file_sha256(checkpoint):
        raise RuntimeError(f"receiver {receiver} result/checkpoint SHA differs")
    return True


def _audit_internal(
    expert_config: Path,
    baseline_config: Path,
    method: str,
    phase: str,
    report_path: Path,
    log_dir: Path,
    status: dict,
) -> None:
    _run_logged(
        f"audit_{phase}",
        [
            PYTHON,
            "scripts/validate_iclr2027_hard_label_baseline.py",
            "--expert-config",
            str(expert_config.relative_to(ROOT)),
            "--baseline-config",
            str(baseline_config.relative_to(ROOT)),
            "--method",
            method,
            "--phase",
            phase,
            "--report-json",
            str(report_path.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )


def _internal_task(method: str, agent_count: int, seed: int, max_parallel: int, status: dict) -> None:
    expert_config = config_path(seed, agent_count, "expert")
    baseline_config = config_path(seed, agent_count, method)
    config = load_config(baseline_config)
    class_split = get_agent_class_split(config)
    run_dir = _run_dir(config)
    task_id = f"internal:{method}:{agent_count}agent:seed{seed}"
    log_dir = LOG_ROOT / "internal" / method / f"{agent_count}agent_seed{seed}"
    status.update({"current_task": task_id, "method": method, "agent_count": agent_count, "seed": seed})
    _write_status(status)

    if all(_internal_receiver_complete(baseline_config, method, receiver) for receiver in class_split):
        _audit_internal(
            expert_config,
            baseline_config,
            method,
            "complete",
            run_dir / "metrics/hard_label_complete_audit.json",
            log_dir,
            status,
        )
        return

    _run_logged(
        "reuse_experts",
        [
            PYTHON,
            "scripts/reuse_iclr2027_experts.py",
            "--source-config",
            str(expert_config.relative_to(ROOT)),
            "--target-config",
            str(baseline_config.relative_to(ROOT)),
            "--resume",
        ],
        log_dir,
        status,
    )

    packet_jobs = []
    for agent in class_split:
        packet = _run_dir(config) / f"agents/agent_{agent}/packets/{method}_packet.pt"
        if packet.is_file():
            continue
        packet_jobs.append(
            (
                f"packet_agent_{agent}",
                [
                    PYTHON,
                    "run_social_pipeline.py",
                    "--config",
                    str(baseline_config.relative_to(ROOT)),
                    "--stage",
                    "build_selection_packets",
                    "--packet-method",
                    method,
                    "--only-agent",
                    str(agent),
                ],
            )
        )
    _run_parallel("build_packets", packet_jobs, max_parallel, log_dir / "packets", status)

    _run_logged(
        "build_communication",
        [
            PYTHON,
            "run_social_pipeline.py",
            "--config",
            str(baseline_config.relative_to(ROOT)),
            "--stage",
            "build_communication",
            "--packet-method",
            method,
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
            str(baseline_config.relative_to(ROOT)),
            "--packet-method",
            method,
        ],
        log_dir,
        status,
    )
    _audit_internal(
        expert_config,
        baseline_config,
        method,
        "preflight",
        run_dir / "metrics/hard_label_preflight.json",
        log_dir,
        status,
    )

    receiver_jobs = []
    for receiver in class_split:
        if _internal_receiver_complete(baseline_config, method, receiver):
            continue
        receiver_jobs.append(
            (
                f"receiver_{receiver}",
                [
                    PYTHON,
                    "run_social_pipeline.py",
                    "--config",
                    str(baseline_config.relative_to(ROOT)),
                    "--stage",
                    "train_receivers",
                    "--packet-method",
                    method,
                    "--only-receiver",
                    str(receiver),
                ],
            )
        )
    _run_parallel("receivers", receiver_jobs, max_parallel, log_dir / "receivers", status)
    _audit_internal(
        expert_config,
        baseline_config,
        method,
        "complete",
        run_dir / "metrics/hard_label_complete_audit.json",
        log_dir,
        status,
    )


def _external_config(agent_count: int, seed: int) -> Path:
    return config_path(seed, agent_count, "expert")


def _external_validate(
    method: str,
    config: Path,
    output_root: Path,
    report: Path,
    log_dir: Path,
    status: dict,
) -> None:
    _run_logged(
        "complete_audit",
        [
            PYTHON,
            "baseline_adapters/validate_external_scaling.py",
            "--method",
            method,
            "--config",
            str(config.relative_to(ROOT)),
            "--output-root",
            str(output_root.relative_to(ROOT)),
            "--report-json",
            str(report.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )


def _desa_task(agent_count: int, seed: int, status: dict) -> None:
    config = _external_config(agent_count, seed)
    project = load_config(config)
    run_dir = DESA_ROOT / project["project"]["run_name"]
    log_dir = LOG_ROOT / "external" / "desa" / f"{agent_count}agent_seed{seed}"
    task_id = f"external:desa:{agent_count}agent:seed{seed}"
    status.update({"current_task": task_id, "method": "desa", "agent_count": agent_count, "seed": seed})
    _write_status(status)
    report = run_dir / "complete_audit.json"
    if (run_dir / "social_results.csv").is_file():
        _external_validate("desa", config, DESA_ROOT, report, log_dir, status)
        return
    _run_logged(
        "train",
        [
            PYTHON,
            "baseline_adapters/run_desa_cil.py",
            "--config",
            str(config.relative_to(ROOT)),
            "--ipc",
            "10",
            "--output-root",
            str(DESA_ROOT.relative_to(ROOT)),
            "--resume",
        ],
        log_dir,
        status,
    )
    _external_validate("desa", config, DESA_ROOT, report, log_dir, status)


def _masc_task(agent_count: int, seed: int, status: dict) -> None:
    config = _external_config(agent_count, seed)
    project = load_config(config)
    run_dir = MASC_ROOT / f"{project['project']['run_name']}_masc_homogeneous_10"
    log_dir = LOG_ROOT / "external" / "masc" / f"{agent_count}agent_seed{seed}"
    task_id = f"external:masc:{agent_count}agent:seed{seed}"
    status.update({"current_task": task_id, "method": "masc", "agent_count": agent_count, "seed": seed})
    _write_status(status)
    report = run_dir / "complete_audit.json"
    if (run_dir / "social_results.csv").is_file():
        _external_validate("masc", config, MASC_ROOT, report, log_dir, status)
        return
    _run_logged(
        "train",
        [
            PYTHON,
            "baseline_adapters/run_masc_complete.py",
            "--config",
            str(config.relative_to(ROOT)),
            "--seed",
            str(seed),
            "--budget",
            "10",
            "--output-root",
            str(MASC_ROOT.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )
    _external_validate("masc", config, MASC_ROOT, report, log_dir, status)


def _fedre_task(agent_count: int, seed: int, status: dict) -> None:
    config = _external_config(agent_count, seed)
    project = load_config(config)
    run_dir = FEDRE_ROOT / f"{project['project']['run_name']}_fedre"
    log_dir = LOG_ROOT / "external" / "fedre" / f"{agent_count}agent_seed{seed}"
    task_id = f"external:fedre:{agent_count}agent:seed{seed}"
    status.update({"current_task": task_id, "method": "fedre", "agent_count": agent_count, "seed": seed})
    _write_status(status)
    validation_command = [
        PYTHON,
        "baseline_adapters/validate_fedre_reproduction.py",
        "--run-dir",
        str(run_dir.relative_to(ROOT)),
        "--expected-clients",
        str(agent_count),
        "--project-config",
        str(config.relative_to(ROOT)),
    ]
    if (run_dir / "status.json").is_file():
        existing_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if existing_status.get("state") == "complete":
            _run_logged("complete_audit", validation_command, log_dir, status)
            return
        raise RuntimeError(
            f"FedRE has a non-complete non-resumable run directory: {run_dir}"
        )

    _run_logged(
        "prepare_dataset",
        [
            PYTHON,
            "baseline_adapters/prepare_fedre_reproduction.py",
            "--project-config",
            str(config.relative_to(ROOT)),
            "--output-root",
            str(FEDRE_DATA_ROOT.relative_to(ROOT)),
            "--fedre-dataset-root",
            str(FEDRE_REPO_DATA_ROOT.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )
    _run_logged(
        "train",
        [
            PYTHON,
            "baseline_adapters/run_fedre_reproduction.py",
            "--project-config",
            str(config.relative_to(ROOT)),
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir.relative_to(ROOT)),
        ],
        log_dir,
        status,
    )
    _run_logged("complete_audit", validation_command, log_dir, status)


def build_matrix() -> list[tuple[str, str, int, int]]:
    tasks = []
    for method in ("heuristic", "fast", "full_real"):
        for seed in (1, 2, 3):
            tasks.append(("internal", method, 5, seed))
    for agent_count in (10, 20):
        for method in ("heuristic", "fast"):
            for seed in (1, 2, 3):
                tasks.append(("internal", method, agent_count, seed))
    for agent_count in (5, 10, 20):
        for method in ("desa", "masc", "fedre"):
            for seed in (1, 2, 3):
                tasks.append(("external", method, agent_count, seed))
    return tasks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    cli = parse_args(argv)
    if cli.max_parallel <= 0 or cli.max_parallel > 5:
        raise ValueError("max-parallel must be in [1, 5] for the single RTX 4090")
    matrix = build_matrix()
    if cli.dry_run:
        print(json.dumps({"task_count": len(matrix), "tasks": matrix}, indent=2))
        return 0
    active = _gpu_processes()
    if active:
        raise RuntimeError(f"refusing to launch while GPU compute processes are active: {active}")
    _storage_gate()
    if STATUS_PATH.exists() and not cli.resume:
        raise FileExistsError(f"queue status exists; inspect it and use --resume: {STATUS_PATH}")
    status = {
        "status": "running",
        "started_at_utc": _now(),
        "matrix_task_count": len(matrix),
        "max_parallel": cli.max_parallel,
        "gpu": "NVIDIA GeForce RTX 4090",
        "completed_tasks": [],
    }
    if cli.resume and STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        status["started_at_utc"] = previous.get("started_at_utc", status["started_at_utc"])
        status["completed_tasks"] = list(previous.get("completed_tasks", []))
    _write_status(status)
    try:
        for family, method, agent_count, seed in matrix:
            task_id = f"{family}:{method}:{agent_count}agent:seed{seed}"
            if task_id in status["completed_tasks"]:
                continue
            if family == "internal":
                _internal_task(method, agent_count, seed, cli.max_parallel, status)
            elif method == "desa":
                _desa_task(agent_count, seed, status)
            elif method == "masc":
                _masc_task(agent_count, seed, status)
            else:
                _fedre_task(agent_count, seed, status)
            status["completed_tasks"].append(task_id)
            status["completed_task_count"] = len(status["completed_tasks"])
            _write_status(status)
    except Exception as error:
        status.update({"status": "failed", "error": str(error)})
        _write_status(status)
        raise
    status.update({"status": "complete", "current_task": None, "current_stage": None, "command": None})
    _write_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

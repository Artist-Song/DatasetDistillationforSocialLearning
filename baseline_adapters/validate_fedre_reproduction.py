from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one completed official FedRE reproduction run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-clients", required=True, type=int)
    parser.add_argument("--expected-round", default=100, type=int)
    parser.add_argument("--expected-paper-accuracy", type=float)
    parser.add_argument("--paper-accuracy-tolerance", type=float, default=10.0)
    parser.add_argument("--minimum-paper-accuracy", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    status = _read_json(run_dir / "status.json")
    summary = _read_json(run_dir / "summary.json")
    protocol = _read_json(run_dir / "resolved_protocol.json")
    local_rows = _read_csv(run_dir / "paper_round_local_metrics.csv")
    social_rows = _read_csv(run_dir / "global_social_metrics.csv")

    _require(status["state"] == "complete", "run status is not complete")
    _require(summary["status"] == "complete", "summary status is not complete")
    _require(int(summary["paper_round"]) == args.expected_round, "unexpected paper round")
    _require(len(local_rows) == args.expected_clients, "wrong number of local client rows")
    _require(len(social_rows) == args.expected_clients, "wrong number of social metric rows")
    _require(
        protocol["training_implementation"] == "official flcore.servers.serverre.FedRE.train",
        "training was not delegated to official FedRE.train",
    )
    _require(protocol["fedre_commit"] == "9e2164343ee0d76be60afe46e2b986b9f77ef1cb", "unexpected FedRE commit")

    paper_accuracy = float(summary["paper_local_accuracy"])
    official_accuracy = float(summary["official_recorded_local_accuracy"])
    _require(math.isfinite(paper_accuracy), "paper accuracy is not finite")
    _require(abs(paper_accuracy - official_accuracy) < 1e-9, "official and recomputed local metrics differ")
    if args.expected_paper_accuracy is not None:
        _require(
            abs(paper_accuracy - args.expected_paper_accuracy) <= args.paper_accuracy_tolerance,
            f"paper accuracy {paper_accuracy:.4f} is outside reproduction gate",
        )
    if args.minimum_paper_accuracy is not None:
        _require(
            paper_accuracy >= args.minimum_paper_accuracy,
            f"paper accuracy {paper_accuracy:.4f} is below minimum gate",
        )

    local_by_client = {int(row["client_id"]): row for row in local_rows}
    social_by_client = {int(row["client_id"]): row for row in social_rows}
    expected_ids = set(range(args.expected_clients))
    _require(set(local_by_client) == expected_ids, "local client IDs are incomplete or duplicated")
    _require(set(social_by_client) == expected_ids, "social client IDs are incomplete or duplicated")

    snapshot_dir = run_dir / f"official_round_{args.expected_round}_models"
    for client_id in sorted(expected_ids):
        local = local_by_client[client_id]
        social = social_by_client[client_id]
        expert = float(social["acc_expert"])
        new = float(social["acc_new"])
        global_accuracy = float(social["acc_global"])
        expert_images = int(social["expert_images"])
        new_images = int(social["new_images"])
        expected_global = (expert * expert_images + new * new_images) / (expert_images + new_images)
        # Different evaluation batch sizes can move one borderline prediction
        # because of floating-point kernel ordering. Reject larger discrepancies.
        one_image_tolerance = 100.0 / int(local["total"]) + 1e-9
        _require(
            abs(float(local["accuracy"]) - expert) <= one_image_tolerance,
            f"client {client_id}: local and expert differ by more than one image",
        )
        _require(abs(global_accuracy - expected_global) < 1e-9, f"client {client_id}: global weighting mismatch")
        _require((snapshot_dir / f"Client_{client_id}_model.pt").exists(), f"client {client_id}: snapshot missing")
    _require((snapshot_dir / "Server_head.pt").exists(), "server head snapshot missing")

    verdict = {
        "status": "passed",
        "run_dir": str(run_dir),
        "clients": args.expected_clients,
        "paper_round": args.expected_round,
        "paper_local_accuracy": paper_accuracy,
        "client_mean_global": float(summary["client_mean_global"]),
        "client_mean_expert": float(summary["client_mean_expert"]),
        "client_mean_new": float(summary["client_mean_new"]),
    }
    print(json.dumps(verdict, indent=2))


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset


OFFICIAL_MODELS = [
    "FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)",
    "torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)",
    "mobilenet_v2(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)",
    "torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)",
    "torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)",
]

MODEL_NAMES = [
    "4-layer CNN",
    "GoogLeNet",
    "MobileNetV2",
    "ResNet-18",
    "ResNet-34",
    "ResNet-50",
    "ResNet-101",
    "ResNet-152",
    "ViT-B/16",
    "ViT-B/32",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official FedRE implementation and add read-only social-learning evaluation."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-clients", required=True, type=int, choices=[4, 10])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fedre-system", default="external_baselines/repos/FedRE/HtFLlib/system")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--local-lr", type=float, default=0.06)
    parser.add_argument("--server-lr", type=float, default=0.01)
    parser.add_argument("--server-batch-size", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--eval-gap", type=int, default=1)
    parser.add_argument("--global-eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    warnings.simplefilter("ignore")
    output_dir = Path(cli.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fedre_system = Path(cli.fedre_system).resolve()
    if str(fedre_system) not in sys.path:
        sys.path.insert(0, str(fedre_system))
    os.chdir(fedre_system)

    from flcore.clients.clientbase import load_item  # noqa: PLC0415
    from flcore.servers.serverre import FedRE  # noqa: PLC0415

    _set_seed(cli.seed)
    args = _build_official_args(cli, output_dir / "official_model_store")
    provenance = _provenance(cli, args, fedre_system)
    _atomic_json(output_dir / "resolved_protocol.json", provenance)

    server = FedRE(args, cli.seed)
    # The public Client constructor resets torch to seed 0 for every client. Reset
    # once after construction so RAP sampling follows the requested experiment seed.
    _set_seed(cli.seed)

    snapshot_dir = output_dir / f"official_round_{cli.rounds}_models"
    final_local_rows: list[dict] = []
    original_evaluate = server.evaluate

    def evaluate_with_final_snapshot(self, acc=None, loss=None):
        result = original_evaluate(acc=acc, loss=loss)
        evaluation_index = len(self.rs_test_acc) - 1 if acc is None else -1
        if evaluation_index == cli.rounds:
            final_local_rows[:] = _evaluate_local_clients(self)
            _snapshot_official_models(self, snapshot_dir)
            _write_dict_csv(output_dir / "paper_round_local_metrics.csv", final_local_rows)
        return result

    server.evaluate = MethodType(evaluate_with_final_snapshot, server)
    _atomic_json(
        output_dir / "status.json",
        {"state": "running", "official_train_called": False, "paper_round": cli.rounds},
    )

    # FedRE training is intentionally delegated in full to the official class.
    server.train()

    if not snapshot_dir.exists() or len(final_local_rows) != cli.num_clients:
        raise RuntimeError(
            "The official training loop did not reach the requested paper-round evaluation snapshot"
        )

    global_rows = _evaluate_union_test(
        server,
        load_item,
        snapshot_dir=snapshot_dir,
        fedre_system=fedre_system,
        batch_size=cli.global_eval_batch_size,
    )
    _write_dict_csv(output_dir / "global_social_metrics.csv", global_rows)
    summary = _build_summary(cli, server, final_local_rows, global_rows, provenance)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "status.json",
        {
            "state": "complete",
            "official_train_called": True,
            "paper_round": cli.rounds,
            "official_loop_updates": cli.rounds + 1,
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def _build_official_args(cli: argparse.Namespace, model_store: Path) -> SimpleNamespace:
    models = OFFICIAL_MODELS[: cli.num_clients]
    # Table 13 reports a homogeneous 512xC classifier broadcast for every model.
    # BaseHeadSplit exposes args.heads specifically for this configuration.
    heads = ["nn.Linear(args.feature_dim, args.num_classes)" for _ in models]
    return SimpleNamespace(
        device=cli.device,
        dataset=cli.dataset,
        num_classes=100,
        global_rounds=cli.rounds,
        local_epochs=cli.local_epochs,
        batch_size=cli.local_batch_size,
        local_learning_rate=cli.local_lr,
        num_clients=cli.num_clients,
        join_ratio=1.0,
        random_join_ratio=False,
        algorithm="FedRE",
        time_select=False,
        goal=f"official_pat_seed{cli.seed}",
        time_threthold=10000,
        auto_break=False,
        save_folder_name=str(model_store),
        eval_gap=cli.eval_gap,
        client_drop_rate=0.0,
        train_slow_rate=0.0,
        send_slow_rate=0.0,
        server_learning_rate=cli.server_lr,
        head_batch_size=cli.server_batch_size,
        feature_dim=cli.feature_dim,
        learning_rate_decay=False,
        learning_rate_decay_gamma=0.99,
        models=models,
        heads=heads,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.set_num_threads(1)


def _evaluate_local_clients(server) -> list[dict]:
    rows = []
    for client in server.clients:
        correct, total, _auc = client.test_metrics()
        rows.append(
            {
                "round": server.args.global_rounds,
                "client_id": client.id,
                "model": MODEL_NAMES[client.id],
                "correct": int(correct),
                "total": int(total),
                "accuracy": 100.0 * correct / max(total, 1),
            }
        )
    return rows


def _snapshot_official_models(server, snapshot_dir: Path) -> None:
    if snapshot_dir.exists():
        raise FileExistsError(f"Refusing to overwrite paper-round snapshot: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    model_root = Path(server.save_folder_name)
    for client in server.clients:
        source = model_root / f"{client.role}_model.pt"
        shutil.copy2(source, snapshot_dir / source.name)
    server_head = model_root / "Server_head.pt"
    shutil.copy2(server_head, snapshot_dir / server_head.name)


def _evaluate_union_test(
    server,
    load_item,
    *,
    snapshot_dir: Path,
    fedre_system: Path,
    batch_size: int,
) -> list[dict]:
    dataset_root = fedre_system.parent / "dataset" / server.dataset
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    test_parts = []
    label_parts = []
    for client_id in range(server.num_clients):
        with np.load(dataset_root / "test" / f"{client_id}.npz", allow_pickle=True) as archive:
            payload = archive["data"].tolist()
        test_parts.append(payload["x"])
        label_parts.append(payload["y"])
    images = torch.from_numpy(np.concatenate(test_parts).astype(np.float32, copy=False))
    labels = torch.from_numpy(np.concatenate(label_parts).astype(np.int64, copy=False))
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    rows: list[dict] = []

    for client in server.clients:
        expert_classes = set(int(cls) for cls in manifest["agents"][str(client.id)]["classes"])
        model = load_item(client.role, "model", str(snapshot_dir))
        model.eval()
        counts = {"global": [0, 0], "expert": [0, 0], "new": [0, 0]}
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(server.device)
                batch_labels = batch_labels.to(server.device)
                matches = model(batch_images).argmax(dim=1) == batch_labels
                expert_mask = torch.zeros_like(batch_labels, dtype=torch.bool)
                for cls in expert_classes:
                    expert_mask |= batch_labels == cls
                new_mask = ~expert_mask
                for name, mask in (("expert", expert_mask), ("new", new_mask)):
                    counts[name][0] += int(matches[mask].sum().item())
                    counts[name][1] += int(mask.sum().item())
                counts["global"][0] += int(matches.sum().item())
                counts["global"][1] += int(batch_labels.numel())
        rows.append(
            {
                "client_id": client.id,
                "model": MODEL_NAMES[client.id],
                "expert_classes": f"{min(expert_classes)}-{max(expert_classes)}",
                "acc_global": _percent(*counts["global"]),
                "acc_expert": _percent(*counts["expert"]),
                "acc_new": _percent(*counts["new"]),
                "global_images": counts["global"][1],
                "expert_images": counts["expert"][1],
                "new_images": counts["new"][1],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _percent(correct: int, total: int) -> float:
    return 100.0 * correct / max(total, 1)


def _build_summary(
    cli: argparse.Namespace,
    server,
    local_rows: list[dict],
    global_rows: list[dict],
    provenance: dict,
) -> dict:
    pooled_local = _percent(
        sum(int(row["correct"]) for row in local_rows),
        sum(int(row["total"]) for row in local_rows),
    )
    return {
        "status": "complete",
        "dataset": cli.dataset,
        "seed": cli.seed,
        "paper_round": cli.rounds,
        "paper_local_accuracy": pooled_local,
        "official_recorded_local_accuracy": 100.0 * float(server.rs_test_acc[-1]),
        "client_mean_global": float(np.mean([row["acc_global"] for row in global_rows])),
        "client_mean_expert": float(np.mean([row["acc_expert"] for row in global_rows])),
        "client_mean_new": float(np.mean([row["acc_new"] for row in global_rows])),
        "client_population_std_global": float(np.std([row["acc_global"] for row in global_rows])),
        "client_population_std_expert": float(np.std([row["acc_expert"] for row in global_rows])),
        "client_population_std_new": float(np.std([row["acc_new"] for row in global_rows])),
        "forgetting": None,
        "provenance": provenance,
    }


def _provenance(cli: argparse.Namespace, args: SimpleNamespace, fedre_system: Path) -> dict:
    repo = fedre_system.parents[1]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--", "HtFLlib/system/flcore/servers/serverre.py"],
        text=True,
    )
    return {
        "method": "FedRE",
        "training_implementation": "official flcore.servers.serverre.FedRE.train",
        "fedre_commit": commit,
        "fedre_server_patch_present": bool(diff.strip()),
        "fedre_server_patch_sha256": _text_sha256(diff),
        "dataset": cli.dataset,
        "num_clients": cli.num_clients,
        "seed": cli.seed,
        "paper_round": cli.rounds,
        "official_loop_note": "Official range(global_rounds + 1); paper metrics snapshot at evaluation index 100",
        "local_epochs": cli.local_epochs,
        "local_batch_size": cli.local_batch_size,
        "local_lr": cli.local_lr,
        "server_lr": cli.server_lr,
        "server_batch_size": cli.server_batch_size,
        "feature_dim": cli.feature_dim,
        "participation": "all clients",
        "representation_mapping": "official AdaptiveAvgPool1d(512)",
        "representation_entanglement": "official RAP",
        "classifier": "official BaseHeadSplit args.heads = Linear(512, 100) for every client",
        "models": MODEL_NAMES[: cli.num_clients],
        "model_expressions": args.models,
        "training_initialization": "official random initialization; no local expert pretraining",
        "extra_evaluation": "read-only union-test evaluation of paper-round checkpoint snapshot",
    }


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


if __name__ == "__main__":
    main()

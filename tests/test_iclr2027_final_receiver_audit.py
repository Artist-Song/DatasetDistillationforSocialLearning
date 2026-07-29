import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import yaml

from config_adapter import build_dsdm_args_from_config, load_config
from scripts.audit_iclr2027_dkp_final_receivers import (
    DEFAULT_CE_RUN,
    DEFAULT_FULL_RUN,
    EXPECTED_MODELS,
    EXPECTED_RUN_NAMES,
    EXPERT_RUN_NAME,
    FinalReceiverAuditError,
    LOSS_IDENTITY_ABS_TOLERANCE,
    V2_SOCIAL_RESULT_FIELDS,
    _report_path,
    _same,
    _sha256,
    build_audit,
    main,
)
import social_output_manager


class TinyAuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 100, bias=False)

    def forward(self, images):
        return self.classifier(images.mean(dim=(2, 3)))

    def get_feature(self, images, idx_from, idx_to=-1):
        return [images.mean(dim=(2, 3))]


def tiny_model_builder(_args):
    return TinyAuditModel()


class ICLR2027FinalReceiverAuditTest(unittest.TestCase):
    def _write_snapshot(self, run_dir, source_path):
        cfg = copy.deepcopy(load_config(source_path))
        config_dir = Path(run_dir) / "config"
        config_dir.mkdir(parents=True)
        snapshot = config_dir / "main.yaml"
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        rebuilt = vars(build_dsdm_args_from_config(cfg, config_path=snapshot))
        rebuilt["config_path"] = str(snapshot.resolve())
        (config_dir / "social_resolved_args.json").write_text(
            json.dumps(rebuilt, sort_keys=True), encoding="utf-8"
        )
        return cfg, snapshot

    def _write_experts(self, root, ce_run, full_run):
        expert_run = Path(root) / EXPERT_RUN_NAME
        (expert_run / "metrics").mkdir(parents=True)
        (expert_run / "config").mkdir(parents=True)
        expert_config = expert_run / "config/main.yaml"
        expert_config.write_text("project:\n  run_name: fixture-expert\n", encoding="utf-8")
        rows = []
        shas = {}
        accuracies = {}
        for agent_id in range(5):
            source = expert_run / f"agents/agent_{agent_id}/checkpoints/expert_model.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(f"fixture-expert-{agent_id}".encode("ascii"))
            shas[agent_id] = _sha256(source)
            accuracies[agent_id] = 70.0 + agent_id
            for run_dir in (ce_run, full_run):
                target = Path(run_dir) / f"agents/agent_{agent_id}/checkpoints/expert_model.pt"
                target.parent.mkdir(parents=True)
                target.write_bytes(source.read_bytes())
            rows.append(
                {
                    "agent_id": agent_id,
                    "model": EXPECTED_MODELS[agent_id],
                    "checkpoint_sha256": shas[agent_id],
                    "output_shape": [2, 100],
                    "local_test_accuracy": accuracies[agent_id],
                }
            )
        expert_preflight = {
            "status": "passed",
            "protocol": "fixture-cosine-expert",
            "agents": rows,
        }
        path = expert_run / "metrics/cosine_expert_preflight.json"
        path.write_text(json.dumps(expert_preflight), encoding="utf-8")
        return expert_run, expert_config, shas, accuracies

    def _write_packet_integrity(self, run_dir, variant):
        require_logits = variant == "full"
        payload = {
            "summary": {
                "total_raw_images": 1000,
                "total_train_images": 4000,
            },
            "packets": [
                {
                    "sender_agent": agent_id,
                    "has_sender_logits": require_logits,
                    "raw_images": 200,
                    "decoded_or_train_images": 800,
                    "sender_logit_bytes": 32000 if require_logits else 0,
                }
                for agent_id in range(5)
            ],
            "warnings": [],
        }
        metrics = Path(run_dir) / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        (metrics / "packet_integrity_dsdm.json").write_text(json.dumps(payload), encoding="utf-8")

    def _rows(self, cfg, variant, shas, accuracies):
        require_logits = variant == "full"
        rows = []
        class_split = {
            int(key.removeprefix("agent_")): value
            for key, value in cfg["agents"]["class_split"].items()
        }
        for receiver_id in range(5):
            expert_before = accuracies[receiver_id]
            expert_after = 60.0 + receiver_id
            new_after = 20.0 + receiver_id
            ce_local = 0.4 + 0.01 * receiver_id
            ce_external = 0.2 + 0.01 * receiver_id
            loss_cls = 0.2 * ce_local + 0.8 * ce_external
            loss_fr = 0.1 + 0.01 * receiver_id if require_logits else 0.0
            loss_kd = 0.3 + 0.01 * receiver_id if require_logits else 0.0
            loss_sc = 0.5 + 0.01 * receiver_id if require_logits else 0.0
            total = loss_cls
            if require_logits:
                total += 0.2 * loss_fr + 0.6 * loss_kd + 0.1 * loss_sc
            row = {field: "" for field in V2_SOCIAL_RESULT_FIELDS}
            row.update(
                {
                    "run_name": EXPECTED_RUN_NAMES[variant],
                    "protocol": "dkp_sl_v1",
                    "dkp_variant": variant,
                    "receiver_agent": receiver_id,
                    "receiver_model": EXPECTED_MODELS[receiver_id],
                    "expert_classes": ",".join(str(value) for value in class_split[receiver_id]),
                    "packet_method": "dsdm",
                    "method": "DKP_SL" if require_logits else "DKP_CE_ONLY",
                    "init_mode": "expert",
                    "self_data_mode": "real",
                    "self_real_per_class": 0,
                    "use_fr": str(require_logits).lower(),
                    "lambda_fr": 0.2 if require_logits else 0.0,
                    "lambda_sc": 0.1 if require_logits else 0.0,
                    "supcon_temperature": 0.07,
                    "use_logits": str(require_logits).lower(),
                    "communication_mode": "logical_all_share_once",
                    "use_generalist_logits": "false",
                    "kd_mix_beta": 0.0,
                    "lambda_kd": 0.6 if require_logits else 0.0,
                    "kd_temperature": 2.0,
                    "ipc": 10,
                    "self_real_images": 10000,
                    "external_comm_images": 800,
                    "external_comm_logit_bytes": 128000 if require_logits else 0,
                    "external_comm_generalist_logit_bytes": 0,
                    "acc_global_before": 0.2 * expert_before,
                    "acc_expert_before": expert_before,
                    "acc_global_after": 0.2 * expert_after + 0.8 * new_after,
                    "acc_expert_after": expert_after,
                    "acc_new_after": new_after,
                    "forgetting": expert_before - expert_after,
                    "loss_cls": loss_cls,
                    "loss": total,
                    "loss_ce_local": ce_local,
                    "loss_ce_external": ce_external,
                    "loss_fr": loss_fr,
                    "loss_kd": loss_kd,
                    "loss_sc": loss_sc,
                    "loss_sender_kd": loss_kd,
                    "loss_generalist_kd": 0.0,
                    "receiver_augment": "true",
                    "freeze_bn_stats": "false",
                    "optimizer_steps": 9420,
                    "training_seconds": 100.0 + receiver_id,
                    "prototype_initialized_classes": 80,
                    "receiver_seed": 100000 + receiver_id,
                    "receiver_init_checkpoint_sha256": shas[receiver_id],
                    "fr_teacher_checkpoint_sha256": shas[receiver_id],
                    "time": "2026-07-27T23:00:00",
                }
            )
            rows.append(row)
        return rows

    def _write_csv(self, run_dir, rows):
        path = Path(run_dir) / "metrics/social_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=V2_SOCIAL_RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_checkpoints(self, run_dir, variant):
        paths = {}
        for receiver_id in range(5):
            path = (
                Path(run_dir)
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}/after_social.pt"
            )
            path.parent.mkdir(parents=True)
            torch.save(TinyAuditModel().state_dict(), path)
            paths[receiver_id] = path
        return paths

    def _write_communication_preflight(self, full_run, shas):
        payload = {
            "status": "passed",
            "protocol": "dkp_sl_iclr2027_cifar100_5x20_ipc10_v1",
            "expert_provenance": [
                {"agent_id": agent_id, "checkpoint_sha256": shas[agent_id]}
                for agent_id in range(5)
            ],
            "receivers": [
                {
                    "receiver_agent": agent_id,
                    "receiver_model": EXPECTED_MODELS[agent_id],
                    "init_checkpoint_sha256": shas[agent_id],
                    "fr_teacher_checkpoint_sha256": shas[agent_id],
                    "prototype_classes": 80,
                    "local_real_images": 10000,
                    "external_decoded_images": 3200,
                }
                for agent_id in range(5)
            ],
        }
        path = Path(full_run) / "metrics/communication_preflight.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _summary_social(self, row):
        return {
            "global": float(row["acc_global_after"]),
            "new": float(row["acc_new_after"]),
            "expert": float(row["acc_expert_after"]),
            "loss_mean": {
                field: float(row[field])
                for field in (
                    "loss",
                    "loss_cls",
                    "loss_ce_local",
                    "loss_ce_external",
                    "loss_fr",
                    "loss_kd",
                    "loss_sc",
                )
            },
            "training_seconds": float(row["training_seconds"]),
            "optimizer_steps": int(row["optimizer_steps"]),
            "raw_external_images": int(row["external_comm_images"]),
            "logit_bytes": int(row["external_comm_logit_bytes"]),
            "prototype_initialized_classes": int(row["prototype_initialized_classes"]),
        }

    def _aggregate(self, rows):
        loss_fields = (
            "loss",
            "loss_cls",
            "loss_ce_local",
            "loss_ce_external",
            "loss_fr",
            "loss_kd",
            "loss_sc",
        )
        result = {
            "metrics_mean": {
                metric: sum(float(row[field]) for row in rows) / 5.0
                for metric, field in (
                    ("global", "acc_global_after"),
                    ("new", "acc_new_after"),
                    ("expert", "acc_expert_after"),
                )
            },
            "loss_mean": {
                field: sum(float(row[field]) for row in rows) / 5.0 for field in loss_fields
            },
            "resources": {},
        }
        resources = result["resources"]
        for name, field in (
            ("training_seconds", "training_seconds"),
            ("optimizer_steps", "optimizer_steps"),
            ("raw_external_images", "external_comm_images"),
            ("logit_bytes", "external_comm_logit_bytes"),
        ):
            total = sum(float(row[field]) for row in rows)
            resources[f"{name}_total"] = total
            resources[f"{name}_mean"] = total / 5.0
        return result

    def _write_summary(
        self,
        expert_run,
        expert_config,
        ce_run,
        full_run,
        ce_snapshot,
        full_snapshot,
        ce_csv,
        full_csv,
        ce_rows,
        full_rows,
        shas,
        accuracies,
    ):
        payload = {
            "status": "complete_diagnostic",
            "formal_result": False,
            "paper_eligible": False,
            "seed": 0,
            "agents": 5,
            "classes_per_agent": 20,
            "ipc": 10,
            "receiver_ids": list(range(5)),
            "sources": {
                "expert_run_dir": str(Path(expert_run).resolve()),
                "expert_config": str(Path(expert_config).resolve()),
                "expert_config_sha256": _sha256(expert_config),
                "ce_only_config": str(Path(ce_snapshot).resolve()),
                "ce_only_config_sha256": _sha256(ce_snapshot),
                "full_config": str(Path(full_snapshot).resolve()),
                "full_config_sha256": _sha256(full_snapshot),
                "ce_only_social_results": str(Path(ce_csv).resolve()),
                "ce_only_social_results_sha256": _sha256(ce_csv),
                "full_social_results": str(Path(full_csv).resolve()),
                "full_social_results_sha256": _sha256(full_csv),
            },
            "per_receiver": [
                {
                    "receiver_agent": receiver_id,
                    "receiver_model": EXPECTED_MODELS[receiver_id],
                    "local_expert": {
                        "expert": accuracies[receiver_id],
                        "checkpoint_sha256": shas[receiver_id],
                    },
                    "ce_only": self._summary_social(ce_rows[receiver_id]),
                    "full_dkp_sl": self._summary_social(full_rows[receiver_id]),
                }
                for receiver_id in range(5)
            ],
            "mean_over_five_receivers": {
                "local_expert": {"expert": sum(accuracies.values()) / 5.0},
                "ce_only": self._aggregate(ce_rows),
                "full_dkp_sl": self._aggregate(full_rows),
            },
        }
        path = Path(full_run) / "metrics/first_round_seed0_summary.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _fixture(self, root):
        root = Path(root)
        ce_run = root / EXPECTED_RUN_NAMES["ce_only"]
        full_run = root / EXPECTED_RUN_NAMES["full"]
        ce_cfg, ce_snapshot = self._write_snapshot(ce_run, DEFAULT_CE_RUN / "config/main.yaml")
        full_cfg, full_snapshot = self._write_snapshot(full_run, DEFAULT_FULL_RUN / "config/main.yaml")
        expert_run, expert_config, shas, accuracies = self._write_experts(root, ce_run, full_run)
        self._write_packet_integrity(ce_run, "ce_only")
        self._write_packet_integrity(full_run, "full")
        ce_rows = self._rows(ce_cfg, "ce_only", shas, accuracies)
        full_rows = self._rows(full_cfg, "full", shas, accuracies)
        ce_csv = self._write_csv(ce_run, ce_rows)
        full_csv = self._write_csv(full_run, full_rows)
        checkpoints = {
            "ce_only": self._write_checkpoints(ce_run, "ce_only"),
            "full": self._write_checkpoints(full_run, "full"),
        }
        communication = self._write_communication_preflight(full_run, shas)
        summary = self._write_summary(
            expert_run,
            expert_config,
            ce_run,
            full_run,
            ce_snapshot,
            full_snapshot,
            ce_csv,
            full_csv,
            ce_rows,
            full_rows,
            shas,
            accuracies,
        )
        return {
            "ce_run": ce_run,
            "full_run": full_run,
            "expert_run": expert_run,
            "summary": summary,
            "communication": communication,
            "ce_csv": ce_csv,
            "full_csv": full_csv,
            "checkpoints": checkpoints,
        }

    def _audit(self, fixture):
        return build_audit(
            ce_run=fixture["ce_run"],
            full_run=fixture["full_run"],
            expert_run=fixture["expert_run"],
            summary_path=fixture["summary"],
            model_builder=tiny_model_builder,
        )

    def test_complete_fixture_audits_ten_checkpoints_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            report_path = _report_path(fixture["full_run"])
            report = self._audit(fixture)
            self.assertFalse(report_path.exists())
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["formal_result"])
        self.assertEqual(len(report["runs"]["ce_only"]["checkpoints"]), 5)
        self.assertEqual(len(report["runs"]["full"]["checkpoints"]), 5)
        self.assertEqual(report["runs"]["full"]["checkpoints"][0]["output_shape"], [2, 100])

    def test_future_global_schema_extension_does_not_change_v2_contract(self):
        future_fields = list(social_output_manager.SOCIAL_RESULT_FIELDS) + ["future_only_field"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            with patch.object(social_output_manager, "SOCIAL_RESULT_FIELDS", future_fields):
                report = self._audit(fixture)
        self.assertEqual(report["status"], "passed")
        self.assertNotEqual(future_fields, V2_SOCIAL_RESULT_FIELDS)

    def test_loss_identity_tolerance_accepts_roundoff_but_rejects_drift(self):
        _same(
            1.0 + LOSS_IDENTITY_ABS_TOLERANCE / 2.0,
            1.0,
            "roundoff",
            tolerance=LOSS_IDENTITY_ABS_TOLERANCE,
        )
        with self.assertRaisesRegex(FinalReceiverAuditError, "mismatch"):
            _same(
                1.0 + LOSS_IDENTITY_ABS_TOLERANCE * 2.0,
                1.0,
                "drift",
                tolerance=LOSS_IDENTITY_ABS_TOLERANCE,
            )

    def test_cli_requires_explicit_write_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            argv = [
                "--ce-run-dir", str(fixture["ce_run"]),
                "--full-run-dir", str(fixture["full_run"]),
                "--expert-run-dir", str(fixture["expert_run"]),
                "--summary-json", str(fixture["summary"]),
            ]
            report_path = _report_path(fixture["full_run"])
            with patch(
                "scripts.audit_iclr2027_dkp_final_receivers._default_model_builder",
                tiny_model_builder,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertFalse(report_path.exists())
                self.assertEqual(main(argv + ["--write-report"]), 0)
                self.assertTrue(report_path.is_file())
                self.assertEqual(main(argv + ["--write-report"]), 1)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "passed")

    def test_nonfinite_or_schema_incompatible_checkpoint_fails_closed(self):
        mutations = ("nonfinite", "missing_key")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp_dir:
                fixture = self._fixture(tmp_dir)
                path = fixture["checkpoints"]["full"][2]
                state = torch.load(path, map_location="cpu", weights_only=True)
                if mutation == "nonfinite":
                    state["classifier.weight"][0, 0] = float("nan")
                    error = "non-finite tensors"
                else:
                    state["unexpected.weight"] = state.pop("classifier.weight")
                    error = "strict checkpoint load failed"
                torch.save(state, path)
                with self.assertRaisesRegex(FinalReceiverAuditError, error):
                    self._audit(fixture)

    def test_csv_loss_metric_and_provenance_drift_fail_closed(self):
        mutations = (
            ("loss identity", "loss", "0.123", "total-loss identity"),
            ("metric identity", "acc_global_after", "99", "global metric identity"),
            ("full regularizer", "loss_sc", "0", "loss_sc is not positive"),
            ("seed", "receiver_seed", "7", "seed mismatch"),
            ("SHA", "receiver_init_checkpoint_sha256", "0" * 64, "init SHA mismatch"),
        )
        for name, field, value, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp_dir:
                fixture = self._fixture(tmp_dir)
                with fixture["full_csv"].open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0][field] = value
                if name == "full regularizer":
                    rows[0]["loss"] = str(
                        float(rows[0]["loss_cls"])
                        + 0.2 * float(rows[0]["loss_fr"])
                        + 0.6 * float(rows[0]["loss_kd"])
                    )
                self._write_csv(fixture["full_run"], rows)
                with self.assertRaisesRegex(FinalReceiverAuditError, error):
                    self._audit(fixture)

    def test_failed_preflight_or_incomplete_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            payload = json.loads(fixture["communication"].read_text(encoding="utf-8"))
            payload["status"] = "failed"
            fixture["communication"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(FinalReceiverAuditError, "did not pass"):
                self._audit(fixture)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            payload = json.loads(fixture["summary"].read_text(encoding="utf-8"))
            payload["formal_result"] = True
            fixture["summary"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(FinalReceiverAuditError, "formal result"):
                self._audit(fixture)


if __name__ == "__main__":
    unittest.main()

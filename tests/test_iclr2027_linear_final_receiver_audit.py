import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from agent_data import build_agent_args
from config_adapter import load_config
from scripts.audit_iclr2027_linear_final_receivers import (
    DEFAULT_CE_CONFIG,
    DEFAULT_EXPERT_CONFIG,
    DEFAULT_FULL_CONFIG,
    DEFAULT_REPORT_NAME,
    LinearFinalReceiverAuditError,
    audit_linear_receiver_checkpoints,
    main,
    validate_linear_receiver_checkpoint,
)
from scripts.prepare_iclr2027_linear_head_ablation import (
    CE_ONLY_RUN,
    EXPERT_RUN,
    FULL_RUN,
)
from scripts.validate_iclr2027_cosine_experts import sha256_file
from social_output_manager import SOCIAL_RESULT_FIELDS


class TinyLinearReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(4, 100)

    def _features(self, images):
        return self.encoder(images).flatten(1)

    def forward(self, images):
        return self.classifier(self._features(images))

    def get_feature(self, images, idx_from, idx_to=-1):
        features = self._features(images)
        return [features], self.classifier(features)


def tiny_model_builder(_args):
    return TinyLinearReceiver()


class LinearCheckpointHelperTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            classifier_type="linear",
            nclass=100,
            nch=3,
            size=32,
            idx_from=0,
            idx_to=-1,
        )

    def test_public_helper_strict_loads_finite_linear_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "after_social.pt"
            torch.save(TinyLinearReceiver().state_dict(), checkpoint)
            report = validate_linear_receiver_checkpoint(
                checkpoint,
                self._args(),
                expected_sha256=sha256_file(checkpoint),
                model_builder=tiny_model_builder,
                random_seed=17,
            )

        self.assertEqual(report["classifier_type"], "linear")
        self.assertEqual(report["output_shape"], [2, 100])
        self.assertEqual(report["feature_shapes"], [[2, 4]])
        self.assertEqual(report["classifier_out_features"], 100)

    def test_public_helper_rejects_nonfinite_and_incomplete_state(self):
        mutations = ("nonfinite", "missing")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = Path(tmp_dir) / "after_social.pt"
                state = TinyLinearReceiver().state_dict()
                if mutation == "nonfinite":
                    state["classifier.weight"][0, 0] = float("nan")
                    expected = "non-finite tensors"
                else:
                    state.pop("classifier.weight")
                    expected = "strict checkpoint load failed"
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(LinearFinalReceiverAuditError, expected):
                    validate_linear_receiver_checkpoint(
                        checkpoint,
                        self._args(),
                        model_builder=tiny_model_builder,
                    )


class LinearFinalReceiverAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            "expert": load_config(DEFAULT_EXPERT_CONFIG),
            "ce_only": load_config(DEFAULT_CE_CONFIG),
            "full": load_config(DEFAULT_FULL_CONFIG),
        }

    @staticmethod
    def _copy_snapshot(source, run_dir):
        target = Path(run_dir) / "config/main.yaml"
        target.parent.mkdir(parents=True)
        target.write_bytes(Path(source).read_bytes())
        return target

    @staticmethod
    def _losses(receiver_id, is_full):
        loss_cls = 1.0 + receiver_id
        loss_fr = 0.1 + receiver_id / 100.0 if is_full else 0.0
        loss_kd = 0.2 + receiver_id / 100.0 if is_full else 0.0
        loss_sc = 0.3 + receiver_id / 100.0 if is_full else 0.0
        return {
            "loss": loss_cls + 0.2 * loss_fr + 0.6 * loss_kd + 0.1 * loss_sc,
            "loss_cls": loss_cls,
            "loss_ce_local": 0.5 + receiver_id,
            "loss_ce_external": 1.125 + receiver_id,
            "loss_fr": loss_fr,
            "loss_kd": loss_kd,
            "loss_sc": loss_sc,
        }

    def _write_variant(self, run_dir, snapshot, variant, expert_shas, accuracies):
        config = load_config(snapshot)
        is_full = variant == "full"
        args_by_agent = {
            receiver_id: build_agent_args(config, snapshot, receiver_id)
            for receiver_id in range(5)
        }
        rows = []
        evidence = {}
        for receiver_id, args in args_by_agent.items():
            checkpoint_dir = (
                Path(run_dir)
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
            )
            checkpoint_dir.mkdir(parents=True)
            final_path = checkpoint_dir / "after_social.pt"
            torch.manual_seed(1000 + receiver_id)
            torch.save(TinyLinearReceiver().state_dict(), final_path)
            final_sha = sha256_file(final_path)
            alpha = 2.0 + receiver_id
            beta = 0.1 + receiver_id / 10.0
            losses = self._losses(receiver_id, is_full)
            global_after = (30.0 if is_full else 20.0) + receiver_id
            new_after = global_after - 5.0
            expert_after = global_after + 20.0
            external_classes = sorted(set(range(100)) - set(args.active_class_ids))
            provenance = {
                "protocol": "dkp_sl_v1",
                "dkp_variant": variant,
                "loss_switches": config["social_learning"]["receiver"]["loss_switches"],
                "receiver_agent": receiver_id,
                "receiver_model": args.model_name,
                "receiver_seed": 100_000 + receiver_id,
                "receiver_init_checkpoint_sha256": expert_shas[receiver_id],
                "fr_teacher_checkpoint_sha256": expert_shas[receiver_id],
                "classifier_type": "linear",
                "prototype_initialization": {
                    "classifier_type": "linear",
                    "mode": "linear_local_row_norm_bias_mean",
                    "alpha": alpha,
                    "beta": beta,
                    "external_weight_norm_min": alpha,
                    "external_weight_norm_max": alpha,
                    "initialized_classes": external_classes,
                },
                "checkpoint_retention": "final_only",
                "checkpoint_artifacts": {
                    "after_social": {
                        "path": str(final_path.resolve()),
                        "sha256": final_sha,
                    }
                },
                "statistics": {
                    "metrics_before": {
                        "acc_global": accuracies[receiver_id] / 5.0,
                        "acc_expert": accuracies[receiver_id],
                    },
                    "metrics_after": {
                        "acc_global": global_after,
                        "acc_new": new_after,
                        "acc_expert": expert_after,
                    },
                    "loss_means": losses,
                    "optimizer_steps": 9_420,
                    "training_seconds": 100.0 + receiver_id,
                    "external_comm_images": 800,
                    "external_comm_logit_bytes": 128_000 if is_full else 0,
                },
            }
            provenance_path = checkpoint_dir / "receiver_provenance.json"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

            row = {field: "" for field in SOCIAL_RESULT_FIELDS}
            row.update(
                {
                    "run_name": config["project"]["run_name"],
                    "protocol": "dkp_sl_v1",
                    "dkp_variant": variant,
                    "receiver_agent": receiver_id,
                    "receiver_model": args.model_name,
                    "expert_classes": ",".join(str(value) for value in args.active_class_ids),
                    "packet_method": "dsdm",
                    "method": "DKP_SL" if is_full else "DKP_CE_ONLY",
                    "init_mode": "expert",
                    "self_data_mode": "real",
                    "self_real_per_class": 0,
                    "use_fr": str(is_full).lower(),
                    "lambda_fr": config["social_learning"]["receiver"]["lambda_fr"],
                    "lambda_sc": config["social_learning"]["receiver"]["lambda_sc"],
                    "supcon_temperature": config["social_learning"]["receiver"]["supcon_temperature"],
                    "use_logits": str(is_full).lower(),
                    "communication_mode": config["communication"]["mode"],
                    "use_generalist_logits": "false",
                    "kd_mix_beta": 0.0,
                    "lambda_kd": config["logits"]["lambda_kd"],
                    "kd_temperature": config["logits"]["temperature"],
                    "ipc": 10,
                    "self_real_images": 10_000,
                    "external_comm_images": 800,
                    "external_comm_logit_bytes": 128_000 if is_full else 0,
                    "external_comm_generalist_logit_bytes": 0,
                    "acc_global_before": accuracies[receiver_id] / 5.0,
                    "acc_expert_before": accuracies[receiver_id],
                    "acc_global_after": global_after,
                    "acc_expert_after": expert_after,
                    "acc_new_after": new_after,
                    "forgetting": accuracies[receiver_id] - expert_after,
                    **losses,
                    "loss_sender_kd": losses["loss_kd"],
                    "loss_generalist_kd": 0.0,
                    "receiver_augment": "true",
                    "freeze_bn_stats": "false",
                    "optimizer_steps": 9_420,
                    "training_seconds": 100.0 + receiver_id,
                    "prototype_initialized_classes": 80,
                    "classifier_type": "linear",
                    "prototype_init_mode": "linear_local_row_norm_bias_mean",
                    "prototype_alpha": alpha,
                    "prototype_beta": beta,
                    "prototype_weight_norm_min": alpha,
                    "prototype_weight_norm_max": alpha,
                    "checkpoint_retention": "final_only",
                    "after_social_checkpoint_sha256": final_sha,
                    "receiver_provenance_path": str(provenance_path.resolve()),
                    "receiver_provenance_sha256": sha256_file(provenance_path),
                    "receiver_seed": 100_000 + receiver_id,
                    "receiver_init_checkpoint_sha256": expert_shas[receiver_id],
                    "fr_teacher_checkpoint_sha256": expert_shas[receiver_id],
                    "time": "2026-07-28T00:00:00",
                }
            )
            rows.append(row)
            evidence[receiver_id] = {
                "final": final_path,
                "provenance": provenance_path,
            }

        csv_path = Path(run_dir) / "metrics/social_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOCIAL_RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return rows, csv_path, evidence

    @staticmethod
    def _summary_social(row):
        return {
            "global": float(row["acc_global_after"]),
            "new": float(row["acc_new_after"]),
            "expert": float(row["acc_expert_after"]),
            "loss_mean": {field: float(row[field]) for field in (
                "loss",
                "loss_cls",
                "loss_ce_local",
                "loss_ce_external",
                "loss_fr",
                "loss_kd",
                "loss_sc",
            )},
            "training_seconds": float(row["training_seconds"]),
            "optimizer_steps": int(row["optimizer_steps"]),
            "raw_external_images": int(row["external_comm_images"]),
            "logit_bytes": int(row["external_comm_logit_bytes"]),
            "prototype_initialized_classes": int(row["prototype_initialized_classes"]),
            "linear_prototype": {
                "prototype_alpha": float(row["prototype_alpha"]),
                "prototype_beta": float(row["prototype_beta"]),
                "prototype_weight_norm_min": float(row["prototype_weight_norm_min"]),
                "prototype_weight_norm_max": float(row["prototype_weight_norm_max"]),
                "after_social_checkpoint_sha256": row["after_social_checkpoint_sha256"],
                "receiver_provenance": row["receiver_provenance_path"],
                "receiver_provenance_sha256": row["receiver_provenance_sha256"],
            },
        }

    @staticmethod
    def _aggregate(rows):
        payload = {
            "metrics_mean": {},
            "loss_mean": {},
            "resources": {},
        }
        for name, field in (
            ("global", "acc_global_after"),
            ("new", "acc_new_after"),
            ("expert", "acc_expert_after"),
        ):
            payload["metrics_mean"][name] = sum(float(row[field]) for row in rows) / 5.0
        for field in (
            "loss",
            "loss_cls",
            "loss_ce_local",
            "loss_ce_external",
            "loss_fr",
            "loss_kd",
            "loss_sc",
        ):
            payload["loss_mean"][field] = sum(float(row[field]) for row in rows) / 5.0
        for name, field in (
            ("training_seconds", "training_seconds"),
            ("optimizer_steps", "optimizer_steps"),
            ("raw_external_images", "external_comm_images"),
            ("logit_bytes", "external_comm_logit_bytes"),
        ):
            total = sum(float(row[field]) for row in rows)
            payload["resources"][f"{name}_total"] = total
            payload["resources"][f"{name}_mean"] = total / 5.0
        return payload

    def _fixture(self, root):
        root = Path(root)
        runs = {
            "expert": root / EXPERT_RUN,
            "ce_only": root / CE_ONLY_RUN,
            "full": root / FULL_RUN,
        }
        runs["expert"].mkdir(parents=True)
        snapshots = {
            "ce_only": self._copy_snapshot(DEFAULT_CE_CONFIG, runs["ce_only"]),
            "full": self._copy_snapshot(DEFAULT_FULL_CONFIG, runs["full"]),
        }
        expert_shas = {receiver_id: f"{receiver_id + 1:064x}" for receiver_id in range(5)}
        accuracies = {receiver_id: 60.0 + receiver_id for receiver_id in range(5)}
        ce_rows, ce_csv, ce_evidence = self._write_variant(
            runs["ce_only"], snapshots["ce_only"], "ce_only", expert_shas, accuracies
        )
        full_rows, full_csv, full_evidence = self._write_variant(
            runs["full"], snapshots["full"], "full", expert_shas, accuracies
        )
        summary = {
            "status": "complete_diagnostic",
            "formal_result": False,
            "paper_eligible": False,
            "result_scope": "matched-linear classifier seed0 internal ablation only",
            "dataset": "cifar100",
            "seed": 0,
            "agents": 5,
            "classes_per_agent": 20,
            "ipc": 10,
            "classifier": "linear",
            "receiver_ids": list(range(5)),
            "sources": {
                "expert_config": str(DEFAULT_EXPERT_CONFIG.resolve()),
                "expert_config_sha256": sha256_file(DEFAULT_EXPERT_CONFIG),
                "expert_run_dir": str(runs["expert"].resolve()),
                "ce_only_config": str(DEFAULT_CE_CONFIG.resolve()),
                "ce_only_config_sha256": sha256_file(DEFAULT_CE_CONFIG),
                "ce_only_social_results": str(ce_csv.resolve()),
                "ce_only_social_results_sha256": sha256_file(ce_csv),
                "full_config": str(DEFAULT_FULL_CONFIG.resolve()),
                "full_config_sha256": sha256_file(DEFAULT_FULL_CONFIG),
                "full_social_results": str(full_csv.resolve()),
                "full_social_results_sha256": sha256_file(full_csv),
            },
            "per_receiver": [
                {
                    "receiver_agent": receiver_id,
                    "receiver_model": ce_rows[receiver_id]["receiver_model"],
                    "local_linear_expert": {
                        "expert": accuracies[receiver_id],
                        "selected_epoch": 100 + receiver_id,
                        "checkpoint_sha256": expert_shas[receiver_id],
                    },
                    "ce_only": self._summary_social(ce_rows[receiver_id]),
                    "full_dkp_sl": self._summary_social(full_rows[receiver_id]),
                }
                for receiver_id in range(5)
            ],
            "mean_over_five_receivers": {
                "local_linear_expert": {"expert": sum(accuracies.values()) / 5.0},
                "ce_only": self._aggregate(ce_rows),
                "full_dkp_sl": self._aggregate(full_rows),
            },
        }
        summary_path = runs["full"] / "metrics/linear_head_seed0_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return {
            "runs": runs,
            "summary": summary_path,
            "csv": {"ce_only": ce_csv, "full": full_csv},
            "evidence": {"ce_only": ce_evidence, "full": full_evidence},
        }

    @staticmethod
    def _audit(fixture):
        return audit_linear_receiver_checkpoints(
            expert_config=DEFAULT_EXPERT_CONFIG,
            ce_config=DEFAULT_CE_CONFIG,
            full_config=DEFAULT_FULL_CONFIG,
            ce_run_dir=fixture["runs"]["ce_only"],
            full_run_dir=fixture["runs"]["full"],
            summary_path=fixture["summary"],
            model_builder=tiny_model_builder,
        )

    def test_complete_fixture_audits_exactly_ten_final_only_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            report = self._audit(fixture)

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["formal_result"])
        self.assertFalse(report["paper_eligible"])
        self.assertEqual(report["checkpoint_count"], 10)
        self.assertEqual(len(report["runs"]["ce_only"]["checkpoints"]), 5)
        self.assertEqual(len(report["runs"]["full"]["checkpoints"]), 5)
        self.assertEqual(report["runs"]["full"]["checkpoints"][4]["output_shape"], [2, 100])

    def test_final_only_and_provenance_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            checkpoint_dir = fixture["evidence"]["ce_only"][0]["final"].parent
            torch.save(TinyLinearReceiver().state_dict(), checkpoint_dir / "before_social.pt")
            with self.assertRaisesRegex(LinearFinalReceiverAuditError, "non-final checkpoints"):
                self._audit(fixture)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            provenance = fixture["evidence"]["full"][1]["provenance"]
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            payload["checkpoint_artifacts"]["after_social"]["sha256"] = "0" * 64
            provenance.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LinearFinalReceiverAuditError, "provenance SHA mismatch"):
                self._audit(fixture)

    def test_summary_source_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            payload = json.loads(fixture["summary"].read_text(encoding="utf-8"))
            payload["sources"]["full_social_results_sha256"] = "0" * 64
            fixture["summary"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LinearFinalReceiverAuditError, "declared SHA mismatch"):
                self._audit(fixture)

    def test_cli_writes_once_and_has_no_overwrite_escape_hatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = self._fixture(tmp_dir)
            argv = [
                "--ce-run-dir",
                str(fixture["runs"]["ce_only"]),
                "--full-run-dir",
                str(fixture["runs"]["full"]),
                "--summary-json",
                str(fixture["summary"]),
            ]
            output = fixture["runs"]["full"] / f"metrics/{DEFAULT_REPORT_NAME}"
            with patch(
                "scripts.audit_iclr2027_linear_final_receivers._default_model_builder",
                tiny_model_builder,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertTrue(output.is_file())
                self.assertEqual(main(argv), 1)
                with self.assertRaises(SystemExit):
                    main(argv + ["--overwrite"])
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "passed")
        self.assertFalse(payload["paper_eligible"])


if __name__ == "__main__":
    unittest.main()

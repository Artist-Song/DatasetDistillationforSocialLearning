import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from agent_data import build_agent_args
from config_adapter import load_config
from DSDM.models.cosine_classifier import CosineClassifier
from scripts.audit_iclr2027_dkp_loss_ablation_final_receivers import (
    CONDITION_SWITCHES,
    DEFAULT_EXPERT_CONFIG,
    EXPECTED_MODELS,
    LossAblationFinalAuditError,
    _report_path,
    _validate_loss_row,
    _validate_receiver_artifacts,
    _validate_summary,
    _write_json_exclusive,
    validate_cosine_receiver_checkpoint,
)
from scripts.prepare_iclr2027_dkp_loss_ablation import variant_id
from scripts.summarize_iclr2027_dkp_first_round import LOSS_FIELDS
from scripts.summarize_iclr2027_dkp_loss_ablation import ALL_SWITCHES
from scripts.validate_iclr2027_cosine_experts import sha256_file


class TinyCosineReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = CosineClassifier(4, 100, scale_init=10.0)

    def _features(self, images):
        return self.encoder(images).flatten(1)

    def forward(self, images):
        return self.classifier(self._features(images))

    def get_feature(self, images, idx_from, idx_to=-1):
        features = self._features(images)
        return [features], self.classifier(features)


def tiny_model_builder(_args):
    return TinyCosineReceiver()


class CosineCheckpointAuditTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            classifier_type="cosine",
            nclass=100,
            nch=3,
            size=32,
            idx_from=0,
            idx_to=-1,
        )

    def test_strict_load_checks_cosine_output_and_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "after_social.pt"
            torch.save(TinyCosineReceiver().state_dict(), checkpoint)
            report = validate_cosine_receiver_checkpoint(
                checkpoint,
                self._args(),
                expected_sha256=sha256_file(checkpoint),
                model_builder=tiny_model_builder,
                random_seed=19,
            )

        self.assertEqual(report["classifier_type"], "cosine")
        self.assertEqual(report["output_shape"], [2, 100])
        self.assertEqual(report["feature_shapes"], [[2, 4]])
        self.assertGreater(report["cosine_scale"], 0.0)

    def test_strict_load_rejects_nonfinite_and_incomplete_state(self):
        for mutation in ("nonfinite", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = Path(tmp_dir) / "after_social.pt"
                state = TinyCosineReceiver().state_dict()
                if mutation == "nonfinite":
                    state["classifier.weight"][0, 0] = float("nan")
                    expected = "non-finite tensors"
                else:
                    state.pop("classifier.weight")
                    expected = "strict checkpoint load failed"
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(LossAblationFinalAuditError, expected):
                    validate_cosine_receiver_checkpoint(
                        checkpoint,
                        self._args(),
                        model_builder=tiny_model_builder,
                    )


class LossAndProvenanceAuditTest(unittest.TestCase):
    CONDITION = "fr1_kd0_sc1"
    SWITCHES = (True, False, True)
    CONFIG = Path(
        "configs/iclr2027/"
        "cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1.yaml"
    )

    @staticmethod
    def _losses():
        losses = {
            "loss_cls": 1.0,
            "loss_ce_local": 0.6,
            "loss_ce_external": 1.1,
            "loss_fr": 0.2,
            "loss_kd": 0.0,
            "loss_sc": 0.3,
        }
        losses["loss"] = 1.0 + 0.2 * 0.2 + 0.1 * 0.3
        return losses

    def test_loss_identity_and_inactive_component_are_fail_closed(self):
        row = {
            **self._losses(),
            "loss_sender_kd": 0.0,
            "loss_generalist_kd": 0.0,
        }
        self.assertEqual(_validate_loss_row(row, self.SWITCHES, "fixture"), self._losses())
        row["loss_kd"] = 0.1
        row["loss_sender_kd"] = 0.1
        with self.assertRaisesRegex(LossAblationFinalAuditError, "inactive loss_kd"):
            _validate_loss_row(row, self.SWITCHES, "fixture")

    def _fixture(self, root):
        cfg = load_config(self.CONFIG)
        args = build_agent_args(cfg, self.CONFIG, 0)
        run_dir = Path(root) / cfg["project"]["run_name"]
        checkpoint_dir = (
            run_dir
            / "social_learning/receiver_agent_0/checkpoints/"
            "dkp_sl_v1_ablation_fr1_kd0_sc1"
        )
        checkpoint_dir.mkdir(parents=True)
        final_path = checkpoint_dir / "after_social.pt"
        torch.save(TinyCosineReceiver().state_dict(), final_path)
        final_sha = sha256_file(final_path)
        expert_sha = "a" * 64
        losses = self._losses()
        metrics = {"global": 31.0, "new": 21.0, "expert": 71.0}
        external_classes = sorted(set(range(100)) - set(args.active_class_ids))
        provenance = {
            "protocol": "dkp_sl_v1",
            "dkp_variant": "ablation_fr1_kd0_sc1",
            "loss_switches": {"fr": True, "kd": False, "supcon": True},
            "receiver_agent": 0,
            "receiver_model": args.model_name,
            "receiver_seed": 100_000,
            "receiver_init_checkpoint_sha256": expert_sha,
            "fr_teacher_checkpoint_sha256": expert_sha,
            "classifier_type": "cosine",
            "prototype_initialization": {
                "classifier_type": "cosine",
                "mode": "cosine_unit_weight_rows",
                "alpha": None,
                "beta": None,
                "external_weight_norm_min": 1.0,
                "external_weight_norm_max": 1.0,
                "local_rows_preserved": True,
                "initialized_classes": external_classes,
            },
            "checkpoint_retention": "final_only",
            "checkpoint_artifacts": {
                "after_social": {"path": str(final_path.resolve()), "sha256": final_sha}
            },
            "statistics": {
                "metrics_before": {"acc_global": 16.0, "acc_expert": 80.0},
                "metrics_after": {
                    "acc_global": metrics["global"],
                    "acc_new": metrics["new"],
                    "acc_expert": metrics["expert"],
                },
                "loss_means": losses,
                "optimizer_steps": 9_420,
                "training_seconds": 123.0,
                "external_comm_images": 800,
                "external_comm_logit_bytes": 0,
            },
        }
        provenance_path = checkpoint_dir / "receiver_provenance.json"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        row = {
            "run_name": cfg["project"]["run_name"],
            "receiver_model": args.model_name,
            "classifier_type": "cosine",
            "dkp_variant": "ablation_fr1_kd0_sc1",
            "checkpoint_retention": "final_only",
            "optimizer_steps": 9_420,
            "external_comm_images": 800,
            "external_comm_logit_bytes": 0,
            "prototype_initialized_classes": 80,
            "receiver_init_checkpoint_sha256": expert_sha,
            "fr_teacher_checkpoint_sha256": expert_sha,
            "loss_sender_kd": 0.0,
            "loss_generalist_kd": 0.0,
            "acc_global_before": 16.0,
            "acc_expert_before": 80.0,
            "acc_global_after": metrics["global"],
            "acc_new_after": metrics["new"],
            "acc_expert_after": metrics["expert"],
            "forgetting": 9.0,
            "training_seconds": 123.0,
            "after_social_checkpoint_sha256": final_sha,
            "receiver_provenance_path": str(provenance_path.resolve()),
            "receiver_provenance_sha256": sha256_file(provenance_path),
            **losses,
        }
        validated = {"metrics": metrics, "losses": losses}
        return cfg, args, run_dir, row, validated, expert_sha, provenance_path, provenance

    def test_receiver_artifact_audit_links_checkpoint_provenance_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg, args, run_dir, row, validated, expert_sha, _path, _provenance = self._fixture(tmp_dir)
            report = _validate_receiver_artifacts(
                run_dir,
                cfg,
                args,
                row,
                validated,
                self.SWITCHES,
                expert_sha,
                tiny_model_builder,
            )
        self.assertEqual(report["checkpoint"]["output_shape"], [2, 100])
        self.assertEqual(report["logit_bytes"], 0)

    def test_receiver_artifact_audit_rejects_provenance_statistic_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg, args, run_dir, row, validated, expert_sha, path, provenance = self._fixture(tmp_dir)
            provenance["statistics"]["external_comm_images"] = 799
            path.write_text(json.dumps(provenance), encoding="utf-8")
            row["receiver_provenance_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(LossAblationFinalAuditError, "provenance images"):
                _validate_receiver_artifacts(
                    run_dir,
                    cfg,
                    args,
                    row,
                    validated,
                    self.SWITCHES,
                    expert_sha,
                    tiny_model_builder,
                )


class SummaryAndPublicationAuditTest(unittest.TestCase):
    @staticmethod
    def _receiver(receiver_id):
        losses = {
            "loss": 1.0 + receiver_id / 100.0,
            "loss_cls": 1.0 + receiver_id / 100.0,
            "loss_ce_local": 1.0 + receiver_id / 100.0,
            "loss_ce_external": 1.0 + receiver_id / 100.0,
            "loss_fr": 0.0,
            "loss_kd": 0.0,
            "loss_sc": 0.0,
        }
        return {
            "metrics": {
                "global": 20.0 + receiver_id,
                "new": 10.0 + receiver_id,
                "expert": 60.0 + receiver_id,
            },
            "losses": losses,
            "training_seconds": 100.0 + receiver_id,
            "optimizer_steps": 9_420,
            "raw_external_images": 800,
            "logit_bytes": 0,
            "prototype_initialized_classes": 80,
        }

    def _summary_fixture(self, root):
        root = Path(root)
        contexts = {"expert_config": Path(DEFAULT_EXPERT_CONFIG).resolve(), "conditions": {}}
        sources = {
            "expert_config": str(Path(DEFAULT_EXPERT_CONFIG).resolve()),
            "expert_config_sha256": sha256_file(DEFAULT_EXPERT_CONFIG),
        }
        per_receiver = [
            {
                "receiver_agent": receiver_id,
                "receiver_model": EXPECTED_MODELS[receiver_id],
                "conditions": {},
            }
            for receiver_id in range(5)
        ]
        aggregates = {}
        for condition, switches in CONDITION_SWITCHES.items():
            condition_root = root / condition
            condition_root.mkdir()
            config = condition_root / "source.yaml"
            config.write_text(f"condition: {condition}\n", encoding="utf-8")
            result = condition_root / "social_results.csv"
            result.write_text("fixture\n", encoding="utf-8")
            receivers = {receiver_id: self._receiver(receiver_id) for receiver_id in range(5)}
            contexts["conditions"][condition] = {
                "config_source": config.resolve(),
                "run_dir": condition_root.resolve(),
                "social_results_csv": result.resolve(),
                "receivers": receivers,
            }
            sources[condition] = {
                "config": str(config.resolve()),
                "config_sha256": sha256_file(config),
                "run_dir": str(condition_root.resolve()),
                "social_results_csv": str(result.resolve()),
                "social_results_sha256": sha256_file(result),
            }
            for receiver_id, row in receivers.items():
                per_receiver[receiver_id]["conditions"][condition] = {
                    **row["metrics"],
                    "loss_mean": row["losses"],
                    "training_seconds": row["training_seconds"],
                    "optimizer_steps": row["optimizer_steps"],
                    "raw_external_images": row["raw_external_images"],
                    "logit_bytes": row["logit_bytes"],
                    "prototype_initialized_classes": row["prototype_initialized_classes"],
                }
            aggregates[condition] = {
                "switches": {"fr": switches[0], "kd": switches[1], "supcon": switches[2]},
                "metrics_mean": {
                    key: sum(row["metrics"][key] for row in receivers.values()) / 5.0
                    for key in ("global", "new", "expert")
                },
                "loss_mean": {
                    key: sum(row["losses"][key] for row in receivers.values()) / 5.0
                    for key in LOSS_FIELDS
                },
                "resources": {
                    f"{key}_{kind}": (
                        sum(row[key] for row in receivers.values())
                        if kind == "total"
                        else sum(row[key] for row in receivers.values()) / 5.0
                    )
                    for key in (
                        "training_seconds",
                        "optimizer_steps",
                        "raw_external_images",
                        "logit_bytes",
                    )
                    for kind in ("total", "mean")
                },
            }
        summary = {
            "status": "complete_diagnostic",
            "formal_result": False,
            "paper_eligible": False,
            "receiver_ids": list(range(5)),
            "seed": 0,
            "agents": 5,
            "classes_per_agent": 20,
            "ipc": 10,
            "condition_order": [variant_id(*switches) for switches in ALL_SWITCHES],
            "sources": sources,
            "per_receiver": per_receiver,
            "mean_over_five_receivers": aggregates,
        }
        path = root / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path, contexts, summary

    def test_summary_sources_and_values_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path, contexts, _summary = self._summary_fixture(tmp_dir)
            report = _validate_summary(path, contexts)
        self.assertEqual(report["status"], "complete_diagnostic")

    def test_summary_value_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path, contexts, summary = self._summary_fixture(tmp_dir)
            summary["per_receiver"][0]["conditions"]["fr1_kd0_sc0"]["global"] += 1.0
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(LossAblationFinalAuditError, "global mismatch"):
                _validate_summary(path, contexts)

    def test_report_writer_is_exclusive_and_has_no_formal_destination(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "audit.json"
            _write_json_exclusive({"status": "passed"}, path)
            with self.assertRaisesRegex(LossAblationFinalAuditError, "refusing to overwrite"):
                _write_json_exclusive({"status": "passed"}, path)
        with self.assertRaisesRegex(LossAblationFinalAuditError, "outputs/diagnostics"):
            _report_path("paper_tables/loss_ablation_checkpoint_audit.json")


if __name__ == "__main__":
    unittest.main()

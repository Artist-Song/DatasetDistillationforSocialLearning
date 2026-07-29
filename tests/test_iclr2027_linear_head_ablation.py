import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.append(str(DSDM_ROOT))

from config_adapter import load_config
from models.convnet import ConvNet
from models.cosine_classifier import get_output_classifier
from scripts.prepare_iclr2027_dkp_protocol import _variant as build_cosine_config
import scripts.prepare_iclr2027_linear_head_ablation as linear_config_generator
from scripts.prepare_iclr2027_linear_head_ablation import (
    CATALOG,
    CE_ONLY_RUN,
    EXPERT_RUN,
    FULL_RUN,
    build_linear_config,
    validate_linear_config,
)
from scripts.summarize_iclr2027_dkp_first_round import REQUIRED_RESULT_FIELDS
from scripts.summarize_iclr2027_linear_head_ablation import (
    DEFAULT_CE_CONFIG,
    DEFAULT_EXPERT_CONFIG,
    DEFAULT_FULL_CONFIG,
    LINEAR_RESULT_FIELDS,
    SummaryError,
    build_summary,
    main as summary_main,
)
from scripts.validate_iclr2027_linear_communication import (
    LinearCommunicationPreflightError,
    validate_config_contract,
)
from scripts.validate_iclr2027_linear_experts import (
    PreflightError,
    sha256_file,
    validate_expert_artifact,
    validate_protocol_config,
)


class TinyLinearAuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 100)

    @staticmethod
    def _features(images):
        return images.mean(dim=(2, 3))

    def forward(self, images):
        return self.classifier(self._features(images))

    def get_feature(self, images, _idx_from, _idx_to):
        return self._features(images)


def tiny_linear_audit_model_builder(_args):
    return TinyLinearAuditModel()


class ICLR2027LinearHeadConfigAndExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert_config = load_config(DEFAULT_EXPERT_CONFIG)
        cls.ce_config = load_config(DEFAULT_CE_CONFIG)
        cls.full_config = load_config(DEFAULT_FULL_CONFIG)
        cls.args_by_agent = validate_protocol_config(cls.expert_config, DEFAULT_EXPERT_CONFIG)

    @staticmethod
    def _conv3_builder(_args):
        return ConvNet(
            100,
            net_norm="instance",
            net_depth=3,
            net_width=128,
            channel=3,
            im_size=(32, 32),
            classifier_type="linear",
        )

    def _write_agent0_artifact(self, run_dir):
        args = self.args_by_agent[0]
        checkpoint_dir = Path(run_dir) / "agents/agent_0/checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "expert_model.pt"
        manifest_path = checkpoint_dir / "expert_manifest.json"
        model = self._conv3_builder(args)
        torch.save(model.state_dict(), checkpoint_path)
        manifest = {
            "agent_id": 0,
            "role": "fully_converged_agent_expert_and_logit_teacher",
            "selection_rule": "best_local_validation_accuracy",
            "test_used_for_selection": False,
            "max_epochs": 500,
            "selected_epoch": 5,
            "best_validation_accuracy": 48.0,
            "official_test_accuracy_report_only": 50.0,
            "validation_fraction": 0.1,
            "retrained_on_full_local_train": True,
            "global_output_dim": 100,
            "labels": "global",
            "active_class_ids": list(args.active_class_ids),
            "masked_local_ce": True,
            "classifier": {
                "type": "linear",
                "bias": True,
                "feature_normalization": False,
                "weight_normalization": False,
                "scale_parameterization": None,
                "final_scale": None,
                "scale_weight_decay": 0.0,
            },
            "expert_path": str(checkpoint_path.resolve()),
            "expert_sha256": sha256_file(checkpoint_path),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return checkpoint_path, manifest_path, manifest, model

    def test_generated_configs_are_matched_linear_and_do_not_mutate_cosine_source(self):
        source = build_cosine_config("source_cosine", "full")
        source_before = copy.deepcopy(source)
        with mock.patch.object(linear_config_generator, "_cosine_variant", return_value=source):
            build_linear_config(FULL_RUN, "full")
        self.assertEqual(source, source_before)

        variants = {
            "local_expert": EXPERT_RUN,
            "ce_only": CE_ONLY_RUN,
            "full": FULL_RUN,
        }
        for variant, run_name in variants.items():
            generated = build_linear_config(run_name, variant)
            validate_linear_config(generated, variant)
            self.assertEqual(generated["project"]["run_name"], run_name)
            self.assertFalse(generated["project"]["paper_eligible"])
            self.assertEqual(generated["communication"]["pool_catalog"], CATALOG)
            for model in generated["model_pool"]["models"].values():
                self.assertEqual(model["classifier"], {"type": "linear"})
                self.assertTrue(model["expert_training"]["masked_local_ce"])
        self.assertEqual(
            self.ce_config["social_learning"]["receiver"]["loss_switches"],
            {"fr": False, "kd": False, "supcon": False},
        )
        self.assertEqual(
            self.full_config["social_learning"]["receiver"]["loss_switches"],
            {"fr": True, "kd": True, "supcon": True},
        )
        self.assertEqual(
            self.ce_config["social_learning"]["receiver"]["checkpoint_retention"],
            "final_only",
        )

    def test_config_contract_rejects_classifier_and_retention_drift(self):
        class_split, model_split = validate_config_contract(
            self.expert_config,
            self.ce_config,
            self.full_config,
        )
        self.assertEqual(sorted(class_split), list(range(5)))
        self.assertEqual(sorted(model_split), list(range(5)))

        wrong_head = copy.deepcopy(self.full_config)
        wrong_head["model_pool"]["models"]["alexnet"]["classifier"] = {"type": "cosine"}
        with self.assertRaisesRegex(LinearCommunicationPreflightError, "matched linear"):
            validate_config_contract(self.expert_config, self.ce_config, wrong_head)

        wrong_retention = copy.deepcopy(self.ce_config)
        wrong_retention["social_learning"]["receiver"]["checkpoint_retention"] = "all"
        with self.assertRaisesRegex(LinearCommunicationPreflightError, "final_only"):
            validate_config_contract(self.expert_config, wrong_retention, self.full_config)

    def test_launcher_gates_disk_before_single_threaded_config_generation(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/run_iclr2027_linear_head_ablation.sh"
        ).read_text(encoding="utf-8")
        disk_gate = 'if (( available_kib < MIN_FREE_KIB )); then'
        generator = (
            'env OMP_NUM_THREADS=1 "$PYTHON_BIN" '
            'scripts/prepare_iclr2027_linear_head_ablation.py'
        )
        self.assertIn("MIN_FREE_KIB=$((2 * 1024 * 1024))", launcher)
        self.assertIn(disk_gate, launcher)
        self.assertIn(generator, launcher)
        self.assertLess(launcher.index(disk_gate), launcher.index(generator))

    def test_linear_checkpoint_strictly_loads_and_exposes_prototype_statistics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            _, _, _, model = self._write_agent0_artifact(run_dir)
            report = validate_expert_artifact(
                self.args_by_agent[0],
                0,
                run_dir,
                min_local_test_accuracy=49.0,
                device="cpu",
                model_builder=self._conv3_builder,
            )
            classifier = get_output_classifier(model)
            local_index = torch.tensor(self.args_by_agent[0].active_class_ids)
            expected_alpha = float(classifier.weight.detach().index_select(0, local_index).norm(dim=1).mean())
            expected_beta = float(classifier.bias.detach().index_select(0, local_index).mean())

        self.assertEqual(report["output_shape"], [2, 100])
        self.assertEqual(report["feature_shapes"], [[2, 128, 4, 4]])
        self.assertAlmostEqual(report["prototype_alpha"], expected_alpha)
        self.assertAlmostEqual(report["prototype_beta"], expected_beta)
        self.assertGreater(report["prototype_alpha"], 0.0)

    def test_linear_expert_sha_and_manifest_geometry_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            _, manifest_path, manifest, _ = self._write_agent0_artifact(run_dir)
            manifest["expert_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "SHA-256"):
                validate_expert_artifact(
                    self.args_by_agent[0], 0, run_dir, model_builder=self._conv3_builder
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            _, manifest_path, manifest, _ = self._write_agent0_artifact(run_dir)
            manifest["classifier"]["bias"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "bias"):
                validate_expert_artifact(
                    self.args_by_agent[0], 0, run_dir, model_builder=self._conv3_builder
                )


class ICLR2027LinearHeadSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert_config = load_config(DEFAULT_EXPERT_CONFIG)
        cls.ce_config = load_config(DEFAULT_CE_CONFIG)
        cls.full_config = load_config(DEFAULT_FULL_CONFIG)
        cls.args_by_agent = validate_protocol_config(cls.expert_config, DEFAULT_EXPERT_CONFIG)

    def _write_experts(self, run_dir):
        accuracies = {}
        checkpoint_hashes = {}
        for agent_id, args in self.args_by_agent.items():
            checkpoint_dir = Path(run_dir) / f"agents/agent_{agent_id}/checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "expert_model.pt"
            checkpoint_path.write_bytes(f"linear-expert-{agent_id}".encode("ascii"))
            checkpoint_hash = sha256_file(checkpoint_path)
            accuracy = 60.0 + agent_id
            manifest = {
                "agent_id": agent_id,
                "role": "fully_converged_agent_expert_and_logit_teacher",
                "test_used_for_selection": False,
                "retrained_on_full_local_train": True,
                "masked_local_ce": True,
                "labels": "global",
                "global_output_dim": 100,
                "active_class_ids": list(args.active_class_ids),
                "classifier": {
                    "type": "linear",
                    "bias": True,
                    "feature_normalization": False,
                    "weight_normalization": False,
                },
                "official_test_accuracy_report_only": accuracy,
                "selected_epoch": 100 + agent_id,
                "expert_path": str(checkpoint_path.resolve()),
                "expert_sha256": checkpoint_hash,
            }
            (checkpoint_dir / "expert_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            accuracies[agent_id] = accuracy
            checkpoint_hashes[agent_id] = checkpoint_hash
        return accuracies, checkpoint_hashes

    def _result_rows(self, run_dir, config, variant, accuracies, checkpoint_hashes):
        is_full = variant == "full"
        receiver_cfg = config["social_learning"]["receiver"]
        logits_cfg = config["logits"]
        rows = []
        for receiver_id, args in self.args_by_agent.items():
            checkpoint_dir = (
                Path(run_dir)
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
            )
            checkpoint_dir.mkdir(parents=True)
            final_path = checkpoint_dir / "after_social.pt"
            torch.save(TinyLinearAuditModel().state_dict(), final_path)
            final_sha = sha256_file(final_path)
            alpha = 2.0 + receiver_id
            beta = 0.1 + receiver_id / 10.0
            provenance = {
                "protocol": "dkp_sl_v1",
                "dkp_variant": variant,
                "loss_switches": receiver_cfg["loss_switches"],
                "receiver_agent": receiver_id,
                "receiver_model": args.model_name,
                "receiver_seed": 100_000 + receiver_id,
                "receiver_init_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "fr_teacher_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "classifier_type": "linear",
                "prototype_initialization": {
                    "classifier_type": "linear",
                    "mode": "linear_local_row_norm_bias_mean",
                    "alpha": alpha,
                    "beta": beta,
                    "external_weight_norm_min": alpha,
                    "external_weight_norm_max": alpha,
                    "initialized_classes": list(range(80)),
                },
                "checkpoint_retention": "final_only",
                "checkpoint_artifacts": {
                    "after_social": {"path": str(final_path.resolve()), "sha256": final_sha}
                },
                "statistics": {"optimizer_steps": 9_420},
            }
            provenance_path = checkpoint_dir / "receiver_provenance.json"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            losses = {
                "loss": 1.0 + receiver_id,
                "loss_cls": 0.8 + receiver_id,
                "loss_ce_local": 0.2 + receiver_id,
                "loss_ce_external": 0.6 + receiver_id,
                "loss_fr": 0.1 + receiver_id if is_full else 0.0,
                "loss_kd": 0.3 + receiver_id if is_full else 0.0,
                "loss_sc": 0.05 + receiver_id if is_full else 0.0,
            }
            metric_offset = 30.0 if is_full else 20.0
            rows.append(
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
                    "use_fr": str(is_full).lower(),
                    "lambda_fr": receiver_cfg["lambda_fr"],
                    "lambda_sc": receiver_cfg["lambda_sc"],
                    "supcon_temperature": receiver_cfg["supcon_temperature"],
                    "use_logits": str(is_full).lower(),
                    "communication_mode": config["communication"]["mode"],
                    "use_generalist_logits": "false",
                    "lambda_kd": logits_cfg["lambda_kd"],
                    "kd_temperature": logits_cfg["temperature"],
                    "ipc": 10,
                    "self_real_images": 10_000,
                    "external_comm_images": 800,
                    "external_comm_logit_bytes": 128_000 if is_full else 0,
                    "external_comm_generalist_logit_bytes": 0,
                    "acc_global_before": accuracies[receiver_id] / 5.0,
                    "acc_expert_before": accuracies[receiver_id],
                    "acc_global_after": metric_offset + receiver_id,
                    "acc_expert_after": metric_offset + 20.0 + receiver_id,
                    "acc_new_after": metric_offset - 5.0 + receiver_id,
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
                    "receiver_init_checkpoint_sha256": checkpoint_hashes[receiver_id],
                    "fr_teacher_checkpoint_sha256": checkpoint_hashes[receiver_id],
                    "time": "2026-07-27T23:00:00",
                    **losses,
                }
            )
        return rows

    @staticmethod
    def _write_results(run_dir, rows):
        path = Path(run_dir) / "metrics/social_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted(REQUIRED_RESULT_FIELDS | LINEAR_RESULT_FIELDS)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    def _fixture(self, root):
        expert_run = Path(root) / "experts"
        ce_run = Path(root) / "ce"
        full_run = Path(root) / "full"
        accuracies, checkpoint_hashes = self._write_experts(expert_run)
        ce_rows = self._result_rows(
            ce_run, self.ce_config, "ce_only", accuracies, checkpoint_hashes
        )
        full_rows = self._result_rows(
            full_run, self.full_config, "full", accuracies, checkpoint_hashes
        )
        self._write_results(ce_run, ce_rows)
        self._write_results(full_run, full_rows)
        return expert_run, ce_run, full_run, ce_rows, full_rows

    @staticmethod
    def _build(expert_run, ce_run, full_run):
        return build_summary(
            expert_config=DEFAULT_EXPERT_CONFIG,
            ce_config=DEFAULT_CE_CONFIG,
            full_config=DEFAULT_FULL_CONFIG,
            expert_run_dir=expert_run,
            ce_run_dir=ce_run,
            full_run_dir=full_run,
            model_builder=tiny_linear_audit_model_builder,
        )

    def test_complete_linear_runs_require_five_receivers_and_report_means(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            summary = self._build(expert_run, ce_run, full_run)

        self.assertEqual(summary["status"], "complete_diagnostic")
        self.assertFalse(summary["formal_result"])
        self.assertFalse(summary["paper_eligible"])
        self.assertEqual(summary["classifier"], "linear")
        self.assertEqual(len(summary["per_receiver"]), 5)
        self.assertEqual(summary["mean_over_five_receivers"]["local_linear_expert"]["expert"], 62.0)
        self.assertEqual(summary["mean_over_five_receivers"]["ce_only"]["metrics_mean"]["global"], 22.0)
        self.assertEqual(summary["mean_over_five_receivers"]["full_dkp_sl"]["metrics_mean"]["global"], 32.0)
        self.assertEqual(
            summary["mean_over_five_receivers"]["full_dkp_sl"]["resources"]["logit_bytes_total"],
            640_000,
        )

    def test_summary_cli_writes_once_and_has_no_overwrite_option(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            output = Path(tmp_dir) / "linear_summary.json"
            argv = [
                "--expert-config",
                str(DEFAULT_EXPERT_CONFIG),
                "--ce-config",
                str(DEFAULT_CE_CONFIG),
                "--full-config",
                str(DEFAULT_FULL_CONFIG),
                "--expert-run-dir",
                str(expert_run),
                "--ce-run-dir",
                str(ce_run),
                "--full-run-dir",
                str(full_run),
                "--output-json",
                str(output),
            ]
            with mock.patch(
                "scripts.summarize_iclr2027_linear_head_ablation._default_model_builder",
                tiny_linear_audit_model_builder,
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(summary_main(argv), 0)
                self.assertEqual(summary_main(argv), 1)
                with self.assertRaises(SystemExit):
                    summary_main(argv + ["--overwrite"])
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(payload["formal_result"])

    def test_summary_rejects_partial_receiver_and_linear_artifact_drift(self):
        mutations = [
            ("retention", "ce", 0, "checkpoint_retention", "all", "final_only"),
            ("alpha", "full", 1, "prototype_alpha", 0.0, "not positive"),
        ]
        for name, target, index, field, value, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp_dir:
                expert_run, ce_run, full_run, ce_rows, full_rows = self._fixture(tmp_dir)
                rows = ce_rows if target == "ce" else full_rows
                rows[index][field] = value
                self._write_results(ce_run if target == "ce" else full_run, rows)
                with self.assertRaisesRegex(SummaryError, error):
                    self._build(expert_run, ce_run, full_run)

        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, ce_rows, _ = self._fixture(tmp_dir)
            self._write_results(ce_run, ce_rows[:-1])
            with self.assertRaisesRegex(SummaryError, "expected exactly five"):
                self._build(expert_run, ce_run, full_run)

    def test_summary_rejects_redundant_or_misdirected_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            checkpoint_dir = ce_run / "social_learning/receiver_agent_0/checkpoints/dkp_sl_v1_ce_only"
            (checkpoint_dir / "before_social.pt").write_bytes(b"redundant")
            with self.assertRaisesRegex(SummaryError, "retained before_social"):
                self._build(expert_run, ce_run, full_run)

        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            provenance_path = (
                full_run
                / "social_learning/receiver_agent_1/checkpoints/dkp_sl_v1_full/receiver_provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["checkpoint_artifacts"]["after_social"]["path"] = str(
                provenance_path.resolve()
            )
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            result_path = full_run / "metrics/social_results.csv"
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["receiver_provenance_sha256"] = sha256_file(provenance_path)
            self._write_results(full_run, rows)
            with self.assertRaisesRegex(SummaryError, "provenance path mismatch"):
                self._build(expert_run, ce_run, full_run)

    def test_summary_strict_load_rejects_nonfinite_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            checkpoint_dir = (
                full_run
                / "social_learning/receiver_agent_2/checkpoints/dkp_sl_v1_full"
            )
            final_path = checkpoint_dir / "after_social.pt"
            state = torch.load(final_path, map_location="cpu", weights_only=True)
            state["classifier.weight"][0, 0] = float("nan")
            torch.save(state, final_path)
            final_sha = sha256_file(final_path)

            provenance_path = checkpoint_dir / "receiver_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["checkpoint_artifacts"]["after_social"]["sha256"] = final_sha
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            provenance_sha = sha256_file(provenance_path)

            result_path = full_run / "metrics/social_results.csv"
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[2]["after_social_checkpoint_sha256"] = final_sha
            rows[2]["receiver_provenance_sha256"] = provenance_sha
            self._write_results(full_run, rows)
            with self.assertRaisesRegex(SummaryError, "non-finite tensors"):
                self._build(expert_run, ce_run, full_run)


if __name__ == "__main__":
    unittest.main()

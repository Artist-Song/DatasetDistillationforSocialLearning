import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from config_adapter import load_config
from scripts.summarize_iclr2027_dkp_first_round import (
    DEFAULT_CE_CONFIG,
    DEFAULT_EXPERT_CONFIG,
    DEFAULT_FULL_CONFIG,
    REQUIRED_RESULT_FIELDS,
    SummaryError,
    _validate_output_path,
    build_summary,
    main,
)
from scripts.validate_iclr2027_cosine_experts import sha256_file, validate_protocol_config


class ICLR2027FirstRoundSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert_cfg = load_config(DEFAULT_EXPERT_CONFIG)
        cls.ce_cfg = load_config(DEFAULT_CE_CONFIG)
        cls.full_cfg = load_config(DEFAULT_FULL_CONFIG)
        cls.expert_args = validate_protocol_config(cls.expert_cfg, DEFAULT_EXPERT_CONFIG)

    def _write_experts(self, run_dir):
        local_accuracies = {}
        checkpoint_hashes = {}
        for agent_id, args in self.expert_args.items():
            checkpoint_dir = Path(run_dir) / f"agents/agent_{agent_id}/checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "expert_model.pt"
            checkpoint_path.write_bytes(f"diagnostic-checkpoint-{agent_id}".encode("ascii"))
            checkpoint_sha256 = sha256_file(checkpoint_path)
            local_accuracy = 60.0 + agent_id
            manifest = {
                "agent_id": agent_id,
                "role": "fully_converged_agent_expert_and_logit_teacher",
                "test_used_for_selection": False,
                "masked_local_ce": True,
                "labels": "global",
                "global_output_dim": 100,
                "active_class_ids": list(args.active_class_ids),
                "classifier": {"type": "cosine"},
                "official_test_accuracy_report_only": local_accuracy,
                "selected_epoch": 100 + agent_id,
                "expert_sha256": checkpoint_sha256,
            }
            (checkpoint_dir / "expert_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            local_accuracies[agent_id] = local_accuracy
            checkpoint_hashes[agent_id] = checkpoint_sha256
        return local_accuracies, checkpoint_hashes

    def _result_rows(self, cfg, variant, local_accuracies, checkpoint_hashes):
        receiver_cfg = cfg["social_learning"]["receiver"]
        logits_cfg = cfg["logits"]
        rows = []
        for receiver_id, args in self.expert_args.items():
            is_full = variant == "full"
            metric_offset = 30.0 if is_full else 20.0
            losses = {
                "loss": 1.0 + receiver_id,
                "loss_cls": 0.8 + receiver_id,
                "loss_ce_local": 0.2 + receiver_id,
                "loss_ce_external": 0.6 + receiver_id,
                "loss_fr": 0.1 + receiver_id if is_full else 0.0,
                "loss_kd": 0.3 + receiver_id if is_full else 0.0,
                "loss_sc": 0.05 + receiver_id if is_full else 0.0,
            }
            row = {
                "run_name": cfg["project"]["run_name"],
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
                "communication_mode": cfg["communication"]["mode"],
                "use_generalist_logits": "false",
                "lambda_kd": logits_cfg["lambda_kd"],
                "kd_temperature": logits_cfg["temperature"],
                "ipc": 10,
                "self_real_images": 10_000,
                "external_comm_images": 800,
                "external_comm_logit_bytes": 128_000 if is_full else 0,
                "external_comm_generalist_logit_bytes": 0,
                "acc_global_before": local_accuracies[receiver_id] / 5.0,
                "acc_expert_before": local_accuracies[receiver_id],
                "acc_global_after": metric_offset + receiver_id,
                "acc_expert_after": metric_offset + 20.0 + receiver_id,
                "acc_new_after": metric_offset - 5.0 + receiver_id,
                "optimizer_steps": 9_420,
                "training_seconds": 100.0 + receiver_id,
                "prototype_initialized_classes": 80,
                "receiver_seed": 100_000 + receiver_id,
                "receiver_init_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "fr_teacher_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "time": "2026-07-27T23:00:00",
                **losses,
            }
            rows.append(row)
        return rows

    def _write_results(self, run_dir, rows):
        path = Path(run_dir) / "metrics/social_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(REQUIRED_RESULT_FIELDS)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
        return path

    def _fixture(self, root):
        expert_run = Path(root) / "experts"
        ce_run = Path(root) / "ce"
        full_run = Path(root) / "full"
        local_accuracies, checkpoint_hashes = self._write_experts(expert_run)
        ce_rows = self._result_rows(self.ce_cfg, "ce_only", local_accuracies, checkpoint_hashes)
        full_rows = self._result_rows(self.full_cfg, "full", local_accuracies, checkpoint_hashes)
        self._write_results(ce_run, ce_rows)
        self._write_results(full_run, full_rows)
        return expert_run, ce_run, full_run, ce_rows, full_rows

    def _build(self, expert_run, ce_run, full_run):
        return build_summary(
            expert_config=DEFAULT_EXPERT_CONFIG,
            ce_config=DEFAULT_CE_CONFIG,
            full_config=DEFAULT_FULL_CONFIG,
            expert_run_dir=expert_run,
            ce_run_dir=ce_run,
            full_run_dir=full_run,
        )

    def test_complete_runs_report_per_receiver_and_exact_arithmetic_means(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            summary = self._build(expert_run, ce_run, full_run)

        self.assertEqual(summary["status"], "complete_diagnostic")
        self.assertFalse(summary["formal_result"])
        self.assertFalse(summary["paper_eligible"])
        self.assertEqual(len(summary["per_receiver"]), 5)
        self.assertEqual(summary["per_receiver"][0]["ce_only"]["global"], 20.0)
        self.assertIn("loss_mean", summary["per_receiver"][0]["full_dkp_sl"])
        self.assertEqual(summary["mean_over_five_receivers"]["local_expert"]["expert"], 62.0)
        self.assertEqual(summary["mean_over_five_receivers"]["ce_only"]["metrics_mean"]["global"], 22.0)
        self.assertEqual(summary["mean_over_five_receivers"]["full_dkp_sl"]["metrics_mean"]["global"], 32.0)
        self.assertEqual(
            summary["mean_over_five_receivers"]["ce_only"]["resources"]["raw_external_images_total"],
            4_000,
        )
        self.assertEqual(summary["mean_over_five_receivers"]["ce_only"]["resources"]["logit_bytes_total"], 0)
        self.assertEqual(
            summary["mean_over_five_receivers"]["full_dkp_sl"]["resources"]["logit_bytes_total"],
            640_000,
        )

    def test_cli_writes_atomic_diagnostic_json_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            output = Path(tmp_dir) / "summary.json"
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
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 1)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["receiver_ids"], list(range(5)))
        self.assertFalse(payload["formal_result"])

    def test_formal_result_destinations_are_rejected(self):
        with self.assertRaises(SummaryError):
            _validate_output_path("paper_tables/first_round.json")
        with self.assertRaises(SummaryError):
            _validate_output_path("outputs/experiment_registry/first_round.json")

    def test_partial_or_duplicate_receiver_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, ce_rows, _ = self._fixture(tmp_dir)
            self._write_results(ce_run, ce_rows[:-1])
            with self.assertRaisesRegex(SummaryError, "expected exactly five"):
                self._build(expert_run, ce_run, full_run)

            duplicate_rows = list(ce_rows)
            duplicate_rows[-1] = dict(duplicate_rows[-1], receiver_agent=3)
            self._write_results(ce_run, duplicate_rows)
            with self.assertRaisesRegex(SummaryError, "receiver ids are incomplete"):
                self._build(expert_run, ce_run, full_run)

    def test_protocol_accounting_and_finite_metric_fail_closed(self):
        mutations = [
            ("ce_bytes", "ce", 0, "external_comm_logit_bytes", 1, "expected 0"),
            ("full_steps", "full", 1, "optimizer_steps", 0, "expected 9420"),
            ("prototype", "full", 2, "prototype_initialized_classes", 79, "expected 80"),
            ("variant", "full", 3, "dkp_variant", "ce_only", "variant mismatch"),
            ("nan", "ce", 4, "acc_global_after", "nan", "not finite"),
            ("model", "ce", 2, "receiver_model", "resnet18_standard", "model mismatch"),
        ]
        for name, target, row_index, field, value, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp_dir:
                expert_run, ce_run, full_run, ce_rows, full_rows = self._fixture(tmp_dir)
                rows = ce_rows if target == "ce" else full_rows
                rows[row_index][field] = value
                self._write_results(ce_run if target == "ce" else full_run, rows)
                with self.assertRaisesRegex(SummaryError, error):
                    self._build(expert_run, ce_run, full_run)

    def test_missing_or_nonfinite_expert_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            (expert_run / "agents/agent_4/checkpoints/expert_manifest.json").unlink()
            with self.assertRaisesRegex(SummaryError, "expected exactly five"):
                self._build(expert_run, ce_run, full_run)

        with tempfile.TemporaryDirectory() as tmp_dir:
            expert_run, ce_run, full_run, _, _ = self._fixture(tmp_dir)
            path = expert_run / "agents/agent_4/checkpoints/expert_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["official_test_accuracy_report_only"] = "nan"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SummaryError, "not finite"):
                self._build(expert_run, ce_run, full_run)


if __name__ == "__main__":
    unittest.main()

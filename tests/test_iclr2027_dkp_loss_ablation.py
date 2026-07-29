import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from config_adapter import build_dsdm_args_from_config, load_config
from scripts.prepare_iclr2027_dkp_loss_ablation import (
    MISSING_COMBINATIONS,
    build_variants,
    config_filename,
    variant_id,
    write_variants,
)
from scripts.summarize_iclr2027_dkp_first_round import REQUIRED_RESULT_FIELDS
from scripts.summarize_iclr2027_dkp_loss_ablation import (
    ALL_SWITCHES,
    LossAblationSummaryError,
    _validate_output_path,
    build_summary,
    default_config_paths,
    main,
    parse_args,
)
from scripts.validate_iclr2027_cosine_experts import (
    DEFAULT_CONFIG as DEFAULT_EXPERT_CONFIG,
    sha256_file,
    validate_protocol_config,
)
from social_output_manager import SOCIAL_RESULT_FIELDS
from social_trainer import resolve_dkp_loss_switches


class ICLR2027DKPLossAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert_cfg = load_config(DEFAULT_EXPERT_CONFIG)
        cls.expert_args = validate_protocol_config(cls.expert_cfg, DEFAULT_EXPERT_CONFIG)
        cls.config_paths = default_config_paths()

    def test_generator_emits_exactly_six_missing_fixed_diagnostics(self):
        variants = build_variants()
        expected_files = {config_filename(*switches) for switches in MISSING_COMBINATIONS}
        self.assertEqual(set(variants), expected_files)
        self.assertEqual(len(variants), 6)
        for switches in MISSING_COMBINATIONS:
            fr, kd, sc = switches
            config = variants[config_filename(*switches)]
            receiver = config["social_learning"]["receiver"]
            self.assertFalse(config["project"]["paper_eligible"])
            self.assertEqual(receiver["checkpoint_retention"], "final_only")
            self.assertEqual(receiver["loss_switches"], {"fr": fr, "kd": kd, "supcon": sc})
            self.assertIs(config["communication"]["use_sender_logits"], kd)
            self.assertEqual(receiver["lambda_fr"], 0.2 if fr else 0.0)
            self.assertEqual(config["logits"]["lambda_kd"], 0.6 if kd else 0.0)
            self.assertEqual(receiver["lambda_sc"], 0.1 if sc else 0.0)

    def test_checked_in_configs_are_exact_generator_outputs(self):
        variants = build_variants()
        for filename, expected in variants.items():
            self.assertEqual(load_config(Path("configs/iclr2027") / filename), expected)

    def test_config_adapter_preserves_explicit_boolean_switches_and_retention(self):
        for switches in MISSING_COMBINATIONS:
            config = load_config(self.config_paths[variant_id(*switches)])
            args = build_dsdm_args_from_config(config)
            self.assertEqual(
                args.dkp_loss_switches,
                {"fr": switches[0], "kd": switches[1], "supcon": switches[2]},
            )
            self.assertEqual(args.receiver_checkpoint_retention, "final_only")

    def test_overnight_launcher_runs_standard_resnet_receivers_together(self):
        launcher = Path("scripts/run_iclr2027_dkp_loss_ablation_overnight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('run_receiver_group "$condition" "$config" 3 4', launcher)
        self.assertNotIn('run_receiver_group "$condition" "$config" 3\n', launcher)
        self.assertNotIn('run_receiver_group "$condition" "$config" 4\n', launcher)

    def test_generator_refuses_different_existing_config(self):
        variants = build_variants()
        filename = next(iter(variants))
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp)
            (target_dir / filename).write_text("project: {run_name: collision}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                write_variants(variants, target_dir)

    def test_switch_parser_preserves_endpoints_and_rejects_conflicts(self):
        self.assertEqual(
            resolve_dkp_loss_switches("ce_only"),
            {"fr": False, "kd": False, "supcon": False},
        )
        self.assertEqual(
            resolve_dkp_loss_switches("full"),
            {"fr": True, "kd": True, "supcon": True},
        )
        switches = {"fr": True, "kd": False, "supcon": True}
        self.assertEqual(resolve_dkp_loss_switches("ablation_fr1_kd0_sc1", switches), switches)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_dkp_loss_switches(
                "ablation_fr1_kd0_sc1",
                {"fr": False, "kd": False, "supcon": True},
            )

    def _write_experts(self, run_dir):
        accuracies = {}
        checkpoint_hashes = {}
        for receiver_id, args in self.expert_args.items():
            checkpoint_dir = Path(run_dir) / f"agents/agent_{receiver_id}/checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "expert_model.pt"
            checkpoint.write_bytes(f"expert-{receiver_id}".encode("ascii"))
            checkpoint_sha = sha256_file(checkpoint)
            accuracy = 70.0 + receiver_id
            manifest = {
                "agent_id": receiver_id,
                "role": "fully_converged_agent_expert_and_logit_teacher",
                "test_used_for_selection": False,
                "masked_local_ce": True,
                "labels": "global",
                "global_output_dim": 100,
                "active_class_ids": list(args.active_class_ids),
                "classifier": {"type": "cosine"},
                "official_test_accuracy_report_only": accuracy,
                "selected_epoch": 100 + receiver_id,
                "expert_sha256": checkpoint_sha,
            }
            (checkpoint_dir / "expert_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            accuracies[receiver_id] = accuracy
            checkpoint_hashes[receiver_id] = checkpoint_sha
        return accuracies, checkpoint_hashes

    def _write_condition(self, run_dir, condition, accuracies, checkpoint_hashes):
        switches = tuple(bool(int(value)) for value in (condition[2], condition[6], condition[10]))
        fr, kd, sc = switches
        config = load_config(self.config_paths[condition])
        args_by_agent = validate_protocol_config(config, self.config_paths[condition])
        variant = "ce_only" if switches == (False, False, False) else "full" if switches == (True, True, True) else f"ablation_{condition}"
        method = "DKP_CE_ONLY" if variant == "ce_only" else "DKP_SL" if variant == "full" else "DKP_SL_ABLATION"
        rows = []
        for receiver_id, args in args_by_agent.items():
            losses = {
                "loss_cls": 1.0 + receiver_id * 0.01,
                "loss_ce_local": 0.5 + receiver_id * 0.01,
                "loss_ce_external": 1.125 + receiver_id * 0.01,
                "loss_fr": 0.1 + receiver_id * 0.01 if fr else 0.0,
                "loss_kd": 0.2 + receiver_id * 0.01 if kd else 0.0,
                "loss_sc": 0.3 + receiver_id * 0.01 if sc else 0.0,
            }
            losses["loss"] = losses["loss_cls"]
            losses["loss"] += (0.2 if fr else 0.0) * losses["loss_fr"]
            losses["loss"] += (0.6 if kd else 0.0) * losses["loss_kd"]
            losses["loss"] += (0.1 if sc else 0.0) * losses["loss_sc"]
            row = {
                "run_name": config["project"]["run_name"],
                "protocol": "dkp_sl_v1",
                "dkp_variant": variant,
                "receiver_agent": receiver_id,
                "receiver_model": args.model_name,
                "expert_classes": ",".join(str(value) for value in args.active_class_ids),
                "packet_method": "dsdm",
                "method": method,
                "init_mode": "expert",
                "self_data_mode": "real",
                "self_real_per_class": 0,
                "use_fr": str(fr).lower(),
                "lambda_fr": 0.2 if fr else 0.0,
                "lambda_sc": 0.1 if sc else 0.0,
                "supcon_temperature": 0.07,
                "use_logits": str(kd).lower(),
                "communication_mode": "logical_all_share_once",
                "use_generalist_logits": "false",
                "kd_mix_beta": 0.0,
                "lambda_kd": 0.6 if kd else 0.0,
                "kd_temperature": 2.0,
                "ipc": 10,
                "self_real_images": 10_000,
                "external_comm_images": 800,
                "external_comm_logit_bytes": 128_000 if kd else 0,
                "external_comm_generalist_logit_bytes": 0,
                "acc_global_before": accuracies[receiver_id] / 5.0,
                "acc_expert_before": accuracies[receiver_id],
                "acc_global_after": 20.0 + receiver_id,
                "acc_expert_after": 60.0 + receiver_id,
                "acc_new_after": 10.0 + receiver_id,
                "forgetting": 10.0,
                "loss_sender_kd": losses["loss_kd"],
                "loss_generalist_kd": 0.0,
                "receiver_augment": "true",
                "freeze_bn_stats": "false",
                "optimizer_steps": 9_420,
                "training_seconds": 100.0 + receiver_id,
                "prototype_initialized_classes": 80,
                "receiver_seed": 100_000 + receiver_id,
                "receiver_init_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "fr_teacher_checkpoint_sha256": checkpoint_hashes[receiver_id],
                "time": "2026-07-28T01:00:00",
                **losses,
            }
            if switches in MISSING_COMBINATIONS:
                checkpoint_dir = (
                    Path(run_dir)
                    / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
                )
                checkpoint_dir.mkdir(parents=True)
                after_path = checkpoint_dir / "after_social.pt"
                after_path.write_bytes(f"after-{condition}-{receiver_id}".encode("ascii"))
                after_sha = sha256_file(after_path)
                external_classes = sorted(set(range(100)) - set(args.active_class_ids))
                provenance = {
                    "protocol": "dkp_sl_v1",
                    "dkp_variant": variant,
                    "loss_switches": {"fr": fr, "kd": kd, "supcon": sc},
                    "receiver_agent": receiver_id,
                    "receiver_model": args.model_name,
                    "receiver_seed": 100_000 + receiver_id,
                    "receiver_init_checkpoint_sha256": checkpoint_hashes[receiver_id],
                    "fr_teacher_checkpoint_sha256": checkpoint_hashes[receiver_id],
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
                        "after_social": {"path": str(after_path.resolve()), "sha256": after_sha}
                    },
                    "statistics": {"optimizer_steps": 9_420},
                }
                provenance_path = checkpoint_dir / "receiver_provenance.json"
                provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
                row.update(
                    {
                        "classifier_type": "cosine",
                        "prototype_init_mode": "cosine_unit_weight_rows",
                        "prototype_alpha": "",
                        "prototype_beta": "",
                        "prototype_weight_norm_min": 1.0,
                        "prototype_weight_norm_max": 1.0,
                        "checkpoint_retention": "final_only",
                        "after_social_checkpoint_sha256": after_sha,
                        "receiver_provenance_path": str(provenance_path.resolve()),
                        "receiver_provenance_sha256": sha256_file(provenance_path),
                    }
                )
            rows.append(row)

        result_path = Path(run_dir) / "metrics/social_results.csv"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys([*SOCIAL_RESULT_FIELDS, *sorted(REQUIRED_RESULT_FIELDS)]))
        with result_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
        return rows

    def _fixture(self, root):
        expert_run = Path(root) / "experts"
        accuracies, checkpoint_hashes = self._write_experts(expert_run)
        run_dirs = {}
        rows = {}
        for switches in ALL_SWITCHES:
            condition = variant_id(*switches)
            run_dir = Path(root) / condition
            rows[condition] = self._write_condition(
                run_dir, condition, accuracies, checkpoint_hashes
            )
            run_dirs[condition] = run_dir
        return expert_run, run_dirs, rows

    def test_strict_summary_requires_and_aggregates_all_eight_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            expert_run, run_dirs, _ = self._fixture(tmp)
            summary = build_summary(
                expert_config=DEFAULT_EXPERT_CONFIG,
                config_paths=self.config_paths,
                expert_run_dir=expert_run,
                run_dirs=run_dirs,
            )
        self.assertEqual(summary["status"], "complete_diagnostic")
        self.assertFalse(summary["paper_eligible"])
        self.assertEqual(len(summary["condition_order"]), 8)
        self.assertEqual(len(summary["per_receiver"]), 5)
        self.assertEqual(
            summary["mean_over_five_receivers"]["fr1_kd0_sc0"]["resources"]["logit_bytes_total"],
            0,
        )
        self.assertEqual(
            summary["mean_over_five_receivers"]["fr0_kd1_sc0"]["resources"]["logit_bytes_total"],
            640_000,
        )

    def test_summary_fails_closed_on_bytes_steps_and_retention_artifacts(self):
        mutations = [
            ("fr1_kd0_sc0", 0, "external_comm_logit_bytes", 1, "expected 0"),
            ("fr0_kd1_sc0", 1, "optimizer_steps", 9_419, "expected 9420"),
            ("fr0_kd0_sc1", 2, "loss_fr", 0.1, "inactive loss_fr"),
        ]
        for condition, receiver_id, field, value, message in mutations:
            with self.subTest(condition=condition, field=field), tempfile.TemporaryDirectory() as tmp:
                expert_run, run_dirs, rows = self._fixture(tmp)
                rows[condition][receiver_id][field] = value
                result_path = run_dirs[condition] / "metrics/social_results.csv"
                with result_path.open("r", encoding="utf-8", newline="") as handle:
                    fieldnames = csv.DictReader(handle).fieldnames
                with result_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows[condition])
                with self.assertRaisesRegex(LossAblationSummaryError, message):
                    build_summary(
                        expert_config=DEFAULT_EXPERT_CONFIG,
                        config_paths=self.config_paths,
                        expert_run_dir=expert_run,
                        run_dirs=run_dirs,
                    )

        with tempfile.TemporaryDirectory() as tmp:
            expert_run, run_dirs, _ = self._fixture(tmp)
            extra = (
                run_dirs["fr1_kd0_sc0"]
                / "social_learning/receiver_agent_0/checkpoints/dkp_sl_v1_ablation_fr1_kd0_sc0/before_social.pt"
            )
            extra.write_bytes(b"must-not-be-retained")
            with self.assertRaisesRegex(LossAblationSummaryError, "contains"):
                build_summary(
                    expert_config=DEFAULT_EXPERT_CONFIG,
                    config_paths=self.config_paths,
                    expert_run_dir=expert_run,
                    run_dirs=run_dirs,
                )

    def test_summary_rejects_formal_destinations(self):
        with self.assertRaises(LossAblationSummaryError):
            _validate_output_path("paper_tables/loss_ablation.json")
        with self.assertRaises(LossAblationSummaryError):
            _validate_output_path("outputs/experiment_registry/loss_ablation.json")

    def test_summary_cli_has_no_overwrite_escape_and_existing_output_fails_closed(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--overwrite"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing.json"
            output.write_text('{"immutable": true}\n', encoding="utf-8")
            before = output.read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = main(["--output-json", str(output)])
            self.assertEqual(exit_code, 1)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

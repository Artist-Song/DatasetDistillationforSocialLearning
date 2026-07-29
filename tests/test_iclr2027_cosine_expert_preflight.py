import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import torch

from config_adapter import load_config
from scripts.validate_iclr2027_cosine_experts import (
    DEFAULT_CONFIG,
    PreflightError,
    _validate_model_definition,
    main,
    sha256_file,
    validate_expert_artifact,
    validate_protocol_config,
)

from models.convnet import ConvNet
from models.cosine_classifier import get_cosine_classifier
from models.resnet import ResNet


class ICLR2027CosineExpertPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)
        cls.args_by_agent = validate_protocol_config(cls.config, DEFAULT_CONFIG)

    def _conv3_builder(self, _args):
        return ConvNet(
            100,
            net_norm="instance",
            net_depth=3,
            net_width=128,
            channel=3,
            im_size=(32, 32),
            classifier_type="cosine",
            cosine_scale_init=10.0,
        )

    def _write_agent0_artifact(self, run_dir, local_test_accuracy=50.0):
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
            "official_test_accuracy_report_only": local_test_accuracy,
            "validation_fraction": 0.1,
            "retrained_on_full_local_train": True,
            "global_output_dim": 100,
            "labels": "global",
            "active_class_ids": list(args.active_class_ids),
            "masked_local_ce": True,
            "classifier": {
                "type": "cosine",
                "bias": False,
                "feature_normalization": True,
                "weight_normalization": True,
                "scale_parameterization": "softplus",
                "scale_init": 10.0,
                "final_scale": float(get_cosine_classifier(model).scale.detach()),
                "scale_weight_decay": 0.0,
            },
            "expert_path": str(checkpoint_path.resolve()),
            "expert_sha256": sha256_file(checkpoint_path),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return checkpoint_path, manifest_path, manifest

    def test_current_config_is_exact_seed0_five_by_twenty_protocol(self):
        self.assertEqual(set(self.args_by_agent), set(range(5)))
        self.assertEqual(
            [self.args_by_agent[index].model_name for index in range(5)],
            ["convnet3w1", "convnet4w15", "alexnet", "resnet10_standard", "resnet18_standard"],
        )
        classes = [class_id for args in self.args_by_agent.values() for class_id in args.active_class_ids]
        self.assertEqual(len(classes), 100)
        self.assertEqual(set(classes), set(range(100)))

    def test_config_overlap_or_nonstandard_resnet_is_rejected(self):
        overlapping = copy.deepcopy(self.config)
        overlapping["agents"]["class_split"]["agent_1"][0] = overlapping["agents"]["class_split"]["agent_0"][0]
        with self.assertRaisesRegex(PreflightError, "overlap"):
            validate_protocol_config(overlapping, "overlapping.yaml")

        wrong_resnet = copy.deepcopy(self.config)
        wrong_resnet["model_pool"]["models"]["resnet10_standard"]["family"] = "resnet"
        with self.assertRaisesRegex(PreflightError, "model family"):
            validate_protocol_config(wrong_resnet, "compact.yaml")

    def test_compact_resnet_checkpoint_definition_is_rejected(self):
        args = self.args_by_agent[3]
        compact = ResNet(
            "cifar100",
            10,
            100,
            size=32,
            cifar_base_width=32,
            classifier_type="cosine",
        )
        with self.assertRaisesRegex(PreflightError, "parameter count mismatch"):
            _validate_model_definition(compact, args, 3, get_cosine_classifier(compact))

    def test_valid_checkpoint_strictly_loads_and_accuracy_floor_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_agent0_artifact(run_dir, local_test_accuracy=50.0)
            report = validate_expert_artifact(
                self.args_by_agent[0],
                0,
                run_dir,
                min_local_test_accuracy=49.0,
                device="cpu",
                model_builder=self._conv3_builder,
            )
            self.assertEqual(report["output_shape"], [2, 100])
            self.assertEqual(report["feature_shapes"], [[2, 128, 4, 4]])
            self.assertEqual(report["local_test_accuracy"], 50.0)
            self.assertGreater(report["cosine_scale"], 0.0)

            with self.assertRaisesRegex(PreflightError, "not above"):
                validate_expert_artifact(
                    self.args_by_agent[0],
                    0,
                    run_dir,
                    min_local_test_accuracy=50.0,
                    device="cpu",
                    model_builder=self._conv3_builder,
                )

    def test_sha_mismatch_and_incomplete_state_dict_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            checkpoint_path, manifest_path, manifest = self._write_agent0_artifact(run_dir)
            manifest["expert_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "SHA-256"):
                validate_expert_artifact(
                    self.args_by_agent[0], 0, run_dir, model_builder=self._conv3_builder
                )

            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            del state["classifier.log_scale"]
            torch.save(state, checkpoint_path)
            manifest["expert_sha256"] = sha256_file(checkpoint_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "strict model loading"):
                validate_expert_artifact(
                    self.args_by_agent[0], 0, run_dir, model_builder=self._conv3_builder
                )

    def test_cli_returns_nonzero_when_any_expert_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir, redirect_stderr(io.StringIO()):
            exit_code = main(
                [
                    "--config",
                    str(DEFAULT_CONFIG),
                    "--run-dir",
                    str(Path(tmp_dir) / "missing-run"),
                ]
            )
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from agent_data import build_agent_args
from agent_trainer import _train_guide_pool
from scripts.prepare_fullclass_conv3_trajectory import (
    CHECKPOINT_EPOCHS,
    TRAJECTORY_MODELS,
    build_trajectory_config,
)
from scripts.prepare_fullclass_dsdm import MODEL_IDS


class FullClassDSDMTrajectoryProtocolTest(unittest.TestCase):
    def test_config_changes_only_the_guide_pool_design(self):
        feature_indices = {"conv3": 2, "conv4": 3, "alexnet": 7}
        for model_name in TRAJECTORY_MODELS:
            with self.subTest(model_name=model_name):
                config = build_trajectory_config(model_name)
                model_id = MODEL_IDS[model_name]
                guide = config["model_pool"]["models"][model_id]["guide_training"]
                self.assertEqual(guide["num_models"], 10)
                self.assertEqual(guide["max_epochs"], 200)
                self.assertEqual(guide["snapshot_epochs"], [200])
                self.assertEqual(guide["trajectory_checkpoint_epochs"], CHECKPOINT_EPOCHS)
                self.assertEqual(guide["trajectory_count"], 1)
                self.assertEqual(guide["training_style"], "dsdm_single_trajectory")

                args = build_agent_args(config, "generated", 0)
                self.assertEqual(args.guide_trajectory_checkpoint_epochs, CHECKPOINT_EPOCHS)
                self.assertEqual(args.guide_trajectory_count, 1)
                self.assertEqual(args.pretrained_model_number, 10)
                self.assertEqual(args.pretrained_epochs, 200)
                self.assertEqual(args.niter, 10000)
                self.assertEqual(args.evaluate_iterations, list(range(500, 10001, 500)))
                self.assertEqual(args.f_idx, str(feature_indices[model_name]))
                self.assertEqual((args.idx_from, args.idx_to), (feature_indices[model_name], -1))

    def test_one_trajectory_is_exposed_as_ten_dsdm_models(self):
        calls = []

        def fake_train_pretrained_trajectory(args, checkpoint_epochs):
            calls.append((args, list(checkpoint_epochs)))
            save_dir = Path(args.save_pretrain_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index, epoch in enumerate(checkpoint_epochs):
                path = save_dir / f"{args.dataset}_model_{index}.pth"
                torch.save({"weight": torch.tensor([float(epoch)])}, path)
                paths.append(path)
            return paths

        fake_module = types.ModuleType("pre_train_model")
        fake_module.train_pretrained_trajectory = fake_train_pretrained_trajectory
        args = SimpleNamespace(
            pretrained_epochs=200,
            guide_max_epochs=200,
            guide_snapshot_epochs=[200],
            guide_epoch=200,
            guide_model_number=10,
            pretrained_model_number=10,
            guide_training_style="dsdm_single_trajectory",
            guide_pool_design="single_trajectory_epoch_snapshots",
            guide_trajectory_count=1,
            guide_trajectory_checkpoint_epochs=CHECKPOINT_EPOCHS,
            guide_scheduler="none",
            guide_batch_size=256,
            batch_size=64,
            batch_real=256,
            active_class_ids=[0, 1, 2],
            num_classes=3,
            nclass=3,
            separate_expert=False,
            seed=0,
            lr=0.01,
            guide_lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            load_memory=True,
            aug_type="color_crop_cutout",
            mixup_net="cut",
            mix_p=0.5,
            dataset="cifar100",
            official_dsdm_commit="cb12851831e39da6b0169da84598166ad7706e01",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = Path(temp_dir)
            with mock.patch.dict(sys.modules, {"pre_train_model": fake_module}):
                paths = _train_guide_pool(args, 0, ckpt_dir, torch.device("cpu"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0].pretrained_model_number, 1)
            self.assertEqual(calls[0][1], CHECKPOINT_EPOCHS)
            self.assertEqual(len(paths), 10)
            manifest = json.loads((ckpt_dir / "guide_pool_manifest.json").read_text())
            self.assertEqual(manifest["trajectory_count"], 1)
            self.assertEqual(manifest["checkpoint_epochs"], CHECKPOINT_EPOCHS)
            self.assertEqual(
                [item["checkpoint_epoch"] for item in manifest["model_artifacts"]],
                CHECKPOINT_EPOCHS,
            )


if __name__ == "__main__":
    unittest.main()

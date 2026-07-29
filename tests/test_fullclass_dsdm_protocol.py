import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from agent_data import build_agent_args, get_agent_class_split
from agent_trainer import _train_guide_pool
from scripts.prepare_fullclass_dsdm import MODELS, MODEL_IDS, OFFICIAL_DSDM_COMMIT, build_config


class FullClassDSDMProtocolTest(unittest.TestCase):
    FEATURE_INDICES = {
        "conv3": 2,
        "conv4": 3,
        "alexnet": 7,
        "resnet10_standard": 5,
        "resnet18_standard": 5,
    }

    def test_all_backbones_use_dsdm_guide_recipe_except_approved_epoch_change(self):
        for name in MODELS:
            config = build_config(name)
            model_id = MODEL_IDS[name]
            guide = config["model_pool"]["models"][model_id]["guide_training"]
            self.assertEqual(guide["num_models"], 10)
            self.assertEqual(guide["max_epochs"], 200)
            self.assertEqual(guide["snapshot_epochs"], [200])
            self.assertEqual(guide["selected_epoch"], 200)
            self.assertEqual(guide["lr"], 0.01)
            self.assertEqual(guide["batch_size"], 256)
            self.assertFalse(guide["augment"])
            self.assertEqual(guide["scheduler"], "none")
            self.assertEqual(guide["scheduler_milestones"], [])
            self.assertEqual(guide["training_style"], "dsdm")
            self.assertNotIn("source_root", guide)

    def test_generated_config_is_all_class_and_maps_runtime_guide_settings(self):
        for name in MODELS:
            config = build_config(name)
            self.assertEqual(get_agent_class_split(config)[0], list(range(100)))
            args = build_agent_args(config, "generated", 0)
            self.assertEqual(args.guide_max_epochs, 200)
            self.assertEqual(args.guide_batch_size, 256)
            self.assertEqual(args.guide_scheduler, "none")
            self.assertEqual(args.guide_training_style, "dsdm")
            self.assertTrue(args.guide_only)
            self.assertEqual(args.ipc, 10)
            self.assertEqual(args.factor, 2)
            self.assertEqual(args.niter, 10000)
            self.assertEqual(args.evaluate_iter, 500)
            self.assertEqual(args.evaluate_iterations, list(range(500, 10001, 500)))
            self.assertEqual(args.lr_img, 0.1)
            self.assertEqual(args.mom_img, 0.5)
            self.assertEqual(args.batch_real, 256)
            self.assertEqual(args.batch_syn_max, 256)
            self.assertEqual(args.smooth_iter, 2000)
            self.assertEqual(args.cov_weight, 50.0)
            self.assertEqual(args.h_p_weight, 0.2)
            self.assertEqual(args.smooth_factor, 0.99)
            self.assertEqual(args.mixup, "cut")
            self.assertEqual(args.mixup_net, "cut")
            self.assertEqual(args.beta, 1.0)
            self.assertEqual(args.mix_p, 0.5)
            self.assertTrue(args.dsa)
            self.assertEqual(args.dsa_strategy, "color_crop_flip_scale_rotate")
            self.assertFalse(args.augment)
            self.assertEqual(args.epochs, 1500)
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.repeat, 1)
            self.assertEqual(args.workers, 8)
            self.assertEqual(args.guide_model_mode, "train")
            self.assertFalse(args.freeze_guide_parameters)
            self.assertEqual(args.grad_clip_norm, 0.0)
            self.assertTrue(args.official_dsdm_protocol)
            self.assertEqual(args.official_dsdm_commit, OFFICIAL_DSDM_COMMIT)
            self.assertEqual(args.match, "grad")
            self.assertTrue(args.reproduce)
            self.assertEqual(args.f_idx, str(self.FEATURE_INDICES[name]))
            self.assertEqual((args.idx_from, args.idx_to), (self.FEATURE_INDICES[name], -1))

    def test_dsdm_guide_pool_calls_official_pretraining_implementation(self):
        calls = []

        def fake_train_pretrained_models(args):
            calls.append(args)
            save_dir = Path(args.save_pretrain_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(args.pretrained_model_number):
                path = save_dir / f"{args.dataset}_model_{index}.pth"
                torch.save({"weight": torch.tensor([float(index)])}, path)
                paths.append(path)
            return paths

        fake_module = types.ModuleType("pre_train_model")
        fake_module.train_pretrained_models = fake_train_pretrained_models
        args = SimpleNamespace(
            pretrained_epochs=200,
            guide_max_epochs=200,
            guide_snapshot_epochs=[200],
            guide_epoch=200,
            guide_model_number=2,
            pretrained_model_number=2,
            guide_training_style="dsdm",
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
            official_dsdm_commit=OFFICIAL_DSDM_COMMIT,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = Path(temp_dir)
            with mock.patch.dict(sys.modules, {"pre_train_model": fake_module}):
                with mock.patch("agent_trainer._train_epoch", side_effect=AssertionError("local path used")):
                    paths = _train_guide_pool(args, 0, ckpt_dir, torch.device("cpu"))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].pretrained_epochs, 200)
            self.assertEqual(calls[0].pretrained_model_number, 2)
            self.assertEqual(calls[0].batch_real, 256)
            self.assertEqual(len(paths), 2)
            manifest = json.loads((ckpt_dir / "guide_pool_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["source_impl"],
                "DSDM/pre_train_model.py::train_pretrained_models",
            )
            self.assertEqual(manifest["augmentation_net_update"], "color_crop")
            self.assertEqual(manifest["mixup"], "cut")
            self.assertEqual(manifest["scheduler"], "none")
            self.assertEqual(manifest["official_commit"], OFFICIAL_DSDM_COMMIT)


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dkp_receiver import CyclingLoader, build_complete_balanced_loader, supervised_contrastive_loss
from packet_consumer import (
    consume_external_manifest_packets,
    consume_receiver_manifest_packet,
)
from packet_integrity import file_sha256
from social_metrics import compute_accuracy
from social_trainer import (
    SocialTrainer,
    _ensure_dsdm_path,
    resolve_class_balanced_ce_weights,
    resolve_dkp_loss_switches,
)

_ensure_dsdm_path()
from models.cosine_classifier import CosineClassifier


class TinyCosineModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        self.classifier = CosineClassifier(3, num_classes, scale_init=10.0)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(3))

    def forward(self, images):
        return self.classifier(self.encoder(images.flatten(1)))


class TinyLinearModel(nn.Module):
    def __init__(self, num_classes=6, bias=True):
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, num_classes, bias=bias)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(3))

    def forward(self, images):
        return self.classifier(self.encoder(images.flatten(1)))


class DKPReceiverUtilityTests(unittest.TestCase):
    def test_class_balanced_ce_weights_cover_scaling_protocols(self):
        for classes_per_agent, expected in ((20, 0.2), (10, 0.1), (5, 0.05)):
            with self.subTest(classes_per_agent=classes_per_agent):
                local, external = resolve_class_balanced_ce_weights(
                    list(range(classes_per_agent)),
                    100,
                )
                self.assertAlmostEqual(local, expected)
                self.assertAlmostEqual(external, 1.0 - expected)

    def test_class_balanced_ce_weights_reject_invalid_partitions(self):
        for classes in ([], [0, 0], [-1], list(range(100))):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    resolve_class_balanced_ce_weights(classes, 100)

    def test_balanced_loader_visits_every_item_once(self):
        images = torch.arange(24, dtype=torch.float).reshape(8, 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        indices = torch.arange(8)
        loader = build_complete_balanced_loader(
            images,
            labels,
            indices,
            batch_size=3,
            shuffle=False,
        )
        visited = torch.cat([batch[2] for batch in loader])
        self.assertEqual(visited.tolist(), list(range(8)))
        self.assertEqual(len(loader), 3)

    def test_balanced_loader_shuffle_is_reproducible_with_generator(self):
        images = torch.arange(24, dtype=torch.float).reshape(8, 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        indices = torch.arange(8)

        def sampled_order(seed):
            loader = build_complete_balanced_loader(
                images,
                labels,
                indices,
                batch_size=3,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            return torch.cat([batch[2] for batch in loader]).tolist()

        self.assertEqual(sampled_order(123), sampled_order(123))
        self.assertNotEqual(sampled_order(123), sampled_order(124))

    def test_padded_loader_keeps_full_batches_and_visits_all_rows(self):
        loader = build_complete_balanced_loader(
            torch.arange(16, dtype=torch.float).reshape(8, 2),
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            torch.arange(8),
            batch_size=3,
            shuffle=False,
            pad_to_full_batch=True,
        )
        batches = list(loader)
        visited = torch.cat([batch[2] for batch in batches])
        self.assertEqual([batch[0].shape[0] for batch in batches], [3, 3, 3])
        self.assertEqual(sorted(set(visited.tolist())), list(range(8)))
        self.assertEqual(visited.tolist(), list(range(8)) + [0])

    def test_balanced_loader_rejects_unequal_class_counts(self):
        with self.assertRaisesRegex(ValueError, "not class-balanced"):
            build_complete_balanced_loader(
                torch.randn(3, 2),
                torch.tensor([0, 0, 1]),
                batch_size=2,
            )

    def test_cycling_loader_restarts(self):
        loader = build_complete_balanced_loader(
            torch.arange(4, dtype=torch.float).reshape(4, 1),
            torch.tensor([0, 0, 1, 1]),
            batch_size=2,
            shuffle=False,
        )
        cycling = CyclingLoader(loader)
        values = [cycling.next()[0].flatten().tolist() for _ in range(3)]
        self.assertEqual(values, [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]])

    def test_supcon_is_finite_and_rewards_matching_views(self):
        labels = torch.tensor([0, 1])
        view1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        matching = view1.clone()
        swapped = torch.flip(view1, dims=(0,))
        matching_loss = supervised_contrastive_loss(view1, matching, labels, temperature=0.1)
        swapped_loss = supervised_contrastive_loss(view1, swapped, labels, temperature=0.1)
        self.assertTrue(torch.isfinite(matching_loss))
        self.assertLess(matching_loss.item(), swapped_loss.item())

    def test_expert_mask_excludes_untrained_classifier_rows(self):
        model = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
        loader = DataLoader(
            TensorDataset(torch.eye(3), torch.tensor([0, 1, 2])),
            batch_size=3,
        )
        self.assertEqual(compute_accuracy(model, loader, torch.device("cpu")), 100.0)
        self.assertAlmostEqual(
            compute_accuracy(model, loader, torch.device("cpu"), allowed_class_ids=[0, 1]),
            200.0 / 3.0,
        )

    def test_external_packet_consumer_excludes_receiver_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet_path = Path(tmp) / "external.pt"
            torch.save(
                {
                    "source": "dsdm",
                    "images": torch.randn(2, 3, 2, 2),
                    "labels": torch.tensor([2, 3]),
                    "class_ids": [2, 3],
                    "factor": 1,
                    "has_sender_logits": True,
                    "sender_logits": torch.randn(2, 2),
                    "sender_logit_class_ids": torch.tensor([2, 3]),
                },
                packet_path,
            )
            result = consume_external_manifest_packets(
                SimpleNamespace(factor=1, decode_type="single", batch_syn_max=16),
                [
                    {"sender_agent": "0", "packet_path": str(Path(tmp) / "unused.pt")},
                    {"sender_agent": "1", "packet_path": str(packet_path)},
                ],
                receiver_agent=0,
                require_logits=True,
            )
        self.assertEqual(result["labels"].tolist(), [2, 3])
        self.assertEqual(result["sender_agents"].tolist(), [1, 1])
        self.assertEqual(tuple(result["sender_logits"].shape), (2, 2))

    def test_self_packet_consumer_requires_one_matching_sender(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet_path = Path(tmp) / "self.pt"
            torch.save(
                {
                    "source": "dsdm",
                    "images": torch.randn(2, 3, 2, 2),
                    "labels": torch.tensor([0, 1]),
                    "class_ids": [0, 1],
                    "factor": 1,
                },
                packet_path,
            )
            args = SimpleNamespace(factor=1, decode_type="single", batch_syn_max=16)
            result = consume_receiver_manifest_packet(
                args,
                [{"sender_agent": "0", "packet_path": str(packet_path)}],
                receiver_agent=0,
            )
            self.assertEqual(result["sender_agent"], 0)
            self.assertEqual(result["raw_images"], 2)
            self.assertEqual(result["num_images"], 2)
            with self.assertRaisesRegex(ValueError, "exactly one self DKP"):
                consume_receiver_manifest_packet(args, [], receiver_agent=0)


class DKPReceiverProtocolTests(unittest.TestCase):
    def _args(self, root, variant):
        switches = resolve_dkp_loss_switches(variant)
        return SimpleNamespace(
            receiver_protocol="dkp_sl_v1",
            dkp_variant=variant,
            dkp_loss_switches=switches if variant.startswith("ablation_") else None,
            use_logits=switches["kd"],
            use_generalist_logits=False,
            lambda_fr=0.2 if switches["fr"] else 0.0,
            lambda_kd=0.6 if switches["kd"] else 0.0,
            lambda_sc=0.1 if switches["supcon"] else 0.0,
            kd_temperature=2.0,
            supcon_temperature=0.07,
            prototype_decoded_per_class=2,
            prototype_batch_size=2,
            receiver_local_batch_size=2,
            receiver_external_batch_size=2,
            receiver_lr=0.01,
            receiver_epochs=1,
            receiver_scheduler="none",
            receiver_scheduler_milestones=[],
            receiver_scheduler_gamma=0.2,
            freeze_bn_stats=False,
            receiver_augment=True,
            lr=0.01,
            epochs=1,
            momentum=0.9,
            weight_decay=5e-4,
            ipc=1,
            batch_size=2,
            packet_method="dsdm",
            communication_mode="all_share_once",
            init_mode="expert",
            dataset="cifar100",
            num_classes=6,
            nclass=6,
            active_class_ids=[0, 1],
            agent_id=0,
            agent_class_split={0: [0, 1], 1: [2, 3], 2: [4, 5]},
            agent_model_split={0: "tiny", 1: "tiny", 2: "tiny"},
            model_name="tiny",
            output_root=str(root),
            run_name=f"test_{variant}",
            device="cpu",
        )

    @staticmethod
    def _streams(include_logits):
        local_images = torch.tensor(
            [
                [[[1.0]], [[0.0]], [[0.0]]],
                [[[0.9]], [[0.1]], [[0.0]]],
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[0.1]], [[0.9]], [[0.0]]],
            ]
        )
        external_images = torch.tensor(
            [
                [[[0.0]], [[0.0]], [[1.0]]],
                [[[0.1]], [[0.0]], [[0.9]]],
                [[[0.7]], [[0.7]], [[0.0]]],
                [[[0.6]], [[0.8]], [[0.0]]],
                [[[0.8]], [[0.0]], [[0.6]]],
                [[[0.9]], [[0.0]], [[0.5]]],
                [[[0.0]], [[0.8]], [[0.6]]],
                [[[0.0]], [[0.9]], [[0.5]]],
            ]
        )
        local = {
            "images": local_images,
            "labels": torch.tensor([0, 0, 1, 1]),
            "num_images": 4,
        }
        packet_meta = [
            {"raw_images": 4, "sender_logit_bytes": 32},
            {"raw_images": 4, "sender_logit_bytes": 32},
        ]
        external = {
            "images": external_images,
            "labels": torch.tensor([2, 2, 3, 3, 4, 4, 5, 5]),
            "sender_agents": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
            "packets": packet_meta,
            "sender_logits": None,
            "sender_logit_class_ids": None,
        }
        if include_logits:
            external["sender_logits"] = torch.tensor(
                [
                    [2.0, -1.0],
                    [1.5, -0.5],
                    [-1.0, 2.0],
                    [-0.5, 1.5],
                    [2.0, -1.0],
                    [1.5, -0.5],
                    [-1.0, 2.0],
                    [-0.5, 1.5],
                ]
            )
            external["sender_logit_class_ids"] = torch.tensor(
                [
                    [2, 3],
                    [2, 3],
                    [2, 3],
                    [2, 3],
                    [4, 5],
                    [4, 5],
                    [4, 5],
                    [4, 5],
                ]
            )
        return local, external

    def test_prototype_initialization_preserves_local_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SocialTrainer.__new__(SocialTrainer)
            trainer.args = self._args(Path(tmp), "full")
            trainer.device = torch.device("cpu")
            trainer.expert_classes = [0, 1]
            old_model = TinyCosineModel()
            new_model = copy.deepcopy(old_model)
            local_before = new_model.classifier.weight[:2].detach().clone()
            _, external = self._streams(include_logits=True)
            external_ids = trainer._initialize_external_prototypes(
                old_model,
                new_model,
                external["images"],
                external["labels"],
            )
            self.assertEqual(external_ids, [2, 3, 4, 5])
            self.assertTrue(torch.equal(local_before, new_model.classifier.weight[:2]))
            self.assertTrue(
                torch.allclose(
                    new_model.classifier.weight[2:].norm(dim=1),
                    torch.ones(4),
                    atol=1e-6,
                    rtol=0,
                )
            )
            self.assertEqual(trainer._prototype_init_stats["classifier_type"], "cosine")
            self.assertEqual(trainer._prototype_init_stats["mode"], "cosine_unit_weight_rows")
            self.assertIsNone(trainer._prototype_init_stats["alpha"])
            self.assertIsNone(trainer._prototype_init_stats["beta"])

    def test_linear_prototypes_use_local_row_norm_and_bias_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SocialTrainer.__new__(SocialTrainer)
            trainer.args = self._args(Path(tmp), "full")
            trainer.device = torch.device("cpu")
            trainer.expert_classes = [0, 1]
            old_model = TinyLinearModel()
            with torch.no_grad():
                old_model.classifier.weight[0].copy_(torch.tensor([3.0, 0.0, 0.0]))
                old_model.classifier.weight[1].copy_(torch.tensor([0.0, 4.0, 0.0]))
                old_model.classifier.bias[0] = 0.25
                old_model.classifier.bias[1] = 0.75
            new_model = copy.deepcopy(old_model)
            local_weight_before = new_model.classifier.weight[:2].detach().clone()
            local_bias_before = new_model.classifier.bias[:2].detach().clone()
            _, external = self._streams(include_logits=True)

            external_ids = trainer._initialize_external_prototypes(
                old_model,
                new_model,
                external["images"],
                external["labels"],
            )

            self.assertEqual(external_ids, [2, 3, 4, 5])
            self.assertTrue(torch.equal(local_weight_before, new_model.classifier.weight[:2]))
            self.assertTrue(torch.equal(local_bias_before, new_model.classifier.bias[:2]))
            self.assertTrue(torch.isfinite(new_model.classifier.weight[2:]).all())
            self.assertTrue(torch.isfinite(new_model.classifier.bias[2:]).all())
            self.assertTrue(
                torch.allclose(
                    new_model.classifier.weight[2:].norm(dim=1),
                    torch.full((4,), 3.5),
                    atol=1e-6,
                    rtol=0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    new_model.classifier.bias[2:],
                    torch.full((4,), 0.5),
                    atol=1e-7,
                    rtol=0,
                )
            )
            expected_direction = nn.functional.normalize(
                external["images"][:2].flatten(1).mean(dim=0), dim=0
            )
            actual_direction = nn.functional.normalize(new_model.classifier.weight[2], dim=0)
            self.assertTrue(torch.allclose(actual_direction, expected_direction, atol=1e-6, rtol=0))
            stats = trainer._prototype_init_stats
            self.assertEqual(stats["classifier_type"], "linear")
            self.assertEqual(stats["mode"], "linear_local_row_norm_bias_mean")
            self.assertAlmostEqual(stats["alpha"], 3.5)
            self.assertAlmostEqual(stats["beta"], 0.5)

    def test_linear_prototypes_without_bias_record_zero_beta(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SocialTrainer.__new__(SocialTrainer)
            trainer.args = self._args(Path(tmp), "full")
            trainer.device = torch.device("cpu")
            trainer.expert_classes = [0, 1]
            old_model = TinyLinearModel(bias=False)
            with torch.no_grad():
                old_model.classifier.weight[0].copy_(torch.tensor([2.0, 0.0, 0.0]))
                old_model.classifier.weight[1].copy_(torch.tensor([0.0, 2.0, 0.0]))
            new_model = copy.deepcopy(old_model)
            _, external = self._streams(include_logits=True)
            trainer._initialize_external_prototypes(
                old_model,
                new_model,
                external["images"],
                external["labels"],
            )
            self.assertEqual(trainer._prototype_init_stats["beta"], 0.0)
            self.assertIsNone(new_model.classifier.bias)

    def test_linear_prototypes_reject_nonpositive_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SocialTrainer.__new__(SocialTrainer)
            trainer.args = self._args(Path(tmp), "full")
            trainer.device = torch.device("cpu")
            trainer.expert_classes = [0, 1]
            old_model = TinyLinearModel()
            with torch.no_grad():
                old_model.classifier.weight[:2].zero_()
            new_model = copy.deepcopy(old_model)
            _, external = self._streams(include_logits=True)
            with self.assertRaisesRegex(ValueError, "alpha must be finite and positive"):
                trainer._initialize_external_prototypes(
                    old_model,
                    new_model,
                    external["images"],
                    external["labels"],
                )

    def test_strict_receiver_checkpoint_provenance_uses_shared_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), "full")
            args.strict_packet_validation = True
            checkpoint = (
                Path(args.output_root)
                / args.run_name
                / "agents"
                / "agent_0"
                / "checkpoints"
                / "expert_model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            torch.save({"weight": torch.ones(1)}, checkpoint)
            trainer = SocialTrainer.__new__(SocialTrainer)
            trainer.args = args
            trainer.receiver_agent = 0
            trainer.manifest_rows = [
                {
                    "sender_agent": "0",
                    "expert_checkpoint_sha256": file_sha256(checkpoint),
                }
            ]
            with mock.patch(
                "packet_integrity.validate_receiver_expert_provenance",
                return_value=file_sha256(checkpoint),
            ) as validator:
                self.assertEqual(
                    trainer._receiver_expert_checkpoint_sha(require_manifest_match=True),
                    file_sha256(checkpoint),
                )
            validator.assert_called_once_with(
                args,
                0,
                trainer.manifest_rows,
                checkpoint,
            )

    def _run_variant(
        self,
        root,
        variant,
        checkpoint_retention=None,
        model_factory=TinyCosineModel,
        local_ce_source="real",
        local_ce_real_fraction=None,
        optimizer_steps=None,
    ):
        args = self._args(root, variant)
        args.receiver_local_ce_source = local_ce_source
        args.receiver_local_ce_real_fraction = local_ce_real_fraction
        args.receiver_optimizer_steps = optimizer_steps
        if checkpoint_retention is not None:
            args.receiver_checkpoint_retention = checkpoint_retention
        trainer = SocialTrainer.__new__(SocialTrainer)
        trainer.args = args
        trainer.receiver_agent = 0
        trainer.manifest_rows = []
        trainer.class_split = args.agent_class_split
        trainer.model_split = args.agent_model_split
        trainer.expert_classes = [0, 1]
        trainer.device = torch.device("cpu")
        model_old = model_factory()
        model_new = copy.deepcopy(model_old)
        trainer._build_models = lambda: (model_old, model_new)
        trainer._normalize_images = lambda images: images.float()
        switches = resolve_dkp_loss_switches(variant)
        local, external = self._streams(include_logits=switches["kd"])
        self_packet_path = Path(root) / "self_packet.pt"
        torch.save({"test": True}, self_packet_path)
        self_packet = {
            "images": local["images"].clone(),
            "labels": local["labels"].clone(),
            "raw_images": 2,
            "num_images": 4,
            "packet_path": str(self_packet_path),
            "manifest_packet_sha256": "",
        }
        metrics = {"acc_global": 25.0, "acc_expert": 50.0, "acc_new": 0.0}
        with mock.patch("social_trainer.load_receiver_local_real_data", return_value=local), mock.patch(
            "social_trainer.consume_external_manifest_packets", return_value=external
        ), mock.patch(
            "social_trainer.consume_receiver_manifest_packet", return_value=self_packet
        ), mock.patch("social_trainer.evaluate_receiver_model", return_value=metrics):
            return trainer.train()

    def test_local_ce_sources_keep_self_packet_out_of_external_communication(self):
        for local_ce_source in ("packet", "real_packet_50_50"):
            with self.subTest(local_ce_source=local_ce_source), tempfile.TemporaryDirectory() as tmp:
                result = self._run_variant(
                    Path(tmp),
                    "full",
                    local_ce_source=local_ce_source,
                    optimizer_steps=3,
                )
                self.assertEqual(result["optimizer_steps"], 3)
                self.assertEqual(result["target_optimizer_steps"], 3)
                self.assertEqual(result["external_comm_images"], 8)
                self.assertEqual(result["external_comm_logit_bytes"], 64)
                self.assertEqual(result["self_packet_raw_images"], 2)
                self.assertEqual(result["self_packet_decoded_images"], 4)
                self.assertEqual(result["local_ce_source"], local_ce_source)
                self.assertTrue(result["self_packet_sha256"])
                if local_ce_source == "packet":
                    self.assertEqual(result["loss_ce_local_real"], 0.0)
                    self.assertAlmostEqual(
                        result["loss_ce_local"], result["loss_ce_local_packet"], places=7
                    )
                else:
                    self.assertGreater(result["loss_ce_local_real"], 0.0)
                    self.assertAlmostEqual(
                        result["loss_ce_local"],
                        0.5
                        * (
                            result["loss_ce_local_real"]
                            + result["loss_ce_local_packet"]
                        ),
                        places=6,
                    )

    def test_packet_heavy_mix_uses_configured_real_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_variant(
                Path(tmp),
                "full",
                local_ce_source="real_packet_mix",
                local_ce_real_fraction=0.1,
                optimizer_steps=3,
            )
        self.assertEqual(result["local_ce_real_fraction"], 0.1)
        self.assertAlmostEqual(
            result["loss_ce_local"],
            0.1 * result["loss_ce_local_real"]
            + 0.9 * result["loss_ce_local_packet"],
            places=6,
        )
        self.assertEqual(result["external_comm_images"], 8)
        self.assertEqual(result["external_comm_logit_bytes"], 64)
        self.assertEqual(result["self_packet_raw_images"], 2)
        self.assertEqual(result["self_packet_decoded_images"], 4)
        self.assertTrue(result["self_packet_sha256"])

    def test_fixed_step_scheduler_decays_after_configured_updates(self):
        trainer = SocialTrainer.__new__(SocialTrainer)
        trainer.args = SimpleNamespace(
            receiver_scheduler="multistep",
            receiver_scheduler_unit="optimizer_step",
            receiver_scheduler_step_milestones=[2, 4],
            receiver_scheduler_gamma=0.2,
        )
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scheduler = trainer._build_receiver_scheduler(optimizer, receiver_epochs=60)
        lrs = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(lrs[0], 0.01)
        self.assertAlmostEqual(lrs[1], 0.002)
        self.assertAlmostEqual(lrs[2], 0.002)
        self.assertAlmostEqual(lrs[3], 0.0004)

    def test_full_protocol_uses_two_stream_losses_and_records_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_variant(Path(tmp), "full")
        self.assertEqual(result["method"], "DKP_SL")
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["external_comm_images"], 8)
        self.assertEqual(result["external_comm_logit_bytes"], 64)
        self.assertEqual(result["prototype_initialized_classes"], 4)
        for field in ["loss", "loss_cls", "loss_ce_local", "loss_ce_external", "loss_fr", "loss_kd", "loss_sc"]:
            self.assertTrue(torch.isfinite(torch.tensor(result[field])), field)

    def test_ce_only_disables_fr_kd_supcon_and_logit_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_variant(Path(tmp), "ce_only")
        self.assertEqual(result["method"], "DKP_CE_ONLY")
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["external_comm_logit_bytes"], 0)
        self.assertEqual(result["loss_fr"], 0.0)
        self.assertEqual(result["loss_kd"], 0.0)
        self.assertEqual(result["loss_sc"], 0.0)

    def test_six_missing_ablation_variants_independently_switch_losses(self):
        missing = [
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
        ]
        for fr, kd, sc in missing:
            variant = f"ablation_fr{int(fr)}_kd{int(kd)}_sc{int(sc)}"
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                result = self._run_variant(
                    Path(tmp),
                    variant,
                    checkpoint_retention="final_only",
                )
            self.assertEqual(result["method"], "DKP_SL_ABLATION")
            self.assertEqual(result["use_fr"], str(fr).lower())
            self.assertEqual(result["use_logits"], str(kd).lower())
            self.assertEqual(result["external_comm_logit_bytes"], 64 if kd else 0)
            self.assertEqual(result["checkpoint_retention"], "final_only")
            for enabled, field in ((fr, "loss_fr"), (kd, "loss_kd"), (sc, "loss_sc")):
                self.assertTrue(torch.isfinite(torch.tensor(result[field])), field)
                if not enabled:
                    self.assertEqual(result[field], 0.0)

    def test_ablation_switch_metadata_must_match_variant(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_dkp_loss_switches(
                "ablation_fr1_kd0_sc0",
                {"fr": False, "kd": False, "supcon": False},
            )
        with self.assertRaisesRegex(ValueError, "000/111 endpoints"):
            resolve_dkp_loss_switches(
                "ablation_fr0_kd0_sc0",
                {"fr": False, "kd": False, "supcon": False},
            )

    def test_default_retention_keeps_all_receiver_checkpoints_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._run_variant(root, "ce_only")
            checkpoint_dir = (
                root
                / "test_ce_only/social_learning/receiver_agent_0/checkpoints/dkp_sl_v1_ce_only"
            )
            self.assertTrue((checkpoint_dir / "before_social.pt").is_file())
            self.assertTrue((checkpoint_dir / "after_prototype_init.pt").is_file())
            self.assertTrue((checkpoint_dir / "after_social.pt").is_file())
            self.assertTrue((checkpoint_dir / "receiver_provenance.json").is_file())
            self.assertEqual(result["checkpoint_retention"], "all")

    def test_final_only_retention_keeps_final_checkpoint_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._run_variant(
                root,
                "ce_only",
                checkpoint_retention="final_only",
                model_factory=TinyLinearModel,
            )
            checkpoint_dir = (
                root
                / "test_ce_only/social_learning/receiver_agent_0/checkpoints/dkp_sl_v1_ce_only"
            )
            self.assertFalse((checkpoint_dir / "before_social.pt").exists())
            self.assertFalse((checkpoint_dir / "after_prototype_init.pt").exists())
            self.assertTrue((checkpoint_dir / "after_social.pt").is_file())
            provenance_path = checkpoint_dir / "receiver_provenance.json"
            self.assertTrue(provenance_path.is_file())
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["checkpoint_retention"], "final_only")
            self.assertEqual(set(provenance["checkpoint_artifacts"]), {"after_social"})
            self.assertIn("prototype_initialization", provenance)
            self.assertIn("statistics", provenance)
            self.assertEqual(result["checkpoint_retention"], "final_only")
            self.assertEqual(result["classifier_type"], "linear")
            self.assertEqual(result["prototype_init_mode"], "linear_local_row_norm_bias_mean")
            self.assertGreater(result["prototype_alpha"], 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(result["prototype_beta"])))
            self.assertTrue(result["after_social_checkpoint_sha256"])
            self.assertTrue(result["receiver_provenance_sha256"])


if __name__ == "__main__":
    unittest.main()

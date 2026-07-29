import unittest

from config_adapter import build_dsdm_args_from_config


class DKPReceiverConfigTests(unittest.TestCase):
    @staticmethod
    def _build(receiver=None):
        config = {}
        if receiver is not None:
            config["social_learning"] = {"receiver": receiver}
        return build_dsdm_args_from_config(config)

    def test_legacy_config_keeps_epoch_scheduler_and_real_local_ce_defaults(self):
        args = self._build()

        self.assertEqual(args.receiver_local_ce_source, "real")
        self.assertIsNone(args.receiver_local_ce_real_fraction)
        self.assertIsNone(args.receiver_optimizer_steps)
        self.assertEqual(args.receiver_self_packet_batch_size, 64)
        self.assertEqual(args.receiver_scheduler_unit, "epoch")
        self.assertEqual(args.receiver_scheduler_step_milestones, [])

    def test_accepts_all_three_local_ce_sources(self):
        for source in ("real", "packet", "real_packet_50_50"):
            with self.subTest(source=source):
                args = self._build({"local_ce_source": source})
                self.assertEqual(args.receiver_local_ce_source, source)

    def test_parses_packet_heavy_real_mix_fraction(self):
        args = self._build(
            {
                "local_ce_source": "real_packet_mix",
                "local_ce_real_fraction": 0.1,
            }
        )
        self.assertEqual(args.receiver_local_ce_source, "real_packet_mix")
        self.assertEqual(args.receiver_local_ce_real_fraction, 0.1)

    def test_real_packet_mix_fraction_is_required_and_strict(self):
        invalid = (
            {"local_ce_source": "real_packet_mix"},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": 0.0},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": 1.0},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": -0.1},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": 1.1},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": float("nan")},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": float("inf")},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": "0.1"},
            {"local_ce_source": "real_packet_mix", "local_ce_real_fraction": True},
            {"local_ce_source": "real", "local_ce_real_fraction": 0.1},
        )
        for receiver in invalid:
            with self.subTest(receiver=receiver), self.assertRaisesRegex(
                ValueError, "local_ce_real_fraction|requires"
            ):
                self._build(receiver)

    def test_parses_fixed_step_receiver_controls(self):
        args = self._build(
            {
                "local_ce_source": "packet",
                "optimizer_steps": 3780,
                "self_packet_batch_size": 64,
                "scheduler_unit": "optimizer_step",
                "scheduler_step_milestones": [2457, 3213],
            }
        )

        self.assertEqual(args.receiver_local_ce_source, "packet")
        self.assertEqual(args.receiver_optimizer_steps, 3780)
        self.assertEqual(args.receiver_self_packet_batch_size, 64)
        self.assertEqual(args.receiver_scheduler_unit, "optimizer_step")
        self.assertEqual(args.receiver_scheduler_step_milestones, [2457, 3213])

    def test_parses_positive_packet_raw_per_class(self):
        args = self._build({"packet_raw_per_class": 500})
        self.assertEqual(args.receiver_packet_raw_per_class, 500)

        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "packet_raw_per_class must be a positive integer"
            ):
                self._build({"packet_raw_per_class": value})

    def test_rejects_unknown_local_ce_source(self):
        with self.assertRaisesRegex(ValueError, "local_ce_source"):
            self._build({"local_ce_source": "all_packet"})

    def test_rejects_nonpositive_or_noninteger_fixed_steps(self):
        for value in (0, -1, 3780.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "optimizer_steps must be a positive integer"
            ):
                self._build({"optimizer_steps": value})

    def test_rejects_nonpositive_or_noninteger_self_packet_batch_size(self):
        for value in (0, -1, 64.0, False):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "self_packet_batch_size must be a positive integer"
            ):
                self._build({"self_packet_batch_size": value})

    def test_rejects_unknown_scheduler_unit(self):
        with self.assertRaisesRegex(ValueError, "scheduler_unit"):
            self._build({"scheduler_unit": "batch"})

    def test_step_scheduler_requires_fixed_steps_and_explicit_milestones(self):
        invalid_receivers = (
            {"scheduler_unit": "optimizer_step", "scheduler_step_milestones": [1]},
            {"scheduler_unit": "optimizer_step", "optimizer_steps": 10},
            {
                "scheduler_unit": "optimizer_step",
                "optimizer_steps": 10,
                "scheduler_step_milestones": [],
            },
        )
        for receiver in invalid_receivers:
            with self.subTest(receiver=receiver), self.assertRaisesRegex(
                ValueError, "requires|non-empty"
            ):
                self._build(receiver)

    def test_rejects_invalid_step_milestone_sequences(self):
        for milestones in ([2, 2], [3, 2], [1, 10], [0, 2], [1.0, 2]):
            with self.subTest(milestones=milestones), self.assertRaises(ValueError):
                self._build(
                    {
                        "scheduler_unit": "optimizer_step",
                        "optimizer_steps": 10,
                        "scheduler_step_milestones": milestones,
                    }
                )

    def test_epoch_scheduler_rejects_step_milestones(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            self._build(
                {
                    "scheduler_unit": "epoch",
                    "scheduler_step_milestones": [1, 2],
                }
            )


if __name__ == "__main__":
    unittest.main()

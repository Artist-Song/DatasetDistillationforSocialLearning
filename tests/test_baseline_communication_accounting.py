import unittest

from baseline_adapters.communication_accounting import desa_communication_accounting


class DesaCommunicationAccountingTests(unittest.TestCase):
    def test_balanced_cifar100_scaling_counts(self):
        expected = {
            4: (750, 7_500_000, 30_000_000),
            5: (800, 6_400_000, 32_000_000),
            10: (900, 3_600_000, 36_000_000),
            20: (950, 1_900_000, 38_000_000),
        }
        for agent_count, (images, per_receiver_bytes, all_agent_bytes) in expected.items():
            with self.subTest(agent_count=agent_count):
                classes_per_agent = 100 // agent_count
                anchors_per_agent = classes_per_agent * 10
                report = desa_communication_accounting(
                    {agent: anchors_per_agent for agent in range(agent_count)},
                    {agent: classes_per_agent for agent in range(agent_count)},
                    rounds=100,
                )
                self.assertEqual(report["unique_sender_images"], 1000)
                self.assertEqual(
                    set(report["external_images_per_receiver"].values()),
                    {images},
                )
                self.assertEqual(
                    set(report["iterative_owner_logit_bytes_per_receiver"].values()),
                    {per_receiver_bytes},
                )
                self.assertEqual(
                    report["receiver_incidence_images"],
                    agent_count * images,
                )
                self.assertEqual(
                    report["iterative_owner_logit_bytes_all_agents"],
                    all_agent_bytes,
                )

    def test_unbalanced_senders_are_counted_per_receiver(self):
        report = desa_communication_accounting(
            {0: 4, 1: 6, 2: 10},
            {0: 2, 1: 3, 2: 5},
            rounds=2,
        )
        self.assertEqual(report["external_images_per_receiver"], {0: 16, 1: 14, 2: 10})
        self.assertEqual(
            report["iterative_owner_logit_bytes_per_receiver"],
            {0: 544, 1: 464, 2: 208},
        )
        self.assertEqual(report["iterative_owner_logit_bytes_all_agents"], 1216)

    def test_invalid_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "owners"):
            desa_communication_accounting({0: 10, 1: 10}, {0: 5}, rounds=1)
        with self.assertRaisesRegex(ValueError, "rounds"):
            desa_communication_accounting({0: 10, 1: 10}, {0: 5, 1: 5}, rounds=0)
        with self.assertRaisesRegex(ValueError, "anchor"):
            desa_communication_accounting({0: 0, 1: 10}, {0: 5, 1: 5}, rounds=1)


if __name__ == "__main__":
    unittest.main()

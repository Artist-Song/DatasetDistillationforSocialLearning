import csv
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from social_output_manager import (
    SOCIAL_RESULT_FIELDS,
    append_social_result,
    get_social_results_path,
    save_social_config,
)


def _append_receiver_result(output_root, receiver_agent):
    args = SimpleNamespace(output_root=output_root, run_name="parallel_receivers")
    append_social_result(
        args,
        {
            "receiver_agent": receiver_agent,
            "receiver_model": f"model_{receiver_agent}",
            "protocol": "dkp_sl_v1",
            "dkp_variant": "full",
        },
    )


def _save_shared_snapshot(output_root, config_path):
    args = SimpleNamespace(
        output_root=output_root,
        run_name="parallel_receivers",
        communication_protocol="dkp_fixture",
        seed=0,
    )
    save_social_config(config_path, args)


class SocialOutputConcurrencyTests(unittest.TestCase):
    def test_parallel_snapshot_writers_reuse_only_identical_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.yaml"
            source.write_text("project:\n  run_name: parallel_receivers\n", encoding="utf-8")
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(target=_save_shared_snapshot, args=(temp_dir, str(source)))
                for _ in range(5)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            config_dir = root / "parallel_receivers" / "config"
            self.assertEqual((config_dir / "main.yaml").read_bytes(), source.read_bytes())
            resolved = json.loads((config_dir / "social_resolved_args.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved["communication_protocol"], "dkp_fixture")

            different = root / "different.yaml"
            different.write_text("project:\n  run_name: different\n", encoding="utf-8")
            args = SimpleNamespace(
                output_root=temp_dir,
                run_name="parallel_receivers",
                communication_protocol="dkp_fixture",
                seed=0,
            )
            with self.assertRaisesRegex(FileExistsError, "config snapshot differs"):
                save_social_config(different, args)

    def test_parallel_receiver_appends_keep_every_complete_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(target=_append_receiver_result, args=(temp_dir, receiver_agent))
                for receiver_agent in range(5)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            args = SimpleNamespace(output_root=temp_dir, run_name="parallel_receivers")
            path = get_social_results_path(args)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, SOCIAL_RESULT_FIELDS)
            self.assertEqual(len(rows), 5)
            self.assertEqual(sorted(int(row["receiver_agent"]) for row in rows), list(range(5)))
            self.assertTrue(all(row["protocol"] == "dkp_sl_v1" for row in rows))


if __name__ == "__main__":
    unittest.main()

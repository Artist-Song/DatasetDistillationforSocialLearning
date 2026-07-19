import sys
import unittest
from pathlib import Path

from validate_packets import _build_warning


# validate_packets supports the legacy top-level DSDM imports by prepending DSDM/.
# Remove that path after importing the helper so package-style tests remain order-independent.
DSDM_ROOT = str(Path(__file__).resolve().parents[1] / "DSDM")
while DSDM_ROOT in sys.path:
    sys.path.remove(DSDM_ROOT)


class PacketValidationActiveClassesTest(unittest.TestCase):
    @staticmethod
    def _args():
        return type(
            "Args",
            (),
            {
                "dataset": "tinyimagenet",
                "num_classes": 200,
                "nclass": 200,
                "ipc": 10,
                "factor": 2,
                "agent_class_split": {0: [0, 1]},
            },
        )()

    def test_dsdm_ignores_inactive_global_classes(self):
        summary = {
            "total_raw_images": 20,
            "total_train_images": 80,
            "per_class_train_images": {"0": 40, "1": 40, "2": 0},
        }
        self.assertEqual(_build_warning(self._args(), "dsdm", summary), [])

    def test_dsdm_reports_missing_active_class(self):
        summary = {
            "total_raw_images": 20,
            "total_train_images": 80,
            "per_class_train_images": {"0": 40, "1": 0},
        }
        warnings = _build_warning(self._args(), "dsdm", summary)
        self.assertTrue(any("class 1" in warning for warning in warnings))

    def test_selection_packet_ignores_inactive_global_classes(self):
        summary = {
            "total_raw_images": 20,
            "per_class_raw_images": {"0": 10, "1": 10, "2": 0},
        }
        self.assertEqual(_build_warning(self._args(), "heuristic", summary), [])


if __name__ == "__main__":
    unittest.main()

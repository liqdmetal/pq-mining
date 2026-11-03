"""Reporting checks use fake measurements; no GPU work is launched."""
import unittest
from unittest.mock import patch

from pqmine.fraction_test import UnitRate, fraction_report


class HardwareReportingTests(unittest.TestCase):
    def test_linear_projection_never_claims_a_hardware_bound(self):
        for gpu_seconds in (0.1, 1.0, 100.0):
            with self.subTest(gpu_seconds=gpu_seconds):
                with patch("pqmine.fraction_test.one_core", return_value=UnitRate("cpu", "core", 10, 100, 1.0)), \
                     patch("pqmine.fraction_test.one_sm", return_value=UnitRate("gpu", "SM", 10, 100, gpu_seconds)):
                    report = fraction_report(100)
                scaling = report["scaling"]
                self.assertFalse(scaling["is_hardware_advantage_bound"])
                self.assertFalse(scaling["is_measured_whole_chip_ratio"])
                self.assertNotIn("chip_gpu_over_cpu_upper_bound", scaling)
                self.assertIn("not a whole-chip", report["verdict"])


if __name__ == "__main__":
    unittest.main()

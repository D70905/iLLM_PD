from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(1, str(PROJECT / "scripts"))

import run_ncat_external_validation as validation
import run_ncat_external_validation_safe as safe


class SafeStatisticsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = safe.core_metrics(np.array([1.0, 2.0, 4.0]),
                                   np.array([1.0, 2.0, 4.0]))
        self.assertEqual(result["mae_mm"], 0.0)
        self.assertEqual(result["rmse_mm"], 0.0)
        self.assertEqual(result["r2"], 1.0)
        self.assertAlmostEqual(result["pearson_r"], 1.0)

    def test_known_error_metrics(self):
        result = safe.core_metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        self.assertAlmostEqual(result["mae_mm"], 1.5)
        self.assertAlmostEqual(result["rmse_mm"], np.sqrt(2.5))
        self.assertAlmostEqual(result["mean_bias_mm"], 1.5)

    def test_bootstrap_reproducible_at_10000(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.2, 1.8, 3.4, 3.7])
        self.assertEqual(safe.bootstrap_intervals(y, pred, 10000, 42),
                         safe.bootstrap_intervals(y, pred, 10000, 42))

    def test_input_json(self):
        payload = json.loads((PROJECT / "experiments/ncat_data/ncat_cases.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(len(payload["cases"]), 7)
        for case in payload["cases"]:
            validation.validate_case(case)

    def test_existing_csv(self):
        with (PROJECT / "ncat_mepdg_rut_summary.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(np.isfinite(float(r["pred_total_mm"])) for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)

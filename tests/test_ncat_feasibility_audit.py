from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import run_ncat_feasibility_audit as audit


class NcatFeasibilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads((PROJECT / "experiments/ncat_data/ncat_cases.json")
                             .read_text(encoding="utf-8"))
        cls.case = payload["cases"][0]

    def test_case_and_equivalent_base(self):
        audit.validate_case(self.case)
        materials = audit.equivalent_materials(self.case)
        self.assertEqual(len(materials["modulus"]), 5)
        self.assertEqual(materials["modulus"][3], materials["modulus"][4])
        self.assertEqual(materials["poisson"][3], materials["poisson"][4])

    def test_candidates_are_deterministic_bounded_and_tied(self):
        first = audit.generate_candidates(self.case, 32, 42)
        second = audit.generate_candidates(self.case, 32, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(len({row["candidate_id"] for row in first}), 32)
        for row in first:
            h = np.asarray(row["thickness_m"], dtype=float)
            self.assertAlmostEqual(h[3], h[4])
            values = np.asarray(list(row["variables"].values()), dtype=float)
            self.assertTrue(np.all(values >= audit.LOWER))
            self.assertTrue(np.all(values <= audit.UPPER))

    def test_summary_does_not_overclaim_failed_search(self):
        result = audit.summarize([
            {"status": "ok", "feasible": False, "dsr": 0.9,
             "cost_cny_m2": 300.0, "candidate_id": "a"}
        ])
        self.assertFalse(result["compliant_design_found"])
        self.assertIn("not proof", result["interpretation"])

    def test_summary_selects_cheapest_compliant(self):
        rows = [
            {"status": "ok", "feasible": True, "dsr": 1.1,
             "cost_cny_m2": 320.0, "candidate_id": "a"},
            {"status": "ok", "feasible": True, "dsr": 1.02,
             "cost_cny_m2": 290.0, "candidate_id": "b"},
        ]
        result = audit.summarize(rows)
        self.assertTrue(result["compliant_design_found"])
        self.assertEqual(result["cheapest_compliant_candidate"]["candidate_id"], "b")


if __name__ == "__main__":
    unittest.main(verbosity=2)

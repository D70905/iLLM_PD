from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NcatInterfacePressureOutputTests(unittest.TestCase):
    def test_abaqus_script_has_unique_interface_outputs(self):
        path = ROOT / "fea" / "abaqus_script.py"
        text = path.read_text(encoding="utf-8")
        ast.parse(text)
        for name in ("p_AC_base_interface_MPa",
                     "p_base_subgrade_interface_MPa"):
            self.assertGreaterEqual(text.count(name), 3)
        self.assertEqual(text.count("p_AC_base_interface_MPa = collect_at("), 1)
        self.assertEqual(text.count("p_base_subgrade_interface_MPa = collect_at("), 1)
        self.assertIn("S_nodal, z_AC_bot", text)
        self.assertIn("S_nodal, z_base_bot", text)

    def test_runner_documents_interface_outputs(self):
        path = ROOT / "fea" / "runner.py"
        text = path.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("p_AC_base_interface_MPa", text)
        self.assertIn("p_base_subgrade_interface_MPa", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Regression checks for canonical HEMS imports, layout and current guide links."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/single_home_hems"


class RepositoryCleanupTests(unittest.TestCase):
    def test_canonical_exports(self):
        from examples.single_home_hems.single_home_hems.controllers import EVBaselineController, EVMPCController
        from examples.single_home_hems.single_home_hems.controllers.ev import EVHorizon, build_ev_problem
        from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics, ev_operating_state
        for symbol in (EVBaselineController, EVMPCController, EVHorizon,
                       build_ev_problem, ev_metrics, ev_operating_state):
            self.assertTrue(callable(symbol))

    def test_obsolete_ev_paths_are_removed(self):
        for name in ("single_home_hems/controllers/ev_formulation.py",
                     "single_home_hems/controllers/ev_mpc.py",
                     "single_home_hems/ev_simulation.py", "run_ev_mpc.py",
                     "config/ev_audi_convex.json", "config/ev_nhts_milp.json"):
            with self.subTest(path=name):
                self.assertFalse((EXAMPLE / name).exists())

    def test_current_document_links_resolve(self):
        # Archived prose retains historical paths; only maintained guides are checked.
        documents = [ROOT / "README.md", ROOT / "CUSTOM_CHANGES.md",
                     EXAMPLE / "README.md", *(EXAMPLE / "docs").rglob("*.md")]
        for document in documents:
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
                if target.startswith(("http:", "https:", "#", "mailto:")):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target.split("#")[0]).exists())


if __name__ == "__main__":
    unittest.main()

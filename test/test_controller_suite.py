"""Focused validation and dispatch checks for the config-only controller front door."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.single_home_hems.single_home_hems.controller_suite import (
    ControllerCase,
    ControllerSuite,
    load_controller_suite,
    run_controller_suite,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "single_home_hems" / "config"
SUITE = CONFIG / "controller_suite.json"
FIVE_MODES = (
    "hvac_only",
    "water_heater_only",
    "joint_thermal",
    "bidirectional_charger_only",
    "whole_home_joint",
)


class ControllerSuiteConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write_suite(self, case: dict) -> Path:
        path = self.directory / "suite.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "output_directory": str(self.directory / "outputs"),
            "selected": ["case"],
            "variants": {"case": case},
        }), encoding="utf-8")
        return path

    def test_all_five_supplied_modes_validate_without_creating_outputs(self) -> None:
        destination = self.directory / "not_created"
        suite = load_controller_suite(
            SUITE, ROOT, case_names=list(FIVE_MODES), output_directory=destination,
        )
        with patch(
            "examples.single_home_hems.single_home_hems.controller_suite.subprocess.run"
        ) as launch:
            run_controller_suite(suite, validate_only=True)
        self.assertEqual(tuple(case.controller for case in suite.cases), FIVE_MODES)
        self.assertFalse(destination.exists())
        launch.assert_not_called()
        resolved = suite.resolved()
        self.assertEqual(resolved["selected_cases"], list(FIVE_MODES))
        self.assertEqual(resolved["cases"]["hvac_only"]["output_directory"],
                         str(destination.resolve() / "hvac_only"))
        self.assertEqual(suite.cases[0].effective_settings["controller"], "hvac")
        self.assertEqual(suite.cases[1].effective_settings["controller"], "water-heater")

    def test_thermal_mode_cannot_override_its_forced_controller(self) -> None:
        path = self.write_suite({
            "controller": "hvac_only",
            "arguments": {"controller": "water-heater", "duration_hours": 1},
        })
        with self.assertRaisesRegex(ValueError, "Unknown runner arguments.*controller"):
            load_controller_suite(path, ROOT)
        self.assertFalse((self.directory / "outputs").exists())

    def test_specialist_mode_rejects_unknown_argument(self) -> None:
        path = self.write_suite({
            "controller": "bidirectional_charger_only",
            "arguments": {"invented_forecast_flag": "value"},
        })
        with self.assertRaisesRegex(ValueError, "Unknown runner arguments"):
            load_controller_suite(path, ROOT)
        self.assertFalse((self.directory / "outputs").exists())

    def test_specialist_mode_rejects_missing_explicit_path(self) -> None:
        path = self.write_suite({
            "controller": "bidirectional_charger_only",
            "arguments": {"hub_seed_csv": "missing-seed.csv"},
        })
        with self.assertRaisesRegex(FileNotFoundError, "hub_seed_csv input does not exist"):
            load_controller_suite(path, ROOT)
        self.assertFalse((self.directory / "outputs").exists())


class ControllerSuiteDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "results"
        self.suite = ControllerSuite(
            path=self.directory / "suite.json",
            requested={"schema_version": 1, "selected": ["heat", "joint"]},
            output_directory=self.output,
            cases=(
                ControllerCase("heat", "hvac_only", ("heat_runner.py", "--x", "1"), {"horizon": 8}),
                ControllerCase("joint", "whole_home_joint", ("joint_runner.py", "--y", "2"), {"horizon": 8}),
            ),
        )

    def test_dispatch_saves_command_status_and_separate_logs(self) -> None:
        commands = []

        def launch(command, **kwargs):
            commands.append((command, kwargs["cwd"], kwargs["check"]))
            kwargs["stdout"].write(f"launched {command[1]}\n")
            return SimpleNamespace(returncode=0)

        with patch(
            "examples.single_home_hems.single_home_hems.controller_suite.subprocess.run",
            side_effect=launch,
        ):
            run_controller_suite(self.suite)
        self.assertEqual(
            [command for command, _, _ in commands],
            [
                [sys.executable, "heat_runner.py", "--x", "1"],
                [sys.executable, "joint_runner.py", "--y", "2"],
            ],
        )
        self.assertTrue(all(cwd == ROOT and check is False for _, cwd, check in commands))
        self.assertEqual(
            [entry["state"] for entry in json.loads((self.output / "run_status.json").read_text())],
            ["complete", "complete"],
        )
        self.assertIn("heat_runner.py", (self.output / "runner_logs" / "heat.log").read_text())
        self.assertIn("joint_runner.py", (self.output / "runner_logs" / "joint.log").read_text())
        self.assertEqual(
            json.loads((self.output / "resolved_controller_config.json").read_text())["selected_cases"],
            ["heat", "joint"],
        )
        self.assertTrue((self.output / "requested_controller_config.json").is_file())

    def test_failed_dispatch_records_failure_and_preserves_log(self) -> None:
        def launch(command, **kwargs):
            kwargs["stdout"].write(f"failed {command[1]}\n")
            return SimpleNamespace(returncode=7)

        with patch(
            "examples.single_home_hems.single_home_hems.controller_suite.subprocess.run",
            side_effect=launch,
        ) as process:
            with self.assertRaisesRegex(RuntimeError, "heat failed with exit code 7"):
                run_controller_suite(self.suite)
        process.assert_called_once()
        status = json.loads((self.output / "run_status.json").read_text())
        self.assertEqual(status[0]["state"], "failed")
        self.assertEqual(status[0]["return_code"], "7")
        self.assertEqual(len(status), 1)
        self.assertIn("failed heat_runner.py", (self.output / "runner_logs" / "heat.log").read_text())
        self.assertFalse((self.output / "runner_logs" / "joint.log").exists())


if __name__ == "__main__":
    unittest.main()

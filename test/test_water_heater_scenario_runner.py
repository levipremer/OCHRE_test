"""Scenario-runner defaults, CLI forwarding, and preservation of prior runs."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from examples.single_home_hems import run_water_heater_mpc_scenario as runner


class ScenarioRunnerTests(unittest.TestCase):
    def test_defaults_select_milp_persistence_operating_point(self):
        args = runner.parse_args([])
        self.assertEqual(args.water_heater_optimization, "milp")
        self.assertEqual(args.water_heater_solver, "GUROBI")
        self.assertEqual(args.controller, "water-heater")
        self.assertEqual(args.water_heater_model, "rc")
        self.assertEqual(args.forecast_type, "scenario")
        self.assertEqual(args.scenario_method, "daily_budget_high_rf_1")
        self.assertEqual(args.duration_hours, 72)
        self.assertEqual(args.water_heater_horizon_hours, 12)
        self.assertEqual(args.time_step_minutes, 1)
        self.assertEqual(args.water_draw_start, dt.datetime(2025, 11, 6))
        self.assertEqual(args.water_draw_file.name, "water_flow_dchouse_l_per_min.csv")
        self.assertEqual(args.water_heater_comfort_weight, .1)
        self.assertEqual(args.water_heater_legionella_weight, .001)
        self.assertEqual(args.water_heater_persistence_weight, 1.0)
        self.assertEqual(args.water_heater_cycle_limit, "off")
        self.assertEqual(args.water_heater_dwell_constraints, "off")
        self.assertFalse(args.skip_optimizer_plan_figures)

    def test_convex_alias_selects_its_solver_and_persistence_operating_point(self):
        for option in ("--optimization", "--water-heater-optimization"):
            with self.subTest(option=option):
                args = runner.parse_args([option, "convex"])
                self.assertEqual(args.water_heater_optimization, "convex")
                self.assertEqual(args.water_heater_solver, "CLARABEL")
                self.assertEqual(args.water_heater_comfort_weight, .01)
                self.assertEqual(args.water_heater_legionella_weight, .01)
                self.assertEqual(args.water_heater_persistence_weight, .001)
                self.assertFalse(args.skip_optimizer_plan_figures)

    def test_explicit_weights_method_duration_and_solver_are_preserved(self):
        args = runner.parse_args([
            "--optimization", "convex", "--water-heater-solver", "GUROBI",
            "--water-heater-comfort-weight", "0",
            "--water-heater-legionella-weight", ".2",
            "--water-heater-persistence-weight", ".3",
            "--scenario-method", "daily_budget_high", "--duration-hours", "24",
            "--water-heater-cycle-limit", "on", "--water-heater-dwell-constraints", "on",
            "--skip-optimizer-plan-figures",
        ])
        self.assertEqual(args.water_heater_solver, "GUROBI")
        self.assertEqual(args.water_heater_comfort_weight, 0)
        self.assertEqual(args.water_heater_legionella_weight, .2)
        self.assertEqual(args.water_heater_persistence_weight, .3)
        self.assertEqual(args.scenario_method, "daily_budget_high")
        self.assertEqual(args.duration_hours, 24)
        self.assertEqual(args.water_heater_cycle_limit, "on")
        self.assertEqual(args.water_heater_dwell_constraints, "on")
        self.assertTrue(args.skip_optimizer_plan_figures)

    def test_titles_are_single_directory_names(self):
        self.assertEqual(runner.run_title("  Morning comparison  "), "Morning comparison")
        self.assertEqual(runner.parse_args(["--run-title", "Morning comparison"]).title, "Morning comparison")
        for title in ("", " ", ".", "..", "../outside", "/absolute", "folder/run", "folder\\run", "run\nname", "run\x00name"):
            with self.subTest(title=repr(title)), self.assertRaises(argparse.ArgumentTypeError):
                runner.run_title(title)

    def test_runner_rejects_changes_that_escape_its_scope_or_output_root(self):
        for arguments in (
            ["--controller", "all"], ["--forecast-type", "perfect"],
            ["--output-directory", "/tmp/elsewhere"], ["--title", "../elsewhere"],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.parse_args(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_repeated_default_and_custom_titles_preserve_previous_run_data(self):
        for arguments, title in (([], "milp"), (["--title", "My comparison"], "My comparison")):
            with self.subTest(title=title), tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                output_root = Path(temporary) / "isolated_runs"
                with patch.object(runner, "OUTPUT_ROOT", output_root), patch.object(runner.run_simulation, "main", return_value=0) as simulate:
                    self.assertEqual(runner.main(arguments), 0)
                    first = output_root / title
                    sentinel = first / "existing_results.csv"
                    sentinel.write_text("timestamp,result\n0,42\n")
                    first_settings = (first / "run_settings.json").read_text()
                    self.assertEqual(runner.main(arguments), 0)
                second = output_root / f"{title}_2"
                self.assertTrue(first.is_dir())
                self.assertTrue(second.is_dir())
                self.assertEqual(sentinel.read_text(), "timestamp,result\n0,42\n")
                self.assertEqual((first / "run_settings.json").read_text(), first_settings)
                self.assertEqual(simulate.call_count, 2)
                for call, destination in zip(simulate.call_args_list, (first, second)):
                    forwarded = call.args[0]
                    self.assertEqual(forwarded.output_directory, destination)
                    self.assertFalse(forwarded.skip_optimizer_plan_figures)
                    settings = json.loads((destination / "run_settings.json").read_text())
                    self.assertEqual(settings["title"], title)
                    self.assertEqual(settings["output_directory"], str(destination))
                    self.assertEqual(settings["forecast_type"], "scenario")
                    self.assertEqual(settings["water_heater_persistence_weight"], 1.0)

    def test_main_forwards_the_standard_simulation_namespace_and_return_code(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output_root = Path(temporary) / "isolated_runs"
            source = Path(temporary) / "custom_draws.csv"
            arguments = [
                "--optimization", "convex", "--home-id", "home_02",
                "--title", "Custom RC", "--water-heater-model", "rc-old",
                "--scenario-method", "daily_budget_high", "--duration-hours", "48",
                "--water-draw-file", str(source), "--water-draw-start", "2025-11-07",
                "--water-heater-hp-only", "--initial-temperature-c", "23.5",
                "--water-heater-persistence-weight", ".07",
            ]
            expected = runner.parse_args(arguments)
            with patch.object(runner, "OUTPUT_ROOT", output_root), patch.object(runner.run_simulation, "main", return_value=17) as simulate:
                self.assertEqual(runner.main(arguments), 17)
            simulate.assert_called_once()
            forwarded = simulate.call_args.args[0]
            expected.output_directory = output_root / "Custom RC"
            self.assertEqual(vars(forwarded), vars(expected))
            self.assertFalse(forwarded.skip_optimizer_plan_figures)
            self.assertEqual(forwarded.output_directory.parent, output_root)
            settings = json.loads((forwarded.output_directory / "run_settings.json").read_text())
            self.assertEqual(settings["water_draw_file"], str(source))
            self.assertEqual(settings["water_draw_start"], "2025-11-07 00:00:00")
            self.assertEqual(settings["water_heater_solver"], "CLARABEL")
            self.assertEqual(settings["water_heater_persistence_weight"], .07)


if __name__ == "__main__":
    unittest.main()

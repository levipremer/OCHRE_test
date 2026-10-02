"""Isolated water-heater CLI tests: no simulations or optimization solves."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import io
import unittest
from contextlib import redirect_stderr
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "examples" / "single_home_hems" / "run_simulation.py"
CONTROLLER = RUNNER.parent / "single_home_hems" / "controllers" / "water_heater.py"


def controller_defaults():
    """Read only the five relevant dataclass defaults, without importing CVXPY."""
    tree = ast.parse(CONTROLLER.read_text())
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WaterHeaterPredictiveController"
    )
    names = {"max_cycles", "enforce_cycle_limit", "enforce_dwell_constraints", "dwell_time_on", "dwell_time_off"}
    defaults = {}
    for node in controller.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name) or node.target.id not in names:
            continue
        if node.target.id.startswith("dwell_time_"):
            keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords}
            defaults[node.target.id] = dt.timedelta(**keywords)
        else:
            defaults[node.target.id] = ast.literal_eval(node.value)
    return defaults


def isolated_runner():
    """Compile only requested functions with all external dependencies mocked."""
    tree = ast.parse(RUNNER.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"default_output_subdirectory", "build_argument_parser", "parse_args", "build_device_controllers", "main"}
    ]
    defaults = controller_defaults()
    water_heater = MagicMock(
        side_effect=lambda **options: SimpleNamespace(
            **{**defaults, **options}, supplied_options=dict(options)
        )
    )
    config = SimpleNamespace(validate=MagicMock())
    home = SimpleNamespace(home_id="home_01", evs=())
    sensors = MagicMock()
    sensors.default_sensor_nodes.return_value = (3, 10)
    sensors.return_value = SimpleNamespace(model_nodes=12, sensor_columns=("upper", "lower"))
    namespace = {
        "argparse": argparse, "dt": dt, "Path": Path,
        "DeviceController": object,
        "EXAMPLE_ROOT": Path("/unused/example"), "REPO_ROOT": Path("/unused/repo"),
        "FIVE_HOME_CONFIG": Path("/unused/config"),
        "HOME_IDS": ("home_01",),
        "CONTROLLER_CHOICES": ("none", "water-heater", "hvac", "ev", "all"),
        "DEFAULT_MODEL_PATH": Path("/unused/state-space"),
        "DEFAULT_RC_MODEL_PATH": Path("/unused/rc"),
        "DEFAULT_RC_OLD_MODEL_PATH": Path("/unused/rc-old"),
        "DEFAULT_RC_STRATIFIED_MODEL_PATH": Path("/unused/rc-stratified"),
        "WaterHeaterPredictiveController": water_heater,
        "HVACPredictiveController": MagicMock(), "EVPredictiveController": MagicMock(),
        "load_configuration": MagicMock(return_value=config),
        "select_home": MagicMock(return_value=home),
        "replace": MagicMock(return_value=config),
        "WaterHeaterSensorLayout": sensors,
        "HVACInternalGainRegime": MagicMock(), "HEMSController": MagicMock(),
        "SingleHomeHEMSSimulation": MagicMock(),
    }
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(functions), type_ignores=[]))
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace


def parsed(namespace, *arguments):
    with patch("sys.argv", ["run_simulation.py", *arguments]):
        return namespace["parse_args"]()


def mocked_main(*arguments):
    namespace = isolated_runner()
    args = parsed(namespace, "--controller", "water-heater", *arguments)
    namespace["parse_args"] = MagicMock(return_value=args)
    namespace["build_device_controllers"] = MagicMock(wraps=namespace["build_device_controllers"])
    result = namespace["main"]()
    return namespace, args, result


class SingleHomeWaterHeaterCliTests(unittest.TestCase):
    def test_scenario_options_reach_controller_and_simulation(self) -> None:
        namespace, args, _ = mocked_main(
            "--forecast-type", "scenario", "--scenario-method", "daily_budget_high",
            "--water-draw-start", "2025-11-06T00:00:00",
            "--skip-optimizer-plan-figures",
        )
        options = namespace["WaterHeaterPredictiveController"].call_args.kwargs
        self.assertEqual(options["forecast_type"], "scenario")
        self.assertEqual(options["scenario_method"], "daily_budget_high")
        self.assertEqual(options["horizon"], dt.timedelta(hours=12))
        simulation = namespace["SingleHomeHEMSSimulation"].call_args.kwargs
        self.assertEqual(simulation["water_draw_file"].name, "water_flow_dchouse_l_per_min.csv")
        self.assertEqual(simulation["water_draw_start"], dt.datetime(2025, 11, 6))
        self.assertFalse(simulation["save_optimizer_plan_figures"])

    def test_scenario_cli_rejects_invalid_configuration(self) -> None:
        for arguments in (
            ("--forecast-type", "scenario", "--water-heater-horizon-hours", "6"),
            ("--water-heater-horizon-hours", "0"),
            ("--water-draw-start", "2025-11-06"),
            ("--water-heater-solver", "CLARABEL", "--water-heater-optimization", "milp"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                mocked_main(*arguments)

    def test_forecast_defaults_preserve_perfect_and_separate_output_names(self) -> None:
        namespace = isolated_runner()
        self.assertEqual(parsed(namespace).forecast_type, "perfect")
        name = namespace["default_output_subdirectory"](
            "water-heater", "convex", "rc", "scenario", "daily_budget_high_rf_1",
        )
        self.assertEqual(name, "water_heater_only_convex_rc_scenario_daily_budget_high_rf_1")

    def test_parser_defaults_keep_cycle_on_and_dwell_off(self) -> None:
        args = parsed(isolated_runner())
        self.assertEqual(args.water_heater_cycle_limit, "on")
        self.assertEqual(args.water_heater_dwell_constraints, "off")

    def test_parser_accepts_all_four_toggle_combinations(self) -> None:
        for cycle, dwell in product(("on", "off"), repeat=2):
            with self.subTest(cycle=cycle, dwell=dwell):
                args = parsed(
                    isolated_runner(), "--water-heater-cycle-limit", cycle,
                    "--water-heater-dwell-constraints", dwell,
                )
                self.assertEqual(args.water_heater_cycle_limit, cycle)
                self.assertEqual(args.water_heater_dwell_constraints, dwell)

    def test_parser_rejects_values_outside_on_off_and_missing_values(self) -> None:
        for option in ("--water-heater-cycle-limit", "--water-heater-dwell-constraints"):
            for value in ("true", "false", "1", "none", "ON", None):
                arguments = (option,) if value is None else (option, value)
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parsed(isolated_runner(), *arguments)

    def test_builder_omitted_or_none_flags_preserve_controller_defaults(self) -> None:
        for options in ({}, {"water_heater_enforce_cycle_limit": None, "water_heater_enforce_dwell_constraints": None}):
            namespace = isolated_runner()
            controller = namespace["build_device_controllers"]("water-heater", (), **options)[0]
            self.assertTrue(controller.enforce_cycle_limit)
            self.assertFalse(controller.enforce_dwell_constraints)
            self.assertNotIn("enforce_cycle_limit", controller.supplied_options)
            self.assertNotIn("enforce_dwell_constraints", controller.supplied_options)

    def test_builder_all_four_combinations_forward_explicit_bools(self) -> None:
        for cycle, dwell in product((True, False), repeat=2):
            with self.subTest(cycle=cycle, dwell=dwell):
                namespace = isolated_runner()
                controller = namespace["build_device_controllers"](
                    "water-heater", (), water_heater_enforce_cycle_limit=cycle,
                    water_heater_enforce_dwell_constraints=dwell,
                )[0]
                self.assertIs(controller.enforce_cycle_limit, cycle)
                self.assertIs(controller.enforce_dwell_constraints, dwell)
                self.assertIs(controller.supplied_options["enforce_cycle_limit"], cycle)
                self.assertIs(controller.supplied_options["enforce_dwell_constraints"], dwell)

    def test_max_cycles_is_independent_of_both_constraint_toggles(self) -> None:
        for limit, cycle, dwell in product((2, 5), (True, False), (True, False)):
            with self.subTest(limit=limit, cycle=cycle, dwell=dwell):
                namespace = isolated_runner()
                controller = namespace["build_device_controllers"](
                    "water-heater", (), water_heater_max_cycles=limit,
                    water_heater_enforce_cycle_limit=cycle,
                    water_heater_enforce_dwell_constraints=dwell,
                )[0]
                self.assertEqual(controller.max_cycles, limit)
                self.assertIs(controller.enforce_cycle_limit, cycle)
                self.assertIs(controller.enforce_dwell_constraints, dwell)

    def test_dwell_toggle_preserves_existing_on_and_off_durations(self) -> None:
        defaults = controller_defaults()
        self.assertEqual(defaults["dwell_time_on"], dt.timedelta(minutes=60))
        self.assertEqual(defaults["dwell_time_off"], dt.timedelta(minutes=15))
        for dwell in (True, False):
            namespace = isolated_runner()
            controller = namespace["build_device_controllers"](
                "water-heater", (), water_heater_enforce_dwell_constraints=dwell,
            )[0]
            self.assertEqual(controller.dwell_time_on, dt.timedelta(minutes=60))
            self.assertEqual(controller.dwell_time_off, dt.timedelta(minutes=15))
            self.assertNotIn("dwell_time_on", controller.supplied_options)
            self.assertNotIn("dwell_time_off", controller.supplied_options)

    def test_non_water_heater_selections_ignore_water_heater_toggles(self) -> None:
        for selection in ("none", "ev", "hvac"):
            with self.subTest(selection=selection):
                namespace = isolated_runner()
                namespace["build_device_controllers"](
                    selection, (), water_heater_enforce_cycle_limit=False,
                    water_heater_enforce_dwell_constraints=True,
                )
                namespace["WaterHeaterPredictiveController"].assert_not_called()

    def test_main_wires_all_four_cli_combinations_to_builder_and_mock_run(self) -> None:
        for cycle, dwell in product(("on", "off"), repeat=2):
            with self.subTest(cycle=cycle, dwell=dwell):
                namespace, _, result = mocked_main(
                    "--water-heater-cycle-limit", cycle,
                    "--water-heater-dwell-constraints", dwell,
                    "--water-heater-max-cycles", "5",
                )
                self.assertEqual(result, 0)
                builder_options = namespace["build_device_controllers"].call_args.kwargs
                self.assertIs(builder_options["water_heater_enforce_cycle_limit"], cycle == "on")
                self.assertIs(builder_options["water_heater_enforce_dwell_constraints"], dwell == "on")
                controller_options = namespace["WaterHeaterPredictiveController"].call_args.kwargs
                self.assertEqual(controller_options["max_cycles"], 5)
                self.assertIs(controller_options["enforce_cycle_limit"], cycle == "on")
                self.assertIs(controller_options["enforce_dwell_constraints"], dwell == "on")
                namespace["SingleHomeHEMSSimulation"].return_value.run.assert_called_once_with()

    def test_main_omitted_flags_keep_default_constraint_behavior(self) -> None:
        namespace, _, _ = mocked_main()
        controller_options = namespace["WaterHeaterPredictiveController"].call_args.kwargs
        self.assertTrue(controller_options["enforce_cycle_limit"])
        self.assertFalse(controller_options["enforce_dwell_constraints"])
        self.assertNotIn("max_cycles", controller_options)

    def test_initial_temperature_defaults_custom_and_random_mode_are_preserved(self) -> None:
        for arguments, expected in (((), 21.0), (("--initial-temperature-c", "19"), 19.0), (("--randomize-initial-temperature",), None)):
            with self.subTest(arguments=arguments):
                namespace, _, _ = mocked_main(*arguments)
                self.assertEqual(
                    namespace["SingleHomeHEMSSimulation"].call_args.kwargs["initial_temperature_c"],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()

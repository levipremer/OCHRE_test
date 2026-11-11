"""HVAC-only command-line wiring without running an OCHRE simulation."""

from __future__ import annotations

import datetime as dt
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from examples.single_home_hems import run_simulation
from examples.single_home_hems.single_home_hems.controllers.hvac import (
    HVACPredictiveController,
)


class SingleHomeHVACCliTests(unittest.TestCase):
    def _run_mocked(self, *arguments: str):
        args = run_simulation.parse_args(["--controller", "hvac", *arguments])
        with patch.object(run_simulation, "SingleHomeHEMSSimulation") as simulation:
            result = run_simulation.main(args)
        self.assertEqual(result, 0)
        simulation.return_value.run.assert_called_once_with()
        return simulation.call_args

    def test_hvac_only_range_and_controller_configuration(self) -> None:
        call = self._run_mocked(
            "--home-id", "home_01",
            "--start-time", "2018-01-03T06:00:00",
            "--duration-hours", "13",
            "--hvac-solver", "CLARABEL",
            "--hvac-comfort-band-c", "2.0",
            "--hvac-comfort-weight", "3.0",
            "--hvac-persistence-weight", "0.4",
        )
        config, home, hems = call.args
        self.assertEqual(home.home_id, "home_01")
        self.assertEqual(config.start_time, dt.datetime(2018, 1, 3, 6))
        self.assertEqual(config.duration, dt.timedelta(hours=13))
        # The plant still runs at its one-minute step; the predictor has a
        # separate fifteen-minute model step and a twelve-hour horizon.
        self.assertEqual(config.time_step, dt.timedelta(minutes=1))
        self.assertEqual(len(hems.device_controllers), 1)
        controller = hems.device_controllers[0]
        self.assertIsInstance(controller, HVACPredictiveController)
        self.assertEqual(controller.horizon, dt.timedelta(hours=12))
        self.assertEqual(controller.model_timestep, dt.timedelta(minutes=15))
        self.assertEqual(controller.optimization_kind, "convex")
        self.assertEqual(controller.solver, "CLARABEL")
        self.assertEqual(controller.comfort_band_c, 2.0)
        self.assertEqual(controller.comfort_weight, 3.0)
        self.assertEqual(controller.persistence_weight, 0.4)
        self.assertNotIn("hvac_warmup_duration", call.kwargs)
        self.assertIsNone(call.kwargs["weather_archive"])

    def test_explicit_hvac_horizon_warmup_and_gurobi_budget_reach_runtime(self) -> None:
        call = self._run_mocked(
            "--duration-hours", "48",
            "--hvac-optimization", "mixed integer",
            "--hvac-horizon-hours", "20",
            "--hvac-warmup-hours", "16",
            "--hvac-max-deviation-slack-weight", "12000",
            "--hvac-time-limit-seconds", "300",
            "--hvac-mip-gap", "0.05",
            "--hvac-solver-threads", "1",
        )
        controller = call.args[2].device_controllers[0]
        self.assertEqual(controller.horizon, pd.Timedelta(hours=20))
        self.assertEqual(call.kwargs["hvac_warmup_duration"], pd.Timedelta(hours=16))
        self.assertEqual(controller.max_deviation_slack_weight, 12_000)
        self.assertEqual(controller.solver_options, {
            "TimeLimit": 300.0, "MIPGap": 0.05, "Threads": 1,
        })

    def test_new_hvac_options_reject_invalid_ranges_and_solver(self) -> None:
        for option, value in (
            ("--hvac-horizon-hours", "0"),
            ("--hvac-horizon-hours", "0.3"),
            ("--hvac-warmup-hours", "-1"),
            ("--hvac-warmup-hours", "0.005"),
            ("--hvac-max-deviation-slack-weight", "nan"),
            ("--hvac-time-limit-seconds", "0"),
            ("--hvac-mip-gap", "1.1"),
            ("--hvac-solver-threads", "0"),
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self._run_mocked("--duration-hours", "1", option, value)
        with self.assertRaisesRegex(ValueError, "supported only with GUROBI"):
            self._run_mocked(
                "--duration-hours", "1", "--hvac-optimization", "mixed integer",
                "--hvac-solver", "CLARABEL",
                "--hvac-mip-gap", "0.05",
            )
        with self.assertRaisesRegex(ValueError, "requires --hvac-optimization"):
            self._run_mocked("--duration-hours", "1", "--hvac-mip-gap", "0.05")

    def test_predict_weather_loads_explicit_prepared_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.parquet"
            run = pd.Timestamp("2018-01-03T00:00:00Z")
            valid = run + pd.Timedelta(hours=1)
            pd.DataFrame([{
                "run_time_utc": run,
                "valid_time_utc": valid,
                "valid_time_local": valid.tz_convert("America/Indiana/Indianapolis"),
                "temp_air": 1.0,
                "relative_humidity": 0.5,
                "ghi": 0.0,
                "dni": 0.0,
                "dhi": 0.0,
            }]).to_parquet(path)
            call = self._run_mocked(
                "--duration-hours", "13",
                "--weather-forecast-mode", "predict",
                "--weather-forecast-file", str(path),
            )
            self.assertEqual(len(call.kwargs["weather_archive"].data), 1)

    def test_predict_weather_requires_archive_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --weather-forecast-file"):
            self._run_mocked(
                "--duration-hours", "13", "--weather-forecast-mode", "predict"
            )

    def test_default_hvac_output_is_under_home_and_identifies_formulation_and_model(self) -> None:
        self.assertEqual(
            run_simulation.default_output_subdirectory(
                "hvac", "convex", "state-space",
                simulation_duration=dt.timedelta(hours=13),
            ),
            "cvx_state_space_regularized_13h",
        )
        call = self._run_mocked(
            "--duration-hours", "13",
        )
        output = call.args[0].output_directory
        self.assertEqual(output.parent.name, "home_01")
        self.assertTrue(output.name.startswith("cvx_state_space_regularized_"))
        self.assertEqual(call.args[2].device_controllers[0].persistence_weight, 1.0)
        self.assertEqual(call.args[2].device_controllers[0].optimization_kind, "convex")
        self.assertFalse(call.args[2].device_controllers[0].enforce_degree_hour_budget)
        self.assertFalse(call.args[2].device_controllers[0].enforce_instantaneous_cap)
        self.assertEqual(call.args[2].device_controllers[0].degree_hour_limit_c_hours, 2.0)
        self.assertEqual(call.args[2].device_controllers[0].max_deviation_c, 1.0)
        self.assertEqual(call.args[2].device_controllers[0].degree_hour_slack_weight, 100.0)
        self.assertFalse(call.args[2].device_controllers[0].enforce_cycle_limit)
        self.assertFalse(call.args[2].device_controllers[0].enforce_dwell_constraints)
        self.assertEqual(call.args[2].device_controllers[0].max_cycles, 3)
        self.assertEqual(
            call.args[2].device_controllers[0].dwell_time_on,
            dt.timedelta(minutes=60),
        )
        self.assertEqual(
            call.args[2].device_controllers[0].dwell_time_off,
            dt.timedelta(minutes=15),
        )

    def test_mixed_integer_hvac_variant_uses_gurobi_and_separate_output_prefix(self) -> None:
        self.assertEqual(
            run_simulation.default_output_subdirectory(
                "hvac", "mixed integer", "state-space",
                simulation_duration=dt.timedelta(hours=13),
            ),
            "mi_state_space_regularized_13h",
        )
        call = self._run_mocked(
            "--duration-hours", "13",
            "--hvac-optimization", "mixed integer",
            "--hvac-solver", "GUROBI",
        )
        controller = call.args[2].device_controllers[0]
        self.assertEqual(controller.optimization_kind, "mixed integer")
        self.assertEqual(controller.solver, "GUROBI")
        self.assertTrue(call.args[0].output_directory.name.startswith(
            "mi_state_space_regularized_13h"
        ))

    def test_mixed_integer_hvac_rejects_clarabel(self) -> None:
        with self.assertRaisesRegex(ValueError, "CLARABEL supports convex HVAC"):
            self._run_mocked(
                "--duration-hours", "13",
                "--hvac-optimization", "mixed integer",
                "--hvac-solver", "CLARABEL",
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_simulation.parse_args([
                "--controller", "hvac", "--duration-hours", "13",
                "--hvac-optimization", "milp",
            ])

    def test_automatic_hvac_directory_advances_past_existing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "cvx_state_space_regularized_13h"
            self.assertEqual(run_simulation.next_available_output_directory(base), base)
            base.mkdir()
            self.assertEqual(
                run_simulation.next_available_output_directory(base),
                base.with_name(base.name + "_2"),
            )
            base.with_name(base.name + "_2").mkdir()
            self.assertEqual(
                run_simulation.next_available_output_directory(base),
                base.with_name(base.name + "_3"),
            )

    def test_explicit_output_directory_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "my_hvac_run"
            call = self._run_mocked(
                "--start-time", "2018-01-03T06:00:00",
                "--duration-hours", "13",
                "--output-directory", str(requested),
            )
            self.assertEqual(call.args[0].output_directory, requested.resolve())

    def test_start_date_selects_midnight_and_duration(self) -> None:
        call = self._run_mocked(
            "--start-date", "2018-01-03",
            "--duration-hours", "13",
        )
        self.assertEqual(call.args[0].start_time, dt.datetime(2018, 1, 3))
        self.assertEqual(call.args[0].duration, dt.timedelta(hours=13))

    def test_hvac_requires_duration_and_rejects_end_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --duration-hours"):
            self._run_mocked("--start-date", "2018-01-03")
        with self.assertRaisesRegex(ValueError, "use --duration-hours"):
            self._run_mocked("--end-time", "2018-01-03T19:00:00")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_simulation.parse_args([
                "--controller", "hvac", "--duration-hours", "13",
                "--end-time", "2018-01-03T19:00:00",
            ])

    def test_hvac_persistence_weight_rejects_negative_and_nonfinite_values(self) -> None:
        for invalid in ("-0.1", "nan", "inf"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "--hvac-persistence-weight must be finite and nonnegative"
            ):
                self._run_mocked(
                    "--duration-hours", "13",
                    "--hvac-persistence-weight", invalid,
                )

    def test_hvac_comfort_switches_are_independent_in_both_formulations(self) -> None:
        for optimization in ("convex", "mixed integer"):
            for budget_enabled in (False, True):
                for cap_enabled in (False, True):
                    with self.subTest(optimization=optimization, budget=budget_enabled, cap=cap_enabled):
                        call = self._run_mocked(
                            "--duration-hours", "13",
                            "--hvac-optimization", optimization,
                            "--hvac-degree-hour-budget", "on" if budget_enabled else "off",
                            "--hvac-instantaneous-cap", "on" if cap_enabled else "off",
                            "--hvac-degree-hour-limit-c-hours", "3.5",
                            "--hvac-max-deviation-c", "0.75",
                            "--hvac-degree-hour-slack-weight", "250",
                        )
                        controller = call.args[2].device_controllers[0]
                        self.assertEqual(controller.enforce_degree_hour_budget, budget_enabled)
                        self.assertEqual(controller.enforce_instantaneous_cap, cap_enabled)
                        self.assertEqual(controller.degree_hour_limit_c_hours, 3.5)
                        self.assertEqual(controller.max_deviation_c, 0.75)
                        self.assertEqual(controller.degree_hour_slack_weight, 250.0)

    def test_hvac_comfort_limits_and_weight_reject_invalid_values(self) -> None:
        for option in (
            "--hvac-degree-hour-limit-c-hours",
            "--hvac-max-deviation-c",
            "--hvac-degree-hour-slack-weight",
        ):
            for invalid in ("-0.1", "nan", "inf"):
                with self.subTest(option=option, value=invalid), self.assertRaisesRegex(
                    ValueError, f"{option} must be finite and nonnegative"
                ):
                    self._run_mocked("--duration-hours", "13", option, invalid)

    def test_mixed_integer_cycle_and_dwell_switches_are_independent(self) -> None:
        for cycle_enabled in (False, True):
            for dwell_enabled in (False, True):
                with self.subTest(cycle=cycle_enabled, dwell=dwell_enabled):
                    call = self._run_mocked(
                        "--duration-hours", "13",
                        "--hvac-optimization", "mixed integer",
                        "--hvac-cycle-limit", "on" if cycle_enabled else "off",
                        "--hvac-dwell-constraints", "on" if dwell_enabled else "off",
                        "--hvac-max-cycles", "4",
                        "--hvac-dwell-on-minutes", "45",
                        "--hvac-dwell-off-minutes", "30",
                    )
                    controller = call.args[2].device_controllers[0]
                    self.assertEqual(controller.enforce_cycle_limit, cycle_enabled)
                    self.assertEqual(controller.enforce_dwell_constraints, dwell_enabled)
                    self.assertEqual(controller.max_cycles, 4)
                    self.assertEqual(controller.dwell_time_on, dt.timedelta(minutes=45))
                    self.assertEqual(controller.dwell_time_off, dt.timedelta(minutes=30))

    def test_hvac_cycle_and_dwell_require_mixed_integer_formulation(self) -> None:
        for switch in ("--hvac-cycle-limit", "--hvac-dwell-constraints"):
            with self.subTest(switch=switch), self.assertRaisesRegex(
                ValueError, "require --hvac-optimization 'mixed integer'"
            ):
                self._run_mocked("--duration-hours", "13", switch, "on")

    def test_hvac_cycle_and_dwell_limits_reject_invalid_values(self) -> None:
        for invalid in ("0", "-1"):
            with self.subTest(max_cycles=invalid), self.assertRaisesRegex(
                ValueError, "--hvac-max-cycles must be positive"
            ):
                self._run_mocked("--duration-hours", "13", "--hvac-max-cycles", invalid)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_simulation.parse_args([
                "--controller", "hvac", "--duration-hours", "13",
                "--hvac-max-cycles", "1.5",
            ])
        for option in ("--hvac-dwell-on-minutes", "--hvac-dwell-off-minutes"):
            for invalid in ("0", "-1", "nan", "inf"):
                with self.subTest(option=option, value=invalid), self.assertRaisesRegex(
                    ValueError, f"{option} must be finite and positive"
                ):
                    self._run_mocked("--duration-hours", "13", option, invalid)


if __name__ == "__main__":
    unittest.main()

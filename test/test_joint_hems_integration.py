"""Joint HEMS runner and simulation wiring without an optimization solve."""

from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems import run_joint_hems
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems import simulation as simulation_module
from examples.single_home_hems.single_home_hems.controllers.joint_hems import (
    JointHEMSPredictiveController,
)
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    _predictive_controllers,
    select_home,
)
from examples.single_home_hems.single_home_hems.training.water_heater import (
    WaterHeaterSensorLayout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPO_ROOT / "examples" / "five_home_block" / "config"


class JointHEMSIntegrationTests(unittest.TestCase):
    def test_cli_builds_selected_k7_scenario_and_common_eight_hour_horizon(self) -> None:
        args = run_joint_hems.parse_args(["--duration-hours", "1"])
        joint = run_joint_hems.build_controller(args)

        self.assertEqual(args.scenario_count, 7)
        self.assertEqual(args.weather_forecast_mode, "perfect")
        self.assertEqual(joint.water_heater_controller.scenario_method, "rmse_volume_6h_k7_rf_1")
        self.assertEqual(joint.water_heater_controller.forecast_type, "scenario")
        self.assertEqual(joint.water_heater_controller.optimization_kind, "milp")
        self.assertEqual(joint.hvac_controller.optimization_kind, "mixed integer")
        self.assertTrue(joint.hvac_controller.enforce_degree_hour_budget)
        self.assertEqual(joint.horizon, pd.Timedelta(hours=8))
        self.assertEqual(joint.water_heater_controller.horizon, joint.hvac_controller.horizon)
        self.assertEqual(joint.water_heater_controller.model_timestep, pd.Timedelta(minutes=15))
        self.assertEqual(joint.hvac_controller.model_timestep, pd.Timedelta(minutes=15))
        self.assertEqual(joint.peak_weight_usd_per_kw, 0.005)
        self.assertEqual(joint.peak_window, "all")
        baseline = run_joint_hems.build_controller(
            run_joint_hems.parse_args([
                "--duration-hours", "1", "--peak-weight-usd-per-kw", "0",
            ])
        )
        self.assertEqual(baseline.peak_weight_usd_per_kw, 0.0)

    def test_cli_builds_both_approved_peak_windows(self) -> None:
        for window, hours in (("tou", (15, 18)), ("night", (3, 7))):
            with self.subTest(window=window):
                joint = run_joint_hems.build_controller(run_joint_hems.parse_args([
                    "--duration-hours", "8", "--peak-window", window, "--mip-gap", "0",
                ]))
                self.assertEqual(joint.peak_window, window)
                self.assertEqual(joint.peak_window_hours, hours)
                self.assertEqual(joint.solver_options["MIPGap"], 0.0)
                self.assertEqual(joint.horizon, pd.Timedelta(hours=8))

    def test_cli_passes_one_joint_controller_to_simulation_and_records_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "joint_run"
            rest_provider = SimpleNamespace(
                method_label="conservative_rf_first_2h__p90_20d_remaining",
                metadata={"history_days": 20},
            )
            with (
                patch.object(run_joint_hems, "SingleHomeHEMSSimulation") as simulation,
                patch.object(
                    run_joint_hems.RestOfHomeHybridForecast, "from_csv",
                    return_value=rest_provider,
                ) as forecast_factory,
                redirect_stdout(io.StringIO()),
            ):
                result = run_joint_hems.main([
                    "--duration-hours", "1", "--output-directory", str(output),
                ])
            self.assertEqual(result, 0)
            simulation.return_value.run.assert_called_once_with()
            forecast_factory.assert_called_once()
            self.assertEqual(forecast_factory.call_args.kwargs["history_days"], 20)
            config, _home, hems = simulation.call_args.args
            self.assertEqual(config.duration, dt.timedelta(hours=1))
            self.assertEqual(config.time_step, dt.timedelta(minutes=1))
            self.assertEqual(len(hems.device_controllers), 1)
            self.assertIsInstance(hems.device_controllers[0], JointHEMSPredictiveController)
            self.assertEqual(hems.device_controllers[0].peak_weight_usd_per_kw, 0.005)
            self.assertTrue(simulation.call_args.kwargs["water_heater_hp_only"])
            self.assertIs(simulation.call_args.kwargs["rest_of_home_forecast"], rest_provider)
            self.assertIsNone(simulation.call_args.kwargs["weather_archive"])
            settings = json.loads((output / "joint_run_settings.json").read_text())
            self.assertEqual(settings["water_scenario_method"], "rmse_volume_6h_k7_rf_1")
            self.assertEqual(settings["horizon_hours"], 8)
            self.assertEqual(settings["hvac_degree_hour_history_hours"], 16)
            self.assertTrue(settings["peak_power_objective_enabled"])
            self.assertEqual(settings["peak_power_weight_usd_per_kw"], 0.005)
            self.assertEqual(settings["peak_power_window"], "all")
            self.assertEqual(settings["peak_power_window_start_hour"], 0)
            self.assertEqual(settings["peak_power_window_end_hour"], 24)
            self.assertTrue(settings["peak_power_excludes_ev"])
            self.assertEqual(
                settings["rest_of_home_forecast_method"], rest_provider.method_label
            )

    def test_joint_cli_routes_predict_weather_only_to_controlled_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "joint_run"
            path = root / "forecast.parquet"
            run = pd.Timestamp("2018-01-01T00:00:00Z")
            pd.DataFrame([{
                "run_time_utc": run,
                "valid_time_utc": valid,
                "valid_time_local": valid.tz_convert("America/Indiana/Indianapolis"),
                "temp_air": 1.0,
                "relative_humidity": 0.5,
                "ghi": 0.0,
                "dni": 0.0,
                "dhi": 0.0,
            } for valid in (run + pd.Timedelta(hours=hour) for hour in range(1, 25))]).to_parquet(path)
            with (
                patch.object(run_joint_hems, "SingleHomeHEMSSimulation") as simulation,
                patch.object(
                    run_joint_hems.RestOfHomeHybridForecast, "from_csv",
                    return_value=SimpleNamespace(method_label="test", metadata={}),
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = run_joint_hems.main([
                    "--duration-hours", "1", "--output-directory", str(output),
                    "--weather-forecast-mode", "predict",
                    "--weather-forecast-file", str(path),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(len(simulation.call_args.kwargs["weather_archive"].data), 24)
            settings = json.loads((output / "joint_run_settings.json").read_text())
            self.assertEqual(settings["weather_forecast_mode"], "predict")

            missing_output = root / "missing_archive"
            with self.assertRaisesRegex(ValueError, "requires --weather-forecast-file"):
                run_joint_hems.main([
                    "--duration-hours", "1", "--output-directory", str(missing_output),
                    "--weather-forecast-mode", "predict",
                ])
            self.assertFalse(missing_output.exists())

    def test_nested_discovery_provides_sixteen_hour_degree_hour_warmup(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        home = select_home(config, "home_01")
        joint = run_joint_hems.build_controller(
            run_joint_hems.parse_args(["--duration-hours", "1"])
        )
        hems = HEMSController(device_controllers=(joint,))
        simulation = SingleHomeHEMSSimulation(config, home, hems)

        water, hvac, found_joint = _predictive_controllers(hems)
        self.assertIs(water, joint.water_heater_controller)
        self.assertIs(hvac, joint.hvac_controller)
        self.assertIs(found_joint, joint)
        self.assertEqual(simulation._effective_hvac_warmup_duration(), pd.Timedelta(hours=16))
        self.assertEqual(simulation.hvac_warmup_duration, pd.Timedelta(hours=12))

    def test_run_saves_joint_histories_plans_and_measured_combined_cost(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        home = select_home(config, "home_01")
        joint = run_joint_hems.build_controller(
            run_joint_hems.parse_args(["--duration-hours", "1"])
        )
        timestamp = pd.Timestamp("2018-01-01 00:00")
        index = pd.date_range(timestamp, periods=2, freq="min", name="timestamp")
        joint._solve_history.append({"timestamp": timestamp, "successful": True})
        joint._objective_component_history.append({
            "timestamp": timestamp, "electricity_objective_cost": 0.04,
        })
        plan = pd.DataFrame({"total_mean_electric_power_kw": [3.0]}, index=index[:1])
        plan.attrs["solved_at"] = timestamp
        joint._optimization_plans.append(plan)

        controlled = pd.DataFrame({
            "water_heating_power_kw": [1.0, 3.0],
            "hvac_power_kw": [2.0, 4.0],
            "ev_power_kw": [0.0, 0.0],
            "total_electric_power_kw": [3.0, 7.0],
        }, index=index)
        baseline = pd.DataFrame({
            "water_heating_power_kw": [2.0, 2.0],
            "hvac_power_kw": [3.0, 3.0],
            "ev_power_kw": [0.0, 0.0],
            "total_electric_power_kw": [5.0, 5.0],
        }, index=index)
        controls = pd.DataFrame({
            "electricity_price_usd_per_kwh": [0.1, 0.3],
        }, index=index)
        empty = pd.DataFrame(index=index)
        sensor_columns = WaterHeaterSensorLayout().sensor_columns

        def stepwise_result(mapped: pd.DataFrame, upper_temperature: list[float]):
            sensors = pd.DataFrame({
                sensor_columns[0]: upper_temperature,
                sensor_columns[-1]: [43.0, 43.5],
            }, index=index)
            raw = pd.DataFrame({
                "Water Heating Deadband Upper Limit (C)": [50.0, 50.0],
                "Water Heating Electric Power (kW)": mapped.water_heating_power_kw,
            }, index=index)
            training = sensors.assign(
                water_heater_draw_l_per_min=0.0,
                water_heater_setpoint_temperature_c=50.0,
                water_heater_electric_power_kw=mapped.water_heating_power_kw,
                water_heater_mode="Heat Pump",
            )
            return raw, controls, empty, sensors, training, empty

        controlled_run = stepwise_result(controlled, [48.0, 49.0])
        baseline_run = stepwise_result(baseline, [49.0, 49.5])
        figure_functions = (
            "plot_hvac", "plot_water_heater", "plot_ev", "plot_whole_home_comparison",
            "plot_water_heater_power_comparison", "plot_water_usage_profile",
            "plot_hvac_operation", "plot_hvac_weather", "plot_hvac_power_comparison",
            "plot_ev_operation",
        )

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                config, start_time=timestamp.to_pydatetime(),
                duration=dt.timedelta(minutes=2), output_directory=Path(directory),
            )
            simulation = SingleHomeHEMSSimulation(
                config, home, HEMSController(device_controllers=(joint,)),
                save_optimizer_plan_figures=False,
            )
            with ExitStack() as patches, redirect_stdout(io.StringIO()):
                patches.enter_context(patch.object(simulation, "_run_stepwise", side_effect=[controlled_run, baseline_run]))
                patches.enter_context(patch.object(simulation, "_validate_results"))
                patches.enter_context(patch.object(
                    simulation_module, "standardize_results", side_effect=[
                        SimpleNamespace(data=controlled, selected_columns={}),
                        SimpleNamespace(data=baseline, selected_columns={}),
                    ],
                ))
                patches.enter_context(patch.object(
                    simulation_module, "hvac_temperature_limits", return_value=(15.0, 25.0),
                ))
                patches.enter_context(patch.object(
                    simulation_module, "hvac_power_limits", return_value=(0.0, 10.0),
                ))
                for name in figure_functions:
                    patches.enter_context(patch.object(simulation_module, name))
                baseline_plot = patches.enter_context(patch.object(
                    simulation_module, "plot_water_heater_baseline",
                ))
                summary = simulation.run()
                baseline_plot.assert_called_once()
                pd.testing.assert_frame_equal(baseline_plot.call_args.args[0], baseline_run[4])
                self.assertEqual(baseline_plot.call_args.args[1], sensor_columns)
                self.assertEqual(
                    baseline_plot.call_args.kwargs["temperature_limits"],
                    simulation_module.plot_water_heater.call_args.kwargs["temperature_limits"],
                )
                self.assertEqual(
                    baseline_plot.call_args.kwargs["power_limits"],
                    simulation_module.plot_water_heater.call_args.kwargs["power_limits"],
                )

            data = Path(directory) / "data"
            self.assertTrue((data / "joint_optimization_solve_history.csv").is_file())
            self.assertTrue((data / "joint_optimization_objective_component_history.csv").is_file())
            self.assertEqual(len(tuple((data / "joint_optimizer_plans").glob("joint_plan_*.csv"))), 1)
            costs = pd.read_csv(data / "home_01_joint_hpwh_hvac_operating_costs.csv", index_col="scenario")
            self.assertAlmostEqual(costs.loc["With predictive control", "energy_kwh"], 10 / 60)
            self.assertAlmostEqual(costs.loc["With predictive control", "operating_cost_usd"], 2.4 / 60)
            pd.testing.assert_frame_equal(summary.joint_operating_costs, costs)


if __name__ == "__main__":
    unittest.main()

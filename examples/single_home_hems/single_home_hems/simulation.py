"""Stepwise single-home simulation runner for HEMS development."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.aggregation import (
    maximum_power_balance_error,
)
from examples.five_home_block.five_home_block.config import (
    HomeConfig,
    NeighborhoodConfig,
)
from examples.five_home_block.five_home_block.home_factory import create_dwelling
from examples.five_home_block.five_home_block.output_mapping import standardize_results

from .controller import HEMSControllerProtocol
from .controllers.water_heater import WaterHeaterPredictiveController
from .forecast import Forecast
from .plotting import (
    plot_ev,
    plot_hvac,
    plot_water_heater,
    plot_water_heater_baseline,
    plot_water_heater_optimization_plans,
    plot_water_heater_power_comparison,
    plot_water_usage_profile,
    plot_whole_home_comparison,
    water_heater_temperature_limits,
)
from .training import (
    HVACInternalGainRegime,
    WaterHeaterSensorLayout,
    hvac_training_record,
    water_heater_training_record,
)


def select_home(config: NeighborhoodConfig, home_id: str) -> HomeConfig:
    """Select one configured home by ID, with a useful error on failure."""
    homes = {home.home_id: home for home in config.homes}
    try:
        return homes[home_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown home ID {home_id!r}; choose one of {sorted(homes)}"
        ) from exc


def water_heater_operating_summary(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    price: pd.Series,
    timestep_hours: float,
) -> pd.DataFrame:
    """Calculate water-heater electricity use and cost for matched runs."""
    if timestep_hours <= 0:
        raise ValueError("Water-heater summary timestep must be positive")
    if not controlled.index.equals(uncontrolled.index) or not controlled.index.equals(
        price.index
    ):
        raise ValueError("Water-heater power and price timestamps must align")

    rows = []
    for scenario, results in (
        ("Without predictive control", uncontrolled),
        ("With predictive control", controlled),
    ):
        power = results["water_heating_power_kw"]
        rows.append(
            {
                "scenario": scenario,
                "energy_kwh": float(power.sum() * timestep_hours),
                "average_power_kw": float(power.mean()),
                "operating_cost_usd": float((power * price).sum() * timestep_hours),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


@dataclass(frozen=True)
class SingleHomeSimulationSummary:
    """Outputs from a completed single-home HEMS simulation."""

    home_id: str
    standardized_results: pd.DataFrame
    uncontrolled_results: pd.DataFrame
    water_heater_sensors: pd.DataFrame
    uncontrolled_water_heater_sensors: pd.DataFrame
    water_heater_training_data: pd.DataFrame
    uncontrolled_water_heater_training_data: pd.DataFrame
    hvac_training_data: pd.DataFrame
    uncontrolled_hvac_training_data: pd.DataFrame
    water_heater_operating_costs: pd.DataFrame
    controls: pd.DataFrame
    data_directory: Path
    figure_directory: Path
    figure_files: tuple[Path, ...]
    optimizer_plan_directory: Path | None
    optimizer_plan_figure_files: tuple[Path, ...]
    peak_power_kw: float


class SingleHomeHEMSSimulation:
    """Run one selected five-home-block dwelling with stepwise HEMS controls."""

    def __init__(
        self,
        config: NeighborhoodConfig,
        home: HomeConfig,
        controller: HEMSControllerProtocol,
        water_heater_sensor_layout: WaterHeaterSensorLayout | None = None,
        hvac_gain_regime: HVACInternalGainRegime | None = None,
        water_heater_hp_only: bool = False,
        initial_temperature_c: float | None = 21.0,
        water_draw_file: Path | None = None,
        water_draw_start: pd.Timestamp | None = None,
        save_optimizer_plan_figures: bool = True,
    ):
        if initial_temperature_c is not None and (
            isinstance(initial_temperature_c, bool)
            or not isinstance(initial_temperature_c, (int, float))
            or not np.isfinite(initial_temperature_c)
        ):
            raise ValueError("initial_temperature_c must be a finite number or None")
        self.config = config
        self.home = home
        self.controller = controller
        self.water_heater_sensor_layout = (
            water_heater_sensor_layout or WaterHeaterSensorLayout()
        )
        self.hvac_gain_regime = hvac_gain_regime or HVACInternalGainRegime()
        self.water_heater_hp_only = water_heater_hp_only
        # A numeric value bypasses OCHRE's random thermostat-deadband offset.
        # None explicitly opts into the original randomized initialization.
        self.initial_temperature_c = initial_temperature_c
        self.save_optimizer_plan_figures = save_optimizer_plan_figures
        self.water_draw_data = None
        self.water_draw_metadata: dict[str, Any] = {}
        if water_draw_file is not None:
            from .water_forecast import DCHouseWaterData

            if pd.Timedelta(config.time_step) != pd.Timedelta(minutes=1):
                raise ValueError("DCHouse water-draw replay requires a one-minute plant timestep")
            if config.initialization_time is not None and pd.Timedelta(config.initialization_time) != pd.Timedelta(0):
                raise ValueError("DCHouse water-draw replay requires zero initialization time")
            self.water_draw_data = DCHouseWaterData.from_csv(
                water_draw_file, simulation_start=pd.Timestamp(config.start_time),
                source_start=water_draw_start,
            )
            self.water_draw_metadata = self.water_draw_data.metadata
        self.data_directory = config.output_directory / "data"
        self.raw_directory = self.data_directory / "raw"
        self.figure_directory = config.output_directory / "figures"

    def _expected_index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.config.start_time,
            self.config.start_time + self.config.duration,
            freq=self.config.time_step,
            inclusive="left",
            name="timestamp",
        )

    @staticmethod
    def _validate_controls(
        controls: object, timestamp: pd.Timestamp
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(controls, dict):
            raise TypeError(
                f"HEMS controls at {timestamp} must be a dictionary, got "
                f"{type(controls).__name__}"
            )
        invalid = [name for name, values in controls.items() if not isinstance(values, dict)]
        if invalid:
            raise TypeError(
                f"HEMS equipment controls at {timestamp} must be dictionaries: {invalid}"
            )
        water_heater_controls = controls.get("Water Heating", {})
        unsupported = set(water_heater_controls) - {"Setpoint"}
        if unsupported:
            raise ValueError(
                "Water Heating may only be controlled through Setpoint; "
                f"unsupported fields at {timestamp}: {sorted(unsupported)}"
            )
        return controls

    def _run_stepwise(
        self,
        *,
        apply_controls: bool,
        raw_directory: Path,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        water_controller = next((
            device for device in getattr(self.controller, "device_controllers", ())
            if isinstance(device, WaterHeaterPredictiveController)
        ), None)
        scenario_mode = water_controller is not None and water_controller.forecast_type == "scenario"
        if scenario_mode and self.water_draw_data is None:
            raise ValueError("Scenario water forecasts require water_draw_file with historical data")
        # Build enough exogenous schedule for the final complete MPC horizon.
        dwelling_config = self.config
        if scenario_mode:
            dwelling_config = replace(
                self.config, duration=self.config.duration + water_controller.horizon,
            )
        dwelling = create_dwelling(
            dwelling_config,
            self.home,
            raw_directory,
            water_nodes=self.water_heater_sensor_layout.model_nodes,
            water_heater_hp_only=self.water_heater_hp_only,
            initial_temp_setpoint=self.initial_temperature_c,
        )
        water_heater = dwelling.get_equipment_by_end_use("Water Heating")
        heating = dwelling.get_equipment_by_end_use("HVAC Heating")
        cooling = dwelling.get_equipment_by_end_use("HVAC Cooling")
        envelope = dwelling.envelope
        if water_heater.model.n_nodes != self.water_heater_sensor_layout.model_nodes:
            raise RuntimeError(
                f"Expected {self.water_heater_sensor_layout.model_nodes} tank nodes, "
                f"got {water_heater.model.n_nodes}"
            )
        schedule_file = raw_directory / f"{self.home.home_id}_schedule.csv"
        schedule = pd.read_csv(schedule_file, parse_dates=["Time"]).set_index("Time")
        water_scenarios = None
        if self.water_draw_data is not None:
            replay_index = schedule.index
            if scenario_mode:
                replay_index = replay_index[
                    replay_index < self.config.start_time + self.config.duration
                ]
            schedule["Water Heating (L/min)"] = np.nan
            schedule.loc[replay_index, "Water Heating (L/min)"] = self.water_draw_data.flow_at(replay_index)
            # Replay this one series as tempered fixture demand, replacing all
            # three original draw channels rather than adding appliance use.
            schedule["Clothes Washer (L/min)"] = 0.0
            schedule["Dishwasher (L/min)"] = 0.0
            schedule.to_csv(schedule_file, index_label="Time")
            if scenario_mode and apply_controls:
                from .water_forecast import ScenarioWaterForecast

                water_scenarios = ScenarioWaterForecast(
                    self.water_draw_data, method=water_controller.scenario_method,
                )
                self.water_draw_metadata.update(water_scenarios.metadata)
        for end_use, column in (
            ("HVAC Heating", "HVAC Heating Setpoint (C)"),
            ("HVAC Cooling", "HVAC Cooling Setpoint (C)"),
            ("Water Heating", "Water Heating Setpoint (C)"),
        ):
            equipment = dwelling.get_equipment_by_end_use(end_use)
            if equipment is not None and column in equipment.schedule:
                schedule[column] = equipment.schedule.loc[schedule.index, column]
        forecast = Forecast.from_schedule(
            schedule,
            self.controller.electricity_price_for_step,
            water_scenarios=water_scenarios,
        )
        previous_status: Mapping[str, Any] | None = None
        control_records: list[dict[str, Any]] = []
        sensor_records: list[dict[str, Any]] = []
        training_records: list[dict[str, Any]] = []
        hvac_training_records: list[dict[str, Any]] = []
        try:
            for timestamp in dwelling.sim_times:
                if timestamp >= self.config.start_time + self.config.duration:
                    break
                if dwelling.current_time != timestamp:
                    raise RuntimeError(
                        f"OCHRE clock mismatch: expected {timestamp}, got "
                        f"{dwelling.current_time}"
                    )

                # Split OCHRE's normal update into its public phases so the HEMS
                # can observe current exogenous schedule inputs before deciding.
                draw_inputs = None
                if self.water_draw_data is not None:
                    draw_inputs = schedule.loc[timestamp, [
                        "Water Heating (L/min)", "Clothes Washer (L/min)", "Dishwasher (L/min)",
                    ]].to_dict()
                dwelling.update_inputs(schedule_inputs=draw_inputs)
                controls = (
                    self.controller.controls_for_step(
                        pd.Timestamp(timestamp),
                        previous_status,
                        forecast.data.loc[pd.Timestamp(timestamp)].to_dict(),
                        forecast,
                    )
                    if apply_controls
                    else {}
                )
                controls = self._validate_controls(controls, pd.Timestamp(timestamp))
                initial_water_heater_states = water_heater.model.states.copy()
                initial_indoor_temperature = envelope.indoor_zone.temperature
                dwelling.update_model(control_signal=controls)
                sensor_records.append(
                    {
                        "timestamp": timestamp,
                        **self.water_heater_sensor_layout.measure(
                            water_heater.model.states
                        ),
                    }
                )
                training_records.append(
                    {
                        "timestamp": timestamp,
                        **water_heater_training_record(
                            water_heater,
                            self.water_heater_sensor_layout,
                            initial_water_heater_states,
                        ),
                    }
                )
                hvac_training_records.append(
                    {
                        "timestamp": timestamp,
                        **hvac_training_record(
                            pd.Timestamp(timestamp),
                            envelope,
                            heating,
                            cooling,
                            initial_indoor_temperature,
                            self.hvac_gain_regime,
                            forecast.data.loc[pd.Timestamp(timestamp)].to_dict(),
                        ),
                    }
                )
                previous_status = dwelling.update_results()
                previous_status.update(
                    self.water_heater_sensor_layout.measure(water_heater.model.states)
                )
                if hasattr(water_heater, "hp_capacity_nominal"):
                    previous_status["Water Heating Nominal Heat Pump Capacity (W)"] = (
                        float(water_heater.hp_capacity_nominal)
                    )
                control_records.append(
                    {
                        "timestamp": timestamp,
                        "electricity_price_usd_per_kwh": (
                            forecast.price_at(pd.Timestamp(timestamp))
                        ),
                        "control_signal_json": json.dumps(controls, sort_keys=True),
                    }
                )
        except Exception:
            dwelling.finalize(failed=True)
            raise

        raw, _, _ = dwelling.finalize()
        if raw is None:
            raise RuntimeError(f"{self.home.home_id}: OCHRE returned no results")
        controls = pd.DataFrame(control_records).set_index("timestamp")
        controls.index = pd.DatetimeIndex(controls.index, name="timestamp")
        training_data = pd.DataFrame(training_records).set_index("timestamp")
        training_data.index = pd.DatetimeIndex(
            training_data.index,
            name="timestamp",
        )
        sensors = pd.DataFrame(sensor_records).set_index("timestamp")
        sensors.index = pd.DatetimeIndex(sensors.index, name="timestamp")
        hvac_training_data = pd.DataFrame(hvac_training_records).set_index(
            "timestamp"
        )
        hvac_training_data.index = pd.DatetimeIndex(
            hvac_training_data.index,
            name="timestamp",
        )
        return raw, controls, schedule, sensors, training_data, hvac_training_data

    def _validate_results(self, raw: pd.DataFrame, standardized: pd.DataFrame) -> None:
        expected = self._expected_index()
        if not standardized.index.equals(expected):
            raise RuntimeError(
                f"{self.home.home_id}: expected {len(expected)} aligned timesteps, "
                f"got {len(standardized)}"
            )
        if not np.isfinite(standardized.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{self.home.home_id}: results contain nonfinite values")
        balance_error = maximum_power_balance_error(standardized)
        if balance_error > self.config.power_balance_tolerance_kw:
            raise RuntimeError(
                f"{self.home.home_id}: maximum power-balance error is "
                f"{balance_error} kW"
            )
        gas_column = "Total Gas Power (therms/hour)"
        if gas_column not in raw or raw[gas_column].abs().max() > 1e-9:
            raise RuntimeError(
                f"{self.home.home_id}: expected zero gas use in the all-electric home"
            )

    def run(self) -> SingleHomeSimulationSummary:
        """Execute the selected home and save data, controls, and figures."""
        simulation_start = time.perf_counter()
        for directory in (self.data_directory, self.raw_directory, self.figure_directory):
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Running {self.home.home_id} with {type(self.controller).__name__}")
        print(f"  {self.home.description}")
        (
            raw,
            controls,
            schedule,
            sensors,
            training_data,
            hvac_training_data,
        ) = self._run_stepwise(
            apply_controls=True,
            raw_directory=self.raw_directory,
        )
        if self.water_draw_metadata:
            (self.data_directory / "water_draw_metadata.json").write_text(
                json.dumps(self.water_draw_metadata, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        mapped = standardize_results(
            raw,
            self.home.home_id,
            numerical_tolerance_kw=self.config.numerical_tolerance_kw,
        )
        self._validate_results(raw, mapped.data)

        print(f"Running matched uncontrolled baseline for {self.home.home_id}")
        (
            uncontrolled_raw,
            _,
            _,
            uncontrolled_sensors,
            uncontrolled_training_data,
            uncontrolled_hvac_training_data,
        ) = self._run_stepwise(
            apply_controls=False,
            raw_directory=self.raw_directory / "uncontrolled",
        )
        uncontrolled_mapped = standardize_results(
            uncontrolled_raw,
            self.home.home_id,
            numerical_tolerance_kw=self.config.numerical_tolerance_kw,
        )
        self._validate_results(uncontrolled_raw, uncontrolled_mapped.data)

        results_file = self.data_directory / f"{self.home.home_id}_standardized.csv"
        uncontrolled_file = (
            self.data_directory / f"{self.home.home_id}_uncontrolled_standardized.csv"
        )
        controls_file = self.data_directory / f"{self.home.home_id}_hems_controls.csv"
        sensors_file = (
            self.data_directory / f"{self.home.home_id}_water_heater_sensors.csv"
        )
        uncontrolled_sensors_file = self.data_directory / (
            f"{self.home.home_id}_uncontrolled_water_heater_sensors.csv"
        )
        training_file = self.data_directory / (
            f"{self.home.home_id}_water_heater_training_data.csv"
        )
        uncontrolled_training_file = self.data_directory / (
            f"{self.home.home_id}_uncontrolled_water_heater_training_data.csv"
        )
        hvac_training_file = self.data_directory / (
            f"{self.home.home_id}_hvac_training_data.csv"
        )
        uncontrolled_hvac_training_file = self.data_directory / (
            f"{self.home.home_id}_uncontrolled_hvac_training_data.csv"
        )
        sensor_metadata_file = self.data_directory / "water_heater_sensor_layout.json"
        mapping_file = self.data_directory / "output_mapping.json"
        mapped.data.to_csv(results_file)
        uncontrolled_mapped.data.to_csv(uncontrolled_file)
        controls.to_csv(controls_file)
        sensors.to_csv(sensors_file)
        uncontrolled_sensors.to_csv(uncontrolled_sensors_file)
        training_data.to_csv(training_file)
        uncontrolled_training_data.to_csv(uncontrolled_training_file)
        hvac_training_data.to_csv(hvac_training_file)
        uncontrolled_hvac_training_data.to_csv(uncontrolled_hvac_training_file)
        with sensor_metadata_file.open("w", encoding="utf-8") as stream:
            json.dump(self.water_heater_sensor_layout.metadata(), stream, indent=2)
        with mapping_file.open("w", encoding="utf-8") as stream:
            json.dump(mapped.selected_columns, stream, indent=2)

        price = controls["electricity_price_usd_per_kwh"]
        water_heater_costs = water_heater_operating_summary(
            mapped.data,
            uncontrolled_mapped.data,
            price,
            self.config.time_step.total_seconds() / 3600,
        )
        water_heater_costs_file = (
            self.data_directory / f"{self.home.home_id}_water_heater_operating_costs.csv"
        )
        water_heater_costs.to_csv(water_heater_costs_file)
        figure_files = (
            self.figure_directory / f"{self.home.home_id}_hvac.png",
            self.figure_directory / f"{self.home.home_id}_water_heater.png",
            self.figure_directory / f"{self.home.home_id}_ev.png",
            self.figure_directory / f"{self.home.home_id}_whole_home_comparison.png",
            self.figure_directory / f"{self.home.home_id}_water_heater_power_comparison.png",
            self.figure_directory / f"{self.home.home_id}_water_usage_profile.png",
        )
        device_controllers = getattr(self.controller, "device_controllers", ())
        water_heater_controller = next(
            (
                device
                for device in device_controllers
                if isinstance(device, WaterHeaterPredictiveController)
            ),
            None,
        )
        water_heater_only = (
            len(device_controllers) == 1
            and isinstance(device_controllers[0], WaterHeaterPredictiveController)
        )
        if water_heater_only:
            figure_files += (
                self.figure_directory / f"{self.home.home_id}_water_heater_baseline.png",
            )
        temperature_limits = None
        if water_heater_only:
            baseline_temperature_columns = [
                *self.water_heater_sensor_layout.sensor_columns,
                "water_heater_setpoint_temperature_c",
            ]
            temperature_limits = water_heater_temperature_limits(
                sensors,
                raw.loc[
                    mapped.data.index,
                    "Water Heating Deadband Upper Limit (C)",
                ],
                uncontrolled_training_data[baseline_temperature_columns],
            )
        plot_hvac(
            mapped.data,
            schedule,
            price,
            self.home.home_id,
            figure_files[0],
            self.config.plot_dpi,
        )
        plot_water_heater(
            mapped.data,
            raw,
            sensors,
            price,
            self.home.home_id,
            figure_files[1],
            self.config.plot_dpi,
            water_usage=training_data["water_heater_draw_l_per_min"],
            temperature_limits=temperature_limits,
        )
        plot_ev(
            mapped.data,
            raw,
            price,
            self.home.home_id,
            figure_files[2],
            self.config.plot_dpi,
        )
        plot_whole_home_comparison(
            mapped.data,
            uncontrolled_mapped.data,
            price,
            self.home.home_id,
            figure_files[3],
            self.config.plot_dpi,
        )
        plot_water_heater_power_comparison(
            mapped.data,
            uncontrolled_mapped.data,
            price,
            self.home.home_id,
            figure_files[4],
            self.config.plot_dpi,
            operating_summary=water_heater_costs,
        )
        plot_water_usage_profile(
            training_data,
            uncontrolled_training_data,
            self.home.home_id,
            figure_files[5],
            self.config.plot_dpi,
        )
        if water_heater_only:
            plot_water_heater_baseline(
                uncontrolled_training_data,
                self.water_heater_sensor_layout.sensor_columns,
                self.home.home_id,
                figure_files[-1],
                self.config.plot_dpi,
                water_usage=uncontrolled_training_data[
                    "water_heater_draw_l_per_min"
                ],
                temperature_limits=temperature_limits,
            )
        optimizer_plan_directory = None
        optimizer_plan_figure_files: tuple[Path, ...] = ()
        if water_heater_controller is not None:
            for name in ("solve_history", "objective_component_history", "scenario_diagnostics_history"):
                history = getattr(water_heater_controller, name, None)
                if isinstance(history, pd.DataFrame) and not history.empty:
                    history.to_csv(self.data_directory / f"optimization_{name}.csv", index=False)
        if water_heater_controller is not None and self.save_optimizer_plan_figures:
            optimizer_plan_directory = (
                self.figure_directory / "water_heater_optimizer_plans"
            )
            optimizer_plan_figure_files = plot_water_heater_optimization_plans(
                water_heater_controller.optimization_plans,
                optimizer_plan_directory,
                self.config.plot_dpi,
            )

        summary = SingleHomeSimulationSummary(
            home_id=self.home.home_id,
            standardized_results=mapped.data,
            uncontrolled_results=uncontrolled_mapped.data,
            water_heater_sensors=sensors,
            uncontrolled_water_heater_sensors=uncontrolled_sensors,
            water_heater_training_data=training_data,
            uncontrolled_water_heater_training_data=uncontrolled_training_data,
            hvac_training_data=hvac_training_data,
            uncontrolled_hvac_training_data=uncontrolled_hvac_training_data,
            water_heater_operating_costs=water_heater_costs,
            controls=controls,
            data_directory=self.data_directory,
            figure_directory=self.figure_directory,
            figure_files=figure_files,
            optimizer_plan_directory=optimizer_plan_directory,
            optimizer_plan_figure_files=optimizer_plan_figure_files,
            peak_power_kw=float(mapped.data["total_electric_power_kw"].max()),
        )
        print("\nSingle-home HEMS simulation complete")
        print(f"  Home: {summary.home_id}")
        print(f"  Timesteps: {len(summary.standardized_results)}")
        for device in getattr(self.controller, "device_controllers", ()):
            for kind, durations in getattr(device, "solve_durations_s", {}).items():
                if durations:
                    print(
                        f"  {kind} solve time: {np.mean(durations):.3f} s average "
                        f"over {len(durations)} calls"
                    )
        print(f"  Total simulation time: {time.perf_counter() - simulation_start:.2f} s")
        print(f"  Peak power: {summary.peak_power_kw:.2f} kW")
        print("  Water-heater operation:")
        for scenario, values in summary.water_heater_operating_costs.iterrows():
            print(
                f"    {scenario}: {values['energy_kwh']:.2f} kWh, "
                f"{values['average_power_kw']:.3f} kW average, "
                f"${values['operating_cost_usd']:.2f}"
            )
        print(f"  Water-heater operating costs: {water_heater_costs_file}")
        print(f"  Data and HEMS controls: {summary.data_directory}")
        print("  Figures:")
        for figure_file in summary.figure_files:
            print(f"    {figure_file}")
        if summary.optimizer_plan_directory is not None:
            print(
                "  Water-heater optimizer plans: "
                f"{summary.optimizer_plan_directory} "
                f"({len(summary.optimizer_plan_figure_files)} figures)"
            )
        return summary

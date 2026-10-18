"""Stepwise single-home simulation runner for HEMS development."""

from __future__ import annotations

import json
from dataclasses import replace
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

from .controller import HEMSControllerProtocol
from .controllers.water_heater import WaterHeaterPredictiveController
from .forecast import Forecast
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
        ev_mobility_schedules: dict | None = None,
        ev_backed_load: pd.Series | None = None,
        unified_context: dict | None = None,
        water_draw_anchor_time: pd.Timestamp | None = None,
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
        self.ev_mobility_schedules = ev_mobility_schedules
        self.ev_backed_load = ev_backed_load
        self.unified_context = unified_context
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
        self.water_draw_file = water_draw_file
        self.water_draw_start = water_draw_start
        self.water_draw_metadata: dict[str, Any] = {}
        if water_draw_file is not None:
            from .water_forecast import DCHouseWaterData

            if pd.Timedelta(config.time_step) != pd.Timedelta(minutes=1):
                raise ValueError("DCHouse water-draw replay requires a one-minute plant timestep")
            if config.initialization_time is not None and pd.Timedelta(config.initialization_time) != pd.Timedelta(0):
                raise ValueError("DCHouse water-draw replay requires zero initialization time")
            self.water_draw_data = DCHouseWaterData.from_csv(
                water_draw_file, simulation_start=pd.Timestamp(water_draw_anchor_time or config.start_time),
                source_start=water_draw_start,
            )
            # Keep the evaluation's source mapping while pre-rolling earlier.
            # Forecast training must remain causal at the actual simulation start.
            self.water_draw_data = replace(self.water_draw_data,
                source_start=self.water_draw_data.source_time(pd.Timestamp(config.start_time)),
                simulation_start=pd.Timestamp(config.start_time))
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
        unsupported = set(water_heater_controls) - {"Setpoint", "HP Duty Cycle", "ER Duty Cycle"}
        if unsupported:
            raise ValueError(
                "Water Heating supports Setpoint and binary HP/ER duty projection only; "
                f"unsupported fields at {timestamp}: {sorted(unsupported)}"
            )
        if "HP Duty Cycle" in water_heater_controls or "ER Duty Cycle" in water_heater_controls:
            if water_heater_controls.get("HP Duty Cycle") not in (0, 1) or water_heater_controls.get("ER Duty Cycle") != 0:
                raise ValueError("HPWH projection requires HP Duty Cycle 0/1 and ER Duty Cycle 0")
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
        from .controllers.ev import EVMPCController
        ev_horizon = max((device.horizon for device in getattr(self.controller, "device_controllers", ())
                          if isinstance(device, EVMPCController)), default=pd.Timedelta(0))
        if ev_horizon:
            dwelling_config = replace(dwelling_config,
                duration=max(dwelling_config.duration, self.config.duration + ev_horizon))
        if self.unified_context:
            dwelling_config = replace(dwelling_config, duration=max(dwelling_config.duration,
                self.config.duration + self.unified_context["forecast_extension"]))
        dwelling = create_dwelling(
            dwelling_config,
            self.home,
            raw_directory,
            water_nodes=self.water_heater_sensor_layout.model_nodes,
            water_heater_hp_only=self.water_heater_hp_only,
            initial_temp_setpoint=self.initial_temperature_c,
            ev_mobility_schedules=self.ev_mobility_schedules,
        )
        water_heater = dwelling.get_equipment_by_end_use("Water Heating")
        heating = dwelling.get_equipment_by_end_use("HVAC Heating")
        cooling = dwelling.get_equipment_by_end_use("HVAC Cooling")
        envelope = dwelling.envelope
        if self.unified_context:
            from .backed_loads import attach_actual_load_cap
            attach_actual_load_cap(dwelling, self.unified_context["backed_loads"])
        if water_heater.model.n_nodes != self.water_heater_sensor_layout.model_nodes:
            raise RuntimeError(
                f"Expected {self.water_heater_sensor_layout.model_nodes} tank nodes, "
                f"got {water_heater.model.n_nodes}"
            )
        schedule_file = raw_directory / f"{self.home.home_id}_schedule.csv"
        if water_controller is not None and (water_controller.enforce_cycle_limit or water_controller.enforce_dwell_constraints):
            for equipment in dwelling.equipment.values():
                if equipment.end_use == "Water Heating":
                    # Projected binary commands are refreshed at each plant step.
                    equipment.ext_time_res = self.config.time_step
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
            ev_schedules=self.ev_mobility_schedules,
            backed_load_kw=self.ev_backed_load,
        )
        if self.unified_context:
            forecast.providers.update(self.unified_context.get("providers", {}))
            if forecast.backed_load_kw is None:
                # Twin/baseline EVs cannot discharge. Zero is decision-irrelevant,
                # avoiding a circular dependency on the twin being constructed.
                forecast = replace(forecast, backed_load_kw=pd.Series(0.0, index=schedule.index))
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
                if self.unified_context and previous_status is None:
                    previous_status = {
                        **self.water_heater_sensor_layout.measure(water_heater.model.states),
                        "Temperature - Indoor (C)": envelope.indoor_zone.temperature,
                        "Water Heating Mode": water_heater.mode,
                        "Water Heating Delivered (W)": 0.0,
                    }
                    if hasattr(water_heater, "hp_capacity_nominal"):
                        previous_status["Water Heating Nominal Heat Pump Capacity (W)"] = water_heater.hp_capacity_nominal
                if self.ev_mobility_schedules:
                    previous_status = dict(previous_status or {})
                    for ev in self.home.evs:
                        if ev.hems is not None:
                            equipment = dwelling.equipment[f"EV_{ev.vehicle_id}"]
                            equipment.prepare_boundary(timestamp)
                            previous_status[f"EV_{ev.vehicle_id} SOC (-)"] = equipment.soc
                            previous_status[f"EV_{ev.vehicle_id} SOH (-)"] = equipment.soh
                            previous_status[f"EV_{ev.vehicle_id} Capacity (kWh)"] = equipment.capacity
                            previous_status[f"EV_{ev.vehicle_id} Parked"] = equipment.in_event
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
        battery_states = {name: equipment.battery_checkpoint()
                          for name, equipment in dwelling.equipment.items()
                          if hasattr(equipment, "battery_checkpoint")}
        if battery_states:
            # Battery-only checkpoint, not a complete dwelling/controller restart.
            (self.data_directory / "ev_battery_checkpoints.json").write_text(json.dumps(
                {"timestamp": str(dwelling.current_time), "batteries": battery_states}, indent=2))
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

    def run(self):
        """Adapt legacy callers to the same three-case pipeline as the unified CLI."""
        from .legacy_adapter import run_legacy
        return run_legacy(self)

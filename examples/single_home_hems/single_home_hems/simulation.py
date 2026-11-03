"""Stepwise single-home simulation runner for HEMS development."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.aggregation import (
    maximum_power_balance_error,
)
from examples.five_home_block.five_home_block.config import (
    HomeConfig,
    NeighborhoodConfig,
)
from examples.five_home_block.five_home_block.home_factory import (
    HEATING_SEASON_COOLING_SETPOINT_C,
    create_dwelling,
)
from examples.five_home_block.five_home_block.output_mapping import standardize_results

from .controller import HEMSControllerProtocol
from .ev_panel_guard import EVPanelGuard
from .controllers.hvac import (
    HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY,
    HVACPredictiveController,
)
from .controllers.water_heater import WaterHeaterPredictiveController
from .forecast import Forecast, PRICE_COLUMN
from .home_hub_observation import project_ev_only_hub_observation
from .hvac_plotting import (
    hvac_power_limits,
    hvac_temperature_limits,
    plot_hvac_operation,
    plot_hvac_optimization_plans,
    plot_hvac_power_comparison,
    plot_hvac_weather,
)
from .plotting import (
    plot_ev,
    plot_ev_operation,
    plot_hvac,
    plot_water_heater,
    plot_water_heater_baseline,
    plot_water_heater_optimization_plans,
    plot_water_heater_power_comparison,
    plot_water_usage_profile,
    plot_whole_home_comparison,
    plot_whole_home_operation,
    water_heater_temperature_limits,
)
from .training import (
    HEATING_SETPOINT_PERIODS,
    HVACInternalGainRegime,
    WaterHeaterSensorLayout,
    heating_setpoint_c,
    heating_setpoints_c,
    hvac_training_record,
    water_heater_training_record,
)

if TYPE_CHECKING:
    from .rest_of_home_forecast import RestOfHomeHybridForecast
    from .frozen_split_hub_forecast import FrozenSplitHubForecast
    from .weather_forecast import ArchivedWeatherForecast


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


def hvac_operating_summary(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    price: pd.Series,
    timestep_hours: float,
) -> pd.DataFrame:
    """Calculate HVAC electricity use and cost for the matched runs."""
    if timestep_hours <= 0:
        raise ValueError("HVAC summary timestep must be positive")
    if not controlled.index.equals(uncontrolled.index) or not controlled.index.equals(price.index):
        raise ValueError("HVAC power and price timestamps must align")
    rows = []
    for scenario, results in (
        ("Without predictive control", uncontrolled),
        ("With predictive control", controlled),
    ):
        power = results["hvac_power_kw"]
        rows.append({
            "scenario": scenario,
            "energy_kwh": float(power.sum() * timestep_hours),
            "average_power_kw": float(power.mean()),
            "operating_cost_usd": float((power * price).sum() * timestep_hours),
        })
    return pd.DataFrame(rows).set_index("scenario")


def solve_time_summary(durations_s: list[float]) -> dict[str, float | int]:
    """Summarize all solve attempts, including the slowest five percent."""
    durations = np.asarray(durations_s, dtype=float)
    if durations.ndim != 1 or durations.size == 0 or not np.isfinite(durations).all():
        raise ValueError("Solve-time summary requires finite solve durations")
    tail_count = max(1, int(np.ceil(0.05 * durations.size)))
    return {
        "solve_count": int(durations.size),
        "average_s": float(durations.mean()),
        "p95_s": float(np.percentile(durations, 95)),
        "slowest_5_percent_count": tail_count,
        "slowest_5_percent_average_s": float(np.sort(durations)[-tail_count:].mean()),
    }


def hvac_optimization_reporting(
    optimization_kind: str,
    solve_durations_s: Mapping[str, list[float]],
) -> tuple[str, dict[str, float | int] | None]:
    """Select the run label and solve statistics for the active HVAC formulation."""
    try:
        optimization_type = {"convex": "cvx", "mixed integer": "mi"}[
            optimization_kind
        ]
    except KeyError as exc:
        raise ValueError(f"Unsupported HVAC optimization kind: {optimization_kind}") from exc
    durations = solve_durations_s.get(optimization_kind, [])
    return optimization_type, solve_time_summary(durations) if durations else None


def _predictive_controllers(
    controller: HEMSControllerProtocol,
) -> tuple[
    WaterHeaterPredictiveController | None,
    HVACPredictiveController | None,
    Any | None,
]:
    """Find standalone predictors or the two predictors owned by a joint device."""
    devices = getattr(controller, "device_controllers", ())
    water = next(
        (device for device in devices if isinstance(device, WaterHeaterPredictiveController)),
        None,
    )
    hvac = next(
        (device for device in devices if isinstance(device, HVACPredictiveController)),
        None,
    )
    joint = next(
        (
            device for device in devices
            if isinstance(getattr(device, "water_heater_controller", None), WaterHeaterPredictiveController)
            and isinstance(getattr(device, "hvac_controller", None), HVACPredictiveController)
        ),
        None,
    )
    if joint is not None:
        if water is not None or hvac is not None:
            raise ValueError("A joint predictor cannot share its devices with standalone predictors")
        water = joint.water_heater_controller
        hvac = joint.hvac_controller
    return water, hvac, joint


def _ev_predictive_controllers(controller: HEMSControllerProtocol) -> tuple[Any, ...]:
    """Find standalone EV MPCs and EV MPCs owned by a joint device."""
    from .controllers.ev import EVMPCController

    devices = getattr(controller, "device_controllers", ())
    ev_controllers = [
        device for device in devices if isinstance(device, EVMPCController)
    ]
    for device in devices:
        nested = getattr(device, "ev_controller", None)
        if isinstance(nested, EVMPCController) and not any(
            nested is existing for existing in ev_controllers
        ):
            ev_controllers.append(nested)
    return tuple(ev_controllers)


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
    hvac_operating_costs: pd.DataFrame
    controls: pd.DataFrame
    data_directory: Path
    figure_directory: Path
    figure_files: tuple[Path, ...]
    optimizer_plan_directory: Path | None
    optimizer_plan_figure_files: tuple[Path, ...]
    hvac_optimizer_plan_directory: Path | None
    hvac_optimizer_plan_figure_files: tuple[Path, ...]
    peak_power_kw: float
    joint_operating_costs: pd.DataFrame | None = None


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
        rest_of_home_forecast: RestOfHomeHybridForecast | None = None,
        hub_other_forecast: FrozenSplitHubForecast | None = None,
        outside_hub_forecast: FrozenSplitHubForecast | None = None,
        save_optimizer_plan_figures: bool = True,
        hvac_warmup_duration: pd.Timedelta | None = None,
        ev_mobility_schedules: dict | None = None,
        ev_backed_load: pd.Series | None = None,
        unified_context: dict | None = None,
        water_draw_anchor_time: pd.Timestamp | None = None,
        weather_archive: ArchivedWeatherForecast | None = None,
        hub_split_replay_kw: pd.DataFrame | None = None,
        resistance_outside_hub: bool = False,
        defrost_er_pause_requires_compressor: bool = True,
        defrost_finish_active_cycle: bool = True,
        ev_panel_guard_limit_kw: float | None = None,
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
        self.hub_split_replay_kw = hub_split_replay_kw
        self.resistance_outside_hub = resistance_outside_hub
        self.defrost_er_pause_requires_compressor = bool(defrost_er_pause_requires_compressor)
        self.defrost_finish_active_cycle = bool(defrost_finish_active_cycle)
        if ev_panel_guard_limit_kw is not None and pd.Timedelta(config.time_step) != pd.Timedelta(minutes=1):
            raise ValueError("EV panel guard requires a one-minute plant timestep")
        self.ev_panel_guard_limit_kw = (
            None if ev_panel_guard_limit_kw is None else EVPanelGuard(ev_panel_guard_limit_kw).limit_kw
        )
        if any(ev.hems is not None for ev in home.evs):
            if pd.Timedelta(config.time_step) != pd.Timedelta(minutes=1):
                raise ValueError("EV comparisons require a one-minute plant timestep")
            start_clock = pd.Timestamp(config.start_time)
            if start_clock.tzinfo is not None:
                start_clock = start_clock.tz_convert("UTC")
            if start_clock != start_clock.floor("min"):
                raise ValueError("EV simulation start must align with the minute grid")
        self.water_heater_sensor_layout = (
            water_heater_sensor_layout or WaterHeaterSensorLayout()
        )
        self.hvac_gain_regime = hvac_gain_regime or HVACInternalGainRegime()
        self.water_heater_hp_only = water_heater_hp_only
        self.rest_of_home_forecast = rest_of_home_forecast
        self.hub_other_forecast = hub_other_forecast
        self.outside_hub_forecast = outside_hub_forecast
        self.weather_archive = weather_archive
        if weather_archive is not None:
            _, hvac_controller, _ = _predictive_controllers(controller)
            if hvac_controller is None:
                raise ValueError("Archived weather requires an HVAC predictive controller")
            start_time = pd.Timestamp(config.start_time)
            weather_archive.validate_coverage(
                start_time,
                start_time + pd.Timedelta(config.duration),
                hvac_controller.horizon,
                hvac_controller.model_timestep,
            )
        # A numeric value bypasses OCHRE's random thermostat-deadband offset.
        # None explicitly opts into the original randomized initialization.
        self.initial_temperature_c = initial_temperature_c
        self.save_optimizer_plan_figures = save_optimizer_plan_figures
        self.hvac_warmup_duration = (
            pd.Timedelta(hours=12)
            if hvac_warmup_duration is None
            else pd.Timedelta(hvac_warmup_duration)
        )
        if (
            pd.isna(self.hvac_warmup_duration)
            or self.hvac_warmup_duration < pd.Timedelta(0)
        ):
            raise ValueError("HVAC warm-up duration must be nonnegative")
        if self.hvac_warmup_duration % pd.Timedelta(config.time_step):
            raise ValueError("HVAC warm-up must contain whole plant timesteps")
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
                water_draw_file, simulation_start=pd.Timestamp(
                    config.start_time if water_draw_anchor_time is None else water_draw_anchor_time
                ),
                source_start=water_draw_start,
            )
            if water_draw_anchor_time is not None:
                # An opt-in EV comparison extends its evaluated window backward.
                # Retain the original draw mapping in all three cases.
                self.water_draw_data = replace(
                    self.water_draw_data,
                    source_start=self.water_draw_data.source_time(pd.Timestamp(config.start_time)),
                    simulation_start=pd.Timestamp(config.start_time),
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

    def _effective_hvac_warmup_duration(self) -> pd.Timedelta:
        """Provide each active rolling budget its complete measured lookback."""
        water, hvac, joint = _predictive_controllers(self.controller)
        if hvac is None:
            warmup = self.hvac_warmup_duration if self.hub_split_replay_kw is not None else pd.Timedelta(0)
        elif joint is not None and hvac.enforce_degree_hour_budget:
            warmup = max(self.hvac_warmup_duration, hvac._comfort_history_window)
        else:
            warmup = self.hvac_warmup_duration
        if water is not None and getattr(water, "rolling_start_limit_24h", None) is not None:
            warmup = max(warmup, pd.Timedelta(hours=24))
        return warmup

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

    def _seasonal_hvac_controls(
        self, controls: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Prevent explicit cooling commands from overriding the seasonal setpoint."""
        if (
            not self.config.heating_season
            or "Setpoint" not in controls.get("HVAC Cooling", {})
        ):
            return controls
        seasonal_controls = {name: values.copy() for name, values in controls.items()}
        seasonal_controls["HVAC Cooling"]["Setpoint"] = (
            HEATING_SEASON_COOLING_SETPOINT_C
        )
        return seasonal_controls

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
        water_controller, hvac_controller, joint_controller = _predictive_controllers(self.controller)
        from .controllers.ev_home_hub import EVHomeHubController

        devices = tuple(getattr(self.controller, "device_controllers", ()))
        ev_only_hub_devices = tuple(
            device for device in devices if isinstance(device, EVHomeHubController)
        )
        ev_only_hub_controller = ev_only_hub_devices[0] if ev_only_hub_devices else None
        if ev_only_hub_controller is not None and (
            len(devices) != 1 or self.hub_split_replay_kw is None or len(self.home.evs) != 1
        ):
            raise ValueError(
                "The EV-only Home Hub controller requires one EV, no other device "
                "controller, and a split Hub plant replay"
            )
        requested_start = pd.Timestamp(self.config.start_time)
        requested_end = requested_start + self.config.duration
        warmup = self._effective_hvac_warmup_duration()
        scenario_mode = water_controller is not None and water_controller.forecast_type == "scenario"
        if scenario_mode and self.water_draw_data is None:
            raise ValueError("Scenario water forecasts require water_draw_file with historical data")
        # Build enough exogenous schedule for the final complete MPC horizon.
        forecast_horizon = max(
            water_controller.horizon if scenario_mode else pd.Timedelta(0),
            hvac_controller.horizon if hvac_controller is not None else pd.Timedelta(0),
        )
        ev_horizon = max(
            (device.horizon for device in _ev_predictive_controllers(self.controller)),
            default=pd.Timedelta(0),
        )
        forecast_horizon = max(forecast_horizon, ev_horizon)
        if self.unified_context:
            forecast_horizon = max(
                forecast_horizon, self.unified_context.get("forecast_extension", pd.Timedelta(0))
            )
        # A one-time plan already spans the complete evaluation. It never
        # requests a new horizon at later minutes, so no second full-period
        # schedule (or unavailable future tariff/weather) is needed.
        if ((ev_only_hub_controller is not None and getattr(ev_only_hub_controller, "one_shot", False))
                or (self.unified_context or {}).get("one_shot", False)):
            forecast_horizon = max(pd.Timedelta(0), forecast_horizon - self.config.duration)
        dwelling_config = replace(
            self.config,
            start_time=requested_start - warmup,
            duration=self.config.duration + warmup + forecast_horizon,
        )
        if self.hub_split_replay_kw is not None:
            replay_times = pd.date_range(
                requested_start - warmup, requested_end,
                freq=self.config.time_step, inclusive="left",
            )
            missing = replay_times.difference(self.hub_split_replay_kw.index)
            if len(missing):
                raise ValueError(
                    f"Home Hub split replay misses {len(missing)} warm-up/evaluation minute(s); "
                    f"first missing interval start {missing[0]}"
                )
        dwelling = create_dwelling(
            dwelling_config,
            self.home,
            raw_directory,
            water_nodes=self.water_heater_sensor_layout.model_nodes,
            water_heater_hp_only=self.water_heater_hp_only,
            initial_temp_setpoint=self.initial_temperature_c,
            **({"ev_mobility_schedules": self.ev_mobility_schedules}
               if self.ev_mobility_schedules is not None else {}),
            **({"hub_split_replay_kw": self.hub_split_replay_kw}
               if self.hub_split_replay_kw is not None else {}),
            **({
                "rest_of_home_simulation_start": requested_start,
                "rest_of_home_duration": self.config.duration,
            } if self.home.rest_of_home.source == "dc_house_test" else {}),
        )
        water_heater = dwelling.get_equipment_by_end_use("Water Heating")
        heating = dwelling.get_equipment_by_end_use("HVAC Heating")
        cooling = dwelling.get_equipment_by_end_use("HVAC Cooling")
        # Apply the same true-plant policy before warm-up in controlled and
        # native runs. Furnaces without defrost do not have these attributes.
        if heating is not None:
            for name in ("defrost_er_pause_requires_compressor", "defrost_finish_active_cycle"):
                if hasattr(heating, name):
                    setattr(heating, name, getattr(self, name))
        if apply_controls and hvac_controller is not None:
            hvac_controller.bind_heating_equipment(heating)
        envelope = dwelling.envelope
        ev_equipment = {
            name: equipment for name, equipment in dwelling.equipment.items()
            if hasattr(equipment, "battery_checkpoint") and hasattr(equipment, "prepare_boundary")
        }
        panel_guard = (
            EVPanelGuard(self.ev_panel_guard_limit_kw)
            if apply_controls and self.ev_panel_guard_limit_kw is not None else None
        )
        if panel_guard is not None and not ev_equipment:
            raise ValueError("EV panel guard requires EV equipment")
        if ev_equipment:
            from .backed_loads import (
                DEFAULT_BACKED_LOADS, HOME_HUB_BACKED_LOADS, attach_actual_load_cap,
            )

            backed_loads = (self.unified_context or {}).get(
                "backed_loads",
                HOME_HUB_BACKED_LOADS if self.hub_split_replay_kw is not None else DEFAULT_BACKED_LOADS,
            )
            if self.hub_split_replay_kw is not None and tuple(backed_loads) != HOME_HUB_BACKED_LOADS:
                raise ValueError(
                    "Home Hub split V2H requires exactly heat pump compressor/fan, "
                    "water heater, and Hub-other in backed_loads"
                )

            attach_actual_load_cap(
                dwelling, backed_loads
            )
            if self.ev_mobility_schedules is None:
                self.ev_mobility_schedules = {
                    equipment.ev_config.vehicle_id: equipment.mobility
                    for equipment in ev_equipment.values()
                }
            for equipment in ev_equipment.values():
                equipment.activation_start = requested_start
        if water_heater.model.n_nodes != self.water_heater_sensor_layout.model_nodes:
            raise RuntimeError(
                f"Expected {self.water_heater_sensor_layout.model_nodes} tank nodes, "
                f"got {water_heater.model.n_nodes}"
            )
        schedule_file = raw_directory / f"{self.home.home_id}_schedule.csv"
        schedule = pd.read_csv(schedule_file)
        raw_times = schedule.pop("Time")
        if requested_start.tzinfo is not None:
            first_time = pd.Timestamp(raw_times.iloc[0])
            if first_time.tzinfo is None:
                raise ValueError(
                    "Timezone-aware simulation requires timezone-aware OCHRE schedule timestamps"
                )
            schedule.index = pd.DatetimeIndex(
                pd.to_datetime(raw_times, utc=True).dt.tz_convert(requested_start.tz),
                name="Time",
            )
        else:
            schedule.index = pd.DatetimeIndex(pd.to_datetime(raw_times), name="Time")
        if schedule.index.has_duplicates or not schedule.index.is_monotonic_increasing:
            raise ValueError("OCHRE schedule timestamps must be unique and increasing")
        water_scenarios = None
        if self.water_draw_data is not None:
            replay_index = schedule.index[
                (schedule.index >= requested_start) & (schedule.index < requested_end)
            ]
            warmup_water_draws = None
            if warmup:
                original_draw_columns = [
                    name for name in (
                        "Water Heating (L/min)", "Clothes Washer (L/min)",
                        "Dishwasher (L/min)",
                    ) if name in schedule
                ]
                warmup_water_draws = schedule.loc[
                    schedule.index < requested_start,
                    original_draw_columns,
                ].copy()
            schedule["Water Heating (L/min)"] = np.nan
            schedule.loc[replay_index, "Water Heating (L/min)"] = self.water_draw_data.flow_at(replay_index)
            # Replay this one series as tempered fixture demand, replacing all
            # three original draw channels rather than adding appliance use.
            schedule["Clothes Washer (L/min)"] = 0.0
            schedule["Dishwasher (L/min)"] = 0.0
            if warmup_water_draws is not None:
                schedule.loc[warmup_water_draws.index, warmup_water_draws.columns] = warmup_water_draws
            schedule.to_csv(schedule_file, index_label="Time")
            if scenario_mode and apply_controls:
                from .water_forecast import ScenarioWaterForecast

                water_scenarios = (self.unified_context or {}).get("water_scenarios")
                if water_scenarios is None:
                    water_scenarios = ScenarioWaterForecast(
                        self.water_draw_data, method=water_controller.scenario_method,
                        horizon=water_controller.horizon,
                        timestep=water_controller.model_timestep,
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
        if (not apply_controls and hvac_controller is not None) or (
            self.hub_split_replay_kw is not None and hvac_controller is None
        ):
            # The matched HVAC baseline follows the envelope-training profile
            # with OCHRE's ordinary thermostat and no predictive decisions.
            schedule["HVAC Heating Setpoint (C)"] = heating_setpoints_c(schedule.index)
            schedule.to_csv(schedule_file, index_label="Time")
        forecast_price_at = self.controller.electricity_price_for_step
        perfect_joint_forecasts = (self.unified_context or {}).get("perfect_forecasts", False)
        if perfect_joint_forecasts or (
            joint_controller is not None
            and getattr(joint_controller, "ev_controller", None) is not None
        ):
            # Joint thermal/EV history needs warmup weather, but the warmup
            # thermostat branch never calls a predictive controller. Leave its
            # unused prices explicitly unpriced instead of requesting prior-day
            # RTP. The issue-time tariff policy below remains unchanged.
            active_price_at = forecast_price_at
            forecast_price_at = lambda time: (
                float("nan") if pd.Timestamp(time) < requested_start else active_price_at(time)
            )
        forecast = Forecast.from_schedule(
            # The EV-only controller is inactive during native thermal warmup.
            # Its tariff forecast starts at the first actual decision; asking
            # for unused prior-day RTP values would reject a valid study week.
            schedule.loc[schedule.index >= requested_start] if ev_only_hub_controller is not None else schedule,
            forecast_price_at,
            water_scenarios=water_scenarios,
            rest_of_home_forecast=self.rest_of_home_forecast if apply_controls else None,
            hub_other_forecast=self.hub_other_forecast if apply_controls else None,
            outside_hub_forecast=self.outside_hub_forecast if apply_controls else None,
            ev_schedules=self.ev_mobility_schedules,
            backed_load_kw=self.ev_backed_load,
            weather_archive=self.weather_archive if apply_controls else None,
            price_as_known_at=(
                getattr(self.controller, "electricity_price_as_known_at", None)
                if apply_controls and not perfect_joint_forecasts and hasattr(
                    getattr(self.controller, "price_schedule", None), "forecast_callback"
                ) else None
            ),
        )
        if self.unified_context:
            forecast.providers.update(self.unified_context.get("providers", {}))
        if ev_equipment and forecast.backed_load_kw is None:
            if (any(equipment.options.v2h for equipment in ev_equipment.values())
                    and ev_horizon and self.hub_split_replay_kw is None):
                raise ValueError("V2H MPC requires an explicit backed-load forecast; use the EV comparison runner")
            if self.hub_split_replay_kw is None:
                forecast = replace(forecast, backed_load_kw=pd.Series(0.0, index=schedule.index))
        ev_only_forecast = None
        if ev_only_hub_controller is not None and apply_controls:
            ev_name = ev_only_hub_controller.equipment_name
            vehicle_id = ev_only_hub_controller.ev.vehicle_id
            if vehicle_id not in forecast.ev_schedules:
                raise ValueError("EV-only Home Hub control requires its EV mobility schedule")
            ev_provider = forecast.providers.get(ev_name)
            ev_only_forecast = replace(
                forecast,
                data=forecast.data.loc[:, [PRICE_COLUMN]].copy(),
                water_scenarios=None,
                rest_of_home_forecast=None,
                hub_other_forecast=None,
                outside_hub_forecast=None,
                weather_archive=None,
                providers={ev_name: ev_provider} if ev_provider is not None else {},
                ev_schedules={vehicle_id: forecast.ev_schedules[vehicle_id]},
                ev_states=forecast.ev_states,
            )
        previous_status: Mapping[str, Any] | None = None
        controller_previous_status: Mapping[str, Any] | None = None
        control_records: list[dict[str, Any]] = []
        sensor_records: list[dict[str, Any]] = []
        training_records: list[dict[str, Any]] = []
        hvac_training_records: list[dict[str, Any]] = []
        try:
            for timestamp in dwelling.sim_times:
                if timestamp >= requested_end:
                    break
                if dwelling.current_time != timestamp:
                    raise RuntimeError(
                        f"OCHRE clock mismatch: expected {timestamp}, got "
                        f"{dwelling.current_time}"
                    )

                # Split OCHRE's normal update into its public phases so the HEMS
                # can observe current exogenous schedule inputs before deciding.
                is_warmup = timestamp < requested_start
                guard_record: dict[str, Any] = {}
                draw_inputs = None
                if self.water_draw_data is not None and not is_warmup:
                    draw_inputs = schedule.loc[timestamp, [
                        "Water Heating (L/min)", "Clothes Washer (L/min)", "Dishwasher (L/min)",
                    ]].to_dict()
                dwelling.update_inputs(schedule_inputs=draw_inputs)
                for name, equipment in ev_equipment.items():
                    equipment.prepare_boundary(timestamp)
                    forecast.ev_states.update({
                        f"{name} SOC (-)": equipment.soc,
                        f"{name} SOH (-)": equipment.soh,
                        f"{name} Capacity (kWh)": equipment.capacity,
                        f"{name} Parked": equipment.in_event,
                    })
                if is_warmup:
                    # Follow the same thermostat schedule used by the matched
                    # baseline and envelope-model training before MPC starts.
                    controls = ({
                        (hvac_controller.heating_name if hvac_controller is not None else "HVAC Heating"): {
                            "Setpoint": heating_setpoint_c(pd.Timestamp(timestamp))
                        }
                    } if hvac_controller is not None or self.hub_split_replay_kw is not None else {})
                elif apply_controls:
                    control_forecast = ev_only_forecast or forecast
                    controls = self.controller.controls_for_step(
                        pd.Timestamp(timestamp),
                        controller_previous_status,
                        control_forecast.data.loc[pd.Timestamp(timestamp)].to_dict(),
                        control_forecast,
                    )
                elif hvac_controller is not None:
                    controls = {
                        hvac_controller.heating_name: {
                            "Setpoint": heating_setpoint_c(pd.Timestamp(timestamp))
                        }
                    }
                else:
                    controls = {}
                if self.hub_split_replay_kw is not None and hvac_controller is None and not is_warmup:
                    if "Setpoint" in controls.get("HVAC Heating", {}):
                        raise ValueError(
                            "EV-only Home Hub replay cannot override the confirmed "
                            "four-period HVAC heating setpoint"
                        )
                    controls = {
                        **controls,
                        "HVAC Heating": {
                            **controls.get("HVAC Heating", {}),
                            "Setpoint": heating_setpoint_c(pd.Timestamp(timestamp)),
                        },
                    }
                if panel_guard is not None and not is_warmup:
                    controls, guard_record = panel_guard.apply(
                        pd.Timestamp(timestamp), previous_status, controls, tuple(ev_equipment)
                    )
                controls = self._validate_controls(controls, pd.Timestamp(timestamp))
                controls = self._seasonal_hvac_controls(controls)
                if not is_warmup:
                    initial_water_heater_states = water_heater.model.states.copy()
                    initial_indoor_temperature = envelope.indoor_zone.temperature
                dwelling.update_model(control_signal=controls)
                if not is_warmup:
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
                previous_status["HVAC Indoor Sensible Heat (W)"] = float(
                    envelope.indoor_zone.hvac_sens_gain
                )
                previous_status.update(
                    self.water_heater_sensor_layout.measure(water_heater.model.states)
                )
                if hasattr(water_heater, "hp_capacity_nominal"):
                    previous_status["Water Heating Nominal Heat Pump Capacity (W)"] = (
                        float(water_heater.hp_capacity_nominal)
                    )
                if apply_controls and hvac_controller is not None:
                    # OCHRE reports the prior minute's starting temperature,
                    # then advances the envelope to this boundary temperature.
                    # Keep this controller-only value out of plant results.
                    previous_status = {
                        **previous_status,
                        HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY: float(
                            envelope.indoor_zone.temperature
                        ),
                    }
                controller_previous_status = (
                    project_ev_only_hub_observation(
                        previous_status, ev_only_hub_controller.equipment_name,
                        include_total_home=ev_only_hub_controller.include_total_home_meter,
                    )
                    if ev_only_hub_controller is not None and apply_controls
                    else previous_status
                )
                if apply_controls:
                    for device in getattr(self.controller, "device_controllers", ()):
                        observe = getattr(device, "observe_plant_step", None)
                        if observe is not None:
                            observe(
                                pd.Timestamp(timestamp),
                                controller_previous_status if device is ev_only_hub_controller
                                else previous_status,
                            )
                    if water_controller is not None and joint_controller is None:
                        water_controller.observe_start_history(timestamp, previous_status)
                if is_warmup:
                    if apply_controls and hvac_controller is not None:
                        hvac_controller.record_warmup_step(
                            pd.Timestamp(timestamp) + pd.Timedelta(self.config.time_step),
                            previous_status,
                            forecast,
                        )
                    continue
                control_records.append(
                    {
                        "timestamp": timestamp,
                        "electricity_price_usd_per_kwh": (
                            forecast.price_at(pd.Timestamp(timestamp))
                        ),
                        **({
                            "weather_forecast_run_utc": str(
                                forecast.weather_archive.selected_run_time(pd.Timestamp(timestamp))
                            )
                        } if forecast.weather_archive is not None else {}),
                        **guard_record,
                        "control_signal_json": json.dumps(controls, sort_keys=True),
                    }
                )
        except Exception:
            dwelling.finalize(failed=True)
            raise

        raw, _, _ = dwelling.finalize()
        if ev_equipment:
            self.data_directory.mkdir(parents=True, exist_ok=True)
            (self.data_directory / "ev_battery_checkpoints.json").write_text(
                json.dumps({
                    "timestamp": str(dwelling.current_time),
                    "batteries": {name: equipment.battery_checkpoint()
                                  for name, equipment in ev_equipment.items()},
                }, indent=2) + "\n", encoding="utf-8",
            )
        if raw is None:
            raise RuntimeError(f"{self.home.home_id}: OCHRE returned no results")
        raw = raw.loc[(raw.index >= requested_start) & (raw.index < requested_end)]
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
            resistance_outside_hub=self.resistance_outside_hub,
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
            resistance_outside_hub=self.resistance_outside_hub,
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
        hvac_costs = hvac_operating_summary(
            mapped.data,
            uncontrolled_mapped.data,
            price,
            self.config.time_step.total_seconds() / 3600,
        )
        hvac_costs_file = self.data_directory / f"{self.home.home_id}_hvac_operating_costs.csv"
        hvac_costs.to_csv(hvac_costs_file)
        water_heater_controller, hvac_controller, joint_controller = (
            _predictive_controllers(self.controller)
        )
        joint_costs = None
        joint_costs_file = None
        if joint_controller is not None:
            # The two device summaries use the same timestamps and scenario labels.
            # This reports measured plant consumption, distinct from the MPC objective.
            joint_costs = water_heater_costs.add(hvac_costs)
            joint_costs_file = self.data_directory / (
                f"{self.home.home_id}_joint_hpwh_hvac_operating_costs.csv"
            )
            joint_costs.to_csv(joint_costs_file)
        figure_files = (
            self.figure_directory / f"{self.home.home_id}_hvac.png",
            self.figure_directory / f"{self.home.home_id}_water_heater.png",
            self.figure_directory / f"{self.home.home_id}_ev.png",
            self.figure_directory / f"{self.home.home_id}_whole_home_comparison.png",
            self.figure_directory / f"{self.home.home_id}_water_heater_power_comparison.png",
            self.figure_directory / f"{self.home.home_id}_water_usage_profile.png",
        )
        water_comparison = _predictive_controllers(self.controller)[0] is not None
        water_baseline_figure = self.figure_directory / f"{self.home.home_id}_water_heater_baseline.png"
        if water_comparison:
            figure_files += (water_baseline_figure,)
        hvac_figure_files: tuple[Path, ...] = ()
        if hvac_controller is not None:
            hvac_figure_files = (
                self.figure_directory / f"{self.home.home_id}_hvac_baseline.png",
                self.figure_directory / f"{self.home.home_id}_hvac_predictive.png",
                self.figure_directory / f"{self.home.home_id}_hvac_weather.png",
                self.figure_directory / f"{self.home.home_id}_hvac_power_comparison.png",
            )
            figure_files += hvac_figure_files
        ev_figure_files: tuple[Path, ...] = ()
        if self.home.evs:
            ev_figure_files = (
                self.figure_directory / f"{self.home.home_id}_ev_baseline.png",
                self.figure_directory / f"{self.home.home_id}_ev_predictive.png",
            )
            figure_files += ev_figure_files
        whole_home_figure_files = (
            self.figure_directory / f"{self.home.home_id}_whole_home_baseline.png",
            self.figure_directory / f"{self.home.home_id}_whole_home_predictive.png",
        )
        figure_files += whole_home_figure_files
        temperature_limits = None
        water_power_limits = None
        if water_comparison:
            maximum_water_power = max(
                float(mapped.data.water_heating_power_kw.max()),
                float(uncontrolled_mapped.data.water_heating_power_kw.max()),
            )
            water_power_limits = (0.0, max(0.1, maximum_water_power * 1.05))
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
            power_limits=water_power_limits,
        )
        plot_ev(
            mapped.data,
            raw,
            price,
            self.home.home_id,
            figure_files[2],
            self.config.plot_dpi,
        )
        if ev_figure_files:
            ev_power_min = min(
                0.0,
                float(mapped.data["ev_power_kw"].min()),
                float(uncontrolled_mapped.data["ev_power_kw"].min()),
            )
            ev_power_max = max(
                0.0,
                float(mapped.data["ev_power_kw"].max()),
                float(uncontrolled_mapped.data["ev_power_kw"].max()),
            )
            ev_power_padding = max(0.5, 0.05 * (ev_power_max - ev_power_min))
            ev_power_limits = (
                ev_power_min - ev_power_padding,
                ev_power_max + ev_power_padding,
            )
            ev_options = self.home.evs[0].hems if len(self.home.evs) == 1 else None
            health_soc_bounds = (
                (ev_options.health_soc_min, ev_options.health_soc_max)
                if ev_options is not None else None
            )
            comfort_soc_min = (
                ev_options.comfort_soc_min if ev_options is not None else None
            )
            for results, raw_results, operation, output_file in (
                (uncontrolled_mapped.data, uncontrolled_raw, "Baseline", ev_figure_files[0]),
                (mapped.data, raw, "Predictive control", ev_figure_files[1]),
            ):
                plot_ev_operation(
                    results,
                    raw_results,
                    price,
                    self.home.home_id,
                    operation,
                    output_file,
                    self.config.plot_dpi,
                    power_limits=ev_power_limits,
                    health_soc_bounds=health_soc_bounds,
                    comfort_soc_min=comfort_soc_min,
                )
        whole_home_min = min(
            0.0,
            float(mapped.data["total_electric_power_kw"].min()),
            float(uncontrolled_mapped.data["total_electric_power_kw"].min()),
        )
        whole_home_max = max(
            0.0,
            float(mapped.data["total_electric_power_kw"].max()),
            float(uncontrolled_mapped.data["total_electric_power_kw"].max()),
        )
        whole_home_padding = max(0.5, 0.05 * (whole_home_max - whole_home_min))
        whole_home_limits = (
            whole_home_min - whole_home_padding,
            whole_home_max + whole_home_padding,
        )
        plot_whole_home_comparison(
            mapped.data,
            uncontrolled_mapped.data,
            price,
            self.home.home_id,
            figure_files[3],
            self.config.plot_dpi,
            power_limits=whole_home_limits,
        )
        for results, operation, output_file in (
            (uncontrolled_mapped.data, "Baseline", whole_home_figure_files[0]),
            (mapped.data, "Predictive control", whole_home_figure_files[1]),
        ):
            plot_whole_home_operation(
                results,
                price,
                self.home.home_id,
                operation,
                output_file,
                self.config.plot_dpi,
                power_limits=whole_home_limits,
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
        if water_comparison:
            plot_water_heater_baseline(
                uncontrolled_training_data,
                self.water_heater_sensor_layout.sensor_columns,
                self.home.home_id,
                water_baseline_figure,
                self.config.plot_dpi,
                water_usage=uncontrolled_training_data[
                    "water_heater_draw_l_per_min"
                ],
                temperature_limits=temperature_limits,
                power_limits=water_power_limits,
            )
        if hvac_controller is not None:
            comfort_setpoint = pd.Series(
                heating_setpoints_c(mapped.data.index),
                index=mapped.data.index,
                name="training_heating_setpoint_c",
            )
            hvac_limits = hvac_temperature_limits(
                mapped.data,
                uncontrolled_mapped.data,
                comfort_setpoint,
                comfort_band_c=hvac_controller.comfort_band_c,
                thermostat_deadband_c=self.home.heating_deadband_c,
                thermostat_deadband_offset=self.home.heating_deadband_offset,
            )
            hvac_power_scale = hvac_power_limits(
                mapped.data, uncontrolled_mapped.data,
            )
            for results, operation, output_file in (
                (uncontrolled_mapped.data, "Baseline", hvac_figure_files[0]),
                (mapped.data, "Predictive control", hvac_figure_files[1]),
            ):
                plot_hvac_operation(
                    results,
                    comfort_setpoint,
                    self.home.home_id,
                    operation,
                    output_file,
                    self.config.plot_dpi,
                    comfort_band_c=hvac_controller.comfort_band_c,
                    thermostat_deadband_c=self.home.heating_deadband_c,
                    thermostat_deadband_offset=self.home.heating_deadband_offset,
                    temperature_limits=hvac_limits,
                    power_limits=hvac_power_scale,
                    price=price,
                )
            plot_hvac_weather(
                schedule,
                mapped.data.index,
                self.home.home_id,
                hvac_figure_files[2],
                self.config.plot_dpi,
            )
            plot_hvac_power_comparison(
                mapped.data,
                uncontrolled_mapped.data,
                price,
                self.home.home_id,
                hvac_figure_files[3],
                self.config.plot_dpi,
                operating_summary=hvac_costs,
            )
        optimizer_plan_directory = None
        optimizer_plan_figure_files: tuple[Path, ...] = ()
        hvac_optimizer_plan_directory = None
        hvac_optimizer_plan_figure_files: tuple[Path, ...] = ()
        if water_heater_controller is not None:
            for name in (
                "solve_history", "objective_component_history", "scenario_diagnostics_history",
                "rolling_cycle_diagnostics_history",
            ):
                history = getattr(water_heater_controller, name, None)
                if isinstance(history, pd.DataFrame) and not history.empty:
                    history.to_csv(self.data_directory / f"optimization_{name}.csv", index=False)
            water_run_settings = {
                "model_kind": water_heater_controller.model_kind,
                "optimization_kind": water_heater_controller.optimization_kind,
                "forecast_type": water_heater_controller.forecast_type,
                "scenario_method": water_heater_controller.scenario_method,
                "prediction_horizon_hours": water_heater_controller.horizon.total_seconds() / 3600,
                "prediction_timestep_minutes": water_heater_controller.model_timestep.total_seconds() / 60,
                "comfort_weight": water_heater_controller.comfort_weight,
                "legionella_weight": water_heater_controller.legionella_weight,
                "persistence_weight": water_heater_controller.persistence_weight,
                "terminal_weight": water_heater_controller.terminal_weight,
                "enforce_cycle_limit": water_heater_controller.enforce_cycle_limit,
                "max_cycles": water_heater_controller.max_cycles,
                "enforce_dwell_constraints": water_heater_controller.enforce_dwell_constraints,
                "rolling_start_limit_24h": water_heater_controller.rolling_start_limit_24h,
                "rolling_start_slack_weight": water_heater_controller.rolling_start_slack_weight,
                "terminal_cycle_reserve_weight": water_heater_controller.terminal_cycle_reserve_weight,
                "effective_native_warmup_hours": self._effective_hvac_warmup_duration().total_seconds() / 3600,
                "rolling_start_window": "(interval_start - 24 hours, interval_start]",
                "observed_start_timestamp": "completed plant feedback Time",
                "cycle_slack_normalization": "prediction timestep hours / full horizon control intervals",
                "terminal_cycle_reserve": "one unused start at the endpoint; one-time cost",
                "comfort_scenario_aggregation": "maximum of complete per-scenario horizon costs",
                "other_scenario_aggregation": "arithmetic mean of complete per-scenario costs",
            }
            (self.data_directory / "water_heater_run_settings.json").write_text(
                json.dumps(water_run_settings, indent=2) + "\n", encoding="utf-8",
            )
        if joint_controller is not None:
            for name in ("solve_history", "objective_component_history"):
                history = getattr(joint_controller, name, None)
                if isinstance(history, pd.DataFrame) and not history.empty:
                    history.to_csv(
                        self.data_directory / f"joint_optimization_{name}.csv",
                        index=False,
                    )
            joint_plans = getattr(joint_controller, "optimization_plans", ())
            if joint_plans:
                joint_plan_directory = self.data_directory / "joint_optimizer_plans"
                joint_plan_directory.mkdir(parents=True, exist_ok=True)
                for position, plan in enumerate(joint_plans, start=1):
                    solved_at = pd.Timestamp(plan.attrs.get("solved_at", plan.index[0]))
                    plan.to_csv(
                        joint_plan_directory
                        / f"joint_plan_{position:04d}_{solved_at:%Y%m%d_%H%M%S}.csv"
                    )
        for ev_controller in _ev_predictive_controllers(self.controller):
            ev_name = ev_controller.equipment_name.replace(" ", "_")
            for attribute, filename in (
                ("solve_history", "solve_history.csv"),
                ("action_history", "action_history.csv"),
                ("peak_comfort_history", "peak_comfort_history.csv"),
            ):
                rows = getattr(ev_controller, attribute, ())
                if rows:
                    pd.DataFrame(rows).to_csv(
                        self.data_directory / f"{ev_name}_{filename}", index=False,
                    )
            plans = getattr(ev_controller, "plans", ())
            if plans:
                (self.data_directory / f"{ev_name}_plans.json").write_text(
                    json.dumps(plans, default=str, indent=2) + "\n", encoding="utf-8",
                )
        if hvac_controller is not None:
            optimization_type, hvac_solve_stats = hvac_optimization_reporting(
                hvac_controller.optimization_kind,
                hvac_controller.solve_durations_s,
            )
            hvac_run_settings = {
                "optimization_type": optimization_type,
                "model_type": "state_space_regularized",
                "model_path": str(hvac_controller.model_path),
                "simulation_start_time": self.config.start_time.isoformat(),
                "simulation_end_time": (self.config.start_time + self.config.duration).isoformat(),
                "simulation_duration_hours": self.config.duration.total_seconds() / 3600,
                "plant_timestep_minutes": self.config.time_step.total_seconds() / 60,
                "prediction_horizon_hours": hvac_controller.horizon.total_seconds() / 3600,
                "prediction_timestep_minutes": hvac_controller.model_timestep.total_seconds() / 60,
                "solver": hvac_controller.solver,
                "comfort_band_c": hvac_controller.comfort_band_c,
                "comfort_weight": hvac_controller.comfort_weight,
                "persistence_weight": hvac_controller.persistence_weight,
                "enforce_degree_hour_budget": hvac_controller.enforce_degree_hour_budget,
                "enforce_instantaneous_cap": hvac_controller.enforce_instantaneous_cap,
                "degree_hour_limit_c_hours": hvac_controller.degree_hour_limit_c_hours,
                "max_deviation_c": hvac_controller.max_deviation_c,
                "degree_hour_slack_weight": hvac_controller.degree_hour_slack_weight,
                "hvac_warmup_hours": self._effective_hvac_warmup_duration().total_seconds() / 3600,
                "comfort_history_warmup_hours": (
                    self._effective_hvac_warmup_duration().total_seconds() / 3600
                    if hvac_controller.enforce_degree_hour_budget else 0
                ),
                "enforce_cycle_limit": hvac_controller.enforce_cycle_limit,
                "max_cycles": int(hvac_controller.max_cycles),
                "enforce_dwell_constraints": hvac_controller.enforce_dwell_constraints,
                "dwell_time_on_minutes": hvac_controller.dwell_time_on.total_seconds() / 60,
                "dwell_time_off_minutes": hvac_controller.dwell_time_off.total_seconds() / 60,
                "baseline_heating_setpoint_profile": list(HEATING_SETPOINT_PERIODS),
                "plant_defrost_policy": {
                    "er_pause_requires_compressor": self.defrost_er_pause_requires_compressor,
                    "finish_active_cycle": self.defrost_finish_active_cycle,
                    "scope": "true plant, including native reference and warm-up",
                    "predictive_model_changed": False,
                },
                "hp_only_er_reset_policy": "pending until fresh nondefrost ER-off feedback",
            }
            (self.data_directory / "hvac_run_settings.json").write_text(
                json.dumps(hvac_run_settings, indent=2) + "\n",
                encoding="utf-8",
            )
            for name in (
                "solve_history", "objective_component_history", "comfort_budget_history"
            ):
                history = getattr(hvac_controller, name, None)
                if isinstance(history, pd.DataFrame) and not history.empty:
                    history.to_csv(
                        self.data_directory / f"hvac_optimization_{name}.csv",
                        index=False,
                    )
            if hvac_solve_stats is not None:
                (self.data_directory / "hvac_optimization_solve_time_summary.json").write_text(
                    json.dumps(hvac_solve_stats, indent=2) + "\n",
                    encoding="utf-8",
                )
            if hvac_controller.optimization_plans:
                hvac_plan_directory = self.data_directory / "hvac_optimizer_plans"
                hvac_plan_directory.mkdir(parents=True, exist_ok=True)
                for position, plan in enumerate(hvac_controller.optimization_plans, start=1):
                    solved_at = pd.Timestamp(plan.attrs["solved_at"])
                    plan.to_csv(
                        hvac_plan_directory
                        / f"hvac_plan_{position:04d}_{solved_at:%Y%m%d_%H%M%S}.csv"
                    )
                if self.save_optimizer_plan_figures:
                    hvac_optimizer_plan_directory = self.figure_directory / "hvac_optimizer_plans"
                    hvac_optimizer_plan_figure_files = plot_hvac_optimization_plans(
                        hvac_controller.optimization_plans,
                        hvac_optimizer_plan_directory,
                        self.config.plot_dpi,
                    )
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
            hvac_operating_costs=hvac_costs,
            controls=controls,
            data_directory=self.data_directory,
            figure_directory=self.figure_directory,
            figure_files=figure_files,
            optimizer_plan_directory=optimizer_plan_directory,
            optimizer_plan_figure_files=optimizer_plan_figure_files,
            hvac_optimizer_plan_directory=hvac_optimizer_plan_directory,
            hvac_optimizer_plan_figure_files=hvac_optimizer_plan_figure_files,
            peak_power_kw=float(mapped.data["total_electric_power_kw"].max()),
            joint_operating_costs=joint_costs,
        )
        print("\nSingle-home HEMS simulation complete")
        print(f"  Home: {summary.home_id}")
        print(f"  Timesteps: {len(summary.standardized_results)}")
        for device in getattr(self.controller, "device_controllers", ()):
            for kind, durations in getattr(device, "solve_durations_s", {}).items():
                if durations:
                    if isinstance(device, HVACPredictiveController):
                        stats = solve_time_summary(durations)
                        print(
                            f"  HVAC {kind} solve time over {stats['solve_count']} calls: "
                            f"{stats['average_s']:.3f} s average, "
                            f"{stats['p95_s']:.3f} s p95, "
                            f"{stats['slowest_5_percent_average_s']:.3f} s "
                            f"average for the slowest 5%"
                        )
                    else:
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
        if hvac_controller is not None:
            print("  HVAC operation:")
            for scenario, values in summary.hvac_operating_costs.iterrows():
                print(
                    f"    {scenario}: {values['energy_kwh']:.2f} kWh, "
                    f"{values['average_power_kw']:.3f} kW average, "
                    f"${values['operating_cost_usd']:.2f}"
                )
            print(f"  HVAC operating costs: {hvac_costs_file}")
        if joint_costs is not None:
            print("  Joint HPWH and HVAC operation:")
            for scenario, values in joint_costs.iterrows():
                print(
                    f"    {scenario}: {values['energy_kwh']:.2f} kWh, "
                    f"{values['average_power_kw']:.3f} kW average, "
                    f"${values['operating_cost_usd']:.2f}"
                )
            print(f"  Joint HPWH and HVAC operating costs: {joint_costs_file}")
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
        if summary.hvac_optimizer_plan_directory is not None:
            print(
                "  HVAC optimizer plans: "
                f"{summary.hvac_optimizer_plan_directory} "
                f"({len(summary.hvac_optimizer_plan_figure_files)} figures)"
            )
        return summary

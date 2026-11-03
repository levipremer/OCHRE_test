#!/usr/bin/env python3
# ruff: noqa: E402
"""Run one selectable five-home-block dwelling with a stepwise HEMS."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
FIVE_HOME_CONFIG = REPO_ROOT / "examples" / "five_home_block" / "config"
HOME_IDS = tuple(f"home_{number:02d}" for number in range(1, 6))
CONTROLLER_CHOICES = ("none", "water-heater", "hvac", "ev", "all")


def _local_weather_time(value: dt.datetime, weather_metadata: dict | None) -> dt.datetime | pd.Timestamp:
    """Interpret CLI wall times in the configured IANA timezone for dated weather."""
    if weather_metadata is None:
        return value
    timezone = str(weather_metadata["timezone"])
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        if timestamp.tzinfo is None else timestamp.tz_convert(timezone)
    )


def load_weather_forecast_archive(
    config, mode: str, file_override: Path | None = None
) -> ArchivedWeatherForecast | None:
    """Resolve the shared HVAC forecast source from CLI or neighborhood config."""
    if mode == "perfect":
        return None
    if mode != "predict":
        raise ValueError("Weather forecast mode must be perfect or predict")
    weather_forecast_file = (
        file_override if file_override is not None else config.weather_forecast_file
    )
    if weather_forecast_file is None:
        raise ValueError(
            "--weather-forecast-mode predict requires --weather-forecast-file "
            "or weather_forecast_file in neighborhood.json"
        )
    weather_forecast_file = Path(weather_forecast_file)
    if not weather_forecast_file.is_absolute():
        weather_forecast_file = REPO_ROOT / weather_forecast_file
    return ArchivedWeatherForecast.from_parquet(
        weather_forecast_file,
        local_timezone=(config.weather_metadata or {}).get("timezone", LOCAL_TIMEZONE),
    )


def default_output_subdirectory(
    controller: str, optimization: str, model: str,
    forecast_type: str = "perfect", scenario_method: str = "daily_budget_high_rf_1",
    simulation_duration: dt.timedelta | None = None,
) -> str:
    if controller == "hvac":
        if simulation_duration is None or simulation_duration <= dt.timedelta(0):
            raise ValueError("HVAC output naming requires a positive simulation duration")
        prefixes = {"convex": "cvx", "mixed integer": "mi"}
        if optimization not in prefixes:
            raise ValueError(f"Unknown HVAC optimization formulation: {optimization!r}")
        hours = simulation_duration.total_seconds() / 3600
        return f"{prefixes[optimization]}_state_space_regularized_{hours:g}h"
    controller_names = {
        "none": "no_control",
        "water-heater": "water_heater_only",
        "hvac": "hvac_only",
        "ev": "ev_only",
        "all": "all_controllers",
    }
    name = controller_names[controller]
    if controller in {"water-heater", "all"}:
        name += f"_{optimization}_{model.replace('-', '_')}"
        if forecast_type == "scenario":
            name += f"_scenario_{scenario_method}"
    return name


def next_available_output_directory(directory: Path) -> Path:
    """Give each automatically named HVAC run its own result directory."""
    if not directory.exists():
        return directory
    for number in range(2, 10000):
        candidate = directory.with_name(f"{directory.name}_{number}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No available output directory beside {directory}")


for local_path in (REPO_ROOT, EXAMPLE_ROOT):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems import (
    SingleHomeHEMSSimulation,
    select_home,
)
from examples.single_home_hems.single_home_hems.controllers import (
    DeviceController,
    EVPredictiveController,
    HVACPredictiveController,
    WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems.controllers.water_heater import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RC_MODEL_PATH,
    DEFAULT_RC_OLD_MODEL_PATH,
    DEFAULT_RC_STRATIFIED_MODEL_PATH,
)
from examples.single_home_hems.single_home_hems.rate_billing import winter_daily_demand_bill
from examples.single_home_hems.single_home_hems.rate_setup import (
    RATE_CHOICES, select_rate_for_simulation,
)
from examples.single_home_hems.single_home_hems.training import (
    HVACInternalGainRegime,
    WaterHeaterSensorLayout,
)
from examples.single_home_hems.single_home_hems.weather_forecast import (
    ArchivedWeatherForecast,
    LOCAL_TIMEZONE,
)


def build_device_controllers(
    selection: str,
    ev_names: tuple[str, ...],
    water_heater_model: str = "state-space",
    water_heater_optimization: str = "milp",
    water_heater_solver: str = "GUROBI",
    water_heater_max_cycles: int | None = None,
    water_heater_switching_weight: float | None = None,
    water_heater_comfort_weight: float | None = None,
    water_heater_legionella_weight: float | None = None,
    water_heater_persistence_weight: float | None = None,
    water_heater_enforce_cycle_limit: bool | None = None,
    water_heater_enforce_dwell_constraints: bool | None = None,
    water_heater_legionella_cooldown_enabled: bool | None = None,
    forecast_type: str = "perfect",
    scenario_method: str = "daily_budget_high_rf_1",
    water_heater_horizon_hours: float = 12.0,
    hvac_optimization: str = "convex",
    hvac_solver: str = "GUROBI",
    hvac_comfort_band_c: float = 1.5,
    hvac_comfort_weight: float = 1.0,
    hvac_persistence_weight: float = 1.0,
    hvac_enforce_degree_hour_budget: bool = False,
    hvac_enforce_instantaneous_cap: bool = False,
    hvac_degree_hour_limit_c_hours: float = 2.0,
    hvac_max_deviation_c: float = 1.0,
    hvac_degree_hour_slack_weight: float = 100.0,
    hvac_enforce_cycle_limit: bool = False,
    hvac_enforce_dwell_constraints: bool = False,
    hvac_max_cycles: int = 3,
    hvac_dwell_on_minutes: float = 60.0,
    hvac_dwell_off_minutes: float = 15.0,
    hvac_gain_regime: HVACInternalGainRegime | None = None,
) -> tuple[DeviceController, ...]:
    """Build the selected set of independently testable device controllers."""
    if selection not in CONTROLLER_CHOICES:
        raise ValueError(f"Unknown controller selection: {selection!r}")
    controllers: list[DeviceController] = []
    if selection in {"water-heater", "all"}:
        model_paths = {
            "state-space": DEFAULT_MODEL_PATH,
            "rc": DEFAULT_RC_MODEL_PATH,
            "rc-old": DEFAULT_RC_OLD_MODEL_PATH,
            "rc-stratified": DEFAULT_RC_STRATIFIED_MODEL_PATH,
        }
        water_heater_options = {
            "model_kind": "rc" if water_heater_model.startswith("rc") else "state-space",
            "model_path": model_paths[water_heater_model],
            "optimization_kind": water_heater_optimization,
            "solver": water_heater_solver,
            "forecast_type": forecast_type,
            "scenario_method": scenario_method,
            "horizon": dt.timedelta(hours=water_heater_horizon_hours),
        }
        if water_heater_max_cycles is not None:
            water_heater_options["max_cycles"] = water_heater_max_cycles
        if water_heater_switching_weight is not None:
            water_heater_options["control_variation_weight"] = water_heater_switching_weight
        if water_heater_comfort_weight is not None:
            water_heater_options["comfort_weight"] = water_heater_comfort_weight
        if water_heater_legionella_weight is not None:
            water_heater_options["legionella_weight"] = water_heater_legionella_weight
        if water_heater_persistence_weight is not None:
            water_heater_options["persistence_weight"] = water_heater_persistence_weight
        if water_heater_enforce_cycle_limit is not None:
            water_heater_options["enforce_cycle_limit"] = water_heater_enforce_cycle_limit
        if water_heater_enforce_dwell_constraints is not None:
            water_heater_options["enforce_dwell_constraints"] = water_heater_enforce_dwell_constraints
        if water_heater_legionella_cooldown_enabled is not None:
            water_heater_options["legionella_cooldown_enabled"] = water_heater_legionella_cooldown_enabled
        controllers.append(WaterHeaterPredictiveController(**water_heater_options))
    if selection in {"hvac", "all"}:
        controllers.append(HVACPredictiveController(
            optimization_kind=hvac_optimization,
            solver=hvac_solver,
            comfort_band_c=hvac_comfort_band_c,
            comfort_weight=hvac_comfort_weight,
            persistence_weight=hvac_persistence_weight,
            enforce_degree_hour_budget=hvac_enforce_degree_hour_budget,
            enforce_instantaneous_cap=hvac_enforce_instantaneous_cap,
            degree_hour_limit_c_hours=hvac_degree_hour_limit_c_hours,
            max_deviation_c=hvac_max_deviation_c,
            degree_hour_slack_weight=hvac_degree_hour_slack_weight,
            enforce_cycle_limit=hvac_enforce_cycle_limit,
            enforce_dwell_constraints=hvac_enforce_dwell_constraints,
            max_cycles=hvac_max_cycles,
            dwell_time_on=dt.timedelta(minutes=hvac_dwell_on_minutes),
            dwell_time_off=dt.timedelta(minutes=hvac_dwell_off_minutes),
            gain_regime=hvac_gain_regime or HVACInternalGainRegime(),
        ))
    if selection in {"ev", "all"}:
        controllers.append(EVPredictiveController(ev_names=ev_names))
    return tuple(controllers)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one five-home-block dwelling with a user-editable HEMS"
    )
    parser.add_argument(
        "--home-id",
        default="home_01",
        choices=HOME_IDS,
        help="Home from the five-home block to simulate (default: home_01)",
    )
    parser.add_argument(
        "--controller",
        choices=CONTROLLER_CHOICES,
        default="ev",
        help="Device controller to test (default: %(default)s)",
    )
    parser.add_argument(
        "--rate-structure", choices=RATE_CHOICES, default="legacy",
        help="Electricity tariff choice; legacy keeps the original example rate",
    )
    parser.add_argument(
        "--water-heater-model", choices=("state-space", "rc", "rc-old", "rc-stratified"),
        default="state-space", help="Predictor used by the water-heater controller",
    )
    parser.add_argument(
        "--water-heater-optimization", choices=("convex", "milp"),
        default="milp", help="Water-heater optimization formulation (default: milp)",
    )
    parser.add_argument(
        "--forecast-type", "--forecast_type", choices=("perfect", "scenario"),
        default="perfect", help="Water-use forecast for the water-heater MPC",
    )
    parser.add_argument(
        "--weather-forecast-mode", choices=("perfect", "predict"),
        default="perfect", help="HVAC future-weather source (default: perfect)",
    )
    parser.add_argument(
        "--weather-forecast-file", type=Path,
        help="Prepared 00Z GFS forecast Parquet; overrides weather_forecast_file in source config",
    )
    parser.add_argument(
        "--scenario-method", "--scenario-forecast-type",
        choices=(
            "daily_budget_high_rf_1", "daily_budget_high",
            *(f"rmse_volume_6h_k{k}{suffix}" for k in (5, 7, 10)
              for suffix in ("", "_rf_1")),
            *(f"{method}_k{k}_rf_1" for method in
              ("day_type", "morning_progress", "tail_anchored", "recent")
              for k in (5, 7, 10)),
        ),
        default="daily_budget_high_rf_1",
        help="Historical scenario method; rf_1 adjusts only the first 15-minute step",
    )
    parser.add_argument(
        "--water-draw-file", type=Path,
        help="Replace fixture water use with a DCHouse timestamp/L-per-minute CSV",
    )
    parser.add_argument(
        "--water-draw-start", type=dt.datetime.fromisoformat,
        help="Local source-data start; time of day must match the simulation start",
    )
    parser.add_argument(
        "--water-heater-horizon-hours", type=float, default=12.0,
        help="Prediction horizon in hours (scenario mode uses 15-minute intervals)",
    )
    parser.add_argument(
        "--skip-optimizer-plan-figures", action="store_true",
        help="Save optimizer diagnostics without rendering one figure per solve",
    )
    parser.add_argument(
        "--water-heater-solver", choices=("GUROBI", "SCIPY", "CLARABEL"),
        default="GUROBI", help="CVXPY solver for water-heater optimization",
    )
    parser.add_argument(
        "--hvac-optimization", choices=("convex", "mixed integer"),
        default="convex", help="HVAC optimization formulation (default: convex)",
    )
    parser.add_argument(
        "--hvac-solver", choices=("GUROBI", "CLARABEL"), default="GUROBI",
        help="CVXPY solver for HVAC optimization (default: GUROBI)",
    )
    parser.add_argument(
        "--hvac-comfort-band-c", type=float, default=1.5,
        help="Comfort band in C around the training heating setpoint (default: 1.5)",
    )
    parser.add_argument(
        "--hvac-comfort-weight", type=float, default=1.0,
        help="Penalty on HVAC comfort slack (default: 1.0)",
    )
    parser.add_argument(
        "--hvac-persistence-weight", type=float, default=1.0,
        help="Penalty on heat-pump heat variation (default: 1.0; 0 disables it)",
    )
    parser.add_argument(
        "--hvac-degree-hour-budget", choices=("on", "off"), default="off",
        help="Enforce the soft 24-hour moving comfort budget (default: off)",
    )
    parser.add_argument(
        "--hvac-instantaneous-cap", choices=("on", "off"), default="off",
        help="Enforce the hard per-step comfort deviation cap (default: off)",
    )
    parser.add_argument(
        "--hvac-degree-hour-limit-c-hours", type=float, default=2.0,
        help="24-hour comfort-violation budget in degree-C-hours (default: 2.0)",
    )
    parser.add_argument(
        "--hvac-max-deviation-c", type=float, default=1.0,
        help="Maximum per-step deviation outside the comfort band in C (default: 1.0)",
    )
    parser.add_argument(
        "--hvac-degree-hour-slack-weight", type=float, default=100.0,
        help="Penalty on degree-hour budget slack before horizon-step averaging (default: 100.0)",
    )
    parser.add_argument(
        "--hvac-cycle-limit", choices=("on", "off"), default="off",
        help="Enforce the mixed-integer HP start limit (default: off)",
    )
    parser.add_argument(
        "--hvac-dwell-constraints", choices=("on", "off"), default="off",
        help="Enforce mixed-integer HP minimum-on and minimum-off times (default: off)",
    )
    parser.add_argument(
        "--hvac-max-cycles", type=int, default=3,
        help="Maximum planned HP starts plus one recent stop when enabled (default: 3)",
    )
    parser.add_argument(
        "--hvac-dwell-on-minutes", type=float, default=60.0,
        help="Minimum HP on time when dwell constraints are enabled (default: 60)",
    )
    parser.add_argument(
        "--hvac-dwell-off-minutes", type=float, default=15.0,
        help="Minimum HP off time when dwell constraints are enabled (default: 15)",
    )
    parser.add_argument(
        "--water-heater-cycle-limit",
        choices=("on", "off"),
        default="on",
        help="Enable or disable the MILP cycle-limit constraint (default: %(default)s)",
    )
    parser.add_argument(
        "--water-heater-dwell-constraints",
        choices=("on", "off"),
        default="off",
        help=(
            "Enable or disable both MILP minimum-on and minimum-off constraints "
            "together (default: %(default)s; on/off times remain 60/15 minutes)"
        ),
    )
    parser.add_argument(
        "--water-heater-max-cycles", type=int,
        help="Maximum predicted heat-pump starts when the MILP cycle limit is on",
    )
    parser.add_argument(
        "--water-heater-switching-weight", type=float,
        help="MILP penalty per kW change in predicted heat input (default: 0.003)",
    )
    parser.add_argument(
        "--water-heater-comfort-weight", type=float,
        help="Convex/MILP penalty on upper-tank temperature below 38 C",
    )
    parser.add_argument(
        "--water-heater-legionella-weight", type=float,
        help="Convex/MILP penalty on low average tank temperature",
    )
    parser.add_argument(
        "--water-heater-legionella-cooldown", action="store_true",
        help="Enable the RC Legionella cooldown allowance (6 h memory, 6 C band)",
    )
    parser.add_argument(
        "--water-heater-persistence-weight", type=float,
        help="Convex/MILP penalty on variation in predicted heat input",
    )
    parser.add_argument(
        "--water-heater-hp-only", action="store_true",
        help="Run the OCHRE plant and its matched baseline in heat-pump-only mode",
    )
    parser.add_argument(
        "--source-config-directory",
        type=Path,
        default=FIVE_HOME_CONFIG,
        help="Five-home configuration directory used as the source",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override the automatically named folder under outputs/HOME_ID",
    )
    parser.add_argument(
        "--time-step-minutes",
        type=float,
        help="Override the timestep inherited from the five-home block",
    )
    simulation_duration = parser.add_mutually_exclusive_group()
    simulation_duration.add_argument(
        "--duration-hours",
        type=float,
        help="Simulation duration in hours (required for HVAC control)",
    )
    simulation_duration.add_argument(
        "--end-time",
        type=dt.datetime.fromisoformat,
        help="End the simulation at this ISO timestamp (exclusive)",
    )
    parser.add_argument(
        "--start-date", "--start-time",
        type=dt.datetime.fromisoformat,
        dest="start_time",
        help="Optional ISO start date or timestamp (default: source configuration)",
    )
    initial_temperature = parser.add_mutually_exclusive_group()
    initial_temperature.add_argument(
        "--initial-temperature-c",
        type=float,
        default=21.0,
        help="Initial indoor temperature before any warm-up (default: 21 C)",
    )
    initial_temperature.add_argument(
        "--randomize-initial-temperature",
        action="store_const",
        const=None,
        dest="initial_temperature_c",
        help="Opt into OCHRE's random initial temperature within the HVAC deadband",
    )
    parser.add_argument(
        "--water-heater-nodes",
        type=int,
        default=12,
        help="Number of stratified water-tank nodes (default: 12)",
    )
    parser.add_argument(
        "--water-heater-sensor-nodes",
        type=int,
        nargs="+",
        help="Virtual sensor nodes from top to bottom (default: nodes 3 and 10)",
    )
    parser.add_argument(
        "--internal-gain-day-start-hour",
        type=int,
        default=6,
        help="Start hour for the HVAC day-gain regime (default: 6)",
    )
    parser.add_argument(
        "--internal-gain-day-end-hour",
        type=int,
        default=22,
        help="End hour for the HVAC day-gain regime (default: 22)",
    )
    parser.add_argument(
        "--heating-deadband-c",
        type=float,
        help="Override the selected home's compressor thermostat deadband",
    )
    parser.add_argument(
        "--heating-deadband-offset",
        type=float,
        help="Override the deadband offset fraction (0.5 centers it)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    if args.rate_structure == "rtp-miso+doe":
        raise ValueError(
            "rtp-miso+doe requires run_whole_home_joint.py: the separate-device "
            "runner cannot enforce the whole-home grid and panel envelopes"
        )
    if args.controller == "hvac":
        if args.end_time is not None:
            raise ValueError("HVAC runs use --duration-hours instead of --end-time")
        if args.duration_hours is None:
            raise ValueError("HVAC runs require --duration-hours")
    for name, value in (
        ("--hvac-comfort-band-c", args.hvac_comfort_band_c),
        ("--hvac-comfort-weight", args.hvac_comfort_weight),
        ("--hvac-persistence-weight", args.hvac_persistence_weight),
        ("--hvac-degree-hour-limit-c-hours", args.hvac_degree_hour_limit_c_hours),
        ("--hvac-max-deviation-c", args.hvac_max_deviation_c),
        ("--hvac-degree-hour-slack-weight", args.hvac_degree_hour_slack_weight),
    ):
        if not 0 <= value < float("inf"):
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.hvac_max_cycles < 1:
        raise ValueError("--hvac-max-cycles must be positive")
    for name, value in (
        ("--hvac-dwell-on-minutes", args.hvac_dwell_on_minutes),
        ("--hvac-dwell-off-minutes", args.hvac_dwell_off_minutes),
    ):
        if not 0 < value < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    if (args.controller in {"hvac", "all"} and args.hvac_optimization == "convex"
            and (args.hvac_cycle_limit == "on" or args.hvac_dwell_constraints == "on")):
        raise ValueError("HVAC cycle and dwell constraints require --hvac-optimization 'mixed integer'")
    if not 0 < args.water_heater_horizon_hours < float("inf"):
        raise ValueError("--water-heater-horizon-hours must be finite and positive")
    if args.forecast_type == "scenario":
        if args.controller not in {"water-heater", "all"}:
            raise ValueError("Scenario forecasts require a water-heater controller")
        if args.water_heater_horizon_hours * 4 != round(args.water_heater_horizon_hours * 4):
            raise ValueError("Scenario forecasts require whole 15-minute horizon intervals")
        if args.water_draw_file is None:
            args.water_draw_file = EXAMPLE_ROOT / "external_data" / "water_flow_dchouse_l_per_min.csv"
    if args.water_draw_start is not None and args.water_draw_file is None:
        raise ValueError("--water-draw-start requires --water-draw-file")
    if args.water_heater_solver == "CLARABEL" and args.water_heater_optimization == "milp":
        raise ValueError("CLARABEL supports convex optimization only; use GUROBI or SCIPY for MILP")
    if (args.controller in {"hvac", "all"} and args.hvac_solver == "CLARABEL"
            and args.hvac_optimization == "mixed integer"):
        raise ValueError("CLARABEL supports convex HVAC optimization only; use GUROBI for mixed integer")
    config = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    home = select_home(config, args.home_id)

    if args.start_time is not None:
        config = replace(
            config, start_time=_local_weather_time(args.start_time, config.weather_metadata)
        )
    if args.time_step_minutes is not None:
        if args.time_step_minutes <= 0:
            raise ValueError("--time-step-minutes must be positive")
        config = replace(
            config,
            time_step=dt.timedelta(minutes=args.time_step_minutes),
        )
    if args.duration_hours is not None:
        if args.duration_hours <= 0:
            raise ValueError("--duration-hours must be positive")
        config = replace(
            config,
            duration=dt.timedelta(hours=args.duration_hours),
        )
    if args.end_time is not None:
        duration = pd.Timestamp(
            _local_weather_time(args.end_time, config.weather_metadata)
        ) - pd.Timestamp(config.start_time)
        if duration <= dt.timedelta(0):
            raise ValueError("--end-time must be after the simulation start")
        config = replace(config, duration=duration)
    if args.water_heater_max_cycles is not None and args.water_heater_max_cycles < 1:
        raise ValueError("--water-heater-max-cycles must be positive")
    if args.water_heater_switching_weight is not None and args.water_heater_switching_weight < 0:
        raise ValueError("--water-heater-switching-weight must be nonnegative")
    for name, value in (
        ("--water-heater-comfort-weight", args.water_heater_comfort_weight),
        ("--water-heater-legionella-weight", args.water_heater_legionella_weight),
        ("--water-heater-persistence-weight", args.water_heater_persistence_weight),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.heating_deadband_c is not None:
        if args.heating_deadband_c <= 0:
            raise ValueError("--heating-deadband-c must be positive")
        home = replace(home, heating_deadband_c=args.heating_deadband_c)
    if args.heating_deadband_offset is not None:
        if not 0 <= args.heating_deadband_offset <= 1:
            raise ValueError("--heating-deadband-offset must be between zero and one")
        home = replace(home, heating_deadband_offset=args.heating_deadband_offset)

    rate_selection = select_rate_for_simulation(args.rate_structure, config)

    default_controller_directory = default_output_subdirectory(
        args.controller,
        args.hvac_optimization if args.controller == "hvac" else args.water_heater_optimization,
        args.water_heater_model,
        args.forecast_type, args.scenario_method,
        config.duration if args.controller == "hvac" else None,
    )
    if args.weather_forecast_mode == "predict":
        default_controller_directory += "_weather_predict"
    if args.rate_structure != "legacy":
        default_controller_directory += "_" + args.rate_structure.lower().replace("+", "_")
    output_directory = (
        args.output_directory
        or EXAMPLE_ROOT / "outputs" / args.home_id / default_controller_directory
    )
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    if args.output_directory is None and (args.controller in {"hvac", "all"} or args.rate_structure != "legacy"):
        output_directory = next_available_output_directory(output_directory)
    config = replace(
        config,
        output_directory=output_directory.resolve(),
        homes=(home,),
    )
    config.validate()

    if args.weather_forecast_mode == "predict" and args.controller not in {"hvac", "all"}:
        raise ValueError("Predicted weather requires an HVAC controller")
    weather_archive = load_weather_forecast_archive(
        config, args.weather_forecast_mode, args.weather_forecast_file
    )

    sensor_nodes = (
        tuple(args.water_heater_sensor_nodes)
        if args.water_heater_sensor_nodes is not None
        else WaterHeaterSensorLayout.default_sensor_nodes(args.water_heater_nodes)
    )
    sensor_layout = WaterHeaterSensorLayout(
        model_nodes=args.water_heater_nodes,
        sensor_nodes=sensor_nodes,
    )
    if args.water_heater_model.startswith("rc") and args.controller in {"water-heater", "all"}:
        model_paths = {
            "rc": DEFAULT_RC_MODEL_PATH,
            "rc-old": DEFAULT_RC_OLD_MODEL_PATH,
            "rc-stratified": DEFAULT_RC_STRATIFIED_MODEL_PATH,
        }
        metadata_file = model_paths[args.water_heater_model] / "metadata.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(
                f"RC water-heater artifact is missing: {metadata_file}; "
                "run train_water_heater_rc.py first"
            )
        with metadata_file.open(encoding="utf-8") as stream:
            rc_metadata = json.load(stream)
        if (
            rc_metadata["columns"]["outputs"] != list(sensor_layout.sensor_columns)
            or rc_metadata["sensor_layout"]["model_nodes"] != sensor_layout.model_nodes
        ):
            raise ValueError("RC artifact and requested water-heater sensors differ")
    ev_names = tuple(f"EV_{ev.vehicle_id}" for ev in home.evs)
    hvac_gain_regime = HVACInternalGainRegime(
        day_start_hour=args.internal_gain_day_start_hour,
        day_end_hour=args.internal_gain_day_end_hour,
    )
    device_controllers = build_device_controllers(
        args.controller, ev_names, args.water_heater_model,
        args.water_heater_optimization, args.water_heater_solver,
        args.water_heater_max_cycles,
        args.water_heater_switching_weight,
        args.water_heater_comfort_weight,
        args.water_heater_legionella_weight,
        args.water_heater_persistence_weight,
        water_heater_enforce_cycle_limit=args.water_heater_cycle_limit == "on",
        water_heater_enforce_dwell_constraints=args.water_heater_dwell_constraints == "on",
        water_heater_legionella_cooldown_enabled=args.water_heater_legionella_cooldown,
        forecast_type=args.forecast_type,
        scenario_method=args.scenario_method,
        water_heater_horizon_hours=args.water_heater_horizon_hours,
        hvac_optimization=args.hvac_optimization,
        hvac_solver=args.hvac_solver,
        hvac_comfort_band_c=args.hvac_comfort_band_c,
        hvac_comfort_weight=args.hvac_comfort_weight,
        hvac_persistence_weight=args.hvac_persistence_weight,
        hvac_enforce_degree_hour_budget=args.hvac_degree_hour_budget == "on",
        hvac_enforce_instantaneous_cap=args.hvac_instantaneous_cap == "on",
        hvac_degree_hour_limit_c_hours=args.hvac_degree_hour_limit_c_hours,
        hvac_max_deviation_c=args.hvac_max_deviation_c,
        hvac_degree_hour_slack_weight=args.hvac_degree_hour_slack_weight,
        hvac_enforce_cycle_limit=args.hvac_cycle_limit == "on",
        hvac_enforce_dwell_constraints=args.hvac_dwell_constraints == "on",
        hvac_max_cycles=args.hvac_max_cycles,
        hvac_dwell_on_minutes=args.hvac_dwell_on_minutes,
        hvac_dwell_off_minutes=args.hvac_dwell_off_minutes,
        hvac_gain_regime=hvac_gain_regime,
    )
    summary = SingleHomeHEMSSimulation(
        config,
        home,
        HEMSController(device_controllers=device_controllers, price_schedule=rate_selection.schedule),
        water_heater_sensor_layout=sensor_layout,
        hvac_gain_regime=hvac_gain_regime,
        water_heater_hp_only=(args.water_heater_hp_only or args.water_heater_model.startswith("rc")),
        initial_temperature_c=args.initial_temperature_c,
        water_draw_file=args.water_draw_file,
        water_draw_start=args.water_draw_start,
        weather_archive=weather_archive,
        save_optimizer_plan_figures=not args.skip_optimizer_plan_figures,
    ).run()
    if args.rate_structure != "legacy":
        price = summary.controls["electricity_price_usd_per_kwh"]
        demand_rate = rate_selection.schedule.demand_charge_usd_per_kw_day
        anytime_demand_rate = float(getattr(rate_selection.schedule, "anytime_demand_charge_usd_per_kw_day", 0))
        bills = {}
        for name, data in (
            ("controlled", summary.standardized_results),
            ("uncontrolled", summary.uncontrolled_results),
        ):
            power = data["total_electric_power_kw"].astype(float)
            aligned_price = price.reindex(power.index)
            if aligned_price.isna().any():
                raise ValueError("Tariff prices do not cover the realized home power")
            energy_cost = float((power * aligned_price).sum() / 60)
            demand_cost = 0.0
            if demand_rate > 0 or anytime_demand_rate > 0:
                demand_bill = winter_daily_demand_bill(
                    power, rate_usd_per_kw_day=demand_rate,
                    anytime_rate_usd_per_kw_day=anytime_demand_rate,
                )
                demand_cost = demand_bill.total_charge_usd
                demand_bill.per_day.to_csv(
                    config.output_directory / "data" / f"{home.home_id}_{name}_daily_demand_bill.csv"
                )
            bills[name] = {
                "energy_charge_usd": energy_cost,
                "demand_charge_usd": demand_cost,
                "total_bill_usd": energy_cost + demand_cost,
            }
        result = {
            "rate_structure": rate_selection.metadata,
            "demand_controller_treatment": "billing_only; existing controller objectives unchanged",
            "bills": bills,
        }
        (config.output_directory / "rate_structure_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

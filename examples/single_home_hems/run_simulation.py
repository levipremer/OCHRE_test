#!/usr/bin/env python3
"""Run one selectable five-home-block dwelling with a stepwise HEMS."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
FIVE_HOME_CONFIG = REPO_ROOT / "examples" / "five_home_block" / "config"
HOME_IDS = tuple(f"home_{number:02d}" for number in range(1, 6))
CONTROLLER_CHOICES = ("none", "water-heater", "hvac", "ev", "all")


def default_output_subdirectory(
    controller: str, optimization: str, model: str,
    forecast_type: str = "perfect", scenario_method: str = "daily_budget_high_rf_1",
) -> str:
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
from examples.single_home_hems.single_home_hems.training import (
    HVACInternalGainRegime,
    WaterHeaterSensorLayout,
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
        controllers.append(HVACPredictiveController())
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
        "--scenario-method", "--scenario-forecast-type",
        choices=("daily_budget_high_rf_1", "daily_budget_high"),
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
        help="Prediction horizon in hours (scenario mode requires 12)",
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
        help="Default: outputs/HOME_ID/CONTROLLER_OPTIMIZATION_MODEL",
    )
    parser.add_argument(
        "--time-step-minutes",
        type=float,
        help="Override the timestep inherited from the five-home block",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        help="Override simulation duration; 48 hours is useful for EV tests",
    )
    parser.add_argument(
        "--start-time",
        type=dt.datetime.fromisoformat,
        help="Override the simulation start using an ISO timestamp",
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
    if args.water_heater_horizon_hours <= 0:
        raise ValueError("--water-heater-horizon-hours must be positive")
    if args.forecast_type == "scenario":
        if args.controller not in {"water-heater", "all"}:
            raise ValueError("Scenario forecasts require a water-heater controller")
        if args.water_heater_horizon_hours != 12:
            raise ValueError("Scenario forecasts require a 12-hour horizon")
        if args.water_draw_file is None:
            args.water_draw_file = EXAMPLE_ROOT / "external_data" / "water_flow_dchouse_l_per_min.csv"
    if args.water_draw_start is not None and args.water_draw_file is None:
        raise ValueError("--water-draw-start requires --water-draw-file")
    if args.water_heater_solver == "CLARABEL" and args.water_heater_optimization == "milp":
        raise ValueError("CLARABEL supports convex optimization only; use GUROBI or SCIPY for MILP")
    config = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    home = select_home(config, args.home_id)

    if args.start_time is not None:
        config = replace(config, start_time=args.start_time)
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

    default_controller_directory = default_output_subdirectory(
        args.controller, args.water_heater_optimization, args.water_heater_model,
        args.forecast_type, args.scenario_method,
    )
    output_directory = (
        args.output_directory
        or EXAMPLE_ROOT / "outputs" / args.home_id / default_controller_directory
    )
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    config = replace(
        config,
        output_directory=output_directory.resolve(),
        homes=(home,),
    )
    config.validate()

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
    )
    hvac_gain_regime = HVACInternalGainRegime(
        day_start_hour=args.internal_gain_day_start_hour,
        day_end_hour=args.internal_gain_day_end_hour,
    )
    SingleHomeHEMSSimulation(
        config,
        home,
        HEMSController(device_controllers=device_controllers),
        water_heater_sensor_layout=sensor_layout,
        hvac_gain_regime=hvac_gain_regime,
        water_heater_hp_only=(args.water_heater_hp_only or args.water_heater_model.startswith("rc")),
        initial_temperature_c=args.initial_temperature_c,
        water_draw_file=args.water_draw_file,
        water_draw_start=args.water_draw_start,
        save_optimizer_plan_figures=not args.skip_optimizer_plan_figures,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

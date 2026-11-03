#!/usr/bin/env python3
"""Run the joint 8-hour HPWH scenario MILP and HVAC degree-hour MILP.

The required choice is the simulation duration. The defaults reproduce the
selected device formulations with seven six-hour RMSE-plus-volume water paths
and an RF correction to only the first forecast interval.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
for local_path in (REPO_ROOT, EXAMPLE_ROOT):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.run_simulation import (
    HOME_IDS, _local_weather_time, load_weather_forecast_archive,
    next_available_output_directory,
)
from examples.single_home_hems.single_home_hems.controllers.hvac import (
    HVACPredictiveController,
)
from examples.single_home_hems.single_home_hems.controllers.joint_hems import (
    JointHEMSPredictiveController,
)
from examples.single_home_hems.single_home_hems.controllers.water_heater import (
    DEFAULT_RC_MODEL_PATH, WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems.controllers.water_heater_variants import (
    add_hpwh_variant_arguments, effective_hpwh_variant, hpwh_options_from_arguments,
)
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation, select_home,
)
from examples.single_home_hems.single_home_hems.rest_of_home_forecast import (
    RestOfHomeHybridForecast,
)


HORIZON = pd.Timedelta(hours=8)
MODEL_STEP = pd.Timedelta(minutes=15)
DEFAULT_WATER_DRAW_FILE = EXAMPLE_ROOT / "external_data/water_flow_dchouse_l_per_min.csv"
DEFAULT_CONFIG_DIRECTORY = REPO_ROOT / "examples/five_home_block/config"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_argument_group("Run")
    run.add_argument("--home-id", choices=HOME_IDS, default="home_01")
    run.add_argument("--start-time", "--start-date", type=dt.datetime.fromisoformat)
    run.add_argument("--duration-hours", type=float, required=True,
                     help="Hours after native warm-up: 16 hours for persistence, 24 for rolling-starts")
    run.add_argument("--source-config-directory", type=Path,
                     default=DEFAULT_CONFIG_DIRECTORY)
    run.add_argument("--output-directory", type=Path,
                     help="New output directory; the default receives a numbered suffix if needed")
    run.add_argument("--skip-optimizer-plan-figures", action="store_true",
                     help="Keep plan CSVs but skip per-solve plan figures")
    run.add_argument(
        "--rest-rf-inference-gap-policy", choices=("strict", "causal_short_gap"),
        default="strict",
        help="RF inference only: strict original inputs, or approved causal short-gap fills with one-minute endpoint hold",
    )

    water = parser.add_argument_group("Water-draw scenarios and HPWH")
    water.add_argument("--water-draw-file", type=Path, default=DEFAULT_WATER_DRAW_FILE)
    water.add_argument("--water-draw-start", type=dt.datetime.fromisoformat,
                       default=dt.datetime(2025, 11, 6),
                       help="Source-data timestamp corresponding to simulation start")
    water.add_argument("--scenario-count", type=int, choices=(5, 7, 10), default=7,
                       help="Number of equally probable six-hour RMSE-plus-volume paths (default: 7)")
    add_hpwh_variant_arguments(water)

    hvac = parser.add_argument_group("HVAC degree-hour MILP")
    hvac.add_argument("--hvac-comfort-band-c", type=float, default=1.5)
    hvac.add_argument("--hvac-persistence-weight", type=float, default=0.1)
    hvac.add_argument("--hvac-degree-hour-limit-c-hours", type=float, default=2.0)
    hvac.add_argument("--hvac-degree-hour-slack-weight", type=float, default=100.0)
    hvac.add_argument("--hvac-max-deviation-c", type=float, default=1.0)
    hvac.add_argument(
        "--weather-forecast-mode", choices=("perfect", "predict"), default="perfect",
        help="HVAC future-weather source (default: perfect)",
    )
    hvac.add_argument(
        "--weather-forecast-file", type=Path,
        help="Prepared 00Z GFS forecast Parquet; overrides weather_forecast_file in source config",
    )

    peak = parser.add_argument_group("Modeled home peak (excluding EV)")
    peak.add_argument(
        "--peak-weight-usd-per-kw", type=float, default=0.005,
        help="Weight on the maximum predicted rest-of-home + HVAC + HPWH power; 0 is the baseline",
    )
    peak.add_argument(
        "--peak-window", choices=("all", "tou", "night"), default="all",
        help="Penalized interval starts: all horizon steps, TOU [15:00,18:00), or night [03:00,07:00); zero peak cost without overlap",
    )

    solver = parser.add_argument_group("Joint solver")
    solver.add_argument("--time-limit-seconds", type=float, default=300.0)
    solver.add_argument("--mip-gap", type=float, default=0.05)
    solver.add_argument("--solver-threads", type=int, default=1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def build_controller(args: argparse.Namespace) -> JointHEMSPredictiveController:
    """Use the developed device formulations with one common 8-hour horizon."""
    solver_options = {
        "TimeLimit": args.time_limit_seconds,
        "MIPGap": args.mip_gap,
        "Threads": args.solver_threads,
    }
    water = WaterHeaterPredictiveController(
        model_path=DEFAULT_RC_MODEL_PATH,
        model_kind="rc",
        optimization_kind="milp",
        forecast_type="scenario",
        scenario_method=f"rmse_volume_6h_k{args.scenario_count}_rf_1",
        horizon=HORIZON,
        model_timestep=MODEL_STEP,
        solver="GUROBI",
        solver_options=solver_options,
        minimum_temperature_c=38.0,
        **hpwh_options_from_arguments(args),
    )
    heating = HVACPredictiveController(
        horizon=HORIZON,
        model_timestep=MODEL_STEP,
        optimization_kind="mixed integer",
        solver="GUROBI",
        solver_options=solver_options,
        comfort_band_c=args.hvac_comfort_band_c,
        comfort_weight=0.0,
        persistence_weight=args.hvac_persistence_weight,
        enforce_degree_hour_budget=True,
        enforce_instantaneous_cap=True,
        degree_hour_limit_c_hours=args.hvac_degree_hour_limit_c_hours,
        max_deviation_c=args.hvac_max_deviation_c,
        degree_hour_slack_weight=args.hvac_degree_hour_slack_weight,
        max_deviation_slack_weight=10_000.0,
        enforce_cycle_limit=False,
        enforce_dwell_constraints=False,
    )
    return JointHEMSPredictiveController(
        water, heating, solver_options=solver_options,
        peak_weight_usd_per_kw=args.peak_weight_usd_per_kw,
        peak_window=args.peak_window,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.duration_hours < float("inf"):
        raise ValueError("--duration-hours must be finite and positive")
    for name in (
        "hvac_comfort_band_c", "hvac_persistence_weight",
        "hvac_degree_hour_limit_c_hours", "hvac_degree_hour_slack_weight",
        "hvac_max_deviation_c",
        "peak_weight_usd_per_kw",
    ):
        value = getattr(args, name)
        if not 0 <= value < float("inf"):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    hpwh_options_from_arguments(args)
    if not 0 < args.time_limit_seconds < float("inf"):
        raise ValueError("--time-limit-seconds must be finite and positive")
    if not 0 <= args.mip_gap < float("inf") or args.solver_threads < 1:
        raise ValueError("--mip-gap and --solver-threads are invalid")

    config = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    home = select_home(config, args.home_id)
    stem = f"joint_milp_rmse_volume_6h_k{args.scenario_count}_rf_1_8h"
    if args.hpwh_variant == "rolling-starts":
        stem += "_hpwh_rolling_starts"
    output = args.output_directory or EXAMPLE_ROOT / "outputs" / home.home_id / stem
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    if args.output_directory is None:
        output = next_available_output_directory(output)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output}")
    config = replace(
        config,
        start_time=(
            _local_weather_time(args.start_time, config.weather_metadata)
            if args.start_time is not None else config.start_time
        ),
        duration=dt.timedelta(hours=args.duration_hours),
        time_step=dt.timedelta(minutes=1),
        output_directory=output,
        homes=(home,),
    )
    config.validate()
    weather_archive = load_weather_forecast_archive(
        config, args.weather_forecast_mode, args.weather_forecast_file
    )
    if weather_archive is not None:
        start_time = pd.Timestamp(config.start_time)
        weather_archive.validate_coverage(
            start_time, start_time + pd.Timedelta(config.duration), HORIZON, MODEL_STEP
        )
    output.mkdir(parents=True, exist_ok=True)
    controller = build_controller(args)
    progress_started = time.monotonic()
    progress_statuses: Counter[str] = Counter()
    progress_path = output / "joint_progress.json"

    def record_progress(row: dict) -> None:
        progress_statuses[str(row["status"])] += 1
        progress = {
            **row,
            "run_state": "optimizing",
            "duration_hours": args.duration_hours,
            "elapsed_wall_seconds": time.monotonic() - progress_started,
            "status_counts": dict(progress_statuses),
        }
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2, default=str) + "\n")
        temporary.replace(progress_path)

    controller.progress_callback = record_progress
    rest_forecast = None
    if home.rest_of_home.source == "dc_house_test":
        rest_forecast = RestOfHomeHybridForecast.from_csv(
            home.rest_of_home.data_file,
            simulation_start=config.start_time,
            source_start=home.rest_of_home.source_start,
            history_days=20,
            inference_gap_policy=args.rest_rf_inference_gap_policy,
        )
    elif args.peak_weight_usd_per_kw > 0:
        raise ValueError(
            "The peak objective requires DC House rest-of-home data for this home; "
            "use --peak-weight-usd-per-kw 0 for the existing device-only objective"
        )
    settings = {
        "controller": "joint HPWH scenario MILP + HVAC degree-hour MILP",
        "hpwh_variant": effective_hpwh_variant(args.hpwh_variant, controller.water_heater_controller),
        "water_scenario_method": controller.water_heater_controller.scenario_method,
        "scenario_count": args.scenario_count,
        "forecast_selection_hours_preceding": 6,
        "rf_first_step_only": True,
        "scenario_probabilities": "equal",
        "hpwh_comfort_aggregation": "maximum of per-scenario horizon totals",
        "hpwh_first_action_shared": True,
        "hvac_trajectory_shared_across_water_scenarios": True,
        "horizon_hours": 8,
        "weather_forecast_mode": args.weather_forecast_mode,
        "weather_forecast_file": (
            str((REPO_ROOT / (args.weather_forecast_file or config.weather_forecast_file)).resolve())
            if weather_archive is not None else None
        ),
        "prediction_step_minutes": 15,
        "hvac_degree_hour_history_hours": 16,
        "electricity_objective": "0.25/32 * sum_t price_t * (hvac_power_t + mean_s hpwh_power_s_t)",
        "rest_of_home_forecast_method": (
            rest_forecast.method_label if rest_forecast is not None else None
        ),
        "rest_of_home_forecast_metadata": (
            rest_forecast.metadata if rest_forecast is not None else None
        ),
        "peak_power_objective_enabled": args.peak_weight_usd_per_kw > 0,
        "peak_power_weight_usd_per_kw": args.peak_weight_usd_per_kw,
        "peak_power_window": args.peak_window,
        "peak_power_window_start_hour": controller.peak_window_hours[0],
        "peak_power_window_end_hour": controller.peak_window_hours[1],
        "peak_power_window_interval_convention": "local interval start; start inclusive, end exclusive",
        "peak_power_empty_window_cost": 0.0,
        "peak_power_formula": (
            "weight * z; z >= rest_of_home_forecast_t + hvac_power_t + "
            "hpwh_power_s_t for every eligible 15-minute step t and water-use scenario s; "
            "eligible steps start within the selected daily window; zero cost without overlap"
        ),
        "peak_power_excludes_ev": True,
        "solver": "GUROBI",
        "arguments": {name: str(value) if isinstance(value, Path) else value
                      for name, value in vars(args).items()},
    }
    (output / "joint_run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Joint HEMS output: {output}", flush=True)
    SingleHomeHEMSSimulation(
        config,
        home,
        HEMSController(device_controllers=(controller,)),
        water_heater_hp_only=True,
        initial_temperature_c=21.0,
        water_draw_file=args.water_draw_file,
        water_draw_start=pd.Timestamp(args.water_draw_start),
        rest_of_home_forecast=rest_forecast,
        weather_archive=weather_archive,
        save_optimizer_plan_figures=not args.skip_optimizer_plan_figures,
    ).run()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        progress["run_state"] = "complete"
        progress["elapsed_wall_seconds"] = time.monotonic() - progress_started
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

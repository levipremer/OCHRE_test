#!/usr/bin/env python3
# ruff: noqa: E402
"""Eight-hour HVAC, HPWH, and bidirectional EV joint MILP replay.

The non-V2H reference supplies the EV's fixed backed-load cost reference.
The split-Hub comparison uses frozen pretest RFs, rolling p90 tails, five EV
scenarios, and one soft whole-home peak threshold. Scenario 3 adds separate
soft grid-import and gross-panel envelopes to the combined controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
for local_path in (REPO_ROOT, EXAMPLE_ROOT):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from examples.five_home_block.five_home_block.config import load_configuration
from examples.five_home_block.five_home_block.ev_schedule import build_ev_schedules
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.run_joint_hems import (
    DEFAULT_CONFIG_DIRECTORY,
    DEFAULT_WATER_DRAW_FILE,
    build_controller as build_thermal_joint_controller,
    parse_args as parse_thermal_joint_args,
)
from examples.single_home_hems.run_simulation import (
    _local_weather_time, load_weather_forecast_archive,
    next_available_output_directory,
)
from examples.single_home_hems.single_home_hems.backed_loads import (
    HOME_HUB_BACKED_LOADS, backed_power,
)
from examples.single_home_hems.single_home_hems.control_config import load_unified
from examples.single_home_hems.single_home_hems.controllers.ev import EVScenarioController
from examples.single_home_hems.single_home_hems.controllers.water_heater_variants import (
    add_hpwh_variant_arguments, effective_hpwh_variant, hpwh_options_from_arguments,
)
from examples.single_home_hems.single_home_hems.controllers.whole_home_joint import (
    DOE_GRID_PENALTY_USD_PER_KWH, DOE_PANEL_PENALTY_USD_PER_KWH,
    WHOLE_HOME_WINDOWS, WINTER_TOU_PEAK_WINDOW, WholeHomeJointPredictiveController,
)
from examples.single_home_hems.single_home_hems.doe_compliance import evaluate_scenario3_doe
from examples.single_home_hems.single_home_hems.ev_scenarios import ScenarioEVForecast
from examples.single_home_hems.single_home_hems.frozen_split_hub_forecast import (
    FrozenSplitHubForecast,
)
from examples.single_home_hems.single_home_hems.home_hub_replay import (
    PreparedHomeHubReplay, prepare_home_hub_replay,
)
from examples.single_home_hems.single_home_hems.joint_approx_resume import (
    JointApproximateReplay, complete_replay_audit, read_partial_raw,
)
from examples.single_home_hems.single_home_hems.rest_of_home_forecast import (
    RestOfHomeHybridForecast,
)
from examples.single_home_hems.single_home_hems.rate_billing import winter_daily_demand_bill
from examples.single_home_hems.single_home_hems.rate_setup import (
    RATE_CHOICES, select_rate_for_simulation,
)
from examples.single_home_hems.single_home_hems.rates import (
    MISO_RTP_DOE, WINTER_TOU_DEMAND_CPP, WinterTOURate,
)
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    select_home,
)


HORIZON = pd.Timedelta(hours=8)
DOE_GRID_LIMIT_KW = 15.0
DOE_PANEL_LIMIT_KW = 38.0
DEFAULT_START = dt.datetime(2018, 1, 1, 3)
# This source day has a complete 03:00-12:00 replay for the one-hour pilot and
# its eight-hour non-V2H forecast tail.
DEFAULT_REST_SOURCE_START = "2026-09-19"
DEFAULT_EV_HOME_CONFIG = EXAMPLE_ROOT / "config/unified_home.json"
DEFAULT_EV_CONTROL_CONFIG = EXAMPLE_ROOT / "config/unified_control.json"
BACKED_LOADS = ("heat_pump", "water_heater")
SPLIT_WARMUP = pd.Timedelta(hours=16)
SPLIT_SOURCE_TAIL = pd.Timedelta(hours=8)
PREDICTIVE_INPUTS = EXAMPLE_ROOT / "inputs/predictive"
DEFAULT_HUB_OTHER_SOURCE = PREDICTIVE_INPUTS / "forecasts/hub_other_forecast_source.csv"
DEFAULT_OUTSIDE_HUB_SOURCE = PREDICTIVE_INPUTS / "forecasts/outside_hub_forecast_source.csv"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home-id", default="home_01")
    parser.add_argument("--start-time", type=dt.datetime.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--rest-source-start", default=DEFAULT_REST_SOURCE_START)
    parser.add_argument("--rest-rf-inference-gap-policy", choices=("strict", "causal_short_gap"), default="strict")
    parser.add_argument("--rate-structure", choices=RATE_CHOICES, default="legacy",
                        help="Electricity tariff choice; legacy keeps the original pilot rate")
    parser.add_argument("--peak-weight-usd-per-kw", type=float,
                        help="Legacy pilot peak weight (default 0.1); selected rates set their own objective")
    parser.add_argument("--peak-limit-kw", type=float,
                        help="Soft threshold; enables normalized worst-positive-exceedance objective")
    parser.add_argument("--peak-window", choices=("night", "all", "evening"),
                        help="Legacy peak window (default night); selected rates set their own windows")
    parser.add_argument("--split-hub-forecast", action="store_true",
                        help="Use separate frozen Hub-other/Outside-Hub forecasts and split plant replay")
    parser.add_argument("--hub-other-source", type=Path, default=DEFAULT_HUB_OTHER_SOURCE)
    parser.add_argument("--outside-hub-source", type=Path, default=DEFAULT_OUTSIDE_HUB_SOURCE)
    parser.add_argument("--ev-scenario-count", type=int, default=7)
    parser.add_argument("--ev-scenario-method", choices=("knn_state", "tod_dow_state_hier",
                                                       "tod_dow_state_hier_limit_persistence"),
                        default="tod_dow_state_hier_limit_persistence")
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--source-config-directory", type=Path, default=DEFAULT_CONFIG_DIRECTORY)
    parser.add_argument("--ev-home-config", type=Path, default=DEFAULT_EV_HOME_CONFIG)
    parser.add_argument("--ev-control-config", type=Path, default=DEFAULT_EV_CONTROL_CONFIG)
    parser.add_argument("--water-draw-file", type=Path, default=DEFAULT_WATER_DRAW_FILE)
    parser.add_argument("--water-draw-start", type=dt.datetime.fromisoformat, default=dt.datetime(2025, 11, 6, 3))
    add_hpwh_variant_arguments(parser.add_argument_group("HPWH variant"))
    parser.add_argument(
        "--weather-forecast-mode", choices=("perfect", "predict"), default="perfect",
        help="HVAC future-weather source (default: perfect)",
    )
    parser.add_argument(
        "--weather-forecast-file", type=Path,
        help="Prepared 00Z GFS forecast Parquet; overrides weather_forecast_file in source config",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--approx-resume-from", type=Path,
                        help="Replay recorded controls from a failed partial run before resuming MPC")
    return parser


def _validated_args(argv: list[str] | None) -> argparse.Namespace:
    args = build_argument_parser().parse_args(argv)
    if not np.isfinite(args.duration_hours) or args.duration_hours <= 0:
        raise ValueError("--duration-hours must be finite and positive")
    if args.duration_hours * 4 != round(args.duration_hours * 4):
        raise ValueError("--duration-hours must contain whole 15-minute control intervals")
    start_clock = pd.Timestamp(args.start_time)
    if start_clock.tzinfo is not None:
        start_clock = start_clock.tz_convert("UTC")
    if start_clock != start_clock.floor("15min"):
        raise ValueError("--start-time must align with the 15-minute control grid")
    if args.rate_structure != "legacy" and args.peak_weight_usd_per_kw is not None:
        raise ValueError("Selected rates set their own peak objective; omit --peak-weight-usd-per-kw")
    if args.peak_weight_usd_per_kw is not None and (
        not np.isfinite(args.peak_weight_usd_per_kw) or args.peak_weight_usd_per_kw < 0
    ):
        raise ValueError("--peak-weight-usd-per-kw must be finite and nonnegative")
    if args.rate_structure == "legacy" and args.peak_weight_usd_per_kw is None:
        args.peak_weight_usd_per_kw = 0.1
    if args.rate_structure == "legacy":
        args.peak_window = args.peak_window or "night"
    elif args.peak_window is not None:
        raise ValueError("Selected rates set their own peak windows; omit --peak-window")
    elif args.rate_structure == MISO_RTP_DOE:
        # The generic peak objective is inactive; the DOE mask is set separately.
        args.peak_window = "night"
    else:
        args.peak_window = WINTER_TOU_PEAK_WINDOW
    if args.peak_limit_kw is not None and (
        not np.isfinite(args.peak_limit_kw) or args.peak_limit_kw <= 0
    ):
        raise ValueError("--peak-limit-kw must be finite and positive")
    if args.rate_structure == MISO_RTP_DOE and args.peak_limit_kw is not None:
        raise ValueError("rtp-miso+doe already sets the 15 kW grid and 38 kW panel limits")
    if args.ev_scenario_count not in (5, 7):
        raise ValueError("--ev-scenario-count must be five or seven")
    if not np.isfinite(args.time_limit_seconds) or args.time_limit_seconds <= 0:
        raise ValueError("--time-limit-seconds must be finite and positive")
    if not np.isfinite(args.mip_gap) or not 0 <= args.mip_gap <= 1:
        raise ValueError("--mip-gap must be a relative fraction between zero and one")
    if args.solver_threads < 1:
        raise ValueError("--solver-threads must be positive")
    hpwh_options_from_arguments(args)
    return args


def _ev_inputs(args: argparse.Namespace):
    """Use the existing EV physical model and the established milp_health costs."""
    unified = load_unified(args.ev_home_config, args.ev_control_config, REPO_ROOT)
    ev_home = select_home(unified.neighborhood, args.home_id)
    if len(ev_home.evs) != 1:
        raise ValueError("The initial whole-home pilot requires exactly one EV")
    ev = ev_home.evs[0]
    if ev.hems is None:
        raise ValueError("The initial whole-home pilot requires the opted-in EV model")
    options = replace(
        ev.hems,
        controller="mpc",
        v2h=True,
        formulation="milp",
        solver="GUROBI",
        solver_options={},
        solver_time_limit_seconds=args.time_limit_seconds,
        horizon_hours=8,
        control_minutes=15,
        electricity_weight=1,
        comfort_weight=1000,
        health_weight=1,
        health_soc_weight=1000,
        health_soc_min=0.2,
        health_soc_max=0.8,
        comfort_soc_min=0.4,
        persistence_weight=0,
        enforce_terminal_soc=True,
        enforce_cycle_limit=False,
        enforce_dwell_constraints=False,
        enforce_discharge_budget=False,
        enable_degradation=False,
        degradation_method="none",
        mip_relative_gap=args.mip_gap,
        history_days=max(ev.hems.history_days, 31),
        forecast_margin_hours=8,
    )
    ev = replace(ev, hems=options)
    control = dict(unified.controls[args.home_id]["evs"][ev.vehicle_id])
    control.update(horizon_hours=8, forecast_hours=8, control_minutes=15)
    control["forecast"] = {
        **control["forecast"], "mode": "scenario",
        "scenario_count": getattr(args, "ev_scenario_count", 7),
        "scenario_method": getattr(args, "ev_scenario_method", "tod_dow_state_hier_limit_persistence"),
    }
    return ev, control


def _thermal_controller(args: argparse.Namespace):
    # The existing builder is the source of truth for HVAC/HPWH formulations.
    thermal_args = parse_thermal_joint_args([
        "--duration-hours", str(args.duration_hours),
        "--scenario-count", "7",
        "--peak-weight-usd-per-kw", "0",  # one combined peak term is added below
        "--peak-window", "night",
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--mip-gap", str(args.mip_gap),
        "--solver-threads", str(args.solver_threads),
        "--hpwh-variant", getattr(args, "hpwh_variant", "persistence"),
    ])
    for name in ("hpwh_comfort_weight", "hpwh_legionella_weight", "hpwh_persistence_weight"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(thermal_args, name, value)
    return build_thermal_joint_controller(thermal_args)


def hpwh_warmup_duration(args: argparse.Namespace) -> pd.Timedelta:
    """Keep the existing prelude unless the selected HPWH needs a full day."""
    return (pd.Timedelta(hours=24)
            if hpwh_options_from_arguments(args)["rolling_start_limit_24h"] is not None
            else SPLIT_WARMUP)


def validate_split_warmup_window(source_start, source_end, warmup) -> None:
    """Reject an unavailable rolling-start prelude before creating run files."""
    if pd.Timedelta(warmup) <= SPLIT_WARMUP:
        return
    from examples.five_home_block.five_home_block.rest_of_home import TEST_START, TEST_END
    from examples.single_home_hems.single_home_hems.home_hub_replay import (
        HOME_HUB_REPLAY_START, HOME_HUB_REPLAY_END,
    )
    first = pd.Timestamp(source_start) - pd.Timedelta(warmup)
    last = pd.Timestamp(source_end)
    valid = (TEST_START <= first < last <= TEST_END
             or HOME_HUB_REPLAY_START <= first < last <= HOME_HUB_REPLAY_END)
    if not valid:
        raise ValueError(
            "The rolling-starts HPWH variant requires a complete 24-hour native history. "
            f"The requested split-Hub source interval [{first}, {last}) exceeds the approved replay windows. "
            "Choose a source start with 24 hours of available history (for the October window, "
            "2026-10-05 instead of 2026-10-04 is an example), or select --hpwh-variant persistence. "
            "History is never truncated or padded."
        )


def _metrics(
    data: pd.DataFrame, price: pd.Series, demand_rate_usd_per_kw_day: float = 0.0,
    *, anytime_demand_rate_usd_per_kw_day: float = 0.0,
    demand_window_step_minutes: int | None = None,
) -> dict[str, float]:
    total = data["total_electric_power_kw"].astype(float)
    price = price.reindex(total.index).astype(float)
    if price.isna().any():
        raise ValueError("Tariff series does not cover the measured replay")
    bin_means = total.resample("15min").mean()
    soc_columns = [name for name in data if name.startswith("ev_") and name.endswith("_soc_fraction")
                   and "end_soc" not in name]
    soc_name = "ev_soc_fraction" if "ev_soc_fraction" in data else (
        soc_columns[0] if len(soc_columns) == 1 else None
    )
    if soc_name is None:
        raise ValueError("Expected one EV SOC series in standardized results")
    result = {
        "energy_kwh": float(total.sum() / 60),
        "electricity_cost_usd": float((total * price).sum() / 60),
        "one_minute_peak_kw": float(total.max()),
        "aligned_15_minute_peak_kw": float(bin_means.max()),
        "ev_net_energy_kwh": float(data["ev_power_kw"].sum() / 60),
        "ev_charge_energy_kwh": float(data["ev_power_kw"].clip(lower=0).sum() / 60),
        "ev_discharge_energy_kwh": max(0.0, float(-data["ev_power_kw"].clip(upper=0).sum() / 60)),
        "ev_soc_start": float(data[soc_name].iloc[0]),
        "ev_soc_end": float(data[soc_name].iloc[-1]),
    }
    demand_bill = (
        winter_daily_demand_bill(
            total, rate_usd_per_kw_day=demand_rate_usd_per_kw_day,
            anytime_rate_usd_per_kw_day=anytime_demand_rate_usd_per_kw_day,
            window_step_minutes=demand_window_step_minutes,
        )
        if demand_rate_usd_per_kw_day > 0 or anytime_demand_rate_usd_per_kw_day > 0
        else None
    )
    demand_cost = demand_bill.total_charge_usd if demand_bill is not None else 0.0
    result["on_peak_demand_charge_usd"] = demand_bill.on_peak_charge_usd if demand_bill is not None else 0.0
    result["anytime_demand_charge_usd"] = demand_bill.anytime_charge_usd if demand_bill is not None else 0.0
    result["demand_charge_usd"] = demand_cost
    result["total_bill_usd"] = result["electricity_cost_usd"] + demand_cost
    return result


def _effective_peak_weight(args: argparse.Namespace, rate_selection) -> float:
    """Return the compatibility on-peak coefficient for the selected rate.

    Revised Grid 2 constructs both daily demand terms in the joint controller.
    """
    if args.rate_structure == "legacy":
        return float(args.peak_weight_usd_per_kw)
    return float(rate_selection.schedule.demand_charge_usd_per_kw_day)


def _ev_boundary_energy(raw_path: Path, ev_name: str,
                        start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    """Read actual chemical energy, including measured capacity fade."""
    first_column = f"{ev_name} Start Stored Energy (kWh)"
    last_column = f"{ev_name} End Stored Energy (kWh)"
    raw = pd.read_csv(raw_path, usecols=["Time", first_column, last_column])
    raw_times = raw.pop("Time")
    if pd.Timestamp(start).tzinfo is not None:
        if pd.Timestamp(raw_times.iloc[0]).tzinfo is None:
            raise ValueError("Timezone-aware replay requires timezone-aware EV result timestamps")
        raw.index = pd.DatetimeIndex(
            pd.to_datetime(raw_times, utc=True).dt.tz_convert(pd.Timestamp(start).tz),
            name="Time",
        )
    else:
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw_times), name="Time")
    window = raw.loc[(raw.index >= start) & (raw.index < end)]
    if window.empty:
        raise ValueError(f"No EV boundary observations in {raw_path}")
    return float(window.iloc[0][first_column]), float(window.iloc[-1][last_column])


def _daily_average_tariff(price_at, timestamp: pd.Timestamp) -> float:
    day = pd.Timestamp(timestamp).normalize()
    minutes = pd.date_range(day, day + pd.Timedelta(days=1), freq="min", inclusive="left")
    return float(np.mean([price_at(minute) for minute in minutes]))


def _map_split_to_simulation(
    prepared: PreparedHomeHubReplay, simulation_start: pd.Timestamp,
    source_start: pd.Timestamp,
) -> pd.DataFrame:
    """Map the approved physical split by one fixed local-clock offset."""
    simulation_start = pd.Timestamp(simulation_start)
    source_start = pd.Timestamp(source_start)
    if source_start.tzinfo is not None:
        raise ValueError("Split replay source clock must be naive local time")
    frame = prepared.power_kw.copy()
    wall = (simulation_start.tz_localize(None)
            if simulation_start.tzinfo is not None else simulation_start)
    mapped_wall = pd.DatetimeIndex(wall + (frame.index - source_start))
    mapped = (mapped_wall.tz_localize(simulation_start.tz, ambiguous="raise", nonexistent="raise")
              if simulation_start.tzinfo is not None else mapped_wall)
    frame.index = pd.DatetimeIndex(mapped, name="timestamp")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Mapped split replay clock must be unique and increasing")
    return frame


def _threshold_metrics(data: pd.DataFrame, limit_kw: float, window: str) -> dict:
    power = data["total_electric_power_kw"].astype(float)
    interval_minutes = 30 if window == WINTER_TOU_PEAK_WINDOW else 15
    grouped = power.resample(f"{interval_minutes}min", closed="left", label="left")
    mean, count = grouped.mean(), grouped.count()
    if window == WINTER_TOU_PEAK_WINDOW:
        active = pd.Series(
            np.fromiter(
                (WinterTOURate.is_on_peak(target) for target in mean.index),
                dtype=bool, count=len(mean),
            ),
            index=mean.index, dtype=bool,
        )
    else:
        first, last = WHOLE_HOME_WINDOWS[window]
        active = pd.Series(
            (mean.index.hour >= first) & (mean.index.hour < last), index=mean.index,
        )
    if not count.loc[active].eq(interval_minutes).all():
        raise ValueError(f"Threshold audit requires complete {interval_minutes}-minute active blocks")
    selected = mean.loc[active]
    exceedance = (selected - limit_kw).clip(lower=0)
    return {
        "threshold_kw": limit_kw,
        "window": window,
        "interval_minutes": interval_minutes,
        "active_blocks": int(active.sum()),
        "peak_in_active_window_kw": float(selected.max()) if not selected.empty else None,
        "worst_exceedance_kw": float(exceedance.max()) if not selected.empty else 0.0,
        "exceeding_blocks": int(exceedance.gt(0).sum()),
        "exceedance_energy_kwh": float(exceedance.sum() * interval_minutes / 60),
    }


def main(argv: list[str] | None = None) -> int:
    args = _validated_args(argv)
    warmup_duration = hpwh_warmup_duration(args)
    if args.split_hub_forecast:
        validate_split_warmup_window(
            args.rest_source_start,
            pd.Timestamp(args.rest_source_start) + pd.Timedelta(hours=args.duration_hours) + SPLIT_SOURCE_TAIL,
            warmup_duration,
        )
    started = time.perf_counter()
    source = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    thermal_home = select_home(source, args.home_id)
    if thermal_home.rest_of_home.source != "dc_house_test":
        raise ValueError("The initial pilot requires the DC House rest-of-home replay")
    ev, ev_control = _ev_inputs(args)
    home = replace(
        thermal_home,
        evs=(ev,),
        rest_of_home=replace(thermal_home.rest_of_home, source_start=args.rest_source_start),
    )
    stem = (f"whole_home_joint_{args.peak_window}_{args.peak_limit_kw:g}kw"
            if args.split_hub_forecast and args.peak_limit_kw is not None
            else "whole_home_joint_8h_milp_k7_night_w0p1")
    if args.rate_structure != "legacy":
        suffix = args.rate_structure.lower().replace("+", "_")
        stem = (stem + "_" + suffix if args.split_hub_forecast
                else "whole_home_joint_8h_milp_k7_" + suffix)
    if args.split_hub_forecast and args.rate_structure == MISO_RTP_DOE:
        stem = "whole_home_joint_split_hub_rtp_miso_doe"
    if args.hpwh_variant == "rolling-starts":
        stem += "_hpwh_rolling_starts"
    output = args.output_directory or EXAMPLE_ROOT / "outputs" / args.home_id / stem
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    if args.output_directory is None:
        output = next_available_output_directory(output)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output}")
    config = replace(
        source,
        start_time=_local_weather_time(args.start_time, source.weather_metadata),
        duration=dt.timedelta(hours=args.duration_hours),
        time_step=dt.timedelta(minutes=1),
        output_directory=output,
        homes=(home,),
    )
    config.validate()
    rate_selection = select_rate_for_simulation(args.rate_structure, config)
    effective_peak_weight = _effective_peak_weight(args, rate_selection)
    doe_selected = args.rate_structure == MISO_RTP_DOE
    if doe_selected:
        rate_selection.schedule.validate_coverage(
            pd.Timestamp(config.start_time) - warmup_duration,
            pd.Timestamp(config.start_time) + pd.Timedelta(config.duration) + HORIZON,
        )
    if args.split_hub_forecast and args.peak_limit_kw is None and not doe_selected:
        raise ValueError("The split-Hub comparison requires an explicit peak threshold")
    if (args.split_hub_forecast and args.rate_structure == "legacy"
            and args.peak_window not in ("all", "evening")):
        raise ValueError("The split-Hub comparison uses all-day or 16:00–22:00 peak windows")
    weather_archive = load_weather_forecast_archive(
        config, args.weather_forecast_mode, args.weather_forecast_file
    )
    if weather_archive is not None:
        start_time = pd.Timestamp(config.start_time)
        weather_archive.validate_coverage(
            start_time, start_time + pd.Timedelta(config.duration),
            HORIZON, pd.Timedelta(minutes=15),
        )
    output.mkdir(parents=True, exist_ok=True)
    solver_options = {
        "TimeLimit": args.time_limit_seconds,
        "MIPGap": args.mip_gap,
        "Threads": args.solver_threads,
    }
    thermal = _thermal_controller(args)
    ev_controller = EVScenarioController(ev)
    controller = WholeHomeJointPredictiveController(
        joint_controller=thermal,
        ev_controller=ev_controller,
        solver_options=solver_options,
        peak_weight_usd_per_kw=effective_peak_weight,
        rate_structure=args.rate_structure,
        peak_window=args.peak_window,
        peak_limit_kw=args.peak_limit_kw,
        doe_panel_limit_kw=DOE_PANEL_LIMIT_KW if doe_selected else None,
        doe_grid_limit_kw=DOE_GRID_LIMIT_KW if doe_selected else None,
        doe_panel_penalty_weight=DOE_PANEL_PENALTY_USD_PER_KWH,
        doe_grid_penalty_weight=DOE_GRID_PENALTY_USD_PER_KWH,
        decision_trace_directory=output / "data" / "joint_decision_traces",
    )
    partial_directory = None
    partial_raw = None
    if args.approx_resume_from is not None:
        partial_directory = args.approx_resume_from.resolve()
        old_settings = json.loads((partial_directory / "whole_home_joint_settings.json").read_text())
        selected_hpwh = effective_hpwh_variant(args.hpwh_variant, thermal.water_heater_controller)
        previous_hpwh = old_settings.get("hpwh_variant")
        if previous_hpwh is None:
            legacy_args = parse_thermal_joint_args(["--duration-hours", "1", "--hpwh-variant", "persistence"])
            legacy = build_thermal_joint_controller(legacy_args)
            previous_hpwh = effective_hpwh_variant("persistence", legacy.water_heater_controller)
        if previous_hpwh != selected_hpwh:
            raise ValueError("Approximate resume requires the same effective HPWH variant and weights")
        for key, expected in (
            ("simulation_start", args.start_time.isoformat()),
            ("duration_hours", args.duration_hours),
            ("rate_structure_choice", args.rate_structure),
            ("peak_window", args.peak_window),
            ("peak_limit_kw", args.peak_limit_kw),
            ("peak_weight_usd_per_kw", effective_peak_weight),
            ("grid2_demand_formulation", controller.grid2_demand_formulation
             if args.rate_structure == WINTER_TOU_DEMAND_CPP else None),
            ("grid2_demand_aggregation", "worst_complete_scenario_v1" if controller.uses_daily_grid2_demand else None),
            ("grid2_demand_window_step_minutes", 15 if args.rate_structure == WINTER_TOU_DEMAND_CPP else None),
            ("doe_grid_limit_kw", DOE_GRID_LIMIT_KW if doe_selected else None),
            ("doe_panel_limit_kw", DOE_PANEL_LIMIT_KW if doe_selected else None),
            ("doe_grid_penalty_weight", DOE_GRID_PENALTY_USD_PER_KWH if doe_selected else None),
            ("doe_panel_penalty_weight", DOE_PANEL_PENALTY_USD_PER_KWH if doe_selected else None),
            ("ev_scenarios", args.ev_scenario_count),
            ("ev_scenario_method", args.ev_scenario_method),
            ("solver_options", solver_options),
        ):
            previous = old_settings.get(
                key, "legacy" if key == "rate_structure_choice" else None,
            )
            if previous != expected:
                raise ValueError(f"Approximate resume mismatch for {key}: "
                                 f"{previous!r} != {expected!r}")
        if args.rate_structure != "legacy" and old_settings.get("rate_structure") != rate_selection.metadata:
            raise ValueError("Approximate resume rate schedule or CPP event differs from the source run")
        if (effective_peak_weight > 0 and args.peak_window == WINTER_TOU_PEAK_WINDOW
                and old_settings.get("peak_demand_interval_minutes") != 30):
            raise ValueError("Approximate resume requires the same 30-minute winter demand objective")
        partial_raw = read_partial_raw(partial_directory / "data/raw" / f"{home.home_id}.csv")
        if partial_raw.index[-1] >= pd.Timestamp(config.start_time) + config.duration:
            raise ValueError("Partial run already covers the full requested week")
        event_source = partial_directory / "whole_home_joint_solver_events.jsonl"
        original_events = [json.loads(line) for line in event_source.read_text().splitlines()]
        for row in original_events:
            controller._solve_history.append(row)
            controller.solve_durations_s["whole_home_joint_milp"].append(float(row["duration_s"]))
        expected_issue = partial_raw.index[-1] + pd.Timedelta(minutes=1)
        if pd.Timestamp(original_events[-1]["timestamp"]) + pd.Timedelta(minutes=15) != expected_issue:
            raise ValueError("Partial solver history does not join the final raw minute")
    progress_path = output / "whole_home_joint_progress.json"
    events_path = output / "whole_home_joint_solver_events.jsonl"
    if partial_directory is not None:
        shutil.copyfile(partial_directory / "whole_home_joint_solver_events.jsonl", events_path)
        progress_path.write_text(json.dumps({
            "timestamp": str(partial_raw.index[-1] + pd.Timedelta(minutes=1)),
            "status": "replaying_partial_controls",
            "solve_count": len(controller._solve_history),
            "replay_source": str(partial_directory),
        }, indent=2) + "\n")
    if hasattr(controller, "progress_callback"):
        def record_progress(row):
            progress = {**row, "wall_elapsed_seconds": time.perf_counter() - started}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2, default=str) + "\n")
            temporary.replace(progress_path)
            if row.get("status") != "solving":
                with events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, default=str) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        controller.progress_callback = record_progress

    split = None
    hub_other = outside_hub = None
    if args.split_hub_forecast:
        source_start = pd.Timestamp(args.rest_source_start)
        simulation_start = pd.Timestamp(config.start_time)
        if (source_start.tzinfo is not None
                or source_start.time() != simulation_start.time()):
            raise ValueError("Split-Hub source and Oiko simulation need matching local start clocks")
        source_end = source_start + pd.Timedelta(config.duration) + SPLIT_SOURCE_TAIL
        prepared = prepare_home_hub_replay(source_start - warmup_duration, source_end)
        split = _map_split_to_simulation(prepared, simulation_start, source_start)
        for source_path, history_days in (
            (args.hub_other_source, 21), (args.outside_hub_source, 28),
        ):
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
        inference_window = (source_start - warmup_duration, source_end)
        hub_other = FrozenSplitHubForecast.from_csv(
            args.hub_other_source, simulation_start=simulation_start,
            source_start=source_start, p90_history_days=21,
            source_window=inference_window,
        )
        outside_hub = FrozenSplitHubForecast.from_csv(
            args.outside_hub_source, simulation_start=simulation_start,
            source_start=source_start, p90_history_days=28,
            source_window=inference_window,
        )
        issues = pd.date_range(simulation_start, simulation_start + config.duration,
                               freq="15min", inclusive="left")
        for issue in issues:
            hub_other.forecast_kw(issue, steps=32)
            outside_hub.forecast_kw(issue, steps=32)
        print(f"Preflighted {len(issues)} split-Hub forecast issues", flush=True)
        (output / "home_hub_replay_quality_counts.csv").parent.mkdir(parents=True, exist_ok=True)
        prepared.meter_quality_counts.to_csv(output / "home_hub_replay_quality_counts.csv")

    # Freeze one mobility realization for the reference and controlled cases.
    twin_config = replace(config, duration=config.duration + HORIZON,
                          output_directory=output / "non_v2h_twin")
    mobility = build_ev_schedules(twin_config, home)
    ev_name = f"EV_{ev.vehicle_id}"
    if partial_directory is None:
        twin_ev = replace(ev, hems=replace(ev.hems, v2h=False))
        twin_home = replace(home, evs=(twin_ev,))
        twin_provider = ScenarioEVForecast(ev.vehicle_id, ev_control)
        twin_controller = EVScenarioController(twin_ev)
        twin_sim = SingleHomeHEMSSimulation(
        replace(twin_config, homes=(twin_home,)), twin_home,
        HEMSController(device_controllers=(twin_controller,), price_schedule=rate_selection.schedule),
        water_heater_hp_only=True,
        initial_temperature_c=21.0,
        water_draw_file=args.water_draw_file,
        water_draw_start=pd.Timestamp(args.water_draw_start),
        save_optimizer_plan_figures=False,
        ev_mobility_schedules=mobility,
        ev_backed_load=(
            pd.Series(0.0, index=pd.date_range(
                pd.Timestamp(config.start_time) - warmup_duration,
                pd.Timestamp(config.start_time) + config.duration + 2 * HORIZON,
                freq="min", inclusive="left",
            )) if split is not None else None
        ),
        hub_split_replay_kw=split,
        hvac_warmup_duration=warmup_duration,
        unified_context={
            "providers": {ev_name: twin_provider},
            "backed_loads": HOME_HUB_BACKED_LOADS if split is not None else BACKED_LOADS,
            "forecast_extension": HORIZON,
        },
        )
        print(f"Whole-home joint output: {output}", flush=True)
        print("Running non-V2H EV reference for the fixed backed-load cost term", flush=True)
        twin_sim.raw_directory.mkdir(parents=True, exist_ok=True)
        twin_raw = twin_sim._run_stepwise(apply_controls=True, raw_directory=twin_sim.raw_directory)[0]
        backed = pd.Series(backed_power(twin_raw,
                                   HOME_HUB_BACKED_LOADS if split is not None else BACKED_LOADS),
                       index=twin_raw.index,
                       name="reference_backed_load_kw")
        backed.to_csv(output / "non_v2h_reference_backed_load.csv")
        pd.DataFrame(twin_provider.history).to_csv(output / "non_v2h_ev_forecast_issuance.csv", index=False)
        twin_history = pd.DataFrame(twin_controller.solve_history)
        twin_history.to_csv(output / "non_v2h_ev_solve_history.csv", index=False)
    else:
        print(f"Whole-home joint approximate continuation: {output}", flush=True)
        print(f"Reusing completed non-V2H reference from {partial_directory}", flush=True)
        for filename in ("non_v2h_reference_backed_load.csv",
                         "non_v2h_ev_forecast_issuance.csv", "non_v2h_ev_solve_history.csv"):
            shutil.copyfile(partial_directory / filename, output / filename)
        backed_source = pd.read_csv(output / "non_v2h_reference_backed_load.csv")
        backed_times = pd.DatetimeIndex(pd.to_datetime(backed_source.pop("Time"), utc=True)
                                        .dt.tz_convert(pd.Timestamp(config.start_time).tz))
        backed = pd.Series(backed_source["reference_backed_load_kw"].to_numpy(dtype=float),
                           index=backed_times, name="reference_backed_load_kw")
        twin_history = pd.read_csv(output / "non_v2h_ev_solve_history.csv")
    if backed.index.min() > pd.Timestamp(config.start_time) or backed.index.max() < (
        pd.Timestamp(config.start_time) + config.duration + HORIZON - config.time_step
    ):
        raise ValueError("Non-V2H backed-load reference does not cover every EV forecast")

    rest = None if args.split_hub_forecast else RestOfHomeHybridForecast.from_csv(
        home.rest_of_home.data_file,
        simulation_start=config.start_time,
        source_start=home.rest_of_home.source_start,
        history_days=20,
        inference_gap_policy=args.rest_rf_inference_gap_policy,
    )
    ev_provider = ScenarioEVForecast(ev.vehicle_id, ev_control)
    if controller.uses_daily_grid2_demand:
        peak_formula = "max_s (0.25/N)*sum_weekdays[(2.34/21)*daily_on_peak_max_30min_kw[s] + (4.47/21)*daily_anytime_max_30min_kw[s]]; each peak stays within a paired water/EV scenario and includes observed earlier-today peaks; rolling 30-minute windows every 15 minutes, fully inside the weekday and eligible on-peak period"
    elif effective_peak_weight == 0:
        peak_formula = "zero"
    else:
        peak_index = (
            "active demand block" if args.peak_window == WINTER_TOU_PEAK_WINDOW
            else "active step"
        )
        peak_power = (
            "P_home_predicted_30min_average"
            if args.peak_window == WINTER_TOU_PEAK_WINDOW else "P_home_predicted"
        )
        if args.peak_limit_kw is not None:
            peak_formula = (
                f"lambda_peak * Delta_t/N * max_{{water,EV,{peak_index}}} "
                f"[{peak_power} - P_limit]_+"
            )
        else:
            peak_formula = f"weight * max_{{water,EV,{peak_index}}} {peak_power}"
    settings = {
        "hpwh_variant": effective_hpwh_variant(args.hpwh_variant, thermal.water_heater_controller),
        "native_warmup_hours": warmup_duration.total_seconds() / 3600,
        "description": (
            f"one CVXPY/Gurobi MILP for HVAC and {7 * args.ev_scenario_count} paired HPWH/EV paths"
        ),
        "simulation_start": args.start_time.isoformat(),
        "duration_hours": args.duration_hours,
        "rate_structure_choice": args.rate_structure,
        "rate_structure": rate_selection.metadata,
        "demand_controller_treatment": (
            "legacy generic peak objective"
            if args.rate_structure == "legacy" else
            "two weekday daily 30-minute demand components, separately worst-case across paired scenarios, with observed-peak floors and 0.25/N normalization; unnormalized realized bill"
            if controller.uses_daily_grid2_demand else
            "generic peak objective disabled; scenario 3 uses separate pointwise grid and panel slacks"
            if doe_selected else
            "generic peak objective disabled because the rate has no demand charge"
            if effective_peak_weight == 0 else
            "eight-hour weekday winter-TOU-window 30-minute demand surrogate "
            "with $0.71/kW weight "
            "scaled by 0.25/32; realized bill uses one daily 30-minute maximum"
            if args.peak_limit_kw is not None else
            "eight-hour weekday winter-TOU-window 30-minute demand surrogate "
            "weighted at $0.71/kW; "
            "realized bill uses one daily 30-minute maximum"
        ),
        "horizon_hours_all_devices": 8,
        "weather_forecast_mode": args.weather_forecast_mode,
        "weather_forecast_file": (
            str((REPO_ROOT / (args.weather_forecast_file or config.weather_forecast_file)).resolve())
            if weather_archive is not None else None
        ),
        "prediction_step_minutes": 15,
        "water_scenarios": 7,
        "ev_scenarios": args.ev_scenario_count,
        "ev_scenario_method": args.ev_scenario_method,
        "peak_scenario_combinations": 7 * args.ev_scenario_count,
        "peak_window": args.peak_window,
        "peak_demand_interval_minutes": (
            30 if args.peak_window == WINTER_TOU_PEAK_WINDOW else 15
        ),
        "peak_limit_kw": args.peak_limit_kw,
        "peak_formula": peak_formula,
        "grid2_demand_aggregation": ("worst_complete_scenario_v1" if controller.uses_daily_grid2_demand else None),
        "grid2_demand_window_step_minutes": (15 if args.rate_structure == WINTER_TOU_DEMAND_CPP else None),
        "grid2_demand_formulation": (controller.grid2_demand_formulation
                                     if args.rate_structure == WINTER_TOU_DEMAND_CPP else None),
        "grid2_on_peak_demand_usd_per_kw_day": (rate_selection.schedule.on_peak_demand_charge_usd_per_kw_day
                                               if controller.uses_daily_grid2_demand else None),
        "grid2_anytime_demand_usd_per_kw_day": (rate_selection.schedule.anytime_demand_charge_usd_per_kw_day
                                               if controller.uses_daily_grid2_demand else None),
        "grid2_demand_normalization_steps": 32 if controller.uses_daily_grid2_demand else None,
        "peak_normalization_hours_per_step": (
            0.25 / 32 if controller.uses_daily_grid2_demand
            or (args.peak_limit_kw is not None and effective_peak_weight > 0) else None
        ),
        "peak_weight_usd_per_kw": effective_peak_weight,
        "doe_grid_limit_kw": DOE_GRID_LIMIT_KW if doe_selected else None,
        "doe_panel_limit_kw": DOE_PANEL_LIMIT_KW if doe_selected else None,
        "doe_grid_penalty_weight": DOE_GRID_PENALTY_USD_PER_KWH if doe_selected else None,
        "doe_panel_penalty_weight": DOE_PANEL_PENALTY_USD_PER_KWH if doe_selected else None,
        "doe_grid_window_local": ["weekday 06:00-09:00", "weekday 17:00-21:00"] if doe_selected else None,
        "doe_panel_window_local": "all hours" if doe_selected else None,
        "doe_slack_scenario_aggregation": "one slack per time step shared across all paired scenarios" if doe_selected else None,
        "doe_slack_objective_normalization": "Delta_t / N = 0.25 / 32" if doe_selected else None,
        "doe_panel_power_definition": "gross home load before V2H offset" if doe_selected else None,
        "doe_reported_bill_treatment": (
            "energy charges only; grid and panel excesses are compliance metrics, "
            "and the DOE slack weights are controller penalties, not billed charges"
            if doe_selected else None
        ),
        "solver_options": solver_options,
        "gurobi_wls_environment_refresh_seconds": 3600,
        "gurobi_solver_error_retries": 2,
        "gurobi_wls_network_error_retries": 8,
        "ev_variant": ("milp_health: electricity + comfort(1000) + soft healthy SOC(1000) "
                       f"+ trip shortfall({ev.hems.trip_shortfall_weight:g}) + terminal energy value"),
        "ev_health_soc_bounds": [0.2, 0.8],
        "ev_discharge_cap_source": (
            "joint HPWH and HVAC compressor/fan decisions plus fixed Hub-other forecast, "
            "paired by water and EV scenario"
            if args.split_hub_forecast else
            "joint HPWH and HVAC compressor/fan decisions, paired by water and EV scenario"
        ),
        "ev_fixed_backed_cost_source": (
            "non-V2H twin, heat_pump, water_heater, and Hub-other loads"
            if args.split_hub_forecast else
            "non-V2H twin, heat_pump and water_heater loads"
        ),
        "ev_fixed_backed_cost_role": (
            "The non-V2H twin supplies a retrospective future backed-load tariff "
            "term that is additive and independent of every joint decision. "
            "It affects the reported objective constant and Gurobi relative "
            "MIP-gap denominator; EV discharge constraints instead use scenario "
            "HPWH/HVAC decisions and the causal Hub-other forecast."
        ),
        "water_scenario_method": thermal.water_heater_controller.scenario_method,
        "rest_forecast_method": None if rest is None else rest.method_label,
        "rest_forecast_metadata": None if rest is None else rest.metadata,
        "hub_other_forecast_method": None if hub_other is None else hub_other.method_label,
        "hub_other_forecast_metadata": None if hub_other is None else hub_other.metadata,
        "outside_hub_forecast_method": None if outside_hub is None else outside_hub.method_label,
        "outside_hub_forecast_metadata": None if outside_hub is None else outside_hub.metadata,
        "split_hub_plant_replay": bool(split is not None),
        "backed_loads": HOME_HUB_BACKED_LOADS if split is not None else BACKED_LOADS,
        "rest_source_start": args.rest_source_start,
        "water_draw_start": args.water_draw_start.isoformat(),
        "mobility_metadata": mobility[ev.vehicle_id].metadata,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if partial_directory is not None:
        settings["approximate_resume"] = {
            "source_partial_directory": str(partial_directory),
            "first_new_issue": str(partial_raw.index[-1] + pd.Timedelta(minutes=1)),
            "method": "replay recorded minute-level HVAC, HPWH, and EV controls from a fresh plant",
            "limitation": "Controller internal state and plant hidden states are reconstructed, not checkpoint restored",
        }
    (output / "whole_home_joint_settings.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
    print(f"Running {args.duration_hours:g}-hour controlled whole-home replay and matched native baseline", flush=True)
    simulation = SingleHomeHEMSSimulation(
        config, home, HEMSController(device_controllers=(controller,), price_schedule=rate_selection.schedule),
        water_heater_hp_only=True,
        initial_temperature_c=21.0,
        water_draw_file=args.water_draw_file,
        water_draw_start=pd.Timestamp(args.water_draw_start),
        rest_of_home_forecast=rest,
        hub_other_forecast=hub_other,
        outside_hub_forecast=outside_hub,
        weather_archive=weather_archive,
        save_optimizer_plan_figures=False,
        ev_mobility_schedules=mobility,
        ev_backed_load=backed,
        hub_split_replay_kw=split,
        hvac_warmup_duration=warmup_duration,
        unified_context={
            "providers": {ev_name: ev_provider},
            "backed_loads": HOME_HUB_BACKED_LOADS if split is not None else BACKED_LOADS,
            "forecast_extension": HORIZON,
        },
    )
    if partial_directory is not None:
        replay = JointApproximateReplay(
            controller, partial_raw, partial_directory, output / "resume_audit.json",
        )
        controller.controls_for_step = replay.controls_for_step
    summary = simulation.run()
    if partial_directory is not None:
        complete_replay_audit(output)
    data_dir = output / "data"
    for name, minute in (
        ("controlled", summary.standardized_results),
        ("uncontrolled", summary.uncontrolled_results),
    ):
        block = minute.resample("15min", closed="left", label="left").mean(numeric_only=True)
        block["minute_count"] = minute["total_electric_power_kw"].resample(
            "15min", closed="left", label="left",
        ).count()
        if not block["minute_count"].eq(15).all():
            raise ValueError("Controlled comparison produced an incomplete 15-minute block")
        block.to_csv(data_dir / f"home_01_{name}_15min.csv")
    history = controller.solve_history
    if not history.empty:
        history.to_csv(data_dir / "whole_home_joint_solve_history.csv", index=False)
    objective_history = controller.objective_component_history
    if not objective_history.empty:
        objective_history.to_csv(data_dir / "whole_home_joint_objective_component_history.csv", index=False)
    pd.DataFrame(ev_provider.history).to_csv(output / "ev_forecast_issuance.csv", index=False)
    price = summary.controls["electricity_price_usd_per_kwh"]
    demand_rate = rate_selection.schedule.demand_charge_usd_per_kw_day if args.rate_structure != "legacy" else 0.0
    anytime_demand_rate = float(getattr(rate_selection.schedule, "anytime_demand_charge_usd_per_kw_day", 0))
    metrics = {
        "controlled": _metrics(summary.standardized_results, price, demand_rate,
                               anytime_demand_rate_usd_per_kw_day=anytime_demand_rate),
        "uncontrolled": _metrics(summary.uncontrolled_results, price, demand_rate,
                                 anytime_demand_rate_usd_per_kw_day=anytime_demand_rate),
    }
    if demand_rate > 0 or anytime_demand_rate > 0:
        for name, minute in (
            ("controlled", summary.standardized_results),
            ("uncontrolled", summary.uncontrolled_results),
        ):
            bill = winter_daily_demand_bill(
                minute["total_electric_power_kw"].astype(float),
                rate_usd_per_kw_day=demand_rate,
                anytime_rate_usd_per_kw_day=anytime_demand_rate,
            )
            bill.per_day.to_csv(data_dir / f"home_01_{name}_daily_demand_bill.csv")
    if args.peak_limit_kw is not None:
        metrics["controlled"]["threshold"] = _threshold_metrics(
            summary.standardized_results, args.peak_limit_kw, args.peak_window,
        )
        metrics["uncontrolled"]["threshold"] = _threshold_metrics(
            summary.uncontrolled_results, args.peak_limit_kw, args.peak_window,
        )
    if doe_selected:
        for name, minute in (
            ("controlled", summary.standardized_results),
            ("uncontrolled", summary.uncontrolled_results),
        ):
            compliance = evaluate_scenario3_doe(
                minute, grid_limit_kw=DOE_GRID_LIMIT_KW,
                panel_limit_kw=DOE_PANEL_LIMIT_KW,
            )
            compliance.intervals.to_csv(data_dir / f"home_01_{name}_doe_15min.csv")
            metrics[name]["doe_compliance"] = compliance.summary
    start = pd.Timestamp(config.start_time)
    end = start + config.duration
    controlled_inventory = _ev_boundary_energy(
        simulation.raw_directory / f"{home.home_id}.csv", ev_name, start, end
    )
    uncontrolled_inventory = _ev_boundary_energy(
        simulation.raw_directory / "uncontrolled" / f"{home.home_id}.csv",
        ev_name, start, end,
    )
    initial_value = _daily_average_tariff(simulation.controller.electricity_price_for_step, start)
    final_value = _daily_average_tariff(simulation.controller.electricity_price_for_step, end)
    raw_savings = (metrics["uncontrolled"]["total_bill_usd"]
                   - metrics["controlled"]["total_bill_usd"])
    inventory_adjustment = (
        final_value * (controlled_inventory[1] - uncontrolled_inventory[1])
        - initial_value * (controlled_inventory[0] - uncontrolled_inventory[0])
    )
    times = history["duration_s"].astype(float) if not history.empty else pd.Series(dtype=float)
    results = {
        "metrics": metrics,
        "electricity_comparison": {
            "raw_bill_savings_usd": raw_savings,
            "energy_charge_savings_usd": (
                metrics["uncontrolled"]["electricity_cost_usd"]
                - metrics["controlled"]["electricity_cost_usd"]
            ),
            "demand_charge_savings_usd": (
                metrics["uncontrolled"]["demand_charge_usd"]
                - metrics["controlled"]["demand_charge_usd"]
            ),
            "controlled_ev_energy_start_end_kwh": controlled_inventory,
            "uncontrolled_ev_energy_start_end_kwh": uncontrolled_inventory,
            "initial_daily_average_tariff_usd_per_kwh": initial_value,
            "final_daily_average_tariff_usd_per_kwh": final_value,
            "ev_inventory_adjustment_usd": inventory_adjustment,
            "ev_inventory_adjusted_bill_savings_usd": raw_savings + inventory_adjustment,
        },
        "non_v2h_reference": {
            "ev_solve_count": len(twin_history),
            "ev_status_counts": dict(Counter(twin_history["status"])) if not twin_history.empty else {},
            "ev_fallback_count": int(twin_history["fallback"].sum()) if not twin_history.empty else 0,
        },
        "aligned_peak_change_kw": (
            metrics["controlled"]["aligned_15_minute_peak_kw"]
            - metrics["uncontrolled"]["aligned_15_minute_peak_kw"]
        ),
        "joint_solver": {
            "solve_count": len(history),
            "status_counts": dict(Counter(history["status"])) if not history.empty else {},
            "mean_solve_seconds": float(times.mean()) if not times.empty else None,
            "max_solve_seconds": float(times.max()) if not times.empty else None,
            "total_solve_seconds": float(times.sum()) if not times.empty else None,
            "max_relative_mip_gap": (
                float(history["relative_gap"].dropna().max())
                if "relative_gap" in history and history["relative_gap"].notna().any() else None
            ),
        },
        "total_wall_seconds": time.perf_counter() - started,
    }
    if doe_selected:
        results["electricity_comparison"]["bill_scope_note"] = (
            "Scenario 3 savings compare energy charges only. The controller's DOE "
            "slack weights are not utility bill charges; realized violations are "
            "reported separately under metrics.*.doe_compliance."
        )
    (output / "whole_home_joint_initial_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

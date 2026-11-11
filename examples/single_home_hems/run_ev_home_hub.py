#!/usr/bin/env python3
# ruff: noqa: E402
"""Run EV-only V2H MPC with live OCHRE thermal loads and a split DC House replay.

The EV receives historical mobility scenarios and an issued Home Hub
forecast. HVAC and the heat-pump water heater remain native OCHRE equipment.
Hub-other and Outside-Hub electricity are fixed measured replay channels.
No non-V2H twin or future plant load enters the controller forecast.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
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
    DEFAULT_CONFIG_DIRECTORY, DEFAULT_WATER_DRAW_FILE,
)
from examples.single_home_hems.run_simulation import (
    _local_weather_time, next_available_output_directory,
)
from examples.single_home_hems.run_whole_home_joint import (
    DEFAULT_EV_CONTROL_CONFIG, DEFAULT_EV_HOME_CONFIG, DEFAULT_START,
    HORIZON, _daily_average_tariff, _ev_boundary_energy, _ev_inputs, _metrics,
)
from examples.single_home_hems.single_home_hems.backed_loads import (
    HOME_HUB_BACKED_LOADS,
)
from examples.single_home_hems.single_home_hems.controllers.ev_home_hub import (
    EVHomeHubController,
)
from examples.single_home_hems.single_home_hems.eligible_hub_rf import EligibleHubRFProvider
from examples.single_home_hems.single_home_hems.submeter_hub_rf import SubmeterHubRFProvider
from examples.single_home_hems.single_home_hems.ev_hub_issued import IssuedEVHubForecasts
from examples.single_home_hems.single_home_hems.ev_scenarios import ScenarioEVForecast
from examples.single_home_hems.single_home_hems.ev_study_forecasts import (
    PerfectHomeHubForecast, PerfectOutsideHubForecast, PerfectMobilityEVForecast,
)
from examples.single_home_hems.single_home_hems.rate_setup import RATE_CHOICES, select_rate_for_simulation
from examples.single_home_hems.single_home_hems.rates import (
    MISO_RTP_DOE, WINTER_TOU_DEMAND_CPP,
)
from examples.single_home_hems.single_home_hems.rate_billing import winter_daily_demand_bill
from examples.single_home_hems.single_home_hems.doe_compliance import evaluate_scenario3_doe
from examples.single_home_hems.single_home_hems.ev_study_metrics import ev_study_metrics
from examples.single_home_hems.single_home_hems.frozen_split_hub_forecast import FrozenSplitHubForecast
from examples.single_home_hems.single_home_hems.home_hub_replay import (
    PreparedHomeHubReplay, prepare_home_hub_replay,
)
from examples.single_home_hems.single_home_hems.home_hub_observation import (
    HUB_POWER_KEY, observation_keys,
)
from examples.single_home_hems.single_home_hems.outside_hub_er_multires_rf import (
    DEFAULT_ARTIFACT as DEFAULT_OUTSIDE_ER_RF_MODEL,
    MultiresWeatherOutsideHubRF,
)
from examples.single_home_hems.single_home_hems.outside_hub_forecast_data import (
    build_outside_hub_forecast_csv,
)
from examples.single_home_hems.single_home_hems.prior_day_load_forecast import (
    prior_complete_local_day,
)
from examples.single_home_hems.single_home_hems.rest_of_home_forecast import (
    RestOfHomeHybridForecast,
)
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation, select_home,
)


WARMUP = pd.Timedelta(hours=16)
DEFAULT_SOURCE_START = pd.Timestamp("2026-09-22 03:00")
CONTROL_STEP = pd.Timedelta(minutes=15)
DEFAULT_ELIGIBLE_RF_MODEL = (EXAMPLE_ROOT / "inputs/predictive/models/eligible_hub_rf.joblib")
DEFAULT_SUBMETER_RF_MODEL = (EXAMPLE_ROOT / "inputs/predictive/models/submeter_hub_rf.joblib")
HUB_CIRCUIT_SEED_COLUMNS = ("water_heater_kw", "outdoor_heat_pump_kw", "indoor_fan_kw")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home-id", default="home_01")
    parser.add_argument("--start-time", type=dt.datetime.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--source-start", type=pd.Timestamp, default=DEFAULT_SOURCE_START,
                        help="Naive local DC House interval start matching simulation start")
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--horizon-hours", type=int, choices=(8, 20), default=8,
                        help="Rolling EV prediction horizon; default Hub RF predicts all 20 hours directly")
    parser.add_argument("--mobility-source-window-hours", type=float,
                        help="Select Audi mobility using this evaluation window, including for a short matched pilot")
    parser.add_argument("--rate-selection-duration-hours", type=float,
                        help="Select the study tariff using this evaluation window, including for a short matched pilot")
    parser.add_argument("--native-warmup-hours", type=int, choices=(16, 24), default=16,
                        help="Measured native plant warm-up; 24 hours matches rolling-start HPWH comparisons")
    parser.add_argument("--hub-issued-forecast-csv", type=Path,
                        help="Pre-issued predictions on the simulation clock for issued_csv mode")
    parser.add_argument("--hub-forecast-mode", choices=("submeter_rf", "issued_csv", "corrected_rf", "perfect"),
                        default="submeter_rf",
                        help="Default: four-week unweighted RF, leaf 15, with three circuit power histories")
    parser.add_argument("--hub-rf-artifact", type=Path,
                        help="Defaults to the circuit-history model for submeter_rf or legacy aggregate model for corrected_rf")
    parser.add_argument("--hub-seed-csv", type=Path,
                        help="Matched pretest 15-minute non-EV Hub history; submeter_rf also requires water_heater_kw, outdoor_heat_pump_kw and indoor_fan_kw")
    parser.add_argument("--outside-rf-inference-gap-policy",
                        choices=("strict", "causal_short_gap"), default="causal_short_gap")
    parser.add_argument("--outside-forecast-mode", choices=(
        "legacy_20d", "frozen_rf28_p90_28d", "rf_multires_weather_persistence", "perfect",
    ),
                        default="rf_multires_weather_persistence")
    parser.add_argument("--outside-rf-artifact", type=Path, default=DEFAULT_OUTSIDE_ER_RF_MODEL,
                        help="Fitted four-week resistance-inclusive multiresolution RF artifact")
    parser.add_argument("--peak-weight-usd-per-kw", type=float, default=0.1)
    parser.add_argument("--peak-limit-kw", type=float,
                        help="Soft exceedance threshold; omit for the legacy raw night-peak objective")
    parser.add_argument("--peak-window", choices=("all", "evening", "night"), default="night")
    parser.add_argument("--ev-scenario-count", type=int, default=7)
    parser.add_argument("--ev-scenario-method", default="tod_dow_state_hier_limit_persistence",
                        choices=("knn_state", "tod_only", "tod_dow", "tod_state",
                                 "tod_dow_state", "tod_dow_state_hier", "tod_dow_state_hier_limit_persistence",
                                 "tod_dow_hier", "tod_state_hier"))
    parser.add_argument("--time-limit-seconds", type=float,
                        help="Per-decision cap; defaults to 600 s at 8 h and 180 s at 20 h")
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-backend", choices=("cvxpy", "sparse_native_v1"), default="cvxpy",
                        help="MILP frontend; sparse native compilation/model reuse preserves the EV formulation")
    parser.add_argument("--hub-mobility-split-policy", default="reject",
                        choices=("reject", "hold_parent_interval_coefficients"),
                        help="Opt in to holding issued 15-minute Hub coefficients on native EV event subsegments")
    parser.add_argument("--rate-structure", choices=RATE_CHOICES, default="legacy")
    parser.add_argument("--grid2-demand-formulation",
                        choices=("weekday_daily_dt_over_n_v1", "legacy_horizon_peak"),
                        default="weekday_daily_dt_over_n_v1",
                        help="Scenario 2 peak objective; actual billing uses the selected current tariff")
    parser.add_argument("--mobility-forecast-mode", choices=("scenario", "perfect"), default="scenario")
    parser.add_argument("--perfect-load-truth-csv", type=Path,
                        help="Explicit offline native minute truth for the approved oracle study")
    parser.add_argument("--one-shot", action="store_true",
                        help="Optimize the full evaluation once, with no solver time limit, then replay modes")
    parser.add_argument("--source-config-directory", type=Path, default=DEFAULT_CONFIG_DIRECTORY)
    parser.add_argument("--ev-home-config", type=Path, default=DEFAULT_EV_HOME_CONFIG)
    parser.add_argument("--ev-control-config", type=Path, default=DEFAULT_EV_CONTROL_CONFIG)
    parser.add_argument("--water-draw-file", type=Path, default=DEFAULT_WATER_DRAW_FILE)
    parser.add_argument("--water-draw-start", type=dt.datetime.fromisoformat,
                        default=dt.datetime(2025, 11, 6, 3))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate replay and Outside-Hub forecast inputs without running OCHRE")
    return parser


def _validated_args(argv: list[str] | None) -> argparse.Namespace:
    args = build_argument_parser().parse_args(argv)
    if args.hub_rf_artifact is None:
        args.hub_rf_artifact = (DEFAULT_SUBMETER_RF_MODEL
                                if args.hub_forecast_mode == "submeter_rf"
                                else DEFAULT_ELIGIBLE_RF_MODEL)
    if args.time_limit_seconds is None:
        args.time_limit_seconds = 180.0 if args.horizon_hours == 20 else 600.0
    if not args.validate_only and args.hub_forecast_mode == "issued_csv" and args.hub_issued_forecast_csv is None:
        raise ValueError("--hub-issued-forecast-csv is required for issued-CSV EV-only simulation")
    if (not args.validate_only and args.hub_forecast_mode in {"corrected_rf", "submeter_rf"}
            and args.hub_seed_csv is None):
        raise ValueError(f"--hub-seed-csv is required for {args.hub_forecast_mode} EV-only simulation")
    if "perfect" in (args.hub_forecast_mode, args.outside_forecast_mode) and args.perfect_load_truth_csv is None:
        raise ValueError("--perfect-load-truth-csv is required for perfect load forecasts")
    if args.one_shot and (args.hub_forecast_mode, args.outside_forecast_mode, args.mobility_forecast_mode) != (
        "perfect", "perfect", "perfect",
    ):
        raise ValueError("--one-shot requires perfect Hub, Outside-Hub and mobility forecasts")
    if not np.isfinite(args.duration_hours) or args.duration_hours <= 0:
        raise ValueError("--duration-hours must be finite and positive")
    if args.mobility_source_window_hours is not None and (
            not np.isfinite(args.mobility_source_window_hours)
            or args.mobility_source_window_hours < args.duration_hours):
        raise ValueError("--mobility-source-window-hours must cover the evaluation duration")
    if args.rate_selection_duration_hours is not None and (
            not np.isfinite(args.rate_selection_duration_hours)
            or args.rate_selection_duration_hours < args.duration_hours):
        raise ValueError("--rate-selection-duration-hours must cover the evaluation duration")
    if args.duration_hours * 4 != round(args.duration_hours * 4):
        raise ValueError("--duration-hours must contain whole 15-minute control intervals")
    start = pd.Timestamp(args.start_time)
    source = pd.Timestamp(args.source_start)
    if start != start.floor(CONTROL_STEP) or source != source.floor(CONTROL_STEP):
        raise ValueError("Simulation and source starts must align to 15-minute intervals")
    if source.tzinfo is not None:
        raise ValueError("--source-start must be a naive local DC House timestamp")
    if start.hour != source.hour or start.minute != source.minute:
        raise ValueError("Simulation and source starts must have the same local clock time")
    for name in ("peak_weight_usd_per_kw", "time_limit_seconds", "mip_gap"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.mip_gap > 1 or args.solver_threads < 1:
        raise ValueError("MIP gap must not exceed 1 and solver threads must be positive")
    if args.peak_limit_kw is not None and (
        not np.isfinite(args.peak_limit_kw) or args.peak_limit_kw <= 0
    ):
        raise ValueError("--peak-limit-kw must be finite and positive")
    if args.ev_scenario_count < 1:
        raise ValueError("--ev-scenario-count must be positive")
    return args


def map_replay_to_simulation(
    prepared: PreparedHomeHubReplay, simulation_start: pd.Timestamp,
    source_start: pd.Timestamp,
) -> pd.DataFrame:
    """Apply one fixed local-clock offset to the complete split plant replay."""
    simulation_start = pd.Timestamp(simulation_start)
    source_start = pd.Timestamp(source_start)
    if source_start.tzinfo is not None:
        raise ValueError("Home Hub replay source clock must be naive local time")
    frame = prepared.power_kw.copy()
    simulation_wall = (simulation_start.tz_localize(None)
                       if simulation_start.tzinfo is not None else simulation_start)
    mapped_wall = pd.DatetimeIndex(simulation_wall + (frame.index - source_start))
    mapped = (mapped_wall.tz_localize(simulation_start.tz, ambiguous="raise", nonexistent="raise")
              if simulation_start.tzinfo is not None else mapped_wall)
    frame.index = pd.DatetimeIndex(mapped, name="timestamp")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Mapped Home Hub replay clock must be unique and increasing")
    return frame


def validate_hub_issuance(
    provider: IssuedEVHubForecasts, start: pd.Timestamp,
    duration: pd.Timedelta, inverter_kw: float,
    horizon: pd.Timedelta = HORIZON,
) -> int:
    """Fail before simulation if any EV solve lacks a complete issued view."""
    starts = pd.date_range(start, start + duration, freq=CONTROL_STEP, inclusive="left")
    for timestamp in starts:
        provider.issue(timestamp, pd.DataFrame(), horizon, inverter_kw)
    count = len(starts)
    provider.history.clear()
    return count


def _jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, (Path, pd.Timestamp, dt.datetime)) else value
            for key, value in vars(args).items()}


def _peak_objective_metadata(controller: EVHomeHubController, schedule) -> dict:
    """Describe the effective peak objective separately from metered billing."""
    horizon_peak = (
        "0.71*maximum whole-home net import over complete weekday TOU 30-minute "
        "blocks and all mobility paths; one horizon maximum; no normalization "
        "or historical peak floor"
    )
    daily_peak = (
        "max_s (0.25/N)*sum_weekdays[(2.34/21)*daily_on_peak_max_30min_kw[s] + "
        "(4.47/21)*daily_anytime_max_30min_kw[s]]; each peak is within one mobility "
        "scenario and includes observed earlier-today peaks; N is the full "
        "horizon control-interval count; rolling 30-minute windows every 15 minutes, "
        "fully within the local weekday and eligible on-peak period"
    )
    formula = {
        "tou-winter": "No generic peak objective",
        "tou-winter+demand": horizon_peak,
        WINTER_TOU_DEMAND_CPP: daily_peak if controller.uses_daily_grid2_demand else horizon_peak,
        MISO_RTP_DOE: "No generic peak; (0.25/N)*(1000*sum(panel slack)+10*sum(active grid slack)); shared per-step slacks",
    }.get(controller.rate_structure) or (
        "weight*(0.25/32)*z, z>=0 and z>=full_non_EV_Hub[t]+Outside_Hub[t]+EV[s,t]-limit "
        "for every EV scenario s and active 15-minute t"
        if controller.peak_limit_kw is not None else
        "weight*z, z>=full_non_EV_Hub[t]+Outside_Hub[t]+EV[s,t] for every active t, EV s"
    )
    demand = None
    if controller.uses_daily_grid2_demand:
        steps = int(controller.horizon / CONTROL_STEP)
        step_hours = CONTROL_STEP.total_seconds() / 3600
        demand = {
            "on_peak_usd_per_kw_day": schedule.on_peak_demand_charge_usd_per_kw_day,
            "anytime_usd_per_kw_day": schedule.anytime_demand_charge_usd_per_kw_day,
            "dt_hours": step_hours,
            "normalization_steps": steps,
            "dt_over_n": step_hours / steps,
            "billing_days": "weekdays_only",
            "aggregation": "worst_complete_scenario_v1",
            "already_observed_peaks": "included as floors for the current weekday",
        }
    elif controller.rate_structure == WINTER_TOU_DEMAND_CPP:
        demand = {
            "on_peak_usd_per_kw": controller.peak_weight_usd_per_kw,
            "anytime_usd_per_kw": 0.0,
            "objective_scale": 1.0,
            "normalization_steps": None,
            "dt_over_n": None,
            "aggregation": "one maximum across complete weekday on-peak 30-minute blocks in the horizon and all mobility scenarios",
            "already_observed_peaks": "not included",
            "billing_tariff": "current Scenario 2 daily on-peak and anytime tariff; independent of the legacy objective",
        }
    return {"peak_formula": formula, "grid2_demand_objective": demand}


def _read_non_ev_hub_seed(path: Path) -> pd.Series:
    seed = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    if "hub_non_ev_power_kw" not in seed:
        raise ValueError("RF seed must contain hub_non_ev_power_kw")
    return seed["hub_non_ev_power_kw"].astype(float)


def _read_hub_circuit_seed(path: Path) -> pd.DataFrame:
    seed = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    missing = set(HUB_CIRCUIT_SEED_COLUMNS) - set(seed)
    if missing:
        raise ValueError(
            f"Circuit-history RF seed lacks {sorted(missing)}; prepare a matched seed "
            "with prepare_ev_submeter_hub_seed.py from completed native OCHRE results"
        )
    return seed.loc[:, list(HUB_CIRCUIT_SEED_COLUMNS)].astype(float)


def _observed_ambient_15min(
    weather_file: Path, timezone: object, first_issue: pd.Timestamp, final_issue: pd.Timestamp,
) -> pd.Series:
    """Read realized Oiko temperature; the RF selects only completed past bins."""
    weather = pd.read_csv(weather_file, usecols=["Time", "temp_air"])
    times = pd.DatetimeIndex(pd.to_datetime(weather["Time"], utc=True)).tz_convert(timezone)
    minute = pd.Series(weather["temp_air"].to_numpy(float), index=times)
    grouped = minute.resample("15min", label="left", closed="left")
    ambient, counts = grouped.mean(), grouped.count()
    # The prepared archive spans autumn DST. Only the causal issue window is
    # needed; narrowing here prevents duplicate naive wall-clock labels.
    return ambient.loc[counts.eq(15) & (ambient.index >= first_issue - pd.Timedelta(days=8))
                       & (ambient.index < final_issue)]


def _threshold_metrics(data: pd.DataFrame, limit_kw: float, window: str) -> dict[str, float | int | str | None]:
    """Evaluate the soft peak target on realized complete 15-minute site means."""
    power = data["total_electric_power_kw"].astype(float)
    groups = power.resample("15min", closed="left", label="left")
    count, means = groups.count(), groups.mean()
    if count.empty or not count.eq(15).all():
        raise ValueError("Peak evaluation requires complete 15-minute site-power bins")
    if window == "all":
        active = np.ones(len(means), dtype=bool)
    elif window == "evening":
        active = np.asarray((means.index.hour >= 16) & (means.index.hour < 22), dtype=bool)
    elif window == "night":
        active = np.asarray((means.index.hour >= 3) & (means.index.hour < 7), dtype=bool)
    else:
        raise ValueError(f"Unknown peak window: {window}")
    selected = means.loc[active]
    exceedance = (selected - limit_kw).clip(lower=0)
    return {
        "limit_kw": float(limit_kw),
        "window": window,
        "active_interval_count": int(len(selected)),
        "active_peak_kw": float(selected.max()) if len(selected) else None,
        "active_peak_time": str(selected.idxmax()) if len(selected) else None,
        "max_exceedance_kw": float(exceedance.max()) if len(selected) else 0.0,
        "intervals_above_limit": int(exceedance.gt(0).sum()),
        "hours_above_limit": float(exceedance.gt(0).sum() * 0.25),
        "exceedance_energy_kwh": float(exceedance.sum() * 0.25),
        "full_week_peak_kw": float(means.max()),
        "full_week_peak_time": str(means.idxmax()),
    }


def _quarter_hour_power(data: pd.DataFrame) -> pd.DataFrame:
    """Return complete comparison bins with the same component meanings as the minute trace."""
    columns = [
        "total_electric_power_kw", "hvac_power_kw", "water_heating_power_kw",
        "ev_power_kw", "rest_of_home_power_kw", "hub_other_power_kw",
        "outside_hub_power_kw", "home_hub_non_ev_power_kw",
    ]
    columns.extend((
        "hvac_resistance_power_kw", "outside_hub_with_resistance_power_kw",
    ))
    missing = set(columns) - set(data)
    if missing:
        raise ValueError(f"Comparison power trace lacks {sorted(missing)}")
    groups = data.loc[:, columns].resample("15min", closed="left", label="left")
    counts = groups.count()
    if counts.empty or not counts.eq(15).all().all():
        raise ValueError("Comparison power trace requires complete 15-minute bins")
    means = groups.mean()
    if not np.allclose(
        means["total_electric_power_kw"],
        means["home_hub_non_ev_power_kw"] + means["ev_power_kw"]
        + means["outside_hub_with_resistance_power_kw"],
        rtol=0, atol=1e-7,
    ):
        raise ValueError("EV-only comparison bins do not balance with resistance outside Hub")
    return means


def main(argv: list[str] | None = None) -> int:
    args = _validated_args(argv)
    started = time.perf_counter()
    source_config = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    thermal_home = select_home(source_config, args.home_id)
    ev, ev_control = _ev_inputs(args)
    if args.horizon_hours == 20 and not args.one_shot:
        ev = replace(ev, hems=replace(
            ev.hems, horizon_hours=20, forecast_margin_hours=20,
            comfort_soc_min=0.30,
        ))
        ev_control.update(horizon_hours=20, forecast_hours=20)
    ev = replace(ev, hems=replace(ev.hems, solver_options={"Threads": args.solver_threads}))
    ev_control["forecast"] = {
        **ev_control["forecast"], "scenario_count": args.ev_scenario_count,
        "scenario_method": args.ev_scenario_method,
    }
    home = replace(thermal_home, evs=(ev,))
    output = args.output_directory or (
        EXAMPLE_ROOT / "outputs" / args.home_id / "ev_only_home_hub_8h_milp_k7_night"
    )
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()
    if not args.validate_only:
        if args.output_directory is None:
            output = next_available_output_directory(output)
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory already contains files: {output}")
    config = replace(
        source_config,
        start_time=_local_weather_time(args.start_time, source_config.weather_metadata),
        duration=dt.timedelta(hours=args.duration_hours),
        time_step=dt.timedelta(minutes=1),
        output_directory=output,
        homes=(home,),
    )
    config.validate()
    rate_window = (config.duration if args.rate_selection_duration_hours is None else
                   dt.timedelta(hours=args.rate_selection_duration_hours))
    horizon = (pd.Timedelta(hours=args.duration_hours) if args.one_shot else
               pd.Timedelta(hours=args.horizon_hours))
    rate_selection = select_rate_for_simulation(
        args.rate_structure, replace(config, duration=rate_window),
        forecast_horizon=horizon,
        rtp_forecast_policy=("same_clock_current_day_until_publication"
                             if args.horizon_hours == 20 else "archived_expost_full"),
    )
    scenario_count = 1 if args.mobility_forecast_mode == "perfect" else args.ev_scenario_count
    # Keep the paired runs' mobility source window fixed. Extend only the
    # exogenous source tail needed for the selected rolling horizon.
    mobility_window = (config.duration if args.mobility_source_window_hours is None else
                       dt.timedelta(hours=args.mobility_source_window_hours))
    mobility = build_ev_schedules(replace(config, duration=mobility_window + horizon), home)
    selected_mobility = mobility[ev.vehicle_id]
    selected_mobility.demand(
        ev.hems.consumption_kwh_per_km,
        soc_depletion_capacity_kwh=ev.hems.soc_depletion_capacity_kwh(
            ev.capacity_kwh * ev.hems.initial_soh),
    )
    if args.one_shot:
        ev = replace(ev, hems=replace(
            ev.hems, horizon_hours=args.duration_hours,
            solver_time_limit_seconds=None, allow_unlimited_solver_time=True,
        ))
        home = replace(home, evs=(ev,))
        config = replace(config, homes=(home,))
    truth = None
    if args.perfect_load_truth_csv is not None:
        truth = pd.read_csv(args.perfect_load_truth_csv)
        times = pd.to_datetime(truth.pop("timestamp"), utc=True)
        truth.index = pd.DatetimeIndex(times).tz_convert(pd.Timestamp(config.start_time).tz)
    source_start = pd.Timestamp(args.source_start)
    duration = pd.Timedelta(config.duration)
    prepared = prepare_home_hub_replay(source_start - WARMUP, source_start + duration)
    warmup = pd.Timedelta(hours=args.native_warmup_hours)
    if warmup > WARMUP:
        from examples.single_home_hems.single_home_hems.study_native_warmup import prepare_home_hub_native_history
        extra = prepare_home_hub_native_history(source_start - warmup, source_start - WARMUP)
        prepared = PreparedHomeHubReplay(
            pd.concat([extra.power_kw, prepared.power_kw]),
            pd.concat([extra.meter_quality, prepared.meter_quality]),
            extra.meter_quality_counts.add(prepared.meter_quality_counts, fill_value=0),
            pd.concat([extra.split_audit_w, prepared.split_audit_w]),
        )
    split = map_replay_to_simulation(prepared, pd.Timestamp(config.start_time), source_start)
    expected = pd.date_range(pd.Timestamp(config.start_time) - warmup,
                             pd.Timestamp(config.start_time) + duration,
                             freq="min", inclusive="left")
    if not split.index.equals(expected):
        raise ValueError("Mapped Hub split does not cover every warm-up/evaluation minute")
    if args.horizon_hours == 20:
        # Outside-Hub (and explicit legacy aggregate Hub) forecasts still use
        # a prior-day tail. The default submeter Hub RF predicts all 80 leads.
        prior_complete_local_day(
            split.loc[:, ["hub_other_power_kw", "outside_hub_power_kw"]],
            pd.Timestamp(config.start_time),
        )
    if args.hub_forecast_mode == "perfect":
        hub_provider = PerfectHomeHubForecast(truth)
        hub_provider.issue(pd.Timestamp(config.start_time), pd.DataFrame(), horizon, ev.hems.max_discharge_kw)
        hub_provider.history.clear()
    elif args.hub_forecast_mode == "submeter_rf" and args.hub_seed_csv is not None:
        hub_provider = SubmeterHubRFProvider.from_artifact(
            args.hub_rf_artifact,
            seed_history=_read_non_ev_hub_seed(args.hub_seed_csv),
            seed_circuit_history=_read_hub_circuit_seed(args.hub_seed_csv),
            ambient_history=_observed_ambient_15min(
                Path(config.weather_file), pd.Timestamp(config.start_time).tz,
                pd.Timestamp(config.start_time), pd.Timestamp(config.start_time) + duration,
            ),
            ev_name=f"EV_{ev.vehicle_id}",
        )
        hub_provider.issue(pd.Timestamp(config.start_time), pd.DataFrame(),
                           horizon, ev.hems.max_discharge_kw)
        hub_provider.history.clear()
    elif args.hub_forecast_mode == "corrected_rf" and args.hub_seed_csv is not None:
        hub_provider = EligibleHubRFProvider.from_artifact(
            args.hub_rf_artifact,
            seed_history=_read_non_ev_hub_seed(args.hub_seed_csv),
            ambient_history=_observed_ambient_15min(
                Path(config.weather_file), pd.Timestamp(config.start_time).tz,
                pd.Timestamp(config.start_time), pd.Timestamp(config.start_time) + duration,
            ),
            ev_name=f"EV_{ev.vehicle_id}",
            tail_policy=("prior_complete_local_day" if args.horizon_hours == 20 else "none"),
        )
        # Only the RF prefix can be checked before the native warm-up has
        # produced completed Hub and EV meter readings for the 20-hour tail.
        hub_provider.issue(pd.Timestamp(config.start_time), pd.DataFrame(),
                           min(horizon, HORIZON), ev.hems.max_discharge_kw)
        hub_provider.history.clear()
    elif args.hub_forecast_mode == "issued_csv" and args.hub_issued_forecast_csv is not None:
        hub_provider = IssuedEVHubForecasts(args.hub_issued_forecast_csv)
    else:
        hub_provider = None
    issuance_count = 1 if args.one_shot else len(pd.date_range(
        pd.Timestamp(config.start_time), pd.Timestamp(config.start_time) + duration,
        freq=CONTROL_STEP, inclusive="left",
    ))
    if isinstance(hub_provider, IssuedEVHubForecasts):
        checked = validate_hub_issuance(
            hub_provider, pd.Timestamp(config.start_time), duration,
            ev.hems.max_discharge_kw, horizon,
        )
        if checked != issuance_count:
            raise AssertionError("Home Hub issuance count changed during validation")

    if args.validate_only:
        forecast_directory = tempfile.TemporaryDirectory(prefix="ev_home_hub_validate_")
        outside_path = Path(forecast_directory.name) / "outside_hub_forecast_source.csv"
    else:
        output.mkdir(parents=True, exist_ok=True)
        forecast_directory = None
        outside_path = output / "outside_hub_forecast_source.csv"
    ev_controller = None
    try:
        if args.outside_forecast_mode != "perfect":
            build_outside_hub_forecast_csv(outside_path)
        if args.outside_forecast_mode == "perfect":
            outside = PerfectOutsideHubForecast(truth)
        elif args.outside_forecast_mode == "rf_multires_weather_persistence":
            outside = MultiresWeatherOutsideHubRF.from_artifact(
                args.outside_rf_artifact,
                weather_file=Path(config.weather_file),
                simulation_start=pd.Timestamp(config.start_time),
                simulation_end=pd.Timestamp(config.start_time) + duration,
                tail_policy=("prior_complete_local_day" if args.horizon_hours == 20 else "none"),
            )
        elif args.outside_forecast_mode == "frozen_rf28_p90_28d":
            outside = FrozenSplitHubForecast.from_csv(
                outside_path,
                simulation_start=pd.Timestamp(config.start_time).tz_localize(None),
                source_start=source_start,
                p90_history_days=28,
                source_window=(source_start.normalize(),
                               (source_start + duration + horizon).ceil("D")),
            )
        else:
            outside = RestOfHomeHybridForecast.from_csv(
                outside_path,
                simulation_start=config.start_time,
                source_start=source_start,
                history_days=20,
                inference_gap_policy=args.outside_rf_inference_gap_policy,
            )
        # Forecast issuance reads completed source bins, never the prepared
        # future plant trace. Check all intended solve boundaries before OCHRE.
        outside_prior_issue_fallback_count = 0
        if not getattr(outside, "requires_total_home_meter", False):
            for issue_time in pd.date_range(
                pd.Timestamp(config.start_time), pd.Timestamp(config.start_time) + duration,
                freq=CONTROL_STEP, inclusive="left",
            ):
                forecast = outside.forecast_kw(issue_time, steps=int(horizon / CONTROL_STEP))
                outside_prior_issue_fallback_count += int(
                    bool(forecast.attrs.get("prior_issue_fallback", False))
                )
        if args.validate_only:
            print(json.dumps({
                "status": "validated",
                "home_id": args.home_id,
                "split_replay_minutes": len(split),
                "source_window_start": str(source_start - warmup),
                "source_window_end_exclusive": str(source_start + duration),
                "scheduled_ev_solves": issuance_count,
                "outside_forecast_method": outside.method_label,
                "outside_prior_issue_fallback_count": outside_prior_issue_fallback_count,
                "outside_dynamic_issue_rule": (
                    "At each solve, use only completed total-home minus corrected Hub aggregate "
                    "meter minutes and completed observed weather; no future plant readings"
                    if isinstance(outside, MultiresWeatherOutsideHubRF) else None
                ),
                "hub_forecast_mode": args.hub_forecast_mode,
                "hub_forecast_file": (str(hub_provider.path)
                                      if isinstance(hub_provider, IssuedEVHubForecasts)
                                      else str(args.hub_rf_artifact) if hub_provider is not None else None),
            }, indent=2))
            return 0

        if hub_provider is None:
            raise AssertionError("A simulation requires an issued Home Hub provider")
        split.to_csv(output / "home_hub_split_simulation_kw.csv")
        prepared.power_kw.to_csv(output / "home_hub_split_source_kw.csv")
        prepared.meter_quality.to_csv(output / "home_hub_meter_quality.csv")
        prepared.meter_quality_counts.to_csv(output / "home_hub_meter_quality_counts.csv")
        prepared.split_audit_w.to_csv(output / "home_hub_split_audit_w.csv")

        ev_name = f"EV_{ev.vehicle_id}"
        mobility_provider = (PerfectMobilityEVForecast(
                                ev.vehicle_id,
                                perfect_tariff=(args.hub_forecast_mode == "perfect" and args.outside_forecast_mode == "perfect"),
                            )
                             if args.mobility_forecast_mode == "perfect"
                             else ScenarioEVForecast(ev.vehicle_id, ev_control))
        ev_controller = EVHomeHubController(
            ev, hub_provider, outside,
            peak_weight_usd_per_kw=args.peak_weight_usd_per_kw,
            peak_limit_kw=args.peak_limit_kw,
            peak_window=args.peak_window,
            scenario_count=scenario_count,
            rate_structure=args.rate_structure,
            one_shot=args.one_shot,
            grid2_demand_formulation=args.grid2_demand_formulation,
            solver_backend=args.solver_backend,
            hub_mobility_split_policy=args.hub_mobility_split_policy,
            peak_arrival_comfort_enabled=(args.horizon_hours == 20 and not args.one_shot),
            hard_panel_limit_kw=(
                38.0 if args.horizon_hours == 20 and not args.one_shot
                and args.rate_structure != MISO_RTP_DOE else None
            ),
        )
        progress_path = output / "ev_home_hub_progress.json"
        progress_ledger = output / "ev_home_hub_progress.jsonl"

        def record_progress(row):
            progress = {**row, "wall_elapsed_seconds": time.perf_counter() - started}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2, default=str) + "\n")
            temporary.replace(progress_path)
            with progress_ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, default=str) + "\n")

        ev_controller.progress_callback = record_progress
        trace_files = {
            "forecast_inputs": output / "ev_forecast_inputs_15min.csv",
            "ev_scenario_inputs": output / "ev_scenario_inputs.csv",
            "scenario_predictions": output / "ev_scenario_predictions_15min.csv",
            "objective_components": output / "ev_objective_components.csv",
            "scenario_objective_components": output / "ev_scenario_objective_components.csv",
        }

        def record_trace(kind: str, rows: pd.DataFrame) -> None:
            """Append each issue as it occurs, preserving diagnostics on interruption."""
            path = trace_files.setdefault(kind, output / f"ev_{kind}.csv")
            rows.to_csv(path, mode="a", header=not path.exists(), index=False)

        ev_controller.trace_callback = record_trace
        settings = {
            "description": "EV-only MILP; native OCHRE HVAC/HPWH; approved parametric-study extensions",
            "simulation_start": str(config.start_time),
            "source_evaluation_start": str(source_start),
            "duration_hours": args.duration_hours,
            "warmup_hours": args.native_warmup_hours,
            "mobility_source_window_hours": mobility_window.total_seconds() / 3600,
            "rate_selection_duration_hours": rate_window.total_seconds() / 3600,
            "prediction_step_minutes": 15,
            "horizon_hours": horizon / pd.Timedelta(hours=1),
            "normalization_steps": int(horizon / CONTROL_STEP),
            "one_shot": args.one_shot,
            "rate_structure": rate_selection.metadata,
            "mobility_forecast_mode": args.mobility_forecast_mode,
            "ev_scenario_count": scenario_count,
            "ev_scenario_method": ("perfect" if args.mobility_forecast_mode == "perfect" else args.ev_scenario_method),
            "peak_window": ev_controller.peak_window,
            "peak_limit_kw": ev_controller.peak_limit_kw,
            **_peak_objective_metadata(ev_controller, rate_selection.schedule),
            "peak_weight_usd_per_kw": ev_controller.peak_weight_usd_per_kw,
            "grid2_demand_aggregation": ("worst_complete_scenario_v1" if ev_controller.uses_daily_grid2_demand else None),
            "grid2_demand_window_step_minutes": (15 if args.rate_structure == WINTER_TOU_DEMAND_CPP else None),
            "grid2_demand_formulation": ev_controller.grid2_demand_formulation,
            "solver_relative_gap": ev.hems.mip_relative_gap,
            "solver_time_limit_seconds": ev.hems.solver_time_limit_seconds,
            "ev_variant": "existing milp_health weights/dynamics; horizon recorded explicitly",
            "hub_observability_variant": (
                "EV plus water-heater, outdoor heat-pump and indoor-fan power submeters; corrected Hub aggregate"
                + (" and total-home aggregate" if ev_controller.include_total_home_meter else "")
                if ev_controller.include_circuit_meters else
                "EV submeter plus corrected Hub and total-home aggregate meters"
                if ev_controller.include_total_home_meter else "only EV submetered"
            ),
            "hub_meter_column": HUB_POWER_KEY,
            "controller_observation_columns": list(observation_keys(
                ev_name, include_total_home=ev_controller.include_total_home_meter,
                include_circuits=ev_controller.include_circuit_meters,
            )),
            "hub_meter_definition": (
                "HVAC heating main + HVAC heating fan + HVAC cooling electric + "
                "water heating + Hub-other + signed EV AC power; excludes HVAC resistance"
            ),
            "ev_discharge_eligible_loads": list(HOME_HUB_BACKED_LOADS),
            "ev_discharge_control": "fixed discharge mode; actual OCHRE minute power clipped to eligible Hub load",
            "ev_charge_control": "fixed charger nameplate while charge mode is active",
            "hub_forecast_mode": args.hub_forecast_mode,
            "hub_forecast_path": str(args.perfect_load_truth_csv if args.hub_forecast_mode == "perfect" else
                                     hub_provider.path if isinstance(hub_provider, IssuedEVHubForecasts)
                                     else args.hub_rf_artifact),
            "hub_forecast_sha256": hashlib.sha256(
                (args.perfect_load_truth_csv if args.hub_forecast_mode == "perfect" else
                 hub_provider.path if isinstance(hub_provider, IssuedEVHubForecasts)
                 else args.hub_rf_artifact).read_bytes()
            ).hexdigest(),
            "hub_seed_path": str(args.hub_seed_csv) if args.hub_seed_csv is not None else None,
            "hub_seed_sha256": (
                hashlib.sha256(args.hub_seed_csv.read_bytes()).hexdigest()
                if args.hub_seed_csv is not None else None
            ),
            "hub_seed_rule": (
                "Matched native pretest non-EV Hub and three circuit power histories with HVAC ER removed"
                if isinstance(hub_provider, SubmeterHubRFProvider) else
                "Matched native pretest non-EV Hub aggregate with HVAC ER removed"
                if isinstance(hub_provider, EligibleHubRFProvider)
                and hub_provider.metadata.get("hub_target", "").endswith("excluding HVAC resistance")
                else "Matched native pretest non-EV Hub aggregate"
            ),
            "hub_rf_metadata": (hub_provider.metadata if isinstance(hub_provider, (EligibleHubRFProvider, SubmeterHubRFProvider))
                                else None),
            "hub_forecast_issuance_count_required": issuance_count,
            "outside_forecast_method": outside.method_label,
            "outside_forecast_metadata": outside.metadata,
            "outside_rf_artifact": (
                str(args.outside_rf_artifact) if isinstance(outside, MultiresWeatherOutsideHubRF)
                else None
            ),
            "outside_forecast_target": (
                "completed total-home aggregate minus corrected Home Hub aggregate; includes HVAC electric resistance"
                if ev_controller.include_total_home_meter
                else "fixed Outside-Hub source without HVAC electric resistance"
            ),
            "outside_prior_issue_fallback_count": outside_prior_issue_fallback_count,
            "diagnostic_trace_files": {name: str(path) for name, path in trace_files.items()},
            "diagnostic_trace_semantics": (
                f"Forecast inputs are saved before solving on the {int(horizon / CONTROL_STEP)}-step issue-time grid; "
                "EV mobility, tariff, and eligible-load scenario segments are also saved before solving; "
                "scenario predictions and objective costs are saved only for accepted solves. "
                "Objective costs are horizon-normalized optimization terms, not metered bills. "
                "EV-only Home Hub excludes HVAC resistance and Outside-Hub includes it when "
                "rf_multires_weather_persistence is selected."
            ),
            "water_draw_start": str(args.water_draw_start),
            "mobility_metadata": mobility[ev.vehicle_id].metadata,
            "objective_weights": {key: getattr(ev.hems, key) for key in (
                "electricity_weight", "health_weight", "health_soc_weight", "comfort_weight",
                "trip_shortfall_weight",
                "persistence_weight", "health_soc_min", "health_soc_max", "comfort_soc_min",
                "enforce_terminal_soc", "enable_degradation", "enforce_cycle_limit",
                "enforce_dwell_constraints", "enforce_discharge_budget", "battery_model",
            )},
            "arguments": _jsonable_args(args),
            "effective_ev_config": asdict(ev),
            "solver_backend": args.solver_backend,
            "hub_mobility_split_policy": args.hub_mobility_split_policy,
            "peak_arrival_comfort_enabled": bool(ev_controller.peak_arrival_comfort_enabled),
            "hard_panel_limit_kw": ev_controller.hard_panel_limit_kw,
            "ev_panel_guard_limit_kw": (
                38.0 if args.horizon_hours == 20 and not args.one_shot
                and args.rate_structure != MISO_RTP_DOE else None
            ),
            "hub_mobility_split_semantics": (
                "Hold each issued 15-minute raw/capped mean and full-bin minimum on native EV event subsegments; "
                "integrate exact durations; no reconstructed minute load profile"
                if args.hub_mobility_split_policy == "hold_parent_interval_coefficients"
                else "Require complete 15-minute Hub forecast bins"
            ),
        }
        (output / "ev_home_hub_settings.json").write_text(
            json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8",
        )
        trip_audit = mobility[ev.vehicle_id].trips.copy(deep=True)
        trip_audit["requested_trip_energy_kwh"] = mobility[ev.vehicle_id].demand(
            ev.hems.consumption_kwh_per_km,
            soc_depletion_capacity_kwh=ev.hems.soc_depletion_capacity_kwh(
                ev.capacity_kwh * ev.hems.initial_soh),
        ).energy_kwh.to_numpy()
        trip_audit.to_csv(output / "ev_realized_mobility_trips.csv", index=False)
        simulation = SingleHomeHEMSSimulation(
            config, home, HEMSController(device_controllers=(ev_controller,), price_schedule=rate_selection.schedule),
            water_heater_hp_only=True,
            initial_temperature_c=21.0,
            water_draw_file=args.water_draw_file,
            water_draw_start=pd.Timestamp(args.water_draw_start),
            save_optimizer_plan_figures=False,
            hvac_warmup_duration=warmup,
            ev_mobility_schedules=mobility,
            hub_split_replay_kw=split,
            resistance_outside_hub=True,
            ev_panel_guard_limit_kw=(
                38.0 if args.horizon_hours == 20 and not args.one_shot
                and args.rate_structure != MISO_RTP_DOE else None
            ),
            unified_context={
                "providers": {ev_name: mobility_provider},
                "backed_loads": HOME_HUB_BACKED_LOADS,
                "forecast_extension": pd.Timedelta(0) if args.one_shot else horizon,
            },
        )
        summary = simulation.run()
        _quarter_hour_power(summary.standardized_results).to_csv(
            output / "ev_only_15min_power.csv"
        )
        _quarter_hour_power(summary.uncontrolled_results).to_csv(
            output / "ev_only_uncontrolled_15min_power.csv"
        )
        observed_end = pd.Timestamp(config.start_time) + duration
        ev_controller._completed_history(observed_end).to_csv(
            output / "ev_controller_completed_hub_meter.csv"
        )
        pd.DataFrame(mobility_provider.history).to_csv(output / "ev_mobility_forecast_issuance.csv", index=False)
        pd.DataFrame(hub_provider.history).to_csv(output / "home_hub_forecast_issuance.csv", index=False)
        pd.DataFrame(ev_controller.issued_load_history).to_csv(
            output / "ev_load_forecast_issuance.csv", index=False,
        )
        if hasattr(outside, "history"):
            pd.DataFrame(outside.history).to_csv(
                output / "outside_hub_forecast_issuance.csv", index=False,
            )
        pd.DataFrame(ev_controller.solve_history).to_csv(output / "ev_solve_history.csv", index=False)
        pd.DataFrame(ev_controller.action_history).to_csv(output / "ev_action_history.csv", index=False)
        (output / "ev_accepted_plans.json").write_text(json.dumps(ev_controller.plans, indent=2, default=str) + "\n")

        price = summary.controls["electricity_price_usd_per_kwh"]
        demand_rate = float(getattr(rate_selection.schedule, "demand_charge_usd_per_kw_day", 0))
        anytime_demand_rate = float(getattr(rate_selection.schedule, "anytime_demand_charge_usd_per_kw_day", 0))
        metrics = {}
        for label, minute in (("controlled", summary.standardized_results),
                              ("uncontrolled", summary.uncontrolled_results)):
            metrics[label] = _metrics(
                minute, price, demand_rate,
                anytime_demand_rate_usd_per_kw_day=anytime_demand_rate,
            )
            if demand_rate > 0 or anytime_demand_rate > 0:
                winter_daily_demand_bill(
                    minute["total_electric_power_kw"], rate_usd_per_kw_day=demand_rate,
                    anytime_rate_usd_per_kw_day=anytime_demand_rate,
                ).per_day.to_csv(
                    output / f"{label}_daily_demand.csv",
                )
            if args.rate_structure == MISO_RTP_DOE:
                compliance = evaluate_scenario3_doe(minute)
                compliance.intervals.to_csv(output / f"{label}_doe_15min.csv")
                metrics[label]["doe_compliance"] = compliance.summary
        start = pd.Timestamp(config.start_time)
        end = start + config.duration
        ev_service = {}
        for label, raw_path in (
            ("controlled", simulation.raw_directory / f"{home.home_id}.csv"),
            ("uncontrolled", simulation.raw_directory / "uncontrolled" / f"{home.home_id}.csv"),
        ):
            raw_ev = pd.read_csv(raw_path)
            clock = pd.DatetimeIndex(pd.to_datetime(raw_ev.pop("Time"), utc=True))
            raw_ev.index = clock.tz_convert(start.tz) if start.tz is not None else clock.tz_localize(None)
            raw_ev = raw_ev.loc[(raw_ev.index >= start) & (raw_ev.index < end)]
            ev_service[label] = ev_study_metrics(
                raw_ev, ev_name, health_soc_min=ev.hems.health_soc_min,
                health_soc_max=ev.hems.health_soc_max, comfort_soc_min=ev.hems.comfort_soc_min,
            )
            metrics[label]["ev_soc_end"] = ev_service[label]["evaluation_end_soc_fraction"]
            metrics[label]["ev_soh_end"] = ev_service[label]["evaluation_end_soh_fraction"]
        pd.DataFrame(ev_service).T.to_csv(output / "ev_service_and_battery_metrics.csv", index_label="case")
        threshold_metrics = (
            {
                "controlled": _threshold_metrics(
                    summary.standardized_results, args.peak_limit_kw, args.peak_window,
                ),
                "uncontrolled": _threshold_metrics(
                    summary.uncontrolled_results, args.peak_limit_kw, args.peak_window,
                ),
            }
            if args.peak_limit_kw is not None and args.rate_structure == "legacy" else None
        )
        start = pd.Timestamp(config.start_time)
        end = start + config.duration
        inventories = {
            "controlled": _ev_boundary_energy(
                simulation.raw_directory / f"{home.home_id}.csv", ev_name, start, end,
            ),
            "uncontrolled": _ev_boundary_energy(
                simulation.raw_directory / "uncontrolled" / f"{home.home_id}.csv",
                ev_name, start, end,
            ),
        }
        initial_value = _daily_average_tariff(simulation.controller.electricity_price_for_step, start)
        final_value = _daily_average_tariff(simulation.controller.electricity_price_for_step, end)
        raw_savings = metrics["uncontrolled"]["total_bill_usd"] - metrics["controlled"]["total_bill_usd"]
        inventory_adjustment = (
            final_value * (inventories["controlled"][1] - inventories["uncontrolled"][1])
            - initial_value * (inventories["controlled"][0] - inventories["uncontrolled"][0])
        )
        history = pd.DataFrame(ev_controller.solve_history)
        solve_seconds = (history["solve_seconds"].astype(float)
                         if "solve_seconds" in history else pd.Series(dtype=float))
        results = {
            "metrics": metrics,
            "ev_service": ev_service,
            "threshold_performance": threshold_metrics,
            "outside_prior_issue_fallback_count": outside_prior_issue_fallback_count,
            "electricity_comparison": {
                "raw_bill_savings_usd": raw_savings,
                "ev_inventory_adjustment_usd": inventory_adjustment,
                "ev_inventory_adjusted_bill_savings_usd": raw_savings + inventory_adjustment,
                "controlled_ev_energy_start_end_kwh": inventories["controlled"],
                "uncontrolled_ev_energy_start_end_kwh": inventories["uncontrolled"],
            },
            "aligned_peak_change_kw": (
                metrics["controlled"]["aligned_15_minute_peak_kw"]
                - metrics["uncontrolled"]["aligned_15_minute_peak_kw"]
            ),
            "ev_solver": {
                "solve_count": len(history),
                "status_counts": dict(Counter(history["status"])) if not history.empty else {},
                "fallback_count": int(history["fallback"].sum()) if "fallback" in history else 0,
                "mean_solve_seconds": float(solve_seconds.mean()) if not solve_seconds.empty else None,
                "total_solve_seconds": float(solve_seconds.sum()) if not solve_seconds.empty else None,
            },
            "total_wall_seconds": time.perf_counter() - started,
        }
        if args.horizon_hours == 20 and not args.one_shot and args.rate_structure != MISO_RTP_DOE:
            controls = summary.controls
            required = {"ev_panel_guard_tripped", "ev_panel_guard_active",
                        "ev_panel_guard_proposed_charge_kw", "ev_panel_guard_applied_charge_kw"}
            if not required <= set(controls):
                raise ValueError("EV panel guard diagnostics are missing from EV-only controls")
            results["ev_panel_guard"] = {
                "limit_kw": 38.0,
                "trip_minutes": int(controls.ev_panel_guard_tripped.astype(bool).sum()),
                "active_minutes": int(controls.ev_panel_guard_active.astype(bool).sum()),
                "proposed_charge_kwh": float(controls.ev_panel_guard_proposed_charge_kw.sum()/60),
                "applied_charge_kwh": float(controls.ev_panel_guard_applied_charge_kw.sum()/60),
            }
        (output / "ev_home_hub_results.json").write_text(
            json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8",
        )
        print(json.dumps({"output_directory": str(output), **results}, indent=2, default=str))
        return 0
    finally:
        if ev_controller is not None:
            ev_controller.close()
        if forecast_directory is not None:
            forecast_directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

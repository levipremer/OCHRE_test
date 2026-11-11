"""Matched EV and whole-home comparisons using the existing simulation loop."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import build_ev_schedules
from examples.five_home_block.five_home_block.output_mapping import standardize_results
from examples.single_home_hems.hems import HEMSController
from .simulation import SingleHomeHEMSSimulation
from .control_config import EV_CONTROL
from .controllers.ev import EVMPCController, EVScenarioController, daily_terminal_energy_value
from .forecast import Forecast
from .ev_scenarios import ScenarioEVForecast, scenario_settings
from .control_forecasts import ExternalForecast
from .backed_loads import backed_power
from .ev_metrics import ev_metrics, ev_operating_state
from .ev_forecast_metrics import score_forecasts
from .weather_forecast import ArchivedWeatherForecast, LOCAL_TIMEZONE
from .controllers.hvac import HVACPredictiveController

CASES = ("all_baseline", "non_v2h_twin", "advanced")
def validate_ev_models(home):
    """Reject legacy EV economics before creating outputs or running any case."""
    legacy = [ev.vehicle_id for ev in home.evs if ev.hems is None]
    if legacy:
        raise ValueError("EV comparison economics requires the new EV model with battery-aging "
                         f"settings (hems); legacy EVs: {', '.join(legacy)}")


def resolve_home_ev_controls(home, controls):
    """Apply outer EV controls to both the plant options and controller options."""
    validate_ev_models(home)
    vehicles = []
    for ev in home.evs:
        requested = controls["evs"][ev.vehicle_id]
        changes = {key: value for key, value in requested.items() if key in EV_CONTROL}
        backend = changes.get("solver_options", {})
        if "solver_time_limit_seconds" not in changes and (
            "TimeLimit" in backend or "time_limit" in backend or
            "time_limit" in backend.get("scipy_options", {})):
            changes["solver_time_limit_seconds"] = None
        changes["forecast_margin_hours"] = max(ev.hems.forecast_margin_hours,
                                                requested.get("forecast_hours", ev.hems.horizon_hours))
        vehicles.append(replace(ev, hems=replace(ev.hems, **changes)))
    return replace(home, evs=tuple(vehicles))


def make_controllers(home, controls, root, baseline=False):
    """Build only EV controllers. Native baseline charges to comfort_soc_min."""
    home = resolve_home_ev_controls(home, controls)
    controllers, providers = [], {}
    if baseline:
        return (), providers
    for ev in home.evs:
        c = controls["evs"][ev.vehicle_id]
        if c["controller"] != "mpc":
            continue
        if c["forecast"]["mode"] == "scenario":
            controllers.append(EVScenarioController(ev))
            providers[f"EV_{ev.vehicle_id}"] = ScenarioEVForecast(ev.vehicle_id, c)
        else:
            controllers.append(EVMPCController(ev))
        if c["forecast"]["mode"] == "deterministic":
            path = Path(c["forecast"]["file"])
            providers[f"EV_{ev.vehicle_id}"] = ExternalForecast(
                path if path.is_absolute() else root/path, f"EV_{ev.vehicle_id}", c["forecast_hours"])
    return tuple(controllers), providers


@dataclass
class UnifiedResult:
    """Matched case trajectories, dual-window metrics, and solver summaries."""
    cases: dict
    metrics: pd.DataFrame
    solver_stats: pd.DataFrame
    data_directory: Path


def _save_case(sim, result, case, controller, original_end):
    raw, commands, schedule, sensors, water_training, hvac_training = result
    original_start = pd.Timestamp(sim.config.start_time)
    index = raw.index[(raw.index >= original_start) & (raw.index < original_end)]
    # Audit trail retains every phase; comparison files score only [start, end).
    sim.data_directory.mkdir(parents=True, exist_ok=True)
    cooldown_end = sim.unified_context.get("cooldown_end", original_end)
    phase = np.where(raw.index < original_start, "warmup",
                     np.where(raw.index < original_end, "evaluation",
                              np.where(raw.index < cooldown_end, "cooldown", "forecast_tail")))
    pd.DataFrame({"in_comparison_window": raw.index.isin(index),
                  "in_excluding_warmup_window": raw.index.isin(index),
                  "in_including_warmup_window": raw.index < original_end,
                  "phase": phase}, index=raw.index).to_csv(
        sim.data_directory/"evaluation_mask.csv")
    raw.to_csv(sim.data_directory/"raw_results_all_phases.csv")
    commands.to_csv(sim.data_directory/"controls_all_phases.csv")
    raw, commands, sensors = raw.loc[index], commands.loc[index], sensors.loc[index]
    water_training, hvac_training = water_training.loc[index], hvac_training.loc[index]
    mapped = standardize_results(raw, sim.home.home_id, expected_ev_count=len(sim.home.evs)).data
    sim._validate_results(raw, mapped)
    directory, figures = sim.data_directory, sim.figure_directory
    directory.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, data in {"raw_results": raw, "standardized": mapped, "controls": commands,
                       "water_heater_sensors": sensors, "water_heater_training": water_training,
                       "hvac_training": hvac_training}.items():
        data.to_csv(directory/f"{name}.csv")
    pd.DataFrame({f"EV_{e.vehicle_id}": ev_operating_state(raw, f"EV_{e.vehicle_id}")
                  for e in sim.home.evs}, index=raw.index).to_csv(directory/"ev_states.csv")
    (directory/"water_heater_sensor_layout.json").write_text(
        json.dumps(sim.water_heater_sensor_layout.metadata(), indent=2), encoding="utf-8")
    (directory/"water_draw_metadata.json").write_text(json.dumps(sim.water_draw_metadata, default=str, indent=2), encoding="utf-8")
    histories, stats = [], []
    by_name = {c.equipment_name: c for c in controller.device_controllers
               if isinstance(c, EVMPCController)}
    for name in [f"EV_{e.vehicle_id}" for e in sim.home.evs]:
        device = by_name.get(name)
        history = pd.DataFrame(device.solve_history) if device else pd.DataFrame()
        if not history.empty:
            history = history.rename(columns={"duration_s": "solve_seconds"})
            times = pd.to_datetime(history.timestamp)
            history["in_comparison_window"] = (times >= original_start) & (times < original_end)
            history = history.assign(home_id=sim.home.home_id, case=case, device=name)
            histories.append(history)
            history.to_csv(directory/f"{name.replace(' ', '_')}_solve_history.csv", index=False)
        scored = history.loc[history.in_comparison_window] if not history.empty else history
        seconds = float(scored.solve_seconds.sum()) if not scored.empty else 0
        stats.append(dict(home_id=sim.home.home_id, case=case, device=name, result_window="excluding_warmup",
            attempts=len(scored), failures=int((~scored.successful).sum()) if not scored.empty else 0,
            mean_solve_seconds=seconds/len(scored) if len(scored) else 0,
            p95_solve_seconds=float(scored.solve_seconds.quantile(.95)) if len(scored) else 0,
            max_solve_seconds=float(scored.solve_seconds.max()) if len(scored) else 0,
            solve_seconds_per_sim_timestep=seconds/len(raw),
            mean_ocp_build_and_solve_seconds=float(scored.ocp_seconds.mean()) if "ocp_seconds" in scored else 0,
            max_ocp_build_and_solve_seconds=float(scored.ocp_seconds.max()) if "ocp_seconds" in scored else 0,
            ocp_seconds_per_sim_timestep=float(scored.ocp_seconds.sum())/len(raw) if "ocp_seconds" in scored else 0,
            backend_solve_seconds=float(scored.backend_solve_seconds.sum(min_count=1)) if "backend_solve_seconds" in scored else np.nan,
            timeouts=int(scored.timed_out.fillna(False).sum()) if "timed_out" in scored else 0,
            limits_reached=int(scored.limit_reached.fillna(False).sum()) if "limit_reached" in scored else 0,
            fallbacks=int(scored.fallback.fillna(False).sum()) if "fallback" in scored else 0,
            accepted_incumbents=int(scored.accepted_incumbent.fillna(False).sum()) if "accepted_incumbent" in scored else 0,
            rainflow_master_solves=int(scored.rainflow_iterations.sum()) if "rainflow_iterations" in scored else 0,
            max_rainflow_iterations=int(scored.rainflow_iterations.max()) if "rainflow_iterations" in scored else 0,
            max_rainflow_relative_gap=float(scored.rainflow_relative_gap.max()) if "rainflow_relative_gap" in scored else np.nan,
            max_rainflow_absolute_gap=float(scored.rainflow_absolute_gap.max()) if "rainflow_absolute_gap" in scored else np.nan,
            max_rainflow_error_usd=float(scored.rainflow_approximation_error_usd.max()) if "rainflow_approximation_error_usd" in scored else np.nan,
            max_relative_mip_gap=float(scored.relative_mip_gap.max()) if "relative_mip_gap" in scored else np.nan,
            max_mip_absolute_gap=float((scored.backend_primal_bound-scored.dual_bound).clip(lower=0).max())
                if {"backend_primal_bound", "dual_bound"} <= set(scored) else np.nan,
            rainflow_uncertified=int((~scored.rainflow_converged.dropna().astype(bool)).sum()) if "rainflow_converged" in scored else 0,
            solver=getattr(device, "solver", getattr(getattr(device, "options", None), "solver", "none"))))
        if isinstance(device, EVMPCController):
            pd.DataFrame(device.action_history).to_csv(directory/f"{name}_actions.csv", index=False)
            (directory/f"{name}_plans.json").write_text(json.dumps(device.plans, default=str, indent=2), encoding="utf-8")
    for name, provider in sim.unified_context.get("providers", {}).items():
        pd.DataFrame(provider.history).to_csv(directory/f"{name.replace(' ', '_')}_forecast_issuance.csv", index=False)
    return dict(raw=raw, mapped=mapped, sensors=sensors, schedule=schedule.loc[index],
                commands=commands, water_training=water_training,
                hvac_training=hvac_training,
                device_controllers=controller.device_controllers), stats, histories


def home_for_comparison_window(home, evaluation_start, simulation_start):
    """Keep measured-load source_start anchored to the public evaluation start."""
    rest = home.rest_of_home
    if rest.source != "dc_house_test":
        return home
    source = pd.Timestamp(rest.source_start)
    evaluation = pd.Timestamp(evaluation_start)
    if len(rest.source_start) == 10:
        source += evaluation - evaluation.normalize()
    source -= evaluation - pd.Timestamp(simulation_start)
    return replace(home, rest_of_home=replace(rest, source_start=source.isoformat()))


def run_home(unified, home, mobility=None, *, thermal_controller_factory=None,
             rest_of_home_forecast_factory=None, comparison_mode="ev",
             include_ev_reference=False, render_plots=True):
    """Run matched cases through the existing plant and controller implementations.

    The legacy EV interface holds thermal policies fixed. The central experiment
    interface selects ``whole_home`` to disable every predictive controller in
    its primary baseline. Thermal objects remain present solely to preserve the
    existing matched initialization, schedules and forecast context.
    """
    if comparison_mode not in {"ev", "whole_home"}:
        raise ValueError("comparison_mode must be ev or whole_home")
    validate_ev_models(home)
    config = unified.neighborhood
    run_config = unified.simulation_window()
    weather_archive = None
    if unified.weather_forecast_mode == "predict":
        if config.weather_forecast_file is None:
            raise ValueError("predict weather requires simulation.weather_forecast_file")
        weather_archive = ArchivedWeatherForecast.from_parquet(
            config.weather_forecast_file,
            local_timezone=(config.weather_metadata or {}).get("timezone", LOCAL_TIMEZONE),
        )
    control = unified.controls[home.home_id]
    home = resolve_home_ev_controls(home, control)
    ev_mpc = any(c["controller"] == "mpc" for c in control["evs"].values())
    if include_ev_reference and (comparison_mode != "whole_home" or not ev_mpc):
        raise ValueError("The optional EV-only reference requires whole-home mode and EV MPC")
    needs_twin = comparison_mode == "ev" or (ev_mpc and any(e.hems.v2h for e in home.evs))
    case_names = ("all_baseline",) + (("non_v2h_twin",) if needs_twin else ()) + ("advanced",)
    if include_ev_reference:
        case_names += ("ev_only_reference",)
    settings = unified.home_settings[home.home_id]
    root = config.output_directory / home.home_id
    case_directories = {
        case: (root / case if comparison_mode == "ev" else root / {
            "all_baseline": "cases/whole_home_baseline",
            "advanced": "cases/controlled",
            "non_v2h_twin": "internal/non_v2h_twin",
            "ev_only_reference": "references/ev_only_reference",
        }[case]) for case in case_names
    }
    original_start = pd.Timestamp(config.start_time)
    original_end = original_start+config.duration
    cooldown_end = pd.Timestamp(run_config.start_time)+run_config.duration
    hours = max([c["forecast_hours"] for c in list(control["evs"].values())
                 if c["controller"] == "mpc"], default=0)
    extension = pd.Timedelta(hours=hours)
    # Full twin trajectory is T+H; its final controller solve needs another H.
    schedule_evs = []
    for ev in home.evs:
        if ev.hems is None:
            schedule_evs.append(ev)
            continue
        c = control["evs"][ev.vehicle_id]
        history_days = ev.hems.history_days
        if c["controller"] == "mpc" and c["forecast"]["mode"] == "scenario":
            s = scenario_settings(c["forecast"], c["control_minutes"], c["horizon_hours"])
            history_days = max(history_days, s["lookback_days"] +
                (s["knn_history_hours"]/24 if s["scenario_method"] == "knn_state" else 0))
            grid = pd.Timedelta(minutes=s["scenario_minutes"])
            if pd.Timestamp(run_config.start_time) != pd.Timestamp(run_config.start_time).floor(grid):
                raise ValueError("EV scenario start must align with scenario_minutes")
        schedule_evs.append(replace(ev, hems=replace(ev.hems, history_days=history_days,
            forecast_margin_hours=max(ev.hems.forecast_margin_hours, 2*hours))))
    schedule_home = replace(home, evs=tuple(schedule_evs))
    mobility = mobility if mobility is not None else build_ev_schedules(run_config, schedule_home)
    for key, schedule in mobility.items():
        if schedule.start > pd.Timestamp(run_config.start_time) or schedule.end < cooldown_end+2*extension:
            raise ValueError(f"EV {key}: mobility must cover warm-up, evaluation, cooldown and twin forecast tail")
        c = control["evs"][key]
        if c["controller"] == "mpc" and c["forecast"]["mode"] == "scenario":
            ev = next(e for e in schedule_evs if e.vehicle_id == key)
            if schedule.start > pd.Timestamp(run_config.start_time)-pd.Timedelta(days=ev.hems.history_days):
                raise ValueError(f"EV {key}: mobility must cover the complete scenario lookback/history before warmup")
    cases, stats, histories, metrics = {}, [], [], []
    backed = None
    twin_home = replace(home, evs=tuple(replace(e, hems=replace(e.hems, v2h=False)) if e.hems else e for e in home.evs))
    # Preflight requested controllers/providers before generating any simulation outputs.
    make_controllers(home, control, config.repo_root)
    thermal_instances, forecast_instances, prepared_cases = [], [], []
    for case in case_names:
        selected_home = home_for_comparison_window(home if case == "advanced" else twin_home,
                                                  original_start, run_config.start_time)
        devices, providers = make_controllers(
            selected_home, control, config.repo_root,
            baseline=case in {"all_baseline", "ev_only_reference"})
        thermal = tuple(thermal_controller_factory(selected_home)) if thermal_controller_factory else ()
        if any(getattr(c, "equipment_name", "").startswith("EV_") for c in thermal):
            raise ValueError("thermal_controller_factory must only return existing non-EV controllers")
        if any(c is previous for c in thermal for previous in thermal_instances):
            raise ValueError("thermal_controller_factory must return fresh controller instances for each case")
        thermal_instances.extend(thermal)
        rest_forecast = rest_of_home_forecast_factory(selected_home) if rest_of_home_forecast_factory else None
        if rest_forecast is not None and any(rest_forecast is previous for previous in forecast_instances):
            raise ValueError("rest_of_home_forecast_factory must return fresh instances for each case")
        if rest_forecast is not None:
            forecast_instances.append(rest_forecast)
        if rest_forecast is None and any(getattr(c, "peak_weight_usd_per_kw", 0) > 0 for c in thermal):
            raise ValueError("Existing joint peak control requires rest_of_home_forecast_factory")
        controller = HEMSController(device_controllers=thermal + devices)
        prepared_cases.append((case, selected_home, devices, providers, controller, rest_forecast))
    if weather_archive is not None and not any(
        isinstance(device, HVACPredictiveController)
        or isinstance(getattr(device, "hvac_controller", None), HVACPredictiveController)
        for _, _, _, _, controller, _ in prepared_cases
        for device in controller.device_controllers
    ):
        raise ValueError("Predicted weather requires an HVAC predictive controller")
    for case, selected_home, devices, providers, controller, rest_forecast in prepared_cases:
        case_has_hvac = any(
            isinstance(device, HVACPredictiveController)
            or isinstance(getattr(device, "hvac_controller", None), HVACPredictiveController)
            for device in controller.device_controllers
        )
        case_config = replace(run_config, homes=(selected_home,), output_directory=case_directories[case],
            duration=run_config.duration+extension if case == "non_v2h_twin" else run_config.duration)
        water_file = settings.get("water_draw_file")
        if water_file:
            water_file = Path(water_file)
            if not water_file.is_absolute():
                water_file = config.repo_root / water_file
        sim = SingleHomeHEMSSimulation(case_config, selected_home, controller,
            water_heater_hp_only=settings.get("water_heater_hp_only", False),
            initial_temperature_c=settings.get("initial_temperature_c", 21.0),
            water_draw_file=water_file, water_draw_start=settings.get("water_draw_start"),
            water_draw_anchor_time=original_start, rest_of_home_forecast=rest_forecast,
            weather_archive=weather_archive if case_has_hvac else None,
            save_optimizer_plan_figures=settings.get("save_optimizer_plan_figures", False), ev_mobility_schedules=mobility, ev_backed_load=backed,
            unified_context={"forecast_extension": extension, "cooldown_end": cooldown_end,
                             "backed_loads": settings["backed_loads"], "providers": providers})
        apply_controls = not (comparison_mode == "whole_home" and case == "all_baseline")
        result = sim._run_stepwise(apply_controls=apply_controls, raw_directory=sim.raw_directory)
        if case == "non_v2h_twin":
            backed = pd.Series(backed_power(result[0], settings["backed_loads"]), index=result[0].index)
        # No plant/controller reset at evaluation start. Preserve its boundary state.
        sim.config = replace(case_config, start_time=config.start_time, duration=config.duration)
        data, case_stats, case_histories = _save_case(sim, result, case, controller, original_end)
        data["output_directory"] = case_directories[case]
        cases[case] = data
        stats.extend(case_stats)
        histories.extend(case_histories)
        data["forecast_errors"] = {}
        window_frames = []
        # Both windows slice one continuous plant/aging/controller trajectory.
        # Cooldown and the twin-only forecast tail never enter reported costs.
        for window, start in (("including_warmup", pd.Timestamp(run_config.start_time)),
                              ("excluding_warmup", original_start)):
            window_raw = result[0].loc[(result[0].index >= start) & (result[0].index < original_end)]
            window_commands = result[1].loc[window_raw.index]
            prior = result[0].loc[result[0].index < start]
            before = prior.iloc[-1] if not prior.empty else None
            frame = window_metrics(window_raw, window_commands, selected_home, devices, controller,
                                   mobility, case, window, start, original_end,
                                   config.time_step.total_seconds()/3600, previous=before)
            for ev in selected_home.evs:
                name = f"EV_{ev.vehicle_id}"
                provider = providers.get(name)
                if isinstance(provider, ScenarioEVForecast):
                    issuance, leads, scores = score_forecasts(provider.issued_forecasts,
                        mobility[ev.vehicle_id], ev.hems.consumption_kwh_per_km,
                        provider.control_minutes, start, original_end)
                    suffix = "" if window == "excluding_warmup" else "_including_warmup"
                    issuance.to_csv(sim.data_directory/f"{name}_forecast_errors{suffix}.csv", index=False)
                    leads.to_csv(sim.data_directory/f"{name}_forecast_errors_by_lead{suffix}.csv", index=False)
                    if window == "excluding_warmup":
                        data["forecast_errors"][name] = (issuance, leads)
                    for field, value in scores.items():
                        frame.loc[frame.device.eq(name), field] = value
            window_frames.append(frame)
        metrics.extend(pd.concat(window_frames, ignore_index=True).to_dict("records"))
    directory = root/"data"
    directory.mkdir(parents=True, exist_ok=True)
    for case, data in cases.items():
        pd.DataFrame({"actual_backed_load_kw": backed_power(data["raw"], settings["backed_loads"]),
                      "reference_backed_load_kw": backed.reindex(data["raw"].index) if backed is not None else 0.0},
                     index=data["raw"].index).to_csv(case_directories[case]/"data/backed_loads.csv")
    for key, schedule in mobility.items():
        schedule.trips.to_csv(directory/f"EV_{key}_trips.csv", index=False)
    (directory/"ev_replay_metadata.json").write_text(json.dumps({k: v.metadata for k, v in mobility.items()}, default=str, indent=2), encoding="utf-8")
    stats = pd.DataFrame(stats) if stats else pd.DataFrame(
        columns=["home_id", "case", "device", "result_window", "attempts", "failures"])
    stats.to_csv(directory/"solver_stats.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(directory/"solve_history.csv", index=False) if histories else (directory/"solve_history.csv").write_text("timestamp,home_id,case,device,status,successful,solve_seconds\n", encoding="utf-8")
    comparison = baseline_economics(pd.DataFrame(metrics))
    comparison.to_csv(directory/"comparison.csv", index=False)
    for case in case_names:
        ev_rows = comparison.loc[comparison.case.eq(case) & comparison.device.str.startswith("EV_")]
        ev_rows.to_csv(case_directories[case]/"data/ev_metrics.csv", index=False)
    if render_plots:
        from .unified_plots import plot_cases
        plot_cases(cases, home, root, config.plot_dpi, metrics=comparison)
    return UnifiedResult(cases, comparison, stats, directory)


def window_metrics(raw, commands, home, devices, controller, mobility, case, window,
                   start, end, step_hours, *, previous=None):
    """Score one half-open window without restarting the continuous aging ledger."""
    if raw.empty:
        raise ValueError("Result window has no plant timesteps")
    price = commands.electricity_price_usd_per_kwh
    ev_data = ev_metrics(raw, home, price, step_hours, devices,
                         previous=previous, mobility=mobility)
    tariff = Forecast(pd.DataFrame(), tariff_price_at=controller.electricity_price_for_step)
    v0 = daily_terminal_energy_value(tariff, raw.index[0])
    vT = daily_terminal_energy_value(tariff, raw.index[-1])
    rows = []
    for row in ev_data.to_dict("records"):
        rows.append(dict(row, device=f"EV_{row['vehicle_id']}",
            energy_kwh=row["net_energy_kwh"], peak_power_kw=row["peak_charge_kw"],
            cost_usd=row["operating_cost_usd"]))
    mapped = standardize_results(raw, home.home_id, expected_ev_count=len(home.evs)).data
    power = mapped.total_electric_power_kw
    whole = dict(device="whole_home", energy_kwh=float(power.sum()*step_hours),
                 peak_power_kw=float(power.max()), cost_usd=float((power*price).sum()*step_hours))
    for field in ("initial_stored_energy_kwh", "final_stored_energy_kwh",
                  "estimated_degradation_cost_usd", "reference_degradation_cost_usd"):
        whole[field] = float(ev_data[field].sum(min_count=1)) if len(ev_data) else 0.0
    rows.append(whole)
    frame = pd.DataFrame(rows)
    frame["home_id"], frame["case"], frame["result_window"] = home.home_id, case, window
    frame["window_start"], frame["window_end_exclusive"] = pd.Timestamp(start), pd.Timestamp(end)
    frame["window_duration_hours"] = (pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/3600
    frame["initial_energy_value_usd_per_kwh"], frame["final_energy_value_usd_per_kwh"] = v0, vT
    return frame


def baseline_economics(metrics):
    """Paired operating plus reference-aging savings versus all-baseline.

    Preserve unadjusted savings; add a separate boundary-inventory correction.
    Missing aging for non-battery devices means zero;
    missing aging for EV/whole-home rows is an error, not free degradation.
    """
    result = metrics.copy()
    aging = result.get("reference_degradation_cost_usd", pd.Series(np.nan, index=result.index))
    battery_rows = result.device.eq("whole_home") | result.device.str.startswith("EV_")
    if aging.loc[battery_rows].isna().any():
        raise ValueError("Reference degradation is required for EV and whole-home economics")
    result["reference_degradation_cost_usd"] = aging.fillna(0.0)
    group_keys = ["home_id", "device"] + (["result_window"] if "result_window" in result else [])
    baseline = result.loc[result.case.eq("all_baseline")].set_index(group_keys)
    def key(row):
        return tuple(getattr(row, field) for field in group_keys)
    if baseline.index.has_duplicates:
        raise ValueError("Economics requires one all-baseline row per home/device/result window")
    result["operating_cost_savings_usd"] = [float(baseline.loc[key(r), "cost_usd"])-r.cost_usd
                                             for r in result.itertuples()]
    result["degradation_cost_savings_usd"] = [
        float(baseline.loc[key(r), "reference_degradation_cost_usd"])-r.reference_degradation_cost_usd
        for r in result.itertuples()]
    result["net_savings_usd"] = result.operating_cost_savings_usd + result.degradation_cost_savings_usd
    fields = ("initial_stored_energy_kwh", "final_stored_energy_kwh",
              "initial_energy_value_usd_per_kwh", "final_energy_value_usd_per_kwh")
    corrections = []
    for row in result.itertuples():
        if row.device != "whole_home" and not row.device.startswith("EV_"):
            corrections.append(0.0)
            continue
        base = baseline.loc[key(row)]
        # Archived result tables lack boundary inventories: do not invent a credit.
        values = [getattr(row, f, np.nan) for f in fields] + [base.get(f, np.nan) for f in fields]
        if not np.isfinite(values).all():
            corrections.append(np.nan)
            continue
        e0, eT, v0, vT, b0, bT, bv0, bvT = values
        if not np.allclose([v0, vT], [bv0, bvT], rtol=0, atol=1e-12):
            raise ValueError("Inventory savings require common boundary valuations across cases")
        corrections.append(vT*(eT-bT)-v0*(e0-b0))
    result["energy_inventory_adjustment_usd"] = corrections
    result["energy_adjusted_net_savings_usd"] = result.net_savings_usd + result.energy_inventory_adjustment_usd
    result = result.drop(columns=["savings_usd", "net_profit_usd"], errors="ignore")
    return result


def run_unified(unified, home_ids=None, *, mobility_by_home=None, thermal_controller_factory=None,
                rest_of_home_forecast_factory=None):
    """Run the normal pipeline, optionally replaying study-frozen EV schedules."""
    config = unified.neighborhood
    selected = [h for h in config.homes if home_ids is None or h.home_id in home_ids]
    if not selected or (home_ids is not None and set(home_ids)-{h.home_id for h in selected}):
        raise ValueError("Unknown or empty selected homes")
    # Preflight all selected homes so a late unsupported backend cannot waste runs.
    for home in selected:
        make_controllers(home, unified.controls[home.home_id], config.repo_root)
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_directory}; choose a fresh run directory")
    save_configuration(unified, selected)
    if mobility_by_home is not None and set(mobility_by_home) != {h.home_id for h in selected}:
        raise ValueError("Frozen mobility must cover exactly the selected homes")
    results = {h.home_id: run_home(unified, h, None if mobility_by_home is None else mobility_by_home[h.home_id],
        thermal_controller_factory=thermal_controller_factory,
        rest_of_home_forecast_factory=rest_of_home_forecast_factory)
               for h in selected}
    write_reports(config, results)
    return results


def save_configuration(unified, selected):
    """Save requested/resolved settings and explicit scoring/forecast windows."""
    config = unified.neighborhood
    run_config = unified.simulation_window()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory/"resolved_configuration.json").write_text(json.dumps({
        "home_document": unified.home_document, "control_document": unified.control_document,
        "resolved_controls": unified.controls, "home_settings": unified.home_settings,
        "effective_ev_options": {home.home_id: {ev.vehicle_id: asdict(ev.hems)
            for ev in resolve_home_ev_controls(home, unified.controls[home.home_id]).evs}
            for home in selected},
        "selected_homes": [h.home_id for h in selected], "runtime": asdict(config),
        "evaluation_window": {"start": config.start_time, "end_exclusive": config.start_time+config.duration,
            "warmup_hours": unified.warmup_hours, "simulation_start": run_config.start_time,
            "cooldown_hours": unified.cooldown_hours,
            "simulation_end_exclusive": run_config.start_time+run_config.duration,
            "initial_soc_applies_at": "simulation_start"},
        "cases": list(CASES), "thermal_policy": "existing native controls or supplied fresh thermal_controller_factory; same in all cases"},
        default=str, indent=2), encoding="utf-8")


def write_reports(config, results):
    """Common root-level diagnostics for canonical and compatibility entry points."""
    directory = config.output_directory/"data"
    directory.mkdir(exist_ok=True)
    stats = pd.concat([r.solver_stats for r in results.values()], ignore_index=True)
    stats.to_csv(directory/"solver_stats.csv", index=False)
    pd.concat([pd.read_csv(r.data_directory/"solve_history.csv") for r in results.values()], ignore_index=True).to_csv(directory/"solve_history.csv", index=False)
    pd.concat([r.metrics for r in results.values()], ignore_index=True).to_csv(directory/"comparison.csv", index=False)
    for case in CASES:
        powers = pd.DataFrame({key: r.cases[case]["mapped"].total_electric_power_kw for key, r in results.items()})
        powers["whole_neighborhood_kw"] = powers.sum(axis=1)
        powers.to_csv(directory/f"{case}_aggregate.csv")
    report = config.output_directory/"reports"
    report.mkdir(exist_ok=True)
    metrics = pd.concat([r.metrics for r in results.values()], ignore_index=True)
    metrics.to_csv(directory/"performance.csv", index=False)
    lines = ["# EV comparison run report", "",
        "Each case is simulated continuously through warmup and evaluation. Result windows are "
        "including_warmup (simulation start to evaluation end) and excluding_warmup "
        "(evaluation start to evaluation end). Both exclude cooldown and the twin forecast tail. "
        "States and controller/aging memory are not reset at evaluation start. Boundary SOC, SOH, "
        "stored energy, damage, dates and duration accompany each window.", "",
        "Baseline charges immediately to comfort_soc_min. The twin uses the selected EV controller "
        "with V2H disabled. Existing thermal control policy is identical in all cases. "
        "Cycle counts measure realized starts; MILP constraints apply to its plan and convex actions "
        "are projected to discrete physical modes. Safety overrides and unmet trips are reported.", "",
        "Each window uses its paired baseline. Raw net savings includes electricity and reference "
        "battery aging. Inventory-adjusted savings adds vT*(case end energy-baseline end energy) "
        "minus v0*(case start energy-baseline start energy), using actual faded-capacity kWh and "
        "common daily-average tariffs. Raw and adjusted savings remain separate. "
        "Continuous aging ledger increments are summed only within the selected window.", "",
        "EV solver limits come from the effective configurable EV time budget; iterative solves "
        "share its remaining time. Backend/canonicalization overhead may overrun. Forecast scores "
        "use final issued scenarios including persistence prefixes; missing scenarios are flagged. "
        "Plots use evaluation-only trajectories; metric plots label their result window.", ""]
    for (home_id, case, window), frame in metrics.groupby(["home_id", "case", "result_window"], sort=False):
        lines += [f"## {home_id} / {case} / {window}", "", "| Device | Metric | Value |",
                  "| --- | --- | ---: |"]
        for row in frame.to_dict("records"):
            for key, value in row.items():
                if key in {"home_id", "case", "device", "vehicle_id"} or pd.isna(value):
                    continue
                display = f"{value:.7g}" if isinstance(value, (float, int)) else str(value)
                lines.append(f"| {row['device']} | {key} | {display} |")
        lines.append("")
    lines += ["## Solver statistics", ""]
    for row in stats.to_dict("records"):
        lines += [f"### {row['home_id']} / {row['case']} / {row['device']}", "",
                  "| Metric | Value |", "| --- | ---: |"]
        for key, value in row.items():
            if key in {"home_id", "case", "device"}:
                continue
            display = "unavailable" if pd.isna(value) else f"{value:.7g}" if isinstance(value, (float, int)) else str(value)
            lines.append(f"| {key} | {display} |")
        lines.append("")
    lines += ["", "Full numeric diagnostics: data/performance.csv, data/solver_stats.csv and data/solve_history.csv. "
              "Resolved configuration and mobility selections/seeds are saved with this run."]
    (report/"run_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

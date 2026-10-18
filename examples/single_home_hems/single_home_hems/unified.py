"""One three-case HEMS pipeline. No independent EV simulation loop."""
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp

from examples.five_home_block.five_home_block.ev_schedule import build_ev_schedules
from examples.five_home_block.five_home_block.output_mapping import standardize_results
from examples.single_home_hems.hems import HEMSController
from .simulation import SingleHomeHEMSSimulation
from .controllers.ev import EVMPCController, EVScenarioController, daily_terminal_energy_value
from .forecast import Forecast
from .ev_scenarios import ScenarioEVForecast, scenario_settings
from .controllers.water_heater import (WaterHeaterPredictiveController,
    DEFAULT_MODEL_PATH, DEFAULT_RC_MODEL_PATH, DEFAULT_RC_OLD_MODEL_PATH, DEFAULT_RC_STRATIFIED_MODEL_PATH)
from .control_forecasts import ExternalForecast
from .backed_loads import backed_power
from .ev_metrics import ev_metrics, ev_operating_state
from .ev_forecast_metrics import score_forecasts
from . import plotting

CASES = ("all_baseline", "non_v2h_twin", "advanced")
MODEL_PATHS = {"rc": DEFAULT_RC_MODEL_PATH, "rc-old": DEFAULT_RC_OLD_MODEL_PATH,
               "rc-stratified": DEFAULT_RC_STRATIFIED_MODEL_PATH, "state-space": DEFAULT_MODEL_PATH}


def make_controllers(home, controls, root, baseline=False):
    """Fresh state for every case. Baseline never checks/uses a solver."""
    if baseline:
        return (), {}
    controllers, providers = [], {}
    for ev in home.evs:
        c = controls["evs"][ev.vehicle_id]
        if c["controller"] == "mpc":
            if c["forecast"]["mode"] == "scenario":
                controllers.append(EVScenarioController(ev))
                providers[f"EV_{ev.vehicle_id}"] = ScenarioEVForecast(ev.vehicle_id, c)
            else:
                controllers.append(EVMPCController(ev))
    c = controls["hpwh"]
    if c["controller"] == "mpc":
        model = c["model"]
        if model not in MODEL_PATHS:
            raise ValueError(f"Unknown HPWH model: {model}")
        params = c["parameters"]
        # Reject non-preference internal/state fields and ambiguous duplicate options.
        allowed = {f.name for f in fields(WaterHeaterPredictiveController) if f.init} - {
            "equipment_name", "model_path", "model_kind", "optimization_kind", "forecast_type",
            "scenario_method", "solver", "solver_options", "horizon", "model_timestep", "control_interval"}
        if set(params) - allowed:
            raise ValueError(f"Invalid HPWH parameters: {sorted(set(params)-allowed)}")
        params = dict(params)
        for key, value in params.items():
            if "weight" in key and (not np.isfinite(value) or value < 0):
                raise ValueError(f"HPWH {key} must be finite and nonnegative")
        for key in ("dwell_time_on", "dwell_time_off", "legionella_cooldown_memory"):
            if key in params:
                params[key] = pd.Timedelta(params[key])
        if c["solver"] not in cp.installed_solvers():
            raise ValueError(f"HPWH solver {c['solver']} is not installed")
        model_path = Path(c.get("model_path") or MODEL_PATHS[model])
        if not model_path.is_absolute():
            model_path = root / model_path
        water = WaterHeaterPredictiveController(model_path=model_path,
            model_kind="rc" if model.startswith("rc") else "state-space",
            optimization_kind=c["formulation"], solver=c["solver"], solver_options=c["solver_options"],
            forecast_type="scenario" if c["forecast"]["mode"] == "scenario" else "perfect",
            scenario_method=c["forecast"].get("scenario_method", "daily_budget_high_rf_1"),
            horizon=pd.Timedelta(hours=c["horizon_hours"]), model_timestep=pd.Timedelta(minutes=c["model_minutes"]),
            control_interval=pd.Timedelta(minutes=c["control_minutes"]), **params)
        if water._load_model() is None:
            raise ValueError("HPWH predictor is missing or incompatible; cannot run requested MPC")
        controllers.append(water)
    for name, c in [("Water Heating", controls["hpwh"]),
                    *[(f"EV_{key}", value) for key, value in controls["evs"].items()]]:
        if c["controller"] == "mpc" and c["forecast"]["mode"] == "deterministic":
            path = Path(c["forecast"]["file"])
            providers[name] = ExternalForecast(path if path.is_absolute() else root/path, name, c["forecast_hours"])
    return tuple(controllers), providers


@dataclass
class UnifiedResult:
    """Three case trajectories plus evaluation-only metrics and solver summaries."""
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
    pd.DataFrame({"in_comparison_window": raw.index.isin(index), "phase": phase}, index=raw.index).to_csv(
        sim.data_directory/"evaluation_mask.csv")
    raw.to_csv(sim.data_directory/"raw_results_all_phases.csv")
    commands.to_csv(sim.data_directory/"controls_all_phases.csv")
    raw, commands, sensors = raw.loc[index], commands.loc[index], sensors.loc[index]
    water_training, hvac_training = water_training.loc[index], hvac_training.loc[index]
    mapped = standardize_results(raw, sim.home.home_id).data
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
    price = commands.electricity_price_usd_per_kwh
    plotting.plot_hvac(mapped, schedule, price, f"{sim.home.home_id}: {case}", figures/"hvac.png", sim.config.plot_dpi)
    plotting.plot_water_heater(mapped, raw, sensors, price, f"{sim.home.home_id}: {case}",
                              figures/"water_heater.png", sim.config.plot_dpi,
                              water_usage=water_training.water_heater_draw_l_per_min)
    histories, stats = [], []
    by_name = {c.equipment_name: c for c in controller.device_controllers}
    for name in ["HVAC", "Water Heating", *[f"EV_{e.vehicle_id}" for e in sim.home.evs]]:
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
        stats.append(dict(home_id=sim.home.home_id, case=case, device=name,
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
        if isinstance(device, WaterHeaterPredictiveController):
            pd.DataFrame(device.action_history).to_csv(directory/"hpwh_actions.csv", index=False)
            for field in ("objective_component_history", "scenario_diagnostics_history", "legionella_mask_history"):
                getattr(device, field).to_csv(directory/f"hpwh_{field}.csv", index=False)
            for i, plan in enumerate(device.optimization_plans):
                plan.to_csv(directory/f"hpwh_plan_{i:04d}.csv")
            if sim.save_optimizer_plan_figures:
                plotting.plot_water_heater_optimization_plans(device.optimization_plans, figures/"hpwh_plans", sim.config.plot_dpi)
    for name, provider in sim.unified_context.get("providers", {}).items():
        pd.DataFrame(provider.history).to_csv(directory/f"{name.replace(' ', '_')}_forecast_issuance.csv", index=False)
    return dict(raw=raw, mapped=mapped, sensors=sensors, schedule=schedule.loc[index],
                commands=commands, water_training=water_training), stats, histories


def run_home(unified, home, mobility=None):
    """Run matched baseline, no-V2H twin and advanced cases for one home."""
    config = unified.neighborhood
    run_config = unified.simulation_window()
    control = unified.controls[home.home_id]
    settings = unified.home_settings[home.home_id]
    root = config.output_directory / home.home_id
    original_start = pd.Timestamp(config.start_time)
    original_end = original_start+config.duration
    cooldown_end = pd.Timestamp(run_config.start_time)+run_config.duration
    hours = max([c["forecast_hours"] for c in [control["hpwh"], *control["evs"].values()]
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
    water_scenario = control["hpwh"]["controller"] == "mpc" and control["hpwh"]["forecast"]["mode"] == "scenario"
    if water_scenario and pd.Timestamp(run_config.start_time) != pd.Timestamp(run_config.start_time).floor("15min"):
        raise ValueError("HPWH scenario starts must align with local quarter hours")
    for case in CASES:
        selected_home = home if case == "advanced" else twin_home
        devices, providers = make_controllers(selected_home, control, config.repo_root, baseline=case == "all_baseline")
        controller = HEMSController(device_controllers=devices)
        case_config = replace(run_config, homes=(selected_home,), output_directory=root/case,
            duration=run_config.duration+extension if case == "non_v2h_twin" else run_config.duration)
        water_file = settings.get("water_draw_file")
        if water_scenario and not water_file:
            water_file = "examples/single_home_hems/external_data/water_flow_dchouse_l_per_min.csv"
        if water_file:
            water_file = Path(water_file)
            if not water_file.is_absolute():
                water_file = config.repo_root / water_file
        sim = SingleHomeHEMSSimulation(case_config, selected_home, controller,
            water_heater_hp_only=settings.get("water_heater_hp_only", False),
            initial_temperature_c=settings.get("initial_temperature_c", 21.0),
            water_draw_file=water_file, water_draw_start=settings.get("water_draw_start"),
            water_draw_anchor_time=original_start,
            save_optimizer_plan_figures=settings.get("save_optimizer_plan_figures", False), ev_mobility_schedules=mobility, ev_backed_load=backed,
            unified_context={"forecast_extension": extension, "cooldown_end": cooldown_end,
                             "backed_loads": settings["backed_loads"], "providers": providers})
        result = sim._run_stepwise(apply_controls=True, raw_directory=sim.raw_directory)
        if case == "non_v2h_twin":
            backed = pd.Series(backed_power(result[0], settings["backed_loads"]), index=result[0].index)
        # No plant/controller reset at evaluation start. Preserve its boundary state.
        preceding = result[0].loc[result[0].index < original_start]
        previous = preceding.iloc[-1] if not preceding.empty else None
        sim.config = replace(case_config, start_time=config.start_time, duration=config.duration)
        data, case_stats, case_histories = _save_case(sim, result, case, controller, original_end)
        cases[case] = data
        stats.extend(case_stats)
        histories.extend(case_histories)
        price = data["commands"].electricity_price_usd_per_kwh
        ev_data = ev_metrics(data["raw"], selected_home, price, config.time_step.total_seconds()/3600,
                             devices, previous=previous, mobility=mobility)
        # Common tariff-based inventory pricing is independent of OCP switches.
        tariff = Forecast(pd.DataFrame(), tariff_price_at=controller.electricity_price_for_step)
        v0 = daily_terminal_energy_value(tariff, data["raw"].index[0])
        vT = daily_terminal_energy_value(tariff, data["raw"].index[-1])
        ev_data["initial_energy_value_usd_per_kwh"] = v0
        ev_data["final_energy_value_usd_per_kwh"] = vT
        data["forecast_errors"] = {}
        for ev in selected_home.evs:
            name = f"EV_{ev.vehicle_id}"
            provider = providers.get(name)
            if isinstance(provider, ScenarioEVForecast):
                issuance, leads, scores = score_forecasts(provider.issued_forecasts,
                    mobility[ev.vehicle_id], ev.hems.consumption_kwh_per_km,
                    provider.control_minutes, original_start, original_end)
                issuance.to_csv(sim.data_directory/f"{name}_forecast_errors.csv", index=False)
                leads.to_csv(sim.data_directory/f"{name}_forecast_errors_by_lead.csv", index=False)
                data["forecast_errors"][name] = (issuance, leads)
                for field, value in scores.items():
                    ev_data.loc[ev_data.vehicle_id.eq(ev.vehicle_id), field] = value
        ev_data["case"] = case
        ev_data.to_csv(sim.data_directory/"ev_metrics.csv", index=False)
        for row in ev_data.to_dict("records"):
            metrics.append(dict(row, home_id=home.home_id, device=f"EV_{row['vehicle_id']}",
                energy_kwh=row["net_energy_kwh"], peak_power_kw=row["peak_charge_kw"], cost_usd=row["operating_cost_usd"]))
        for name, col in (("whole_home", "total_electric_power_kw"), ("hpwh", "water_heating_power_kw"), ("hvac", "hvac_power_kw")):
            power = data["mapped"][col]
            row = dict(home_id=home.home_id, case=case, device=name,
                energy_kwh=float(power.sum()*config.time_step.total_seconds()/3600),
                peak_power_kw=float(power.max()), cost_usd=float((power*price).sum()*config.time_step.total_seconds()/3600))
            step_hours = config.time_step.total_seconds()/3600
            if name == "whole_home":
                row.update(initial_energy_value_usd_per_kwh=v0, final_energy_value_usd_per_kwh=vT)
                for metric in ("initial_stored_energy_kwh", "final_stored_energy_kwh"):
                    row[metric] = float(ev_data[metric].sum()) if len(ev_data) else 0.0
                for metric in ("estimated_degradation_cost_usd", "reference_degradation_cost_usd"):
                    row[metric] = float(ev_data[metric].sum(min_count=1)) if metric in ev_data else 0.0
            def starts(on):
                # Detect transitions on the complete trajectory, then score T only.
                return int((on & ~on.shift(fill_value=False)).reindex(data["raw"].index).sum())
            if name == "hpwh":
                on = result[0]["Water Heating Mode"].eq("Heat Pump On")
                row["cycle_count"] = row["compressor_starts"] = starts(on)
                threshold = control["hpwh"].get("parameters", {}).get("minimum_temperature_c", 38.0)
                row["tank_average_shortfall_c_hours"] = float((threshold-data["mapped"].water_heater_temperature_c).clip(lower=0).sum()*step_hours)
                row["comfort_reference_c"] = threshold
                water = next((c for c in devices if isinstance(c, WaterHeaterPredictiveController)), None)
                actions = [a for a in water.action_history if original_start <= a["timestamp"] < original_end] if water else []
                row["projection_overrides"] = sum(a["projection_override"] for a in actions)
                row["physical_overrides"] = sum(a["physical_override"] for a in actions)
            elif name == "hvac":
                row["heating_starts"] = starts(result[0]["HVAC Heating Main Power (kW)"].gt(1e-7))
                row["cooling_starts"] = starts(result[0]["HVAC Cooling Main Power (kW)"].gt(1e-7))
                row["cycle_count"] = row["heating_starts"] + row["cooling_starts"]
                temp = data["mapped"].indoor_temperature_c
                row["heating_shortfall_c_hours"] = float((home.heating_setpoint_c-temp).clip(lower=0).sum()*step_hours)
                row["cooling_excess_c_hours"] = float((temp-home.cooling_setpoint_c).clip(lower=0).sum()*step_hours)
            metrics.append(row)
    directory = root/"data"
    directory.mkdir(parents=True, exist_ok=True)
    for case, data in cases.items():
        pd.DataFrame({"actual_backed_load_kw": backed_power(data["raw"], settings["backed_loads"]),
                      "reference_backed_load_kw": backed.reindex(data["raw"].index)}, index=data["raw"].index).to_csv(root/case/"data/backed_loads.csv")
    for key, schedule in mobility.items():
        schedule.trips.to_csv(directory/f"EV_{key}_trips.csv", index=False)
    (directory/"ev_replay_metadata.json").write_text(json.dumps({k: v.metadata for k, v in mobility.items()}, default=str, indent=2), encoding="utf-8")
    stats = pd.DataFrame(stats)
    stats.to_csv(directory/"solver_stats.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(directory/"solve_history.csv", index=False) if histories else (directory/"solve_history.csv").write_text("timestamp,home_id,case,device,status,successful,solve_seconds\n", encoding="utf-8")
    comparison = baseline_economics(pd.DataFrame(metrics))
    comparison.to_csv(directory/"comparison.csv", index=False)
    for case in CASES:
        ev_rows = comparison.loc[comparison.case.eq(case) & comparison.device.str.startswith("EV_")]
        ev_rows.to_csv(root/case/"data/ev_metrics.csv", index=False)
    from .unified_plots import plot_cases
    plot_cases(cases, home, root, config.plot_dpi, metrics=comparison)
    return UnifiedResult(cases, comparison, stats, directory)


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
    baseline = result.loc[result.case.eq("all_baseline")].set_index(["home_id", "device"])
    if baseline.index.has_duplicates:
        raise ValueError("Economics requires one all-baseline row per home/device")
    result["operating_cost_savings_usd"] = [float(baseline.loc[(r.home_id, r.device), "cost_usd"])-r.cost_usd
                                             for r in result.itertuples()]
    result["degradation_cost_savings_usd"] = [
        float(baseline.loc[(r.home_id, r.device), "reference_degradation_cost_usd"])-r.reference_degradation_cost_usd
        for r in result.itertuples()]
    result["net_savings_usd"] = result.operating_cost_savings_usd + result.degradation_cost_savings_usd
    fields = ("initial_stored_energy_kwh", "final_stored_energy_kwh",
              "initial_energy_value_usd_per_kwh", "final_energy_value_usd_per_kwh")
    corrections = []
    for row in result.itertuples():
        if row.device != "whole_home" and not row.device.startswith("EV_"):
            corrections.append(0.0)
            continue
        base = baseline.loc[(row.home_id, row.device)]
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


def run_unified(unified, home_ids=None, *, mobility_by_home=None):
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
    results = {h.home_id: run_home(unified, h, None if mobility_by_home is None else mobility_by_home[h.home_id])
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
        "selected_homes": [h.home_id for h in selected], "runtime": asdict(config),
        "evaluation_window": {"start": config.start_time, "end_exclusive": config.start_time+config.duration,
            "warmup_hours": unified.warmup_hours, "simulation_start": run_config.start_time,
            "cooldown_hours": unified.cooldown_hours,
            "simulation_end_exclusive": run_config.start_time+run_config.duration,
            "initial_soc_applies_at": "simulation_start"},
        "cases": list(CASES), "initialization": "measured initial states; repeated measured HPWH lifted history"},
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
    lines = ["# Unified HEMS run report", "",
        "All-baseline, non-V2H twin, and advanced results use the same comparison window. "
        "Comparison metrics and plots exclude controller-active warm-up, cooldown and the twin forecast tail; "
        "native raw outputs and detailed histories retain them. See resolved_configuration.json and evaluation_mask.csv. "
        "States and controller memory are not reset at evaluation start. EV aging metrics sum scored continuous-ledger "
        "increments, retaining prior cycle residue but excluding warm-up/cooldown increments. "
        "SOH and usable capacity evolve in every case; MPC freezes measured SOH only within each horizon. "
        "Nominal pack capacity sets dollar value; faded capacity sets SOC and stored energy. Initial damage is not billed again. "
        "Cooldown continues normal controls without a state reset or forced terminal recharge; it is not a terminal-SOC guarantee.", "",
        "Cycle count means realized operating starts, not battery equivalent-full cycles. "
        "A positive first sample counts only if it did not continue from warm-up. EV charge/discharge starts are separate; "
        "HPWH counts compressor starts, HVAC sums heating/cooling compressor starts. "
        "Safety interruptions and subsequent restarts count.", "",
        "EV MILP caps use historical ends plus planned starts: future=min(horizon,12h), history=24h-future. "
        "EV projection never enforces caps/dwell; convex EV has neither as hard constraints. "
        "HPWH OCP caps use horizon starts plus "
        "min(1, stops in the preceding hour); its projector uses that first-step budget. "
        "HPWH convex caps/dwell remain actuator-only. Physical safeties take precedence.", "",
        "Comfort metrics are realized degree-hours, not weighted OCP costs: HPWH uses "
        "average tank temperature below the reported reference; HVAC uses configured thermostat setpoints. "
        "EV health/comfort violations are SOC-fraction-hours. Compare final stored energy before claiming savings. "
        "EV baseline charges to comfort minimum. Full forecast trips can make the hard-bounded OCP infeasible; "
        "baseline fallback is logged. Plant clipping and unserved trip energy are reported separately.", "",
        "Solve-call timing includes canonicalization/backend overhead; OCP timing also includes preparation. "
        "Per-timestep averages divide total comparison-window time (including failed attempts) by plant steps. "
        "EV rainflow iterations share one 300 s elapsed-time budget, with remaining time passed to each backend. "
        "Other native backend limits are 300 s; backend stopping/canonicalization overhead may overrun. "
        "limits_reached can include non-time limits. Missing backend timings are unavailable, not zero.", "",
        "Operating-cost savings = all-baseline electricity cost minus case electricity cost. "
        "Degradation-cost savings = all-baseline NMC reference cost minus case NMC reference cost. Net savings is their sum; "
        "all-baseline net savings is zero. Unadjusted net savings excludes stored-energy credits. "
        "energy_adjusted_net_savings_usd adds vT*(case end energy-baseline end energy) "
        "minus v0*(case start energy-baseline start energy), using actual faded-capacity chemical kWh "
        "and common daily-minimum tariff values at the first/last evaluation timesteps. "
        "The correction is separate from realized electricity/aging costs and includes inventory "
        "revaluation if boundary prices differ. No twin-relative savings are used. "
        "Degradation dollars are unweighted and use new minus residual battery value; all models include driving "
        "and residue half cycles. NMC reference aging uses additive fade and signed SOC stress, not early-life "
        "nonlinear accumulation or measured true aging. The same battery calibration is used in all cases.", "",
        "EV scenario errors score the final issued forecast, including any persistence prefix. "
        "Availability is sampled at control-bin starts; arrival demand is distance-derived chemical kWh "
        "integrated in each bin, not served energy. Timestep means are duration-weighted over scored "
        "issuance/target pairs; overlapping forecasts intentionally retain distinct lead predictions. "
        "Horizon/next-new-trip scores require a complete horizon inside evaluation. No-scenario "
        "issuances are flagged and counted, not scored as zero error. Missing actual next trips have "
        "NaN MAE; missed predicted trips use zero energy when an actual next trip exists. "
        "Warmup/cooldown are excluded; forecast files retain sample counts.", ""]
    for (home_id, case), frame in metrics.groupby(["home_id", "case"], sort=False):
        lines += [f"## {home_id} / {case}", "", "| Device | Metric | Value |",
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

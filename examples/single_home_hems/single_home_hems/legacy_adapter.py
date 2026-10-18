"""Compatibility inputs delegate to the unified pipeline, never a second loop."""
from dataclasses import asdict, fields
from types import SimpleNamespace
import warnings
import pandas as pd

from .control_config import UnifiedConfiguration, resolve_control, EV_CONTROL
from .controllers.water_heater import WaterHeaterPredictiveController


def run_legacy(sim):
    """Adapt active legacy callers to three cases with zero phase padding."""
    warnings.warn("Legacy HEMS configuration is adapted in memory; prefer --home-config and --control-config", UserWarning)
    home = sim.home
    doc = {"homes": {home.home_id: {"evs": {}}}}
    entry = doc["homes"][home.home_id]
    for ev in home.evs:
        if ev.hems:
            entry["evs"][ev.vehicle_id] = {k: v for k, v in asdict(ev.hems).items() if k in EV_CONTROL}
    for c in sim.controller.device_controllers:
        if isinstance(c, WaterHeaterPredictiveController):
            excluded = {"equipment_name", "model_path", "model_kind", "optimization_kind", "forecast_type", "scenario_method",
                        "solver", "solver_options", "horizon", "model_timestep", "control_interval"}
            entry["hpwh"] = dict(controller="mpc", model=c.model_kind, model_path=str(c.model_path),
                formulation=c.optimization_kind, solver=c.solver, solver_options=dict(c.solver_options),
                horizon_hours=c.horizon.total_seconds()/3600, model_minutes=c.model_timestep.total_seconds()/60,
                control_minutes=c.model_timestep.total_seconds()/60,
                forecast={"mode": c.forecast_type, "scenario_method": c.scenario_method},
                parameters={f.name: getattr(c, f.name) for f in fields(c) if f.init and f.name not in excluded})
    controls = {name: resolve_control(doc, home.home_id, name) for name in ("hpwh", "hvac")}
    controls["evs"] = {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id) for ev in home.evs}
    settings = {"backed_loads": ["heat_pump", "water_heater"], "water_heater_hp_only": sim.water_heater_hp_only,
                "initial_temperature_c": sim.initial_temperature_c,
                "water_draw_file": sim.water_draw_file, "water_draw_start": sim.water_draw_start,
                "save_optimizer_plan_figures": sim.save_optimizer_plan_figures}
    config = UnifiedConfiguration(sim.config, {home.home_id: controls}, {home.home_id: settings}, {}, doc,
                                  warmup_hours=0, cooldown_hours=0)  # preserve the legacy entry point's window
    from .unified import run_home, write_reports, save_configuration
    save_configuration(config, [home])
    result = run_home(config, home, sim.ev_mobility_schedules)
    write_reports(sim.config, {home.home_id: result})
    root = sim.config.output_directory/home.home_id
    advanced, twin = result.cases["advanced"], result.cases["non_v2h_twin"]
    metrics = pd.concat([pd.read_csv(root/case/"data/ev_metrics.csv").assign(
        scenario={"all_baseline": "baseline", "non_v2h_twin": "twin", "advanced": "controlled"}[case])
        for case in result.cases], ignore_index=True)
    return SimpleNamespace(home_id=home.home_id, cases=result.cases, metrics=metrics,
        solver_stats=result.solver_stats, raw_results=advanced["raw"], standardized_results=advanced["mapped"],
        uncontrolled_results=twin["mapped"], data_directory=result.data_directory,
        raw_directory=root/"advanced/data/raw", figure_directory=root/"figures",
        figure_files=tuple(root.rglob("*.png")), peak_power_kw=float(advanced["mapped"].total_electric_power_kw.max()))

"""Opt-in EV comparison documents and independently resolved preferences.

The home document is self-contained; missing physics never consult legacy files.
EVOptions remains an internal compatibility adapter, not the new file schema.
"""
from dataclasses import asdict, dataclass, fields, replace
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import math

import pandas as pd

from examples.five_home_block.five_home_block.config import (configuration_from_documents,
    HomeConfig, EVConfig, ConfigurationError)
from examples.five_home_block.five_home_block.ev_config import EVOptions
from .backed_loads import validate_backed_loads
from .ev_scenarios import SCENARIO_DEFAULTS, scenario_settings

PHYSICAL = {"consumption_kwh_per_km", "charge_efficiency", "discharge_efficiency",
            "max_discharge_kw", "v2h", "battery_replacement_cost_per_kwh", "end_of_life_soh",
            "calibration_end_of_life_soh", "initial_soh",
            "battery_residual_value_per_kwh", "reference_cycle_depth", "reference_mean_soc",
            "rainflow_exponent", "reference_cycle_count", "soc_stress_coefficient"}
PHYSICAL_OPTIONAL = {"battery_model", "standby_vehicle_kw", "soc_depletion_reference_capacity_kwh"}
MOBILITY = {"source", "source_file", "source_vehicle", "seed", "history_days"}
MOBILITY_OPTIONAL = {"missing_soc_depletion_policy", "return_delay_minutes", "return_delay_collision_policy"}
EV_CONTROL = {f.name for f in fields(EVOptions)} - PHYSICAL - PHYSICAL_OPTIONAL - MOBILITY - MOBILITY_OPTIONAL - {
    "forecast_margin_hours", "health_min_soc", "health_max_soc", "health_low_weight", "health_high_weight", "trip_weight",
    "target_soc_min", "target_soc_max", "target_soc_low_weight", "target_soc_high_weight"}
COMMON = {"controller", "formulation", "solver", "solver_options", "horizon_hours",
          "forecast_hours", "control_minutes", "forecast"}


def merge(base, override):
    """Recursively copy overrides without mutating either configuration."""
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def keys(data, allowed, label):
    """Reject unknown configuration keys with their section name."""
    if not isinstance(data, dict):
        raise ConfigurationError(f"{label} must be an object")
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")


def resolve_control(document, home_id, device, vehicle_id=None):
    """Precedence: device defaults < home device preferences < vehicle preferences."""
    default = dict(controller="baseline", formulation="convex", solver="CLARABEL",
                   solver_options={}, horizon_hours=12, control_minutes=15,
                   forecast={"mode": "perfect"})
    home = document.get("homes", {}).get(home_id, {})
    if device == "ev":
        specific = merge(home.get("ev_defaults", {}), home.get("evs", {}).get(vehicle_id, {}))
    else:
        specific = home.get(device, {})
    result = merge(merge(default, document.get("defaults", {}).get(device, {})), specific)
    if device != "ev":
        requested = merge(document.get("defaults", {}).get(device, {}), specific)
        keys(requested, {"controller"}, f"{device} in EV-only configuration")
        if requested.get("controller", "baseline") != "baseline":
            raise ValueError("EV comparison JSON cannot configure HVAC/water-heater controllers; "
                             "use a fresh thermal_controller_factory with existing controller settings")
        return {"controller": "baseline"}
    result.setdefault("forecast_hours", result["horizon_hours"])
    keys(result, COMMON | EV_CONTROL, device)
    if result["controller"] not in {"baseline", "mpc"}:
        raise ValueError("controller must be baseline or mpc")
    if result["formulation"] not in {"convex", "milp"} or result["solver"] not in {"CLARABEL", "SCIPY", "GUROBI"}:
        raise ValueError("Invalid formulation/solver")
    if result["formulation"] == "milp" and result["solver"] == "CLARABEL":
        raise ValueError("CLARABEL does not support MILP")
    for name in ("horizon_hours", "forecast_hours", "control_minutes"):
        if not isinstance(result[name], (float, int)) or not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if result["forecast_hours"] < result["horizon_hours"]:
        raise ValueError("forecast_hours must cover horizon_hours")
    if result["horizon_hours"] * 60 < result["control_minutes"]:
        raise ValueError("horizon must cover the control interval")
    forecast = result["forecast"]
    keys(forecast, {"mode", "file", "scenario_method"} | set(SCENARIO_DEFAULTS), "forecast")
    if forecast["mode"] not in {"perfect", "deterministic", "scenario"}:
        raise ValueError("Unknown forecast mode")
    if result["controller"] == "mpc":
        if forecast["mode"] == "scenario":
            forecast.update(scenario_settings(forecast, result["control_minutes"], result["horizon_hours"]))
            if result["control_minutes"] % forecast["scenario_minutes"]:
                raise ValueError("EV scenario control interval must align with scenario_minutes")
        if forecast["mode"] == "deterministic" and not forecast.get("file"):
            raise ValueError("Deterministic mode requires an explicit external forecast file")
    return result


@dataclass
class UnifiedConfiguration:
    """Validated physical inputs, resolved EV controls and phase lengths."""
    neighborhood: object
    controls: dict
    home_settings: dict
    home_document: dict
    control_document: dict
    warmup_hours: float = 24.0
    cooldown_hours: float = 24.0
    weather_forecast_mode: str = "perfect"

    def simulation_window(self):
        """Controller-active pre/post-roll; public start/duration describe evaluation."""
        intervals = [self.neighborhood.time_step.total_seconds()]
        for home in self.controls.values():
            intervals += [c["control_minutes"] * 60 for c in list(home["evs"].values())
                          if c["controller"] == "mpc"]
        for name in ("warmup_hours", "cooldown_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if any(not math.isclose(value*3600 / step, round(value*3600 / step), abs_tol=1e-9) for step in intervals):
                raise ValueError(f"{name} must align with plant and active control intervals")
        warmup = pd.Timedelta(hours=self.warmup_hours)
        return replace(
            self.neighborhood,
            start_time=pd.Timestamp(self.neighborhood.start_time) - warmup,
            duration=(pd.Timedelta(self.neighborhood.duration) + warmup
                      + pd.Timedelta(hours=self.cooldown_hours)),
        )


def required(data, names, label):
    """Reject absent required fields; never consult legacy configuration files."""
    missing = set(names) - set(data)
    if missing:
        raise ConfigurationError(f"{label}: missing required fields {sorted(missing)}; no legacy configuration fallback")


def load_unified(home_file, control_file, repo_root):
    """Load strict standalone JSON; root _comments metadata is ignored."""
    homes_doc = json.loads(Path(home_file).read_text(encoding="utf-8"))
    control_doc = json.loads(Path(control_file).read_text(encoding="utf-8"))
    return load_unified_documents(homes_doc, control_doc, home_file, repo_root)


def load_unified_documents(homes_doc, control_doc, home_file, repo_root):
    """Validate copied in-memory documents with the same rules as JSON files.

    ``home_file`` supplies the physical-input path anchor; no temporary JSON
    files or changes to the individual EV runner are needed by experiments.
    """
    root = Path(repo_root).resolve()
    homes_doc, control_doc = deepcopy(homes_doc), deepcopy(control_doc)
    keys(homes_doc, {"simulation", "homes", "_comments"}, "home document")
    required(homes_doc, {"simulation", "homes"}, "home document")
    keys(control_doc, {"defaults", "homes", "_comments"}, "control document")
    keys(control_doc.get("defaults", {}), {"ev", "hpwh", "hvac"}, "control defaults")
    simulation = homes_doc["simulation"]
    sim_fields = {"start_time", "duration_hours", "time_step_minutes", "initialization_hours",
        "weather_file", "ev_trip_file", "NHTS_location_CDIVMSAR_ID", "output_directory",
        "random_seed", "verbosity", "plot_dpi", "numerical_tolerance_kw", "power_balance_tolerance_kw",
        "maximum_home_power_kw", "defrost_model", "defrost_control_type", "defrost_er_strategy", "transformer"}
    keys(simulation, sim_fields | {
        "warmup_hours", "cooldown_hours", "heating_season",
        "weather_metadata", "ground_mains_epw_file", "weather_interpolation",
        "weather_forecast_file", "weather_forecast_mode",
    }, "simulation")
    required(simulation, sim_fields, "simulation")
    weather_forecast_mode = simulation.get("weather_forecast_mode", "perfect")
    if weather_forecast_mode not in {"perfect", "predict"}:
        raise ValueError("simulation.weather_forecast_mode must be perfect or predict")
    if weather_forecast_mode == "predict" and not simulation.get("weather_forecast_file"):
        raise ValueError("predict weather requires simulation.weather_forecast_file")
    required(simulation["transformer"], {"power_rating_kva", "overload_capacity_percent"}, "transformer")
    if simulation["initialization_hours"] != 0:
        raise ValueError("Unified comparisons require zero initialization time")
    if simulation["time_step_minutes"] != 1:
        raise ValueError("EV comparisons require a one-minute plant timestep")
    start = dt.datetime.fromisoformat(simulation["start_time"])
    if start.second or start.microsecond:
        raise ValueError("EV comparison start_time must align to a whole minute")
    known = set(homes_doc["homes"])
    if set(control_doc.get("homes", {})) - known:
        raise ValueError("Unknown home ID in control configuration")
    home_fields = {f.name for f in fields(HomeConfig)} - {"home_id", "evs"}
    extras = {"evs", "backed_loads", "water_heater_hp_only", "water_draw_file", "water_draw_start",
              "initial_temperature_c", "save_optimizer_plan_figures"}
    ev_fields = {f.name for f in fields(EVConfig)} - {"vehicle_id", "hems"}
    resolved, settings, homes = {}, {}, []
    for home_id, spec in homes_doc["homes"].items():
        label = f"homes.{home_id}"
        keys(spec, home_fields | extras, label)
        required(spec, (home_fields - {"rest_of_home"}) | extras, label)
        selected = spec["backed_loads"]
        validate_backed_loads(selected)
        ctl_home = control_doc.get("homes", {}).get(home_id, {})
        keys(ctl_home, {"ev_defaults", "evs", "hpwh", "hvac"}, f"{label} controls")
        if set(ctl_home.get("evs", {})) - set(spec["evs"]):
            raise ValueError(f"Unknown EV ID for {home_id}")
        resolved[home_id] = {d: resolve_control(control_doc, home_id, d) for d in ("hpwh", "hvac")}
        resolved[home_id]["evs"] = {}
        vehicles = []
        for vehicle_id, entry in spec["evs"].items():
            ev_label = f"{label}.evs.{vehicle_id}"
            keys(entry, ev_fields | {"physical", "mobility"}, ev_label)
            required(entry, ev_fields | {"physical", "mobility"}, ev_label)
            for name in ("capacity_kwh", "charger_power_kw", "initial_soc_fraction", "vehicle_type", "work_from_home_probability"):
                if entry[name] is None:
                    raise ConfigurationError(f"{ev_label}.{name} must not be null")
            physical, mobility = entry["physical"], entry["mobility"]
            keys(physical, PHYSICAL | PHYSICAL_OPTIONAL, f"{ev_label}.physical")
            required(physical, PHYSICAL, f"{ev_label}.physical")
            keys(mobility, MOBILITY | MOBILITY_OPTIONAL, f"{ev_label}.mobility")
            required(mobility, MOBILITY, f"{ev_label}.mobility")
            if not mobility["source_file"]:
                raise ConfigurationError(f"{ev_label}.mobility.source_file must explicitly name the dataset")
            c = resolve_control(control_doc, home_id, "ev", vehicle_id)
            required(c, {"health_soc_min", "health_soc_max", "comfort_soc_min"}, f"{ev_label} controls")
            resolved[home_id]["evs"][vehicle_id] = c
            effective_mobility = dict(mobility)
            if c["controller"] == "mpc" and c["forecast"]["mode"] == "scenario":
                f = c["forecast"]
                # Reserve the complete candidate pool AND candidate kNN history
                # before random Audi replay selection (including warmup start).
                history = f["lookback_days"] + (f["knn_history_hours"]/24 if f["scenario_method"] == "knn_state" else 0)
                effective_mobility["history_days"] = max(mobility["history_days"], history)
            options = EVOptions(**physical, **effective_mobility,
                **{k: v for k, v in c.items() if k in EV_CONTROL},
                forecast_margin_hours=c["forecast_hours"])
            native = {k: entry[k] for k in ev_fields if entry[k] is not None}
            vehicles.append(dict(native, vehicle_id=vehicle_id, hems=asdict(options)))
        homes.append({**{k: spec[k] for k in home_fields if k in spec}, "home_id": home_id, "evs": vehicles})
        settings[home_id] = {k: spec[k] for k in extras if k != "evs"}
    config = configuration_from_documents(simulation, homes, Path(home_file).resolve().parent, root)
    for home_id, controls in resolved.items():
        for c in list(controls["evs"].values()):
            if (c["control_minutes"]*60) % config.time_step.total_seconds() or (c["horizon_hours"]*3600) % config.time_step.total_seconds():
                raise ValueError(f"{home_id}: control intervals/horizons must align with the plant step")
        for c in controls["evs"].values():
            if c["controller"] == "mpc" and c["forecast"]["mode"] == "scenario":
                if c["forecast"]["scenario_minutes"]*60 % config.time_step.total_seconds():
                    raise ValueError("EV scenario grid must align with the plant step")
    config.validate()
    unified = UnifiedConfiguration(
        config, resolved, settings, homes_doc, control_doc,
        simulation.get("warmup_hours", 24.0), simulation.get("cooldown_hours", 24.0),
        weather_forecast_mode=weather_forecast_mode,
    )
    window = unified.simulation_window()  # validate before any outputs are created
    import pandas as pd
    for controls in resolved.values():
        for c in controls["evs"].values():
            if c["controller"] == "mpc" and c["forecast"]["mode"] == "scenario":
                start = pd.Timestamp(window.start_time)
                if start != start.floor(pd.Timedelta(minutes=c["forecast"]["scenario_minutes"])):
                    raise ValueError("EV scenario start must align with scenario_minutes")
    return unified

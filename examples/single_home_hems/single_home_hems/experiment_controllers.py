"""Adapters to the existing independent and joint thermal controllers.

This module translates configuration only. Objectives, forecasts and dynamics
remain in the existing controllers and the existing joint-runner preset.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from .control_config import keys
from .controllers.hvac import HVACPredictiveController
from .controllers.water_heater import (
    WaterHeaterPredictiveController, DEFAULT_MODEL_PATH, DEFAULT_RC_MODEL_PATH,
    DEFAULT_RC_OLD_MODEL_PATH, DEFAULT_RC_STRATIFIED_MODEL_PATH,
)
from .controllers.water_heater_variants import hpwh_options_from_arguments
from .training.hvac import HVACInternalGainRegime


def jsonable(value):
    """Stable JSON representation of resolved settings, without runtime state."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (Path, dt.datetime, dt.date, dt.time, dt.timedelta, pd.Timedelta)):
        return str(value)
    return value


def _parameters(controller):
    return {field.name: deepcopy(getattr(controller, field.name))
            for field in fields(controller) if field.init}


def _defaults(device):
    # The public individual runner is the authority for its existing defaults.
    from examples.single_home_hems.run_simulation import build_device_controllers
    selection = "hvac" if device == "hvac" else "water-heater"
    return _parameters(build_device_controllers(selection, ())[0])


def _joint_defaults():
    from examples.single_home_hems.run_joint_hems import build_argument_parser
    args = build_argument_parser().parse_args(["--duration-hours", "1"])
    names = {
        "scenario_count", "hpwh_variant", "hpwh_comfort_weight", "hpwh_legionella_weight",
        "hpwh_persistence_weight", "hvac_comfort_band_c", "hvac_persistence_weight",
        "hvac_degree_hour_limit_c_hours", "hvac_degree_hour_slack_weight",
        "hvac_max_deviation_c", "peak_weight_usd_per_kw", "peak_window",
        "time_limit_seconds", "mip_gap", "solver_threads", "rest_rf_inference_gap_policy",
    }
    return {name: getattr(args, name) for name in names}


def _finite_positive(value, label, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")


def _validate_solver(device, parameters):
    solver, formulation = parameters["solver"], parameters["optimization_kind"]
    if solver not in {"GUROBI", "SCIPY", "CLARABEL"}:
        raise ValueError(f"Unsupported thermal solver: {solver}")
    if device == "hvac" and solver == "SCIPY" and not parameters["enforce_degree_hour_budget"]:
        raise ValueError("SCIPY does not support the HVAC norm2 comfort objective; select a conic-capable solver for this formulation")
    if formulation in {"milp", "mixed integer"} and solver == "CLARABEL":
        raise ValueError("CLARABEL does not support thermal mixed-integer control")


def resolve_thermal_control(device, request, *, base_directory, repo_root):
    """Validate and resolve one thermal selection to native constructor inputs."""
    request = deepcopy(request)
    if device == "joint":
        return _resolve_joint(request)
    parameters = _defaults(device)
    aliases = {"controller", "formulation", "horizon_hours", "control_minutes",
               "forecast", "model", "dwell_on_minutes", "dwell_off_minutes"}
    # Timedelta/native names are represented by explicit unit-bearing keys in JSON.
    excluded = {"horizon", "model_timestep", "dwell_time_on", "dwell_time_off"}
    allowed = (set(parameters) - excluded) | aliases
    keys(request, allowed, device)
    selected = request.pop("controller", "baseline")
    if selected not in {"baseline", "mpc"}:
        raise ValueError(f"{device}.controller must be baseline or mpc")
    for key in ("horizon_hours", "control_minutes", "dwell_on_minutes", "dwell_off_minutes"):
        if key in request:
            _finite_positive(request[key], f"{device}.{key}")
    for public, native, unit in (("horizon_hours", "horizon", "hours"),
                                 ("control_minutes", "model_timestep", "minutes"),
                                 ("dwell_on_minutes", "dwell_time_on", "minutes"),
                                 ("dwell_off_minutes", "dwell_time_off", "minutes")):
        if public in request:
            parameters[native] = pd.Timedelta(**{unit: request.pop(public)})
    if "formulation" in request:
        formulation = request.pop("formulation")
        if formulation not in {"convex", "milp"}:
            raise ValueError(f"{device}.formulation must be convex or milp")
        if "optimization_kind" in request:
            raise ValueError("Use formulation or optimization_kind, not both")
        parameters["optimization_kind"] = "mixed integer" if device == "hvac" and formulation == "milp" else formulation
    forecast = request.pop("forecast", None)
    if forecast is not None:
        keys(forecast, {"mode", "scenario_method"}, f"{device}.forecast")
        if "forecast_type" in request or "scenario_method" in request:
            raise ValueError("Use forecast or native forecast fields, not both")
        if device == "hvac":
            if forecast != {"mode": "perfect"}:
                raise ValueError("Existing HVAC controller supports only perfect forecasts")
        else:
            parameters["forecast_type"] = forecast.get("mode", "perfect")
            if "scenario_method" in forecast:
                parameters["scenario_method"] = forecast["scenario_method"]
    if "model" in request:
        model = request.pop("model")
        if "model_path" in request or "model_kind" in request:
            raise ValueError("Use model or explicit model_path/model_kind, not both")
        if device == "hvac":
            if model != "state-space":
                raise ValueError("Existing HVAC controller supports the regularized state-space model")
        else:
            paths = {"state-space": DEFAULT_MODEL_PATH, "rc": DEFAULT_RC_MODEL_PATH,
                     "rc-old": DEFAULT_RC_OLD_MODEL_PATH, "rc-stratified": DEFAULT_RC_STRATIFIED_MODEL_PATH}
            if model not in paths:
                raise ValueError(f"Unknown water-heater model: {model}")
            parameters.update(model_path=paths[model], model_kind="rc" if model.startswith("rc") else "state-space")
    if "gain_regime" in request:
        keys(request["gain_regime"], {f.name for f in fields(HVACInternalGainRegime)}, "hvac.gain_regime")
        request["gain_regime"] = HVACInternalGainRegime(**request["gain_regime"])
    parameters.update(request)
    if parameters.get("model_path") is not None:
        path = Path(parameters["model_path"]).expanduser()
        if not path.is_absolute():
            local = Path(base_directory) / path
            path = local if local.exists() else Path(repo_root) / path
        parameters["model_path"] = path.resolve()
    if not isinstance(parameters["solver_options"], dict):
        raise ValueError(f"{device}.solver_options must be an object")
    for key, value in parameters.items():
        if key.startswith("enforce_") or key.endswith("_enabled"):
            if not isinstance(value, bool):
                raise ValueError(f"{device}.{key} must be boolean")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"{device}.{key} must be finite")
            if ("weight" in key or key == "max_cycles") and value < 0:
                raise ValueError(f"{device}.{key} must be nonnegative")
    horizon, interval = parameters["horizon"], parameters["model_timestep"]
    if horizon <= pd.Timedelta(0) or interval <= pd.Timedelta(0) or horizon % interval:
        raise ValueError(f"{device}: horizon must contain complete positive control intervals")
    if interval % pd.Timedelta(minutes=1):
        raise ValueError(f"{device}: control interval must align with the one-minute plant")
    _validate_solver(device, parameters)
    if device == "water_heater" and parameters["forecast_type"] == "scenario":
        from .water_forecast import SCENARIO_METHODS
        if parameters["scenario_method"] not in SCENARIO_METHODS:
            raise ValueError("Unsupported water-heater scenario_method")
    cls = HVACPredictiveController if device == "hvac" else WaterHeaterPredictiveController
    cls(**parameters)  # Preserve each controller's existing validation.
    return {"controller": selected, "parameters": jsonable(parameters)}


def _resolve_joint(request):
    parameters = _joint_defaults()
    keys(request, set(parameters) | {"controller", "horizon_hours", "control_minutes", "solver", "solver_options"}, "joint")
    selected = request.pop("controller", "baseline")
    if selected not in {"baseline", "mpc"}:
        raise ValueError("joint.controller must be baseline or mpc")
    for name, expected in (("horizon_hours", 8), ("control_minutes", 15), ("solver", "GUROBI")):
        if name in request and request.pop(name) != expected:
            raise ValueError(f"Existing joint preset requires {name}={expected!r}")
    solver = request.pop("solver_options", {})
    keys(solver, {"TimeLimit", "MIPGap", "Threads"}, "joint.solver_options")
    for key, name in (("TimeLimit", "time_limit_seconds"), ("MIPGap", "mip_gap"), ("Threads", "solver_threads")):
        if key in solver:
            if name in request and request[name] != solver[key]:
                raise ValueError(f"Conflicting joint {name} and solver_options.{key}")
            parameters[name] = solver[key]
    parameters.update(request)
    for key, value in parameters.items():
        if key in {"hpwh_comfort_weight", "hpwh_legionella_weight", "hpwh_persistence_weight"} and value is None:
            continue
        if key.endswith("weight") or key in {"hvac_comfort_band_c", "hvac_degree_hour_limit_c_hours", "hvac_max_deviation_c", "peak_weight_usd_per_kw", "mip_gap"}:
            _finite_positive(value, f"joint.{key}", zero=True)
    from argparse import Namespace
    hpwh_options_from_arguments(Namespace(**parameters))
    _finite_positive(parameters["time_limit_seconds"], "joint.time_limit_seconds")
    if type(parameters["solver_threads"]) is not int or parameters["solver_threads"] < 1:
        raise ValueError("joint.solver_threads must be a positive integer")
    if type(parameters["scenario_count"]) is not int or parameters["scenario_count"] not in {5, 7, 10}:
        raise ValueError("Existing joint preset supports scenario_count 5, 7 or 10")
    if parameters["peak_window"] not in {"all", "tou", "night"}:
        raise ValueError("joint.peak_window must be all, tou or night")
    if parameters["rest_rf_inference_gap_policy"] not in {"strict", "causal_short_gap"}:
        raise ValueError("Unsupported rest_rf_inference_gap_policy")
    resolved = {"controller": selected, "parameters": jsonable(parameters),
                "preset": ("existing_joint_8h_scenario_milp_degree_hour"
                           if parameters["hpwh_variant"] == "persistence"
                           else "joint_8h_rolling_starts_scenario_milp_degree_hour")}
    controller = _construct("joint", resolved)
    resolved["effective"] = {
        "solver": controller.solver,
        "solver_options": jsonable(controller.solver_options),
        "horizon_hours": controller.horizon.total_seconds() / 3600,
        "control_minutes": controller.hvac_controller.model_timestep.total_seconds() / 60,
        "hvac": jsonable(_parameters(controller.hvac_controller)),
        "water_heater": jsonable(_parameters(controller.water_heater_controller)),
    }
    return resolved


def _construct(device, resolved):
    parameters = deepcopy(resolved["parameters"])
    if device == "joint":
        from argparse import Namespace
        from examples.single_home_hems.run_joint_hems import build_controller
        return build_controller(Namespace(**parameters))
    if device == "hvac":
        parameters["gain_regime"] = HVACInternalGainRegime(**parameters["gain_regime"])
    cls = HVACPredictiveController if device == "hvac" else WaterHeaterPredictiveController
    return cls(**parameters)


def build_thermal_controllers(variant, home_id, home):
    """Return fresh objects for every case; no optimizer history is shared."""
    controls = variant.thermal_controls[home_id]
    return tuple(_construct(device, controls[device])
                 for device in ("hvac", "water_heater", "joint")
                 if controls[device]["controller"] == "mpc")


def validate_thermal_controllers(variant, home_id, home):
    """Preflight supported paths and model artifacts before starting a run."""
    controllers = build_thermal_controllers(variant, home_id, home)
    joint = variant.thermal_controls[home_id]["joint"]
    if joint["controller"] == "mpc" and joint["parameters"]["peak_weight_usd_per_kw"] > 0 and home.rest_of_home.source != "dc_house_test":
        raise ValueError(f"{home_id}: joint peak objective requires DC House rest-of-home data; explicitly select peak_weight_usd_per_kw=0 for device-only optimization")
    from .controllers.joint_hems import JointHEMSPredictiveController
    settings = variant.unified.home_settings[home_id]
    window_start = pd.Timestamp(variant.unified.simulation_window().start_time)
    for controller in controllers:
        devices = (controller.water_heater_controller, controller.hvac_controller) if isinstance(controller, JointHEMSPredictiveController) else (controller,)
        for device in devices:
            if device._load_model() is None:
                raise FileNotFoundError(f"Predictive control needs a compatible trained model: {device.model_path}")
            if isinstance(device, WaterHeaterPredictiveController) and device.model_kind == "state-space":
                metadata = json.loads((Path(device.model_path) / "metadata.json").read_text(encoding="utf-8"))
                model_seconds = metadata.get("configuration", {}).get("model_timestep_seconds")
                if model_seconds != device.model_timestep.total_seconds():
                    raise ValueError("Water-heater control interval does not match its trained state-space model")
            interval = device.model_timestep
            if window_start != window_start.floor(interval):
                raise ValueError("Thermal predictive-control start must align with its control interval")
            if isinstance(device, WaterHeaterPredictiveController) and device.forecast_type == "scenario":
                if not settings.get("water_draw_file"):
                    raise ValueError("Water-heater scenario control requires water_draw_file in the household inputs")
                if not Path(settings["water_draw_file"]).is_file():
                    raise FileNotFoundError(f"Water-draw input not found: {settings['water_draw_file']}")
    return controllers


def build_rest_forecast(variant, home_id, home):
    """Use the case home's shifted source anchor on the matching warmup clock."""
    joint = variant.thermal_controls[home_id]["joint"]
    if joint["controller"] != "mpc" or home.rest_of_home.source != "dc_house_test":
        return None
    from .rest_of_home_forecast import RestOfHomeHybridForecast
    return RestOfHomeHybridForecast.from_csv(
        home.rest_of_home.data_file,
        simulation_start=variant.unified.simulation_window().start_time,
        source_start=home.rest_of_home.source_start,
        history_days=20,
        inference_gap_policy=joint["parameters"]["rest_rf_inference_gap_policy"],
    )

"""Central named experiments using the existing controllers and plant loop."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import build_ev_schedules
from .experiment_controllers import build_thermal_controllers, build_rest_forecast, jsonable
from .experiment_reporting import comparison_plot_limits, write_experiment_comparisons
from .simulation import _predictive_controllers
from .unified import run_home, baseline_economics


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, default=str) + "\n", encoding="utf-8")


def _select(experiment, variant_names=None, home_ids=None):
    available = {variant.name: variant for variant in experiment.variants}
    names = list(available) if variant_names is None else list(variant_names)
    if not names or len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError("Select existing, unique experiment variants")
    selected = []
    for name in names:
        variant = available[name]
        homes = {h.home_id: h for h in variant.unified.neighborhood.homes}
        chosen = list(homes) if home_ids is None else list(home_ids)
        if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(homes):
            raise ValueError(f"{name}: select existing, unique homes")
        config = replace(variant.unified.neighborhood, homes=tuple(homes[h] for h in chosen))
        selected.append(replace(variant, unified=replace(variant.unified, neighborhood=config)))
    return selected


def freeze_experiment_mobility(variants):
    """One mobility realization per home, covering every selected forecast horizon.

    Changing forecast method or horizon must not resample the realized Audi trip
    window. All variants share household inputs and the experiment time window.
    """
    grouped = {}
    for variant in variants:
        for home in variant.unified.neighborhood.homes:
            grouped.setdefault(home.home_id, []).append((variant, home))
    schedules = {}
    for home_id, entries in grouped.items():
        first, home = entries[0]
        config = first.unified.simulation_window()
        for variant, other in entries[1:]:
            window = variant.unified.simulation_window()
            if (window.start_time, window.duration, window.time_step) != (config.start_time, config.duration, config.time_step):
                raise ValueError("Compared variants must share simulation and reporting windows")
            if {e.vehicle_id for e in other.evs} != {e.vehicle_id for e in home.evs}:
                raise ValueError("Compared variants must share household EV equipment")
        vehicles = []
        for ev in home.evs:
            peers = [next(e for e in h.evs if e.vehicle_id == ev.vehicle_id) for _, h in entries]
            vehicles.append(replace(ev, hems=replace(ev.hems,
                history_days=max(e.hems.history_days for e in peers),
                forecast_margin_hours=max(2 * max(e.hems.horizon_hours, e.hems.forecast_margin_hours)
                                          for e in peers))))
        schedules[home_id] = build_ev_schedules(config, replace(home, evs=tuple(vehicles))) if vehicles else {}
    return schedules


def _report_settings(variant, home):
    from examples.single_home_hems.hems import HEMSController
    devices = build_thermal_controllers(variant, home.home_id, home)
    water, hvac, _ = _predictive_controllers(HEMSController(device_controllers=devices))
    selected = set()
    metadata = {"time_step_hours": variant.unified.neighborhood.time_step.total_seconds()/3600,
                "baseline_scope": "whole_home", "result_window": "excluding_warmup"}
    if hvac is not None:
        selected.add("hvac")
        metadata["hvac_comfort_band_c"] = hvac.comfort_band_c
    if water is not None:
        selected.add("water_heater")
        metadata["water_heater_minimum_temperature_c"] = water.minimum_temperature_c
    selected.update(f"EV_{key}" for key, c in variant.unified.controls[home.home_id]["evs"].items()
                    if c["controller"] == "mpc")
    return selected, metadata


def _controller_name(controller):
    if hasattr(controller, "water_heater_controller"):
        return "joint_thermal"
    name = getattr(controller, "equipment_name", None)
    return str(name).replace(" ", "_") if name else "HVAC"


def save_controller_diagnostics(devices, directory, warmup_start, evaluation_start, end):
    """Retain actual solver statuses and plans without changing fallback policies."""
    summaries = []
    for device in devices:
        name = _controller_name(device)
        folder = directory / name
        folder.mkdir(parents=True, exist_ok=True)
        history = pd.DataFrame(getattr(device, "solve_history", [])).copy()
        if "timestamp" in history:
            times = pd.to_datetime(history.timestamp)
            history["in_including_warmup_window"] = (times >= warmup_start) & (times < end)
            history["in_excluding_warmup_window"] = (times >= evaluation_start) & (times < end)
        history.to_csv(folder / "solve_history.csv", index=False)
        components = pd.DataFrame(getattr(device, "objective_component_history", []))
        if not components.empty:
            components.to_csv(folder / "objective_components.csv", index=False)
        plans = getattr(device, "optimization_plans", ())
        if plans:
            pd.concat(plans, keys=range(len(plans)), names=["solve", "row"]).to_csv(folder / "optimization_plans.csv")
        water = getattr(device, "water_heater_controller", device)
        cycle_history = getattr(water, "rolling_cycle_diagnostics_history", None)
        if isinstance(cycle_history, pd.DataFrame) and not cycle_history.empty:
            cycle_history.to_csv(folder / "water_heater_rolling_cycle_diagnostics.csv", index=False)
        for window in ("including_warmup", "excluding_warmup"):
            selected = history.loc[history[f"in_{window}_window"]] if not history.empty else history
            row = {"controller": name, "result_window": window, "attempts": len(selected)}
            row["failed_solves"] = int((~selected.successful.astype(bool)).sum()) if "successful" in selected else 0
            row["fallback_steps_or_solves"] = int(selected.fallback.fillna(False).astype(bool).sum()) if "fallback" in selected else None
            duration = next((k for k in ("solve_seconds", "duration_s") if k in selected), None)
            row["solve_seconds"] = float(selected[duration].sum()) if duration else 0.0
            summaries.append(row)
    return pd.DataFrame(summaries, columns=["controller", "result_window", "attempts", "failed_solves",
                                            "fallback_steps_or_solves", "solve_seconds"])


def _union_limits(target, source):
    for key, value in source.items():
        prior = target.get(key, value)
        target[key] = (min(prior[0], value[0]), max(prior[1], value[1]))


def run_experiment(experiment, variant_names=None, home_ids=None, output_directory=None, *, validate_only=False):
    """Execute explicit variants and fresh whole-home baselines, sequentially."""
    variants = _select(experiment, variant_names, home_ids)
    root = Path(output_directory or variants[0].unified.neighborhood.output_directory).expanduser().resolve()
    variants = [replace(v, unified=replace(v.unified, neighborhood=replace(
        v.unified.neighborhood, output_directory=root/v.name))) for v in variants]
    # Resolve all controllers before starting any variant, including Python callers.
    from .experiment_controllers import validate_thermal_controllers
    for variant in variants:
        for home in variant.unified.neighborhood.homes:
            validate_thermal_controllers(variant, home.home_id, home)
            if variant.options.get("ev_only_comparison") and not any(
                c["controller"] == "mpc" for c in variant.unified.controls[home.home_id]["evs"].values()):
                raise ValueError("ev_only_comparison requires at least one predictive EV controller")
    if validate_only:
        return {"validated": True, "variants": [v.name for v in variants], "output_directory": str(root)}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}")
    mobility = freeze_experiment_mobility(variants)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root/"requested_experiment.json", experiment.requested)
    _write_json(root/"resolved_experiment.json", {"experiment_file": str(experiment.path),
        "precedence": experiment.resolved()["precedence"],
        "variants": {v.name: v.resolved() for v in variants},
        "baseline": "Fresh whole-home native baseline for each variant; all predictive actions disabled",
        "internal_twin": "Non-V2H forecast source; distinct from optional EV-only reference"})
    for home_id, vehicles in mobility.items():
        for key, schedule in vehicles.items():
            folder = root/"inputs"/home_id
            folder.mkdir(parents=True, exist_ok=True)
            schedule.trips.to_csv(folder/f"EV_{key}_trips.csv", index=False)
            _write_json(folder/f"EV_{key}_metadata.json", schedule.metadata)
    completed, status, metrics, diagnostics, limits = {}, [], [], [], {}
    for variant in variants:
        _write_json(root/variant.name/"resolved_configuration.json", variant.resolved())
        for home in variant.unified.neighborhood.homes:
            entry = {"variant": variant.name, "home_id": home.home_id, "state": "running"}
            status.append(entry)
            _write_json(root/"run_status.json", status)
            try:
                result = run_home(variant.unified, home, mobility[home.home_id],
                    thermal_controller_factory=lambda selected, v=variant, h=home.home_id: build_thermal_controllers(v, h, selected),
                    rest_of_home_forecast_factory=lambda selected, v=variant, h=home.home_id: build_rest_forecast(v, h, selected),
                    comparison_mode="whole_home", include_ev_reference=variant.options.get("ev_only_comparison", False),
                    render_plots=False)
                selected, metadata = _report_settings(variant, home)
                _union_limits(limits, comparison_plot_limits(result.cases, home, selected, metadata))
                completed[(variant.name, home.home_id)] = (result, home, selected, metadata)
                frame = result.metrics.copy()
                frame["variant"] = variant.name
                frame["comparison_role"] = frame.case.map({"all_baseline": "whole_home_baseline", "advanced": "controlled",
                    "ev_only_reference": "optional_ev_reference", "non_v2h_twin": "internal_forecast_source"})
                metrics.append(frame)
                if "ev_only_reference" in result.cases:
                    pair = result.metrics.loc[result.metrics.case.isin(["ev_only_reference", "advanced"])].copy()
                    pair["case"] = pair.case.replace({"ev_only_reference": "all_baseline"})
                    pair = baseline_economics(pair)
                    pair["case"] = pair.case.replace({"all_baseline": "ev_only_reference"})
                    pair["reference_case"] = "ev_only_reference"
                    reference_folder = root/variant.name/home.home_id/"ev_only_comparison"
                    reference_folder.mkdir(parents=True, exist_ok=True)
                    pair.to_csv(reference_folder/"economics_both_windows.csv", index=False)
                for case, data in result.cases.items():
                    start = pd.Timestamp(variant.unified.neighborhood.start_time)
                    summary = save_controller_diagnostics(data["device_controllers"], data["output_directory"]/"diagnostics",
                        pd.Timestamp(variant.unified.simulation_window().start_time), start,
                        start+variant.unified.neighborhood.duration)
                    summary["case"], summary["variant"], summary["home_id"] = case, variant.name, home.home_id
                    diagnostics.append(summary)
                entry["state"] = "simulated"
            except Exception as error:
                entry.update(state="failed", error=f"{type(error).__name__}: {error}")
                _write_json(root/"run_status.json", status)
                raise
            _write_json(root/"run_status.json", status)
    for (name, home_id), (result, home, selected, metadata) in completed.items():
        entry = next(row for row in status if row["variant"] == name and row["home_id"] == home_id)
        try:
            write_experiment_comparisons(result.cases, home, root/name/home_id, selected,
                dpi=next(v.unified.neighborhood.plot_dpi for v in variants if v.name == name),
                metadata={**metadata, "plot_limits": limits})
            entry["state"] = "complete"
        except Exception as error:
            entry.update(state="failed", phase="reporting", error=f"{type(error).__name__}: {error}")
            _write_json(root/"run_status.json", status)
            raise
        _write_json(root/"run_status.json", status)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(root/"all_case_metrics.csv", index=False)
    all_metrics.loc[all_metrics.case.isin(["all_baseline", "advanced"])].to_csv(root/"comparison_metrics.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(root/"solver_summary.csv", index=False)
    _write_json(root/"shared_plot_limits.json", limits)
    return {key: value[0] for key, value in completed.items()}

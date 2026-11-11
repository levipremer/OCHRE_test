"""Matched household and equipment reports from already simulated trajectories.

This module does not simulate devices or inspect optimizer predictions. Applied
plant values and requested controller commands are deliberately separate outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import plt, mdates, WATER_HEATER_COMFORT_THRESHOLD_C
from .unified_plots import value_limits


CASE_LABELS = {"all_baseline": "Whole-home baseline", "advanced": "Controlled home",
               "ev_only_reference": "EV-only reference (thermal control retained)"}
_COLORS = {"all_baseline": "#666666", "advanced": "#166b9b", "ev_only_reference": "#a05a13"}


def _series(data, table, column, index):
    frame = data.get(table)
    if frame is None or column not in frame:
        return None
    result = frame[column].reindex(index)
    if result.isna().any():
        raise ValueError(f"Comparison column {table}.{column} has missing or misaligned values")
    return result


def _numeric(data, table, column, index):
    result = _series(data, table, column, index)
    if result is not None:
        result = pd.to_numeric(result, errors="raise")
        if not np.isfinite(result).all():
            raise ValueError(f"Comparison column {table}.{column} must be finite")
    return result


def _put(frame, name, value):
    if value is not None:
        frame[name] = value


def _validate_cases(cases, metadata):
    for name in ("all_baseline", "advanced"):
        if name not in cases:
            raise ValueError(f"Whole-home comparison requires {name}")
    index = cases["all_baseline"]["mapped"].index
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Comparison needs a nonempty, increasing, unique DatetimeIndex")
    step = float(metadata.get("time_step_hours", (index[1] - index[0]).total_seconds()/3600 if len(index) > 1 else 1/60))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("time_step_hours must be finite and positive")
    if len(index) > 1 and not np.allclose(np.diff(index.asi8)/3.6e12, step):
        raise ValueError("Comparison timestamps must use the configured regular time step")
    price = _numeric(cases["all_baseline"], "commands", "electricity_price_usd_per_kwh", index)
    if price is None:
        raise ValueError("Comparison requires the realized electricity tariff")
    for name, data in cases.items():
        if name not in CASE_LABELS:
            continue
        if not data["mapped"].index.equals(index):
            raise ValueError(f"Comparison time windows do not align for {name}")
        other = _numeric(data, "commands", "electricity_price_usd_per_kwh", index)
        if other is None or not np.allclose(other, price, rtol=0, atol=1e-12):
            raise ValueError(f"Matched cases require the same electricity tariff: {name}")
    return index, step, price


def _device_frames(cases, home, selected_devices, metadata):
    index, step, price = _validate_cases(cases, metadata)
    baseline = cases["all_baseline"]
    scheduled = _numeric(baseline, "schedule", "HVAC Heating Setpoint (C)", index)
    if scheduled is None:
        scheduled = _numeric(baseline, "mapped", "heating_setpoint_c", index)
    baseline_wh = _numeric(baseline, "water_training", "water_heater_setpoint_temperature_c", index)
    devices = {"whole_home": {}}
    selected = set(selected_devices)
    if "hvac" in selected:
        devices["hvac"] = {}
    if selected.intersection({"water_heater", "hpwh"}):
        devices["water_heater"] = {}
    evs = [ev for ev in home.evs if "ev" in selected or f"EV_{ev.vehicle_id}" in selected or f"ev_{ev.vehicle_id}" in selected]
    for ev in evs:
        devices[f"ev_{ev.vehicle_id}"] = {}
    band = metadata.get("hvac_comfort_band_c", metadata.get("comfort_band_c"))
    if band is not None and (not np.isfinite(float(band)) or float(band) < 0):
        raise ValueError("HVAC comfort band must be finite and nonnegative")
    for case, data in cases.items():
        if case not in CASE_LABELS:
            continue
        whole = pd.DataFrame(index=index)
        power = _numeric(data, "mapped", "total_electric_power_kw", index)
        if power is None:
            raise ValueError("Whole-home comparison requires total_electric_power_kw")
        whole["net_electric_power_kw"] = power
        whole["grid_import_power_kw"] = power.clip(lower=0)
        whole["grid_export_power_kw"] = (-power).clip(lower=0)
        whole["cumulative_net_energy_kwh"] = (power*step).cumsum()
        whole["cumulative_electricity_cost_usd"] = (power*price*step).cumsum()
        devices["whole_home"][case] = whole
        if "hvac" in devices:
            hvac = pd.DataFrame(index=index)
            for col in ("indoor_temperature_c", "heating_setpoint_c", "cooling_setpoint_c"):
                _put(hvac, col, _numeric(data, "mapped", col, index))
            _put(hvac, "electric_power_kw", _numeric(data, "mapped", "hvac_power_kw", index))
            for col in ("heating_mode", "cooling_mode"):
                _put(hvac, col, _series(data, "hvac_training", col, index))
            if scheduled is not None:
                hvac["scheduled_heating_setpoint_c"] = scheduled
                if "heating_setpoint_c" in hvac:
                    hvac["applied_heating_setpoint_shift_c"] = hvac.heating_setpoint_c-scheduled
                if band is not None:
                    hvac["comfort_lower_c"], hvac["comfort_upper_c"] = scheduled-float(band), scheduled+float(band)
            devices["hvac"][case] = hvac
        if "water_heater" in devices:
            water = pd.DataFrame(index=index)
            sensors = data.get("sensors", pd.DataFrame())
            for col in sensors:
                _put(water, col, _numeric(data, "sensors", col, index))
            _put(water, "applied_setpoint_c", _numeric(data, "water_training", "water_heater_setpoint_temperature_c", index))
            _put(water, "electric_power_kw", _numeric(data, "mapped", "water_heating_power_kw", index))
            _put(water, "operating_mode", _series(data, "water_training", "water_heater_mode", index))
            _put(water, "water_draw_l_per_min", _numeric(data, "water_training", "water_heater_draw_l_per_min", index))
            if baseline_wh is not None and "applied_setpoint_c" in water:
                water["applied_setpoint_shift_c"] = water.applied_setpoint_c-baseline_wh
            water["reporting_temperature_threshold_c"] = float(metadata.get("water_heater_minimum_temperature_c", WATER_HEATER_COMFORT_THRESHOLD_C))
            devices["water_heater"][case] = water
        for ev in evs:
            name = f"EV_{ev.vehicle_id}"
            frame = pd.DataFrame(index=index)
            for output, choices in {
                "soc_fraction": ("End SOC (-)", "SOC (-)"),
                "soh_fraction": ("End SOH (-)", "SOH (-)"),
                "capacity_kwh": ("End Capacity (kWh)", "Capacity (kWh)"),
                "electric_power_kw": ("Electric Power (kW)",),
                "reference_degradation_cost_usd": ("Reference Degradation Cost ($)",),
            }.items():
                for suffix in choices:
                    value = _numeric(data, "raw", f"{name} {suffix}", index)
                    if value is not None:
                        frame[output] = value
                        break
            _put(frame, "operating_mode", _series(data, "raw", f"{name} Mode", index))
            if "reference_degradation_cost_usd" in frame:
                frame["cumulative_reference_degradation_cost_usd"] = frame.reference_degradation_cost_usd.cumsum()
            devices[f"ev_{ev.vehicle_id}"][case] = frame
    return devices, step, price


def _groups(device, frame):
    columns = frame.columns
    if device == "whole_home":
        candidates = [("power", "Electrical power (kW)", ["net_electric_power_kw"]),
                      ("energy", "Net energy (kWh)", ["cumulative_net_energy_kwh"]),
                      ("cost", "Electricity cost (USD)", ["cumulative_electricity_cost_usd"])]
    else:
        temperatures = [c for c in columns if c.endswith("_temperature_c") or c in {"indoor_temperature_c", "heating_setpoint_c", "cooling_setpoint_c", "applied_setpoint_c", "comfort_lower_c", "comfort_upper_c", "reporting_temperature_threshold_c"}]
        candidates = [("temperatures", "Temperature (°C)", temperatures),
                      ("setpoint_shifts", "Setpoint shift (°C)", [c for c in columns if "shift_c" in c]),
                      ("power", "Electrical power (kW)", ["electric_power_kw"]),
                      ("soc", "SOC (fraction)", ["soc_fraction"]),
                      ("soh", "SOH (fraction)", ["soh_fraction"]),
                      ("capacity", "Usable capacity (kWh)", ["capacity_kwh"]),
                      ("aging", "Reference aging cost (USD)", ["cumulative_reference_degradation_cost_usd"])]
    return [(key, ylabel, [c for c in cols if c in columns]) for key, ylabel, cols in candidates if any(c in columns for c in cols)]


def comparison_plot_limits(cases, home, selected_devices, metadata=None):
    """Return mergeable limits; callers can union these across named variants."""
    devices, _, _ = _device_frames(cases, home, selected_devices, metadata or {})
    result = {}
    for device, frames in devices.items():
        union = pd.concat(list(frames.values()), axis=0)
        for key, _, columns in _groups(device, union):
            result[f"{device}/{key}"] = value_limits([frame.reindex(columns=columns) for frame in frames.values()])
    return result


def _requested_commands(data, device, index):
    source = data.get("commands", pd.DataFrame(index=index))
    records = []
    for value in source.get("control_signal_json", pd.Series("{}", index=index)).reindex(index):
        command = json.loads(value) if isinstance(value, str) else value
        if not isinstance(command, dict):
            raise ValueError("Control signal must be a JSON object")
        selected = {}
        selected_signal = {}
        for equipment, values in command.items():
            matches = (device == "hvac" and equipment.startswith("HVAC")) or (device == "water_heater" and equipment.startswith("Water")) or (device.startswith("ev_") and equipment == "EV_"+device[3:])
            if matches and isinstance(values, dict):
                selected_signal[equipment] = values
                selected.update({f"requested_{equipment}_{key}": val for key, val in values.items()})
        selected["requested_signal_json"] = json.dumps(selected_signal, sort_keys=True)
        records.append(selected)
    return pd.DataFrame(records, index=index)


def _metrics(device, case, frame, step, price):
    result = {"device": device, "case": case, "duration_hours": len(frame)*step}
    power_col = "net_electric_power_kw" if device == "whole_home" else "electric_power_kw"
    if power_col in frame:
        power = frame[power_col]
        result.update(net_energy_kwh=float(power.sum()*step), peak_electric_power_kw=float(power.max()))
        if device == "whole_home":
            result["electricity_cost_usd"] = float((power*price).sum()*step)
            result["import_energy_kwh"] = float(power.clip(lower=0).sum()*step)
            result["export_energy_kwh"] = float((-power).clip(lower=0).sum()*step)
        else:
            result["attributed_electricity_cost_usd"] = float((power*price).sum()*step)
            on = power.abs().gt(1e-6)
            result["active_hours"] = float(on.sum()*step)
            result["observed_off_to_on_transitions"] = int((on & ~on.shift(1, fill_value=bool(on.iloc[0]))).sum())
    if {"indoor_temperature_c", "comfort_lower_c", "comfort_upper_c"}.issubset(frame):
        low = (frame.comfort_lower_c-frame.indoor_temperature_c).clip(lower=0)
        high = (frame.indoor_temperature_c-frame.comfort_upper_c).clip(lower=0)
        result["comfort_deviation_degree_hours"] = float((low+high).sum()*step)
        result["outside_comfort_hours"] = float((low+high).gt(0).sum()*step)
    for col in frame:
        if "shift_c" in col:
            result[f"mean_absolute_{col}"] = float(frame[col].abs().mean())
            result[f"max_absolute_{col}"] = float(frame[col].abs().max())
        if col.startswith("water_heater_sensor_node_"):
            result[f"{col}_minimum"] = float(frame[col].min())
            result[f"{col}_below_reporting_threshold_hours"] = float(frame[col].lt(frame.reporting_temperature_threshold_c).sum()*step)
    for col in ("soc_fraction", "soh_fraction", "capacity_kwh"):
        if col in frame:
            result[f"minimum_{col}"] = float(frame[col].min())
            result[f"final_{col}"] = float(frame[col].iloc[-1])
    if "reference_degradation_cost_usd" in frame:
        result["reference_degradation_cost_usd"] = float(frame.reference_degradation_cost_usd.sum())
    return result


def _plot(device, frames, folder, title, dpi, limits):
    union = pd.concat(list(frames.values()), axis=0)
    groups = _groups(device, union)
    if groups:
        fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3*len(groups)), sharex=True, squeeze=False)
        for axis, (key, ylabel, columns) in zip(axes.flat, groups):
            for case, frame in frames.items():
                for n, column in enumerate(columns):
                    if column not in frame:
                        continue
                    label = f"{CASE_LABELS[case]}: {column.replace('_', ' ')}" if len(columns) > 1 else CASE_LABELS[case]
                    axis.plot(frame.index, frame[column], label=label, color=_COLORS[case], linestyle=("-", "--", ":", "-.")[n % 4], alpha=.85)
            axis.set_ylabel(ylabel)
            axis.set_ylim(limits[f"{device}/{key}"])
            axis.grid(alpha=.2)
            axis.legend(fontsize="x-small", ncol=2)
        axes[0, 0].set_title(title)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1, 0].xaxis.set_major_locator(locator)
        axes[-1, 0].xaxis.set_major_formatter(formatter)
        fig.tight_layout()
        fig.savefig(folder/"trajectory_comparison.png", dpi=dpi)
        plt.close(fig)
    modes = sorted({column for frame in frames.values() for column in frame if "mode" in column})
    if modes:
        fig, axes = plt.subplots(len(modes), 1, figsize=(12, 2.7*len(modes)), sharex=True, squeeze=False)
        for axis, column in zip(axes.flat, modes):
            labels = sorted({str(value) for frame in frames.values() if column in frame for value in frame[column].unique()})
            mapping = {label: n for n, label in enumerate(labels)}
            for case, frame in frames.items():
                if column in frame:
                    axis.step(frame.index, frame[column].map(mapping), where="post", label=CASE_LABELS[case], color=_COLORS[case])
            axis.set_yticks(range(len(labels)), labels)
            axis.set_ylabel(column.replace("_", " "))
            axis.legend(fontsize="small")
            axis.grid(alpha=.2)
        axes[0, 0].set_title(title+": observed modes")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder/"operating_state_comparison.png", dpi=dpi)
        plt.close(fig)


def write_experiment_comparisons(cases: dict, home, output: Path, selected_devices: set[str], dpi=120, metadata=None):
    """Save primary reports and a separately labeled optional EV-only comparison.

    Cases are evaluation slices of the same household simulations. The internal
    non-V2H forecast twin is never presented as the whole-home baseline.
    """
    metadata = metadata or {}
    output = Path(output)
    devices, step, price = _device_frames(cases, home, selected_devices, metadata)
    limits = comparison_plot_limits(cases, home, selected_devices, metadata)
    limits.update(metadata.get("plot_limits", {}))
    rows = []
    for device, all_frames in devices.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", device):
            raise ValueError(f"Unsafe device folder name: {device}")
        groups = [(output/device, {key: all_frames[key] for key in ("all_baseline", "advanced")})]
        if "ev_only_reference" in all_frames and (device == "whole_home" or device.startswith("ev_")):
            groups.append((output/"ev_only_comparison"/device, {key: all_frames[key] for key in ("ev_only_reference", "advanced")}))
        for folder, frames in groups:
            folder.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, axis=1).to_csv(folder/"trajectory_comparison.csv", index_label="timestamp")
            local_rows = [_metrics(device, case, frame, step, price) for case, frame in frames.items()]
            for row in local_rows:
                for field, savings in (("net_energy_kwh", "energy_savings_kwh"),
                                       ("electricity_cost_usd", "electricity_cost_savings_usd"),
                                       ("attributed_electricity_cost_usd", "attributed_electricity_cost_savings_usd"),
                                       ("peak_electric_power_kw", "peak_reduction_kw")):
                    if field in row:
                        row[savings] = local_rows[0][field]-row[field]
            pd.DataFrame(local_rows).to_csv(folder/"comparison_metrics.csv", index=False)
            if folder == output/device:
                rows.extend(local_rows)
            if device != "whole_home":
                pd.concat({case: _requested_commands(cases[case], device, frame.index) for case, frame in frames.items()}, axis=1).to_csv(folder/"requested_commands.csv", index_label="timestamp")
            _plot(device, frames, folder, f"{home.home_id}: {device.replace('_', ' ')}", dpi, limits)
    summary = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/"comparison_metrics.csv", index=False)
    (output/"report_settings.json").write_text(json.dumps({
        "primary_cases": ["all_baseline", "advanced"], "case_labels": CASE_LABELS,
        "internal_forecast_twin_excluded": True, "time_step_hours": step,
        "selected_devices": sorted(selected_devices), "plot_limits": limits,
        "whole_home_cost": "Signed total electric power times the same tariff used by existing run economics. Import/export energy is also reported separately.",
        "device_cost": "Signed device power times tariff; attribution, not a separate household bill.",
        "commands": "requested_commands.csv contains commands issued to OCHRE. trajectory_comparison.csv contains observed plant values. Optimizer predictions are separate diagnostics.",
        "comfort": "HVAC scheduled heating setpoint plus/minus configured comfort band; water threshold is a reporting threshold, not a new control constraint.",
    }, indent=2)+"\n", encoding="utf-8")
    return summary

"""Fit the paper-style two-node RC water-heater model from observable signals."""

from __future__ import annotations

import json
from dataclasses import replace
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - slower pure-Python fallback
    def njit(function):
        return function

from ..water_heater_rc import (
    WATER_HEAT_CAPACITY_KWH_PER_L_K,
    WaterHeaterRCModel,
    rc_transition,
    reconstruct_heat_input,
)
from .model_diagnostics import COLORS, _plot_three_split_predictions, plt
from .water_heater import WaterHeaterSensorLayout
from .water_heater_model import WaterHeaterSplitWindows, load_water_heater_training_data


REQUIRED_COLUMNS = (
    "water_heater_sensor_node_03_temperature_c",
    "water_heater_sensor_node_10_temperature_c",
    "next_water_heater_sensor_node_03_temperature_c",
    "next_water_heater_sensor_node_10_temperature_c",
    "water_heater_setpoint_temperature_c",
    "water_heater_draw_l_per_min",
    "water_heater_mains_temperature_c",
    "water_heater_tank_ambient_temperature_c",
    "water_heater_electric_power_kw",
    "water_heater_mode",
    "water_heater_tank_volume_l",
)


@njit
def _rollout(
    upper, lower, setpoint, draw, mains, ambient,
    previous_on, transitions, integrals, z, heat_fraction, cop,
    capacity, ua_kw, on_power, off_power, deadband, reset_steps,
):
    """Open-loop prediction with measured state/mode resets every horizon."""
    count = len(upper)
    predicted_upper = np.empty(count)
    predicted_lower = np.empty(count)
    predicted_power = np.empty(count)
    upper_state = upper[0]
    lower_state = lower[0]
    on = previous_on[0]
    for index in range(count):
        if index % reset_steps == 0:
            upper_state = upper[index]
            lower_state = lower[index]
            on = previous_on[index]
        control_temp = 0.75 * upper_state + 0.25 * lower_state
        if control_temp < setpoint[index] - deadband:
            on = True
        elif control_temp >= setpoint[index] or upper_state >= setpoint[index] + 1:
            on = False
        power = on_power if on else off_power
        heat_kw = cop * (on_power - off_power) if on else 0.0
        flow_cp = max(draw[index], 0.0) * 60 * WATER_HEAT_CAPACITY_KWH_PER_L_K
        g1 = (ua_kw * ambient[index] + heat_fraction * heat_kw / z) / capacity
        g2 = (
            ua_kw * ambient[index]
            + flow_cp * mains[index] / (1 - z)
            + (1 - heat_fraction) * heat_kw / (1 - z)
        ) / capacity
        new_upper = (
            transitions[index, 0, 0] * upper_state
            + transitions[index, 0, 1] * lower_state
            + integrals[index, 0, 0] * g1
            + integrals[index, 0, 1] * g2
        )
        new_lower = (
            transitions[index, 1, 0] * upper_state
            + transitions[index, 1, 1] * lower_state
            + integrals[index, 1, 0] * g1
            + integrals[index, 1, 1] * g2
        )
        predicted_upper[index] = new_upper
        predicted_lower[index] = new_lower
        predicted_power[index] = power
        upper_state = new_upper
        lower_state = new_lower
    return predicted_upper, predicted_lower, predicted_power


def _split(data: pd.DataFrame, windows: WaterHeaterSplitWindows) -> dict[str, pd.DataFrame]:
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"RC training data is missing columns: {sorted(missing)}")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("RC training timestamps must be unique and increasing")
    numeric = [column for column in REQUIRED_COLUMNS if column != "water_heater_mode"]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("RC training data contains nonfinite values")
    data = data.copy()
    data["_previous_hp_on"] = data["water_heater_mode"].shift().eq("Heat Pump On")
    splits = {}
    for name in ("training", "validation", "test"):
        window = getattr(windows, name)
        part = data.loc[(data.index >= window.start) & (data.index < window.end)].copy()
        if part.empty:
            raise ValueError(f"No RC samples in {name} interval")
        if len(part) < 2:
            raise ValueError(f"RC {name} needs at least two samples")
        if not part.index.to_series().diff().iloc[1:].eq(part.index[1] - part.index[0]).all():
            raise ValueError(f"RC {name} timestamps are not uniformly spaced")
        modes = set(part["water_heater_mode"])
        if not modes <= {"Off", "Heat Pump On"}:
            raise ValueError(f"RC {name} requires HP-only data; found modes {sorted(modes)}")
        splits[name] = part
    if not {"Off", "Heat Pump On"} <= set(splits["training"]["water_heater_mode"]):
        raise ValueError("RC training needs both off and heat-pump-on intervals")
    return splits


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    names = REQUIRED_COLUMNS[:9]
    values = tuple(frame[name].to_numpy(dtype=np.float64) for name in names)
    on = frame["_previous_hp_on"].to_numpy(dtype=np.bool_)
    return (*values, on)


def _transition_arrays(
    frame: pd.DataFrame, model: WaterHeaterRCModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse each exact ZOH transition for repeated observed draw rates."""
    draw = frame[REQUIRED_COLUMNS[5]].to_numpy(dtype=float)
    unique_draws, inverse = np.unique(draw, return_inverse=True)
    timestep_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    transitions = np.empty((len(unique_draws), 2, 2))
    integrals = np.empty_like(transitions)
    for index, rate in enumerate(unique_draws):
        transitions[index], integrals[index] = rc_transition(
            model, float(rate), timestep_hours
        )
    return transitions[inverse], integrals[inverse]


def _predict(
    frame: pd.DataFrame, model: WaterHeaterRCModel, reset_steps: int,
    transition_arrays: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    upper, lower, _, _, setpoint, draw, mains, ambient, _, on = _arrays(frame)
    if transition_arrays is None:
        transition_arrays = _transition_arrays(frame, model)
    return np.vstack(_rollout(
        upper, lower, setpoint, draw, mains, ambient,
        on, *transition_arrays, model.upper_fraction,
        model.upper_heat_fraction, model.cop,
        model.heat_capacity_kwh_per_k, model.ua_w_per_k / 1000,
        model.on_power_kw, model.off_power_kw, model.deadband_c, reset_steps,
    )).T


def _errors(frame: pd.DataFrame, prediction: np.ndarray, scales: dict[str, float]) -> dict[str, float]:
    actual = frame[list(REQUIRED_COLUMNS[2:4]) + [REQUIRED_COLUMNS[8]]].to_numpy(dtype=float)
    error = prediction - actual
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    tank = (0.75 * mae[0] + 0.25 * mae[1]) / scales["temperature_c"]
    power = mae[2] / scales["power_kw"]
    return {
        "score": float((tank + power) / 2),
        "tank_error": float(tank),
        "power_error": float(power),
        "upper_mae_c": float(mae[0]),
        "lower_mae_c": float(mae[1]),
        "power_mae_kw": float(mae[2]),
        "upper_rmse_c": float(rmse[0]),
        "lower_rmse_c": float(rmse[1]),
        "power_rmse_kw": float(rmse[2]),
    }


def _geometry(arguments_path: Path, training: pd.DataFrame) -> tuple[float, float, float, float, float]:
    with arguments_path.open(encoding="utf-8") as stream:
        equipment = json.load(stream)["Equipment"]["Water Heating"]
    if equipment.get("hp_only_mode") is not True:
        raise ValueError(
            "RC training requires an OCHRE run created with --water-heater-hp-only"
        )
    volume = float(equipment["Tank Volume (L)"])
    height = float(equipment["Tank Height (m)"])
    if "UA (W/K)" in equipment:
        ua = float(equipment["UA (W/K)"])
    else:
        radius = sqrt((volume / 1000 / height) / np.pi)
        area = 2 * volume / 1000 / height + 2 * np.pi * radius * height
        ua = float(equipment["Heat Transfer Coefficient (W/m^2/K)"]) * area
    if not np.allclose(training["water_heater_tank_volume_l"], volume):
        raise ValueError("OCHRE equipment volume disagrees with the training CSV")
    deadband = float(equipment.get("Deadband Temperature (C)", 8.17))
    delivery = float(equipment.get("Mixed Delivery Temperature (C)", (105 - 32) * 5 / 9))
    return volume, height, ua, deadband, delivery


def _pareto(sweep: pd.DataFrame) -> pd.DataFrame:
    ordered = sweep.sort_values(["validation_tank_error", "validation_power_error"])
    best_power = float("inf")
    rows = []
    for row in ordered.itertuples():
        if row.validation_power_error < best_power:
            best_power = row.validation_power_error
            rows.append(row.Index)
    return sweep.loc[rows].sort_values("validation_tank_error")


def _diagnostics(
    splits: dict[str, pd.DataFrame], model: WaterHeaterRCModel,
    sweep: pd.DataFrame | None, selected_index: int | None, output: Path, reset_steps: int,
    *, predictor=None, label: str = "Two-node RC", heat_reconstructor=None,
    split_labels: dict[str, str] | None = None,
) -> list[str]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    predictor = predictor or _predict
    heat_reconstructor = heat_reconstructor or reconstruct_heat_input
    for split, frame in splits.items():
        if predictor is _predict:
            transitions = _transition_arrays(frame, model)
            rolling = predictor(frame, model, reset_steps, transitions)
            one_step = predictor(frame, model, 1, transitions)
        else:
            rolling = predictor(frame, model, reset_steps)
            one_step = predictor(frame, model, 1)
        for index, timestamp in enumerate(frame.index):
            records.append({
                "timestamp": timestamp, "split": split,
                "measured_upper_c": frame.iloc[index][REQUIRED_COLUMNS[2]],
                "measured_lower_c": frame.iloc[index][REQUIRED_COLUMNS[3]],
                "measured_power_kw": frame.iloc[index][REQUIRED_COLUMNS[8]],
                "rolling_upper_c": rolling[index, 0],
                "rolling_lower_c": rolling[index, 1],
                "rolling_power_kw": rolling[index, 2],
                "one_step_upper_c": one_step[index, 0],
                "one_step_lower_c": one_step[index, 1],
                "one_step_power_kw": one_step[index, 2],
                "horizon_step": index % reset_steps + 1,
            })
    predictions = pd.DataFrame(records)
    predictions.to_csv(figures / "predictions.csv", index=False)
    for kind in ("upper_c", "lower_c", "power_kw"):
        predictions[f"rolling_{kind}_error"] = predictions[f"rolling_{kind}"] - predictions[f"measured_{kind}"]
        predictions[f"one_step_{kind}_error"] = predictions[f"one_step_{kind}"] - predictions[f"measured_{kind}"]
    residual_columns = ["timestamp", "split", "horizon_step"] + [name for name in predictions if name.endswith("_error")]
    predictions[residual_columns].to_csv(figures / "residuals.csv", index=False)

    paths = []
    labels = (("upper_c", "Upper temperature (°C)"), ("lower_c", "Lower temperature (°C)"), ("power_kw", "Power (kW)"))
    bundles = {}
    heat_input = {}
    for split, frame in splits.items():
        part = predictions.loc[predictions.split.eq(split)]
        bundles[split] = {
            "index": frame.index,
            "measured": part[["measured_upper_c", "measured_lower_c"]].to_numpy(),
            "one_step": part[["one_step_upper_c", "one_step_lower_c"]].to_numpy(),
            "rolling": part[["rolling_upper_c", "rolling_lower_c"]].to_numpy(),
        }
        heat_input[split] = heat_reconstructor(frame, model)
    pd.concat(heat_input, names=["split", "timestamp"]).to_csv(figures / "heat_reconstruction.csv")
    _plot_three_split_predictions(
        bundles, tuple(REQUIRED_COLUMNS[:2]), label,
        figures / "three_split_predictions.png", heat_input=heat_input,
        split_labels=split_labels,
    )
    paths.append("three_split_predictions.png")

    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    for row, (kind, label) in enumerate(labels):
        error = predictions[f"rolling_{kind}_error"]
        axes[row, 0].plot(predictions.timestamp, error, lw=.7, color=COLORS["residual"])
        axes[row, 0].axhline(0, color="black", lw=.7)
        axes[row, 0].set_ylabel(label + " error")
        axes[row, 1].hist(error, bins=50, color=COLORS["residual"], alpha=.75)
        axes[row, 1].set_xlabel(label + " error")
    figure.suptitle(f"{label}: 12-hour rolling prediction residuals")
    figure.tight_layout()
    figure.savefig(figures / "residual_diagnostics.png", dpi=150)
    plt.close(figure)
    paths.append("residual_diagnostics.png")

    horizon = predictions.groupby(["split", "horizon_step"], sort=True)
    horizon_rows = []
    for (split, step), part in horizon:
        horizon_rows.append({"split": split, "horizon_step": step, **{
            f"{kind}_rmse": float(np.sqrt(np.square(part[f"rolling_{kind}_error"]).mean()))
            for kind, _ in labels
        }})
    horizon_frame = pd.DataFrame(horizon_rows)
    horizon_frame.to_csv(figures / "rmse_by_horizon.csv", index=False)
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, (kind, label) in zip(axes, labels):
        for split, part in horizon_frame.groupby("split"):
            axis.plot(part.horizon_step, part[f"{kind}_rmse"], label=split)
        axis.set_ylabel(label + " RMSE")
        axis.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Steps since measured-state reset")
    figure.suptitle(f"{label}: prediction error by horizon")
    figure.tight_layout()
    figure.savefig(figures / "rmse_by_horizon.png", dpi=150)
    plt.close(figure)
    paths.append("rmse_by_horizon.png")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (kind, label) in zip(axes, labels):
        for split, part in predictions.groupby("split"):
            axis.scatter(part[f"measured_{kind}"], part[f"rolling_{kind}"], s=3, alpha=.35, label=split)
        low = min(predictions[f"measured_{kind}"].min(), predictions[f"rolling_{kind}"].min())
        high = max(predictions[f"measured_{kind}"].max(), predictions[f"rolling_{kind}"].max())
        axis.plot([low, high], [low, high], color="black", lw=.8)
        axis.set(xlabel="Measured " + label, ylabel="Predicted " + label)
    axes[0].legend()
    figure.suptitle(f"{label}: measured versus predicted")
    figure.tight_layout()
    figure.savefig(figures / "measured_vs_predicted.png", dpi=150)
    plt.close(figure)
    paths.append("measured_vs_predicted.png")

    if sweep is None:
        return paths
    front = _pareto(sweep)
    front.to_csv(figures / "pareto_front.csv", index=False)
    selected = sweep.loc[selected_index]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(sweep.validation_tank_error, sweep.validation_power_error, s=12, alpha=.3, color="#69859b", label="Grid candidates")
    axis.plot(front.validation_tank_error, front.validation_power_error, color="#235f70", lw=1.5, label="Pareto front")
    axis.scatter([selected.validation_tank_error], [selected.validation_power_error], marker="*", s=280, color="#cf3d24", edgecolor="black", label="Selected")
    axis.annotate(
        f"Selected: hₛ={selected.stratification_thickness_m:.3f} m, z={selected.upper_fraction:.2f}, λ={selected.upper_heat_fraction:.2f}, COP={selected.cop:.2f}" if "stratification_thickness_m" in sweep else f"Selected: z={selected.upper_fraction:.1f}, λ={selected.upper_heat_fraction:.1f}, COP={selected.cop:.1f}",
        (selected.validation_tank_error, selected.validation_power_error),
        xytext=(12, 16), textcoords="offset points", fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#cf3d24"},
    )
    metric = "RMSE" if "stratification_thickness_m" in sweep else "MAE"
    axis.set(xlabel=f"Validation normalized tank {metric}", ylabel=f"Validation normalized power {metric}", title="RC parameter sweep: temperature versus power")
    if metric == "MAE":
        axis.text(.02, .02, "Selected from fits within 5% of best training score", transform=axis.transAxes, fontsize=9, va="bottom")
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(figures / "parameter_pareto.png", dpi=150)
    plt.close(figure)
    paths.append("parameter_pareto.png")
    return paths


def train_water_heater_rc(
    training_data_path: Path,
    arguments_path: Path,
    windows: WaterHeaterSplitWindows,
    output: Path,
    *,
    horizon_hours: float = 12,
) -> dict:
    """Old method: thermostat-driven grid search over z, heat split, and COP."""
    data = load_water_heater_training_data(training_data_path)
    splits = _split(data, windows)
    training = splits["training"]
    volume, height, ua, deadband, delivery = _geometry(arguments_path, training)
    step = training.index[1] - training.index[0]
    if any((part.index[1] - part.index[0]) != step for part in splits.values()):
        raise ValueError("RC splits must have the same sampling period")
    reset_steps = int(pd.Timedelta(hours=horizon_hours) / step)
    if reset_steps < 1:
        raise ValueError("RC prediction horizon must be at least one timestep")
    upper_range = float(training[REQUIRED_COLUMNS[0]].max() - training[REQUIRED_COLUMNS[0]].min())
    lower_range = float(training[REQUIRED_COLUMNS[1]].max() - training[REQUIRED_COLUMNS[1]].min())
    power_range = float(training[REQUIRED_COLUMNS[8]].max() - training[REQUIRED_COLUMNS[8]].min())
    scales = {"temperature_c": 0.75 * upper_range + 0.25 * lower_range, "power_kw": power_range}
    if min(scales.values()) <= 0:
        raise ValueError("RC training requires nonzero temperature and power ranges")
    on_power = float(training.loc[training.water_heater_mode.eq("Heat Pump On"), REQUIRED_COLUMNS[8]].mean())
    off_power = float(training.loc[training.water_heater_mode.eq("Off"), REQUIRED_COLUMNS[8]].mean())
    base = WaterHeaterRCModel(volume, height, ua, .5, .5, 3, on_power, off_power, deadband)

    rows = []
    transitions_by_z = {}
    for z in np.round(np.arange(.1, 1, .1), 2):
        z_model = replace(base, upper_fraction=float(z))
        transitions_by_z[float(z)] = {
            name: _transition_arrays(part, z_model)
            for name, part in splits.items()
        }
        for heat_fraction in np.round(np.arange(0, 1.01, .1), 2):
            for cop in np.round(np.arange(1, 5.01, .1), 2):
                candidate = replace(base, upper_fraction=float(z), upper_heat_fraction=float(heat_fraction), cop=float(cop))
                predicted = _predict(
                    training, candidate, reset_steps,
                    transitions_by_z[float(z)]["training"],
                )
                metrics = _errors(training, predicted, scales)
                rows.append({"upper_fraction": float(z), "upper_heat_fraction": float(heat_fraction), "cop": float(cop), **{f"training_{key}": value for key, value in metrics.items()}})
    sweep = pd.DataFrame(rows)
    # Validation selects only among fits close to the minimum training loss.
    cutoff = float(sweep.training_score.min() * 1.05)
    candidates = sweep.index[sweep.training_score <= cutoff]
    sweep["validation_score"] = np.nan
    sweep["validation_tank_error"] = np.nan
    sweep["validation_power_error"] = np.nan
    for index in sweep.index:
        row = sweep.loc[index]
        candidate = replace(base, upper_fraction=float(row.upper_fraction), upper_heat_fraction=float(row.upper_heat_fraction), cop=float(row.cop))
        metrics = _errors(
            splits["validation"],
            _predict(
                splits["validation"], candidate, reset_steps,
                transitions_by_z[float(row.upper_fraction)]["validation"],
            ),
            scales,
        )
        for key, value in metrics.items():
            sweep.loc[index, f"validation_{key}"] = value
    selected_index = int(sweep.loc[candidates, "validation_score"].idxmin())
    selected_row = sweep.loc[selected_index]
    selected = replace(base, upper_fraction=float(selected_row.upper_fraction), upper_heat_fraction=float(selected_row.upper_heat_fraction), cop=float(selected_row.cop))
    test_metrics = _errors(
        splits["test"],
        _predict(
            splits["test"], selected, reset_steps,
            transitions_by_z[selected.upper_fraction]["test"],
        ),
        scales,
    )
    output.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output / "parameter_sweep.csv", index=False)
    figure_files = _diagnostics(splits, selected, sweep, selected_index, output, reset_steps)
    metadata = {
        "schema_version": 1,
        "model_type": "two_node_water_heater_rc",
        "fitting_method": "old_thermostat_grid",
        "parameters": selected.metadata(),
        "derived": {
            "heat_capacity_kwh_per_k": selected.heat_capacity_kwh_per_k,
            "ambient_resistance_k_per_kw": selected.ambient_resistance_k_per_kw,
            "inter_node_resistance_k_per_kw": selected.inter_node_resistance_k_per_kw,
            "maximum_heat_kw": selected.maximum_heat_kw,
        },
        "thermostat": "OCHRE HP-only hysteresis; no minimum on/off times; fixed mean on/off power",
        "tempered_draw_temperature_c": delivery,
        "inter_node_resistance_method": "fixed_half_tank_height",
        "normalization": scales,
        "objective": "0.5*(0.75*upper_MAE+0.25*lower_MAE)/temperature_scale + 0.5*power_MAE/power_scale",
        "selection": {"training_tolerance_fraction": .05, "eligible_candidates": len(candidates), "selected_grid_index": selected_index, "training_score": float(selected_row.training_score), "validation_score": float(selected_row.validation_score)},
        "test_metrics": test_metrics,
        "sample_period_seconds": step.total_seconds(),
        "prediction_horizon_seconds": pd.Timedelta(hours=horizon_hours).total_seconds(),
        "windows": {name: {"start": getattr(windows, name).start.isoformat(), "end": getattr(windows, name).end.isoformat()} for name in splits},
        "sensor_layout": WaterHeaterSensorLayout().metadata(),
        "columns": {"outputs": list(REQUIRED_COLUMNS[:2]), "power": REQUIRED_COLUMNS[8], "disturbances": list(REQUIRED_COLUMNS[5:8])},
        "figure_files": figure_files,
        "training_data": str(training_data_path),
        "ochre_arguments": str(arguments_path),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata

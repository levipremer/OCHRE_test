"""Fit a two-node RC water heater from measured power and one-step temperatures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..water_heater_rc import WATER_HEAT_CAPACITY_KWH_PER_L_K, WaterHeaterRCModel
from .model_diagnostics import COLORS, SPLIT_FILLS, SPLIT_LABELS, _style_time_axis, plt
from .water_heater import WaterHeaterSensorLayout
from .water_heater_model import WaterHeaterSplitWindows, load_water_heater_training_data
from .water_heater_rc import REQUIRED_COLUMNS, _diagnostics, _geometry, _split


def _signals(frame: pd.DataFrame, model: WaterHeaterRCModel) -> tuple[np.ndarray, ...]:
    """Observed one-minute signals, with standby electricity removed."""
    upper, lower, next_upper, next_lower = (
        frame[name].to_numpy(float) for name in REQUIRED_COLUMNS[:4]
    )
    draw, mains, ambient, power = (
        frame[name].to_numpy(float) for name in REQUIRED_COLUMNS[5:9]
    )
    flow_cp = np.maximum(draw, 0) * 60 * WATER_HEAT_CAPACITY_KWH_PER_L_K
    active_power = np.maximum(power - model.off_power_kw, 0)
    return upper, lower, next_upper, next_lower, flow_cp, mains, ambient, active_power


def _fit_temperature(frame: pd.DataFrame, base: WaterHeaterRCModel) -> WaterHeaterRCModel:
    """Fit z and the shared-COP heat split from one-step temperature error."""
    upper, lower, next_upper, next_lower, flow, mains, ambient, power = _signals(frame, base)
    step_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    capacity = base.heat_capacity_kwh_per_k
    ua = base.ua_w_per_k / 1000
    conductance = 1 / base.inter_node_resistance_k_per_kw
    scale = step_hours / capacity
    upper_transfer = (conductance + flow) * (lower - upper)
    lower_transfer = conductance * (upper - lower) + flow * (mains - lower)
    upper_ambient = ua * (ambient - upper)
    lower_ambient = ua * (ambient - lower)

    def squared_temperature_error(x: np.ndarray) -> tuple[float, np.ndarray]:
        z, beta_upper, beta_lower = x
        upper_heat = upper_transfer + beta_upper * power
        lower_heat = lower_transfer + beta_lower * power
        upper_error = upper + scale * (upper_heat / z + upper_ambient) - next_upper
        lower_error = lower + scale * (lower_heat / (1 - z) + lower_ambient) - next_lower
        loss = 0.5 * np.mean(0.75 * upper_error**2 + 0.25 * lower_error**2)
        gradient = np.array([
            np.mean(scale * (-0.75 * upper_error * upper_heat / z**2
                             + 0.25 * lower_error * lower_heat / (1 - z)**2)),
            np.mean(0.75 * upper_error * scale * power / z),
            np.mean(0.25 * lower_error * scale * power / (1 - z)),
        ])
        return float(loss), gradient

    constraints = (
        {"type": "ineq", "fun": lambda x: x[1] + x[2] - 1,
         "jac": lambda x: np.array([0., 1., 1.])},
        {"type": "ineq", "fun": lambda x: 5 - x[1] - x[2],
         "jac": lambda x: np.array([0., -1., -1.])},
    )
    fits = [minimize(
        squared_temperature_error, x0=np.array([start, 1.5, 1.5]),
        jac=True, method="SLSQP", bounds=((0.1, 0.9), (0, 5), (0, 5)),
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000},
    ) for start in (0.2, 0.5, 0.8)]
    successful = [fit for fit in fits if fit.success]
    if not successful:
        raise RuntimeError(f"Constrained RC temperature fit failed: {fits[0].message}")
    result = min(successful, key=lambda fit: fit.fun)
    z, beta_upper, beta_lower = result.x
    cop = beta_upper + beta_lower
    return replace(base, upper_fraction=float(z), cop=float(cop),
                   upper_heat_fraction=float(beta_upper / cop))


def _one_step(frame: pd.DataFrame, model: WaterHeaterRCModel) -> tuple[np.ndarray, np.ndarray]:
    """Euler temperature prediction and inverse power from measured temperatures."""
    upper, lower, next_upper, next_lower, flow, mains, ambient, power = _signals(frame, model)
    step_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    capacity = model.heat_capacity_kwh_per_k
    z = model.upper_fraction
    ua = model.ua_w_per_k / 1000
    g = 1 / model.inter_node_resistance_k_per_kw
    beta_upper = model.cop * model.upper_heat_fraction
    beta_lower = model.cop * (1 - model.upper_heat_fraction)
    unheated_upper = upper + step_hours * (
        (g + flow) * (lower - upper) + z * ua * (ambient - upper)
    ) / (z * capacity)
    unheated_lower = lower + step_hours * (
        g * (upper - lower) + flow * (mains - lower)
        + (1 - z) * ua * (ambient - lower)
    ) / ((1 - z) * capacity)
    gain_upper = step_hours * beta_upper / (z * capacity)
    gain_lower = step_hours * beta_lower / ((1 - z) * capacity)
    predicted = np.column_stack((unheated_upper + gain_upper * power,
                                 unheated_lower + gain_lower * power))
    reconstructed = (
        gain_upper * (next_upper - unheated_upper)
        + gain_lower * (next_lower - unheated_lower)
    ) / (gain_upper**2 + gain_lower**2) + model.off_power_kw
    return predicted, reconstructed


def _predict_power(frame: pd.DataFrame, model: WaterHeaterRCModel, reset_steps: int) -> np.ndarray:
    """Measured-power Euler rollout; power column is inverse reconstruction."""
    upper, lower, _, _, flow, mains, ambient, power = _signals(frame, model)
    step_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    z = model.upper_fraction
    capacity = model.heat_capacity_kwh_per_k
    ua = model.ua_w_per_k / 1000
    g = 1 / model.inter_node_resistance_k_per_kw
    beta_upper = model.cop * model.upper_heat_fraction
    beta_lower = model.cop * (1 - model.upper_heat_fraction)
    predicted = np.empty((len(frame), 3))
    predicted[:, 2] = _one_step(frame, model)[1]
    state_u, state_l = upper[0], lower[0]
    for index in range(len(frame)):
        if index % reset_steps == 0:
            state_u, state_l = upper[index], lower[index]
        next_u = state_u + step_hours * (
            (g + flow[index]) * (state_l - state_u)
            + z * ua * (ambient[index] - state_u)
            + beta_upper * power[index]
        ) / (z * capacity)
        next_l = state_l + step_hours * (
            g * (state_u - state_l) + flow[index] * (mains[index] - state_l)
            + (1 - z) * ua * (ambient[index] - state_l)
            + beta_lower * power[index]
        ) / ((1 - z) * capacity)
        predicted[index, :2] = next_u, next_l
        state_u, state_l = next_u, next_l
    return predicted


def _metrics(frame: pd.DataFrame, model: WaterHeaterRCModel, scales: dict[str, float]) -> dict[str, float]:
    predicted, reconstructed = _one_step(frame, model)
    actual = frame[list(REQUIRED_COLUMNS[2:4])].to_numpy(float)
    temperature_rmse = np.sqrt(np.square(predicted - actual).mean(axis=0))
    power_rmse = float(np.sqrt(np.square(
        reconstructed - frame[REQUIRED_COLUMNS[8]].to_numpy(float)
    ).mean()))
    tank = float((0.75 * temperature_rmse[0] + 0.25 * temperature_rmse[1]) / scales["temperature_c"])
    power = power_rmse / scales["power_kw"]
    return {
        "score": (tank + power) / 2,
        "tank_error": tank, "power_error": power,
        "upper_rmse_c": float(temperature_rmse[0]),
        "lower_rmse_c": float(temperature_rmse[1]),
        "power_rmse_kw": power_rmse,
    }


def reconstruct_heat_input_euler(frame: pd.DataFrame, model: WaterHeaterRCModel) -> pd.DataFrame:
    """Euler inverse heat from sensors, independent of measured power."""
    _, reconstructed_power = _one_step(frame, model)
    return pd.DataFrame({
        "reference_kw": frame["water_heater_delivered_heat_kw"].to_numpy(float),
        "reconstructed_kw": model.cop * (reconstructed_power - model.off_power_kw),
    }, index=frame.index)


def _plot_power_reconstruction(
    splits: dict[str, pd.DataFrame], model: WaterHeaterRCModel,
    output: Path, split_labels: dict[str, str],
) -> None:
    figure, axis = plt.subplots(figsize=(16.5, 4.8))
    first = splits["training"].index[0]
    last = splits["test"].index[-1]
    for name, frame in splits.items():
        measured = frame[REQUIRED_COLUMNS[8]].to_numpy(float)
        reconstructed = _one_step(frame, model)[1]
        axis.axvspan(frame.index[0], frame.index[-1], color=SPLIT_FILLS[name], alpha=0.55)
        axis.plot(frame.index, measured, color=COLORS["measured"], lw=1,
                  label="Measured power" if name == "training" else None)
        axis.plot(frame.index, reconstructed, color=COLORS["rolling"], lw=0.9,
                  label="RC reconstructed power" if name == "training" else None)
        error = np.sqrt(np.square(reconstructed - measured).mean())
        midpoint = frame.index[0] + (frame.index[-1] - frame.index[0]) / 2
        axis.text(float((midpoint - first) / (last - first)), 0.98,
                  f"{split_labels[name]}\nRMSE {error:.3f} kW",
                  transform=axis.transAxes, ha="center", va="top", fontsize=8,
                  bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#888888"})
    for name in ("validation", "test"):
        axis.axvline(splits[name].index[0], color="#555555", ls="--", lw=1)
    axis.set(xlabel="Time", ylabel="Electric power (kW)",
             title="Water-heater power reconstructed from measured temperatures")
    axis.legend(loc="lower left", frameon=False)
    _style_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output / "figures" / "power_reconstruction.png", dpi=150)
    plt.close(figure)


def train_water_heater_rc_power(
    training_data_path: Path, arguments_path: Path, windows: WaterHeaterSplitWindows,
    output: Path, *, fit_stratification: bool = False, horizon_hours: float = 12,
    stratification_fractions: tuple[float, ...] = tuple(np.round(np.arange(.1, 1.01, .1), 2)),
    extra_stratification_m: tuple[float, ...] = (),
) -> dict:
    """Fit measured-power RC model, optionally selecting h_s on validation."""
    splits = _split(load_water_heater_training_data(training_data_path), windows)
    training = splits["training"]
    volume, height, ua, deadband, delivery = _geometry(arguments_path, training)
    step = training.index[1] - training.index[0]
    if any((part.index[1] - part.index[0]) != step for part in splits.values()):
        raise ValueError("RC splits must have the same sampling period")
    reset_steps = int(pd.Timedelta(hours=horizon_hours) / step)
    if reset_steps < 1:
        raise ValueError("RC prediction horizon must be at least one timestep")
    scales = {
        "temperature_c": 0.75 * (training[REQUIRED_COLUMNS[0]].max() - training[REQUIRED_COLUMNS[0]].min())
                         + 0.25 * (training[REQUIRED_COLUMNS[1]].max() - training[REQUIRED_COLUMNS[1]].min()),
        "power_kw": training[REQUIRED_COLUMNS[8]].max() - training[REQUIRED_COLUMNS[8]].min(),
    }
    scales = {key: float(value) for key, value in scales.items()}
    if min(scales.values()) <= 0:
        raise ValueError("RC training requires nonzero temperature and power ranges")
    fit_frame = training if fit_stratification else pd.concat((training, splits["validation"]))
    on_power = float(fit_frame.loc[fit_frame.water_heater_mode.eq("Heat Pump On"), REQUIRED_COLUMNS[8]].mean())
    off_power = float(fit_frame.loc[fit_frame.water_heater_mode.eq("Off"), REQUIRED_COLUMNS[8]].mean())
    base = WaterHeaterRCModel(volume, height, ua, .5, .5, 3, on_power, off_power, deadband)

    sweep = None
    selected_index = None
    if fit_stratification:
        if not stratification_fractions and not extra_stratification_m:
            raise ValueError("The h_s grid must contain at least one value")
        rows = []
        for fraction in stratification_fractions:
            if not 0 < fraction <= 1:
                raise ValueError("h_s fractions must lie in (0, 1]")
        heights = [float(height * fraction) for fraction in stratification_fractions]
        if any(thickness <= 0 for thickness in extra_stratification_m):
            raise ValueError("Extra h_s values must be positive")
        heights.extend(float(thickness) for thickness in extra_stratification_m)
        for thickness in heights:
            candidate = _fit_temperature(training, replace(base, stratification_thickness_m=thickness))
            scores = _metrics(splits["validation"], candidate, scales)
            rows.append({"stratification_thickness_m": thickness,
                         "upper_fraction": candidate.upper_fraction,
                         "upper_heat_fraction": candidate.upper_heat_fraction,
                         "cop": candidate.cop,
                         **{f"validation_{key}": value for key, value in scores.items()}})
        sweep = pd.DataFrame(rows)
        selected_index = int(sweep.validation_score.idxmin())
        chosen = sweep.loc[selected_index]
        selected = _fit_temperature(training, replace(base, stratification_thickness_m=float(chosen.stratification_thickness_m)))
    else:
        selected = _fit_temperature(fit_frame, base)

    test_metrics = _metrics(splits["test"], selected, scales)
    output.mkdir(parents=True, exist_ok=True)
    if sweep is not None:
        sweep.to_csv(output / "parameter_sweep.csv", index=False)
    else:
        for stale in (output / "parameter_sweep.csv", output / "figures" / "parameter_pareto.png",
                      output / "figures" / "pareto_front.csv"):
            stale.unlink(missing_ok=True)
    split_labels = dict(SPLIT_LABELS)
    if not fit_stratification:
        split_labels["validation"] = "Fit extension"
    figure_files = _diagnostics(
        splits, selected, sweep, selected_index, output, reset_steps,
        predictor=_predict_power,
        label="Two-node RC (measured power, fitted hₛ)" if fit_stratification else "Two-node RC (measured power)",
        heat_reconstructor=reconstruct_heat_input_euler, split_labels=split_labels,
    )
    _plot_power_reconstruction(splits, selected, output, split_labels)
    figure_files.append("power_reconstruction.png")
    power_rows = []
    for name, frame in splits.items():
        power_rows.append(pd.DataFrame({
            "split": name, "timestamp": frame.index,
            "measured_power_kw": frame[REQUIRED_COLUMNS[8]].to_numpy(float),
            "reconstructed_power_kw": _one_step(frame, selected)[1],
        }))
    pd.concat(power_rows, ignore_index=True).to_csv(output / "figures" / "power_reconstruction.csv", index=False)
    metadata = {
        "schema_version": 2,
        "model_type": "two_node_water_heater_rc",
        "fitting_method": "measured_power_euler_stratification_grid" if fit_stratification else "measured_power_euler_fixed_resistance",
        "parameters": selected.metadata(),
        "derived": {
            "heat_capacity_kwh_per_k": selected.heat_capacity_kwh_per_k,
            "ambient_resistance_k_per_kw": selected.ambient_resistance_k_per_kw,
            "inter_node_resistance_k_per_kw": selected.inter_node_resistance_k_per_kw,
            "maximum_heat_kw": selected.maximum_heat_kw,
        },
        "tempered_draw_temperature_c": delivery,
        "inter_node_resistance_method": "grid_searched_stratification_thickness" if fit_stratification else "fixed_half_tank_height",
        "fit_objective": "0.5*mean(0.75*upper_one_step_temperature_error_c^2 + 0.25*lower_one_step_temperature_error_c^2) with measured electric power",
        "selection_objective": "0.5*(0.75*upper_RMSE+0.25*lower_RMSE)/temperature_scale + 0.5*reconstructed_power_RMSE/power_scale" if fit_stratification else None,
        "normalization": scales,
        "selection": {"selected_grid_index": selected_index, "validation_score": float(sweep.loc[selected_index, "validation_score"])} if sweep is not None else None,
        "test_metrics": test_metrics,
        "sample_period_seconds": step.total_seconds(),
        "prediction_horizon_seconds": pd.Timedelta(hours=horizon_hours).total_seconds(),
        "windows": {name: {"start": getattr(windows, name).start.isoformat(),
                            "end": getattr(windows, name).end.isoformat()} for name in splits},
        "fit_windows": ["training"] if fit_stratification else ["training", "validation"],
        "stratification_grid_m": heights if fit_stratification else None,
        "sensor_layout": WaterHeaterSensorLayout().metadata(),
        "columns": {"outputs": list(REQUIRED_COLUMNS[:2]), "power": REQUIRED_COLUMNS[8],
                    "disturbances": list(REQUIRED_COLUMNS[5:8])},
        "figure_files": figure_files,
        "training_data": str(training_data_path),
        "ochre_arguments": str(arguments_path),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata

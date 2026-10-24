"""Diagnostic figures and CSV exports for trained state-space models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_matplotlib_cache = Path(tempfile.gettempdir()) / "ochre_training_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "measured": "#111111",
    "one_step": "#2f6690",
    "rolling": "#3a7d44",
    "residual": "#9e2a2b",
    "training_fill": "#dbe9f4",
    "validation_fill": "#fff2cc",
    "test_fill": "#d9ead3",
}
SPLIT_ORDER = ("training", "validation", "test")
SPLIT_LABELS = {
    "training": "Training",
    "validation": "Validation",
    "test": "Held-out test",
}
SPLIT_FILLS = {
    "training": COLORS["training_fill"],
    "validation": COLORS["validation_fill"],
    "test": COLORS["test_fill"],
}
SPLIT_LINES = {
    "training": COLORS["one_step"],
    "validation": "#c28e00",
    "test": COLORS["rolling"],
}


def create_state_space_diagnostic_outputs(
    *,
    model: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    model_output: Path,
    output_names: tuple[str, ...],
    disturbance_names: tuple[str, ...],
    model_label: str,
    solar_azimuth_by_disturbance: dict[str, int] | None = None,
    heat_input: dict[str, pd.DataFrame] | None = None,
) -> tuple[Path, ...]:
    """Write the requested prediction, residual, selection, and interpretation files."""
    figure_directory = Path(model_output) / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    bundles = {
        split: _prediction_bundle(model, frames[split])
        for split in SPLIT_ORDER
    }

    _write_prediction_csv(bundles, output_names, figure_directory)
    _write_residual_csv(bundles, output_names, figure_directory)
    horizon_data = _horizon_rmse_frame(model, frames, output_names)
    horizon_data.to_csv(figure_directory / "rmse_by_horizon.csv", index=False)

    figure_paths = (
        figure_directory / "three_split_predictions.png",
        figure_directory / "residual_diagnostics.png",
        figure_directory / "rmse_by_horizon.png",
        figure_directory / "measured_vs_predicted.png",
        figure_directory / "memory_selection.png",
        figure_directory / "lambda_selection.png",
        figure_directory / "selected_disturbances.png",
    )
    _plot_three_split_predictions(
        bundles,
        output_names,
        model_label,
        figure_paths[0],
        heat_input=heat_input,
    )
    _plot_residual_diagnostics(
        bundles,
        output_names,
        model_label,
        figure_paths[1],
    )
    _plot_rmse_by_horizon(
        horizon_data,
        output_names,
        model_label,
        figure_paths[2],
    )
    _plot_measured_vs_predicted(
        bundles,
        output_names,
        model_label,
        figure_paths[3],
    )
    _plot_memory_selection(model, model_label, figure_paths[4])
    _plot_lambda_selection(model, model_label, figure_paths[5])
    _plot_selected_disturbances(
        model,
        disturbance_names,
        model_label,
        figure_paths[6],
        solar_azimuth_by_disturbance or {},
    )
    return figure_paths


def _state_space_arrays(
    model: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nu = int(model["nu"])
    ny = int(model["ny"])
    nw = int(model["nw"])
    inputs = frame[
        [f"Input_{index + 1}" for index in range(nu)]
    ].to_numpy(float).T
    outputs = frame[
        [f"Output_{index + 1}" for index in range(ny)]
    ].to_numpy(float).T
    disturbances = frame[
        [f"Disturbance_{index + 1}" for index in range(nw)]
    ].to_numpy(float).T
    history_signals = np.vstack([inputs, disturbances, outputs])
    return inputs, outputs, disturbances, history_signals


def _prediction_bundle(
    model: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    inputs, outputs, disturbances, history_signals = _state_space_arrays(
        model, frame
    )
    memory = int(model["m"])
    horizon = int(model["horizon_steps"])
    sample_count = len(frame)
    output_count = outputs.shape[0]
    one_step = np.full((sample_count, output_count), np.nan)
    rolling = np.full((sample_count, output_count), np.nan)

    for step in range(memory, sample_count):
        history = history_signals[:, step - memory : step].reshape(
            -1, order="F"
        )
        one_step[step] = (
            model["P1"] @ history
            + model["F11"] @ inputs[:, step]
            + model["W11"] @ disturbances[:, step]
        )

    for origin in range(memory, sample_count, horizon):
        state = history_signals[:, origin - memory : origin].reshape(
            -1, order="F"
        )
        stop = min(origin + horizon, sample_count)
        for step in range(origin, stop):
            rolling[step] = (
                model["P1"] @ state
                + model["F11"] @ inputs[:, step]
                + model["W11"] @ disturbances[:, step]
            )
            state = (
                model["A"] @ state
                + model["B"] @ inputs[:, step]
                + model["E"] @ disturbances[:, step]
            )

    return {
        "index": frame.index,
        "measured": outputs.T,
        "one_step": one_step,
        "rolling": rolling,
    }


def _rmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(measured) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((measured[mask] - predicted[mask]) ** 2)))


def _bias_and_std(
    measured: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    residual = measured - predicted
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(residual)), float(np.std(residual))


def _output_label(output_name: str) -> str:
    if output_name == "indoor_temperature_c":
        return "Indoor temperature"
    marker = "water_heater_sensor_node_"
    if output_name.startswith(marker):
        node = int(output_name.removeprefix(marker).split("_")[0])
        position = "Upper" if node <= 6 else "Lower"
        return f"{position} sensor (node {node})"
    return output_name.replace("_", " ").removesuffix(" c").title()


def _write_prediction_csv(
    bundles: dict[str, dict[str, Any]],
    output_names: tuple[str, ...],
    directory: Path,
) -> None:
    frames = []
    for split in SPLIT_ORDER:
        bundle = bundles[split]
        values: dict[str, Any] = {
            "timestamp": bundle["index"],
            "split": split,
        }
        for output_index, output_name in enumerate(output_names):
            values[f"measured_{output_name}"] = bundle["measured"][
                :, output_index
            ]
            values[f"one_step_{output_name}"] = bundle["one_step"][
                :, output_index
            ]
            values[f"rolling_12h_{output_name}"] = bundle["rolling"][
                :, output_index
            ]
        frames.append(pd.DataFrame(values))
    pd.concat(frames, ignore_index=True).to_csv(
        directory / "predictions.csv", index=False
    )


def _write_residual_csv(
    bundles: dict[str, dict[str, Any]],
    output_names: tuple[str, ...],
    directory: Path,
) -> None:
    frames = []
    for split in SPLIT_ORDER:
        bundle = bundles[split]
        values: dict[str, Any] = {
            "timestamp": bundle["index"],
            "split": split,
        }
        for output_index, output_name in enumerate(output_names):
            measured = bundle["measured"][:, output_index]
            values[f"one_step_residual_{output_name}"] = (
                measured - bundle["one_step"][:, output_index]
            )
            values[f"rolling_12h_residual_{output_name}"] = (
                measured - bundle["rolling"][:, output_index]
            )
        frames.append(pd.DataFrame(values))
    pd.concat(frames, ignore_index=True).to_csv(
        directory / "residuals.csv", index=False
    )


def _horizon_rmse_frame(
    model: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    output_names: tuple[str, ...],
) -> pd.DataFrame:
    memory = int(model["m"])
    horizon = int(model["horizon_steps"])
    rows = []
    for split in SPLIT_ORDER:
        inputs, outputs, disturbances, history_signals = _state_space_arrays(
            model, frames[split]
        )
        squared_error = np.zeros((outputs.shape[0], horizon))
        origin_count = 0
        for origin in range(memory, outputs.shape[1] - horizon + 1):
            state = history_signals[:, origin - memory : origin].reshape(
                -1, order="F"
            )
            for lead in range(horizon):
                step = origin + lead
                prediction = (
                    model["P1"] @ state
                    + model["F11"] @ inputs[:, step]
                    + model["W11"] @ disturbances[:, step]
                )
                squared_error[:, lead] += (outputs[:, step] - prediction) ** 2
                state = (
                    model["A"] @ state
                    + model["B"] @ inputs[:, step]
                    + model["E"] @ disturbances[:, step]
                )
            origin_count += 1
        if origin_count == 0:
            raise ValueError(
                f"{split} split has no complete rolling prediction horizon"
            )

        rmse = np.sqrt(squared_error / origin_count)
        for lead in range(horizon):
            for output_index, output_name in enumerate(output_names):
                rows.append(
                    {
                        "split": split,
                        "lead_step": lead + 1,
                        "lead_minutes": 15 * (lead + 1),
                        "output": output_name,
                        "rmse_c": float(rmse[output_index, lead]),
                        "origin_count": origin_count,
                    }
                )
            rows.append(
                {
                    "split": split,
                    "lead_step": lead + 1,
                    "lead_minutes": 15 * (lead + 1),
                    "output": "combined",
                    "rmse_c": float(
                        np.sqrt(np.mean(squared_error[:, lead]) / origin_count)
                    ),
                    "origin_count": origin_count,
                }
            )
    return pd.DataFrame(rows)


def _style_time_axis(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.grid(True, alpha=0.22, linewidth=0.7)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_three_split_predictions(
    bundles: dict[str, dict[str, Any]],
    output_names: tuple[str, ...],
    model_label: str,
    path: Path,
    *,
    heat_input: dict[str, pd.DataFrame] | None = None,
    split_labels: dict[str, str] | None = None,
) -> None:
    split_labels = split_labels or SPLIT_LABELS
    figure, axes = plt.subplots(
        len(output_names) + (heat_input is not None),
        1,
        figsize=(16.5, 4.6 * (len(output_names) + (heat_input is not None))),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    first_timestamp = bundles["training"]["index"][0]
    last_timestamp = bundles["test"]["index"][-1]
    full_duration = last_timestamp - first_timestamp

    for output_index, (axis, output_name) in enumerate(
        zip(axes_flat, output_names)
    ):
        for split in SPLIT_ORDER:
            bundle = bundles[split]
            index = bundle["index"]
            axis.axvspan(
                index[0],
                index[-1],
                color=SPLIT_FILLS[split],
                alpha=0.55,
                zorder=0,
            )
            axis.plot(
                index,
                bundle["measured"][:, output_index],
                color=COLORS["measured"],
                linewidth=1.15,
                label="Measured" if split == "training" else None,
                zorder=3,
            )
            axis.plot(
                index,
                bundle["one_step"][:, output_index],
                color=COLORS["one_step"],
                linewidth=0.9,
                alpha=0.9,
                label="One-step" if split == "training" else None,
            )
            axis.plot(
                index,
                bundle["rolling"][:, output_index],
                color=COLORS["rolling"],
                linewidth=1.0,
                label="12-hour rolling/reset" if split == "training" else None,
            )

            midpoint = index[0] + (index[-1] - index[0]) / 2
            x_fraction = (midpoint - first_timestamp) / full_duration
            one_step_rmse = _rmse(
                bundle["measured"][:, output_index],
                bundle["one_step"][:, output_index],
            )
            rolling_rmse = _rmse(
                bundle["measured"][:, output_index],
                bundle["rolling"][:, output_index],
            )
            axis.text(
                float(x_fraction),
                0.985,
                f"{split_labels[split]}\n"
                f"1-step RMSE {one_step_rmse:.3f} °C\n"
                f"12 h RMSE {rolling_rmse:.3f} °C",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=7.5,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.86,
                    "edgecolor": "#888888",
                },
                zorder=8,
            )

        axis.set_ylabel("Temperature (°C)")
        axis.set_title(_output_label(output_name), loc="left", fontsize=10)

    if heat_input is not None:
        axis = axes_flat[-1]
        for split in SPLIT_ORDER:
            part = heat_input[split]
            axis.axvspan(part.index[0], part.index[-1], color=SPLIT_FILLS[split], alpha=0.55, zorder=0)
            axis.plot(part.index, part["reference_kw"], color=COLORS["measured"], linewidth=1.15,
                      label="OCHRE delivered heat" if split == "training" else None, zorder=3)
            axis.plot(part.index, part["reconstructed_kw"], color=COLORS["rolling"], linewidth=1,
                      label="RC reconstructed heat" if split == "training" else None)
            midpoint = part.index[0] + (part.index[-1] - part.index[0]) / 2
            x_fraction = (midpoint - first_timestamp) / full_duration
            error = _rmse(part["reference_kw"].to_numpy(), part["reconstructed_kw"].to_numpy())
            axis.text(float(x_fraction), 0.985, f"{split_labels[split]}\nReconstruction RMSE {error:.3f} kW",
                      transform=axis.transAxes, ha="center", va="top", fontsize=7.5,
                      bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#888888"}, zorder=8)
        axis.set_ylabel("Heat input (kW)")
        axis.set_title("Delivered heat: RC balance reconstruction", loc="left", fontsize=10)

    for axis in axes_flat:
        for split in ("validation", "test"):
            boundary = bundles[split]["index"][0]
            axis.axvline(boundary, color="#555555", linestyle="--", linewidth=1.1)
            axis.annotate(f"{split_labels[split]} start", xy=(boundary, 1),
                          xycoords=("data", "axes fraction"), xytext=(4, -4),
                          textcoords="offset points", rotation=90, va="top",
                          ha="left", fontsize=7.5, color="#444444")
        axis.legend(loc="lower left", ncol=3, frameon=False, fontsize=8)
        _style_time_axis(axis)

    axes_flat[-1].set_xlabel("Time")
    figure.suptitle(
        f"{model_label}: {split_labels['training'].lower()}, "
        f"{split_labels['validation'].lower()}, and {split_labels['test'].lower()} predictions",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_residual_diagnostics(
    bundles: dict[str, dict[str, Any]],
    output_names: tuple[str, ...],
    model_label: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15, 10.5),
        gridspec_kw={"width_ratios": [2.4, 1]},
    )
    linestyles = ("-", "--", ":", "-.")
    for row, split in enumerate(SPLIT_ORDER):
        bundle = bundles[split]
        trace_axis, histogram_axis = axes[row]
        annotation_lines = []
        for output_index, output_name in enumerate(output_names):
            measured = bundle["measured"][:, output_index]
            label = _output_label(output_name)
            for prediction_name, color in (
                ("one_step", COLORS["one_step"]),
                ("rolling", COLORS["rolling"]),
            ):
                residual = measured - bundle[prediction_name][:, output_index]
                finite = np.isfinite(residual)
                display_prediction = (
                    "One-step" if prediction_name == "one_step" else "12-hour"
                )
                trace_axis.plot(
                    bundle["index"],
                    residual,
                    color=color,
                    linestyle=linestyles[output_index % len(linestyles)],
                    linewidth=0.8,
                    alpha=0.85,
                    label=f"{label}, {display_prediction}",
                )
                histogram_axis.hist(
                    residual[finite],
                    bins=30,
                    density=True,
                    histtype="step",
                    color=color,
                    linestyle=linestyles[output_index % len(linestyles)],
                    linewidth=1.2,
                )
                bias, standard_deviation = _bias_and_std(
                    measured, bundle[prediction_name][:, output_index]
                )
                annotation_lines.append(
                    f"{label}, {display_prediction}: "
                    f"bias {bias:+.3f}, σ {standard_deviation:.3f} °C"
                )

        trace_axis.axhline(
            0,
            color=COLORS["residual"],
            linewidth=1.0,
            linestyle="--",
        )
        histogram_axis.axvline(
            0,
            color=COLORS["residual"],
            linewidth=1.0,
            linestyle="--",
        )
        trace_axis.set_title(f"{SPLIT_LABELS[split]} residual trace", loc="left")
        histogram_axis.set_title("Residual distribution", loc="left")
        trace_axis.set_ylabel("Measured − predicted (°C)")
        histogram_axis.set_xlabel("Residual (°C)")
        histogram_axis.set_ylabel("Density")
        trace_axis.text(
            0.01,
            0.97,
            "\n".join(annotation_lines),
            transform=trace_axis.transAxes,
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        trace_axis.legend(loc="lower left", frameon=False, fontsize=7, ncol=2)
        _style_time_axis(trace_axis)
        histogram_axis.grid(True, alpha=0.22, linewidth=0.7)

    axes[-1, 0].set_xlabel("Time")
    figure.suptitle(f"{model_label}: residual diagnostics", fontsize=13)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_rmse_by_horizon(
    data: pd.DataFrame,
    output_names: tuple[str, ...],
    model_label: str,
    path: Path,
) -> None:
    panels = (*output_names, "combined") if len(output_names) > 1 else output_names
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.0 * len(panels), 4.8),
        squeeze=False,
        sharex=True,
    )
    for axis, output_name in zip(axes[0], panels):
        for split in SPLIT_ORDER:
            selected = data[
                (data["split"] == split) & (data["output"] == output_name)
            ]
            axis.plot(
                selected["lead_minutes"] / 60,
                selected["rmse_c"],
                color=SPLIT_LINES[split],
                linewidth=1.7,
                label=SPLIT_LABELS[split],
            )
        title = "Combined RMSE" if output_name == "combined" else _output_label(
            output_name
        )
        axis.set_title(title)
        axis.set_xlabel("Prediction lead time (hours)")
        axis.set_ylabel("RMSE (°C)")
        axis.set_xlim(0.25, float(data["lead_minutes"].max() / 60))
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{model_label}: RMSE by rolling-forecast horizon", fontsize=13)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_measured_vs_predicted(
    bundles: dict[str, dict[str, Any]],
    output_names: tuple[str, ...],
    model_label: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(output_names),
        3,
        figsize=(15, 4.7 * len(output_names)),
        squeeze=False,
    )
    for output_index, output_name in enumerate(output_names):
        for split_index, split in enumerate(SPLIT_ORDER):
            axis = axes[output_index, split_index]
            bundle = bundles[split]
            measured = bundle["measured"][:, output_index]
            predictions = (
                ("one_step", "One-step", COLORS["one_step"]),
                ("rolling", "12-hour", COLORS["rolling"]),
            )
            finite_values = [measured[np.isfinite(measured)]]
            annotation_lines = []
            for key, label, color in predictions:
                predicted = bundle[key][:, output_index]
                mask = np.isfinite(measured) & np.isfinite(predicted)
                finite_values.append(predicted[mask])
                axis.scatter(
                    measured[mask],
                    predicted[mask],
                    s=10,
                    alpha=0.48,
                    color=color,
                    edgecolors="none",
                    label=label,
                )
                rmse = _rmse(measured, predicted)
                bias, _ = _bias_and_std(measured, predicted)
                annotation_lines.append(
                    f"{label}: RMSE {rmse:.3f} °C, bias {bias:+.3f} °C"
                )
            finite_values = [values for values in finite_values if values.size]
            minimum = min(float(np.min(values)) for values in finite_values)
            maximum = max(float(np.max(values)) for values in finite_values)
            padding = max((maximum - minimum) * 0.04, 0.1)
            limits = (minimum - padding, maximum + padding)
            axis.plot(limits, limits, color=COLORS["measured"], linewidth=1.1)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(
                f"{SPLIT_LABELS[split]} — {_output_label(output_name)}",
                fontsize=9.5,
            )
            axis.set_xlabel("Measured temperature (°C)")
            axis.set_ylabel("Predicted temperature (°C)")
            axis.text(
                0.03,
                0.97,
                "\n".join(annotation_lines),
                transform=axis.transAxes,
                va="top",
                fontsize=7.2,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.83,
                    "edgecolor": "none",
                },
            )
            axis.grid(True, alpha=0.22, linewidth=0.7)
            axis.legend(loc="lower right", frameon=False, fontsize=7.5)
    figure.suptitle(f"{model_label}: measured versus predicted", fontsize=13)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_memory_selection(
    model: dict[str, Any],
    model_label: str,
    path: Path,
) -> None:
    memories = np.asarray(model["memory_candidates"], dtype=int)
    aicc = np.asarray(model["aicc_values"], dtype=float)
    delta = aicc - np.min(aicc)
    standard_memory = int(model["standard_aicc_memory"])
    selected_memory = int(model["relative_aicc_memory"])
    value_by_memory = dict(zip(memories.tolist(), delta.tolist()))

    data = pd.DataFrame(
        {
            "memory_steps": memories,
            "memory_minutes": memories * 15,
            "aicc": aicc,
            "delta_aicc": delta,
            "minimum_aicc": memories == standard_memory,
            "relative_aicc_selected": memories == selected_memory,
        }
    )
    data.to_csv(path.parent / "memory_selection.csv", index=False)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.plot(memories, delta, color=COLORS["one_step"], marker="o", linewidth=1.7)
    axis.scatter(
        [standard_memory],
        [value_by_memory[standard_memory]],
        marker="D",
        s=95,
        color=COLORS["measured"],
        zorder=5,
        label="Minimum AICc",
    )
    axis.scatter(
        [selected_memory],
        [value_by_memory[selected_memory]],
        marker="s",
        s=120,
        facecolor="none",
        edgecolor=COLORS["residual"],
        linewidth=2,
        zorder=6,
        label="Relative-AICc selection",
    )
    axis.set_xlabel("Memory (15-minute steps)")
    axis.set_ylabel("AICc increase above minimum")
    axis.set_title(f"{model_label}: memory selection")
    axis.set_xticks(memories)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, path)


def _pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return points
    best_by_count = (
        points.sort_values("validation_rmse_c")
        .drop_duplicates("selected_disturbance_count")
        .sort_values("selected_disturbance_count")
    )
    keep = []
    best_rmse = np.inf
    for row in best_by_count.itertuples():
        if row.validation_rmse_c < best_rmse:
            keep.append(row.Index)
            best_rmse = row.validation_rmse_c
    return best_by_count.loc[keep].sort_values("selected_disturbance_count")


def _plot_lambda_selection(
    model: dict[str, Any],
    model_label: str,
    path: Path,
) -> None:
    diagnostics = model.get("regularization_diagnostics", {}).get(
        int(model["m"]), []
    )
    rows = []
    selected_lambda = model.get("selected_lambda")
    for diagnostic in diagnostics:
        validation_rmse = diagnostic.get("validation_rmse")
        if diagnostic.get("status") != "ok" or not np.isfinite(validation_rmse):
            continue
        active = diagnostic.get("active_disturbances", [])
        rows.append(
            {
                "lambda": float(diagnostic["lambda"]),
                "selected_disturbance_count": len(active),
                "validation_rmse_c": float(validation_rmse),
                "active_disturbance_indices": ";".join(map(str, active)),
                "selected": (
                    selected_lambda is not None
                    and np.isclose(diagnostic["lambda"], selected_lambda)
                ),
            }
        )
    data = pd.DataFrame(
        rows,
        columns=(
            "lambda",
            "selected_disturbance_count",
            "validation_rmse_c",
            "active_disturbance_indices",
            "selected",
        ),
    )
    data.to_csv(path.parent / "lambda_selection.csv", index=False)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    if data.empty:
        axis.text(
            0.5,
            0.5,
            "Feature selection disabled\nNo lambda path",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_axis_off()
    else:
        front = _pareto_front(data)
        axis.scatter(
            data["selected_disturbance_count"],
            data["validation_rmse_c"],
            color="0.65",
            s=32,
            label="Lambda candidate",
        )
        axis.plot(
            front["selected_disturbance_count"],
            front["validation_rmse_c"],
            color=COLORS["one_step"],
            marker="o",
            linewidth=1.8,
            label="Pareto front",
        )
        selected = data[data["selected"]]
        if not selected.empty:
            axis.scatter(
                selected["selected_disturbance_count"],
                selected["validation_rmse_c"],
                marker="*",
                s=190,
                color=COLORS["residual"],
                zorder=5,
                label=f"Selected λ={selected_lambda:.3g}",
            )
        axis.set_xlabel("Selected disturbance count")
        axis.set_ylabel("Validation rolling-horizon RMSE (°C)")
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.legend(frameon=False)
    axis.set_title(f"{model_label}: lambda selection")
    figure.tight_layout()
    _save_figure(figure, path)


def _disturbance_strengths(
    model: dict[str, Any],
    disturbance_names: tuple[str, ...],
) -> pd.DataFrame:
    memory = int(model["m"])
    nu = int(model["nu"])
    nw = int(model["nw"])
    ny = int(model["ny"])
    nz = nu + nw + ny
    selected = {int(index) for index in model["selected_disturbance_indices"]}
    rows = []
    for disturbance_index, name in enumerate(disturbance_names):
        history_columns = [
            lag * nz + nu + disturbance_index for lag in range(memory)
        ]
        coefficients = np.concatenate(
            [
                model["P1"][:, history_columns].reshape(-1),
                model["W11"][:, disturbance_index].reshape(-1),
            ]
        )
        rows.append(
            {
                "disturbance": name,
                "selected": disturbance_index in selected,
                "coefficient_norm": float(np.linalg.norm(coefficients)),
            }
        )
    return pd.DataFrame(rows)


def _plot_selected_disturbances(
    model: dict[str, Any],
    disturbance_names: tuple[str, ...],
    model_label: str,
    path: Path,
    solar_azimuth_by_disturbance: dict[str, int],
) -> None:
    data = _disturbance_strengths(model, disturbance_names)
    data["azimuth_degrees"] = data["disturbance"].map(
        solar_azimuth_by_disturbance
    )
    data["disturbance_type"] = np.where(
        data["azimuth_degrees"].notna(), "projected_dni", "non_solar"
    )
    data.to_csv(path.parent / "selected_disturbances.csv", index=False)

    has_solar = bool(solar_azimuth_by_disturbance)
    if has_solar:
        figure = plt.figure(figsize=(14, 5.8))
        grid = figure.add_gridspec(1, 2, width_ratios=[1.2, 1])
        bar_axis = figure.add_subplot(grid[0, 0])
        polar_axis = figure.add_subplot(grid[0, 1], projection="polar")
    else:
        figure, bar_axis = plt.subplots(figsize=(9.5, 5.2))
        polar_axis = None

    non_solar = data[
        (data["disturbance_type"] == "non_solar") & data["selected"]
    ]
    if non_solar.empty:
        bar_axis.text(
            0.5,
            0.5,
            "No non-solar disturbances selected",
            transform=bar_axis.transAxes,
            ha="center",
            va="center",
        )
    else:
        labels = [
            name.replace("_", " ") for name in non_solar["disturbance"]
        ]
        bar_axis.barh(
            labels,
            non_solar["coefficient_norm"],
            color=COLORS["one_step"],
        )
        bar_axis.invert_yaxis()
    bar_axis.set_xlabel("One-step coefficient group norm")
    bar_axis.set_title("Selected non-solar disturbances")
    bar_axis.grid(axis="x", alpha=0.22, linewidth=0.7)

    if polar_axis is not None:
        solar = data[
            (data["disturbance_type"] == "projected_dni") & data["selected"]
        ].sort_values("azimuth_degrees")
        polar_axis.set_theta_zero_location("N")
        polar_axis.set_theta_direction(-1)
        if not solar.empty:
            polar_axis.bar(
                np.deg2rad(solar["azimuth_degrees"].to_numpy(float)),
                solar["coefficient_norm"],
                width=np.deg2rad(8),
                color=COLORS["rolling"],
                alpha=0.82,
            )
        else:
            polar_axis.text(
                0.5,
                0.5,
                "No projected-DNI\norientation selected",
                transform=polar_axis.transAxes,
                ha="center",
                va="center",
            )
        polar_axis.set_xticks(
            np.deg2rad([0, 90, 180, 270]),
            labels=["N (0°)", "E (90°)", "S (180°)", "W (270°)"],
        )
        polar_axis.set_title("Selected projected-DNI orientations", pad=18)
        polar_axis.grid(alpha=0.25)

    figure.suptitle(f"{model_label}: selected disturbances", fontsize=13)
    figure.tight_layout()
    _save_figure(figure, path)

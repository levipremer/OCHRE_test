"""TPC-style rollout diagnostics for the HVAC state-space predictor.

The predictor observes indoor air temperature only.  Its history state is not
the physical latent-mass temperature of the separate 2R2C model.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_cache = Path(tempfile.gettempdir()) / "ochre_training_matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


PREDICTION_NAMES = ("one_step", "8_hour", "24_hour", "open_loop")
COLORS = {
    "measured": "#111111",
    "one_step": "#245b9a",
    "8_hour": "#5d9bd3",
    "24_hour": "#dc8125",
    "open_loop": "#328049",
}
LABELS = {
    "one_step": "One-step",
    "8_hour": "8-hour",
    "24_hour": "24-hour",
    "open_loop": "Open-loop",
}


def rollout_predictions(
    model: dict[str, Any], frame: pd.DataFrame, model_timestep: pd.Timedelta
) -> pd.DataFrame:
    """Predict each split independently using measured-history initialization."""
    timestep = pd.Timedelta(model_timestep)
    if timestep <= pd.Timedelta(0) or pd.Timedelta(hours=8) % timestep:
        raise ValueError("model_timestep must divide eight hours")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Prediction timestamps must be unique and increasing")
    if len(frame) > 1 and not np.all(np.diff(frame.index.asi8) == timestep.value):
        raise ValueError("Prediction frame must contain consecutive model intervals")

    nu, nw, ny = (int(model[key]) for key in ("nu", "nw", "ny"))
    if ny != 1:
        raise ValueError("HVAC envelope evaluation expects one indoor output")
    inputs = frame[[f"Input_{i + 1}" for i in range(nu)]].to_numpy(float).T
    disturbances = frame[
        [f"Disturbance_{i + 1}" for i in range(nw)]
    ].to_numpy(float).T
    measured = frame["Output_1"].to_numpy(float)
    if not all(np.isfinite(array).all() for array in (inputs, disturbances, measured)):
        raise ValueError("Prediction frame contains nonfinite values")
    history_signals = np.vstack((inputs, disturbances, measured[None, :]))
    memory = int(model["m"])
    if len(frame) <= memory:
        raise ValueError("Prediction frame is shorter than model memory")
    reset_steps = {
        "8_hour": int(pd.Timedelta(hours=8) // timestep),
        "24_hour": int(pd.Timedelta(hours=24) // timestep),
        "open_loop": None,
    }
    result = pd.DataFrame(index=frame.index)
    result["measured"] = measured
    one_step = np.full(len(frame), np.nan)
    for step in range(memory, len(frame)):
        history = history_signals[:, step - memory : step].reshape(-1, order="F")
        one_step[step] = (
            model["P1"] @ history
            + model["F11"] @ inputs[:, step]
            + model["W11"] @ disturbances[:, step]
        ).item()
    result["one_step"] = one_step
    for name, reset_interval in reset_steps.items():
        predictions = np.full(len(frame), np.nan)
        state = history_signals[:, :memory].reshape(-1, order="F")
        for step in range(memory, len(frame)):
            if reset_interval is not None and (step - memory) % reset_interval == 0:
                state = history_signals[:, step - memory : step].reshape(-1, order="F")
            predictions[step] = (
                model["P1"] @ state
                + model["F11"] @ inputs[:, step]
                + model["W11"] @ disturbances[:, step]
            ).item()
            state = (
                model["A"] @ state
                + model["B"] @ inputs[:, step]
                + model["E"] @ disturbances[:, step]
            )
        result[name] = predictions
    return result


def _rmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(measured) & np.isfinite(predicted)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((measured[mask] - predicted[mask]) ** 2)))


def save_rollout_evaluation(
    model: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    model_timestep: pd.Timedelta,
    output_directory: Path,
    temperature_limits_c: tuple[float, float] | None = None,
) -> tuple[dict[str, dict[str, float]], Path, tuple[float, float]]:
    """Save train/validation/test traces and a four-rollout comparison figure."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    order = ("training", "validation", "test")
    predictions = {
        split: rollout_predictions(model, frames[split], model_timestep)
        for split in order
    }
    metrics = {
        split: {
            name: _rmse(
                values["measured"].to_numpy(), values[name].to_numpy()
            )
            for name in PREDICTION_NAMES
        }
        for split, values in predictions.items()
    }
    pd.concat(
        [values.assign(split=split) for split, values in predictions.items()],
        names=["timestamp"],
    ).rename_axis("timestamp").to_csv(directory / "rollout_predictions.csv")
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("split").to_csv(
        directory / "rollout_metrics.csv"
    )

    if temperature_limits_c is None:
        finite_values = np.concatenate(
            [
                values[column].to_numpy()[np.isfinite(values[column].to_numpy())]
                for values in predictions.values()
                for column in ("measured", *PREDICTION_NAMES)
            ]
        )
        lower, upper = float(finite_values.min()), float(finite_values.max())
        padding = max(0.5, 0.04 * (upper - lower))
        temperature_limits_c = (lower - padding, upper + padding)
    if temperature_limits_c[0] >= temperature_limits_c[1]:
        raise ValueError("temperature_limits_c must be increasing")
    (directory / "axis_limits.json").write_text(
        json.dumps({"indoor_temperature_c": list(temperature_limits_c)}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(3, 1, figsize=(15.5, 10.5), sharey=True)
    for axis, split in zip(axes, order):
        values = predictions[split]
        axis.plot(values.index, values["measured"], color=COLORS["measured"],
                  linewidth=1.3, label="Measured", zorder=5)
        for name in PREDICTION_NAMES:
            axis.plot(values.index, values[name], color=COLORS[name],
                      linewidth=1.0, alpha=0.93, label=LABELS[name])
        axis.set_ylim(*temperature_limits_c)
        axis.set_ylabel("Indoor air (°C)")
        axis.set_title(split.capitalize(), loc="left")
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
        annotation = "RMSE (°C)  " + "  |  ".join(
            f"{LABELS[name]} {metrics[split][name]:.2f}" for name in PREDICTION_NAMES
        )
        axis.text(0.01, 0.97, annotation, transform=axis.transAxes, va="top",
                  fontsize=8, bbox={"facecolor": "white", "alpha": 0.88,
                                    "edgecolor": "#777777"})
    axes[0].legend(loc="lower left", ncol=5, frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time")
    figure.suptitle("Regularized state-space HVAC envelope: indoor air forecasts")
    figure.tight_layout()
    figure_path = directory / "rollout_predictions.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return metrics, figure_path, temperature_limits_c

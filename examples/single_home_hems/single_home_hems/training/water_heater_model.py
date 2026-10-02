"""Dataset preparation and user-owned training hook for a water-heater model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_diagnostics import create_state_space_diagnostic_outputs
from .state_space_predictor_regularized import (
    train_state_space_predictor_all_features,
    train_state_space_predictor_group_lasso,
)
from .water_heater import WaterHeaterSensorLayout
from ..water_heater_rc import load_rc_model, reconstruct_heat_input


DELIVERED_HEAT_COLUMN = "water_heater_delivered_heat_kw"

DISTURBANCE_COLUMNS = (
    "water_heater_draw_l_per_min",
    "water_heater_mains_temperature_c",
    "water_heater_tank_ambient_temperature_c",
)

SETPOINT_COLUMN = "water_heater_setpoint_temperature_c"
POWER_COLUMN = "water_heater_electric_power_kw"

DIAGNOSTIC_COLUMNS = (
    SETPOINT_COLUMN,
    POWER_COLUMN,
    "water_heater_mode",
    "water_heater_tank_volume_l",
    "water_heater_resistance_capacity_w",
    "water_heater_nominal_heat_pump_capacity_w",
    "water_heater_nominal_heat_pump_cop",
)


@dataclass(frozen=True)
class WaterHeaterTrainingWindow:
    """Half-open training interval, kept separate from deployment dates."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if end <= start:
            raise ValueError("Water-heater training end must be after its start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class WaterHeaterTrainingDataset:
    """Validated feature, target, and diagnostic tables for model fitting."""

    features: pd.DataFrame
    targets: pd.DataFrame
    diagnostics: pd.DataFrame
    window: WaterHeaterTrainingWindow

    @property
    def sample_count(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class WaterHeaterSplitWindows:
    """Chronological, non-overlapping train/validation/test windows."""

    training: WaterHeaterTrainingWindow
    validation: WaterHeaterTrainingWindow
    test: WaterHeaterTrainingWindow

    def __post_init__(self) -> None:
        if self.training.end > self.validation.start:
            raise ValueError("Water-heater training and validation windows overlap")
        if self.validation.end > self.test.start:
            raise ValueError("Water-heater validation and test windows overlap")


@dataclass(frozen=True)
class WaterHeaterStateSpaceConfig:
    """Settings for the regularized water-heater state-space model."""

    source_timestep: pd.Timedelta = pd.Timedelta(minutes=1)
    model_timestep: pd.Timedelta = pd.Timedelta(minutes=15)
    prediction_horizon: pd.Timedelta = pd.Timedelta(hours=12)

    disturbance_columns: tuple[str, ...] = DISTURBANCE_COLUMNS

    memory_min: int = 1
    memory_max: int = 12

    feature_selection_method: str = "group_lasso"
    scaling: str | None = None
    cvxpy_solver: str = "gurobi"
    auto_lambda_grid_points: int = 30
    relative_aicc_threshold: float = 0.05

    def __post_init__(self) -> None:
        source_timestep = pd.Timedelta(self.source_timestep)
        model_timestep = pd.Timedelta(self.model_timestep)
        prediction_horizon = pd.Timedelta(self.prediction_horizon)
        disturbance_columns = tuple(self.disturbance_columns)

        for name, value in (
            ("source_timestep", source_timestep),
            ("model_timestep", model_timestep),
            ("prediction_horizon", prediction_horizon),
        ):
            if value <= pd.Timedelta(0):
                raise ValueError(f"{name} must be positive")
        if model_timestep % source_timestep:
            raise ValueError(
                "model_timestep must be an integer multiple of source_timestep"
            )
        if prediction_horizon % model_timestep:
            raise ValueError(
                "prediction_horizon must be an integer multiple of model_timestep"
            )
        if not disturbance_columns:
            raise ValueError("At least one water-heater disturbance is required")
        if len(set(disturbance_columns)) != len(disturbance_columns):
            raise ValueError("Water-heater disturbance columns must be unique")
        unsupported = set(disturbance_columns) - set(DISTURBANCE_COLUMNS)
        if unsupported:
            raise ValueError(
                f"Unsupported water-heater disturbances: {sorted(unsupported)}"
            )
        for name, value in (
            ("memory_min", self.memory_min),
            ("memory_max", self.memory_max),
            ("auto_lambda_grid_points", self.auto_lambda_grid_points),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_min > self.memory_max:
            raise ValueError("memory_min cannot exceed memory_max")
        if self.feature_selection_method not in {"group_lasso", "none"}:
            raise ValueError(
                "feature_selection_method must be 'group_lasso' or 'none'"
            )
        scaling = self.scaling
        if scaling is None:
            scaling = (
                "standard_deviation"
                if self.feature_selection_method == "group_lasso"
                else "none"
            )
        if scaling not in {"none", "standard_deviation"}:
            raise ValueError("scaling must be 'none' or 'standard_deviation'")
        if self.feature_selection_method == "none" and scaling != "none":
            raise ValueError(
                "scaling must be 'none' when feature selection is disabled"
            )
        if self.cvxpy_solver not in {"gurobi", "clarabel", "scs"}:
            raise ValueError("cvxpy_solver must be 'gurobi', 'clarabel', or 'scs'")
        if not 0 <= self.relative_aicc_threshold <= 1:
            raise ValueError("relative_aicc_threshold must be between zero and one")

        object.__setattr__(self, "source_timestep", source_timestep)
        object.__setattr__(self, "model_timestep", model_timestep)
        object.__setattr__(self, "prediction_horizon", prediction_horizon)
        object.__setattr__(self, "disturbance_columns", disturbance_columns)
        object.__setattr__(self, "scaling", scaling)

    @property
    def horizon_steps(self) -> int:
        return self.prediction_horizon // self.model_timestep


@dataclass(frozen=True)
class WaterHeaterDatasetSplits:
    """Prepared datasets for fitting, tuning, and final evaluation."""

    training: WaterHeaterTrainingDataset
    validation: WaterHeaterTrainingDataset
    test: WaterHeaterTrainingDataset


def next_sensor_columns(layout: WaterHeaterSensorLayout) -> tuple[str, ...]:
    """Return next-state target names matching the configured sensors."""
    return tuple(f"next_{column}" for column in layout.sensor_columns)


def load_water_heater_training_data(path: Path) -> pd.DataFrame:
    """Load a simulation training table with a validated time index."""
    if not path.is_file():
        raise FileNotFoundError(f"Water-heater training data not found: {path}")
    data = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    data.index = pd.DatetimeIndex(data.index, name="timestamp")
    return data


def prepare_water_heater_training_dataset(
    data: pd.DataFrame,
    window: WaterHeaterTrainingWindow,
    sensor_layout: WaterHeaterSensorLayout | None = None,
) -> WaterHeaterTrainingDataset:
    """Select a leakage-free training window and separate model I/O columns.

    Features contain measured temperatures, delivered heat, and forecastable
    disturbances. A separate wrapper maps the physical setpoint command to the
    delivered-heat input used by this model.
    """
    layout = sensor_layout or WaterHeaterSensorLayout()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Water-heater training data must use a DatetimeIndex")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Training timestamps must be unique and increasing")

    feature_columns = (
        *layout.sensor_columns,
        DELIVERED_HEAT_COLUMN,
        *DISTURBANCE_COLUMNS,
    )

    target_columns = next_sensor_columns(layout)
    required_columns = (*feature_columns, *target_columns, *DIAGNOSTIC_COLUMNS)
    missing = [column for column in required_columns if column not in data]
    if missing:
        raise ValueError(f"Water-heater training data is missing columns: {missing}")

    selected = data.loc[
        (data.index >= window.start) & (data.index < window.end),
        required_columns,
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No water-heater samples fall in [{window.start}, {window.end})"
        )

    numeric_columns = [
        *feature_columns,
        *target_columns,
        *(
            column
            for column in DIAGNOSTIC_COLUMNS
            if column != "water_heater_mode"
        ),
    ]
    numeric = selected.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Water-heater training data contains nonfinite values")

    return WaterHeaterTrainingDataset(
        features=selected.loc[:, feature_columns],
        targets=selected.loc[:, target_columns],
        diagnostics=selected.loc[:, DIAGNOSTIC_COLUMNS],
        window=window,
    )


def prepare_water_heater_dataset_splits(
    data: pd.DataFrame,
    windows: WaterHeaterSplitWindows,
    sensor_layout: WaterHeaterSensorLayout | None = None,
) -> WaterHeaterDatasetSplits:
    """Prepare three chronological partitions without data leakage."""
    layout = sensor_layout or WaterHeaterSensorLayout()
    return WaterHeaterDatasetSplits(
        training=prepare_water_heater_training_dataset(
            data,
            windows.training,
            layout,
        ),
        validation=prepare_water_heater_training_dataset(
            data,
            windows.validation,
            layout,
        ),
        test=prepare_water_heater_training_dataset(
            data,
            windows.test,
            layout,
        ),
    )


def aggregate_water_heater_dataset(
    dataset: WaterHeaterTrainingDataset,
    config: WaterHeaterStateSpaceConfig,
    layout: WaterHeaterSensorLayout,
) -> pd.DataFrame:
    """Aggregate validated one-minute records into model intervals."""
    source_step = pd.Timedelta(config.source_timestep)
    model_step = pd.Timedelta(config.model_timestep)

    if model_step % source_step:
        raise ValueError(
            "model_timestep must be an integer multiple of source_timestep"
        )
    if (dataset.window.end - dataset.window.start) % model_step:
        raise ValueError("Dataset window must contain complete model intervals")

    data = pd.concat(
        [dataset.features, dataset.targets, dataset.diagnostics],
        axis=1,
    )
    if data.columns.has_duplicates:
        raise ValueError("Dataset tables contain duplicate column names")

    expected_index = pd.date_range(
        dataset.window.start,
        dataset.window.end,
        freq=source_step,
        inclusive="left",
        name=data.index.name,
    )
    if not data.index.equals(expected_index):
        raise ValueError(
            "Dataset must contain consecutive source-timestep samples"
        )

    next_sensors = next_sensor_columns(layout)
    aggregation = {
        **{column: "first" for column in layout.sensor_columns},
        DELIVERED_HEAT_COLUMN: "mean",
        **{column: "mean" for column in config.disturbance_columns},
        **{column: "last" for column in next_sensors},
        POWER_COLUMN: "mean",
    }

    missing = set(aggregation) - set(data.columns)
    if missing:
        raise ValueError(f"Missing aggregation columns: {sorted(missing)}")

    return (
        data.loc[:, list(aggregation)]
        .resample(
            model_step,
            origin=dataset.window.start,
            closed="left",
            label="right",
        )
        .agg(aggregation)
    )


def to_state_space_frame(
    aggregated: pd.DataFrame,
    config: WaterHeaterStateSpaceConfig,
    layout: WaterHeaterSensorLayout,
) -> pd.DataFrame:
    """Map domain-named interval data to the generic predictor schema."""
    frame = pd.DataFrame(index=aggregated.index)

    frame["Input_1"] = aggregated[DELIVERED_HEAT_COLUMN]
    for index, column in enumerate(next_sensor_columns(layout), start=1):
        frame[f"Output_{index}"] = aggregated[column]

    for index, column in enumerate(config.disturbance_columns, start=1):
        frame[f"Disturbance_{index}"] = aggregated[column]

    return frame


def evaluate_water_heater_state_space_model(
    model: dict[str, Any],
    frame: pd.DataFrame,
    output_names: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate one-step and rolling-horizon temperature predictions."""
    nu = int(model["nu"])
    ny = int(model["ny"])
    nw = int(model["nw"])
    memory = int(model["m"])
    horizon = int(model["horizon_steps"])
    if len(output_names) != ny:
        raise ValueError("Output names do not match the fitted model")
    if len(frame) < memory + horizon:
        raise ValueError(
            "Test data is too short for the selected memory and prediction horizon"
        )

    inputs = frame[[f"Input_{i + 1}" for i in range(nu)]].to_numpy(float).T
    outputs = frame[[f"Output_{i + 1}" for i in range(ny)]].to_numpy(float).T
    disturbances = frame[
        [f"Disturbance_{i + 1}" for i in range(nw)]
    ].to_numpy(float).T
    history_signals = np.vstack([inputs, disturbances, outputs])

    one_step_errors = []
    for step in range(memory, len(frame)):
        history = history_signals[:, step - memory : step].reshape(
            -1, order="F"
        )
        prediction = (
            model["P1"] @ history
            + model["F11"] @ inputs[:, step]
            + model["W11"] @ disturbances[:, step]
        )
        one_step_errors.append(outputs[:, step] - prediction)
    one_step_errors = np.column_stack(one_step_errors)

    rolling_squared_error = np.zeros(ny)
    rolling_value_count = 0
    rolling_origin_count = 0
    for origin in range(memory, len(frame) - horizon + 1):
        history = history_signals[:, origin - memory : origin].reshape(
            -1, order="F"
        )
        future_inputs = inputs[:, origin : origin + horizon].reshape(
            -1, order="F"
        )
        future_disturbances = disturbances[
            :, origin : origin + horizon
        ].reshape(-1, order="F")
        prediction = (
            model["P"] @ history
            + model["F"] @ future_inputs
            + model["W"] @ future_disturbances
        ).reshape(ny, horizon, order="F")
        error = outputs[:, origin : origin + horizon] - prediction
        rolling_squared_error += np.sum(error**2, axis=1)
        rolling_value_count += horizon
        rolling_origin_count += 1

    one_step_rmse = np.sqrt(np.mean(one_step_errors**2, axis=1))
    rolling_rmse = np.sqrt(rolling_squared_error / rolling_value_count)
    return {
        "one_step_rmse_c": {
            name: float(value) for name, value in zip(output_names, one_step_rmse)
        },
        "one_step_combined_rmse_c": float(
            np.sqrt(np.mean(one_step_errors**2))
        ),
        "rolling_horizon_rmse_c": {
            name: float(value) for name, value in zip(output_names, rolling_rmse)
        },
        "rolling_horizon_combined_rmse_c": float(
            np.sqrt(np.sum(rolling_squared_error) / (rolling_value_count * ny))
        ),
        "one_step_prediction_count": int(one_step_errors.shape[1]),
        "rolling_horizon_origin_count": rolling_origin_count,
    }


def _regularization_summary(
    model: dict[str, Any],
    disturbance_columns: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Return JSON-safe lambda diagnostics without candidate matrices."""
    summary = {}
    for memory, diagnostics in model["regularization_diagnostics"].items():
        rows = []
        for diagnostic in diagnostics:
            active = diagnostic.get("active_disturbances", [])
            row = {
                "lambda": float(diagnostic["lambda"]),
                "active_disturbances": [
                    disturbance_columns[index] for index in active
                ],
                "status": diagnostic["status"],
            }
            for key in ("AICc", "BIC", "validation_rmse"):
                if key in diagnostic:
                    row[key] = float(diagnostic[key])
            rows.append(row)
        summary[str(memory)] = rows
    return summary


def _save_water_heater_model(
    model: dict[str, Any],
    model_output: Path,
    config: WaterHeaterStateSpaceConfig,
    layout: WaterHeaterSensorLayout,
    splits: WaterHeaterDatasetSplits,
    test_metrics: dict[str, Any],
) -> None:
    """Save predictor matrices and human-readable training metadata."""
    model_output.mkdir(parents=True, exist_ok=True)
    array_names = (
        "A",
        "B",
        "E",
        "P",
        "F",
        "W",
        "P1",
        "F11",
        "W11",
        "eSig",
    )
    np.savez_compressed(
        model_output / "model_arrays.npz",
        **{name: model[name] for name in array_names},
    )

    selected_indices = [int(i) for i in model["selected_disturbance_indices"]]
    uses_group_lasso = config.feature_selection_method == "group_lasso"
    metadata = {
        "schema_version": 2,
        "model_type": (
            "regularized_exogenous_state_space"
            if uses_group_lasso
            else "all_feature_exogenous_state_space"
        ),
        "columns": {
            "input": [DELIVERED_HEAT_COLUMN],
            "outputs": list(layout.sensor_columns),
            "disturbances": list(config.disturbance_columns),
        },
        "sensor_layout": layout.metadata(),
        "configuration": {
            "source_timestep_seconds": config.source_timestep.total_seconds(),
            "model_timestep_seconds": config.model_timestep.total_seconds(),
            "prediction_horizon_seconds": config.prediction_horizon.total_seconds(),
            "horizon_steps": config.horizon_steps,
            "memory_min": config.memory_min,
            "memory_max": config.memory_max,
            "feature_selection_method": config.feature_selection_method,
            "scaling": config.scaling,
            "lambda_grid_mode": "auto" if uses_group_lasso else None,
            "auto_lambda_grid_points": (
                config.auto_lambda_grid_points if uses_group_lasso else None
            ),
            "relative_aicc_threshold": config.relative_aicc_threshold,
            "group_lasso_solver": "cvxpy" if uses_group_lasso else None,
            "cvxpy_solver": model["cvxpy_solver"] if uses_group_lasso else None,
            "lambda_selection_metric": (
                "validation_rolling_horizon_rmse" if uses_group_lasso else None
            ),
            "memory_selection_metric": "relative_AICc",
        },
        "selection": {
            "memory": int(model["m"]),
            "standard_aicc_memory": int(model["standard_aicc_memory"]),
            "relative_aicc_memory": int(model["relative_aicc_memory"]),
            "lambda": (
                float(model["selected_lambda"]) if uses_group_lasso else None
            ),
            "lambda_grid": [float(value) for value in model["lambda_grid"]],
            "selected_disturbances": [
                config.disturbance_columns[index] for index in selected_indices
            ],
            "validation_rmse_c": (
                float(model["selected_lambda_validation_rmse"])
                if uses_group_lasso
                else None
            ),
        },
        "windows": {
            name: {
                "start": getattr(splits, name).window.start.isoformat(),
                "end": getattr(splits, name).window.end.isoformat(),
            }
            for name in ("training", "validation", "test")
        },
        "test_metrics": test_metrics,
        "diagnostic_outputs": {
            "figure_directory": "figures",
            "figures": [
                Path(path).name for path in model.get("figure_files", ())
            ],
            "csv_files": [
                "predictions.csv",
                "residuals.csv",
                "rmse_by_horizon.csv",
                "memory_selection.csv",
                "lambda_selection.csv",
                "selected_disturbances.csv",
                *(["heat_reconstruction.csv"] if (model_output / "figures" / "heat_reconstruction.csv").is_file() else []),
            ],
        },
        "regularization_summary": (
            _regularization_summary(model, config.disturbance_columns)
            if uses_group_lasso
            else {}
        ),
    }
    with (model_output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def train_water_heater_model(
    splits: WaterHeaterDatasetSplits,
    model_output: Path,
    config: WaterHeaterStateSpaceConfig | None = None,
    sensor_layout: WaterHeaterSensorLayout | None = None,
    rc_model_path: Path | None = None,
) -> dict[str, Any]:
    """Fit, evaluate, and save the regularized water-heater model."""
    config = config or WaterHeaterStateSpaceConfig()
    layout = sensor_layout or WaterHeaterSensorLayout()
    aggregated = {
        name: aggregate_water_heater_dataset(getattr(splits, name), config, layout)
        for name in ("training", "validation", "test")
    }
    frames = {
        name: to_state_space_frame(data, config, layout)
        for name, data in aggregated.items()
    }

    common_fit_arguments = {
        "df_train": frames["training"],
        "nu": 1,
        "ny": len(layout.sensor_columns),
        "nw": len(config.disturbance_columns),
        "horizon_steps": config.horizon_steps,
        "m_min": config.memory_min,
        "m_max": config.memory_max,
        "relative_aicc_pct_thresh": config.relative_aicc_threshold,
    }
    if config.feature_selection_method == "group_lasso":
        model = train_state_space_predictor_group_lasso(
            **common_fit_arguments,
            df_val=frames["validation"],
            lambda_grid=None,
            auto_lambda_grid_points=config.auto_lambda_grid_points,
            group_lasso_solver="cvxpy",
            cvxpy_solver=config.cvxpy_solver,
            group_lasso_scaling=config.scaling,
            lambda_selection_metric="validation_rolling_horizon_rmse",
            memory_selection_method="relative_aic",
        )
        model["feature_selection_method"] = "group_lasso"
    else:
        model = train_state_space_predictor_all_features(
            **common_fit_arguments,
        )

    test_metrics = evaluate_water_heater_state_space_model(
        model, frames["test"], layout.sensor_columns
    )
    model["test_metrics"] = test_metrics
    model["selected_disturbance_columns"] = [
        config.disturbance_columns[int(index)]
        for index in model["selected_disturbance_indices"]
    ]
    model_output = Path(model_output)
    rc_metadata = Path(rc_model_path) if rc_model_path is not None else model_output.parent / "water_heater_rc_model" / "metadata.json"
    if rc_model_path is not None and not rc_metadata.is_file():
        raise FileNotFoundError(f"RC model metadata not found: {rc_metadata}")
    heat_input = None
    if rc_metadata.is_file():
        rc_model = load_rc_model(rc_metadata)
        with rc_metadata.open(encoding="utf-8") as stream:
            rc_fitting_method = json.load(stream).get("fitting_method", "")
        if rc_fitting_method.startswith("measured_power_euler"):
            from .water_heater_rc_power import reconstruct_heat_input_euler
            heat_reconstructor = reconstruct_heat_input_euler
        else:
            heat_reconstructor = reconstruct_heat_input
        heat_input = {}
        for name in ("training", "validation", "test"):
            dataset = getattr(splits, name)
            raw = pd.concat([
                dataset.features, dataset.targets,
                dataset.diagnostics[[POWER_COLUMN]],
            ], axis=1)
            reconstructed = heat_reconstructor(raw, rc_model)
            heat_input[name] = reconstructed.resample(
                config.model_timestep, origin=dataset.window.start,
                closed="left", label="right",
            ).mean().reindex(frames[name].index)
        figures = model_output / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        pd.concat(heat_input, names=["split", "timestamp"]).to_csv(figures / "heat_reconstruction.csv")
    model["figure_files"] = create_state_space_diagnostic_outputs(
        model=model,
        frames=frames,
        model_output=model_output,
        output_names=layout.sensor_columns,
        disturbance_names=config.disturbance_columns,
        model_label="Water-heater temperature model",
        heat_input=heat_input,
    )
    _save_water_heater_model(
        model,
        model_output,
        config,
        layout,
        splits,
        test_metrics,
    )
    return model

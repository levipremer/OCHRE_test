"""Training integration for the HVAC building-envelope model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hvac import (
    DAY_INDICATOR_COLUMN,
    NIGHT_INDICATOR_COLUMN,
    DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN,
    DIRECT_NORMAL_IRRADIANCE_COLUMN,
    GLOBAL_SOLAR_COLUMN,
    HVAC_COIL_ER_HEAT_COLUMN,
    HVAC_HEAT_COLUMN,
    HVAC_HEATING_POWER_COLUMN,
    HVAC_ER_HEAT_COLUMN,
    HVAC_FAN_POWER_COLUMN,
    HVAC_HP_COIL_HEAT_COLUMN,
    HVAC_HP_MAIN_POWER_COLUMN,
    HVAC_HP_PROXY_COLUMN,
    HVAC_POWER_COLUMN,
    HVAC_PROXY_HEAT_COLUMN,
    INDOOR_TEMPERATURE_COLUMN,
    NEXT_INDOOR_TEMPERATURE_COLUMN,
    OUTDOOR_RELATIVE_HUMIDITY_COLUMN,
    OUTDOOR_TEMPERATURE_COLUMN,
    SOLAR_AZIMUTH_COLUMN,
    SOLAR_ZENITH_COLUMN,
)
from .hvac_2r2c import HVACHeatInput
from .model_diagnostics import create_state_space_selection_outputs
from .building_envelope_evaluation import save_rollout_evaluation
from .state_space_predictor_regularized import (
    train_state_space_predictor_group_lasso,
)


HVAC_FEATURE_COLUMNS = (
    INDOOR_TEMPERATURE_COLUMN,
    HVAC_HEAT_COLUMN,
    OUTDOOR_TEMPERATURE_COLUMN,
    OUTDOOR_RELATIVE_HUMIDITY_COLUMN,
    GLOBAL_SOLAR_COLUMN,
    DAY_INDICATOR_COLUMN,
)
HVAC_TARGET_COLUMNS = (NEXT_INDOOR_TEMPERATURE_COLUMN,)
STATE_SPACE_INPUT_COLUMN = "state_space_thermal_heat_w"
HVAC_DIAGNOSTIC_COLUMNS = (
    HVAC_POWER_COLUMN,
    HVAC_HEATING_POWER_COLUMN,
    "actual_internal_sensible_gain_w",
    "occupancy_persons",
    "heating_setpoint_c",
    "cooling_setpoint_c",
    "heating_mode",
    "cooling_mode",
    DIRECT_NORMAL_IRRADIANCE_COLUMN,
    DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN,
    SOLAR_ZENITH_COLUMN,
    SOLAR_AZIMUTH_COLUMN,
    "ground_temperature_c",
    "wind_speed_m_per_s",
    "ambient_humidity_ratio",
    "sky_temperature_c",
)
HVAC_OPTIONAL_HEAT_DIAGNOSTIC_COLUMNS = (
    HVAC_HEAT_COLUMN,
    HVAC_COIL_ER_HEAT_COLUMN,
    HVAC_PROXY_HEAT_COLUMN,
    HVAC_HP_PROXY_COLUMN,
    HVAC_HP_MAIN_POWER_COLUMN,
    HVAC_FAN_POWER_COLUMN,
    HVAC_HP_COIL_HEAT_COLUMN,
    HVAC_ER_HEAT_COLUMN,
)


def projected_dni_column(surface_azimuth_degrees: int) -> str:
    """Return the stable feature name for one vertical surface orientation."""
    return f"projected_dni_azimuth_{surface_azimuth_degrees:03d}_w_per_m2"


@dataclass(frozen=True)
class HVACTrainingWindow:
    """Half-open HVAC training interval, separate from deployment dates."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if end <= start:
            raise ValueError("HVAC training end must be after its start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class HVACTrainingDataset:
    """Validated HVAC feature, target, and diagnostic tables."""

    features: pd.DataFrame
    targets: pd.DataFrame
    diagnostics: pd.DataFrame
    window: HVACTrainingWindow

    @property
    def sample_count(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class HVACSplitWindows:
    """Chronological, non-overlapping train/validation/test windows."""

    training: HVACTrainingWindow
    validation: HVACTrainingWindow
    test: HVACTrainingWindow

    def __post_init__(self) -> None:
        if self.training.end > self.validation.start:
            raise ValueError("HVAC training and validation windows overlap")
        if self.validation.end > self.test.start:
            raise ValueError("HVAC validation and test windows overlap")


@dataclass(frozen=True)
class HVACDatasetSplits:
    """Prepared datasets for fitting, tuning, and final evaluation."""

    training: HVACTrainingDataset
    validation: HVACTrainingDataset
    test: HVACTrainingDataset


@dataclass(frozen=True)
class HVACStateSpaceConfig:
    """Settings for the regularized building-envelope model."""

    source_timestep: pd.Timedelta = pd.Timedelta(minutes=1)
    model_timestep: pd.Timedelta = pd.Timedelta(minutes=15)
    prediction_horizon: pd.Timedelta = pd.Timedelta(hours=12)

    solar_azimuth_step_degrees: int = 20
    include_ghi: bool = True
    control_input: HVACHeatInput = field(
        default_factory=lambda: HVACHeatInput(
            heat_column=HVAC_COIL_ER_HEAT_COLUMN,
            power_column=HVAC_HEATING_POWER_COLUMN
        )
    )

    memory_min: int = 1
    memory_max: int = 12
    scaling: str = "standard_deviation"
    cvxpy_solver: str = "gurobi"
    auto_lambda_grid_points: int = 30
    relative_aicc_threshold: float = 0.05

    def __post_init__(self) -> None:
        source_timestep = pd.Timedelta(self.source_timestep)
        model_timestep = pd.Timedelta(self.model_timestep)
        prediction_horizon = pd.Timedelta(self.prediction_horizon)

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
        step = self.solar_azimuth_step_degrees
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("solar_azimuth_step_degrees must be a positive integer")
        if 360 % step:
            raise ValueError("solar_azimuth_step_degrees must divide 360 evenly")
        if not isinstance(self.include_ghi, bool):
            raise ValueError("include_ghi must be a boolean")
        if not isinstance(self.control_input, HVACHeatInput):
            raise TypeError("control_input must be an HVACHeatInput adapter")
        self.control_input.required_columns()
        for name, value in (
            ("memory_min", self.memory_min),
            ("memory_max", self.memory_max),
            ("auto_lambda_grid_points", self.auto_lambda_grid_points),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_min > self.memory_max:
            raise ValueError("memory_min cannot exceed memory_max")
        if self.scaling not in {"none", "standard_deviation"}:
            raise ValueError("scaling must be 'none' or 'standard_deviation'")
        if self.cvxpy_solver not in {"gurobi", "clarabel", "scs"}:
            raise ValueError("cvxpy_solver must be 'gurobi', 'clarabel', or 'scs'")
        if not 0 <= self.relative_aicc_threshold <= 1:
            raise ValueError("relative_aicc_threshold must be between zero and one")

        object.__setattr__(self, "source_timestep", source_timestep)
        object.__setattr__(self, "model_timestep", model_timestep)
        object.__setattr__(self, "prediction_horizon", prediction_horizon)

    @property
    def horizon_steps(self) -> int:
        return self.prediction_horizon // self.model_timestep

    @property
    def solar_azimuths_degrees(self) -> tuple[int, ...]:
        return tuple(range(0, 360, self.solar_azimuth_step_degrees))

    @property
    def projected_dni_columns(self) -> tuple[str, ...]:
        return tuple(
            projected_dni_column(azimuth)
            for azimuth in self.solar_azimuths_degrees
        )

    @property
    def disturbance_columns(self) -> tuple[str, ...]:
        columns = [
            OUTDOOR_RELATIVE_HUMIDITY_COLUMN,
            DAY_INDICATOR_COLUMN,
            NIGHT_INDICATOR_COLUMN,
            DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN,
            *self.projected_dni_columns,
        ]
        if self.include_ghi:
            columns.append(GLOBAL_SOLAR_COLUMN)
        columns.append(OUTDOOR_TEMPERATURE_COLUMN)
        return tuple(columns)

    @property
    def current_only_disturbance_indices(self) -> tuple[int, ...]:
        """Day/night gain intercepts have no lagged coefficients."""
        return tuple(
            self.disturbance_columns.index(column)
            for column in (DAY_INDICATOR_COLUMN, NIGHT_INDICATOR_COLUMN)
        )


def load_hvac_training_data(path: Path) -> pd.DataFrame:
    """Load an HVAC simulation training table with a time index."""
    if not path.is_file():
        raise FileNotFoundError(f"HVAC training data not found: {path}")
    data = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    data.index = pd.DatetimeIndex(data.index, name="timestamp")
    return data


def prepare_hvac_training_dataset(
    data: pd.DataFrame,
    window: HVACTrainingWindow,
    config: HVACStateSpaceConfig | None = None,
) -> HVACTrainingDataset:
    """Apply a chronological window and expose the common 2R2C/ARX I/O."""
    config = config or HVACStateSpaceConfig()
    control_input = config.control_input
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("HVAC training data must use a DatetimeIndex")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("HVAC training timestamps must be unique and increasing")

    feature_columns = tuple(
        column for column in HVAC_FEATURE_COLUMNS if column != HVAC_HEAT_COLUMN
    )
    if control_input.representation == "thermal_heat":
        feature_columns = (
            feature_columns[0], control_input.heat_column, *feature_columns[1:]
        )
    diagnostic_columns = tuple(
        column for column in dict.fromkeys(
            (
                *HVAC_DIAGNOSTIC_COLUMNS,
                *control_input.required_columns(),
                *(
                    column for column in HVAC_OPTIONAL_HEAT_DIAGNOSTIC_COLUMNS
                    if column in data
                ),
            )
        ) if column not in feature_columns
    )
    required_columns = tuple(dict.fromkeys(
        (*feature_columns, *HVAC_TARGET_COLUMNS, *diagnostic_columns)
    ))
    missing = [column for column in required_columns if column not in data]
    if missing:
        raise ValueError(f"HVAC training data is missing columns: {missing}")

    selected = data.loc[
        (data.index >= window.start) & (data.index < window.end),
        required_columns,
    ].copy()
    if selected.empty:
        raise ValueError(f"No HVAC samples fall in [{window.start}, {window.end})")

    numeric_columns = [
        *feature_columns,
        *HVAC_TARGET_COLUMNS,
        *(
            column for column in diagnostic_columns
            if column not in {"heating_mode", "cooling_mode", control_input.mode_column}
        ),
    ]
    numeric = selected.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("HVAC training data contains nonfinite values")

    return HVACTrainingDataset(
        features=selected.loc[:, feature_columns],
        targets=selected.loc[:, HVAC_TARGET_COLUMNS],
        diagnostics=selected.loc[:, diagnostic_columns],
        window=window,
    )


def prepare_hvac_dataset_splits(
    data: pd.DataFrame,
    windows: HVACSplitWindows,
    config: HVACStateSpaceConfig | None = None,
) -> HVACDatasetSplits:
    """Prepare three chronological partitions without data leakage."""
    return HVACDatasetSplits(
        training=prepare_hvac_training_dataset(data, windows.training, config),
        validation=prepare_hvac_training_dataset(data, windows.validation, config),
        test=prepare_hvac_training_dataset(data, windows.test, config),
    )


def add_projected_dni_candidates(
    data: pd.DataFrame,
    config: HVACStateSpaceConfig,
) -> pd.DataFrame:
    """Add beam irradiance projected onto candidate vertical surfaces."""
    required = (
        DIRECT_NORMAL_IRRADIANCE_COLUMN,
        SOLAR_ZENITH_COLUMN,
        SOLAR_AZIMUTH_COLUMN,
    )
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"HVAC solar data is missing columns: {missing}")

    result = data.copy()
    dni = result[DIRECT_NORMAL_IRRADIANCE_COLUMN].to_numpy(float).clip(min=0)
    zenith_degrees = result[SOLAR_ZENITH_COLUMN].to_numpy(float)
    solar_azimuth = np.deg2rad(
        result[SOLAR_AZIMUTH_COLUMN].to_numpy(float)
    )
    sin_zenith = np.sin(np.deg2rad(zenith_degrees))
    sun_is_up = zenith_degrees < 90

    for surface_azimuth, column in zip(
        config.solar_azimuths_degrees,
        config.projected_dni_columns,
    ):
        incidence_cosine = sin_zenith * np.cos(
            solar_azimuth - np.deg2rad(surface_azimuth)
        )
        result[column] = dni * np.maximum(incidence_cosine, 0) * sun_is_up
    return result


def _uses_recorded_sensible_heat(control_input: HVACHeatInput) -> bool:
    """Keep the current W input path unchanged for the default adapter."""
    return (
        control_input.representation == "thermal_heat"
        and control_input.heat_column == HVAC_HEAT_COLUMN
        and control_input.heat_multiplier_to_kw == 0.001
    )


def _state_space_heat_input_w(
    data: pd.DataFrame, control_input: HVACHeatInput
) -> pd.Series:
    """Convert a selected source representation to signed thermal input W."""
    if control_input.representation == "power_mode":
        if control_input.power_column == HVAC_POWER_COLUMN:
            cooling_mode = data["cooling_mode"].astype(str).str.strip().str.casefold()
            if cooling_mode.ne("off").any():
                raise ValueError(
                    "Total HVAC electric power cannot be mapped through heating "
                    "mode while cooling is active; use heating-only power"
                )
    return control_input.thermal_heat_kw(data) * 1000.0


def aggregate_hvac_dataset(
    dataset: HVACTrainingDataset,
    config: HVACStateSpaceConfig,
) -> pd.DataFrame:
    """Aggregate validated one-minute records into model intervals."""
    if (dataset.window.end - dataset.window.start) % config.model_timestep:
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
        freq=config.source_timestep,
        inclusive="left",
        name=data.index.name,
    )
    if not data.index.equals(expected_index):
        raise ValueError(
            "Dataset must contain consecutive source-timestep samples"
        )

    data = add_projected_dni_candidates(data, config)
    # The old training CSV records only the day regime. Derive its complement
    # here so the same historical dates can be retrained without regeneration.
    data[NIGHT_INDICATOR_COLUMN] = 1 - data[DAY_INDICATOR_COLUMN]
    input_column = HVAC_HEAT_COLUMN
    if not _uses_recorded_sensible_heat(config.control_input):
        input_column = STATE_SPACE_INPUT_COLUMN
        data[input_column] = _state_space_heat_input_w(data, config.control_input)
    aggregation = {
        INDOOR_TEMPERATURE_COLUMN: "first",
        input_column: "mean",
        **{
            column: "first" if column in {DAY_INDICATOR_COLUMN, NIGHT_INDICATOR_COLUMN} else "mean"
            for column in config.disturbance_columns
        },
        NEXT_INDOOR_TEMPERATURE_COLUMN: "last",
        HVAC_POWER_COLUMN: "mean",
    }
    aggregated = (
        data.loc[:, list(aggregation)]
        .resample(
            config.model_timestep,
            origin=dataset.window.start,
            closed="left",
            label="right",
        )
        .agg(aggregation)
    )
    # The target at a label equal to window.end is the first indoor state of
    # the next split. Keep that boundary state entirely out of this split.
    return aggregated.loc[aggregated.index < dataset.window.end]


def to_hvac_state_space_frame(
    aggregated: pd.DataFrame,
    config: HVACStateSpaceConfig,
) -> pd.DataFrame:
    """Map HVAC interval data to the generic predictor schema."""
    frame = pd.DataFrame(index=aggregated.index)
    input_column = (
        HVAC_HEAT_COLUMN if _uses_recorded_sensible_heat(config.control_input)
        else STATE_SPACE_INPUT_COLUMN
    )
    frame["Input_1"] = aggregated[input_column]
    frame["Output_1"] = aggregated[NEXT_INDOOR_TEMPERATURE_COLUMN]
    for index, column in enumerate(config.disturbance_columns, start=1):
        frame[f"Disturbance_{index}"] = aggregated[column]
    return frame


def evaluate_hvac_state_space_model(
    model: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate one-step and rolling-horizon indoor-temperature predictions."""
    memory = int(model["m"])
    horizon = int(model["horizon_steps"])
    disturbance_count = int(model["nw"])
    if len(frame) < memory + horizon:
        raise ValueError(
            "Test data is too short for the selected memory and prediction horizon"
        )

    inputs = frame[["Input_1"]].to_numpy(float).T
    outputs = frame[["Output_1"]].to_numpy(float).T
    disturbances = frame[
        [f"Disturbance_{i + 1}" for i in range(disturbance_count)]
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
        one_step_errors.append(float(outputs[0, step] - prediction[0]))

    rolling_squared_error = 0.0
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
        )
        error = outputs[0, origin : origin + horizon] - prediction
        rolling_squared_error += float(np.sum(error**2))
        rolling_value_count += horizon
        rolling_origin_count += 1

    one_step_errors = np.asarray(one_step_errors)
    return {
        "one_step_rmse_c": float(np.sqrt(np.mean(one_step_errors**2))),
        "rolling_horizon_rmse_c": float(
            np.sqrt(rolling_squared_error / rolling_value_count)
        ),
        "one_step_prediction_count": int(one_step_errors.size),
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
            row = {
                "lambda": float(diagnostic["lambda"]),
                "active_disturbances": [
                    disturbance_columns[index]
                    for index in diagnostic.get("active_disturbances", [])
                ],
                "status": diagnostic["status"],
            }
            for key in ("AICc", "BIC", "validation_rmse"):
                if key in diagnostic:
                    row[key] = float(diagnostic[key])
            rows.append(row)
        summary[str(memory)] = rows
    return summary


def _save_hvac_model(
    model: dict[str, Any],
    model_output: Path,
    config: HVACStateSpaceConfig,
    splits: HVACDatasetSplits,
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
    selected_disturbances = [
        config.disturbance_columns[index] for index in selected_indices
    ]
    control_input = config.control_input
    input_sources = (
        [control_input.heat_column]
        if control_input.representation == "thermal_heat"
        else [control_input.power_column, control_input.mode_column]
    )
    metadata = {
        "schema_version": 1,
        "model_type": "regularized_exogenous_state_space",
        "columns": {
            "input": input_sources,
            "model_input": ["Input_1"],
            "model_input_units": (
                "W of HP coil heat plus ER heat"
                if control_input.representation == "thermal_heat"
                and control_input.heat_column == HVAC_COIL_ER_HEAT_COLUMN
                else "W of signed thermal heat"
            ),
            "outputs": [INDOOR_TEMPERATURE_COLUMN],
            "disturbances": list(config.disturbance_columns),
        },
        "configuration": {
            "control_input": {
                "representation": control_input.representation,
                "heat_column": control_input.heat_column,
                "heat_multiplier_to_kw": control_input.heat_multiplier_to_kw,
                "power_column": control_input.power_column,
                "mode_column": control_input.mode_column,
                "mode_heat_factors": dict(control_input.mode_heat_factors),
                "total_power_cooling_guard": (
                    control_input.representation == "power_mode"
                    and control_input.power_column == HVAC_POWER_COLUMN
                ),
            },
            "source_timestep_seconds": config.source_timestep.total_seconds(),
            "model_timestep_seconds": config.model_timestep.total_seconds(),
            "prediction_horizon_seconds": config.prediction_horizon.total_seconds(),
            "horizon_steps": config.horizon_steps,
            "solar_azimuth_step_degrees": config.solar_azimuth_step_degrees,
            "surface_tilt_degrees": 90,
            "include_ghi": config.include_ghi,
            "current_only_disturbances": [
                config.disturbance_columns[index]
                for index in config.current_only_disturbance_indices
            ],
            "memory_min": config.memory_min,
            "memory_max": config.memory_max,
            "scaling": config.scaling,
            "lambda_grid_mode": "auto",
            "auto_lambda_grid_points": config.auto_lambda_grid_points,
            "relative_aicc_threshold": config.relative_aicc_threshold,
            "group_lasso_solver": "cvxpy",
            "cvxpy_solver": model["cvxpy_solver"],
            "allow_rank_deficient_refit": model["allow_rank_deficient_refit"],
            "lambda_selection_metric": "validation_rolling_horizon_rmse",
            "memory_selection_metric": "relative_AICc",
        },
        "selection": {
            "memory": int(model["m"]),
            "standard_aicc_memory": int(model["standard_aicc_memory"]),
            "relative_aicc_memory": int(model["relative_aicc_memory"]),
            "lambda": float(model["selected_lambda"]),
            "lambda_grid": [float(value) for value in model["lambda_grid"]],
            "selected_disturbances": selected_disturbances,
            "selected_solar_azimuths_degrees": [
                azimuth
                for azimuth, column in zip(
                    config.solar_azimuths_degrees,
                    config.projected_dni_columns,
                )
                if column in selected_disturbances
            ],
            "validation_rmse_c": float(model["selected_lambda_validation_rmse"]),
        },
        "windows": {
            name: {
                "start": getattr(splits, name).window.start.isoformat(),
                "end": getattr(splits, name).window.end.isoformat(),
            }
            for name in ("training", "validation", "test")
        },
        "test_metrics": test_metrics,
        "rollout_metrics": model.get("rollout_metrics", {}),
        "diagnostic_outputs": {
            "figure_directory": "figures",
            "figures": [
                Path(path).name for path in model.get("figure_files", ())
            ],
            "csv_files": [
                "memory_selection.csv",
                "lambda_selection.csv",
                "selected_disturbances.csv",
                "rollout_predictions.csv",
                "rollout_metrics.csv",
            ],
            "axis_limits_file": "axis_limits.json",
            "indoor_temperature_axis_limits_c": model.get("temperature_limits_c"),
        },
        "regularization_summary": _regularization_summary(
            model, config.disturbance_columns
        ),
    }
    with (model_output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def train_hvac_model(
    splits: HVACDatasetSplits,
    model_output: Path,
    config: HVACStateSpaceConfig | None = None,
    temperature_limits_c: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit, evaluate, and save the regularized building-envelope model."""
    config = config or HVACStateSpaceConfig()
    aggregated = {
        name: aggregate_hvac_dataset(getattr(splits, name), config)
        for name in ("training", "validation", "test")
    }
    frames = {
        name: to_hvac_state_space_frame(data, config)
        for name, data in aggregated.items()
    }

    model = train_state_space_predictor_group_lasso(
        df_train=frames["training"],
        df_val=frames["validation"],
        nu=1,
        ny=1,
        nw=len(config.disturbance_columns),
        horizon_steps=config.horizon_steps,
        m_min=config.memory_min,
        m_max=config.memory_max,
        lambda_grid=None,
        auto_lambda_grid_points=config.auto_lambda_grid_points,
        group_lasso_solver="cvxpy",
        cvxpy_solver=config.cvxpy_solver,
        group_lasso_scaling=config.scaling,
        current_only_disturbance_indices=config.current_only_disturbance_indices,
        lambda_selection_metric="validation_rolling_horizon_rmse",
        memory_selection_method="relative_aic",
        relative_aicc_pct_thresh=config.relative_aicc_threshold,
        allow_rank_deficient_refit=True,
    )

    test_metrics = evaluate_hvac_state_space_model(model, frames["test"])
    model["test_metrics"] = test_metrics
    model["selected_disturbance_columns"] = [
        config.disturbance_columns[int(index)]
        for index in model["selected_disturbance_indices"]
    ]
    model_output = Path(model_output)
    model["figure_files"] = create_state_space_selection_outputs(
        model=model,
        model_output=model_output,
        disturbance_names=config.disturbance_columns,
        model_label="HVAC indoor-temperature model",
        solar_azimuth_by_disturbance=dict(
            zip(config.projected_dni_columns, config.solar_azimuths_degrees)
        ),
        model_timestep_minutes=config.model_timestep.total_seconds() / 60,
    )
    rollout_metrics, rollout_figure, plotted_limits = save_rollout_evaluation(
        model,
        frames,
        config.model_timestep,
        model_output / "figures",
        temperature_limits_c=temperature_limits_c,
    )
    model["rollout_metrics"] = rollout_metrics
    model["temperature_limits_c"] = list(plotted_limits)
    model["figure_files"] = (*model["figure_files"], rollout_figure)
    _save_hvac_model(model, model_output, config, splits, test_metrics)
    return model

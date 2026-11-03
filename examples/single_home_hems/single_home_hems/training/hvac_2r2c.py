"""Train and evaluate the reference latent-mass 2R2C HVAC envelope model.

The physical model always receives signed thermal heat in kW. Input conversion
is separate, so a later power-and-mode data source can provide that same heat
input without changing the fitted state equations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hvac import HVAC_HEATING_POWER_COLUMN, HVAC_POWER_COLUMN
from .rc_2r2c_reference import TwoR2CModel
from .rc_2r2c_reference.joint_fitter import day_mask_from_hours


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "building_envelope_training"
    / "2r2c"
)

COLORS = {
    "measured": "#111111",
    "one_step": "#245b9a",
    "8_hour": "#5d9bd3",
    "24_hour": "#dc8125",
    "open_loop": "#328049",
}
SPLIT_FILLS = {
    "training": "#dbe9f4",
    "validation": "#fff2cc",
    "test": "#d9ead3",
}


@dataclass(frozen=True)
class HalfOpenWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if start.tz is not None or end.tz is not None:
            raise ValueError("Window timestamps must be timezone-naive")
        if end <= start:
            raise ValueError("Window end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def overlaps(self, other: HalfOpenWindow) -> bool:
        return self.start < other.end and other.start < self.end

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class HVAC2R2CWindows:
    training: HalfOpenWindow
    validation: HalfOpenWindow
    test: HalfOpenWindow
    controller_simulations: tuple[HalfOpenWindow, ...] = ()

    def __post_init__(self) -> None:
        if self.training.end > self.validation.start:
            raise ValueError("Training and validation windows overlap")
        if self.validation.end > self.test.start:
            raise ValueError("Validation and test windows overlap")
        for name in ("training", "validation", "test"):
            window = getattr(self, name)
            for simulation in self.controller_simulations:
                if window.overlaps(simulation):
                    raise ValueError(
                        f"{name} data overlaps HVAC controller simulation "
                        f"[{simulation.start}, {simulation.end})"
                    )

    def named(self) -> dict[str, HalfOpenWindow]:
        return {
            "training": self.training,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class HVACHeatInput:
    """Map source columns to the 2R2C model's signed thermal heat in kW.

    `power_mode` uses a caller-provided signed thermal-kW/electric-kW factor
    for each heating-mode label. The default power column contains heating
    alone. If the total heating-plus-cooling column is selected, cooling must
    be off. No efficiency or heat-pump COP is inferred here.
    """

    representation: Literal["thermal_heat", "power_mode"] = "thermal_heat"
    heat_column: str = "hvac_indoor_sensible_heat_w"
    heat_multiplier_to_kw: float = 0.001
    power_column: str = HVAC_HEATING_POWER_COLUMN
    mode_column: str = "heating_mode"
    cooling_mode_column: str = "cooling_mode"
    mode_heat_factors: Mapping[str, float] = field(default_factory=dict)

    def required_columns(self) -> tuple[str, ...]:
        if self.representation == "thermal_heat":
            return (self.heat_column,)
        if self.representation == "power_mode":
            columns = (self.power_column, self.mode_column)
            return (
                (*columns, self.cooling_mode_column)
                if self.power_column == HVAC_POWER_COLUMN
                else columns
            )
        raise ValueError(f"Unknown heat-input representation {self.representation!r}")

    def thermal_heat_kw(self, frame: pd.DataFrame) -> pd.Series:
        if self.representation == "thermal_heat":
            multiplier = float(self.heat_multiplier_to_kw)
            if not np.isfinite(multiplier) or multiplier == 0:
                raise ValueError("heat_multiplier_to_kw must be finite and nonzero")
            return pd.to_numeric(frame[self.heat_column], errors="coerce") * multiplier
        if self.representation != "power_mode":
            raise ValueError(f"Unknown heat-input representation {self.representation!r}")
        if not self.mode_heat_factors:
            raise ValueError(
                "power_mode requires explicit signed thermal-kW/electric-kW "
                "factors for every observed mode"
            )
        if self.power_column == HVAC_POWER_COLUMN:
            if self.cooling_mode_column not in frame:
                raise ValueError(
                    "power_mode requires a cooling-mode column to verify that "
                    "total HVAC electric power contains no cooling power"
                )
            cooling_mode = frame[self.cooling_mode_column]
            cooling_off = (
                cooling_mode.notna()
                & cooling_mode.astype(str).str.strip().str.casefold().isin(
                    {"off", "0", "false"}
                )
            )
            if not cooling_off.all():
                raise ValueError(
                    "power_mode cannot convert total HVAC electric power while "
                    "cooling is active or its mode is unknown; use the heating-only "
                    "power column or thermal_heat input"
                )
        mode = frame[self.mode_column].astype(str)
        missing = sorted(set(mode) - set(self.mode_heat_factors))
        if missing:
            raise ValueError(f"No thermal-heat factor for HVAC mode(s): {missing}")
        factors = mode.map(self.mode_heat_factors).astype(float)
        if not np.isfinite(factors).all():
            raise ValueError("Mode heat factors must be finite")
        power = pd.to_numeric(frame[self.power_column], errors="coerce")
        return power * factors


@dataclass(frozen=True)
class HVAC2R2CData:
    frame: pd.DataFrame
    sampling_minutes: float
    disturbance_model: Literal["constant", "day_night"]
    day_start_hour: float
    night_start_hour: float

    @property
    def segment_ids(self) -> np.ndarray:
        times = self.frame.index.to_numpy(dtype="datetime64[ns]")
        expected_ns = int(round(self.sampling_minutes * 60 * 1e9))
        discontinuity = np.diff(times).astype("timedelta64[ns]").astype(np.int64) != expected_ns
        segments = np.zeros(len(times), dtype=int)
        segments[1:] = np.cumsum(discontinuity)
        return segments

    def fit_arguments(self) -> dict[str, np.ndarray | None]:
        index = self.frame.index
        hours = (
            index.hour.to_numpy(dtype=float)
            + index.minute.to_numpy(dtype=float) / 60.0
            + index.second.to_numpy(dtype=float) / 3600.0
        )
        return {
            "timestamps": index.to_numpy(dtype="datetime64[ns]"),
            "indoor_temperature": self.frame["indoor_c"].to_numpy(dtype=float),
            "ambient_temperature": self.frame["ambient_c"].to_numpy(dtype=float),
            "heating_input": self.frame["thermal_heat_kw"].to_numpy(dtype=float),
            "solar": self.frame["ghi_w_per_m2"].to_numpy(dtype=float),
            "is_day": (
                day_mask_from_hours(hours, self.day_start_hour, self.night_start_hour)
                if self.disturbance_model == "day_night"
                else None
            ),
            "segment_ids": self.segment_ids,
        }


def load_hvac_2r2c_data(
    csv_path: str | Path,
    *,
    sampling_minutes: float = 30.0,
    heat_input: HVACHeatInput = HVACHeatInput(),
    disturbance_model: Literal["constant", "day_night"] = "constant",
    day_start_hour: float = 7.0,
    night_start_hour: float = 19.0,
) -> HVAC2R2CData:
    """Resample OCHRE HVAC transition records as the reference fitter does."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not np.isfinite(sampling_minutes) or sampling_minutes <= 0:
        raise ValueError("sampling_minutes must be positive and finite")
    if disturbance_model not in {"constant", "day_night"}:
        raise ValueError("disturbance_model must be constant or day_night")

    source_columns = (
        "timestamp",
        "indoor_temperature_c",
        "outdoor_temperature_c",
        "global_horizontal_irradiance_w_per_m2",
        *heat_input.required_columns(),
    )
    source = pd.read_csv(path, usecols=list(dict.fromkeys(source_columns)))
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    if source["timestamp"].isna().any():
        raise ValueError("HVAC timestamp column contains unparseable values")
    if source["timestamp"].dt.tz is not None:
        raise ValueError("HVAC timestamps must be timezone-naive")
    source = source.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    heat_kw = heat_input.thermal_heat_kw(source)
    numeric = pd.DataFrame(
        {
            "timestamp": source["timestamp"],
            "indoor_c": pd.to_numeric(source["indoor_temperature_c"], errors="coerce"),
            "ambient_c": pd.to_numeric(source["outdoor_temperature_c"], errors="coerce"),
            "ghi_w_per_m2": pd.to_numeric(
                source["global_horizontal_irradiance_w_per_m2"], errors="coerce"
            ),
            "thermal_heat_kw": heat_kw,
        }
    ).set_index("timestamp")
    interval = pd.Timedelta(minutes=float(sampling_minutes))
    frame = numeric.resample(interval).mean().dropna()
    if len(frame) < 2:
        raise ValueError("Fewer than two complete HVAC samples remain")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("HVAC data contains nonfinite values after resampling")
    return HVAC2R2CData(
        frame=frame,
        sampling_minutes=float(sampling_minutes),
        disturbance_model=disturbance_model,
        day_start_hour=float(day_start_hour),
        night_start_hour=float(night_start_hour),
    )


def fit_hvac_2r2c(
    data: HVAC2R2CData,
    windows: HVAC2R2CWindows,
    *,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    lambda_temperature: float = 0.5,
    indoor_capacitance_grid: Sequence[float] | None = None,
    mass_capacitance_grid: Sequence[float] | None = None,
    mass_time_constant_grid_hours: Sequence[float] | None = None,
    initial_mass_offset_bounds: tuple[float, float] = (-10.0, 10.0),
    indoor_ylim: tuple[float, float] | None = None,
    input_description: Mapping[str, object] | None = None,
    source_path: str | Path | None = None,
) -> TwoR2CModel:
    """Fit on training only; save split metrics, traces, and two-state plots."""
    for name, window in windows.named().items():
        if not ((data.frame.index >= window.start) & (data.frame.index < window.end)).any():
            raise ValueError(f"{name} window contains no HVAC samples")

    model = TwoR2CModel(
        sampling_minutes=data.sampling_minutes,
        disturbance_model=data.disturbance_model,
        lambda_temperature=lambda_temperature,
        indoor_capacitance_grid=indoor_capacitance_grid,
        mass_capacitance_grid=mass_capacitance_grid,
        mass_time_constant_grid_hours=mass_time_constant_grid_hours,
        initial_mass_offset_bounds=initial_mass_offset_bounds,
    ).fit(
        **data.fit_arguments(),
        training_range=(windows.training.start, _inclusive_end(windows.training)),
        validation_range=(windows.validation.start, _inclusive_end(windows.validation)),
        test_range=(windows.test.start, _inclusive_end(windows.test)),
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    traces = prediction_frame(data, model, windows)
    limits = plot_2r2c_splits(traces, model, output / "figures", indoor_ylim=indoor_ylim)
    traces.to_csv(output / "predictions.csv", index_label="timestamp")
    (output / "axis_limits.json").write_text(
        json.dumps(limits, indent=2) + "\n", encoding="utf-8"
    )
    summary = model.summary()
    summary["source_csv"] = str(source_path) if source_path is not None else None
    summary["input_conversion"] = dict(input_description or {})
    if data.disturbance_model == "day_night":
        summary["day_night_indicator"] = {
            "day_start_hour": data.day_start_hour,
            "night_start_hour": data.night_start_hour,
            "applied_to": "current interval",
        }
    summary["half_open_windows"] = {
        **{name: window.as_dict() for name, window in windows.named().items()},
        "controller_simulations": [
            window.as_dict() for window in windows.controller_simulations
        ],
    }
    summary["plot_axis_limits"] = limits
    (output / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_value) + "\n", encoding="utf-8"
    )
    return model


def prediction_frame(
    data: HVAC2R2CData,
    model: TwoR2CModel,
    windows: HVAC2R2CWindows,
) -> pd.DataFrame:
    """Return aligned measured air and four predictions of both states."""
    frame = data.frame.copy()
    frame["split"] = "outside"
    for name, window in windows.named().items():
        frame.loc[
            (frame.index >= window.start) & (frame.index < window.end), "split"
        ] = name
    frame["indoor_one_step_c"] = model.temperature_predictions_["indoor"]
    frame["mass_one_step_latent_c"] = model.temperature_predictions_["mass"]
    for label in ("8_hour", "12_hour", "24_hour", "open_loop"):
        frame[f"indoor_{label}_c"] = model.rollout_predictions_[label]["indoor"]
        frame[f"mass_{label}_latent_c"] = model.rollout_predictions_[label]["mass"]
    frame["heat_reconstructed_kw"] = model.heating_input_predictions_
    # The shared boundary state can hold a prediction from the preceding split.
    # Each split's first state is an initialization, not its own prediction.
    prediction_columns = [
        column
        for column in frame
        if column.startswith(("indoor_", "mass_")) and column != "indoor_c"
    ]
    for window in windows.named().values():
        first = frame.index.searchsorted(window.start)
        if first < len(frame) and frame.index[first] < window.end:
            frame.loc[frame.index[first], prediction_columns] = np.nan
    return frame


def plot_2r2c_splits(
    traces: pd.DataFrame,
    model: TwoR2CModel,
    figure_directory: str | Path,
    *,
    indoor_ylim: tuple[float, float] | None = None,
) -> dict[str, list[float]]:
    """Use the TPC comparison palette with consistent axes across splits."""
    figures = Path(figure_directory)
    figures.mkdir(parents=True, exist_ok=True)
    indoor_columns = [
        "indoor_c",
        "indoor_one_step_c",
        "indoor_8_hour_c",
        "indoor_24_hour_c",
        "indoor_open_loop_c",
    ]
    mass_columns = [
        "mass_one_step_latent_c",
        "mass_8_hour_latent_c",
        "mass_24_hour_latent_c",
        "mass_open_loop_latent_c",
    ]
    relevant = traces[traces["split"] != "outside"]
    indoor_limits = (
        _validated_limits(indoor_ylim)
        if indoor_ylim is not None
        else _common_limits(relevant[indoor_columns].to_numpy(dtype=float))
    )
    mass_limits = _common_limits(relevant[mass_columns].to_numpy(dtype=float))
    axis_limits = {
        "indoor_temperature_c": list(indoor_limits),
        "latent_mass_temperature_c": list(mass_limits),
    }
    series = (
        ("one_step", "One-step"),
        ("8_hour", "8-hour reset"),
        ("24_hour", "24-hour reset"),
        ("open_loop", "Open loop"),
    )

    for name in ("training", "validation", "test"):
        part = traces.loc[traces["split"] == name]
        if part.empty:
            continue
        fig, (air_ax, mass_ax) = plt.subplots(
            2,
            1,
            figsize=(16.5, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1.55, 1.0]},
        )
        for axis in (air_ax, mass_ax):
            axis.axvspan(part.index[0], part.index[-1], color=SPLIT_FILLS[name], alpha=0.18)
            axis.axvline(part.index[0], color="#555555", linestyle="--", linewidth=1.0)
            axis.axvline(part.index[-1], color="#555555", linestyle="--", linewidth=1.0)
            axis.grid(alpha=0.25)
        air_ax.plot(
            part.index,
            part["indoor_c"],
            color=COLORS["measured"],
            linewidth=1.25,
            label="Measured indoor",
            zorder=5,
        )
        for label, display in series:
            key = "one_step" if label == "one_step" else label
            air_ax.plot(
                part.index,
                part[f"indoor_{key}_c"],
                color=COLORS[label],
                linewidth=1.0,
                alpha=0.88,
                label=display,
            )
            mass_ax.plot(
                part.index,
                part[f"mass_{key}_latent_c"],
                color=COLORS[label],
                linewidth=1.0,
                alpha=0.88,
                label=display,
            )
        metrics = model.split_metrics_[name]
        air_ax.text(
            0.01,
            0.98,
            (
                f"{name.capitalize()} indoor RMSE (°C)\n"
                f"1-step {metrics['temperature_rmse']:.3f} | "
                f"8 h {metrics['temperature_rmse_8_hour']:.3f}\n"
                f"24 h {metrics['temperature_rmse_24_hour']:.3f} | "
                f"open {metrics['temperature_rmse_open_loop']:.3f}"
            ),
            transform=air_ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#777777"},
            zorder=8,
        )
        mass_ax.text(
            0.01,
            0.98,
            "Latent mass state; no measured mass-temperature target",
            transform=mass_ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#777777"},
            zorder=8,
        )
        air_ax.set_ylim(*indoor_limits)
        mass_ax.set_ylim(*mass_limits)
        air_ax.set_title(f"2R2C: {name} indoor and latent mass predictions")
        air_ax.set_ylabel("Indoor temperature (°C)")
        mass_ax.set_ylabel("Latent mass temperature (°C)")
        mass_ax.set_xlabel("Time")
        air_ax.legend(loc="lower left", ncol=5, fontsize=8)
        mass_ax.legend(loc="lower left", ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{name}_two_state_predictions.png", dpi=300)
        plt.close(fig)
    return axis_limits


def _common_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot determine comparison axes without finite values")
    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    margin = max(0.5, 0.05 * (maximum - minimum))
    return minimum - margin, maximum + margin


def _validated_limits(limits: tuple[float, float]) -> tuple[float, float]:
    low, high = map(float, limits)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("indoor_ylim must contain finite increasing bounds")
    return low, high


def _inclusive_end(window: HalfOpenWindow) -> pd.Timestamp:
    """Exclude the next split's first state from reference-fit targets."""
    return window.end - pd.Timedelta(1, unit="ns")


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")

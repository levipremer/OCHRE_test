"""Predictive control for the trained HVAC state-space model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import sparse

from ..forecast import PRICE_COLUMN, Forecast
from ..training.hvac import (
    DAY_INDICATOR_COLUMN,
    HVAC_COIL_ER_HEAT_COLUMN,
    HVACInternalGainRegime,
    NIGHT_INDICATOR_COLUMN,
    heating_setpoints_c,
)


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "building_envelope_training"
    / "state_space_predictor_regularized"
)
HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY = "HVAC Indoor Boundary Temperature (C)"
HP_ONLY_SETPOINT_OFFSET_C = 1.0
ER_CALL_SETPOINT_OFFSET_C = 1.5
OFF_SETPOINT_MARGIN_C = 0.25


@dataclass(frozen=True)
class PreparedHVACMixedIntegerStep:
    """An HVAC MILP ready to join another device's optimization problem."""

    timestamp: pd.Timestamp
    previous_status: Mapping[str, Any]
    problem: cp.Problem
    heat_hvac: cp.Expression
    heat_hp: cp.Expression
    heat_er: cp.Expression
    prices: np.ndarray
    cop: np.ndarray
    initial_state: np.ndarray
    disturbances: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model: Mapping[str, Any]
    setpoints: np.ndarray
    historical_degree_hours: float | np.ndarray | None
    objective_components: Mapping[str, cp.Expression]
    comfort_budget_expressions: Mapping[str, cp.Expression]
    mixed_int_state_expressions: Mapping[str, cp.Expression]

    @property
    def electric_power_kw(self) -> cp.Expression:
        """The existing HP-plus-fan and ER electricity proxy for each interval."""
        return cp.multiply(1 / self.cop, self.heat_hp / 1000) + self.heat_er / 1000

    @property
    def non_electric_objective(self) -> cp.Expression:
        """All existing HVAC penalties, leaving electricity for one joint term."""
        return sum(
            expression for name, expression in self.objective_components.items()
            if name != "electricity_objective_cost"
        )


@dataclass
class HVACPredictiveController:
    """Use the regularized predictor at a 15-minute receding-horizon cadence."""

    model_path: Path = DEFAULT_MODEL_PATH
    horizon: pd.Timedelta = pd.Timedelta(hours=12)
    model_timestep: pd.Timedelta = pd.Timedelta(minutes=15)
    heating_name: str = "HVAC Heating"
    cooling_name: str = "HVAC Cooling"
    optimization_kind: str = "convex"
    solver: str = "GUROBI"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    gain_regime: HVACInternalGainRegime = field(default_factory=HVACInternalGainRegime)
    comfort_band_c: float = 1.5
    comfort_weight: float = 1.0
    persistence_weight: float = 1.0
    enforce_degree_hour_budget: bool = False
    enforce_instantaneous_cap: bool = False
    # Explicit full-period extension of the established 8-hour prediction plus
    # 16-hour measured-history comfort budget. Ordinary MPC keeps its original
    # single horizon-total constraint and configurable lookback.
    full_horizon_rolling_budget: bool = False
    degree_hour_limit_c_hours: float = 2.0
    max_deviation_c: float = 1.0
    degree_hour_slack_weight: float = 100.0
    max_deviation_slack_weight: float = 10_000.0
    max_cycles: int = 3
    dwell_time_off: pd.Timedelta = pd.Timedelta(minutes=15)
    dwell_time_on: pd.Timedelta = pd.Timedelta(minutes=60)
    enforce_cycle_limit: bool = False
    enforce_dwell_constraints: bool = False

    _model: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _heating_equipment: Any | None = field(default=None, init=False, repr=False)
    _history: list[tuple[float, np.ndarray, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _observations: dict[pd.Timestamp, tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _comfort_temperature_observations: dict[pd.Timestamp, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _comfort_history_step: pd.Timedelta | None = field(
        default=None, init=False, repr=False
    )
    _last_completed_history_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _last_observed_hp_on: int | None = field(default=None, init=False, repr=False)
    _last_observed_hp_transition_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _recent_hp_off_times: list[pd.Timestamp] = field(
        default_factory=list, init=False, repr=False
    )
    _last_decision_time: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _last_setpoint: float | None = field(default=None, init=False, repr=False)
    _off_requested: bool = field(default=False, init=False, repr=False)
    _hp_only_requested: bool = field(default=False, init=False, repr=False)
    _hp_only_command_sent: bool = field(default=False, init=False, repr=False)
    _hp_only_er_off_pending: bool = field(default=False, init=False, repr=False)
    _hp_only_er_reset_attempted: bool = field(default=False, init=False, repr=False)
    _hp_only_er_reset_feedback_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _er_target_c: float | None = field(default=None, init=False, repr=False)
    _optimization_plans: list[pd.DataFrame] = field(
        default_factory=list, init=False, repr=False
    )
    _objective_component_expressions: dict[str, cp.Expression] = field(
        default_factory=dict, init=False, repr=False
    )
    _comfort_budget_expressions: dict[str, cp.Expression] = field(
        default_factory=dict, init=False, repr=False
    )
    _comfort_budget_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _mixed_int_state_expressions: dict[str, cp.Expression] = field(
        default_factory=dict, init=False, repr=False
    )
    _objective_component_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _solve_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    solve_durations_s: dict[str, list[float]] = field(
        default_factory=lambda: {"convex": [], "mixed integer": []},
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self.horizon = pd.Timedelta(self.horizon)
        self.model_timestep = pd.Timedelta(self.model_timestep)
        self.dwell_time_off = pd.Timedelta(self.dwell_time_off)
        self.dwell_time_on = pd.Timedelta(self.dwell_time_on)
        if self.optimization_kind not in {"convex", "mixed integer"}:
            raise ValueError("HVAC optimization kind must be convex or mixed integer")
        if self.optimization_kind == "mixed integer" and self.solver == "CLARABEL":
            raise ValueError("Mixed-integer HVAC optimization requires a MIP-capable solver")
        if self.optimization_kind == "convex" and (
            self.enforce_cycle_limit or self.enforce_dwell_constraints
        ):
            raise ValueError("HVAC cycle and dwell constraints require mixed integer optimization")
        if (
            isinstance(self.max_cycles, bool)
            or not isinstance(self.max_cycles, (int, np.integer))
            or self.max_cycles < 1
            or pd.isna(self.dwell_time_off)
            or pd.isna(self.dwell_time_on)
            or self.dwell_time_off <= pd.Timedelta(0)
            or self.dwell_time_on <= pd.Timedelta(0)
        ):
            raise ValueError("HVAC cycle limit and dwell times must be positive")
        if (
            self.horizon <= pd.Timedelta(0)
            or self.model_timestep <= pd.Timedelta(0)
            or self.horizon % self.model_timestep
        ):
            raise ValueError("HVAC horizon must contain complete positive model steps")
        if (
            not np.isfinite([
                self.comfort_band_c, self.comfort_weight, self.persistence_weight,
                self.degree_hour_limit_c_hours, self.max_deviation_c,
                self.degree_hour_slack_weight, self.max_deviation_slack_weight,
            ]).all()
            or self.comfort_band_c < 0
            or self.comfort_weight < 0
            or self.persistence_weight < 0
            or self.degree_hour_limit_c_hours < 0
            or self.max_deviation_c < 0
            or self.degree_hour_slack_weight < 0
            or self.max_deviation_slack_weight < 0
        ):
            raise ValueError("HVAC comfort limits and weights must be nonnegative")
        if not isinstance(self.full_horizon_rolling_budget, bool):
            raise ValueError("HVAC full-horizon rolling budget flag must be boolean")
        if self.full_horizon_rolling_budget and (
            not self.enforce_degree_hour_budget
            or self.horizon < pd.Timedelta(hours=8)
            or self.model_timestep != pd.Timedelta(minutes=15)
        ):
            raise ValueError(
                "HVAC full-horizon rolling budget requires the degree-hour budget, "
                "at least eight prediction hours, and 15-minute model steps"
            )
        if (self.enforce_degree_hour_budget and not self.full_horizon_rolling_budget
                and self.horizon >= pd.Timedelta(hours=24)):
            raise ValueError("HVAC degree-hour budget requires a horizon under 24 hours")

    @property
    def _comfort_history_window(self) -> pd.Timedelta:
        if self.full_horizon_rolling_budget:
            return pd.Timedelta(hours=16)
        if self.enforce_degree_hour_budget:
            return pd.Timedelta(hours=24) - self.horizon
        return pd.Timedelta(hours=12)

    @property
    def optimization_plans(self) -> tuple[pd.DataFrame, ...]:
        return tuple(self._optimization_plans)

    @property
    def objective_component_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._objective_component_history)

    @property
    def solve_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._solve_history)

    @property
    def comfort_budget_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._comfort_budget_history)

    def bind_heating_equipment(self, equipment: Any) -> None:
        """Expose OCHRE's heat-pump performance curve to forecast preparation."""
        self._heating_equipment = equipment

    def _load_model(self) -> dict[str, Any]:
        """Load exactly the input, output, and disturbance order that was fit."""
        if self._model is not None:
            return self._model
        model_path = Path(self.model_path)
        metadata_path = model_path / "metadata.json"
        arrays_path = model_path / "model_arrays.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"HVAC state-space artifact is missing: {model_path}")
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("model_type") != "regularized_exogenous_state_space":
            raise ValueError("Selected HVAC artifact is not a regularized state-space model")
        if metadata["columns"]["input"] != [HVAC_COIL_ER_HEAT_COLUMN]:
            raise ValueError("HVAC predictor must use HP coil plus ER heat in watts")
        output_names = tuple(metadata["columns"]["outputs"])
        disturbance_names = tuple(metadata["columns"]["disturbances"])
        memory = int(metadata["selection"]["memory"])
        if output_names != ("indoor_temperature_c",) or memory < 1:
            raise ValueError("HVAC output or memory does not match the controller")
        if metadata["configuration"]["model_timestep_seconds"] != self.model_timestep.total_seconds():
            raise ValueError("HVAC model timestep does not match its training artifact")
        with np.load(arrays_path) as arrays:
            matrices = {name: arrays[name] for name in ("A", "B", "E")}
        state_size = memory * (1 + len(disturbance_names) + len(output_names))
        if (
            matrices["A"].shape != (state_size, state_size)
            or matrices["B"].shape != (state_size, 1)
            or matrices["E"].shape != (state_size, len(disturbance_names))
            or not all(np.isfinite(matrix).all() for matrix in matrices.values())
        ):
            raise ValueError("HVAC model matrices do not match its metadata")
        output_matrix = np.zeros((1, state_size))
        output_matrix[0, -1] = 1.0
        self._model = {
            **matrices,
            "m": memory,
            "output_matrix": output_matrix,
            "output_names": output_names,
            "disturbance_names": disturbance_names,
        }
        return self._model

    @staticmethod
    def _source_timestep(forecast: Forecast) -> pd.Timedelta:
        index = forecast.data.index
        if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
            raise ValueError("HVAC forecast needs a regular DatetimeIndex")
        source_step = pd.Timedelta(index[1] - index[0])
        if source_step <= pd.Timedelta(0):
            raise ValueError("HVAC forecast timestep must be positive")
        return source_step

    def _planning_index(self, timestamp: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(
            pd.Timestamp(timestamp),
            periods=self.horizon // self.model_timestep,
            freq=self.model_timestep,
        )

    def _is_model_step_boundary(self, timestamp: pd.Timestamp) -> bool:
        """Check elapsed-time alignment without ambiguous local DST flooring."""
        timestamp = pd.Timestamp(timestamp)
        clock = timestamp.tz_convert("UTC") if timestamp.tzinfo is not None else timestamp
        return clock == clock.floor(self.model_timestep)

    @staticmethod
    def _training_setpoints(index: pd.DatetimeIndex) -> np.ndarray:
        """Four-period schedule from generate_building_envelope_data.py."""
        return heating_setpoints_c(index)

    def _comfort_bounds(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        setpoints = self._training_setpoints(index)
        return setpoints - self.comfort_band_c, setpoints + self.comfort_band_c

    def record_measured_temperature(
        self,
        measured_at: pd.Timestamp,
        temperature_c: float,
        sample_step: pd.Timedelta,
    ) -> None:
        """Retain measured temperatures for the degree-hour lookback window."""
        measured_at = pd.Timestamp(measured_at)
        sample_step = pd.Timedelta(sample_step)
        temperature_c = float(temperature_c)
        if (
            pd.isna(measured_at)
            or sample_step <= pd.Timedelta(0)
            or self._comfort_history_window % sample_step
            or (self.full_horizon_rolling_budget and self.model_timestep % sample_step)
            or not np.isfinite(temperature_c)
        ):
            raise ValueError("HVAC comfort history needs finite regular measurements")
        if self._comfort_history_step is None:
            self._comfort_history_step = sample_step
        elif self._comfort_history_step != sample_step:
            raise ValueError("HVAC comfort history timestep changed during the run")
        self._comfort_temperature_observations[measured_at] = temperature_c
        cutoff = measured_at - self._comfort_history_window
        self._comfort_temperature_observations = {
            time: value for time, value in self._comfort_temperature_observations.items()
            if time >= cutoff
        }

    def _historical_degree_hours(
        self, timestamp: pd.Timestamp,
    ) -> float | np.ndarray | None:
        """Integrate measured violations over the degree-hour lookback window.

        The opt-in full-period mode returns the remaining measured contribution
        to every 24-hour window ending from issue+8 hours through the final
        prediction endpoint. The first window uses all preceding 16 hours;
        subsequent windows discard the oldest completed measurements, and no
        measured contribution remains once the endpoint reaches issue+24 hours.
        """
        sample_step = self._comfort_history_step
        if sample_step is None:
            return None
        timestamp = pd.Timestamp(timestamp)
        expected = pd.date_range(
            timestamp - self._comfort_history_window,
            timestamp - sample_step,
            freq=sample_step,
        )
        if any(time not in self._comfort_temperature_observations for time in expected):
            return None
        temperature = np.asarray(
            [self._comfort_temperature_observations[time] for time in expected],
            dtype=float,
        )
        lower, upper = self._comfort_bounds(expected)
        violation = np.maximum(temperature - upper, 0) + np.maximum(
            lower - temperature, 0
        )
        if self.full_horizon_rolling_budget:
            window_ends = np.arange(
                int(pd.Timedelta(hours=8) / self.model_timestep),
                int(self.horizon / self.model_timestep) + 1,
            )
            samples_per_step = int(self.model_timestep / sample_step)
            discarded = np.minimum(
                (window_ends - window_ends[0]) * samples_per_step,
                len(violation),
            )
            # Suffix sums retain nonnegative values without cancellation when
            # only a small final measured contribution remains in a window.
            suffix = np.r_[np.cumsum(violation[::-1])[::-1], 0.0]
            return sample_step.total_seconds() / 3600 * suffix[discarded]
        return float(sample_step.total_seconds() / 3600 * violation.sum())

    @staticmethod
    def _forecast_prices(index: pd.DatetimeIndex, forecast: Forecast) -> np.ndarray | None:
        if PRICE_COLUMN not in forecast.data:
            return None
        prices = forecast.data.reindex(index)[PRICE_COLUMN].to_numpy(dtype=float)
        return prices if np.isfinite(prices).all() else None

    def _power_and_er_limits_w(
        self, previous_status: Mapping[str, Any]
    ) -> tuple[float, float] | None:
        """Return rated HP compressor-plus-fan power and rated ER heat."""
        equipment = self._heating_equipment
        if equipment is None:
            return None
        er_rating = previous_status.get(f"{self.heating_name} ER Rated Capacity (W)")
        capacity_list = getattr(equipment, "capacity_list", None)
        eir_list = getattr(equipment, "eir_list", None)
        if er_rating is None or not capacity_list or not eir_list:
            return None
        values = np.asarray([
            capacity_list[-1], eir_list[-1],
            getattr(equipment, "fan_power_max", np.nan), er_rating,
        ], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            return None
        return float(values[0] * values[1] + values[2]), float(values[3])

    def _observed_coil_er_heat_w(self) -> tuple[float, float] | None:
        """Read the last OCHRE step's HP coil and total heating capacities."""
        equipment = self._heating_equipment
        if equipment is None:
            return None
        try:
            total = float(equipment.capacity)
            er = float(equipment.er_capacity)
        except (AttributeError, TypeError, ValueError):
            return None
        if not np.isfinite([total, er]).all() or er < 0:
            return None
        return total - er, total

    def _observed_hp_coil_heat_w(self) -> float | None:
        """Return the preceding minute's signed HP coil heat."""
        values = self._observed_coil_er_heat_w()
        return values[0] if values is not None else None

    def _observe_hp_state(
        self, timestamp: pd.Timestamp, previous_status: Mapping[str, Any]
    ) -> int | None:
        """Track the observed compressor mode and stops over the preceding hour."""
        timestamp = pd.Timestamp(timestamp)
        cutoff = timestamp - pd.Timedelta(hours=1)
        self._recent_hp_off_times = [
            stopped_at for stopped_at in self._recent_hp_off_times
            if stopped_at > cutoff
        ]

        mode = previous_status.get(f"{self.heating_name} Mode")
        if mode in {"HP On", "HP and ER On"}:
            on = 1
        elif mode in {"Off", "ER On"}:
            on = 0
        else:
            equipment = self._heating_equipment
            if equipment is None:
                return None
            mode = getattr(equipment, "mode", None)
            if mode in {"HP On", "HP and ER On"}:
                on = 1
            elif mode in {"Off", "ER On"}:
                on = 0
            else:
                return None

        if self._last_observed_hp_on is not None and self._last_observed_hp_on != on:
            self._last_observed_hp_transition_time = timestamp
            if self._last_observed_hp_on == 1:
                self._recent_hp_off_times.append(timestamp)
        self._last_observed_hp_on = on
        return on

    def _remaining_hp_dwell_steps(self, timestamp: pd.Timestamp, on: int) -> int:
        """Return full model intervals still required in the observed HP mode."""
        if self._last_observed_hp_transition_time is None:
            return 0
        required = self.dwell_time_on if on else self.dwell_time_off
        remaining = required - (
            pd.Timestamp(timestamp) - self._last_observed_hp_transition_time
        )
        if remaining <= pd.Timedelta(0):
            return 0
        return int(np.ceil(remaining / self.model_timestep))

    def _forecast_cop(
        self, outdoor_temperature_c: np.ndarray, indoor_setpoint_c: np.ndarray
    ) -> np.ndarray | None:
        """Convert forecast coil COP to coil heat per HP-plus-fan electricity."""
        equipment = self._heating_equipment
        if equipment is None:
            return None
        outdoor = np.asarray(outdoor_temperature_c, dtype=float).reshape(-1)
        indoor = np.asarray(indoor_setpoint_c, dtype=float).reshape(-1)
        if outdoor.shape != indoor.shape or not np.isfinite(outdoor).all():
            return None
        if not np.isfinite(indoor).all():
            return None
        enabled_speeds = np.flatnonzero(~equipment.disable_speeds)
        if enabled_speeds.size == 0:
            return None
        full_speed = int(enabled_speeds[-1]) + 1
        eir = np.asarray([
            equipment.calculate_biquadratic_param(
                "eir",
                full_speed,
                ambient_temperature_c=float(weather),
                coil_input_temperature_c=float(setpoint),
            )
            for weather, setpoint in zip(outdoor, indoor)
        ], dtype=float)
        if not np.isfinite(eir).all() or np.any(eir <= 0):
            return None
        rated_main = float(equipment.capacity_list[-1] * equipment.eir_list[-1])
        rated_fan = float(equipment.fan_power_max)
        if not np.isfinite([rated_main, rated_fan]).all() or rated_main <= 0 or rated_fan < 0:
            return None
        return (1 / eir) * rated_main / (rated_main + rated_fan)

    def _disturbances_for_intervals(
        self,
        start: pd.Timestamp,
        steps: int,
        forecast: Forecast,
        model: Mapping[str, Any],
        *,
        future_weather: bool = False,
    ) -> np.ndarray | None:
        """Recreate training features before taking each 15-minute mean."""
        source_step = self._source_timestep(forecast)
        if self.model_timestep % source_step:
            raise ValueError("HVAC model timestep must contain whole forecast steps")
        samples_per_step = self.model_timestep // source_step
        expected = pd.date_range(
            pd.Timestamp(start), periods=steps * samples_per_step, freq=source_step
        )
        rows = forecast.data.reindex(expected)
        if future_weather and forecast.weather_archive is not None:
            archived_weather = forecast.weather_archive.weather_for_hvac(
                start,
                expected,
                forecast.data.loc[pd.Timestamp(start)],
                self.model_timestep,
            )
            rows.loc[:, archived_weather.columns] = archived_weather
        raw_columns = {
            "outdoor_temperature_c": "Ambient Dry Bulb (C)",
            "outdoor_relative_humidity_fraction": "Ambient Relative Humidity (-)",
            "diffuse_horizontal_irradiance_w_per_m2": "DHI (W/m^2)",
            "global_horizontal_irradiance_w_per_m2": "GHI (W/m^2)",
        }
        names = model["disturbance_names"]
        required = {raw_columns[name] for name in names if name in raw_columns}
        if any(name.startswith("projected_dni_azimuth_") for name in names):
            required.update(("DNI (W/m^2)", "zenith", "azimuth"))
        if any(column not in rows for column in required):
            return None
        numeric = rows.loc[:, list(required)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            return None
        dni = np.maximum(rows["DNI (W/m^2)"].to_numpy(float), 0) if "DNI (W/m^2)" in required else None
        zenith = rows["zenith"].to_numpy(float) if "zenith" in required else None
        azimuth = np.deg2rad(rows["azimuth"].to_numpy(float)) if "azimuth" in required else None
        values = []
        for name in names:
            if name in {DAY_INDICATOR_COLUMN, NIGHT_INDICATOR_COLUMN}:
                is_day = np.asarray(
                    [self.gain_regime.is_day(time) for time in expected],
                    dtype=float,
                )
                if name == NIGHT_INDICATOR_COLUMN:
                    is_day = 1 - is_day
                values.append(is_day)
            elif name in raw_columns:
                values.append(rows[raw_columns[name]].to_numpy(float))
            elif name.startswith("projected_dni_azimuth_") and name.endswith("_w_per_m2"):
                surface_degrees = int(name.removeprefix("projected_dni_azimuth_").removesuffix("_w_per_m2"))
                incidence = np.sin(np.deg2rad(zenith)) * np.cos(azimuth - np.deg2rad(surface_degrees))
                values.append(dni * np.maximum(incidence, 0) * (zenith < 90))
            else:
                raise ValueError(f"Unsupported HVAC disturbance: {name}")
        result = np.empty((len(names), steps))
        for position, (name, samples) in enumerate(zip(names, values)):
            blocks = samples.reshape(steps, samples_per_step)
            result[position] = (
                blocks[:, 0]
                if name in {DAY_INDICATOR_COLUMN, NIGHT_INDICATOR_COLUMN}
                else blocks.mean(axis=1)
            )
        return result

    def _observe_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any],
        forecast: Forecast,
    ) -> None:
        source_step = self._source_timestep(forecast)
        observed_at = pd.Timestamp(timestamp) - source_step
        temperature = previous_status.get("Temperature - Indoor (C)")
        boundary_temperature = previous_status.get(
            HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY
        )
        heat = self._observed_coil_er_heat_w()
        if (
            temperature is not None
            and boundary_temperature is not None
            and heat is not None
            and np.isfinite([heat[1], temperature, boundary_temperature]).all()
        ):
            # The heat belongs to the completed minute; the trained output is
            # its ending temperature. Comfort history uses the minute start.
            self._observations[observed_at] = (
                heat[1], float(boundary_temperature)
            )
            if self.enforce_degree_hour_budget:
                self.record_measured_temperature(
                    observed_at, float(temperature), source_step
                )
        cutoff = pd.Timestamp(timestamp) - self._load_model()["m"] * self.model_timestep
        self._observations = {
            time: sample for time, sample in self._observations.items() if time >= cutoff
        }

    def _complete_history_interval(
        self,
        timestamp: pd.Timestamp,
        forecast: Forecast,
        model: Mapping[str, Any],
    ) -> bool:
        if self._last_completed_history_time == pd.Timestamp(timestamp):
            return True
        source_step = self._source_timestep(forecast)
        samples_per_step = self.model_timestep // source_step
        start = pd.Timestamp(timestamp) - self.model_timestep
        index = pd.date_range(start, periods=samples_per_step, freq=source_step)
        if any(time not in self._observations for time in index):
            return False
        disturbances = self._disturbances_for_intervals(start, 1, forecast, model)
        if disturbances is None:
            return False
        heat = float(np.mean([self._observations[time][0] for time in index]))
        temperature = self._observations[index[-1]][1]
        self._history.append((heat, disturbances[:, 0], temperature))
        self._history = self._history[-model["m"]:]
        self._last_completed_history_time = pd.Timestamp(timestamp)
        return True

    def record_warmup_step(
        self,
        next_timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any],
        forecast: Forecast,
    ) -> None:
        """Seed model memory and measured comfort history during baseline warm-up."""
        next_timestamp = pd.Timestamp(next_timestamp)
        self._observe_step(next_timestamp, previous_status, forecast)
        if self.optimization_kind == "mixed integer":
            self._observe_hp_state(next_timestamp, previous_status)
        if self._is_model_step_boundary(next_timestamp):
            self._complete_history_interval(next_timestamp, forecast, self._load_model())

    def _lifted_state(self, memory: int) -> np.ndarray:
        if len(self._history) != memory:
            raise ValueError("HVAC history is not ready")
        return np.concatenate([
            np.concatenate(([heat], disturbances, [temperature]))
            for heat, disturbances, temperature in self._history
        ])

    def _degree_hour_terms(
        self,
        indoor_temperature: cp.Expression,
        lower: np.ndarray,
        upper: np.ndarray,
        steps: int,
        timestep_hours: float,
        historical_degree_hours: float | np.ndarray | None,
    ) -> tuple[list[cp.Constraint], cp.Expression, cp.Expression]:
        """Build the selected accumulated-budget and maximum-deviation rules."""
        violation = cp.pos(indoor_temperature - upper) + cp.pos(
            lower - indoor_temperature
        )
        predicted_degree_hours = (
            self.model_timestep.total_seconds() / 3600 * cp.sum(violation)
        )
        constraints: list[cp.Constraint] = []
        cap_slack: cp.Expression = cp.Constant(np.zeros(steps))
        cap_cost: cp.Expression = cp.Constant(0.0)
        if self.enforce_instantaneous_cap:
            if self.enforce_degree_hour_budget:
                cap_slack = cp.Variable(steps, nonneg=True)
                constraints.append(violation <= self.max_deviation_c + cap_slack)
                cap_cost = (
                    self.max_deviation_slack_weight * timestep_hours / steps
                    * cp.sum(cap_slack)
                )
            else:
                constraints.append(violation <= self.max_deviation_c)
        budget_slack: cp.Expression = cp.Constant(0.0)
        budget_cost: cp.Expression = cp.Constant(0.0)
        rolling_expressions: dict[str, cp.Expression] = {}
        if self.enforce_degree_hour_budget:
            budget_slack = cp.Variable(nonneg=True)
            if self.full_horizon_rolling_budget:
                window_ends = np.arange(
                    int(pd.Timedelta(hours=8) / self.model_timestep), steps + 1,
                )
                history = (
                    np.asarray(historical_degree_hours, dtype=float).reshape(-1)
                    if historical_degree_hours is not None else np.array([])
                )
                window_steps = int(pd.Timedelta(hours=24) / self.model_timestep)
                if (history.shape != window_ends.shape
                        or not np.isfinite(history).all() or np.any(history < 0)
                        or np.any(np.diff(history) > 1e-10)
                        or np.any(history[window_ends >= window_steps] != 0)):
                    raise ValueError(
                        "HVAC full-horizon rolling budget needs one nonnegative "
                        "measured-history remainder per window, decreasing to zero "
                        "at the 24-hour endpoint"
                    )
                # Each predicted endpoint violation represents its preceding
                # 15-minute interval, exactly as in the original horizon sum.
                # Nonnegative sparse weights preserve convexity of the pos()
                # violations while avoiding duplicated full-horizon graphs.
                starts = np.maximum(0, window_ends - window_steps)
                lengths = window_ends - starts
                row = np.repeat(np.arange(len(window_ends)), lengths)
                column = np.concatenate([
                    np.arange(first, last) for first, last in zip(starts, window_ends)
                ])
                weights = sparse.csr_matrix(
                    (np.full(len(row), timestep_hours), (row, column)),
                    shape=(len(window_ends), steps),
                )
                predicted_by_window = weights @ violation
                totals = predicted_by_window + history
                # One shared slack charges the worst 24-hour window, once.
                constraints.append(totals <= self.degree_hour_limit_c_hours + budget_slack)
                rolling_expressions = {
                    "rolling_24h_window_end_step": cp.Constant(window_ends),
                    "rolling_24h_historical_degree_hours_c_hours": cp.Constant(history),
                    "rolling_24h_predicted_degree_hours_c_hours": predicted_by_window,
                    "rolling_24h_degree_hours_c_hours": totals,
                }
            else:
                if (
                    historical_degree_hours is None
                    or not np.isfinite(historical_degree_hours)
                    or historical_degree_hours < 0
                ):
                    raise ValueError("HVAC degree-hour budget needs measured history")
                remaining = self.degree_hour_limit_c_hours - historical_degree_hours
                constraints.append(predicted_degree_hours <= remaining + budget_slack)
            budget_cost = (self.degree_hour_slack_weight * timestep_hours /
                           steps * budget_slack)
        self._comfort_budget_expressions = {
            "predicted_violation_c": violation,
            "predicted_degree_hours_c_hours": predicted_degree_hours,
            "degree_hour_slack_c_hours": budget_slack,
            "maximum_deviation_slack_c": cap_slack,
            **rolling_expressions,
        }
        return constraints, budget_cost, cap_cost

    def optimization_problem_convex(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any] | None = None,
        *,
        power_limit_hp_w: float,
        heat_limit_er_w: float,
        temperature_lower_c: np.ndarray,
        temperature_upper_c: np.ndarray,
        cop: float | np.ndarray,
        previous_heat_hp_kw: float,
        comfort_weight: float = 1.0,
        historical_degree_hours: float | np.ndarray | None = None,
    ) -> tuple[cp.Problem, cp.Expression, cp.Expression, cp.Expression]:
        """Build the HP/ER split with the selected comfort formulation."""
        model = model or self._load_model()
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        prices = np.asarray(prices, dtype=float).reshape(-1)
        disturbances = np.asarray(disturbances, dtype=float)
        lower = np.asarray(temperature_lower_c, dtype=float).reshape(-1)
        upper = np.asarray(temperature_upper_c, dtype=float).reshape(-1)
        previous_heat_hp_kw = float(previous_heat_hp_kw)
        steps = prices.size
        if self.enforce_degree_hour_budget and steps != self.horizon // self.model_timestep:
            raise ValueError("HVAC degree-hour budget needs the complete configured horizon")
        cop_values = np.asarray(cop, dtype=float)
        if cop_values.ndim == 0:
            cop_values = np.full(steps, float(cop_values))
        else:
            cop_values = cop_values.reshape(-1)
        if (
            steps == 0
            or disturbances.shape != (model["E"].shape[1], steps)
            or initial_state.shape != (model["A"].shape[0],)
            or lower.shape != (steps,)
            or upper.shape != (steps,)
            or cop_values.shape != (steps,)
            or not np.isfinite(initial_state).all()
            or not np.isfinite(prices).all()
            or not np.isfinite(disturbances).all()
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or not np.isfinite(cop_values).all()
            or np.any(lower > upper)
            or not np.isfinite([
                power_limit_hp_w, heat_limit_er_w, comfort_weight, previous_heat_hp_kw
            ]).all()
            or power_limit_hp_w < 0
            or heat_limit_er_w < 0
            or np.any(cop_values <= 0)
            or comfort_weight < 0
        ):
            raise ValueError("HVAC state, forecast, bounds, or equipment limits are invalid")

        timestep_hours = self.model_timestep.total_seconds() / 3600
        state = cp.Variable((model["A"].shape[0], steps + 1))
        # Solve in kW for conditioning while the trained B matrix receives W.
        heat_hvac_kw = cp.Variable(steps, nonneg=True)
        heat_hp_kw = cp.Variable(steps, nonneg=True)
        heat_er_kw = cp.Variable(steps, nonneg=True)
        heat_hvac = 1000 * heat_hvac_kw
        heat_hp = 1000 * heat_hp_kw
        heat_er = 1000 * heat_er_kw
        indoor_temperature = (model["output_matrix"] @ state[:, 1:])[0, :]
        constraints = [
            state[:, 0] == initial_state,
            heat_hvac_kw == heat_hp_kw + heat_er_kw,
            cp.multiply(1 / cop_values, heat_hp_kw) <= power_limit_hp_w / 1000,
            heat_er_kw <= heat_limit_er_w / 1000,
        ]
        degree_hour_mode = self.enforce_degree_hour_budget
        if degree_hour_mode:
            comfort_cost: cp.Expression = cp.Constant(0.0)
        else:
            slack = cp.Variable(steps)
            constraints.extend([
                lower <= indoor_temperature + slack,
                indoor_temperature + slack <= upper,
            ])
            comfort_cost = comfort_weight * cp.norm2(slack)
        degree_hour_constraints, degree_hour_cost, cap_cost = self._degree_hour_terms(
            indoor_temperature, lower, upper, steps,
            timestep_hours, historical_degree_hours
        )
        constraints.extend(degree_hour_constraints)
        for step in range(steps):
            constraints.append(
                state[:, step + 1]
                == model["A"] @ state[:, step]
                + model["B"][:, 0] * heat_hvac[step]
                + model["E"] @ disturbances[:, step]
            )

        electricity_cost = cp.sum(
            cp.multiply(
                prices,
                (cp.multiply(1 / cop_values, heat_hp_kw) + heat_er_kw)
                * timestep_hours,
            )
        ) / steps

        heat_hp_variation = cp.abs(heat_hp_kw[0] - previous_heat_hp_kw)
        if steps > 1:
            heat_hp_variation += cp.norm1(heat_hp_kw[1:] - heat_hp_kw[:-1])
        persistence_cost = (
            self.persistence_weight / steps * timestep_hours * heat_hp_variation
        )

        self._objective_component_expressions = {
            "electricity_objective_cost": electricity_cost,
            "comfort_objective_cost": comfort_cost,
            "persistence_objective_cost": persistence_cost,
            "degree_hour_objective_cost": degree_hour_cost,
            "max_deviation_objective_cost": cap_cost,
        }
        problem = cp.Problem(
            cp.Minimize(
                electricity_cost + comfort_cost + persistence_cost
                + degree_hour_cost + cap_cost
            ),
            constraints,
        )
        return problem, heat_hvac, heat_hp, heat_er

    def optimization_problem_mixed_int(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any] | None = None,
        *,
        power_limit_hp_w: float,
        heat_limit_er_w: float,
        er_stage_capacities_w: np.ndarray,
        temperature_lower_c: np.ndarray,
        temperature_upper_c: np.ndarray,
        cop: float | np.ndarray,
        previous_heat_hp_kw: float,
        comfort_weight: float = 1.0,
        historical_degree_hours: float | np.ndarray | None = None,
        prior_on: int | None = None,
        recent_cycles: int = 0,
        prior_dwell_steps: int = 0,
    ) -> tuple[cp.Problem, cp.Expression, cp.Expression, cp.Expression]:
        """Run the HP at its modeled maximum and select OCHRE's ER stages."""
        model = model or self._load_model()
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        prices = np.asarray(prices, dtype=float).reshape(-1)
        disturbances = np.asarray(disturbances, dtype=float)
        lower = np.asarray(temperature_lower_c, dtype=float).reshape(-1)
        upper = np.asarray(temperature_upper_c, dtype=float).reshape(-1)
        previous_heat_hp_kw = float(previous_heat_hp_kw)
        steps = prices.size
        if self.enforce_degree_hour_budget and steps != self.horizon // self.model_timestep:
            raise ValueError("HVAC degree-hour budget needs the complete configured horizon")
        cop_values = np.asarray(cop, dtype=float)
        if cop_values.ndim == 0:
            cop_values = np.full(steps, float(cop_values))
        else:
            cop_values = cop_values.reshape(-1)
        er_capacities = np.asarray(er_stage_capacities_w, dtype=float).reshape(-1)
        if (
            steps == 0
            or disturbances.shape != (model["E"].shape[1], steps)
            or initial_state.shape != (model["A"].shape[0],)
            or lower.shape != (steps,)
            or upper.shape != (steps,)
            or cop_values.shape != (steps,)
            or not np.isfinite(initial_state).all()
            or not np.isfinite(prices).all()
            or not np.isfinite(disturbances).all()
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or not np.isfinite(cop_values).all()
            or er_capacities.size == 0
            or not np.isfinite(er_capacities).all()
            or er_capacities[0] != 0
            or np.any(np.diff(er_capacities) <= 0)
            or np.any(lower > upper)
            or not np.isfinite([
                power_limit_hp_w, heat_limit_er_w, comfort_weight, previous_heat_hp_kw
            ]).all()
            or power_limit_hp_w <= 0
            or heat_limit_er_w < 0
            or np.any(cop_values <= 0)
            or comfort_weight < 0
        ):
            raise ValueError("HVAC state, forecast, bounds, or equipment limits are invalid")
        track_transitions = self.enforce_cycle_limit or self.enforce_dwell_constraints
        if (track_transitions or self.persistence_weight > 0) and (
            prior_on not in (0, 1) or isinstance(prior_on, bool)
        ):
            raise ValueError("HVAC prior HP state is invalid")
        if track_transitions and (
            isinstance(recent_cycles, bool)
            or not isinstance(recent_cycles, (int, np.integer))
            or recent_cycles < 0
            or isinstance(prior_dwell_steps, bool)
            or not isinstance(prior_dwell_steps, (int, np.integer))
            or prior_dwell_steps < 0
        ):
            raise ValueError("HVAC prior HP state and transition history are invalid")

        timestep_hours = self.model_timestep.total_seconds() / 3600
        state = cp.Variable((model["A"].shape[0], steps + 1))
        # Solve in kW for conditioning while the trained B matrix receives W.
        heat_hp_state = cp.Variable(steps, boolean=True)
        heat_hvac_kw = cp.Variable(steps, nonneg=True)
        heat_hp_kw = cp.Variable(steps, nonneg=True)
        er_stages = er_capacities.size - 1

        constraints = []
        if er_stages == 0:
            heat_er_state = cp.Constant(np.zeros(steps))
            heat_er_kw = cp.Constant(np.zeros(steps))
            er_constraints: list[cp.Constraint] = []
        elif np.allclose(
            np.diff(er_capacities), er_capacities[-1] / er_stages,
            rtol=1e-9, atol=1e-6,
        ):
            # Equal cumulative increments need only one integer per interval.
            heat_er_state = cp.Variable(steps, integer=True)
            heat_er_kw = er_capacities[-1] / (1000 * er_stages) * heat_er_state
            er_constraints = [heat_er_state >= 0, heat_er_state <= er_stages]
            constraints.append(
                heat_hp_state >= heat_er_state/er_stages
            )
        else:
            # A one-hot choice preserves unequal OCHRE cumulative stage sizes.
            stage_choice = cp.Variable((er_stages + 1, steps), boolean=True)
            heat_er_state = cp.sum(
                cp.multiply(np.arange(er_stages + 1)[:, None], stage_choice), axis=0
            )
            heat_er_kw = cp.sum(
                cp.multiply(er_capacities[:, None] / 1000, stage_choice), axis=0
            )
            er_constraints = [cp.sum(stage_choice, axis=0) == 1]
            constraints.append(
                heat_hp_state >= heat_er_state/er_stages
            )
        heat_hvac = 1000 * heat_hvac_kw
        heat_hp = 1000 * heat_hp_kw
        heat_er = 1000 * heat_er_kw
        indoor_temperature = (model["output_matrix"] @ state[:, 1:])[0, :]
        constraints.extend([
            state[:, 0] == initial_state,
            heat_hvac_kw == heat_hp_kw + heat_er_kw,
            heat_hp_kw == cp.multiply(
                cop_values * (power_limit_hp_w / 1000), heat_hp_state
            ),
            heat_er_kw <= heat_limit_er_w / 1000,
            *er_constraints,
        ])
        degree_hour_mode = self.enforce_degree_hour_budget
        if degree_hour_mode:
            comfort_cost: cp.Expression = cp.Constant(0.0)
        else:
            slack = cp.Variable(steps)
            constraints.extend([
                lower <= indoor_temperature + slack,
                indoor_temperature + slack <= upper,
            ])
            comfort_cost = comfort_weight * cp.norm2(slack)
        degree_hour_constraints, degree_hour_cost, cap_cost = self._degree_hour_terms(
            indoor_temperature, lower, upper, steps,
            timestep_hours, historical_degree_hours
        )
        constraints.extend(degree_hour_constraints)
        heat_on = heat_off = None
        if track_transitions:
            heat_on = cp.Variable(steps, boolean=True)
            heat_off = cp.Variable(steps, boolean=True)
            previous_heat_state = cp.hstack([prior_on, heat_hp_state[:-1]])
            constraints.extend([
                heat_hp_state - previous_heat_state == heat_on - heat_off,
                heat_on + heat_off <= 1,
            ])
            if self.enforce_cycle_limit:
                constraints.append(cp.sum(heat_on) + recent_cycles <= self.max_cycles)
        if self.enforce_dwell_constraints and prior_dwell_steps:
            constraints.append(
                heat_hp_state[:min(prior_dwell_steps, steps)] == prior_on
            )
        if self.enforce_dwell_constraints:
            assert heat_on is not None and heat_off is not None
            dwell_steps_on = int(np.ceil(self.dwell_time_on / self.model_timestep))
            dwell_steps_off = int(np.ceil(self.dwell_time_off / self.model_timestep))
            for step in range(steps):
                on_window = min(dwell_steps_on, steps - step)
                off_window = min(dwell_steps_off, steps - step)
                constraints.extend([
                    cp.sum(heat_hp_state[step:step + on_window])
                    >= on_window * heat_on[step],
                    cp.sum(1 - heat_hp_state[step:step + off_window])
                    >= off_window * heat_off[step],
                ])
        for step in range(steps):
            constraints.append(
                state[:, step + 1]
                == model["A"] @ state[:, step]
                + model["B"][:, 0] * heat_hvac[step]
                + model["E"] @ disturbances[:, step]
            )

        electricity_cost = cp.sum(
            cp.multiply(
                prices,
                (cp.multiply(1 / cop_values, heat_hp_kw) + heat_er_kw)
                * timestep_hours,
            )
        ) / steps

        if self.persistence_weight > 0:
            hp_state_variation = cp.abs(heat_hp_state[0] - prior_on)
            if steps > 1:
                hp_state_variation += cp.norm1(
                    heat_hp_state[1:] - heat_hp_state[:-1]
                )
            persistence_cost = (
                self.persistence_weight / steps * timestep_hours * hp_state_variation
            )
        else:
            persistence_cost = cp.Constant(0.0)

        self._mixed_int_state_expressions = {
            "hp_on": heat_hp_state,
            "er_stage": heat_er_state,
        }
        if heat_on is not None and heat_off is not None:
            self._mixed_int_state_expressions.update({
                "hp_start": heat_on,
                "hp_stop": heat_off,
            })
        self._objective_component_expressions = {
            "electricity_objective_cost": electricity_cost,
            "comfort_objective_cost": comfort_cost,
            "persistence_objective_cost": persistence_cost,
            "degree_hour_objective_cost": degree_hour_cost,
            "max_deviation_objective_cost": cap_cost,
        }
        problem = cp.Problem(
            cp.Minimize(
                electricity_cost + comfort_cost + persistence_cost
                + degree_hour_cost + cap_cost
            ),
            constraints,
        )
        return problem, heat_hvac, heat_hp, heat_er

    def _record_objective_components(
        self,
        timestamp: pd.Timestamp,
        problem: cp.Problem | None,
        *,
        expressions: Mapping[str, cp.Expression] | None = None,
    ) -> None:
        values = {
            name: float(np.asarray(expression.value).squeeze())
            for name, expression in (
                self._objective_component_expressions if expressions is None else expressions
            ).items()
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("Solved HVAC objective has a nonfinite component")
        component_sum = sum(values.values())
        scalar_objective = component_sum if problem is None else float(problem.value)
        self._objective_component_history.append({
            "timestamp": pd.Timestamp(timestamp),
            **values,
            "scalar_objective_cost": scalar_objective,
            "objective_balance_error": scalar_objective - component_sum,
        })

    def _record_comfort_budget(
        self,
        timestamp: pd.Timestamp,
        historical_degree_hours: float | np.ndarray | None,
        *,
        expressions: Mapping[str, cp.Expression] | None = None,
    ) -> None:
        if not (self.enforce_degree_hour_budget or self.enforce_instantaneous_cap):
            return
        expressions = self._comfort_budget_expressions if expressions is None else expressions
        violation = np.asarray(
            expressions["predicted_violation_c"].value, dtype=float
        ).reshape(-1)
        predicted = float(expressions["predicted_degree_hours_c_hours"].value)
        budget_slack = float(expressions["degree_hour_slack_c_hours"].value)
        cap_slack = np.asarray(
            expressions["maximum_deviation_slack_c"].value, dtype=float
        ).reshape(-1)
        if (
            not np.isfinite(violation).all()
            or not np.isfinite(cap_slack).all()
            or not np.isfinite([predicted, budget_slack]).all()
        ):
            raise RuntimeError("Solved HVAC comfort budget has nonfinite diagnostics")
        initial_history = (
            None if historical_degree_hours is None else
            float(np.asarray(historical_degree_hours, dtype=float).reshape(-1)[0])
        )
        rolling_diagnostics: dict[str, Any] = {}
        if self.full_horizon_rolling_budget:
            ends = np.asarray(
                expressions["rolling_24h_window_end_step"].value, dtype=int,
            ).reshape(-1)
            historical = np.asarray(
                expressions["rolling_24h_historical_degree_hours_c_hours"].value,
                dtype=float,
            ).reshape(-1)
            predicted_by_window = np.asarray(
                expressions["rolling_24h_predicted_degree_hours_c_hours"].value,
                dtype=float,
            ).reshape(-1)
            totals = np.asarray(
                expressions["rolling_24h_degree_hours_c_hours"].value, dtype=float,
            ).reshape(-1)
            if (not np.isfinite(np.r_[historical, predicted_by_window, totals]).all()
                    or not (len(ends) == len(historical) == len(predicted_by_window) == len(totals))):
                raise RuntimeError("Solved HVAC rolling comfort windows have invalid diagnostics")
            window_times = [pd.Timestamp(timestamp) + int(end) * self.model_timestep for end in ends]
            worst = int(np.argmax(totals))
            rolling_diagnostics = {
                "full_horizon_rolling_budget": True,
                "rolling_24h_window_count": len(totals),
                "rolling_24h_window_end_times": [time.isoformat() for time in window_times],
                "rolling_24h_historical_degree_hours_c_hours": historical.tolist(),
                "rolling_24h_predicted_degree_hours_c_hours": predicted_by_window.tolist(),
                "rolling_24h_degree_hours_c_hours": totals.tolist(),
                "maximum_rolling_24h_degree_hours_c_hours": float(totals[worst]),
                "worst_rolling_24h_window_end": window_times[worst],
            }
        self._comfort_budget_history.append({
            "timestamp": pd.Timestamp(timestamp),
            "historical_degree_hours_c_hours": initial_history,
            "predicted_degree_hours_c_hours": predicted,
            "remaining_degree_hours_c_hours": (
                self.degree_hour_limit_c_hours - initial_history
                if initial_history is not None else None
            ),
            "degree_hour_slack_c_hours": budget_slack,
            "maximum_predicted_violation_c": float(violation.max()),
            "maximum_deviation_slack_c": float(cap_slack.max()),
            "total_maximum_deviation_slack_c": float(cap_slack.sum()),
            "history_coverage_hours": (
                self._comfort_history_window.total_seconds() / 3600
                if historical_degree_hours is not None else None
            ),
            **rolling_diagnostics,
        })

    @staticmethod
    def _finite_gurobi_attribute(model: Any, name: str) -> float | None:
        try:
            value = float(getattr(model, name))
        except (AttributeError, TypeError, ValueError):
            return None
        except Exception:  # Gurobi raises when an attribute is unavailable.
            return None
        return value if np.isfinite(value) and abs(value) < 1e99 else None

    def _solver_metrics(self, problem: cp.Problem | None) -> dict[str, Any]:
        stats = problem.solver_stats if problem is not None else None
        model = (
            stats.extra_stats
            if stats is not None and str(self.solver).upper() == "GUROBI"
            else None
        )
        status_code = (
            self._finite_gurobi_attribute(model, "Status") if model is not None else None
        )
        sol_count = (
            self._finite_gurobi_attribute(model, "SolCount") if model is not None else None
        )
        incumbent = sol_count is not None and sol_count > 0
        objective = (
            self._finite_gurobi_attribute(model, "ObjVal") if incumbent else None
        )
        bound = (
            self._finite_gurobi_attribute(model, "ObjBound")
            if model is not None and self.optimization_kind == "mixed integer"
            else None
        )
        gap = (
            self._finite_gurobi_attribute(model, "MIPGap")
            if incumbent and self.optimization_kind == "mixed integer"
            else None
        )
        solver_time = None if stats is None else stats.solve_time
        if solver_time is None and model is not None:
            solver_time = self._finite_gurobi_attribute(model, "Runtime")
        return {
            "solver_time_s": solver_time,
            "solver_status_code": int(status_code) if status_code is not None else None,
            "time_limit_hit": status_code == 9,
            "sol_count": int(sol_count) if sol_count is not None else None,
            "incumbent_available": incumbent,
            "objective_value": objective,
            "objective_bound": bound,
            "relative_gap": gap,
            "absolute_gap": abs(objective - bound) if objective is not None and bound is not None else None,
        }

    @staticmethod
    def _time_limited_solution_is_feasible(problem: cp.Problem) -> bool:
        """Check the returned primal point before using a time-limited incumbent."""
        if problem.value is None or not np.isfinite(problem.value):
            return False
        for variable in problem.variables():
            value = variable.value
            if value is None or not np.isfinite(value).all():
                return False
            if variable.attributes["boolean"] or variable.attributes["integer"]:
                if np.max(np.abs(value - np.rint(value))) > 1e-4:
                    return False
        for constraint in problem.constraints:
            try:
                violation = constraint.violation()
            except ValueError:
                return False
            if violation is None or not np.isfinite(violation).all():
                return False
            if np.max(np.abs(violation)) > 1e-3:
                return False
        return True

    def _record_solve_attempt(
        self, timestamp: pd.Timestamp, duration_s: float, status: str, successful: bool,
        *, problem: cp.Problem | None = None, metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if metrics is None:
            metrics = self._solver_metrics(problem)
        self.solve_durations_s[self.optimization_kind].append(float(duration_s))
        self._solve_history.append({
            "timestamp": pd.Timestamp(timestamp),
            "optimization_kind": self.optimization_kind,
            "duration_s": float(duration_s),
            "status": status,
            "successful": bool(successful),
            **metrics,
        })

    def _record_optimization_plan(
        self,
        timestamp: pd.Timestamp,
        initial_state: np.ndarray,
        heat_hvac: np.ndarray,
        heat_hp: np.ndarray,
        heat_er: np.ndarray,
        prices: np.ndarray,
        cop: np.ndarray,
        disturbances: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        model: Mapping[str, Any],
        mixed_state: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        heat_hvac = np.asarray(heat_hvac, dtype=float).reshape(-1)
        heat_hp = np.asarray(heat_hp, dtype=float).reshape(-1)
        heat_er = np.asarray(heat_er, dtype=float).reshape(-1)
        states = [np.asarray(initial_state, dtype=float).reshape(-1)]
        for step, heat in enumerate(heat_hvac):
            states.append(
                model["A"] @ states[-1]
                + model["B"][:, 0] * heat
                + model["E"] @ disturbances[:, step]
            )
        temperatures = (model["output_matrix"] @ np.column_stack(states))[0]
        index = pd.date_range(timestamp, periods=len(states), freq=self.model_timestep)
        plan = pd.DataFrame(index=index)
        plan.index.name = "timestamp"
        plan["indoor_temperature_c"] = temperatures
        initial_lower, initial_upper = self._comfort_bounds(index[:1])
        for name, values in (
            ("hvac_heat_w", heat_hvac),
            ("hp_heat_w", heat_hp),
            ("er_heat_w", heat_er),
            ("electric_power_kw", (heat_hp / cop + heat_er) / 1000),
            ("hp_cop", cop),
            (PRICE_COLUMN, prices),
        ):
            plan[name] = np.append(values, values[-1])
        plan["comfort_lower_c"] = np.insert(lower, 0, initial_lower[0])
        plan["comfort_upper_c"] = np.insert(upper, 0, initial_upper[0])
        if mixed_state is not None:
            for name, values in mixed_state.items():
                values = np.rint(np.asarray(values, dtype=float).reshape(-1)).astype(int)
                plan[name] = np.append(values, values[-1])
        if not self.enforce_degree_hour_budget:
            plan["comfort_slack_c"] = np.insert(
                np.clip(lower - temperatures[1:], 0, None)
                - np.clip(temperatures[1:] - upper, 0, None),
                0, 0.0,
            )
        violation = np.maximum(temperatures[1:] - upper, 0) + np.maximum(
            lower - temperatures[1:], 0
        )
        plan["comfort_violation_c"] = np.insert(violation, 0, 0.0)
        plan["comfort_degree_hours_c_hours"] = np.insert(
            violation * self.model_timestep.total_seconds() / 3600, 0, 0.0
        )
        if (self.full_horizon_rolling_budget
                and "rolling_24h_window_end_step" in self._comfort_budget_expressions):
            expressions = self._comfort_budget_expressions
            window_ends = np.asarray(
                expressions["rolling_24h_window_end_step"].value, dtype=int,
            ).reshape(-1)
            for name in (
                "rolling_24h_historical_degree_hours_c_hours",
                "rolling_24h_predicted_degree_hours_c_hours",
                "rolling_24h_degree_hours_c_hours",
            ):
                values = np.full(len(plan), np.nan)
                values[window_ends] = np.asarray(expressions[name].value, dtype=float).reshape(-1)
                plan[name] = values
            plan.attrs.update(
                full_horizon_rolling_budget=True,
                comfort_budget_window_hours=24,
                comfort_budget_first_endpoint_hours=8,
                shared_degree_hour_slack_c_hours=float(expressions["degree_hour_slack_c_hours"].value),
            )
        plan.attrs["solved_at"] = pd.Timestamp(timestamp)
        self._optimization_plans.append(plan)

    def prepare_mixed_int_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> PreparedHVACMixedIntegerStep | None:
        """Observe one plant step and build the existing HVAC MILP when due."""
        if self.optimization_kind != "mixed integer":
            raise ValueError("HVAC mixed-integer preparation requires mixed integer optimization")
        if previous_status is None:
            return None
        observed_hp_on = self._observe_hp_state(timestamp, previous_status)
        model = self._load_model()
        self._observe_step(timestamp, previous_status, forecast)
        timestamp = pd.Timestamp(timestamp)
        if not self._is_model_step_boundary(timestamp):
            return None
        if self._last_decision_time == timestamp:
            return None
        if not self._complete_history_interval(timestamp, forecast, model):
            return None
        if len(self._history) < model["m"]:
            return None
        index = self._planning_index(timestamp)
        disturbances = self._disturbances_for_intervals(
            timestamp, len(index), forecast, model, future_weather=True
        )
        prices = self._forecast_prices(index, forecast)
        limits = self._power_and_er_limits_w(previous_status)
        if disturbances is None or prices is None or limits is None:
            return None
        outdoor_name = "outdoor_temperature_c"
        if outdoor_name not in model["disturbance_names"]:
            raise ValueError("HVAC model lacks the outdoor-temperature disturbance")
        setpoints = self._training_setpoints(index)
        outdoor = disturbances[model["disturbance_names"].index(outdoor_name)]
        cop = self._forecast_cop(outdoor, setpoints)
        if cop is None:
            return None
        initial_state = self._lifted_state(model["m"])
        lower, upper = self._comfort_bounds(index + self.model_timestep)
        historical_degree_hours = None
        if self.enforce_degree_hour_budget:
            historical_degree_hours = self._historical_degree_hours(timestamp)
            if historical_degree_hours is None:
                raise RuntimeError(
                    "HVAC degree-hour budget requires a complete "
                    f"{self._comfort_history_window.total_seconds() / 3600:g}-hour "
                    "measured-temperature warm start"
                )
        equipment = self._heating_equipment
        if equipment is None or not hasattr(equipment, "er_capacity_list"):
            return None
        extra_options: dict[str, Any] = {
            "er_stage_capacities_w": equipment.er_capacity_list,
        }
        prior_on = observed_hp_on
        if prior_on is None and (
            self.persistence_weight > 0
            or self.enforce_cycle_limit
            or self.enforce_dwell_constraints
        ):
            return None
        if prior_on is not None:
            extra_options["prior_on"] = prior_on
        if self.enforce_cycle_limit or self.enforce_dwell_constraints:
            extra_options.update({
                "recent_cycles": min(1, len(self._recent_hp_off_times)),
                "prior_dwell_steps": self._remaining_hp_dwell_steps(
                    timestamp, prior_on
                ),
            })
        problem, heat_hvac, heat_hp, heat_er = self.optimization_problem_mixed_int(
            initial_state,
            prices,
            disturbances,
            model,
            power_limit_hp_w=limits[0],
            heat_limit_er_w=limits[1],
            temperature_lower_c=lower,
            temperature_upper_c=upper,
            cop=cop,
            previous_heat_hp_kw=0.0,
            comfort_weight=self.comfort_weight,
            historical_degree_hours=historical_degree_hours,
            **extra_options,
        )
        return PreparedHVACMixedIntegerStep(
            timestamp=timestamp,
            previous_status=dict(previous_status),
            problem=problem,
            heat_hvac=heat_hvac,
            heat_hp=heat_hp,
            heat_er=heat_er,
            prices=prices,
            cop=cop,
            initial_state=initial_state,
            disturbances=disturbances,
            lower=lower,
            upper=upper,
            model=model,
            setpoints=setpoints,
            historical_degree_hours=historical_degree_hours,
            objective_components=MappingProxyType(dict(self._objective_component_expressions)),
            comfort_budget_expressions=MappingProxyType(dict(self._comfort_budget_expressions)),
            mixed_int_state_expressions=MappingProxyType(dict(self._mixed_int_state_expressions)),
        )

    def held_controls_for_step(
        self, previous_status: Mapping[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        """Continue the most recent thermostat request between optimizer solves."""
        if previous_status is None:
            return {}
        return self._setpoint_control(previous_status)

    def apply_mixed_int_solution(
        self,
        prepared: PreparedHVACMixedIntegerStep,
        *,
        standalone_problem: cp.Problem | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Record a solved HVAC plan and command its first interval."""
        if self.optimization_kind != "mixed integer":
            raise ValueError("HVAC mixed-integer application requires mixed integer optimization")
        if any(
            expression.value is None or not np.isfinite(expression.value).all()
            for expression in (
                prepared.heat_hvac, prepared.heat_hp, prepared.heat_er,
                *prepared.mixed_int_state_expressions.values(),
            )
        ):
            raise ValueError("HVAC mixed-integer solution has missing or nonfinite values")
        self._record_objective_components(
            prepared.timestamp,
            standalone_problem,
            expressions=prepared.objective_components,
        )
        self._record_comfort_budget(
            prepared.timestamp,
            prepared.historical_degree_hours,
            expressions=prepared.comfort_budget_expressions,
        )
        mixed_state = {
            name: np.asarray(expression.value)
            for name, expression in prepared.mixed_int_state_expressions.items()
        }
        self._record_optimization_plan(
            prepared.timestamp,
            prepared.initial_state,
            prepared.heat_hvac.value,
            prepared.heat_hp.value,
            prepared.heat_er.value,
            prepared.prices,
            prepared.cop,
            prepared.disturbances,
            prepared.lower,
            prepared.upper,
            prepared.model,
            mixed_state,
        )
        next_temperature = float(
            self._optimization_plans[-1]["indoor_temperature_c"].iloc[1]
        )
        self._start_setpoint_interval(
            prepared.previous_status,
            planned_heat_w=float(prepared.heat_hvac.value[0]),
            planned_hp_heat_w=float(prepared.heat_hp.value[0]),
            planned_er_heat_w=float(prepared.heat_er.value[0]),
            training_setpoint_c=float(prepared.setpoints[0]),
            predicted_next_temperature_c=next_temperature,
        )
        self._last_decision_time = prepared.timestamp
        return self._setpoint_control(prepared.previous_status)

    def _solve_prepared_mixed_int_step(
        self, prepared: PreparedHVACMixedIntegerStep
    ) -> dict[str, dict[str, Any]]:
        """Keep the standalone HVAC solve and its diagnostics unchanged."""
        problem = prepared.problem
        solve_start = time.perf_counter()
        try:
            solver_options = dict(self.solver_options)
            if self.solver == "GUROBI":
                solver_options["Threads"] = 1
            problem.solve(solver=self.solver, **solver_options)
        except cp.error.SolverError:
            self._record_solve_attempt(
                prepared.timestamp,
                time.perf_counter() - solve_start,
                "solver_error",
                False,
            )
            return self.held_controls_for_step(prepared.previous_status)
        duration = time.perf_counter() - solve_start
        metrics = self._solver_metrics(problem)
        time_limited_incumbent = (
            problem.status == cp.USER_LIMIT
            and metrics["time_limit_hit"]
            and metrics["incumbent_available"]
            and self._time_limited_solution_is_feasible(problem)
        )
        successful = (
            (problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
             or time_limited_incumbent)
            and all(
                expression.value is not None
                and np.isfinite(expression.value).all()
                for expression in (
                    prepared.heat_hvac, prepared.heat_hp, prepared.heat_er,
                    *prepared.mixed_int_state_expressions.values(),
                )
            )
        )
        self._record_solve_attempt(
            prepared.timestamp,
            duration,
            str(problem.status),
            successful,
            problem=problem,
            metrics=metrics,
        )
        if not successful:
            return self.held_controls_for_step(prepared.previous_status)
        return self.apply_mixed_int_solution(
            prepared, standalone_problem=problem
        )

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Update measured history and issue the latest heating setpoint."""
        if previous_status is None:
            return {}
        if self.optimization_kind == "mixed integer":
            prepared = self.prepare_mixed_int_step(
                timestamp, previous_status, current_schedule, forecast
            )
            if prepared is None:
                return self.held_controls_for_step(previous_status)
            return self._solve_prepared_mixed_int_step(prepared)
        model = self._load_model()
        self._observe_step(timestamp, previous_status, forecast)
        if not self._is_model_step_boundary(timestamp):
            return self._setpoint_control(previous_status)
        if self._last_decision_time == pd.Timestamp(timestamp):
            return self._setpoint_control(previous_status)
        if not self._complete_history_interval(timestamp, forecast, model):
            return self._setpoint_control(previous_status)
        if len(self._history) < model["m"]:
            return self._setpoint_control(previous_status)
        index = self._planning_index(timestamp)
        disturbances = self._disturbances_for_intervals(
            timestamp, len(index), forecast, model, future_weather=True
        )
        prices = self._forecast_prices(index, forecast)
        limits = self._power_and_er_limits_w(previous_status)
        previous_hp_heat_w = self._observed_hp_coil_heat_w()
        if (
            disturbances is None or prices is None or limits is None
            or previous_hp_heat_w is None
        ):
            return self._setpoint_control(previous_status)
        outdoor_name = "outdoor_temperature_c"
        if outdoor_name not in model["disturbance_names"]:
            raise ValueError("HVAC model lacks the outdoor-temperature disturbance")
        setpoints = self._training_setpoints(index)
        outdoor = disturbances[model["disturbance_names"].index(outdoor_name)]
        cop = self._forecast_cop(outdoor, setpoints)
        if cop is None:
            return self._setpoint_control(previous_status)
        initial_state = self._lifted_state(model["m"])
        lower, upper = self._comfort_bounds(index + self.model_timestep)
        historical_degree_hours = None
        if self.enforce_degree_hour_budget:
            historical_degree_hours = self._historical_degree_hours(timestamp)
            if historical_degree_hours is None:
                raise RuntimeError(
                    "HVAC degree-hour budget requires a complete "
                    f"{self._comfort_history_window.total_seconds() / 3600:g}-hour "
                    "measured-temperature warm start"
                )

        problem, heat_hvac, heat_hp, heat_er = self.optimization_problem_convex(
            initial_state,
            prices,
            disturbances,
            model,
            power_limit_hp_w=limits[0],
            heat_limit_er_w=limits[1],
            temperature_lower_c=lower,
            temperature_upper_c=upper,
            cop=cop,
            previous_heat_hp_kw=previous_hp_heat_w / 1000,
            comfort_weight=self.comfort_weight,
            historical_degree_hours=historical_degree_hours,
        )
        solve_start = time.perf_counter()
        try:
            solver_options = dict(self.solver_options)
            if self.solver == "GUROBI":
                solver_options["Threads"] = 1
            problem.solve(solver=self.solver, **solver_options)
        except cp.error.SolverError:
            self._record_solve_attempt(
                timestamp, time.perf_counter() - solve_start, "solver_error", False
            )
            return self._setpoint_control(previous_status)
        duration = time.perf_counter() - solve_start
        metrics = self._solver_metrics(problem)
        time_limited_incumbent = (
            problem.status == cp.USER_LIMIT
            and metrics["time_limit_hit"]
            and metrics["incumbent_available"]
            and self._time_limited_solution_is_feasible(problem)
        )
        successful = (
            (problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
             or time_limited_incumbent)
            and all(
                variable.value is not None and np.isfinite(variable.value).all()
                for variable in (heat_hvac, heat_hp, heat_er)
            )
        )
        self._record_solve_attempt(
            timestamp, duration, str(problem.status), successful,
            problem=problem, metrics=metrics,
        )
        if not successful:
            return self._setpoint_control(previous_status)

        self._record_objective_components(timestamp, problem)
        self._record_comfort_budget(timestamp, historical_degree_hours)
        self._record_optimization_plan(
            timestamp, initial_state, heat_hvac.value, heat_hp.value, heat_er.value,
            prices, cop, disturbances, lower, upper, model,
        )
        next_temperature = float(
            self._optimization_plans[-1]["indoor_temperature_c"].iloc[1]
        )
        self._start_setpoint_interval(
            previous_status,
            planned_heat_w=float(heat_hvac.value[0]),
            planned_hp_heat_w=float(heat_hp.value[0]),
            planned_er_heat_w=float(heat_er.value[0]),
            training_setpoint_c=float(setpoints[0]),
            predicted_next_temperature_c=next_temperature,
        )
        self._last_decision_time = pd.Timestamp(timestamp)
        return self._setpoint_control(previous_status)

    @staticmethod
    def _boundary_temperature(
        previous_status: Mapping[str, Any] | None,
    ) -> float | None:
        observed = (
            previous_status.get(HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY)
            if previous_status is not None else None
        )
        try:
            temperature = float(observed)
        except (TypeError, ValueError):
            return None
        return temperature if np.isfinite(temperature) else None

    def _observed_er_on(self, previous_status: Mapping[str, Any]) -> bool | None:
        er_power = previous_status.get(f"{self.heating_name} ER Power (kW)")
        if er_power is not None:
            try:
                power = float(er_power)
            except (TypeError, ValueError):
                power = np.nan
            if np.isfinite(power):
                return power > 0.0
        mode = previous_status.get(f"{self.heating_name} Mode")
        if mode in {"ER On", "HP and ER On"}:
            return True
        if mode in {"Off", "HP On"}:
            return False
        return None

    def _running_defrost_at_boundary(
        self, previous_status: Mapping[str, Any]
    ) -> bool:
        """Read a pending cycle with a running compressor, not a paused timer.

        Defrost remaining is reported after the completed minute advances its
        timer. A positive active fraction with zero remaining therefore does
        not defer a reset in the next minute: that cycle has just finished.
        """
        try:
            remaining = float(previous_status.get(
                f"{self.heating_name} Defrost Remaining (minutes)", 0.0
            ))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(remaining) or remaining <= 0.0:
            return False
        try:
            power = float(previous_status.get(
                f"{self.heating_name} Main Power (kW)"
            ))
        except (TypeError, ValueError):
            power = np.nan
        if np.isfinite(power):
            return power > 0.0
        return previous_status.get(f"{self.heating_name} Mode") in {
            "HP On", "HP and ER On"
        }

    def _defrost_active_in_feedback(
        self, previous_status: Mapping[str, Any]
    ) -> bool:
        """Do not acknowledge normal ER-off using defrost-minute feedback."""
        for suffix in ("Defrost Active Fraction (-)", "Defrost Active (-)"):
            try:
                value = float(previous_status.get(f"{self.heating_name} {suffix}"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return True
        return self._running_defrost_at_boundary(previous_status)

    @staticmethod
    def _plant_feedback_time(previous_status: Mapping[str, Any]) -> pd.Timestamp | None:
        """Identify the completed plant minute whose feedback is being read."""
        try:
            timestamp = pd.Timestamp(previous_status.get("Time"))
        except (TypeError, ValueError):
            return None
        return None if pd.isna(timestamp) else timestamp

    def _hp_only_setpoint(
        self, temperature: float, previous_status: Mapping[str, Any] | None
    ) -> float:
        """Keep normal ER-off pending until fresh post-reset feedback confirms it.

        The existing Tin reset pulse only requests normal ER release; native
        compressor minimum times, defrost backup, and safety rules retain
        authority. A cycle can start during a pulse, so sending it is not an
        acknowledgment and a later eligible minute must retry if needed.
        """
        status = previous_status or {}
        er_on = self._observed_er_on(status)
        if er_on is True:
            self._hp_only_er_off_pending = True
        feedback_time = self._plant_feedback_time(status)
        fresh_after_pulse = False
        if feedback_time is not None and self._hp_only_er_reset_feedback_time is not None:
            try:
                fresh_after_pulse = feedback_time > self._hp_only_er_reset_feedback_time
            except TypeError:
                # Inconsistent clocks cannot acknowledge an actuator request.
                pass
        if self._hp_only_er_reset_attempted and fresh_after_pulse:
            if er_on is False and not self._defrost_active_in_feedback(status):
                self._hp_only_er_off_pending = False
            # A failed or defrost-overlapped pulse is eligible to retry. A
            # successful pulse clears the pending request through the branch
            # above, rather than consuming the request at command time.
            self._hp_only_er_reset_attempted = False
            self._hp_only_er_reset_feedback_time = None
        elif (
            self._hp_only_er_reset_attempted
            and self._hp_only_er_reset_feedback_time is None
            and feedback_time is not None
        ):
            # An un-timestamped sample cannot confirm the previous pulse.
            # Retry against this identifiable observation instead.
            self._hp_only_er_reset_attempted = False

        if (
            self._hp_only_command_sent
            and self._hp_only_er_off_pending
            and not self._hp_only_er_reset_attempted
            and not self._running_defrost_at_boundary(status)
        ):
            self._hp_only_er_reset_attempted = True
            self._hp_only_er_reset_feedback_time = feedback_time
            return temperature
        self._hp_only_command_sent = True
        return temperature + HP_ONLY_SETPOINT_OFFSET_C

    def _off_setpoint(self, temperature: float) -> float | None:
        """Request thermostat off with a margin beyond its turn-off threshold."""
        equipment = self._heating_equipment
        if equipment is None:
            return None
        try:
            deadband = float(equipment.temp_deadband)
            offset = float(equipment.deadband_offset)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            not np.isfinite([deadband, offset]).all()
            or deadband <= 0
            or not 0 <= offset <= 1
        ):
            return None
        return temperature - deadband * offset - OFF_SETPOINT_MARGIN_C

    def _start_setpoint_interval(
        self,
        previous_status: Mapping[str, Any],
        *,
        planned_heat_w: float,
        planned_hp_heat_w: float,
        planned_er_heat_w: float,
        training_setpoint_c: float,
        predicted_next_temperature_c: float,
    ) -> None:
        self._off_requested = planned_heat_w <= 10.0
        hp_only_requested = (
            planned_hp_heat_w > 10.0 and planned_er_heat_w <= 10.0
        )
        if not (self._hp_only_requested and hp_only_requested):
            self._hp_only_command_sent = False
            self._hp_only_er_off_pending = False
            self._hp_only_er_reset_attempted = False
            self._hp_only_er_reset_feedback_time = None
        self._hp_only_requested = hp_only_requested
        self._er_target_c = None
        # Retain the fixed mapping when the boundary temperature or thermostat
        # deadband is unavailable. Valid off, HP-only, and ER plans override it.
        self._last_setpoint = float(
            training_setpoint_c + self.comfort_band_c
            if planned_heat_w > 10.0
            else training_setpoint_c - self.comfort_band_c
        )
        start_temperature = self._boundary_temperature(previous_status)
        if self._off_requested and start_temperature is not None:
            off_setpoint = self._off_setpoint(start_temperature)
            if off_setpoint is not None:
                self._last_setpoint = off_setpoint
        elif (
            planned_er_heat_w > 10.0
            and start_temperature is not None
            and np.isfinite(predicted_next_temperature_c)
        ):
            self._last_setpoint = max(
                start_temperature + ER_CALL_SETPOINT_OFFSET_C,
                predicted_next_temperature_c,
            )
            if predicted_next_temperature_c > start_temperature:
                self._er_target_c = predicted_next_temperature_c

    def _setpoint_control(
        self, previous_status: Mapping[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        temperature = self._boundary_temperature(previous_status)
        if temperature is not None:
            if self._off_requested:
                off_setpoint = self._off_setpoint(temperature)
                if off_setpoint is not None:
                    self._last_setpoint = off_setpoint
            elif self._hp_only_requested:
                self._last_setpoint = self._hp_only_setpoint(temperature, previous_status)
            elif self._er_target_c is not None and temperature >= self._er_target_c:
                self._last_setpoint = self._er_target_c
                self._er_target_c = None
        if self._last_setpoint is None:
            return {}
        return {self.heating_name: {"Setpoint": self._last_setpoint}}

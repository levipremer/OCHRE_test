"""Linear-regression and capacitance-grid fitting for physical RC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear


DisturbanceKind = Literal["constant", "day_night"]


@dataclass(frozen=True)
class CoefficientSpec:
    """One physical coefficient in the linear heat-balance equations."""

    name: str
    lower: float = -np.inf
    upper: float = np.inf
    nonnegative: bool = False


@dataclass(frozen=True)
class PreparedData:
    indoor: np.ndarray
    ambient: np.ndarray
    heating: np.ndarray
    mass: np.ndarray
    solar: np.ndarray | None
    is_day: np.ndarray | None
    timestamps: np.ndarray
    available_current: np.ndarray
    heating_output_length: int


def day_mask_from_hours(
    hours: Sequence[float] | np.ndarray,
    day_start_hour: float = 7.0,
    night_start_hour: float = 19.0,
) -> np.ndarray:
    """Return a daytime mask, including schedules that cross midnight."""

    if not 0.0 <= day_start_hour < 24.0:
        raise ValueError("day_start_hour must be in [0, 24).")
    if not 0.0 <= night_start_hour < 24.0:
        raise ValueError("night_start_hour must be in [0, 24).")
    if day_start_hour == night_start_hour:
        raise ValueError("day_start_hour and night_start_hour must differ.")
    values = np.mod(np.asarray(hours, dtype=float), 24.0)
    if not np.all(np.isfinite(values)):
        raise ValueError("hours must contain only finite values.")
    if day_start_hour < night_start_hour:
        return (values >= day_start_hour) & (values < night_start_hour)
    return (values >= day_start_hour) | (values < night_start_hour)


def causal_moving_average(
    values: Sequence[float] | np.ndarray,
    window_steps: int,
    segment_ids: Sequence[Any] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute a trailing mean without crossing discontinuous data segments."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("values must be a finite one-dimensional array.")
    if not isinstance(window_steps, (int, np.integer)) or window_steps < 1:
        raise ValueError("window_steps must be a positive integer.")
    if segment_ids is None:
        segments = np.zeros(len(samples), dtype=int)
    else:
        segments = np.asarray(segment_ids)
        if segments.ndim != 1 or len(segments) != len(samples):
            raise ValueError("segment_ids must match values.")

    result = np.empty_like(samples)
    start = 0
    while start < len(samples):
        stop = start + 1
        while stop < len(samples) and segments[stop] == segments[start]:
            stop += 1
        block = samples[start:stop]
        cumulative = np.concatenate(([0.0], np.cumsum(block)))
        for local_index in range(len(block)):
            left = max(0, local_index - int(window_steps) + 1)
            result[start + local_index] = (
                cumulative[local_index + 1] - cumulative[left]
            ) / (local_index - left + 1)
        start = stop
    return result


class ForwardEulerJointFitter(ABC):
    """Base class for linear RC fits nested inside capacitance grid searches.

    For fixed capacitances, every unknown conductance and forcing coefficient
    enters the forward-Euler heat balances linearly and is estimated with
    bounded linear least squares. Capacitances are selected only from explicit
    grids. No initial guesses or nonlinear multistart optimization are used.
    """

    model_name = "RC model"
    state_names: tuple[str, ...] = ("indoor",)
    observed_state_names: tuple[str, ...] = ("indoor",)
    requires_mass_state = False
    mass_is_latent = False
    has_mass_ambient_path = False

    def __init__(
        self,
        sampling_minutes: float = 30.0,
        disturbance_model: DisturbanceKind = "constant",
        lambda_temperature: float = 0.5,
        indoor_capacitance_grid: Sequence[float] | np.ndarray | None = None,
        mass_capacitance_grid: Sequence[float] | np.ndarray | None = None,
        coefficient_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        disturbance_model = str(disturbance_model).lower()
        if not np.isfinite(sampling_minutes) or sampling_minutes <= 0:
            raise ValueError("sampling_minutes must be positive and finite.")
        if disturbance_model not in {"constant", "day_night"}:
            raise ValueError("disturbance_model must be 'constant' or 'day_night'.")
        if not np.isfinite(lambda_temperature) or not 0 <= lambda_temperature <= 1:
            raise ValueError("lambda_temperature must be in [0, 1].")

        self.sampling_minutes = float(sampling_minutes)
        self.dt_hours = self.sampling_minutes / 60.0
        self.disturbance_model: DisturbanceKind = disturbance_model  # type: ignore[assignment]
        self.lambda_temperature = float(lambda_temperature)
        self.indoor_capacitance_grid = self._validated_grid(
            indoor_capacitance_grid, "indoor_capacitance_grid"
        )
        self.mass_capacitance_grid = (
            self._validated_grid(mass_capacitance_grid, "mass_capacitance_grid")
            if self.requires_mass_state
            else None
        )
        self.coefficient_bounds = dict(coefficient_bounds or {})

        self.coefficient_names_: tuple[str, ...] | None = None
        self.coefficients_: np.ndarray | None = None
        self.linear_coefficient_names_: tuple[str, ...] | None = None
        self.linear_coefficients_: np.ndarray | None = None
        self.selected_capacitances_: dict[str, float] = {}
        self.physical_parameters_: dict[str, float] = {}
        self.temperature_predictions_: dict[str, np.ndarray] = {}
        self.rollout_predictions_: dict[str, dict[str, np.ndarray]] = {}
        self.rollout_evaluation_: dict[str, Any] = {}
        self.heating_input_predictions_: np.ndarray | None = None
        self.metrics_: dict[str, float] = {}
        self.split_metrics_: dict[str, dict[str, float]] = {}
        self.data_ranges_: dict[str, dict[str, str]] = {}
        self.optimizer_status_: dict[str, Any] = {}
        self.grid_search_: dict[str, Any] = {}
        self.stability_warnings_: list[str] = []
        self.uses_solar_: bool | None = None

    def fit(
        self,
        indoor_temperature: Sequence[float] | np.ndarray,
        ambient_temperature: Sequence[float] | np.ndarray,
        heating_input: Sequence[float] | np.ndarray,
        mass_temperature: Sequence[float] | np.ndarray | None = None,
        *,
        timestamps: Sequence[Any] | np.ndarray,
        training_range: tuple[Any, Any],
        validation_range: tuple[Any, Any],
        test_range: tuple[Any, Any],
        solar: Sequence[float] | np.ndarray | None = None,
        is_day: Sequence[bool] | np.ndarray | None = None,
        segment_ids: Sequence[Any] | np.ndarray | None = None,
    ) -> "ForwardEulerJointFitter":
        """Fit on training and evaluate validation/test without leakage."""

        data = self._prepare_data(
            indoor_temperature=indoor_temperature,
            ambient_temperature=ambient_temperature,
            heating_input=heating_input,
            mass_temperature=mass_temperature,
            timestamps=timestamps,
            solar=solar,
            is_day=is_day,
            segment_ids=segment_ids,
        )
        split_indices, range_report = self._split_transition_indices(
            data,
            training_range=training_range,
            validation_range=validation_range,
            test_range=test_range,
        )
        self._fit_and_finalize(data, split_indices, range_report)
        return self

    def _fit_and_finalize(
        self,
        data: PreparedData,
        split_indices: dict[str, np.ndarray],
        range_report: dict[str, dict[str, str]],
    ) -> None:
        training_indices = split_indices["training"]
        if data.is_day is not None and (
            np.all(data.is_day[training_indices])
            or not np.any(data.is_day[training_indices])
        ):
            raise ValueError(
                "The training range must contain daytime and nighttime transitions."
            )

        fit_record = self._fit_on_indices(data, training_indices)
        self.coefficients_ = fit_record["rate_parameters"]
        self.linear_coefficients_ = fit_record["linear_parameters"]
        self.selected_capacitances_ = dict(fit_record["capacitances"])
        self.grid_search_ = {
            "grid_shape": fit_record["grid_shape"],
            "selected_grid_index": fit_record["selected_grid_index"],
            "selected_at_grid_boundary": fit_record["selected_at_grid_boundary"],
            "candidate_results": fit_record["grid_records"],
        }
        self.physical_parameters_ = self._physical_parameters(
            self._coefficient_dict(self.coefficients_)
        )
        self.data_ranges_ = range_report

        temperature_normalizer = fit_record["temperature_normalizer"]
        heating_normalizer = fit_record["heating_normalizer"]
        self.split_metrics_ = {
            name: self._metrics_for_indices(
                self.coefficients_,
                data,
                indices,
                temperature_normalizer,
                heating_normalizer,
            )
            for name, indices in split_indices.items()
        }
        self._evaluate_rollouts(data, split_indices)
        self.metrics_ = dict(self.split_metrics_["training"])
        self.metrics_.update(
            {
                "temperature_normalizer": temperature_normalizer,
                "heating_input_normalizer": heating_normalizer,
                "temperature_only_rmse": fit_record["temperature_only_rmse"],
                "temperature_only_fit_heating_input_rmse": fit_record[
                    "temperature_only_fit_heating_rmse"
                ],
                "heating_only_fit_temperature_rmse": fit_record[
                    "heating_only_fit_temperature_rmse"
                ],
                "heating_only_rmse": fit_record["heating_only_rmse"],
            }
        )
        self.split_metrics_["training"].update(self.metrics_)

        selected = np.sort(np.concatenate(list(split_indices.values())))
        state_predictions, heating_predictions = self._aligned_predictions(
            self.coefficients_, data, selected
        )
        self.temperature_predictions_ = state_predictions
        self.heating_input_predictions_ = heating_predictions
        self.stability_warnings_ = self._stability_warnings(
            self._coefficient_dict(self.coefficients_)
        )
        if fit_record["selected_at_grid_boundary"]:
            self.stability_warnings_.append(
                "A selected outer-grid parameter is on the edge of its grid; "
                "widen the grid before interpreting it as identified."
            )
        self.optimizer_status_ = {
            "integration": "forward Euler",
            "method": "capacitance grid with bounded linear least squares",
            "linear_parameterization": (
                "physical conductances and heat-gain coefficients"
            ),
            "joint_objective": (
                "lambda * normalized temperature RMSE + "
                "(1-lambda) * normalized heating-input RMSE"
            ),
            "lambda_temperature": self.lambda_temperature,
            "normalization": "best grid RMSE on training only",
            "selected_capacitances": dict(self.selected_capacitances_),
            "grid_shape": fit_record["grid_shape"],
            "grid_evaluations": fit_record["grid_evaluations"],
            "selected_grid_index": fit_record["selected_grid_index"],
            "selected_at_grid_boundary": fit_record["selected_at_grid_boundary"],
            "linear_solver": "scipy.optimize.lsq_linear",
            "linear_solver_status": fit_record["linear_solver_status"],
            "split_transition_counts": {
                name: int(len(indices)) for name, indices in split_indices.items()
            },
            "rollout_metrics_used_for_fitting": False,
        }

    def _fit_on_indices(
        self,
        data: PreparedData,
        training_indices: np.ndarray,
    ) -> dict[str, Any]:
        self.uses_solar_ = data.solar is not None
        specs = self._resolved_specs(self.uses_solar_)
        self.linear_coefficient_names_ = tuple(spec.name for spec in specs)
        if len(training_indices) * len(self.observed_state_names) <= len(specs):
            raise ValueError(
                "The training range does not contain enough state transitions "
                "for the selected model."
            )
        lower = np.asarray([spec.lower for spec in specs], dtype=float)
        upper = np.asarray([spec.upper for spec in specs], dtype=float)
        grid = list(self._capacity_combinations())
        records: list[dict[str, Any]] = []
        for grid_index, capacitances in enumerate(grid):
            design, target = self._linear_system(
                data, training_indices, capacitances, specs
            )
            result = lsq_linear(design, target, bounds=(lower, upper), method="trf")
            if not result.success or not np.all(np.isfinite(result.x)):
                continue
            rates = self._rate_parameters(
                dict(zip(self.linear_coefficient_names_, result.x, strict=True)),
                capacitances,
            )
            if self.coefficient_names_ is None:
                self.coefficient_names_ = tuple(rates)
            rate_vector = np.asarray(
                [rates[name] for name in self.coefficient_names_], dtype=float
            )
            temperature_rmse, heating_rmse, _ = self._rmse_pair(
                rate_vector, data, training_indices
            )
            records.append(
                {
                    "grid_index": grid_index,
                    "capacitances": capacitances,
                    "linear_parameters": np.asarray(result.x, dtype=float),
                    "rate_parameters": rate_vector,
                    "temperature_rmse": temperature_rmse,
                    "heating_rmse": heating_rmse,
                    "cost": float(result.cost),
                    "optimality": float(result.optimality),
                    "active_mask": np.asarray(result.active_mask, dtype=int),
                    "iterations": int(result.nit),
                }
            )
        if not records:
            raise RuntimeError("All capacitance-grid linear fits failed.")

        measured_states = self._measured_next_states(data, training_indices)
        temperature_record = min(records, key=lambda item: item["temperature_rmse"])
        heating_record = min(records, key=lambda item: item["heating_rmse"])
        temperature_normalizer = self._safe_normalizer(
            temperature_record["temperature_rmse"], float(np.std(measured_states))
        )
        heating_normalizer = self._safe_normalizer(
            heating_record["heating_rmse"],
            float(np.std(data.heating[training_indices])),
        )
        for record in records:
            record["normalized_temperature_rmse"] = (
                record["temperature_rmse"] / temperature_normalizer
            )
            record["normalized_heating_rmse"] = (
                record["heating_rmse"] / heating_normalizer
            )
            record["joint_objective"] = (
                self.lambda_temperature
                * record["normalized_temperature_rmse"]
                + (1.0 - self.lambda_temperature)
                * record["normalized_heating_rmse"]
            )
        selected = min(
            records,
            key=lambda item: (
                item["joint_objective"],
                tuple(item["capacitances"].values()),
            ),
        )

        self.coefficient_names_ = tuple(
            self._rate_parameters(
                dict(
                    zip(
                        self.linear_coefficient_names_,
                        selected["linear_parameters"],
                        strict=True,
                    )
                ),
                selected["capacitances"],
            )
        )
        grid_shape = [len(self.indoor_capacitance_grid)]
        if self.requires_mass_state:
            assert self.mass_capacitance_grid is not None
            grid_shape.append(len(self.mass_capacitance_grid))
        return {
            "rate_parameters": selected["rate_parameters"],
            "linear_parameters": selected["linear_parameters"],
            "capacitances": selected["capacitances"],
            "temperature_normalizer": temperature_normalizer,
            "heating_normalizer": heating_normalizer,
            "temperature_only_rmse": temperature_record["temperature_rmse"],
            "temperature_only_fit_heating_rmse": temperature_record["heating_rmse"],
            "heating_only_fit_temperature_rmse": heating_record["temperature_rmse"],
            "heating_only_rmse": heating_record["heating_rmse"],
            "grid_shape": grid_shape,
            "grid_evaluations": len(records),
            "selected_grid_index": int(selected["grid_index"]),
            "selected_at_grid_boundary": self._at_grid_boundary(
                selected["capacitances"]
            ),
            "linear_solver_status": {
                "cost": selected["cost"],
                "optimality": selected["optimality"],
                "iterations": selected["iterations"],
                "active_bounds": int(np.count_nonzero(selected["active_mask"])),
            },
            "grid_records": [
                {
                    "grid_index": int(record["grid_index"]),
                    **{
                        name: float(value)
                        for name, value in record["capacitances"].items()
                    },
                    "temperature_rmse": float(record["temperature_rmse"]),
                    "heating_input_rmse": float(record["heating_rmse"]),
                    "normalized_temperature_rmse": float(
                        record["normalized_temperature_rmse"]
                    ),
                    "normalized_heating_input_rmse": float(
                        record["normalized_heating_rmse"]
                    ),
                    "joint_objective": float(record["joint_objective"]),
                }
                for record in records
            ],
        }

    def _base_coefficient_specs(self) -> list[CoefficientSpec]:
        """Return physical coefficients used by the linear heat balances."""

        specs = [
            CoefficientSpec(
                "outside_air_conductance_kW_per_degC",
                lower=0.0,
                nonnegative=True,
            ),
            CoefficientSpec(
                "air_mass_conductance_kW_per_degC",
                lower=0.0,
                nonnegative=True,
            ),
        ]
        if self.has_mass_ambient_path:
            specs.append(
                CoefficientSpec(
                    "mass_ambient_conductance_kW_per_degC",
                    lower=0.0,
                    nonnegative=True,
                )
            )
        return specs

    @abstractmethod
    def _predict_states(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        """Return next-state forward-Euler predictions."""

    @abstractmethod
    def _physical_parameters(
        self, coefficients: dict[str, float]
    ) -> dict[str, float]:
        """Map fitted rate coefficients to R/C parameters algebraically."""

    @abstractmethod
    def _stability_warnings(
        self, coefficients: dict[str, float]
    ) -> list[str]:
        """Return warnings for timestep/time-constant concerns."""

    def _air_rate_without_heating(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        rate = (
            coefficients["air_ambient_rate"]
            * (data.ambient[current] - data.indoor[current])
            + coefficients["air_mass_rate"]
            * (data.mass[current] - data.indoor[current])
        )
        if data.solar is not None:
            rate = rate + coefficients["solar_rate"] * data.solar[current]
        if self.disturbance_model == "constant":
            rate = rate + coefficients["disturbance_rate"]
        else:
            assert data.is_day is not None
            daytime = data.is_day[current].astype(float)
            rate = rate + coefficients["day_disturbance_rate"] * daytime
            rate = rate + coefficients["night_disturbance_rate"] * (1.0 - daytime)
        return rate

    def _reconstruct_heating(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        measured_rate = (
            data.indoor[current + 1] - data.indoor[current]
        ) / self.dt_hours
        return (
            measured_rate
            - self._air_rate_without_heating(coefficients, data, current)
        ) / coefficients["heating_rate"]

    def _resolved_specs(self, uses_solar: bool) -> list[CoefficientSpec]:
        specs = self._base_coefficient_specs()
        if uses_solar:
            specs.append(
                CoefficientSpec(
                    "solar_gain_kW_per_input_unit",
                    lower=0.0,
                    nonnegative=True,
                )
            )
        if self.disturbance_model == "constant":
            specs.append(CoefficientSpec("Q_disturbance_kW"))
        else:
            specs.extend(
                [
                    CoefficientSpec("Q_day_kW"),
                    CoefficientSpec("Q_night_kW"),
                ]
            )
        known = {spec.name for spec in specs}
        unknown = set(self.coefficient_bounds) - known
        if unknown:
            raise ValueError(
                "Unknown coefficient bound name(s): " + ", ".join(sorted(unknown))
            )
        resolved = []
        for spec in specs:
            if spec.name not in self.coefficient_bounds:
                resolved.append(spec)
                continue
            lower, upper = map(float, self.coefficient_bounds[spec.name])
            if np.isnan(lower) or np.isnan(upper) or lower >= upper:
                raise ValueError(f"Invalid bounds for {spec.name}.")
            if spec.nonnegative and lower < 0:
                raise ValueError(f"{spec.name} cannot have a negative lower bound.")
            resolved.append(
                replace(
                    spec,
                    lower=lower,
                    upper=upper,
                )
            )
        return resolved

    def _linear_system(
        self,
        data: PreparedData,
        indices: np.ndarray,
        capacitances: Mapping[str, float],
        specs: Sequence[CoefficientSpec],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a state-temperature-error least-squares system."""

        names = [spec.name for spec in specs]
        columns = {name: position for position, name in enumerate(names)}
        sample_count = len(indices)
        air_design = np.zeros((sample_count, len(specs)), dtype=float)
        air_target = np.empty(sample_count, dtype=float)
        indoor_capacitance = capacitances["C_i_kWh_per_degC"]

        air_design[:, columns["outside_air_conductance_kW_per_degC"]] = (
            data.ambient[indices] - data.indoor[indices]
        )
        air_design[:, columns["air_mass_conductance_kW_per_degC"]] = (
            data.mass[indices] - data.indoor[indices]
        )
        if data.solar is not None:
            air_design[:, columns["solar_gain_kW_per_input_unit"]] = data.solar[
                indices
            ]
        if self.disturbance_model == "constant":
            air_design[:, columns["Q_disturbance_kW"]] = 1.0
        else:
            assert data.is_day is not None
            daytime = data.is_day[indices].astype(float)
            air_design[:, columns["Q_day_kW"]] = daytime
            air_design[:, columns["Q_night_kW"]] = 1.0 - daytime

        indoor_rate = (
            data.indoor[indices + 1] - data.indoor[indices]
        ) / self.dt_hours
        air_target[:] = indoor_capacitance * indoor_rate - data.heating[indices]

        # Scaling each physical balance by its capacitance makes the linear
        # residual proportional to its one-step temperature prediction error.
        design_blocks = [air_design / indoor_capacitance]
        target_blocks = [air_target / indoor_capacitance]
        if self.requires_mass_state:
            mass_capacitance = capacitances["C_m_kWh_per_degC"]
            mass_design = np.zeros_like(air_design)
            mass_design[:, columns["air_mass_conductance_kW_per_degC"]] = (
                data.indoor[indices] - data.mass[indices]
            )
            if self.has_mass_ambient_path:
                mass_design[
                    :, columns["mass_ambient_conductance_kW_per_degC"]
                ] = data.ambient[indices] - data.mass[indices]
            mass_rate = (
                data.mass[indices + 1] - data.mass[indices]
            ) / self.dt_hours
            design_blocks.append(mass_design / mass_capacitance)
            target_blocks.append(mass_rate)

        return np.vstack(design_blocks), np.concatenate(target_blocks)

    def _rate_parameters(
        self,
        linear: Mapping[str, float],
        capacitances: Mapping[str, float],
    ) -> dict[str, float]:
        """Convert fitted physical coefficients to forward-Euler rates."""

        indoor_capacitance = capacitances["C_i_kWh_per_degC"]
        rates = {
            "air_ambient_rate": (
                linear["outside_air_conductance_kW_per_degC"]
                / indoor_capacitance
            ),
            "air_mass_rate": (
                linear["air_mass_conductance_kW_per_degC"]
                / indoor_capacitance
            ),
            "heating_rate": 1.0 / indoor_capacitance,
        }
        if self.requires_mass_state:
            mass_capacitance = capacitances["C_m_kWh_per_degC"]
            rates["mass_air_rate"] = (
                linear["air_mass_conductance_kW_per_degC"] / mass_capacitance
            )
            if self.has_mass_ambient_path:
                rates["mass_ambient_rate"] = (
                    linear["mass_ambient_conductance_kW_per_degC"]
                    / mass_capacitance
                )
        if "solar_gain_kW_per_input_unit" in linear:
            rates["solar_rate"] = (
                linear["solar_gain_kW_per_input_unit"] / indoor_capacitance
            )
        if self.disturbance_model == "constant":
            rates["disturbance_rate"] = (
                linear["Q_disturbance_kW"] / indoor_capacitance
            )
        else:
            rates["day_disturbance_rate"] = linear["Q_day_kW"] / indoor_capacitance
            rates["night_disturbance_rate"] = (
                linear["Q_night_kW"] / indoor_capacitance
            )
        return rates

    def _capacity_combinations(self):
        if not self.requires_mass_state:
            for indoor_capacitance in self.indoor_capacitance_grid:
                yield {"C_i_kWh_per_degC": float(indoor_capacitance)}
            return
        assert self.mass_capacitance_grid is not None
        for indoor_capacitance, mass_capacitance in product(
            self.indoor_capacitance_grid, self.mass_capacitance_grid
        ):
            yield {
                "C_i_kWh_per_degC": float(indoor_capacitance),
                "C_m_kWh_per_degC": float(mass_capacitance),
            }

    def _at_grid_boundary(self, capacitances: Mapping[str, float]) -> bool:
        grids = {"C_i_kWh_per_degC": self.indoor_capacitance_grid}
        if self.requires_mass_state:
            assert self.mass_capacitance_grid is not None
            grids["C_m_kWh_per_degC"] = self.mass_capacitance_grid
        return any(
            np.isclose(value, grid[0]) or np.isclose(value, grid[-1])
            for name, value in capacitances.items()
            for grid in (grids[name],)
        )

    def _prepare_data(
        self,
        *,
        indoor_temperature: Sequence[float] | np.ndarray,
        ambient_temperature: Sequence[float] | np.ndarray,
        heating_input: Sequence[float] | np.ndarray,
        mass_temperature: Sequence[float] | np.ndarray | None,
        timestamps: Sequence[Any] | np.ndarray,
        solar: Sequence[float] | np.ndarray | None,
        is_day: Sequence[bool] | np.ndarray | None,
        segment_ids: Sequence[Any] | np.ndarray | None,
    ) -> PreparedData:
        indoor = self._one_dimensional(indoor_temperature, "indoor_temperature")
        if len(indoor) < 2:
            raise ValueError("At least two temperature timestamps are required.")
        ambient = self._interval_array(
            ambient_temperature, "ambient_temperature", len(indoor)
        )
        heating_raw = self._one_dimensional(heating_input, "heating_input")
        heating = self._interval_array(heating_raw, "heating_input", len(indoor))
        if self.mass_is_latent:
            if mass_temperature is not None:
                raise ValueError(
                    "mass_temperature must not be supplied to a latent-mass model."
                )
            mass = np.full(len(indoor), np.nan, dtype=float)
        elif self.requires_mass_state:
            if mass_temperature is None:
                raise ValueError("mass_temperature is required for this model.")
            mass = self._one_dimensional(mass_temperature, "mass_temperature")
            if len(mass) != len(indoor):
                raise ValueError(
                    "mass_temperature must match indoor_temperature for this model."
                )
        else:
            if mass_temperature is None:
                raise ValueError("mass_temperature proxy is required for this model.")
            mass = self._interval_array(
                mass_temperature, "mass_temperature", len(indoor)
            )
        solar_array = (
            None
            if solar is None
            else self._interval_array(solar, "solar", len(indoor))
        )
        timestamps_array = self._datetime_array(timestamps, len(indoor))

        if self.disturbance_model == "day_night":
            if is_day is None:
                raise ValueError("is_day is required for a day_night disturbance.")
            raw_day = np.asarray(is_day)
            if raw_day.ndim != 1 or len(raw_day) not in {
                len(indoor) - 1,
                len(indoor),
            }:
                raise ValueError("is_day must have one value per interval or timestamp.")
            day_array = raw_day.astype(bool)
            if len(day_array) == len(indoor) - 1:
                day_array = np.append(day_array, day_array[-1])
        else:
            if is_day is not None:
                raise ValueError("is_day is only valid for a day_night disturbance.")
            day_array = None

        if segment_ids is None:
            available = np.arange(len(indoor) - 1, dtype=int)
        else:
            segments = np.asarray(segment_ids)
            if segments.ndim != 1 or len(segments) != len(indoor):
                raise ValueError("segment_ids must match indoor_temperature.")
            available = np.flatnonzero(segments[:-1] == segments[1:])
        if len(available) == 0:
            raise ValueError("No continuous adjacent transitions are available.")
        return PreparedData(
            indoor=indoor,
            ambient=ambient,
            heating=heating,
            mass=mass,
            solar=solar_array,
            is_day=day_array,
            timestamps=timestamps_array,
            available_current=available,
            heating_output_length=len(heating_raw),
        )

    def _split_transition_indices(
        self,
        data: PreparedData,
        *,
        training_range: tuple[Any, Any],
        validation_range: tuple[Any, Any],
        test_range: tuple[Any, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
        supplied = {
            "training": training_range,
            "validation": validation_range,
            "test": test_range,
        }
        normalized: dict[str, tuple[np.datetime64, np.datetime64]] = {}
        for name, bounds in supplied.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"{name}_range must be a (start, end) pair.")
            try:
                start = np.datetime64(bounds[0], "ns")
                end = np.datetime64(bounds[1], "ns")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid datetime bound in {name}_range.") from error
            if np.isnat(start) or np.isnat(end) or start > end:
                raise ValueError(f"Invalid bounds in {name}_range.")
            normalized[name] = (start, end)
        train_start, train_end = normalized["training"]
        validation_start, validation_end = normalized["validation"]
        test_start, test_end = normalized["test"]
        if not (
            train_start <= train_end < validation_start
            <= validation_end < test_start <= test_end
        ):
            raise ValueError(
                "Ranges must be chronological and non-overlapping: training, "
                "validation, then test."
            )

        indices: dict[str, np.ndarray] = {}
        report: dict[str, dict[str, str]] = {}
        for name, (start, end) in normalized.items():
            selected = data.available_current[
                (data.timestamps[data.available_current] >= start)
                & (data.timestamps[data.available_current + 1] <= end)
            ]
            if len(selected) == 0:
                raise ValueError(f"{name}_range has no complete transitions.")
            indices[name] = selected
            report[name] = {
                "requested_start": str(start),
                "requested_end": str(end),
                "first_timestamp": str(data.timestamps[selected[0]]),
                "last_timestamp": str(data.timestamps[selected[-1] + 1]),
            }
        return indices, report

    def _errors(
        self,
        parameters: np.ndarray,
        data: PreparedData,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = self._coefficient_dict(parameters)
        predicted = self._predict_states(coefficients, data, indices)
        measured = self._measured_next_states(data, indices)
        heating = self._reconstruct_heating(coefficients, data, indices)
        return predicted - measured, heating - data.heating[indices]

    def _rmse_pair(
        self,
        parameters: np.ndarray,
        data: PreparedData,
        indices: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        temperature_error, heating_error = self._errors(parameters, data, indices)
        per_state = np.sqrt(np.mean(temperature_error**2, axis=0))
        return (
            float(np.sqrt(np.mean(temperature_error**2))),
            float(np.sqrt(np.mean(heating_error**2))),
            per_state,
        )

    def _metrics_for_indices(
        self,
        parameters: np.ndarray,
        data: PreparedData,
        indices: np.ndarray,
        temperature_normalizer: float,
        heating_normalizer: float,
    ) -> dict[str, float]:
        temperature_rmse, heating_rmse, per_state = self._rmse_pair(
            parameters, data, indices
        )
        normalized_temperature = temperature_rmse / temperature_normalizer
        normalized_heating = heating_rmse / heating_normalizer
        metrics = {
            "temperature_rmse": temperature_rmse,
            "heating_input_rmse": heating_rmse,
            "normalized_temperature_rmse": normalized_temperature,
            "normalized_heating_input_rmse": normalized_heating,
            "joint_objective": (
                self.lambda_temperature * normalized_temperature
                + (1.0 - self.lambda_temperature) * normalized_heating
            ),
            "number_of_transitions": float(len(indices)),
        }
        for state_name, value in zip(
            self.observed_state_names, per_state, strict=True
        ):
            metrics[f"{state_name}_temperature_rmse"] = float(value)
        return metrics

    def _evaluate_rollouts(
        self,
        data: PreparedData,
        split_indices: Mapping[str, np.ndarray],
    ) -> None:
        """Add report-only periodic-reset and open-loop metrics by split."""

        eight_hour_steps = max(1, int(round(8.0 / self.dt_hours)))
        twelve_hour_steps = max(1, int(round(12.0 / self.dt_hours)))
        twenty_four_hour_steps = max(1, int(round(24.0 / self.dt_hours)))
        horizons: tuple[tuple[str, int | None], ...] = (
            ("8_hour", eight_hour_steps),
            ("12_hour", twelve_hour_steps),
            ("24_hour", twenty_four_hour_steps),
            ("open_loop", None),
        )
        self.rollout_evaluation_ = {
            "metrics_used_for_fitting": False,
            "8_hour_reset_steps": eight_hour_steps,
            "8_hour_actual_reset_hours": eight_hour_steps * self.dt_hours,
            "12_hour_reset_steps": twelve_hour_steps,
            "12_hour_actual_reset_hours": twelve_hour_steps * self.dt_hours,
            "24_hour_reset_steps": twenty_four_hour_steps,
            "24_hour_actual_reset_hours": twenty_four_hour_steps * self.dt_hours,
            "open_loop_reset_steps": None,
        }
        self.rollout_predictions_ = {
            label: {
                state_name: np.full(len(data.indoor), np.nan, dtype=float)
                for state_name in self.state_names
            }
            for label, _ in horizons
        }

        coefficients = self._coefficient_dict(self.coefficients_)
        for split_name, indices in split_indices.items():
            for label, reset_steps in horizons:
                predictions = self._rollout_states(
                    coefficients, data, indices, reset_steps
                )
                for state_name in self.state_names:
                    values = predictions[state_name]
                    mask = np.isfinite(values)
                    self.rollout_predictions_[label][state_name][mask] = values[mask]
                self.split_metrics_[split_name].update(
                    self._rollout_metrics(data, predictions, label)
                )

    def _rollout_states(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        indices: np.ndarray,
        reset_steps: int | None,
    ) -> dict[str, np.ndarray]:
        """Propagate measured air/mass states within each contiguous block."""

        if not self.requires_mass_state:
            raise NotImplementedError(
                "Single-state subclasses must define their rollout memory."
            )
        predictions = {
            state_name: np.full(len(data.indoor), np.nan, dtype=float)
            for state_name in self.state_names
        }
        for block in self._contiguous_index_blocks(indices):
            first = int(block[0])
            state = np.array([data.indoor[first], data.mass[first]], dtype=float)
            for offset, current_value in enumerate(block):
                current = int(current_value)
                if reset_steps is not None and offset > 0 and offset % reset_steps == 0:
                    state[:] = (data.indoor[current], data.mass[current])
                indoor, mass = state
                air_rate = self._air_rate_at_state(
                    coefficients, data, current, indoor, mass
                )
                mass_rate = coefficients["mass_air_rate"] * (indoor - mass)
                if self.has_mass_ambient_path:
                    mass_rate += coefficients["mass_ambient_rate"] * (
                        data.ambient[current] - mass
                    )
                state = np.array(
                    [
                        indoor + self.dt_hours * air_rate,
                        mass + self.dt_hours * mass_rate,
                    ]
                )
                predictions["indoor"][current + 1] = state[0]
                predictions["mass"][current + 1] = state[1]
        return predictions

    def _air_rate_at_state(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        current: int,
        indoor: float,
        mass: float,
    ) -> float:
        rate = (
            coefficients["air_ambient_rate"] * (data.ambient[current] - indoor)
            + coefficients["air_mass_rate"] * (mass - indoor)
            + coefficients["heating_rate"] * data.heating[current]
        )
        if data.solar is not None:
            rate += coefficients["solar_rate"] * data.solar[current]
        if self.disturbance_model == "constant":
            rate += coefficients["disturbance_rate"]
        else:
            assert data.is_day is not None
            rate += (
                coefficients["day_disturbance_rate"]
                if data.is_day[current]
                else coefficients["night_disturbance_rate"]
            )
        return float(rate)

    def _rollout_metrics(
        self,
        data: PreparedData,
        predictions: Mapping[str, np.ndarray],
        label: str,
    ) -> dict[str, float]:
        actual = {"indoor": data.indoor, "mass": data.mass}
        squared_errors = []
        metrics: dict[str, float] = {}
        for state_name in self.observed_state_names:
            predicted = predictions[state_name]
            mask = np.isfinite(predicted)
            errors = predicted[mask] - actual[state_name][mask]
            squared_errors.append(errors**2)
            metrics[f"{state_name}_temperature_rmse_{label}"] = float(
                np.sqrt(np.mean(errors**2))
            )
        metrics[f"temperature_rmse_{label}"] = float(
            np.sqrt(np.mean(np.concatenate(squared_errors)))
        )
        return metrics

    @staticmethod
    def _contiguous_index_blocks(indices: np.ndarray) -> list[np.ndarray]:
        if len(indices) == 0:
            return []
        cuts = np.flatnonzero(np.diff(indices) != 1) + 1
        return [block for block in np.split(indices, cuts) if len(block)]

    def _aligned_predictions(
        self,
        parameters: np.ndarray,
        data: PreparedData,
        indices: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        coefficients = self._coefficient_dict(parameters)
        predicted = self._predict_states(coefficients, data, indices)
        aligned_states = {
            name: np.full(len(data.indoor), np.nan, dtype=float)
            for name in self.state_names
        }
        for column, name in enumerate(self.state_names):
            aligned_states[name][indices + 1] = predicted[:, column]
        aligned_heating = np.full(data.heating_output_length, np.nan, dtype=float)
        aligned_heating[indices] = self._reconstruct_heating(
            coefficients, data, indices
        )
        return aligned_states, aligned_heating

    def _measured_next_states(
        self, data: PreparedData, indices: np.ndarray
    ) -> np.ndarray:
        if self.requires_mass_state and not self.mass_is_latent:
            return np.column_stack(
                (data.indoor[indices + 1], data.mass[indices + 1])
            )
        return data.indoor[indices + 1, None]

    @property
    def coefficient_values(self) -> dict[str, float]:
        self._require_fitted()
        assert self.coefficients_ is not None
        return self._coefficient_dict(self.coefficients_)

    @property
    def linear_coefficient_values(self) -> dict[str, float]:
        self._require_fitted()
        assert self.linear_coefficient_names_ is not None
        assert self.linear_coefficients_ is not None
        return dict(
            zip(
                self.linear_coefficient_names_,
                self.linear_coefficients_.tolist(),
                strict=True,
            )
        )

    def summary(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "model": self.model_name,
            "integration": "forward Euler",
            "sampling_minutes": self.sampling_minutes,
            "disturbance_model": self.disturbance_model,
            "lambda_temperature": self.lambda_temperature,
            "rate_coefficients": self.coefficient_values,
            "linear_physical_coefficients": self.linear_coefficient_values,
            "selected_capacitances": dict(self.selected_capacitances_),
            "physical_parameters": dict(self.physical_parameters_),
            "split_metrics": {
                name: dict(metrics) for name, metrics in self.split_metrics_.items()
            },
            "data_ranges": dict(self.data_ranges_),
            "stability_warnings": list(self.stability_warnings_),
            "optimizer": dict(self.optimizer_status_),
            "capacitance_grid_search": dict(self.grid_search_),
            "rollout_evaluation": dict(self.rollout_evaluation_),
        }

    def _coefficient_dict(self, parameters: np.ndarray) -> dict[str, float]:
        if self.coefficient_names_ is None:
            raise RuntimeError("Coefficient names are not initialized.")
        return dict(zip(self.coefficient_names_, parameters.tolist(), strict=True))

    @staticmethod
    def _one_dimensional(
        values: Sequence[float] | np.ndarray, name: str
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite one-dimensional array.")
        return array

    @classmethod
    def _interval_array(
        cls,
        values: Sequence[float] | np.ndarray,
        name: str,
        timestamp_count: int,
    ) -> np.ndarray:
        array = cls._one_dimensional(values, name)
        if len(array) == timestamp_count:
            return array
        if len(array) == timestamp_count - 1:
            return np.append(array, array[-1])
        raise ValueError(
            f"{name} must have one value per timestamp or per interval."
        )

    @staticmethod
    def _datetime_array(
        values: Sequence[Any] | np.ndarray, expected_length: int
    ) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 1 or len(raw) != expected_length:
            raise ValueError("timestamps must match indoor_temperature.")
        try:
            converted = raw.astype("datetime64[ns]")
        except (TypeError, ValueError) as error:
            raise ValueError("timestamps must contain datetime-like values.") from error
        if np.any(np.isnat(converted)) or np.any(converted[1:] <= converted[:-1]):
            raise ValueError("timestamps must be finite and strictly increasing.")
        return converted

    @staticmethod
    def _safe_normalizer(primary: float, fallback: float) -> float:
        epsilon = 100.0 * np.finfo(float).eps
        if np.isfinite(primary) and primary > epsilon:
            return float(primary)
        if np.isfinite(fallback) and fallback > epsilon:
            return float(fallback)
        return 1.0

    @staticmethod
    def _validated_grid(
        values: Sequence[float] | np.ndarray | None,
        name: str,
    ) -> np.ndarray:
        if values is None:
            grid = np.linspace(0.1, 30.0, 61)
        else:
            grid = np.asarray(values, dtype=float)
        if grid.ndim != 1 or len(grid) == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional sequence.")
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise ValueError(f"{name} values must be positive and finite.")
        return np.unique(grid)

    def _require_fitted(self) -> None:
        if self.coefficients_ is None:
            raise RuntimeError("fit() must be called first.")

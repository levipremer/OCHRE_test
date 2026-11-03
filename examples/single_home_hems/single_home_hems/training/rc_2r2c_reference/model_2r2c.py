"""Linear-inner/grid-outer 2R2C model with latent mass temperature."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from .joint_fitter import (
    CoefficientSpec,
    DisturbanceKind,
    ForwardEulerJointFitter,
    PreparedData,
)
from .model_2r1c import _euler_warnings, _resistance_from_conductance


class TwoR2CModel(ForwardEulerJointFitter):
    """Air/mass model whose unmeasured mass state is propagated internally.

    ``C_i``, ``C_m``, and the mass-air time constant are outer-grid
    parameters. For each candidate, the latent mass trajectory is affine in
    its initial offset. The offset, outside conductance, solar gain, and
    disturbance powers are therefore estimated by bounded linear least
    squares. The air-mass conductance is fixed within a candidate as
    ``C_m / tau_im``.
    """

    model_name = "2R2C linear-grid latent-mass model"
    state_names = ("indoor", "mass")
    observed_state_names = ("indoor",)
    requires_mass_state = True
    mass_is_latent = True

    def __init__(
        self,
        sampling_minutes: float = 30.0,
        disturbance_model: DisturbanceKind = "constant",
        lambda_temperature: float = 0.5,
        indoor_capacitance_grid: Sequence[float] | np.ndarray | None = None,
        mass_capacitance_grid: Sequence[float] | np.ndarray | None = None,
        mass_time_constant_grid_hours: Sequence[float] | np.ndarray | None = None,
        initial_mass_offset_bounds: tuple[float, float] = (-10.0, 10.0),
        coefficient_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(
            sampling_minutes=sampling_minutes,
            disturbance_model=disturbance_model,
            lambda_temperature=lambda_temperature,
            indoor_capacitance_grid=indoor_capacitance_grid,
            mass_capacitance_grid=mass_capacitance_grid,
            coefficient_bounds=coefficient_bounds,
        )
        default_time_constants = np.geomspace(2.0, 240.0, 7)
        self.mass_time_constant_grid_hours = self._validated_grid(
            default_time_constants
            if mass_time_constant_grid_hours is None
            else mass_time_constant_grid_hours,
            "mass_time_constant_grid_hours",
        )
        lower, upper = map(float, initial_mass_offset_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "initial_mass_offset_bounds must be finite with lower < upper."
            )
        self.initial_mass_offset_bounds = (lower, upper)
        self.selected_latent_grid_parameters_: dict[str, float] = {}

    def _fit_and_finalize(
        self,
        data: PreparedData,
        split_indices: dict[str, np.ndarray],
        range_report: dict[str, dict[str, str]],
    ) -> None:
        super()._fit_and_finalize(data, split_indices, range_report)
        self.grid_search_["selected_grid_parameters"] = dict(
            self.selected_latent_grid_parameters_
        )
        self.optimizer_status_.update(
            {
                "method": "latent-dynamics grid with bounded linear least squares",
                "state_observation": (
                    "indoor air measured; mass temperature latent and propagated"
                ),
                "selected_latent_grid_parameters": dict(
                    self.selected_latent_grid_parameters_
                ),
            }
        )

    def _fit_on_indices(
        self,
        data: PreparedData,
        training_indices: np.ndarray,
    ) -> dict[str, Any]:
        self.uses_solar_ = data.solar is not None
        fit_specs = self._latent_fit_specs(self.uses_solar_)
        fit_names = tuple(spec.name for spec in fit_specs)
        if len(training_indices) <= len(fit_specs):
            raise ValueError(
                "The training range does not contain enough indoor-temperature "
                "transitions for the latent-mass model."
            )
        lower = np.asarray([spec.lower for spec in fit_specs], dtype=float)
        upper = np.asarray([spec.upper for spec in fit_specs], dtype=float)

        records: list[dict[str, Any]] = []
        for grid_index, grid_parameters in enumerate(
            self._latent_grid_combinations()
        ):
            if not self._mass_update_is_monotone(grid_parameters):
                continue
            design, target = self._latent_linear_system(
                data, training_indices, grid_parameters, fit_names
            )
            result = lsq_linear(design, target, bounds=(lower, upper), method="trf")
            if not result.success or not np.all(np.isfinite(result.x)):
                continue
            fitted = dict(zip(fit_names, result.x, strict=True))
            linear = self._complete_linear_parameters(fitted, grid_parameters)
            rates = self._latent_rate_parameters(linear, grid_parameters)
            if not self._rate_vector_is_stable(rates):
                continue
            rate_names = tuple(rates)
            self.coefficient_names_ = rate_names
            self.selected_latent_grid_parameters_ = dict(grid_parameters)
            rate_vector = np.asarray([rates[name] for name in rate_names], dtype=float)
            temperature_rmse, heating_rmse, _ = self._rmse_pair(
                rate_vector, data, training_indices
            )
            report_linear_names = tuple(linear)
            records.append(
                {
                    "grid_index": grid_index,
                    "grid_parameters": dict(grid_parameters),
                    "capacitances": {
                        "C_i_kWh_per_degC": grid_parameters[
                            "C_i_kWh_per_degC"
                        ],
                        "C_m_kWh_per_degC": grid_parameters[
                            "C_m_kWh_per_degC"
                        ],
                    },
                    "linear_parameter_names": report_linear_names,
                    "linear_parameters": np.asarray(
                        [linear[name] for name in report_linear_names], dtype=float
                    ),
                    "rate_parameter_names": rate_names,
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
            raise RuntimeError(
                "All latent-dynamics grid fits failed or were Euler-unstable."
            )

        temperature_record = min(records, key=lambda item: item["temperature_rmse"])
        heating_record = min(records, key=lambda item: item["heating_rmse"])
        temperature_normalizer = self._safe_normalizer(
            temperature_record["temperature_rmse"],
            float(np.std(data.indoor[training_indices + 1])),
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
                self.lambda_temperature * record["normalized_temperature_rmse"]
                + (1.0 - self.lambda_temperature)
                * record["normalized_heating_rmse"]
            )
        selected = min(
            records,
            key=lambda item: (
                item["joint_objective"],
                tuple(item["grid_parameters"].values()),
            ),
        )

        self.selected_latent_grid_parameters_ = dict(selected["grid_parameters"])
        self.coefficient_names_ = selected["rate_parameter_names"]
        self.linear_coefficient_names_ = selected["linear_parameter_names"]
        selected_at_boundary = self._latent_grid_boundary(
            selected["grid_parameters"]
        )
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
            "grid_shape": self._latent_grid_shape(),
            "grid_evaluations": len(records),
            "selected_grid_index": int(selected["grid_index"]),
            "selected_at_grid_boundary": selected_at_boundary,
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
                        for name, value in record["grid_parameters"].items()
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

    def _latent_fit_specs(self, uses_solar: bool) -> list[CoefficientSpec]:
        specs = [
            CoefficientSpec(
                "outside_air_conductance_kW_per_degC",
                lower=0.0,
                nonnegative=True,
            ),
            CoefficientSpec(
                "initial_mass_offset_degC",
                lower=self.initial_mass_offset_bounds[0],
                upper=self.initial_mass_offset_bounds[1],
            ),
        ]
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
                [CoefficientSpec("Q_day_kW"), CoefficientSpec("Q_night_kW")]
            )

        known = {spec.name for spec in specs}
        unknown = set(self.coefficient_bounds) - known
        if unknown:
            raise ValueError(
                "Unknown or grid-fixed coefficient bound name(s): "
                + ", ".join(sorted(unknown))
            )
        resolved: list[CoefficientSpec] = []
        for spec in specs:
            if spec.name not in self.coefficient_bounds:
                resolved.append(spec)
                continue
            lower, upper = map(float, self.coefficient_bounds[spec.name])
            if np.isnan(lower) or np.isnan(upper) or lower >= upper:
                raise ValueError(f"Invalid bounds for {spec.name}.")
            if spec.nonnegative and lower < 0.0:
                raise ValueError(f"{spec.name} cannot have a negative lower bound.")
            resolved.append(
                CoefficientSpec(spec.name, lower, upper, spec.nonnegative)
            )
        return resolved

    def _latent_grid_combinations(self):
        assert self.mass_capacitance_grid is not None
        for c_i, c_m, tau_im in product(
            self.indoor_capacitance_grid,
            self.mass_capacitance_grid,
            self.mass_time_constant_grid_hours,
        ):
            yield {
                "C_i_kWh_per_degC": float(c_i),
                "C_m_kWh_per_degC": float(c_m),
                "tau_im_hours": float(tau_im),
            }

    def _latent_grid_shape(self) -> list[int]:
        assert self.mass_capacitance_grid is not None
        return [
            len(self.indoor_capacitance_grid),
            len(self.mass_capacitance_grid),
            len(self.mass_time_constant_grid_hours),
        ]

    def _latent_grid_boundary(self, parameters: Mapping[str, float]) -> bool:
        assert self.mass_capacitance_grid is not None
        grids = {
            "C_i_kWh_per_degC": self.indoor_capacitance_grid,
            "C_m_kWh_per_degC": self.mass_capacitance_grid,
            "tau_im_hours": self.mass_time_constant_grid_hours,
        }
        return any(
            np.isclose(parameters[name], grid[0])
            or np.isclose(parameters[name], grid[-1])
            for name, grid in grids.items()
        )

    def _mass_rates_from_grid(
        self, grid_parameters: Mapping[str, float]
    ) -> tuple[float, float]:
        return 1.0 / grid_parameters["tau_im_hours"], 0.0

    def _mass_conductances_from_grid(
        self, grid_parameters: Mapping[str, float]
    ) -> tuple[float, float]:
        c_m = grid_parameters["C_m_kWh_per_degC"]
        mass_air_rate, mass_ambient_rate = self._mass_rates_from_grid(
            grid_parameters
        )
        return c_m * mass_air_rate, c_m * mass_ambient_rate

    def _mass_update_is_monotone(
        self, grid_parameters: Mapping[str, float]
    ) -> bool:
        mass_air_rate, mass_ambient_rate = self._mass_rates_from_grid(
            grid_parameters
        )
        return self.dt_hours * (mass_air_rate + mass_ambient_rate) <= 1.0

    def _latent_basis(
        self,
        data: PreparedData,
        indices: np.ndarray,
        grid_parameters: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return offset-zero latent mass and sensitivity to initial offset."""

        baseline = np.full(len(data.indoor), np.nan, dtype=float)
        sensitivity = np.full(len(data.indoor), np.nan, dtype=float)
        mass_air_rate, mass_ambient_rate = self._mass_rates_from_grid(
            grid_parameters
        )
        retention = 1.0 - self.dt_hours * (
            mass_air_rate + mass_ambient_rate
        )
        for block in self._contiguous_index_blocks(indices):
            first = int(block[0])
            baseline[first] = data.indoor[first]
            sensitivity[first] = 1.0
            for current_value in block:
                current = int(current_value)
                baseline[current + 1] = (
                    retention * baseline[current]
                    + self.dt_hours * mass_air_rate * data.indoor[current]
                    + self.dt_hours * mass_ambient_rate * data.ambient[current]
                )
                sensitivity[current + 1] = retention * sensitivity[current]
        return baseline, sensitivity

    def _latent_linear_system(
        self,
        data: PreparedData,
        indices: np.ndarray,
        grid_parameters: Mapping[str, float],
        fit_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = {name: position for position, name in enumerate(fit_names)}
        design = np.zeros((len(indices), len(fit_names)), dtype=float)
        c_i = grid_parameters["C_i_kWh_per_degC"]
        g_im, _ = self._mass_conductances_from_grid(grid_parameters)
        baseline, sensitivity = self._latent_basis(
            data, indices, grid_parameters
        )

        design[:, columns["outside_air_conductance_kW_per_degC"]] = (
            data.ambient[indices] - data.indoor[indices]
        )
        design[:, columns["initial_mass_offset_degC"]] = (
            g_im * sensitivity[indices]
        )
        if data.solar is not None:
            design[:, columns["solar_gain_kW_per_input_unit"]] = data.solar[
                indices
            ]
        if self.disturbance_model == "constant":
            design[:, columns["Q_disturbance_kW"]] = 1.0
        else:
            assert data.is_day is not None
            daytime = data.is_day[indices].astype(float)
            design[:, columns["Q_day_kW"]] = daytime
            design[:, columns["Q_night_kW"]] = 1.0 - daytime

        indoor_rate = (
            data.indoor[indices + 1] - data.indoor[indices]
        ) / self.dt_hours
        target = (
            c_i * indoor_rate
            - data.heating[indices]
            - g_im * (baseline[indices] - data.indoor[indices])
        )
        return design / c_i, target / c_i

    def _complete_linear_parameters(
        self,
        fitted: Mapping[str, float],
        grid_parameters: Mapping[str, float],
    ) -> dict[str, float]:
        g_im, _ = self._mass_conductances_from_grid(grid_parameters)
        return {
            "outside_air_conductance_kW_per_degC": fitted[
                "outside_air_conductance_kW_per_degC"
            ],
            "air_mass_conductance_kW_per_degC": g_im,
            **{
                name: value
                for name, value in fitted.items()
                if name != "outside_air_conductance_kW_per_degC"
            },
        }

    def _latent_rate_parameters(
        self,
        linear: Mapping[str, float],
        grid_parameters: Mapping[str, float],
    ) -> dict[str, float]:
        c_i = grid_parameters["C_i_kWh_per_degC"]
        mass_air_rate, mass_ambient_rate = self._mass_rates_from_grid(
            grid_parameters
        )
        rates = {
            "air_ambient_rate": linear[
                "outside_air_conductance_kW_per_degC"
            ]
            / c_i,
            "air_mass_rate": linear["air_mass_conductance_kW_per_degC"] / c_i,
            "heating_rate": 1.0 / c_i,
            "mass_air_rate": mass_air_rate,
            "initial_mass_offset_degC": linear["initial_mass_offset_degC"],
        }
        if mass_ambient_rate:
            rates["mass_ambient_rate"] = mass_ambient_rate
        if "solar_gain_kW_per_input_unit" in linear:
            rates["solar_rate"] = linear["solar_gain_kW_per_input_unit"] / c_i
        if self.disturbance_model == "constant":
            rates["disturbance_rate"] = linear["Q_disturbance_kW"] / c_i
        else:
            rates["day_disturbance_rate"] = linear["Q_day_kW"] / c_i
            rates["night_disturbance_rate"] = linear["Q_night_kW"] / c_i
        return rates

    def _selected_grid_parameters(self) -> Mapping[str, float]:
        if not self.selected_latent_grid_parameters_:
            raise RuntimeError("Latent grid parameters have not been selected.")
        return self.selected_latent_grid_parameters_

    def _teacher_forced_latent_mass(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        indices: np.ndarray,
    ) -> np.ndarray:
        baseline, sensitivity = self._latent_basis(
            data, indices, self._selected_grid_parameters()
        )
        return baseline + coefficients["initial_mass_offset_degC"] * sensitivity

    def _predict_states(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        latent_mass = self._teacher_forced_latent_mass(
            coefficients, data, current
        )
        air_rate = self._air_rate_with_latent(
            coefficients, data, current, latent_mass[current], include_heating=True
        )
        return np.column_stack(
            (
                data.indoor[current] + self.dt_hours * air_rate,
                latent_mass[current + 1],
            )
        )

    def _air_rate_with_latent(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        current: np.ndarray,
        latent_mass: np.ndarray,
        *,
        include_heating: bool,
    ) -> np.ndarray:
        rate = (
            coefficients["air_ambient_rate"]
            * (data.ambient[current] - data.indoor[current])
            + coefficients["air_mass_rate"]
            * (latent_mass - data.indoor[current])
        )
        if include_heating:
            rate = rate + coefficients["heating_rate"] * data.heating[current]
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

    def _air_rate_without_heating(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        latent_mass = self._teacher_forced_latent_mass(
            coefficients, data, current
        )
        return self._air_rate_with_latent(
            coefficients,
            data,
            current,
            latent_mass[current],
            include_heating=False,
        )

    def _errors(
        self,
        parameters: np.ndarray,
        data: PreparedData,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = self._coefficient_dict(parameters)
        predicted = self._predict_states(coefficients, data, indices)
        heating = self._reconstruct_heating(coefficients, data, indices)
        temperature_error = (
            predicted[:, :1] - data.indoor[indices + 1, None]
        )
        return temperature_error, heating - data.heating[indices]

    def _rollout_states(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        indices: np.ndarray,
        reset_steps: int | None,
    ) -> dict[str, np.ndarray]:
        predictions = {
            name: np.full(len(data.indoor), np.nan, dtype=float)
            for name in self.state_names
        }
        mass_air_rate = coefficients["mass_air_rate"]
        mass_ambient_rate = coefficients.get("mass_ambient_rate", 0.0)
        for block in self._contiguous_index_blocks(indices):
            first = int(block[0])
            indoor = float(data.indoor[first])
            mass = indoor + coefficients["initial_mass_offset_degC"]
            for offset, current_value in enumerate(block):
                current = int(current_value)
                if reset_steps is not None and offset > 0 and offset % reset_steps == 0:
                    indoor = float(data.indoor[current])
                air_rate = self._air_rate_at_state(
                    coefficients, data, current, indoor, mass
                )
                mass_rate = mass_air_rate * (indoor - mass)
                mass_rate += mass_ambient_rate * (data.ambient[current] - mass)
                indoor = indoor + self.dt_hours * air_rate
                mass = mass + self.dt_hours * mass_rate
                predictions["indoor"][current + 1] = indoor
                predictions["mass"][current + 1] = mass
        return predictions

    def _physical_parameters(
        self, coefficients: dict[str, float]
    ) -> dict[str, float]:
        linear = self.linear_coefficient_values
        parameters = {
            "C_i_kWh_per_degC": self.selected_capacitances_[
                "C_i_kWh_per_degC"
            ],
            "C_m_kWh_per_degC": self.selected_capacitances_[
                "C_m_kWh_per_degC"
            ],
            "R_oa_degC_per_kW": _resistance_from_conductance(
                linear["outside_air_conductance_kW_per_degC"]
            ),
            "R_im_degC_per_kW": _resistance_from_conductance(
                linear["air_mass_conductance_kW_per_degC"]
            ),
            "tau_im_hours": self.selected_latent_grid_parameters_["tau_im_hours"],
            "initial_mass_offset_degC": coefficients[
                "initial_mass_offset_degC"
            ],
        }
        self._add_forcing_parameters(
            parameters, coefficients, coefficients["heating_rate"]
        )
        return parameters

    def _add_forcing_parameters(
        self,
        parameters: dict[str, float],
        coefficients: Mapping[str, float],
        heating_rate: float,
    ) -> None:
        if "solar_rate" in coefficients:
            parameters["solar_gain_kW_per_input_unit"] = (
                coefficients["solar_rate"] / heating_rate
            )
        if self.disturbance_model == "constant":
            parameters["Q_disturbance_kW"] = (
                coefficients["disturbance_rate"] / heating_rate
            )
        else:
            parameters["Q_day_kW"] = (
                coefficients["day_disturbance_rate"] / heating_rate
            )
            parameters["Q_night_kW"] = (
                coefficients["night_disturbance_rate"] / heating_rate
            )

    def _continuous_state_matrix(
        self, coefficients: Mapping[str, float]
    ) -> np.ndarray:
        return np.array(
            [
                [
                    -(
                        coefficients["air_ambient_rate"]
                        + coefficients["air_mass_rate"]
                    ),
                    coefficients["air_mass_rate"],
                ],
                [coefficients["mass_air_rate"], -coefficients["mass_air_rate"]],
            ],
            dtype=float,
        )

    def _rate_vector_is_stable(self, coefficients: Mapping[str, float]) -> bool:
        discrete_eigenvalues = 1.0 + self.dt_hours * np.linalg.eigvals(
            self._continuous_state_matrix(coefficients)
        )
        return bool(np.all(np.abs(discrete_eigenvalues) <= 1.0 + 1e-10))

    def _stability_warnings(
        self, coefficients: dict[str, float]
    ) -> list[str]:
        eigenvalues = np.linalg.eigvals(self._continuous_state_matrix(coefficients))
        warnings = _euler_warnings(
            self.dt_hours,
            1.0 + self.dt_hours * eigenvalues,
            _time_constants(eigenvalues),
        )
        if not self._mass_update_is_monotone(self._selected_grid_parameters()):
            warnings.append(
                "The selected latent-mass Euler update is not a monotone weighted "
                "average; use longer mass time constants or a shorter timestep."
            )
        return warnings

    def summary(self) -> dict[str, Any]:
        report = super().summary()
        report["latent_mass"] = {
            "measured_mass_temperature_used": False,
            "training": "measured indoor air teacher-forces latent mass propagation",
            "periodic_reset": "indoor air only; latent mass continues",
            "initial_offset_bounds_degC": list(self.initial_mass_offset_bounds),
            "selected_grid_parameters": dict(self.selected_latent_grid_parameters_),
        }
        return report


def _time_constants(eigenvalues: np.ndarray) -> np.ndarray:
    real_parts = np.real(eigenvalues)
    time_constants = np.full(real_parts.shape, np.inf, dtype=float)
    nonzero = np.abs(real_parts) > 100.0 * np.finfo(float).eps
    time_constants[nonzero] = -1.0 / real_parts[nonzero]
    return time_constants

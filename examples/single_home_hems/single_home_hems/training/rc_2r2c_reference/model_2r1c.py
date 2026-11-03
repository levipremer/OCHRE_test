"""Linear-grid 2R1C model with a causal moving-average mass proxy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .joint_fitter import (
    DisturbanceKind,
    ForwardEulerJointFitter,
    PreparedData,
    causal_moving_average,
)


class TwoR1CModel(ForwardEulerJointFitter):
    """Two-resistance, one-capacitance indoor-temperature model."""

    model_name = "2R1C linear-grid causal moving-mass model"
    state_names = ("indoor",)
    requires_mass_state = False

    def __init__(
        self,
        mass_window_hours: float,
        sampling_minutes: float = 30.0,
        disturbance_model: DisturbanceKind = "constant",
        lambda_temperature: float = 0.5,
        indoor_capacitance_grid: Sequence[float] | np.ndarray | None = None,
        coefficient_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(
            sampling_minutes=sampling_minutes,
            disturbance_model=disturbance_model,
            lambda_temperature=lambda_temperature,
            indoor_capacitance_grid=indoor_capacitance_grid,
            coefficient_bounds=coefficient_bounds,
        )
        if not np.isfinite(mass_window_hours) or mass_window_hours <= 0:
            raise ValueError("mass_window_hours must be positive and finite.")
        self.mass_window_hours = float(mass_window_hours)
        self.mass_window_steps = max(
            1,
            int(round(self.mass_window_hours * 60.0 / self.sampling_minutes)),
        )
        self.window_selection_: dict[str, Any] | None = None

    def fit(
        self,
        indoor_temperature: Sequence[float] | np.ndarray,
        ambient_temperature: Sequence[float] | np.ndarray,
        heating_input: Sequence[float] | np.ndarray,
        *,
        timestamps: Sequence[Any] | np.ndarray,
        training_range: tuple[Any, Any],
        validation_range: tuple[Any, Any],
        test_range: tuple[Any, Any],
        solar: Sequence[float] | np.ndarray | None = None,
        is_day: Sequence[bool] | np.ndarray | None = None,
        segment_ids: Sequence[Any] | np.ndarray | None = None,
    ) -> "TwoR1CModel":
        proxy = causal_moving_average(
            indoor_temperature,
            self.mass_window_steps,
            segment_ids=segment_ids,
        )
        super().fit(
            indoor_temperature=indoor_temperature,
            ambient_temperature=ambient_temperature,
            heating_input=heating_input,
            mass_temperature=proxy,
            timestamps=timestamps,
            training_range=training_range,
            validation_range=validation_range,
            test_range=test_range,
            solar=solar,
            is_day=is_day,
            segment_ids=segment_ids,
        )
        return self

    def _predict_states(
        self,
        coefficients: dict[str, float],
        data: PreparedData,
        current: np.ndarray,
    ) -> np.ndarray:
        air_rate = self._air_rate_without_heating(coefficients, data, current)
        air_rate += coefficients["heating_rate"] * data.heating[current]
        return (data.indoor[current] + self.dt_hours * air_rate)[:, None]

    def _rollout_states(
        self,
        coefficients: Mapping[str, float],
        data: PreparedData,
        indices: np.ndarray,
        reset_steps: int | None,
    ) -> dict[str, np.ndarray]:
        """Roll out air temperature with a prediction-only causal mass buffer."""

        predictions = {
            "indoor": np.full(len(data.indoor), np.nan, dtype=float)
        }
        for block in self._contiguous_index_blocks(indices):
            first = int(block[0])
            state = float(data.indoor[first])
            mass_buffer = [state]
            for offset, current_value in enumerate(block):
                current = int(current_value)
                if reset_steps is not None and offset > 0 and offset % reset_steps == 0:
                    state = float(data.indoor[current])
                    history_start = max(
                        first, current - self.mass_window_steps + 1
                    )
                    mass_buffer = data.indoor[
                        history_start : current + 1
                    ].tolist()
                mass_proxy = float(np.mean(mass_buffer))
                state += self.dt_hours * self._air_rate_at_state(
                    coefficients, data, current, state, mass_proxy
                )
                predictions["indoor"][current + 1] = state
                mass_buffer.append(state)
                if len(mass_buffer) > self.mass_window_steps:
                    mass_buffer.pop(0)
        return predictions

    def _physical_parameters(
        self, coefficients: dict[str, float]
    ) -> dict[str, float]:
        heating_rate = coefficients["heating_rate"]
        linear = self.linear_coefficient_values
        parameters = {
            "C_i_kWh_per_degC": self.selected_capacitances_[
                "C_i_kWh_per_degC"
            ],
            "R_oa_degC_per_kW": _resistance_from_conductance(
                linear["outside_air_conductance_kW_per_degC"]
            ),
            "R_im_degC_per_kW": _resistance_from_conductance(
                linear["air_mass_conductance_kW_per_degC"]
            ),
            "mass_window_hours": self.mass_window_hours,
            "mass_window_steps": float(self.mass_window_steps),
        }
        self._add_forcing_parameters(parameters, coefficients, heating_rate)
        return parameters

    def _stability_warnings(
        self, coefficients: dict[str, float]
    ) -> list[str]:
        total_rate = (
            coefficients["air_ambient_rate"] + coefficients["air_mass_rate"]
        )
        return _euler_warnings(
            self.dt_hours,
            np.array([1.0 - self.dt_hours * total_rate]),
            np.array(
                [float("inf") if total_rate <= 0.0 else 1.0 / total_rate]
            ),
        )

    def _add_forcing_parameters(
        self,
        parameters: dict[str, float],
        coefficients: dict[str, float],
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

    def summary(self) -> dict[str, Any]:
        report = super().summary()
        report["mass_window_selection"] = self.window_selection_
        return report


def select_and_fit_2r1c(
    *,
    indoor_temperature: Sequence[float] | np.ndarray,
    ambient_temperature: Sequence[float] | np.ndarray,
    heating_input: Sequence[float] | np.ndarray,
    timestamps: Sequence[Any] | np.ndarray,
    training_range: tuple[Any, Any],
    validation_range: tuple[Any, Any],
    test_range: tuple[Any, Any],
    mass_window_candidates_hours: Sequence[float],
    window_selection_fraction: float = 0.20,
    sampling_minutes: float = 30.0,
    disturbance_model: DisturbanceKind = "constant",
    lambda_temperature: float = 0.5,
    solar: Sequence[float] | np.ndarray | None = None,
    is_day: Sequence[bool] | np.ndarray | None = None,
    segment_ids: Sequence[Any] | np.ndarray | None = None,
    indoor_capacitance_grid: Sequence[float] | np.ndarray | None = None,
    coefficient_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> TwoR1CModel:
    """Select the mass window on a held-out tail of training, then refit."""

    if not np.isfinite(window_selection_fraction) or not (
        0.0 < window_selection_fraction < 0.5
    ):
        raise ValueError("window_selection_fraction must be between 0 and 0.5.")
    candidates = np.asarray(mass_window_candidates_hours, dtype=float)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("mass_window_candidates_hours must not be empty.")
    if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0):
        raise ValueError("Mass-window candidates must be positive and finite.")
    candidates = np.unique(candidates)
    candidate_steps = np.maximum(
        1, np.rint(candidates * 60.0 / sampling_minutes).astype(int)
    )
    if len(np.unique(candidate_steps)) != len(candidate_steps):
        raise ValueError(
            "Two mass-window candidates map to the same number of samples."
        )

    common_options = {
        "sampling_minutes": sampling_minutes,
        "disturbance_model": disturbance_model,
        "lambda_temperature": lambda_temperature,
        "indoor_capacitance_grid": indoor_capacitance_grid,
        "coefficient_bounds": coefficient_bounds,
    }
    selection_records: list[dict[str, float]] = []
    selection_bounds: dict[str, str] | None = None
    for candidate in candidates:
        model = TwoR1CModel(float(candidate), **common_options)
        proxy = causal_moving_average(
            indoor_temperature, model.mass_window_steps, segment_ids
        )
        data = model._prepare_data(
            indoor_temperature=indoor_temperature,
            ambient_temperature=ambient_temperature,
            heating_input=heating_input,
            mass_temperature=proxy,
            timestamps=timestamps,
            solar=solar,
            is_day=is_day,
            segment_ids=segment_ids,
        )
        split_indices, _ = model._split_transition_indices(
            data,
            training_range=training_range,
            validation_range=validation_range,
            test_range=test_range,
        )
        full_training = split_indices["training"]
        selection_count = max(
            1, int(np.ceil(window_selection_fraction * len(full_training)))
        )
        fitting_indices = full_training[:-selection_count]
        selection_indices = full_training[-selection_count:]
        if len(fitting_indices) == 0:
            raise ValueError("The training range is too short for window selection.")
        if data.is_day is not None and (
            np.all(data.is_day[fitting_indices])
            or not np.any(data.is_day[fitting_indices])
        ):
            raise ValueError(
                "The window-fitting subset must contain day and night transitions."
            )
        fit_record = model._fit_on_indices(data, fitting_indices)
        parameters = fit_record["rate_parameters"]
        temperature_rmse, heating_rmse, _ = model._rmse_pair(
            parameters, data, selection_indices
        )
        temperature_scale = model._safe_normalizer(
            float(np.std(data.indoor[selection_indices + 1])), 1.0
        )
        heating_scale = model._safe_normalizer(
            float(np.std(data.heating[selection_indices])), 1.0
        )
        normalized_temperature = temperature_rmse / temperature_scale
        normalized_heating = heating_rmse / heating_scale
        score = (
            lambda_temperature * normalized_temperature
            + (1.0 - lambda_temperature) * normalized_heating
        )
        selection_records.append(
            {
                "mass_window_hours": float(candidate),
                "mass_window_steps": float(model.mass_window_steps),
                "temperature_rmse": temperature_rmse,
                "heating_input_rmse": heating_rmse,
                "temperature_scale": temperature_scale,
                "heating_input_scale": heating_scale,
                "normalized_temperature_rmse": normalized_temperature,
                "normalized_heating_input_rmse": normalized_heating,
                "selection_score": score,
            }
        )
        if selection_bounds is None:
            selection_bounds = {
                "fit_start": str(data.timestamps[fitting_indices[0]]),
                "fit_end": str(data.timestamps[fitting_indices[-1] + 1]),
                "selection_start": str(data.timestamps[selection_indices[0]]),
                "selection_end": str(data.timestamps[selection_indices[-1] + 1]),
            }

    selected = min(
        selection_records,
        key=lambda record: (record["selection_score"], record["mass_window_hours"]),
    )
    final_model = TwoR1CModel(selected["mass_window_hours"], **common_options)
    final_model.fit(
        indoor_temperature=indoor_temperature,
        ambient_temperature=ambient_temperature,
        heating_input=heating_input,
        timestamps=timestamps,
        training_range=training_range,
        validation_range=validation_range,
        test_range=test_range,
        solar=solar,
        is_day=is_day,
        segment_ids=segment_ids,
    )
    final_model.window_selection_ = {
        "method": "chronological held-out tail of training data",
        "selection_fraction": float(window_selection_fraction),
        "selection_objective": (
            "lambda * temperature_RMSE/selection_temperature_std + "
            "(1-lambda) * heating_RMSE/selection_heating_std"
        ),
        "ranges": selection_bounds,
        "selected_mass_window_hours": selected["mass_window_hours"],
        "candidate_results": selection_records,
    }
    return final_model


def _euler_warnings(
    dt_hours: float,
    discrete_eigenvalues: np.ndarray,
    time_constants: np.ndarray,
) -> list[str]:
    warnings = []
    if np.any(np.abs(discrete_eigenvalues) >= 1.0):
        warnings.append(
            "The fitted forward-Euler transition is not asymptotically stable "
            "at the selected sampling interval."
        )
    minimum_time_constant = float(np.min(time_constants))
    if minimum_time_constant < 5.0 * dt_hours:
        warnings.append(
            "The shortest fitted thermal time constant is less than five sample "
            "intervals; forward-Euler parameter bias may be material."
        )
    return warnings


def _resistance_from_conductance(conductance: float) -> float:
    """Return an infinite resistance for a fitted zero-conductance path."""

    if conductance <= 100.0 * np.finfo(float).eps:
        return float("inf")
    return 1.0 / conductance

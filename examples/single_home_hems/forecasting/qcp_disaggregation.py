"""Quantile change-point (QCP) AC disaggregation from Jin et al. (2024).

This module implements the paper's observed-hour procedure, not a future-load
forecast. The QCP input includes the total energy *measured in the hour being
disaggregated*. A forecast runner must choose a causal future quantile without
using that future measurement; this module makes no such choice.

Paper: Jin et al., Journal of Building Engineering 97 (2024) 110729,
sections 3.2-3.4, equations (1)-(8), DOI 10.1016/j.jobe.2024.110729.
Energy can use any consistent unit; hourly kWh is the paper's convention.
Temperatures are grouped into intervals [floor(T), floor(T)+1) degrees C.
The article does not state an interval-edge convention, so this choice is
explicit here. Sparse or unseen temperature bins are never silently pooled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


GroupKey = tuple[bool, int]


def temperature_bin(temperature_c: float) -> int:
    """Return the lower edge of a 1-degree-Celsius temperature interval."""
    if not math.isfinite(temperature_c):
        raise ValueError("Temperature must be finite")
    return math.floor(temperature_c)


def _group_key(timestamp: pd.Timestamp | str) -> GroupKey:
    local = pd.Timestamp(timestamp)
    if local.minute or local.second or local.microsecond or local.nanosecond:
        raise ValueError("QCP groups require an hourly local timestamp")
    return bool(local.dayofweek >= 5), int(local.hour)


@dataclass(frozen=True)
class ChangePointCurve:
    """The five coefficients of the paper's three-section change-point curve."""

    heating_change_point_c: float
    cooling_change_point_c: float
    base_energy: float
    heating_slope: float
    cooling_slope: float

    def predict(self, temperature_c: float | np.ndarray) -> float | np.ndarray:
        temperature = np.asarray(temperature_c, dtype=float)
        prediction = (
            self.base_energy
            + self.heating_slope
            * np.minimum(temperature - self.heating_change_point_c, 0.0)
            + self.cooling_slope
            * np.maximum(temperature - self.cooling_change_point_c, 0.0)
        )
        return float(prediction) if prediction.ndim == 0 else prediction


@dataclass(frozen=True)
class CoolingEstimate:
    """Cooling result for an hour whose total energy is already measured."""

    observed_total_energy: float
    ambient_temperature_c: float
    conditional_quantile: float
    base_energy: float
    cooling_energy: float
    curve: ChangePointCurve


class QCPSurface:
    """Paper QCP model for one weekday/weekend and hour-of-day sub-dataset.

    A temperature bin contains its sorted observed total-energy samples. The
    empirical CDF counts observations less than or equal to the query (Eq. 1).
    Its inverse uses the generalized inverse, the smallest observation whose
    CDF is at least ``q``. Curve fitting is the unweighted least-squares problem
    over occupied 1-degree bins in Eqs. (2)-(8), solved using SLSQP.
    """

    def __init__(self, temperature_bins: Mapping[int, Sequence[float]]) -> None:
        if not temperature_bins:
            raise ValueError("A QCP surface needs at least one temperature bin")
        bins: dict[int, np.ndarray] = {}
        for key, values in temperature_bins.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                raise TypeError("Temperature-bin keys must be integer lower edges")
            samples = np.asarray(values, dtype=float).reshape(-1)
            if len(samples) == 0 or not np.isfinite(samples).all() or (samples < 0).any():
                raise ValueError("Each temperature bin needs finite nonnegative samples")
            bins[int(key)] = np.sort(samples)
        self._bins = dict(sorted(bins.items()))
        self._curves: dict[float, ChangePointCurve] = {}

    @classmethod
    def from_temperature_bins(
        cls, temperature_bins: Mapping[int, Sequence[float]]
    ) -> QCPSurface:
        """Construct from pre-aggregated 1-degree-Celsius empirical samples."""
        return cls(temperature_bins)

    @property
    def temperature_bins(self) -> tuple[int, ...]:
        return tuple(self._bins)

    @property
    def sample_counts(self) -> dict[int, int]:
        return {key: len(samples) for key, samples in self._bins.items()}

    def cdf(self, temperature_c: float, observed_total_energy: float) -> float:
        """Compute the observed hour's conditional quantile by Eq. (1)."""
        if not math.isfinite(observed_total_energy) or observed_total_energy < 0:
            raise ValueError("Observed total energy must be finite and nonnegative")
        key = temperature_bin(temperature_c)
        if key not in self._bins:
            raise KeyError(f"No observed temperature bin at {key} C")
        samples = self._bins[key]
        return float(np.searchsorted(samples, observed_total_energy, side="right") / len(samples))

    def inverse_cdf(self, temperature_bin_c: int, quantile: float) -> float:
        """Return the generalized inverse of a bin's empirical CDF."""
        if not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("Quantile must lie in [0, 1]")
        samples = self._bins[temperature_bin_c]
        index = max(0, math.ceil(quantile * len(samples)) - 1)
        return float(samples[index])

    def quantile_curve_samples(self, quantile: float) -> tuple[np.ndarray, np.ndarray]:
        """Return Eq. (1) inverse-CDF loads at each occupied temperature bin."""
        temperatures = np.fromiter(self._bins, dtype=float)
        loads = np.array(
            [self.inverse_cdf(key, quantile) for key in self._bins], dtype=float
        )
        return temperatures, loads

    def curve(self, quantile: float) -> ChangePointCurve:
        """Solve the constrained three-section regression in Eqs. (2)-(8)."""
        if not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("Quantile must lie in [0, 1]")
        q = float(quantile)
        if q in self._curves:
            return self._curves[q]
        temperatures, loads = self.quantile_curve_samples(q)
        if len(temperatures) < 3:
            raise ValueError("At least three occupied temperature bins are needed for QCP fitting")
        t_min, t_max = float(temperatures[0]), float(temperatures[-1])
        w_min, w_max = float(loads.min()), float(loads.max())
        t_h = float(np.quantile(temperatures, 0.25))
        t_c = float(np.quantile(temperatures, 0.75))
        base = float(np.median(loads))
        heating_slope = min(0.0, (loads[0] - base) / (t_min - t_h))
        cooling_slope = max(0.0, (loads[-1] - base) / (t_max - t_c))
        initial = np.array([t_h, t_c, base, heating_slope, cooling_slope])

        def objective(x: np.ndarray) -> float:
            th, tc, wb, kh, kc = x
            fitted = (
                wb + kh * np.minimum(temperatures - th, 0.0)
                + kc * np.maximum(temperatures - tc, 0.0)
            )
            return float(np.square(loads - fitted).sum())

        result = minimize(
            objective, initial, method="SLSQP",
            bounds=[(t_min, t_max), (t_min, t_max), (w_min, w_max),
                    (None, 0.0), (0.0, None)],
            constraints=[{"type": "ineq", "fun": lambda x: x[1] - x[0]}],
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"QCP SLSQP regression failed: {result.message}")
        th, tc, wb, kh, kc = (float(value) for value in result.x)
        fitted_curve = ChangePointCurve(th, tc, wb, kh, kc)
        self._curves[q] = fitted_curve
        return fitted_curve

    def predict(self, temperature_c: float, quantile: float) -> float:
        """Evaluate the fitted total-energy curve at temperature and quantile."""
        return float(self.curve(quantile).predict(temperature_c))


class QCPDisaggregator:
    """The paper's 48 weekday/weekend-by-hour QCP sub-datasets."""

    def __init__(self) -> None:
        self._surfaces: dict[GroupKey, QCPSurface] = {}

    @classmethod
    def from_group_bins(
        cls, group_bins: Mapping[GroupKey, Mapping[int, Sequence[float]]]
    ) -> QCPDisaggregator:
        model = cls()
        model._surfaces = {
            group: QCPSurface.from_temperature_bins(bins)
            for group, bins in group_bins.items()
        }
        return model

    def fit(
        self, hourly_frame: pd.DataFrame, *,
        timestamp_col: str = "timestamp_local",
        temperature_col: str = "temperature_c",
        total_col: str = "total_energy_kwh",
    ) -> QCPDisaggregator:
        """Fit empirical CDFs from observed hourly total energy and weather."""
        data = hourly_frame[[timestamp_col, temperature_col, total_col]].copy()
        stamps = pd.to_datetime(data[timestamp_col], errors="raise")
        if stamps.isna().any() or stamps.duplicated().any():
            raise ValueError("Hourly timestamps must be present and unique")
        temperatures = pd.to_numeric(data[temperature_col], errors="raise").to_numpy(float)
        totals = pd.to_numeric(data[total_col], errors="raise").to_numpy(float)
        if not np.isfinite(temperatures).all() or not np.isfinite(totals).all() or (totals < 0).any():
            raise ValueError("Fit data must have finite temperature and nonnegative energy")
        group_bins: dict[GroupKey, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for stamp, temperature, total in zip(stamps, temperatures, totals):
            group_bins[_group_key(stamp)][temperature_bin(float(temperature))].append(float(total))
        self._surfaces = {
            group: QCPSurface.from_temperature_bins(bins)
            for group, bins in group_bins.items()
        }
        return self

    @property
    def groups(self) -> tuple[GroupKey, ...]:
        return tuple(sorted(self._surfaces))

    def surface_at(self, timestamp: pd.Timestamp | str) -> QCPSurface:
        """Return the empirical surface for this local weekday/weekend hour."""
        group = _group_key(timestamp)
        if group not in self._surfaces:
            raise KeyError(f"No QCP sub-dataset for group {group}")
        return self._surfaces[group]

    def estimate_cooling(
        self, timestamp: pd.Timestamp | str, temperature_c: float,
        observed_total_energy: float,
    ) -> CoolingEstimate:
        """Disaggregate cooling from the *measured target-hour* total energy."""
        surface = self.surface_at(timestamp)
        q = surface.cdf(temperature_c, observed_total_energy)
        curve = surface.curve(q)
        cooling = (
            observed_total_energy - curve.base_energy
            if temperature_c > curve.cooling_change_point_c else 0.0
        )
        return CoolingEstimate(
            observed_total_energy=float(observed_total_energy),
            ambient_temperature_c=float(temperature_c),
            conditional_quantile=q,
            base_energy=curve.base_energy,
            cooling_energy=float(cooling),
            curve=curve,
        )

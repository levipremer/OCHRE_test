"""Causal, observed-hour features from an already fitted QCP surface.

The QCP model's three-section curve contains an explicit heating term.  This
module exposes that *modeled temperature term* and the other fitted terms as
historical features after the source hour has completed and its meter reading
has arrived.  It does not introduce a heating-unit disaggregation equation or
fit/pool a QCP model.  In particular, ``heating_curve_term_kw`` must not be
interpreted as a measured HVAC submeter value.

The input hourly frame uses the same schema as
``evaluate_v2h_qcp.hourly_observations``.  Its total is the aggregate Home Hub
meter less the separately metered EV power, expressed in hourly kWh.  The
caller is responsible for fitting ``model`` strictly before the forecast
origin; ``model_fit_end`` makes that cutoff explicit and is checked here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import pandas as pd

from .qcp_disaggregation import QCPSurface, temperature_bin


class QCPSurfaceProvider(Protocol):
    """Existing 48-group or separately fitted hour-pooled QCP surfaces."""

    def surface_at(self, timestamp: pd.Timestamp | str) -> QCPSurface: ...


@dataclass(frozen=True)
class IssuedQCPCurveFeatures:
    """Terms of the fitted QCP curve evaluated at one completed source hour."""

    source_hour_local: pd.Timestamp
    source_ready_at_utc: pd.Timestamp
    source_temperature_c: float
    source_total_energy_kwh: float
    source_quantile: float
    temperature_edge_fallback: bool
    heating_change_point_c: float
    cooling_change_point_c: float
    base_kw: float
    heating_curve_term_kw: float
    cooling_curve_term_kw: float
    fitted_total_kw: float

    def numeric_features(self) -> dict[str, float]:
        """Return features suitable for a lead-conditioned prediction model."""
        return {
            "qcp_source_temperature_c": self.source_temperature_c,
            "qcp_source_total_kw": self.source_total_energy_kwh,
            "qcp_source_quantile": self.source_quantile,
            "qcp_temperature_edge_fallback": float(self.temperature_edge_fallback),
            "qcp_heating_change_point_c": self.heating_change_point_c,
            "qcp_cooling_change_point_c": self.cooling_change_point_c,
            "qcp_base_kw": self.base_kw,
            "qcp_heating_curve_term_kw": self.heating_curve_term_kw,
            "qcp_cooling_curve_term_kw": self.cooling_curve_term_kw,
            "qcp_fitted_total_kw": self.fitted_total_kw,
        }


def _utc_timestamp(value: pd.Timestamp | str, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def issue_qcp_curve_features(
    model: QCPSurfaceProvider,
    hourly: pd.DataFrame,
    origin: pd.Timestamp | str,
    *,
    model_fit_end: pd.Timestamp | str,
) -> IssuedQCPCurveFeatures:
    """Use only a complete historical hour available at the issue time.

    This is a historical feature extraction, not a forecast.  A future hourly
    row may be present in ``hourly`` but cannot affect the returned features.
    ``model_fit_end`` is the last timestamp whose labels entered the fitted
    QCP model; it must be no later than the issue time.
    """
    origin_utc = _utc_timestamp(origin, "origin")
    fit_end_utc = _utc_timestamp(model_fit_end, "model_fit_end")
    if fit_end_utc > origin_utc:
        raise ValueError("QCP fit includes labels after the forecast origin")
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("QCP hourly observations need a timezone-aware hourly index")
    required = {"total_energy_kwh", "temperature_c", "ready_at_utc", "hour_end_utc"}
    missing = required - set(hourly)
    if missing:
        raise ValueError(f"QCP hourly observations lack {sorted(missing)}")
    if hourly.index.has_duplicates or not hourly.index.is_monotonic_increasing:
        raise ValueError("QCP hourly timestamps must be unique and increasing")
    if any(stamp.minute or stamp.second or stamp.microsecond or stamp.nanosecond
           for stamp in hourly.index):
        raise ValueError("QCP source timestamps must start on a local clock hour")
    end_utc = pd.to_datetime(hourly["hour_end_utc"], utc=True)
    ready_utc = pd.to_datetime(hourly["ready_at_utc"], utc=True)
    expected_end = (hourly.index + pd.Timedelta(hours=1)).tz_convert("UTC")
    if not np.array_equal(end_utc.to_numpy(), expected_end.to_numpy()):
        raise ValueError("QCP hourly observation end times do not match source hours")
    if (ready_utc < end_utc).any():
        raise ValueError("QCP hourly meter readings cannot arrive before hour end")
    eligible = hourly.loc[(end_utc <= origin_utc) & (ready_utc <= origin_utc)]
    if eligible.empty:
        raise ValueError(f"No completed QCP hourly observation at {origin_utc}")
    source = eligible.iloc[-1]
    source_hour = eligible.index[-1]
    temperature = float(source["temperature_c"])
    total_kwh = float(source["total_energy_kwh"])
    if not math.isfinite(temperature) or not math.isfinite(total_kwh) or total_kwh < 0:
        raise ValueError("QCP source temperature and total energy must be finite and nonnegative")

    surface = model.surface_at(source_hour)
    source_bin = temperature_bin(temperature)
    nearest_bin = min(surface.temperature_bins, key=lambda b: (abs(b - source_bin), b))
    quantile = surface.cdf(nearest_bin + 0.5, total_kwh)
    curve = surface.curve(quantile)
    heating = curve.heating_slope * min(temperature - curve.heating_change_point_c, 0.0)
    cooling = curve.cooling_slope * max(temperature - curve.cooling_change_point_c, 0.0)
    return IssuedQCPCurveFeatures(
        source_hour_local=source_hour,
        source_ready_at_utc=ready_utc.loc[source_hour],
        source_temperature_c=temperature,
        source_total_energy_kwh=total_kwh,
        source_quantile=quantile,
        temperature_edge_fallback=nearest_bin != source_bin,
        heating_change_point_c=curve.heating_change_point_c,
        cooling_change_point_c=curve.cooling_change_point_c,
        base_kw=curve.base_energy,
        heating_curve_term_kw=heating,
        cooling_curve_term_kw=cooling,
        fitted_total_kw=float(curve.predict(temperature)),
    )

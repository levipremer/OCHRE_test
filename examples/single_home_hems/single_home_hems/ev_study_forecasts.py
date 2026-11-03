"""Explicit offline perfect-information inputs for the EV charger study.

These providers deliberately read future native plant truth. They are oracle
comparisons, not deployable forecasts. They retain the same interval load and
scenario interfaces as the predictive controller without changing its costs,
trip-energy timing, or physical discharge limits.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import EVSchedule

from .backed_loads import HOME_HUB_BACKED_LOADS, backed_power
from .control_forecasts import EVScenarioSet
from .ev_hub_forecast import EVHubIntervalForecast, INTERVAL
from .ev_scenarios import last_observed_arrival


MINUTE = pd.Timedelta(minutes=1)
TRUTH_COLUMNS = ("eligible_load_kw", "hub_non_ev_kw", "outside_hub_with_resistance_kw")
ORACLE_SOURCE = "perfect_native_truth_offline_oracle"


def _minute_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
            or index.has_duplicates or not index.is_monotonic_increasing):
        raise ValueError(f"{name} needs nonempty, unique, increasing minute timestamps")
    if (index.asi8 % MINUTE.value).any() or (np.diff(index.asi8) != MINUTE.value).any():
        raise ValueError(f"{name} must contain every consecutive minute without gaps")
    return frame.copy()


def _clock(timestamp: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if pd.isna(timestamp) or str(timestamp.tz) != str(index.tz):
        raise ValueError("Oracle issue and truth timestamps must use the same clock/timezone")
    if timestamp.value % INTERVAL.value:
        raise ValueError("Oracle issues must align to 15-minute intervals")
    return timestamp


def _power(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(result.to_numpy()).all() or result.lt(-1e-7).any():
        raise ValueError(f"{name} must contain finite nonnegative power")
    # Retain the plant's existing tolerance for floating-point meter residuals.
    return result.clip(lower=0)


def derive_native_minute_truth(
    standardized: pd.DataFrame,
    split: pd.DataFrame | None = None,
    *,
    raw_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract matched native non-EV loads without equating Hub and V2H load.

    ``standardized`` uses the resistance-outside-Hub output mapping. If it
    lacks split aggregate fields, ``split`` supplies ``hub_other_power_kw``
    and ``outside_hub_power_kw`` on exactly the same minute clock. Exact
    eligible power comes from ``eligible_load_kw`` or the native raw component
    readings, since a Hub aggregate cannot establish discharge eligibility.
    """
    native = _minute_frame(standardized, "Native standardized truth")
    if split is not None:
        split = _minute_frame(split, "Fixed split truth")
        if not split.index.equals(native.index):
            raise ValueError("Fixed split and native truth must have identical minute timestamps")
        for name in ("hub_other_power_kw", "outside_hub_power_kw"):
            if name not in split:
                raise ValueError(f"Fixed split truth lacks {name}")
            if name in native and not np.allclose(native[name], split[name], rtol=0, atol=1e-7):
                raise ValueError(f"Native and fixed split {name} disagree")
            native[name] = _power(split[name], name)
    if "eligible_load_kw" in native:
        eligible = _power(native["eligible_load_kw"], "Eligible native load")
    elif raw_results is not None:
        raw = _minute_frame(raw_results, "Raw native truth")
        if not raw.index.equals(native.index):
            raise ValueError("Raw and standardized truth must have identical minute timestamps")
        eligible = pd.Series(backed_power(raw, HOME_HUB_BACKED_LOADS), index=native.index)
    else:
        raise ValueError("Exact eligible native truth requires eligible_load_kw or raw_results")

    if "home_hub_non_ev_power_kw" in native:
        hub = _power(native["home_hub_non_ev_power_kw"], "Native non-EV Hub")
    else:
        required = {"hvac_power_kw", "hvac_resistance_power_kw", "water_heating_power_kw", "hub_other_power_kw"}
        if not required <= set(native):
            raise ValueError(f"Native Hub truth lacks {sorted(required - set(native))}")
        hub = _power(native.hvac_power_kw - native.hvac_resistance_power_kw
                     + native.water_heating_power_kw + native.hub_other_power_kw, "Native non-EV Hub")
    if "outside_hub_with_resistance_power_kw" in native:
        outside = _power(native["outside_hub_with_resistance_power_kw"], "Native Outside-Hub plus resistance")
    else:
        required = {"outside_hub_power_kw", "hvac_resistance_power_kw"}
        if not required <= set(native):
            raise ValueError(f"Native Outside-Hub truth lacks {sorted(required - set(native))}")
        outside = _power(native.outside_hub_power_kw + native.hvac_resistance_power_kw,
                         "Native Outside-Hub plus resistance")
    if (eligible > hub + 1e-7).any():
        raise ValueError("Eligible native load exceeds the full non-EV Home Hub")
    if {"total_electric_power_kw", "ev_power_kw"} <= set(native):
        residual = native.total_electric_power_kw - native.ev_power_kw - hub - outside
        if not np.isfinite(residual).all() or residual.abs().gt(1e-6).any():
            raise ValueError("Native total-home minus EV does not balance Hub and Outside-Hub truth")
    result = pd.DataFrame(dict(zip(TRUTH_COLUMNS, (eligible, hub, outside))), index=native.index)
    result.attrs.update(source=ORACLE_SOURCE, perfect_forecast=True,
                        information_policy="offline_oracle_future_native_truth")
    return result


class _PerfectLoadTruth:
    def __init__(self, minute_truth: pd.DataFrame):
        self.minute_truth = _minute_frame(minute_truth, "Oracle minute truth")
        missing = set(TRUTH_COLUMNS) - set(self.minute_truth)
        if missing:
            raise ValueError(f"Oracle minute truth lacks {sorted(missing)}")
        for name in TRUTH_COLUMNS:
            self.minute_truth[name] = _power(self.minute_truth[name], name)
        if (self.minute_truth.eligible_load_kw > self.minute_truth.hub_non_ev_kw + 1e-7).any():
            raise ValueError("Eligible native load exceeds the full non-EV Home Hub")
        self.history: list[dict] = []
        self.metadata = {
            "method_label": ORACLE_SOURCE,
            "perfect_forecast": True,
            "information_policy": "offline_oracle_future_native_truth",
            "truth_start": str(self.minute_truth.index[0]),
            "truth_end_exclusive": str(self.minute_truth.index[-1] + MINUTE),
            "truth_interval_label": "minute_start",
        }

    def _window(self, timestamp, horizon) -> tuple[pd.Timestamp, pd.DataFrame]:
        timestamp = _clock(timestamp, self.minute_truth.index)
        horizon = pd.Timedelta(horizon)
        if pd.isna(horizon) or horizon <= pd.Timedelta(0) or horizon % INTERVAL:
            raise ValueError("Oracle horizon must be positive and contain complete 15-minute intervals")
        expected = pd.date_range(timestamp, timestamp + horizon, freq=MINUTE, inclusive="left")
        rows = self.minute_truth.reindex(expected)
        if rows.loc[:, list(TRUTH_COLUMNS)].isna().any().any():
            raise ValueError("Oracle native truth lacks complete requested horizon coverage")
        return timestamp, rows

    def _observations(self, timestamp, completed_history):
        if not isinstance(completed_history, pd.DataFrame):
            raise TypeError("Completed observations must be a DataFrame")
        if not completed_history.empty:
            index = completed_history.index
            if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
                    or not index.is_monotonic_increasing or str(index.tz) != str(timestamp.tz)):
                raise ValueError("Completed observations must use unique increasing times on the issue clock")
            if index[-1] > timestamp:
                raise ValueError("Completed observations contain a future/uncompleted interval")


class PerfectHomeHubForecast(_PerfectLoadTruth):
    """Offline oracle for all four Home Hub interval outputs, with exact clipping."""

    def __init__(self, minute_truth: pd.DataFrame):
        super().__init__(minute_truth)
        self.metadata["target"] = "four exact non-EV Home Hub/V2H-eligible 15-minute aggregates; excludes resistance"

    def issue(self, timestamp, completed_history, horizon, inverter_kw) -> EVHubIntervalForecast:
        timestamp, rows = self._window(timestamp, horizon)
        self._observations(timestamp, completed_history)
        if not np.isfinite(inverter_kw) or inverter_kw <= 0:
            raise ValueError("EV inverter limit must be positive and finite")
        rows["eligible_capped_kw"] = rows.eligible_load_kw.clip(upper=inverter_kw)
        groups = rows.resample(INTERVAL, label="left", closed="left")
        intervals = pd.DataFrame({
            "eligible_load_mean_kw": groups.eligible_load_kw.mean(),
            "eligible_mean_kw": groups.eligible_capped_kw.mean(),
            "eligible_min_kw": groups.eligible_capped_kw.min(),
            "hub_mean_kw": groups.hub_non_ev_kw.mean(),
        })
        # The oracle experiment grants the future truth at issue time. Explicit
        # source/metadata identify that grant; this is not observational readiness.
        result = EVHubIntervalForecast(timestamp, timestamp, intervals, ORACLE_SOURCE)
        self.history.append({"timestamp": timestamp, "issued_at": timestamp,
                             "inputs_ready_at": timestamp, "source": ORACLE_SOURCE,
                             "perfect_forecast": True, "truth_end_exclusive": timestamp + pd.Timedelta(horizon)})
        return result


class PerfectOutsideHubForecast(_PerfectLoadTruth):
    """Offline oracle of Outside-Hub aggregate power including electric resistance."""

    requires_total_home_meter = True
    method_label = ORACLE_SOURCE

    def __init__(self, minute_truth: pd.DataFrame):
        super().__init__(minute_truth)
        self.metadata["target"] = "15-minute mean Outside-Hub aggregate including HVAC electric resistance"

    def forecast_kw(self, timestamp, *, steps=32) -> pd.Series:
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("Oracle Outside-Hub steps must be a positive integer")
        timestamp, rows = self._window(timestamp, steps * INTERVAL)
        result = rows.outside_hub_with_resistance_kw.resample(INTERVAL, label="left", closed="left").mean()
        result.name = "outside_hub_with_resistance_forecast_kw"
        result.attrs.update(source=ORACLE_SOURCE, perfect_forecast=True,
                            information_policy="offline_oracle_future_native_truth",
                            truth_end_exclusive=timestamp + steps * INTERVAL)
        self.history.append({"timestamp": timestamp, "issued_at": timestamp,
                             "source": ORACLE_SOURCE, "perfect_forecast": True,
                             "steps": int(steps), "truth_end_exclusive": timestamp + steps * INTERVAL})
        return result

    def forecast_from_completed(self, timestamp, completed_history, *, steps=32) -> pd.Series:
        timestamp = _clock(timestamp, self.minute_truth.index)
        self._observations(timestamp, completed_history)
        return self.forecast_kw(timestamp, steps=steps)


class PerfectMobilityEVForecast:
    """One probability-one true mobility path through the existing scenario math.

    Returning a scenario set retains departure-time trip impulses, including an
    already-away bout, and the observed return time for ongoing comfort grace.
    It never applies the historical provider's persistence-prefix modification.
    """

    def __init__(self, vehicle_id: str, *, perfect_tariff: bool = False):
        self.vehicle_id = vehicle_id
        self.perfect_tariff = bool(perfect_tariff)
        self.history: list[dict] = []
        self.issued_forecasts: list[tuple] = []
        self.metadata = {"method_label": "perfect_mobility_offline_oracle",
                         "perfect_forecast": True, "scenario_count": 1,
                         "information_policy": "offline_oracle_future_mobility_truth",
                         "perfect_tariff": self.perfect_tariff,
                         "trip_energy_timing": "departure"}

    def at(self, timestamp, realized, horizon) -> EVScenarioSet:
        mobility = realized.ev_schedules[self.vehicle_id]
        timestamp = _clock(timestamp, pd.DatetimeIndex([mobility.start]))
        horizon = pd.Timedelta(horizon)
        if pd.isna(horizon) or horizon <= pd.Timedelta(0) or horizon % INTERVAL:
            raise ValueError("Perfect mobility horizon must contain complete 15-minute intervals")
        end = timestamp + horizon
        if timestamp < mobility.start or end > mobility.end:
            raise ValueError("Perfect EV mobility lacks complete requested horizon coverage")
        expected = pd.date_range(timestamp, end, freq=MINUTE, inclusive="left")
        if not realized.data.index.intersection(expected).equals(expected):
            raise ValueError("Perfect EV mobility needs complete minute tariff horizon coverage")
        if (realized.ev_hub_intervals is None) == (realized.backed_load_kw is None):
            raise ValueError("Perfect EV mobility needs either interval Hub or minute backed-load inputs")
        rows = mobility.trips.loc[(mobility.trips.arrival > timestamp) & (mobility.trips.departure < end)].copy()
        identifier = "perfect_actual_mobility"
        path = EVSchedule(rows, timestamp, end, {**mobility.metadata, **self.metadata,
                                               "candidate_id": identifier,
                                               "last_observed_arrival": last_observed_arrival(mobility, timestamp),
                                               "source_origin": str(timestamp), "source_end": str(end)})
        data = realized.data.copy()
        price_at = realized.tariff_price_at
        known_at = realized.tariff_price_as_known_at
        if self.perfect_tariff and price_at is None:
            raise ValueError("Perfect tariff foresight requires the complete realized tariff callable")
        if self.perfect_tariff or known_at is not None:
            prices = [price_at(target) if self.perfect_tariff else known_at(timestamp, target)
                      for target in expected]
            if not np.isfinite(prices).all():
                raise ValueError("EV oracle tariff horizon must contain finite prices")
            from .forecast import PRICE_COLUMN

            data.loc[expected, PRICE_COLUMN] = prices
        forecast = replace(realized, data=data,
                           ev_schedules={**realized.ev_schedules, self.vehicle_id: path}, providers={},
                           tariff_price_as_known_at=None if self.perfect_tariff else known_at)
        diagnostics = {**self.metadata, "timestamp": str(timestamp), "requested_count": 1,
                       "candidate_count": 1, "candidate_ids": [identifier],
                       "forecast_hours": horizon.total_seconds() / 3600,
                       "optimization_probability": 1.0, "probability_policy": "uniform",
                       "mobility_replay_metadata": mobility.metadata, "no_scenarios_selected": False}
        self.history.append(diagnostics)
        self.issued_forecasts.append((timestamp, horizon, (path,)))
        return EVScenarioSet(timestamp, (path,), (forecast.backed_load_kw,), (forecast,),
                             (identifier,), diagnostics)

"""Causal use of OikoLab's archived daily GFS forecasts for HVAC MPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "America/Indiana/Indianapolis"
FORECAST_WEATHER_COLUMNS = {
    "temp_air": "Ambient Dry Bulb (C)",
    "relative_humidity": "Ambient Relative Humidity (-)",
    "ghi": "GHI (W/m^2)",
    "dni": "DNI (W/m^2)",
    "dhi": "DHI (W/m^2)",
}
FORECAST_TIME_COLUMNS = ("run_time_utc", "valid_time_utc", "valid_time_local")


def _to_utc(timestamp: pd.Timestamp, local_timezone: str) -> pd.Timestamp:
    """Interpret OCHRE's naive clock in the declared local timezone."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            local_timezone, ambiguous="raise", nonexistent="raise"
        )
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class ArchivedWeatherForecast:
    """One table of immutable GFS runs indexed by UTC initialization and valid time.

    Each decision uses the latest archived 00Z run initialized no later than
    the decision's UTC day. A selected run is never replaced partway through
    the prediction horizon, even across the next 00Z.
    """

    data: pd.DataFrame
    local_timezone: str = LOCAL_TIMEZONE
    _runs: dict[pd.Timestamp, pd.DataFrame] = field(init=False, repr=False)
    _run_times: pd.DatetimeIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        missing = (set(FORECAST_TIME_COLUMNS) | set(FORECAST_WEATHER_COLUMNS)) - set(self.data)
        if missing:
            raise ValueError(f"Weather forecast archive is missing columns: {sorted(missing)}")
        for column in FORECAST_TIME_COLUMNS:
            if not isinstance(self.data[column].dtype, pd.DatetimeTZDtype):
                raise ValueError(f"{column} must contain timezone-aware timestamps")
        run_times = self.data["run_time_utc"].dt.tz_convert("UTC")
        valid_times = self.data["valid_time_utc"].dt.tz_convert("UTC")
        if self.data.empty:
            raise ValueError("Weather forecast archive is empty")
        if (run_times.dt.hour != 0).any() or (run_times.dt.minute != 0).any() or (run_times.dt.second != 0).any():
            raise ValueError("Archived weather forecast runs must initialize at 00Z")
        if (valid_times <= run_times).any():
            raise ValueError("Forecast valid times must follow their run time")
        if pd.MultiIndex.from_arrays((run_times, valid_times)).has_duplicates:
            raise ValueError("Weather forecast archive has duplicate run/valid times")
        if not np.isfinite(self.data[list(FORECAST_WEATHER_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError("Weather forecast archive contains nonfinite weather values")
        if not self.data["relative_humidity"].between(0, 1).all():
            raise ValueError("Weather forecast relative humidity must be between 0 and 1")
        for column in ("ghi", "dni", "dhi"):
            if (self.data[column] < 0).any():
                raise ValueError(f"Weather forecast {column} must be nonnegative")
        expected_local = pd.DatetimeIndex(valid_times.dt.tz_convert(self.local_timezone))
        actual_local = pd.DatetimeIndex(
            self.data["valid_time_local"].dt.tz_convert(self.local_timezone)
        )
        if not expected_local.equals(actual_local):
            raise ValueError("valid_time_local does not match valid_time_utc")
        runs = {
            pd.Timestamp(run_time): frame.sort_values("valid_time_utc").copy()
            for run_time, frame in self.data.groupby(run_times, sort=False)
        }
        hour_ns = pd.Timedelta(hours=1).value
        for run_time, frame in runs.items():
            valid = pd.DatetimeIndex(frame["valid_time_utc"].dt.tz_convert("UTC"))
            lead_ns = valid.asi8 - run_time.value
            if (lead_ns % hour_ns).any():
                raise ValueError(f"Run {run_time} has forecast points off the hourly grid")
            last_hour = int(lead_ns[-1] // hour_ns)
            expected = np.r_[
                np.arange(1, min(last_hour, 120) + 1),
                np.arange(123, last_hour + 1, 3),
            ]
            if not np.array_equal(lead_ns // hour_ns, expected):
                raise ValueError(
                    f"Run {run_time} has missing or irregular forecast valid times"
                )
        object.__setattr__(self, "_runs", runs)
        object.__setattr__(self, "_run_times", pd.DatetimeIndex(sorted(runs)))

    @classmethod
    def from_parquet(
        cls, path: str | Path, *, local_timezone: str = LOCAL_TIMEZONE
    ) -> "ArchivedWeatherForecast":
        """Read the prepared run/valid-time archive produced by the weather pull."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Prepared weather forecast archive is missing: {path}")
        return cls(pd.read_parquet(path), local_timezone=local_timezone)

    def selected_run_time(self, decision_time: pd.Timestamp) -> pd.Timestamp:
        """Use the latest available 00Z run no later than this UTC decision."""
        decision_utc = _to_utc(decision_time, self.local_timezone)
        position = self._run_times.searchsorted(decision_utc, side="right") - 1
        if position < 0:
            raise ValueError(
                f"No archived 00Z weather forecast at or before {decision_utc}"
            )
        return self._run_times[position]

    def validate_coverage(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        horizon: pd.Timedelta,
        decision_step: pd.Timedelta,
    ) -> None:
        """Check the selected run for each UTC decision day up front."""
        start_utc = _to_utc(start_time, self.local_timezone)
        end_utc = _to_utc(end_time, self.local_timezone)
        horizon = pd.Timedelta(horizon)
        decision_step = pd.Timedelta(decision_step)
        if end_utc <= start_utc or horizon <= pd.Timedelta(0) or decision_step <= pd.Timedelta(0):
            raise ValueError("Forecast coverage requires positive window, horizon, and decision step")
        decisions = pd.date_range(
            start_utc.ceil(decision_step), end_utc, freq=decision_step, inclusive="left"
        )
        decision_days = decisions.floor("D")
        for decision_day in decision_days.unique():
            run_time = self.selected_run_time(decision_day)
            last_decision = decisions[decision_days == decision_day][-1]
            valid_utc = pd.DatetimeIndex(
                self._runs[run_time]["valid_time_utc"].dt.tz_convert("UTC")
            )
            hold_end = last_decision + decision_step
            if valid_utc[-1] < last_decision + horizon:
                raise ValueError(
                    f"Run {run_time} ends at {valid_utc[-1]} before the final HVAC horizon"
                )
            if valid_utc.searchsorted(hold_end, side="right") >= len(valid_utc):
                raise ValueError(f"Run {run_time} lacks weather after the first held HVAC step")

    def weather_for_hvac(
        self,
        decision_time: pd.Timestamp,
        target_index: pd.DatetimeIndex,
        realized_now: pd.Series,
        hold_duration: pd.Timedelta,
    ) -> pd.DataFrame:
        """Hold current weather for one MPC step, then blend within one run.

        The first forecast anchor is the earliest archived valid time *strictly
        after* the held step. All later values are linearly interpolated between
        valid times of that same run. No future realized weather is consulted.
        """
        if not isinstance(target_index, pd.DatetimeIndex) or target_index.empty:
            raise ValueError("HVAC weather target must be a nonempty DatetimeIndex")
        if target_index.has_duplicates or not target_index.is_monotonic_increasing:
            raise ValueError("HVAC weather target timestamps must be unique and increasing")
        hold_duration = pd.Timedelta(hold_duration)
        if hold_duration <= pd.Timedelta(0):
            raise ValueError("HVAC weather hold duration must be positive")
        decision_utc = _to_utc(decision_time, self.local_timezone)
        target_utc = pd.DatetimeIndex(target_index)
        if target_utc.tz is None:
            target_utc = target_utc.tz_localize(
                self.local_timezone, ambiguous="raise", nonexistent="raise"
            )
        target_utc = target_utc.tz_convert("UTC")
        if target_utc[0] != decision_utc or (target_utc < decision_utc).any():
            raise ValueError("HVAC weather target must start at the decision time")

        columns = list(FORECAST_WEATHER_COLUMNS)
        schedule_columns = list(FORECAST_WEATHER_COLUMNS.values())
        if any(column not in realized_now for column in schedule_columns):
            raise ValueError("Current realized weather lacks an HVAC forecast field")
        observed = realized_now[schedule_columns].to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError("Current realized weather contains nonfinite HVAC fields")

        run_time = self.selected_run_time(decision_time)
        run = self._runs[run_time].loc[:, ["valid_time_utc", *columns]]
        valid_utc = pd.DatetimeIndex(run["valid_time_utc"].dt.tz_convert("UTC"))
        hold_end = decision_utc + hold_duration
        first_pos = valid_utc.searchsorted(hold_end, side="right")
        if first_pos >= len(valid_utc):
            raise ValueError(f"Run {run_time} has no forecast point after the held HVAC step")
        if target_utc[-1] > valid_utc[-1]:
            raise ValueError(
                f"Run {run_time} ends at {valid_utc[-1]}, before HVAC horizon ends at {target_utc[-1]}"
            )
        anchor_times = np.r_[hold_end.value, valid_utc[first_pos:].asi8]
        values = np.vstack((observed, run[columns].iloc[first_pos:].to_numpy(dtype=float)))
        result = np.column_stack([
            np.interp(target_utc.asi8, anchor_times, values[:, position])
            for position in range(len(columns))
        ])
        return pd.DataFrame(result, index=target_index, columns=schedule_columns)

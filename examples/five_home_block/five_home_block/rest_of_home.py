"""Measured rest-of-home electricity from reserved forecast test windows.

This is plant input data, not a forecast model. The source CSV is never edited.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RestOfHomeConfig


TEST_START = pd.Timestamp("2026-09-15")
TEST_END = pd.Timestamp("2026-09-29")
# The later week is held out by both the simulated Home Hub and Outside-Hub
# forecast studies. Keep it separate from the original rest-of-home test split.
HOME_HUB_TEST_START = pd.Timestamp("2026-10-04")
HOME_HUB_TEST_END = pd.Timestamp("2026-10-11")
MINUTE = pd.Timedelta(minutes=1)
POWER_COLUMN = "rest_of_home_power_kw"


def _wall_time(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize(None) if value.tz is not None else value


@dataclass
class RestOfHomeReplay:
    """Interval-start measurements mapped by a fixed local calendar offset."""

    data: pd.DataFrame
    simulation_start: pd.Timestamp
    source_start: pd.Timestamp
    source_path: Path
    missing_data_policy: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_split: str


    @classmethod
    def from_config(
        cls, config: RestOfHomeConfig, *, simulation_start: pd.Timestamp,
    ) -> "RestOfHomeReplay":
        if config.source != "dc_house_test" or config.data_file is None:
            raise ValueError("DC House replay requires source=dc_house_test and a data_file")
        if config.missing_data_policy not in {"error", "linear_interpolate"}:
            raise ValueError("Unsupported rest-of-home missing_data_policy")
        simulation_start = pd.Timestamp(simulation_start)
        wall = _wall_time(simulation_start)
        source_start = pd.Timestamp(config.source_start)
        if source_start.tz is not None:
            raise ValueError("rest_of_home.source_start must use naive local time")
        if len(config.source_start) == 10:
            source_start += wall - wall.normalize()
        if wall.time() != source_start.time():
            raise ValueError("Rest-of-home source and simulation starts must have the same local clock time")
        if TEST_START <= source_start < TEST_END:
            test_start, test_end = TEST_START, TEST_END
            test_split = "rest_of_home_forecast_test"
        elif HOME_HUB_TEST_START <= source_start < HOME_HUB_TEST_END:
            test_start, test_end = HOME_HUB_TEST_START, HOME_HUB_TEST_END
            test_split = "home_hub_outside_hub_held_out_test"
        else:
            raise ValueError(
                "Rest-of-home source_start must fall in the September 15–28, 2026 "
                "rest-of-home forecast test or October 4–10, 2026 Home Hub/Outside-Hub test"
            )
        if wall != wall.floor("min"):
            raise ValueError("Rest-of-home simulation start must align to a whole minute")
        raw = pd.read_csv(config.data_file)
        if not {"Time", "Value", "Quality"}.issubset(raw.columns) or raw.empty:
            raise ValueError("DC House rest-of-home CSV needs Time, Value, and Quality columns")
        ends = pd.DatetimeIndex(pd.to_datetime(raw["Time"], errors="raise"))
        if (ends.tz is not None or ends.hasnans or ends.has_duplicates
                or not ends.is_monotonic_increasing or not ends.equals(ends.floor("min"))):
            raise ValueError("DC House timestamps must be naive, unique, increasing whole minutes")
        # The 00:01 source row is the mean power during [00:00, 00:01).
        raw.index = ends - MINUTE
        raw = raw.reindex(pd.date_range(test_start, test_end, freq=MINUTE, inclusive="left"))
        values = pd.to_numeric(raw["Value"], errors="raise").astype(float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError("DC House rest-of-home watts must be nonnegative and finite or missing")
        approved = raw["Quality"].isin(["observed", "interpolated"])
        values = values.where(approved)
        data = pd.DataFrame({
            POWER_COLUMN: values / 1000.0,
            "source_quality": raw["Quality"].fillna("unavailable"),
        })
        data["simulation_interpolated"] = False
        if config.missing_data_policy == "linear_interpolate":
            # Specifically approved for the holdout's bounded 1–4 minute gaps.
            # Never use training data, extrapolate endpoints, or partially fill
            # a longer gap if another source file is selected.
            missing = values.isna()
            run_lengths = missing.groupby((~missing).cumsum()).transform("sum")
            filled = (values / 1000.0).interpolate(method="time", limit_area="inside")
            use_fill = missing & (run_lengths <= 4) & filled.notna()
            data.loc[use_fill, POWER_COLUMN] = filled.loc[use_fill]
            data.loc[use_fill, "simulation_interpolated"] = True
        return cls(
            data, simulation_start, source_start, config.data_file,
            config.missing_data_policy, test_start, test_end, test_split,
        )

    def _sample_minutes(
        self, index: pd.DatetimeIndex, timestep: pd.Timedelta | None,
    ) -> tuple[pd.DatetimeIndex, pd.DataFrame, int]:
        if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
                or index.has_duplicates or not index.is_monotonic_increasing):
            raise ValueError("Rest-of-home simulation timestamps must be unique and increasing")
        step = pd.Timedelta(timestep) if timestep is not None else (
            index[1] - index[0] if len(index) > 1 else MINUTE
        )
        minutes = step / MINUTE
        if not np.isfinite(minutes) or minutes < 1 or minutes != int(minutes):
            raise ValueError("Rest-of-home timestep must be a whole positive number of minutes")
        if len(index) and not index.equals(pd.date_range(index[0], periods=len(index), freq=step)):
            raise ValueError("Rest-of-home simulation timestamps must be regularly spaced")
        wall = index.tz_localize(None) if index.tz is not None else index
        if not wall.equals(wall.floor("min")):
            raise ValueError("Rest-of-home timestamps must align to whole minutes")
        starts = wall + (self.source_start - _wall_time(self.simulation_start))
        if len(starts) and (starts[0] < self.test_start or starts[-1] + step > self.test_end):
            raise ValueError(
                f"Rest-of-home replay exceeds the {self.test_split} split "
                f"[{self.test_start}, {self.test_end}); "
                "shorten the run, change source_start, or select source=ochre. Data is not repeated."
            )
        expanded = pd.DatetimeIndex(
            np.repeat(starts.to_numpy(), int(minutes))
            + np.tile(pd.to_timedelta(np.arange(int(minutes)), unit="min").to_numpy(), len(starts))
        )
        selected = self.data.reindex(expanded)
        missing = selected[POWER_COLUMN].isna()
        if missing.any():
            first = selected.index[missing][0]
            raise ValueError(
                f"DC House rest-of-home test data has {int(missing.sum())} excluded or missing "
                f"minute(s) in this replay, first interval start {first}; "
                "choose another source_start or select source=ochre."
            )
        return starts, selected, int(minutes)

    def power_at(
        self, index: pd.DatetimeIndex, *, timestep: pd.Timedelta | None = None,
    ) -> pd.Series:
        """Return interval-mean kW under the configured gap policy, without repetition."""
        _, selected, minutes = self._sample_minutes(index, timestep)
        values = selected[POWER_COLUMN].to_numpy().reshape(len(index), minutes).mean(axis=1)
        return pd.Series(values, index=index, name=POWER_COLUMN)

    def audit_frame(
        self, index: pd.DatetimeIndex, *, timestep: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Expose source alignment and quality for every replayed simulation step."""
        starts, selected, minutes = self._sample_minutes(index, timestep)
        values = selected[POWER_COLUMN].to_numpy().reshape(len(index), minutes).mean(axis=1)
        approved_fills = (selected["source_quality"] == "interpolated").to_numpy()
        simulation_fills = selected["simulation_interpolated"].to_numpy()
        return pd.DataFrame({
            "source_interval_start": starts,
            POWER_COLUMN: values,
            "source_interpolated_minutes": approved_fills.reshape(len(index), minutes).sum(axis=1),
            "simulation_interpolated_minutes": simulation_fills.reshape(len(index), minutes).sum(axis=1),
        }, index=index.rename("timestamp"))

    @property
    def interpolated_minutes(self) -> pd.DataFrame:
        """List each newly filled source minute with its original quality flag."""
        return self.data.loc[
            self.data["simulation_interpolated"], ["source_quality", POWER_COLUMN]
        ].rename_axis("source_interval_start").reset_index()

    @property
    def metadata(self) -> dict[str, object]:
        metadata = {
            "source": "dc_house_test",
            "source_file": str(self.source_path),
            "test_split": self.test_split,
            "test_start": self.test_start.isoformat(),
            "test_end_exclusive": self.test_end.isoformat(),
            "source_start": self.source_start.isoformat(),
            "simulation_start": self.simulation_start.isoformat(),
            "source_units": "W",
            "simulation_units": "kW",
            "source_timestamp_position": "interval end",
            "simulation_timestamp_position": "interval start",
            "mapping": "fixed local calendar-date offset; no repetition",
            "missing_data_policy": self.missing_data_policy,
            "test_split_simulation_interpolated_minutes": int(self.data["simulation_interpolated"].sum()),
            "interpolation_max_gap_minutes": 4,
            "native_internal_heat_gains": "preserved",
            "warmup_electric_load": "native OCHRE",
            "reactive_power": "native rest-of-home removed; DC House has real power only",
        }
        if self.test_split == "home_hub_outside_hub_held_out_test":
            metadata["forecast_fit_label_end_exclusive"] = HOME_HUB_TEST_START.isoformat()
            metadata["held_out_from_training_for"] = [
                "simulated_home_hub", "outside_hub",
            ]
        return metadata

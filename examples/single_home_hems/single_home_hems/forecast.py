"""Forecast data shared by the individual predictive controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

import pandas as pd

if TYPE_CHECKING:
    from .water_forecast import ScenarioWaterForecast


PRICE_COLUMN = "electricity_price_usd_per_kwh"


class PriceSchedule(Protocol):
    """Common interface for TOU or future day-ahead hourly prices."""

    def price_at(self, timestamp: pd.Timestamp) -> float:
        """Return the price in dollars per kWh at ``timestamp``."""
        ...


@dataclass(frozen=True)
class TimeOfUsePrice:
    """Simple daily time-of-use electricity price."""

    off_peak_price: float = 0.15
    peak_price: float = 0.30
    peak_start_hour: int = 15
    peak_end_hour: int = 18

    def is_peak(self, timestamp: pd.Timestamp) -> bool:
        """Return whether ``timestamp`` is in the 3-6 p.m. peak period."""
        return self.peak_start_hour <= timestamp.hour < self.peak_end_hour

    def price_at(self, timestamp: pd.Timestamp) -> float:
        """Return the price in dollars per kWh at ``timestamp``."""
        return self.peak_price if self.is_peak(timestamp) else self.off_peak_price


@dataclass(frozen=True)
class Forecast:
    """Weather, schedules, prices, and an optional causal water forecast."""

    data: pd.DataFrame
    water_scenarios: ScenarioWaterForecast | None = None
    ev_schedules: dict = field(default_factory=dict)
    backed_load_kw: pd.Series | None = None
    providers: dict = field(default_factory=dict)
    # Known tariff, separate from the finite device forecast. Retained when an
    # external forecast replaces load/mobility/price rows; not a realized tail.
    tariff_price_at: Callable[[pd.Timestamp], float] | None = None

    def for_device(self, name, timestamp, horizon):
        """Route a device forecast; leave shared measured history/tariff available."""
        provider = self.providers.get(name)
        return self if provider is None else provider.at(timestamp, self, horizon)

    @classmethod
    def from_schedule(
        cls,
        schedule: pd.DataFrame,
        price_at: Callable[[pd.Timestamp], float],
        *,
        water_scenarios: ScenarioWaterForecast | None = None,
        ev_schedules: dict | None = None,
        backed_load_kw: pd.Series | None = None,
    ) -> "Forecast":
        """Build a forecast from the deterministic OCHRE input schedule."""
        if not isinstance(schedule.index, pd.DatetimeIndex):
            raise TypeError("Forecast schedule must use a DatetimeIndex")
        if (
            schedule.index.has_duplicates
            or not schedule.index.is_monotonic_increasing
        ):
            raise ValueError("Forecast timestamps must be unique and increasing")

        data = schedule.copy()
        data[PRICE_COLUMN] = [price_at(pd.Timestamp(time)) for time in data.index]
        return cls(data=data, water_scenarios=water_scenarios,
                   ev_schedules=ev_schedules or {}, backed_load_kw=backed_load_kw,
                   tariff_price_at=price_at)

    def window(
        self,
        timestamp: pd.Timestamp,
        horizon: pd.Timedelta,
    ) -> pd.DataFrame:
        """Return available forecast rows in ``[timestamp, timestamp + horizon)``."""
        if horizon <= pd.Timedelta(0):
            raise ValueError("Prediction horizon must be positive")
        end = timestamp + horizon
        selected = (self.data.index >= timestamp) & (self.data.index < end)
        return self.data.loc[selected].copy()

    def price_at(self, timestamp: pd.Timestamp) -> float:
        """Return the forecast electricity price at one timestep."""
        return float(self.data.at[timestamp, PRICE_COLUMN])

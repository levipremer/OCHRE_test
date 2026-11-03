"""Study-defined winter TOU, daily-demand, CPP, and MISO RTP rate choices.

The Indiana and Carolinas tariffs supply price benchmarks. The weekday-only
peak rule, daily demand amount, and combined CPP schedule are study choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


WINTER_TOU = "tou-winter"
WINTER_TOU_DEMAND = "tou-winter+demand"
WINTER_TOU_DEMAND_CPP = "tou-winter+demand+CPP"
MISO_RTP_DOE = "rtp-miso+doe"
WINTER_RATE_CHOICES = (
    WINTER_TOU,
    WINTER_TOU_DEMAND,
    WINTER_TOU_DEMAND_CPP,
)

DISCOUNT_PRICE_USD_PER_KWH = 0.086
OFF_PEAK_PRICE_USD_PER_KWH = 0.143
ON_PEAK_PRICE_USD_PER_KWH = 0.214
CPP_ADDER_USD_PER_KWH = 0.190
DEMAND_CHARGE_USD_PER_KW_DAY = 0.71
GRID2_TOU_PRICE_MULTIPLIER = 0.75
GRID2_DEMAND_AMORTIZATION_WEEKDAYS = 21
GRID2_ON_PEAK_DEMAND_USD_PER_KW_MONTH = 2.34
GRID2_ANYTIME_DEMAND_USD_PER_KW_MONTH = 4.47
GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY = (
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_MONTH / GRID2_DEMAND_AMORTIZATION_WEEKDAYS
)
GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY = (
    GRID2_ANYTIME_DEMAND_USD_PER_KW_MONTH / GRID2_DEMAND_AMORTIZATION_WEEKDAYS
)
CPP_NOTICE_LOCAL_TIME = time(16, 0)
MISO_DA_NOTICE_LOCAL_TIME = time(13, 30)
LOCAL_TIMEZONE = "America/Indiana/Indianapolis"
MISO_CIN_PSI_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "external_data/miso_cin_psi/"
    "cin_psi_da_expost_lmp_2025-11-30_to_2025-12-08_hourly.csv"
)


def _local_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Treat naive timestamps as local; convert aware instants to local time."""
    value = pd.Timestamp(timestamp)
    return value if value.tzinfo is None else value.tz_convert(LOCAL_TIMEZONE)


def _aware_local_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Interpret naive timestamps as local and convert aware instants to local."""
    value = pd.Timestamp(timestamp)
    return (
        value.tz_localize(LOCAL_TIMEZONE)
        if value.tzinfo is None
        else value.tz_convert(LOCAL_TIMEZONE)
    )


@dataclass(frozen=True)
class WinterTOURate:
    """One of the three winter study rates, using local wall-clock times.

    ``price_at`` is the actual billed energy price. A controller should instead
    receive ``forecast_callback(issue_time)`` at each decision: before the
    prior-day 16:00 notice that callback gives the ordinary TOU price for the
    future CPP event, and afterward it includes the CPP adder.
    """

    choice: str = WINTER_TOU
    cpp_event_date: date | None = None

    def __post_init__(self) -> None:
        if self.choice not in WINTER_RATE_CHOICES:
            raise ValueError(
                f"Unknown winter rate {self.choice!r}; choose from {WINTER_RATE_CHOICES}"
            )
        has_cpp = self.choice == WINTER_TOU_DEMAND_CPP
        if has_cpp and self.cpp_event_date is None:
            raise ValueError("The CPP rate requires cpp_event_date")
        if not has_cpp and self.cpp_event_date is not None:
            raise ValueError("cpp_event_date applies only to the CPP rate")
        if self.cpp_event_date is not None:
            if type(self.cpp_event_date) is not date:
                raise TypeError("cpp_event_date must be a local datetime.date")
            if self.cpp_event_date.weekday() >= 5:
                raise ValueError("The CPP event must be on a weekday")

    @property
    def demand_charge_usd_per_kw_day(self) -> float:
        """Compatibility alias for the on-peak component, excluding anytime."""
        return self.on_peak_demand_charge_usd_per_kw_day

    @property
    def on_peak_demand_charge_usd_per_kw_day(self) -> float:
        """Daily weekday charge for the combined on-peak 30-minute maximum."""
        if self.choice == WINTER_TOU_DEMAND_CPP:
            return GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY
        return (
            0.0
            if self.choice == WINTER_TOU
            else DEMAND_CHARGE_USD_PER_KW_DAY
        )

    @property
    def anytime_demand_charge_usd_per_kw_day(self) -> float:
        """Grid 2's additional weekday 24-hour 30-minute maximum charge."""
        return (GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY
                if self.choice == WINTER_TOU_DEMAND_CPP else 0.0)

    @property
    def energy_price_multiplier(self) -> float:
        """The approved 25% ordinary-energy reduction applies to Grid 2 only."""
        return GRID2_TOU_PRICE_MULTIPLIER if self.choice == WINTER_TOU_DEMAND_CPP else 1.0

    @property
    def discount_price_usd_per_kwh(self) -> float:
        return DISCOUNT_PRICE_USD_PER_KWH * self.energy_price_multiplier

    @property
    def off_peak_price_usd_per_kwh(self) -> float:
        return OFF_PEAK_PRICE_USD_PER_KWH * self.energy_price_multiplier

    @property
    def on_peak_price_usd_per_kwh(self) -> float:
        return ON_PEAK_PRICE_USD_PER_KWH * self.energy_price_multiplier

    @property
    def cpp_notice_at(self) -> pd.Timestamp | None:
        """Naive timestamp for the event's preceding local day at 16:00."""
        if self.cpp_event_date is None:
            return None
        notice_date = self.cpp_event_date - timedelta(days=1)
        return pd.Timestamp.combine(notice_date, CPP_NOTICE_LOCAL_TIME)

    @staticmethod
    def is_discount(timestamp: pd.Timestamp) -> bool:
        """The daily 00:00–04:00 discount applies on weekdays and weekends."""
        return _local_timestamp(timestamp).hour < 4

    @staticmethod
    def is_on_peak(timestamp: pd.Timestamp) -> bool:
        """Both winter peak windows share one weekday on-peak designation."""
        target = _local_timestamp(timestamp)
        return target.weekday() < 5 and (6 <= target.hour < 8 or 17 <= target.hour < 21)

    def _price_at(self, timestamp: pd.Timestamp, *, cpp_visible: bool) -> float:
        target = _local_timestamp(timestamp)
        if self.is_discount(target):
            return self.discount_price_usd_per_kwh
        if not self.is_on_peak(target):
            return self.off_peak_price_usd_per_kwh
        price = self.on_peak_price_usd_per_kwh
        if (
            cpp_visible
            and self.cpp_event_date == target.date()
            and 6 <= target.hour < 8
        ):
            price += CPP_ADDER_USD_PER_KWH
        return price

    def price_at(self, timestamp: pd.Timestamp) -> float:
        """Return the energy price actually billed at the target time."""
        return self._price_at(timestamp, cpp_visible=True)

    def forecast_callback(
        self, issued_at: pd.Timestamp
    ) -> Callable[[pd.Timestamp], float]:
        """Freeze the energy prices known when this forecast is issued."""
        issue = _local_timestamp(issued_at)
        notice = self.cpp_notice_at
        cpp_visible = notice is not None and (
            issue.date() > notice.date()
            or (issue.date() == notice.date() and issue.time() >= CPP_NOTICE_LOCAL_TIME)
        )
        return lambda target: self._price_at(target, cpp_visible=cpp_visible)


@dataclass(frozen=True)
class MISOMultiplicativeRTPRate:
    """Historical CIN.PSI hourly LMP scaled to each winter TOU daily mean.

    The source includes November 30 for controller warmup and the full
    December 8 market day for the final eight-hour forecast.
    This class supplies a historical replay price path; it does not assert when
    the source report's ExPost values became available to an operator.
    The approved archived source has 24 hours on every local market day.
    Clock-change days require a separately specified 23/25-hour normalization
    and same-clock forecast mapping, so they fail source validation.
    """

    prices_by_hour: dict[pd.Timestamp, float]
    lmp_by_hour: dict[pd.Timestamp, float]
    daily_target_usd_per_kwh: dict[date, float]
    daily_multiplier: dict[date, float]
    source_path: Path
    forecast_policy: str = "archived_expost_full"

    @classmethod
    def from_csv(
        cls, source_path: Path = MISO_CIN_PSI_SOURCE, *,
        forecast_policy: str = "archived_expost_full",
    ) -> "MISOMultiplicativeRTPRate":
        if forecast_policy not in {
            "archived_expost_full", "same_clock_current_day_until_publication",
        }:
            raise ValueError("Unknown MISO forecast policy")
        source = Path(source_path).resolve()
        frame = pd.read_csv(source)
        required = {"interval_start_local", "lmp_usd_per_mwh"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CIN.PSI source is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("CIN.PSI source has no hourly prices")

        times = pd.to_datetime(frame["interval_start_local"], utc=True).dt.tz_convert(
            LOCAL_TIMEZONE
        )
        wholesale = pd.to_numeric(frame["lmp_usd_per_mwh"], errors="raise") / 1000.0
        hourly = pd.Series(wholesale.to_numpy(dtype=float), index=pd.DatetimeIndex(times))
        hourly = hourly.sort_index()
        if not np.isfinite(hourly.to_numpy()).all():
            raise ValueError("CIN.PSI source contains non-finite LMPs")
        if hourly.index.has_duplicates:
            raise ValueError("CIN.PSI source contains duplicate hourly timestamps")
        expected = pd.date_range(hourly.index[0], periods=len(hourly), freq="h")
        if not hourly.index.equals(expected):
            raise ValueError("CIN.PSI source must have one price for every local hour")

        tou = WinterTOURate()
        prices: dict[pd.Timestamp, float] = {}
        targets: dict[date, float] = {}
        multipliers: dict[date, float] = {}
        for day, group in hourly.groupby(hourly.index.date):
            if len(group) != 24 or group.index.hour.tolist() != list(range(24)):
                raise ValueError(
                    "CIN.PSI source needs an unambiguous 24-hour local market "
                    f"day for {day}; DST transition days need a separately "
                    "approved same-clock mapping"
                )
            wholesale_mean = float(group.mean())
            if wholesale_mean <= 0:
                raise ValueError(
                    f"Multiplicative RTP scaling needs a positive daily LMP mean on {day}"
                )
            target = float(np.mean([tou.price_at(hour) for hour in group.index]))
            multiplier = target / wholesale_mean
            targets[day] = target
            multipliers[day] = multiplier
            prices.update({hour: float(price * multiplier) for hour, price in group.items()})
        return cls(prices, hourly.to_dict(), targets, multipliers, source, forecast_policy)

    @property
    def demand_charge_usd_per_kw_day(self) -> float:
        """The utility import envelope is separate from a demand-charge bill."""
        return 0.0

    @property
    def first_hour(self) -> pd.Timestamp:
        return min(self.prices_by_hour)

    @property
    def end_exclusive(self) -> pd.Timestamp:
        return max(self.prices_by_hour) + pd.Timedelta(hours=1)

    def validate_coverage(
        self, start: pd.Timestamp, end_exclusive: pd.Timestamp,
    ) -> None:
        start_local = _aware_local_timestamp(start)
        end_local = _aware_local_timestamp(end_exclusive)
        if start_local < self.first_hour or end_local > self.end_exclusive:
            raise ValueError(
                "The CIN.PSI hourly source does not cover the simulation and forecast: "
                f"requested [{start_local}, {end_local}), available "
                f"[{self.first_hour}, {self.end_exclusive})"
            )

    def price_at(self, timestamp: pd.Timestamp) -> float:
        """Return the scaled retail energy price in USD/kWh for a local hour."""
        hour = _aware_local_timestamp(timestamp).floor("h")
        try:
            return self.prices_by_hour[hour]
        except KeyError as exc:
            raise ValueError(f"No CIN.PSI price is available for {hour}") from exc

    def forecast_callback(
        self, issued_at: pd.Timestamp,
    ) -> Callable[[pd.Timestamp], float]:
        """Issue the selected archived or publication-aware hourly price view.

        With the joint runner's eight-hour horizon, the first forecast that
        reaches the next market day issues after the usual 13:30 Eastern
        day-ahead publication. The archive may contain subsequent revisions.
        """
        issue = _aware_local_timestamp(issued_at)
        if self.forecast_policy == "archived_expost_full":
            return self.price_at
        current_day_prices = {
            hour.hour: float(price) for hour, price in self.prices_by_hour.items()
            if hour.date() == issue.date()
        }
        if len(current_day_prices) != 24:
            raise ValueError("MISO same-clock fallback needs a complete current market day")

        def as_known_at(target: pd.Timestamp) -> float:
            local_target = _aware_local_timestamp(target)
            if local_target.date() <= issue.date():
                return self.price_at(local_target)
            # A twenty-hour issue can enter tomorrow's market day before the
            # prior-day announcement. Reuse today's same-clock hourly price
            # until that announcement, then use the archived next-day path.
            publication_day = local_target.date() - timedelta(days=1)
            publication = pd.Timestamp.combine(
                publication_day, MISO_DA_NOTICE_LOCAL_TIME,
            ).tz_localize(LOCAL_TIMEZONE)
            if issue >= publication:
                return self.price_at(local_target)
            if local_target.date() != issue.date() + timedelta(days=1):
                raise ValueError("MISO same-clock fallback only covers the next market day")
            return current_day_prices[local_target.hour]

        return as_known_at

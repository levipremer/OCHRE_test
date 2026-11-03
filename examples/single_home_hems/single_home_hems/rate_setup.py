"""Build the selected tariff and record how its CPP event was chosen."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cpp_event import select_retrospective_coldest_weekday
from .forecast import PriceSchedule, TimeOfUsePrice
from .rates import (
    CPP_ADDER_USD_PER_KWH,
    GRID2_ANYTIME_DEMAND_USD_PER_KW_MONTH,
    GRID2_DEMAND_AMORTIZATION_WEEKDAYS,
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_MONTH,
    MISO_RTP_DOE,
    MISOMultiplicativeRTPRate,
    WINTER_RATE_CHOICES,
    WINTER_TOU_DEMAND_CPP,
    WinterTOURate,
)


RATE_CHOICES = ("legacy", *WINTER_RATE_CHOICES, MISO_RTP_DOE)
RTP_FORECAST_TAIL = pd.Timedelta(hours=8)


@dataclass(frozen=True)
class RateSelection:
    schedule: PriceSchedule
    metadata: dict


def select_rate_for_simulation(
    choice: str, config, *,
    forecast_horizon: pd.Timedelta = RTP_FORECAST_TAIL,
    rtp_forecast_policy: str = "archived_expost_full",
) -> RateSelection:
    """Use published-rate adaptations and one retrospective CPP event."""
    if choice not in RATE_CHOICES:
        raise ValueError(f"Unknown rate structure {choice!r}")
    forecast_horizon = pd.Timedelta(forecast_horizon)
    if pd.isna(forecast_horizon) or forecast_horizon <= pd.Timedelta(0):
        raise ValueError("Rate forecast horizon must be positive")
    if choice == "legacy":
        return RateSelection(TimeOfUsePrice(), {"choice": choice})
    if choice == MISO_RTP_DOE:
        rate = MISOMultiplicativeRTPRate.from_csv(forecast_policy=rtp_forecast_policy)
        simulation_start = pd.Timestamp(config.start_time)
        simulation_end = simulation_start + pd.Timedelta(config.duration)
        rate.validate_coverage(simulation_start, simulation_end + forecast_horizon)
        causal_tail = rtp_forecast_policy == "same_clock_current_day_until_publication"
        return RateSelection(rate, {
            "choice": choice,
            "energy_price_method": "daily multiplicative CIN.PSI day-ahead ExPost LMP scaling",
            "source_file": str(rate.source_path),
            "source_market_report": "MISO DA ExPost LMP, CIN.PSI Loadzone",
            "source_first_hour_local": rate.first_hour.isoformat(),
            "source_end_exclusive_local": rate.end_exclusive.isoformat(),
            "daily_target_usd_per_kwh": {
                day.isoformat(): target for day, target in rate.daily_target_usd_per_kwh.items()
            },
            "daily_multiplier": {
                day.isoformat(): multiplier for day, multiplier in rate.daily_multiplier.items()
            },
            "demand_usd_per_kw_day": 0.0,
            "forecast_horizon_hours": forecast_horizon.total_seconds() / 3600,
            "forecast_policy": rtp_forecast_policy,
            "clock_change_scope": (
                "The archived winter source has 24 unambiguous local hours per day; "
                "23/25-hour DST market days require a separately approved mapping"
            ),
            "forecast_price_note": (
                "Before the next market day's 13:30 Eastern publication, use "
                "the current day's same-clock hourly price for unavailable "
                "future hours; afterward use the archived next-day path. "
                "Archived values may contain subsequent revisions."
                if causal_tail else
                "The controller replays the archived DA ExPost path as the issued "
                "hourly price forecast. The eight-hour horizon first reaches "
                "the next market day after the usual prior-day 13:30 Eastern "
                "publication; archived values may contain subsequent revisions."
            ),
            "reference_note": "examples/single_home_hems/RATE_SCENARIO_SOURCES.md",
        })

    event = None
    if choice == WINTER_TOU_DEMAND_CPP:
        metadata = config.weather_metadata or {}
        if config.weather_file.suffix.lower() != ".csv" or not metadata.get("timezone"):
            raise ValueError(
                "The CPP choice needs a dated realized-weather CSV with weather_metadata.timezone "
                "to select the coldest weekday morning"
            )
        event = select_retrospective_coldest_weekday(
            config.weather_file,
            pd.Timestamp(config.start_time),
            pd.Timedelta(config.duration),
            local_timezone=str(metadata["timezone"]),
        )

    rate = WinterTOURate(choice, cpp_event_date=None if event is None else event.event_date)
    details = {
        "choice": choice,
        "discount_usd_per_kwh": rate.discount_price_usd_per_kwh,
        "off_peak_usd_per_kwh": rate.off_peak_price_usd_per_kwh,
        "on_peak_usd_per_kwh": rate.on_peak_price_usd_per_kwh,
        "ordinary_tou_price_multiplier": rate.energy_price_multiplier,
        "weekday_peak_windows_local": ["06:00-08:00", "17:00-21:00"],
        "daily_discount_window_local": "00:00-04:00",
        "demand_usd_per_kw_day": rate.demand_charge_usd_per_kw_day,
        "on_peak_demand_usd_per_kw_day": rate.on_peak_demand_charge_usd_per_kw_day,
        "anytime_demand_usd_per_kw_day": rate.anytime_demand_charge_usd_per_kw_day,
        "demand_interval_minutes": 30,
        "demand_aggregation": "one daily maximum across both weekday peak windows",
        "cpp_adder_usd_per_kwh": CPP_ADDER_USD_PER_KWH if event is not None else 0.0,
        "reference_note": "examples/single_home_hems/RATE_SCENARIO_SOURCES.md",
    }
    if choice == WINTER_TOU_DEMAND_CPP:
        details.update(
            on_peak_demand_reference_usd_per_kw_month=GRID2_ON_PEAK_DEMAND_USD_PER_KW_MONTH,
            anytime_demand_reference_usd_per_kw_month=GRID2_ANYTIME_DEMAND_USD_PER_KW_MONTH,
            demand_amortization_weekdays=GRID2_DEMAND_AMORTIZATION_WEEKDAYS,
            demand_aggregation=(
                "sum of independent daily weekday on-peak and 24-hour maxima; "
                "both use complete clock-aligned 30-minute means"
            ),
            demand_weekday_only=True,
            demand_usd_per_kw_day_note="compatibility field contains the on-peak component only",
        )
    if event is not None:
        details["cpp_event"] = {
            "date": event.event_date.isoformat(),
            "start": event.event_start.isoformat(),
            "end": event.event_end.isoformat(),
            "notice": event.notice_time.isoformat(),
            "morning_mean_temperature_c": event.morning_mean_temperature_c,
            "eligible_morning_means_c": {
                date.isoformat(): value for date, value in event.eligible_morning_means_c.items()
            },
            "selection_basis": event.selection_basis,
            "weather_file": str(event.weather_file),
            "retrospective_selection": True,
        }
    return RateSelection(rate, details)

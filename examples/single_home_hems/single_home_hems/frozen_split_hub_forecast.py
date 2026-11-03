"""Deploy the pretest RF28 -> rolling p90 Hub-other/Outside-Hub studies.

The RF and its conservative calibration remain frozen at August 30, 2026.
The source replay may start later; changing the replay clock must never refit
on the held-out October comparison week.  One unavailable October 8 issue
reuses the preceding causal issue, shifted forward by one control interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .rest_of_home_forecast import (
    INTERVAL,
    RestOfHomeHybridForecast,
    _causal_short_gap_inference_data,
    fit_forest,
)


FROZEN_FIT_END = pd.Timestamp("2026-08-30 00:00:00")
APPROVED_FALLBACK_SOURCE_ISSUE = pd.Timestamp("2026-10-08 00:15:00")
_MISSING_LATEST_PREFIX = "Conservative RF needs the latest approved 15-minute bin available by "


@dataclass
class FrozenSplitHubForecast:
    """One frozen 28-day conservative RF plus a causal 21/28-day p90 tail."""

    provider: RestOfHomeHybridForecast
    p90_history_days: int
    metadata: dict[str, Any]

    @property
    def method_label(self) -> str:
        return f"rf28d_2h_p90_{self.p90_history_days}d"

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        simulation_start: str | pd.Timestamp,
        source_start: str | pd.Timestamp,
        p90_history_days: int,
        source_window: tuple[pd.Timestamp, pd.Timestamp],
    ) -> "FrozenSplitHubForecast":
        """Reconstruct the saved study fit, then map issuance to the test replay.

        Training and the q75/peak-continuation calibration use only originally
        observed source bins before ``FROZEN_FIT_END``.  The rolling p90 may
        use earlier completed test days, as specified for this comparison.
        """
        if type(p90_history_days) is not int or p90_history_days not in (21, 28):
            raise ValueError("p90_history_days must be 21 or 28")
        simulation_start = pd.Timestamp(simulation_start)
        simulation_wall = (simulation_start.tz_localize(None)
                           if simulation_start.tz is not None else simulation_start)
        source_start = pd.Timestamp(source_start)
        if (source_start.tz is not None
                or simulation_wall != simulation_wall.floor(INTERVAL)
                or source_start != source_start.floor(INTERVAL)
                or simulation_wall.time() != source_start.time()):
            raise ValueError("Split-Hub forecast clocks must share a 15-minute local clock")
        window_start, window_end = map(pd.Timestamp, source_window)
        if not window_start <= source_start < window_end:
            raise ValueError("Inference source window must cover the source replay start")
        # The saved offline study allowed causal short-gap inference from the
        # August 30 baseline onward.  The first October issue can need a
        # completed bin from the preceding evening, so retain that same
        # pretest inference history even when callers request only October.
        window_start = min(window_start, FROZEN_FIT_END)

        # Build the *frozen* calibration at the study's fit cutoff.  The
        # established provider fits a 20-day RF while building q75 and peak
        # continuation; replace only its forest with the exact 28-day study
        # fit.  The fitting function uses originally observed bins exclusively.
        base = RestOfHomeHybridForecast.from_csv(
            path,
            simulation_start=FROZEN_FIT_END,
            source_start=FROZEN_FIT_END,
            history_days=20,
        )
        forest, forest_fit = fit_forest(base.observed, FROZEN_FIT_END, train_days=28)
        base.forest = forest
        base.simulation_start = simulation_wall
        base.source_start = source_start
        base.history_days = p90_history_days
        base.inference_gap_policy = "causal_short_gap"
        base._inference_bins, base._inference_minutes = _causal_short_gap_inference_data(
            base.source_path, simulation_wall, source_start,
            (window_start, window_end),
        )
        method = f"rf28d_2h_p90_{p90_history_days}d"
        metadata = {
            **base.metadata,
            "method_label": method,
            "rf_fit": forest_fit,
            "rf_fit_days": 28,
            "rf_fit_end_exclusive": FROZEN_FIT_END.isoformat(),
            "rf_calibration_end_exclusive": FROZEN_FIT_END.isoformat(),
            "simulation_start": simulation_start.isoformat(),
            "source_start": source_start.isoformat(),
            "inference_window_start": window_start.isoformat(),
            "inference_window_end_exclusive": window_end.isoformat(),
            "p90_history_days": p90_history_days,
            "p90_rule": "clock-slot p90 from prior completed source calendar days",
            "missing_issue_rule": (
                "At source 2026-10-08 00:15, reuse the 00:00 causal issue "
                "shifted one 15-minute step; the RF-to-p90 handoff is one "
                "step earlier only for this issue"
            ),
        }
        base.metadata = metadata
        return cls(base, p90_history_days, metadata)

    def forecast_kw(self, timestamp: str | pd.Timestamp, steps: int = 32) -> pd.Series:
        """Issue a complete aligned forecast without future meter information."""
        timestamp = pd.Timestamp(timestamp)
        try:
            result = self.provider.forecast_kw(timestamp, steps=steps)
            result.attrs["prior_issue_fallback"] = False
            return result
        except ValueError as error:
            source_issue = self.provider.source_timestamp(timestamp)
            if (not str(error).startswith(_MISSING_LATEST_PREFIX)
                    or source_issue != APPROVED_FALLBACK_SOURCE_ISSUE):
                raise
        prior_timestamp = timestamp - INTERVAL
        prior = self.provider.forecast_kw(prior_timestamp, steps=steps + 1)
        result = prior.iloc[1:].copy()
        expected = pd.date_range(timestamp, periods=steps, freq=INTERVAL)
        if not result.index.equals(expected):
            raise AssertionError("Shifted prior Hub forecast does not cover the requested horizon")
        result.attrs.update(prior.attrs)
        result.attrs["prior_issue_fallback"] = True
        result.attrs["actual_issue_time"] = prior_timestamp.isoformat()
        return result

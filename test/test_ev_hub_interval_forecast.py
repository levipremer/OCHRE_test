"""Exact EV coefficients from issued Home Hub interval forecasts."""

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from test.test_ev_v2h import forecast, schedule, vehicle
from test.test_ev_scenario import historical, scenario_control
from examples.single_home_hems.single_home_hems.controllers.ev import EVHorizon, EVMPCController
from examples.single_home_hems.single_home_hems.ev_hub_forecast import EVHubIntervalForecast
from examples.single_home_hems.single_home_hems.ev_scenarios import ScenarioEVForecast


def paired_forecasts(mobility=None):
    ev = vehicle(formulation="milp", solver="SCIPY")
    mobility = mobility or schedule()
    minute = forecast(ev, mobility, price=0.2)
    now = mobility.start
    # Include demand above the 4 kW inverter limit: the capped mean cannot be
    # reconstructed by clipping the raw 15-minute mean afterward.
    pattern = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 24)
    load = pd.Series(pattern, index=minute.data.index)
    minute = replace(minute, backed_load_kw=load)
    rows = []
    for start in pd.date_range(now, periods=4, freq="15min"):
        values = load.loc[start:start + pd.Timedelta(minutes=14)].to_numpy()
        cap = np.minimum(values, ev.hems.max_discharge_kw)
        rows.append(dict(eligible_load_mean_kw=values.mean(), eligible_mean_kw=cap.mean(),
                         eligible_min_kw=cap.min(), hub_mean_kw=values.mean() + 0.7))
    issued = EVHubIntervalForecast(now, now - pd.Timedelta(minutes=1),
                                   pd.DataFrame(rows, index=pd.date_range(now, periods=4, freq="15min")),
                                   "controlled_test_forecast")
    interval = replace(minute, backed_load_kw=None, ev_hub_intervals=issued)
    return ev, minute, interval


class EVHubIntervalForecastTests(unittest.TestCase):
    def test_interval_coefficients_match_minute_path(self):
        ev, minute, interval = paired_forecasts()
        controller = EVMPCController(ev)
        old = EVHorizon.from_forecast(controller, minute.ev_schedules[ev.vehicle_id].start, minute)
        new = EVHorizon.from_forecast(controller, interval.ev_schedules[ev.vehicle_id].start, interval)
        self.assertTrue(old.t.equals(new.t))
        for name in ("dt", "z", "price", "backed_cost", "discharge_cost",
                     "discharge_mean", "discharge_min", "trip_kwh", "group"):
            np.testing.assert_allclose(getattr(old, name), getattr(new, name))
        self.assertGreater(interval.ev_hub_intervals.intervals.hub_mean_kw.iloc[0],
                           interval.ev_hub_intervals.intervals.eligible_load_mean_kw.iloc[0])

    def test_mid_interval_mobility_requires_issued_subintervals(self):
        ev, _, interval = paired_forecasts(schedule([(7, 20, 1)]))
        with self.assertRaisesRegex(ValueError, "mobility boundary splits"):
            EVHorizon.from_forecast(EVMPCController(ev), interval.ev_schedules[ev.vehicle_id].start, interval)

    def test_full_hub_power_does_not_expand_ev_eligible_cap(self):
        ev, _, interval = paired_forecasts()
        now = interval.ev_schedules[ev.vehicle_id].start
        controller = EVMPCController(ev)
        original = EVHorizon.from_forecast(controller, now, interval)
        rows = interval.ev_hub_intervals.intervals.copy()
        rows["hub_mean_kw"] += 5.0  # e.g. an ineligible HVAC resistance circuit
        changed = EVHubIntervalForecast(now, now, rows, "controlled_test_forecast")
        horizon = EVHorizon.from_forecast(controller, now,
                                          replace(interval, ev_hub_intervals=changed))
        np.testing.assert_allclose(horizon.discharge_mean, original.discharge_mean)
        np.testing.assert_allclose(horizon.backed_cost, original.backed_cost)

    def test_price_covariance_is_not_silently_approximated(self):
        ev, _, interval = paired_forecasts()
        data = interval.data.copy()
        data.loc[data.index[3], "electricity_price_usd_per_kwh"] = 0.3
        changed = replace(interval, data=data)
        with self.assertRaisesRegex(ValueError, "constant tariff"):
            EVHorizon.from_forecast(EVMPCController(ev), changed.ev_schedules[ev.vehicle_id].start, changed)

    def test_issuance_and_full_coverage_are_required(self):
        ev, _, interval = paired_forecasts()
        now = interval.ev_schedules[ev.vehicle_id].start
        rows = interval.ev_hub_intervals.intervals
        with self.assertRaisesRegex(ValueError, "ready no later"):
            EVHubIntervalForecast(now, now + pd.Timedelta(minutes=1), rows, "test")
        future = EVHubIntervalForecast(now + pd.Timedelta(minutes=1), now, rows, "test")
        with self.assertRaisesRegex(ValueError, "not available"):
            EVHorizon.from_forecast(EVMPCController(ev), now,
                                    replace(interval, ev_hub_intervals=future))
        incomplete = EVHubIntervalForecast(now, now, rows.iloc[:-1], "test")
        with self.assertRaisesRegex(ValueError, "complete 15-minute horizon coverage"):
            EVHorizon.from_forecast(EVMPCController(ev), now,
                                    replace(interval, ev_hub_intervals=incomplete))

    def test_interval_and_minute_load_cannot_conflict(self):
        ev, minute, interval = paired_forecasts()
        with self.assertRaisesRegex(ValueError, "either interval Hub power or minute backed load"):
            EVHorizon.from_forecast(EVMPCController(ev), minute.ev_schedules[ev.vehicle_id].start,
                                    replace(interval, backed_load_kw=minute.backed_load_kw))

    def test_mobility_scenarios_retain_issued_hub_forecast(self):
        ev, _, interval = paired_forecasts()
        now = interval.ev_schedules[ev.vehicle_id].start
        interval = replace(interval, ev_schedules={ev.vehicle_id: historical()})
        provider = ScenarioEVForecast(ev.vehicle_id, scenario_control())
        scenarios = provider.at(now, interval, pd.Timedelta(hours=1))
        self.assertTrue(all(path is None for path in scenarios.backed_load_paths))
        self.assertTrue(all(path is interval.ev_hub_intervals for path in scenarios.hub_interval_paths))
        self.assertIn("issued Home Hub interval forecast", scenarios.diagnostics["backed_load_source"])
        self.assertNotIn("twin", scenarios.diagnostics["backed_load_source"])


if __name__ == "__main__":
    unittest.main()

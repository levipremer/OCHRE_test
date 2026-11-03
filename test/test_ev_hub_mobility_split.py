"""The explicit study policy retains native EV events inside forecast bins."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from examples.five_home_block.five_home_block.ev_schedule import EVSchedule
from examples.single_home_hems.single_home_hems.controllers.ev import EVHorizon, EVMPCController
from examples.single_home_hems.single_home_hems.ev_hub_forecast import EVHubIntervalForecast
from examples.single_home_hems.single_home_hems.forecast import Forecast
from test.test_ev_v2h import schedule, vehicle


def forecast_case(policy="reject"):
    ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=8,
        battery_model="power_loss", initial_soh=.95, soc_depletion_reference_capacity_kwh=60)
    ev = replace(ev, capacity_kwh=80, initial_soc_fraction=.2)
    raw = schedule([(0, 14, 10), (15, 30, 5)])
    rows = raw.trips.copy()
    rows["arrival_net_soc_depletion_pp"] = [10., 5.]
    mobility = EVSchedule(rows, raw.start, raw.end, raw.metadata)
    starts = pd.date_range(raw.start, periods=32, freq="15min")
    coefficients = pd.DataFrame(dict(eligible_load_mean_kw=3., eligible_mean_kw=2.5,
                                    eligible_min_kw=1., hub_mean_kw=3.5), index=starts)
    issued = EVHubIntervalForecast(raw.start, raw.start, coefficients, "test_issued_coefficients",
                                  mobility_split_policy=policy)
    minute = pd.DataFrame(index=pd.date_range(raw.start, periods=480, freq="min"))
    forecast = Forecast.from_schedule(minute, lambda time: .2,
        ev_schedules={ev.vehicle_id: mobility}, ev_hub_intervals=issued)
    return ev, mobility, forecast


def test_default_rejects_fourteen_plus_one_minute_bin():
    ev, mobility, issued = forecast_case()
    with pytest.raises(ValueError, match="mobility boundary splits"):
        EVHorizon.from_forecast(EVMPCController(ev), mobility.start, issued, trip_timing="departure")


def test_opt_in_holds_issued_coefficients_with_exact_native_durations_and_trip_energy():
    ev, mobility, issued = forecast_case("hold_parent_interval_coefficients")
    before = mobility.trips.copy(deep=True)
    controller = EVMPCController(ev)
    horizon = EVHorizon.from_forecast(controller, mobility.start, issued, trip_timing="departure")
    np.testing.assert_allclose(horizon.dt[:2], [14 / 60, 1 / 60])
    np.testing.assert_array_equal(horizon.z[:3], [0, 1, 0])
    np.testing.assert_allclose(horizon.discharge_mean[:2], [2.5, 2.5])
    np.testing.assert_allclose(horizon.discharge_min[:2], [1., 1.])
    assert np.dot(horizon.dt[:2], horizon.backed_cost[:2]) == pytest.approx(.25 * .2 * 3)
    assert np.dot(horizon.dt[:2], horizon.discharge_cost[:2]) == pytest.approx(.25 * .2 * 2.5)
    assert horizon.N == 32 and len(horizon.dt) == 33
    np.testing.assert_allclose(horizon.trip_kwh[horizon.trip_kwh > 0], [6., 3.])
    assert horizon.t[np.flatnonzero(horizon.trip_kwh > 0)].equals(pd.DatetimeIndex(before.departure))
    assert horizon.comfort_dt[1] == 0  # Only one minute home, still in the return grace period.
    grace_end = mobility.start + pd.Timedelta(minutes=90)  # Second return at30min +60min.
    assert horizon.comfort_dt[horizon.t.get_loc(grace_end)] == pytest.approx(.25)
    pd.testing.assert_frame_equal(mobility.trips, before)


def test_opt_in_keeps_whole_parent_bin_tariff_covariance_guard():
    ev, mobility, issued = forecast_case("hold_parent_interval_coefficients")
    data = issued.data.copy()
    data.loc[mobility.start + pd.Timedelta(minutes=14), "electricity_price_usd_per_kwh"] = .3
    with pytest.raises(ValueError, match="constant tariff"):
        EVHorizon.from_forecast(EVMPCController(ev), mobility.start, replace(issued, data=data),
                                trip_timing="departure")


def test_unknown_split_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown Home Hub mobility split policy"):
        forecast_case("snap_mobility")

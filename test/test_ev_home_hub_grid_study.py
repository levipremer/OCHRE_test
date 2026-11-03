"""Approved EV-only grid terms and a single accepted full-period schedule."""

from dataclasses import replace
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.controllers.ev import EVScenarioController
from examples.single_home_hems.single_home_hems.controllers.ev_home_hub import EVHomeHubController
from examples.single_home_hems.single_home_hems.ev_hub_forecast import EVHubIntervalForecast
from examples.single_home_hems.single_home_hems.forecast import Forecast
from examples.single_home_hems.single_home_hems.home_hub_observation import TOTAL_POWER_KEY
from examples.single_home_hems.single_home_hems.rates import (
    MISO_RTP_DOE, WINTER_TOU, WINTER_TOU_DEMAND_CPP,
)
from test.test_ev_v2h import schedule, vehicle
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule


class Outside:
    def __init__(self, power=0.0):
        self.power = power

    def forecast_kw(self, timestamp, steps=32):
        return pd.Series(self.power, index=pd.date_range(timestamp, periods=steps, freq="15min"))


class Hub:
    def __init__(self, power=10.0):
        self.power = power
        self.issue_count = 0

    def issue(self, timestamp, completed_history, horizon, inverter_kw):
        self.issue_count += 1
        return EVHubIntervalForecast(timestamp, timestamp, pd.DataFrame({
            "eligible_load_mean_kw": self.power,
            "eligible_mean_kw": min(self.power, inverter_kw),
            "eligible_min_kw": min(self.power, inverter_kw),
            "hub_mean_kw": self.power,
        }, index=pd.date_range(timestamp, periods=int(horizon / pd.Timedelta(minutes=15)), freq="15min")), "test_hub")


def fixture(rate, now="2025-12-01 00:00", hours=8, count=2, one_shot=False, load=10.0,
            grid2_demand_formulation="weekday_daily_dt_over_n_v1"):
    now = pd.Timestamp(now)
    ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=hours,
                 enforce_terminal_soc=False, comfort_weight=0, health_weight=0,
                 persistence_weight=0)
    hub = Hub(load)
    controller = EVHomeHubController(ev, hub, Outside(), rate_structure=rate,
                                     scenario_count=count, one_shot=one_shot,
                                     grid2_demand_formulation=grid2_demand_formulation)
    mobility = replace(schedule(), start=now, end=now + pd.Timedelta(hours=hours, minutes=15))
    forecast = Forecast.from_schedule(
        pd.DataFrame(index=pd.date_range(now, periods=int(hours * 60), freq="min")),
        lambda _: 0.2, ev_schedules={ev.vehicle_id: mobility},
    )
    issued = hub.issue(now, pd.DataFrame(), controller.horizon, ev.hems.max_discharge_kw)
    path_forecast = replace(forecast, ev_hub_intervals=issued)
    scenarios = EVScenarioSet(now, (mobility,) * count, (None,) * count,
                              (path_forecast,) * count, tuple(map(str, range(count))))
    return controller, forecast, scenarios


def fixed_base(scenarios, ac_values, charge_values=None):
    starts = scenarios.forecasts[0].ev_hub_intervals.intervals.index
    boundaries = starts.append(pd.DatetimeIndex([starts[-1] + pd.Timedelta(minutes=15)]))
    paths = []
    for index, ac in enumerate(ac_values):
        values = np.broadcast_to(ac, (len(starts),))
        charge = np.maximum(values, 0) if charge_values is None else np.broadcast_to(charge_values[index], values.shape)
        paths.append({"boundaries": boundaries, "ac": cp.Constant(values),
                      "charge": cp.Constant(charge)})
    binary = cp.Variable(boolean=True)
    return cp.Problem(cp.Minimize(0), [binary == 0]), {
        "objective_terms": {"electricity": cp.Constant(0.0)}, "scenario_plans": paths,
    }


def test_scenario1_removes_legacy_peak_penalty():
    c, _, scenarios = fixture(WINTER_TOU, "2025-12-01 03:00")
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [0, 4])):
        problem, plan = c.build_problem(scenarios.issued_at, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert problem.value == 0
    assert not plan["peak_objective_active"].any()
    assert plan["modeled_home_peak_kw"] is None
    assert set(plan["objective_terms"]) == {"electricity", "modeled_home_peak"}


def test_historical_scenario2_uses_worst_scenario_complete_30_minute_mean():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 05:45",
                              grid2_demand_formulation="legacy_horizon_peak")
    first = np.zeros(32)
    first[1] = 20  # 06:00 has 30 kW, but its 30-minute mean is 20 kW.
    second = np.zeros(32)
    second[1:3] = 12  # This scenario's 22 kW mean sets the shared maximum.
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [first, second])):
        problem, plan = c.build_problem(scenarios.issued_at, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert plan["modeled_home_peak_kw"].value == pytest.approx(22)
    assert problem.value == pytest.approx(0.71 * 22)
    assert list(plan["peak_demand_block_starts"]) == list(pd.date_range("2025-12-01 06:00", periods=4, freq="30min"))
    np.testing.assert_allclose(plan["peak_demand_block_power_scenarios_kw"].value[:, 0], [20, 22])


def test_historical_scenario2_midblock_uses_actual_first_half_and_excludes_weekend():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-05 20:45",
                              grid2_demand_formulation="legacy_horizon_peak")
    now = scenarios.issued_at
    c._completed_plant_rows = [(time, {TOTAL_POWER_KEY: 20.0}) for time in
                               pd.date_range(now - pd.Timedelta(minutes=14), now, freq="min")]
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [0, 0])):
        problem, plan = c.build_problem(now, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert plan["peak_observed_first_half_kw"] == 20
    assert plan["peak_demand_block_starts"] == (now - pd.Timedelta(minutes=15),)
    assert plan["peak_objective_active"].sum() == 1
    assert problem.value == pytest.approx(0.71 * 15)
    c._completed_plant_rows.pop()
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [0, 0])):
        with pytest.raises(ValueError, match="15 completed"):
            c.build_problem(now, 0.5, scenarios)


def test_historical_scenario2_excludes_incomplete_last_demand_block():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 09:15",
                              grid2_demand_formulation="legacy_horizon_peak")
    # The horizon ends 17:15; the evening 17:00 block is incomplete.
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [0, 0])):
        problem, plan = c.build_problem(scenarios.issued_at, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert plan["peak_demand_block_starts"] == ()
    assert problem.value == 0


def test_scenario3_gross_panel_and_net_grid_use_shared_step_slacks():
    c, _, scenarios = fixture(MISO_RTP_DOE, "2025-12-01 05:45", load=40)
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [-4, 4])):
        problem, plan = c.build_problem(scenarios.issued_at, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(plan["modeled_home_power_scenarios_kw"].value[:, 0], [36, 44])
    np.testing.assert_allclose(plan["modeled_panel_power_scenarios_kw"].value[:, 0], [40, 44])
    assert plan["doe_panel_slack_kw"].shape == (32,)
    np.testing.assert_allclose(plan["doe_panel_slack_kw"].value, 6)
    assert plan["doe_grid_slack_kw"].shape == (12,)
    np.testing.assert_allclose(plan["doe_grid_slack_kw"].value, 29)
    assert plan["objective_terms"]["doe_panel_violation"].value == pytest.approx(1000 * .25 / 32 * 32 * 6)
    assert plan["objective_terms"]["doe_grid_violation"].value == pytest.approx(10 * .25 / 32 * 12 * 29)
    assert not plan["doe_grid_active"][0]  # 05:45
    assert plan["doe_grid_active"][12]  # 08:45, original 09:00 endpoint
    assert not plan["doe_grid_active"][13]
    assert not plan["peak_objective_active"].any()


def test_full_week_keeps_approved_n672_envelope_normalization():
    c, _, scenarios = fixture(MISO_RTP_DOE, hours=168, count=1, one_shot=True, load=40)
    with patch.object(EVScenarioController, "build_problem", return_value=fixed_base(scenarios, [-4])):
        problem, plan = c.build_problem(scenarios.issued_at, 0.5, scenarios)
    problem.solve(solver="SCIPY")
    assert len(c.forecast_trace[-1]) == 672
    assert plan["doe_panel_slack_kw"].shape == (672,)
    assert plan["doe_grid_active"].sum() == 5 * (3 + 4) * 4
    assert plan["objective_terms"]["doe_panel_violation"].value == pytest.approx(1000 * .25 / 672 * 672 * 2)
    assert plan["objective_terms"]["doe_grid_violation"].value == pytest.approx(10 * .25 / 672 * 140 * 21)


@pytest.mark.parametrize("rate", [WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE])
def test_accepted_grid_solve_saves_constraint_margins_and_weighted_objectives(rate):
    c, forecast, scenarios = fixture(rate, count=1, load=40)
    record = {}
    c.trace_callback = lambda kind, rows: record.setdefault(kind, rows.copy())
    provider = SimpleNamespace(at=lambda timestamp, source, horizon: EVScenarioSet(
        timestamp, (source.ev_schedules[c.ev.vehicle_id],), (None,), (source,), ("perfect",),
    ))
    c._solve(scenarios.issued_at, .5, replace(
        forecast, ev_hub_intervals=scenarios.forecasts[0].ev_hub_intervals,
        providers={c.equipment_name: provider},
    ))
    assert c.solve_history[0]["successful"]
    predicted = record["scenario_predictions"]
    assert len(predicted) == 32
    assert record["objective_components"].iloc[0].objective_balance_error == pytest.approx(0, abs=1e-7)
    if rate == MISO_RTP_DOE:
        assert predicted.doe_panel_slack_kw.max() >= 2 - 1e-6
        assert predicted.doe_panel_constraint_margin_kw.min() >= -1e-6
        assert predicted.doe_grid_constraint_margin_kw.dropna().min() >= -1e-6
        assert len(c.plans[0]["doe_panel_slack_kw"]) == 32
        assert "doe_panel_violation" in c.plans[0]["objective_terms"]
    else:
        assert predicted.peak_constraint_margin_kw.dropna().min() >= -1e-6
        assert predicted.demand_block_power_kw.notna().sum() == 8
        assert len(c.plans[0]["peak_demand_block_starts"]) == 4
        assert predicted.daily_anytime_constraint_margin_kw.dropna().min() >= -1e-6
        saved = c.plans[0]
        assert saved["grid2_demand_formulation"] == "weekday_daily_dt_over_n_v1"
        assert saved["daily_demand_days"] == ["2025-12-01T00:00:00"]
        assert isinstance(saved["daily_on_peak_demand_kw"], list)
        assert isinstance(saved["daily_anytime_demand_kw"], list)
        assert saved["daily_demand_scale"] == pytest.approx(.25 / 32)
        assert sum(saved["daily_demand_objective_terms"].values()) == pytest.approx(
            saved["objective_terms"]["modeled_home_peak"])
        assert record["decision_state"].iloc[0].grid2_demand_formulation == "weekday_daily_dt_over_n_v1"
        assert json.loads(record["decision_state"].iloc[0].observed_daily_demand_peaks_json) == {
            "2025-12-01T00:00:00": {"on_peak_kw": 0, "anytime_kw": 0},
        }


def test_one_shot_solves_once_replays_all_modes_and_saves_accepted_plan():
    c, forecast, _ = fixture(WINTER_TOU, "2018-01-01", hours=1, count=1, one_shot=True, load=2)
    now = forecast.data.index[0]
    record = {}
    c.trace_callback = lambda kind, rows: record.setdefault(kind, rows.copy())
    # Use the real EV optimization; later physical projections may differ from
    # its schedule, but no realized discrepancy may trigger an oracle re-solve.
    status = {
        "Home Hub Electric Power (kW)": 2.0, TOTAL_POWER_KEY: 2.0,
        f"{c.equipment_name} SOC (-)": .5, f"{c.equipment_name} End SOC (-)": .5,
        f"{c.equipment_name} SOH (-)": 1.0,
        f"{c.equipment_name} Capacity (kWh)": c.ev.capacity_kwh,
        f"{c.equipment_name} Electric Power (kW)": 0.0,
        f"{c.equipment_name} Parked": True,
    }
    first_forecast = replace(forecast, providers={
        c.equipment_name: SimpleNamespace(at=lambda timestamp, source, horizon: EVScenarioSet(
            timestamp, (source.ev_schedules[c.ev.vehicle_id],), (None,), (source,), ("perfect",),
        ))
    })
    for minute in range(60):
        action = c.controls_for_step(now + pd.Timedelta(minutes=minute), status, {}, first_forecast)
        mode = c._one_shot_modes[minute // 15]
        assert action[c.equipment_name]["P Setpoint"] == (4 if mode == 1 else -4 if mode == -1 else 0)
    assert len(c.solve_history) == 1
    assert c.hub_forecast_provider.issue_count == 2  # fixture once, control once
    assert len(c._one_shot_modes) == 4
    saved = json.loads(record["accepted_plan"].iloc[0].plan_json)
    assert saved["normalization_steps"] == 4
    assert saved["one_shot_modes"] == c._one_shot_modes.tolist()
    assert len(saved["scenarios"][0]["charge_mode"]) == 4
    assert "soc_turning_points_json" in record["decision_state"]
    assert "doe_grid_constraint_margin_kw" in record["scenario_predictions"]
    native = json.loads(record["scenario_snapshot"].iloc[0].horizon_json)
    assert native["N"] == 4
    assert native["group"] == [0, 1, 2, 3]
    assert native["budget_dt"] == [.25] * 4
    assert json.loads(record["scenario_snapshot"].iloc[0].mobility_trips_json) == []
    assert all(row["one_shot"] for row in c.action_history)
    with pytest.raises(ValueError, match="does not cover"):
        c.controls_for_step(now + pd.Timedelta(hours=1), status, {}, first_forecast)


def test_one_shot_requires_one_path_and_does_not_fallback_after_failed_solve():
    with pytest.raises(ValueError, match="one perfect"):
        fixture(WINTER_TOU, one_shot=True)
    c, forecast, _ = fixture(WINTER_TOU, hours=1, count=1, one_shot=True)
    def fail(*args):
        c.solve_history.append({"successful": False, "status": "infeasible", "fallback": True})
    with patch.object(EVScenarioController, "_solve", side_effect=fail):
        with pytest.raises(RuntimeError, match="accepted schedule"):
            c._solve(forecast.data.index[0], .5, forecast)


def test_zoneinfo_snapshot_preserves_zero_demand_trip_and_local_offsets():
    now = pd.Timestamp("2025-12-01 03:00", tz=ZoneInfo("America/Indiana/Indianapolis"))
    c, _, scenarios = fixture(WINTER_TOU, now=now, count=1)
    departure, arrival = now + pd.Timedelta(hours=1), now + pd.Timedelta(hours=2)
    mobility = EVSchedule(pd.DataFrame([{
        "departure": departure, "arrival": arrival, "distance_km": 0.0,
        "arrival_net_soc_depletion_pp": 0.0,
    }]), now, now + c.horizon, {"zero_trip": True})
    assert isinstance(mobility.trips.departure.dt.tz, ZoneInfo)
    path = replace(scenarios.forecasts[0], ev_schedules={c.ev.vehicle_id: mobility})
    scenarios = replace(scenarios, paths=(mobility,), forecasts=(path,))
    record = {}
    c.trace_callback = lambda kind, rows: record.setdefault(kind, rows.copy())
    c.build_problem(now, .5, scenarios)
    snapshot = record["scenario_snapshot"].iloc[0]
    saved = json.loads(snapshot.mobility_trips_json)
    assert len(saved) == 1
    assert pd.Timestamp(saved[0]["departure"]) == departure
    assert pd.Timestamp(saved[0]["arrival"]) == arrival
    assert saved[0]["departure"].endswith("-05:00")
    assert saved[0]["distance_km"] == 0
    assert saved[0]["arrival_net_soc_depletion_pp"] == 0
    assert json.loads(snapshot.mobility_trip_columns_json) == mobility.trips.columns.tolist()
    native = json.loads(snapshot.horizon_json)
    assert native["departure_indices"] == [4]
    assert not any(native["trip_kwh"])
    assert native["z"][4:8] == [0, 0, 0, 0]

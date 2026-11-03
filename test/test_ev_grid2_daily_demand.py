"""Approved Grid 2 weekday maxima use the fixed 15-minute dt/full-horizon N."""
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.single_home_hems.controllers.ev import EVScenarioController
from examples.single_home_hems.single_home_hems.home_hub_observation import TOTAL_POWER_KEY
from examples.single_home_hems.single_home_hems.rates import (
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY as ON_RATE,
    GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY as ANY_RATE,
    WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP,
)
from test.test_ev_home_hub_grid_study import fixture, fixed_base


def build(controller, scenarios, powers):
    with patch.object(EVScenarioController, "build_problem",
                      return_value=fixed_base(scenarios, powers)):
        problem, plan = controller.build_problem(scenarios.issued_at, .5, scenarios)
    problem.solve(solver="SCIPY")
    assert problem.status == cp.OPTIMAL
    return problem, plan


def observations(controller, start, values):
    controller._completed_plant_rows = [
        (time + pd.Timedelta(minutes=1), {TOTAL_POWER_KEY: float(value)})
        for time, value in zip(pd.date_range(start, periods=len(values), freq="min"), values)
    ]


def test_two_components_have_distinct_shared_scenario_maxima_and_n32_scaling():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 05:30")
    first, second = np.zeros(32), np.zeros(32)
    first[2:4] = 12.0  # Scenario 0 establishes the on-peak 22 kW maximum.
    second[14:16] = 30.0  # Scenario 1 establishes the off-peak 40 kW maximum.
    problem, plan = build(c, scenarios, [first, second])
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, [22])
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, [40])
    assert plan["daily_demand_normalization_steps"] == 32
    assert plan["daily_demand_scale"] == pytest.approx(.25 / 32)
    assert problem.value == pytest.approx(.25 / 32 * (ON_RATE * 22 + ANY_RATE * 40))
    assert set(plan["objective_terms"]) == {"electricity", "modeled_home_peak"}
    assert plan["daily_demand_objective_terms"]["on_peak"].value == pytest.approx(.25 / 32 * ON_RATE * 22)
    assert plan["daily_demand_objective_terms"]["anytime"].value == pytest.approx(.25 / 32 * ANY_RATE * 40)


def test_full_week_uses_n672_and_sums_five_independent_weekdays():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-11-30", hours=168,
                              count=1, one_shot=True)
    starts = scenarios.forecasts[0].ev_hub_intervals.intervals.index
    power = np.zeros(672)
    for day in range(5):
        date = pd.Timestamp("2025-12-01") + pd.Timedelta(days=day)
        power[(starts >= date + pd.Timedelta(hours=6))
              & (starts < date + pd.Timedelta(hours=6, minutes=30))] = day + 1
        power[(starts >= date + pd.Timedelta(hours=12))
              & (starts < date + pd.Timedelta(hours=12, minutes=30))] = 10 + day
    power[starts.dayofweek >= 5] = 1000  # Weekend imports never enter either term.
    problem, plan = build(c, scenarios, [power])
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, np.arange(11, 16))
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, np.arange(20, 25))
    assert len(plan["daily_demand_days"]) == 5
    assert plan["daily_demand_normalization_steps"] == 672
    assert plan["daily_demand_scale"] == pytest.approx(.25 / 672)
    assert len(plan["daily_demand_block_starts"]) == 5 * 48
    assert problem.value == pytest.approx(.25 / 672 * (ON_RATE * 65 + ANY_RATE * 110))


def test_observed_complete_daily_peaks_floor_both_terms_and_reset_at_midnight():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 10:00")
    values = np.full(10 * 60, 4.0)
    values[2 * 60:2 * 60 + 30] = 40  # Earlier anytime peak.
    values[6 * 60:6 * 60 + 30] = 30  # Earlier on-peak peak.
    observations(c, "2025-12-01", values)
    problem, plan = build(c, scenarios, [0, 0])
    np.testing.assert_allclose(plan["daily_demand_observed_on_peak_kw"], [30])
    np.testing.assert_allclose(plan["daily_demand_observed_anytime_kw"], [40])
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, [30])
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, [40])
    assert problem.value == pytest.approx(.25 / 32 * (ON_RATE * 30 + ANY_RATE * 40))
    assert c._observed_daily_demand_peaks(pd.Timestamp("2025-12-02")) == {
        "2025-12-02T00:00:00": {"on_peak_kw": 0.0, "anytime_kw": 0.0},
    }


def test_reconstruction_restores_daily_floors_without_full_meter_history():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 10:00")
    c._restored_daily_demand_peaks = {
        "2025-12-01T00:00:00": {"on_peak_kw": 30, "anytime_kw": 40},
    }
    _, plan = build(c, scenarios, [0, 0])
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, [30])
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, [40])


def test_midblock_anytime_demand_combines_actual_and_forecast_then_positive_part():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 09:15", load=0)
    observations(c, "2025-12-01 09:00", [30] * 15)
    first, second = np.zeros(32), np.zeros(32)
    first[0] = -10
    second[0] = 10
    problem, plan = build(c, scenarios, [first, second])
    np.testing.assert_allclose(plan["daily_demand_block_power_scenarios_kw"].value[:, 0], [10, 20])
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, [20])
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, [0])
    assert plan["daily_demand_block_starts"][-1] == pd.Timestamp("2025-12-01 16:30")
    assert plan["peak_demand_block_starts"] == ()  # 17:00–17:30 is not fully covered.
    assert problem.value == pytest.approx(.25 / 32 * ANY_RATE * 20)
    c._completed_plant_rows.pop()
    with pytest.raises(ValueError, match="15 completed"):
        build(c, scenarios, [first, second])


def test_positive_part_applies_to_signed_thirty_minute_mean_not_each_quarter():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-01 06:00", load=0)
    power = np.zeros(32)
    power[:2] = [-20, 10]  # Signed mean is -5, rather than mean positive import5.
    problem, plan = build(c, scenarios, [power, power])
    assert problem.value == pytest.approx(0)
    np.testing.assert_allclose(plan["daily_on_peak_demand_kw"].value, [0])
    np.testing.assert_allclose(plan["daily_anytime_demand_kw"].value, [0])


def test_weekend_only_horizon_has_no_daily_demand_variables_or_cost():
    c, _, scenarios = fixture(WINTER_TOU_DEMAND_CPP, "2025-12-06", load=100)
    problem, plan = build(c, scenarios, [100, 100])
    assert plan["daily_demand_days"] == ()
    assert plan["daily_on_peak_demand_kw"] is None
    assert plan["daily_anytime_demand_kw"] is None
    assert not plan["daily_demand_active"].any()
    assert problem.value == 0


def test_non_cpp_demand_and_historical_grid2_keep_original_horizon_peak():
    for rate, formulation in ((WINTER_TOU_DEMAND, "weekday_daily_dt_over_n_v1"),
                              (WINTER_TOU_DEMAND_CPP, "legacy_horizon_peak")):
        c, _, scenarios = fixture(rate, "2025-12-01 06:00", grid2_demand_formulation=formulation)
        problem, plan = build(c, scenarios, [0, 12])
        assert problem.value == pytest.approx(.71 * 22)
        assert plan["modeled_home_peak_kw"].value == pytest.approx(22)
        assert "daily_demand_days" not in plan

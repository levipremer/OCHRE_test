"""Grid 2 joint HEMS retains paired scenarios and fixed dt/full-horizon N."""
from dataclasses import replace
import json
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.single_home_hems.controllers.whole_home_joint import WINTER_TOU_PEAK_WINDOW
from examples.single_home_hems.single_home_hems.rates import (
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY as ON_RATE,
    GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY as ANY_RATE,
    WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP,
)
from test import test_whole_home_joint as existing


def controller(hours=8, *, perfect=False, rate=WINTER_TOU_DEMAND_CPP,
               formulation="weekday_daily_dt_over_n_v1"):
    c = existing.WholeHomeJointTests().make_controller(peak_weight=.71)
    c.joint_controller.horizon = pd.Timedelta(hours=hours)
    c.ev_controller.horizon = pd.Timedelta(hours=hours)
    return replace(c, rate_structure=rate, grid2_demand_formulation=formulation,
                   peak_window=WINTER_TOU_PEAK_WINDOW,
                   perfect_forecasts=perfect, one_shot=hours != 8)


def inputs(c, start, ev_power=None, water_power=None, *, water_count=7, ev_count=5):
    steps = int(c.horizon / pd.Timedelta(minutes=15))
    start = pd.Timestamp(start)
    if ev_power is None:
        ev_power = np.zeros((water_count * ev_count, steps))
    if water_power is None:
        water_power = np.zeros((water_count, steps))
    water = SimpleNamespace(
        electric_power_kw=cp.Constant(water_power), prices=np.zeros(steps),
        problem=cp.Problem(cp.Minimize(0), [cp.Variable(boolean=True) == 1]),
        objective_components={},
    )
    hvac = SimpleNamespace(
        electric_power_kw=cp.Constant(np.zeros(steps)), prices=np.zeros(steps), timestamp=start,
        problem=cp.Problem(cp.Minimize(0), []), objective_components={},
    )
    boundaries = pd.date_range(start, periods=steps + 1, freq="15min")
    paths = [dict(boundaries=boundaries, ac=cp.Constant(ev_power[w * ev_count + e]),
                  water_scenario_index=w, ev_scenario_index=e)
             for w in range(water_count) for e in range(ev_count)]
    return water, hvac, cp.Problem(cp.Minimize(0), []), dict(
        scenario_plans=paths, scenario_shape=(water_count, ev_count), objective_terms={})


def solve(c, args, rest=0, observed_half=None):
    steps = args[0].electric_power_kw.shape[1]
    problem = c.optimization_problem_milp(*args, np.full(steps, rest),
                                         observed_first_half_kw=observed_half)
    problem.solve(solver="SCIPY")
    assert problem.status == cp.OPTIMAL
    return problem


def observe(c, start, values):
    for t, value in zip(pd.date_range(start, periods=len(values), freq="min"), values):
        c.observe_plant_step(t, {"Total Electric Power (kW)": value})


def test_joint_daily_components_use_distinct_worst_paired_scenarios_not_their_mean():
    c = controller()
    start = pd.Timestamp("2025-12-01 05:30")
    ev = np.zeros((35, 32))
    water = np.ones((7, 32))
    water[6, 2:4] = 3
    ev[34, 2:4] = 20  # Water6/EV4 gives an on-peak whole-home maximum33.
    ev[3, 14:16] = 40  # Water0/EV3 gives a different, off-peak maximum51.
    args = inputs(c, start, ev, water)
    problem = solve(c, args, rest=10)
    meta = c.daily_demand_metadata(solved=True)
    assert c._modeled_home_power_scenarios_kw.shape == (35, 32)
    np.testing.assert_allclose(meta["daily_on_peak_demand_kw"], [33])
    np.testing.assert_allclose(meta["daily_anytime_demand_kw"], [51])
    assert meta["daily_demand_scale"] == pytest.approx(.25 / 32)
    assert problem.value == pytest.approx(.25 / 32 * (ON_RATE * 33 + ANY_RATE * 51))
    assert sum(meta["daily_demand_objective_terms"].values()) == pytest.approx(problem.value)
    assert set(c._objective_components) == {"electricity_objective_cost", "modeled_home_peak_objective_cost"}
    assert all(path["water_scenario_index"] == i // 5 and path["ev_scenario_index"] == i % 5
               for i, path in enumerate(args[3]["scenario_plans"]))
    json.dumps(meta, allow_nan=False)


def test_full_week_sums_five_daily_maxima_with_n672_and_no_weekend_charge():
    c = controller(168, perfect=True)
    start = pd.Timestamp("2025-11-30")
    starts = pd.date_range(start, periods=672, freq="15min")
    ev = np.zeros((1, 672))
    for day in range(5):
        date = pd.Timestamp("2025-12-01") + pd.Timedelta(days=day)
        ev[0, (starts >= date + pd.Timedelta(hours=6))
           & (starts < date + pd.Timedelta(hours=6, minutes=30))] = 11 + day
        ev[0, (starts >= date + pd.Timedelta(hours=12))
           & (starts < date + pd.Timedelta(hours=12, minutes=30))] = 20 + day
    ev[:, starts.dayofweek >= 5] = 1000
    args = inputs(c, start, ev, water_count=1, ev_count=1)
    problem = solve(c, args)
    meta = c.daily_demand_metadata(solved=True)
    assert meta["daily_demand_normalization_steps"] == 672
    assert meta["daily_demand_scale"] == pytest.approx(.25 / 672)
    assert len(meta["daily_demand_days"]) == 5
    assert len(meta["daily_demand_block_starts"]) == 5 * 48
    np.testing.assert_allclose(meta["daily_on_peak_demand_kw"], np.arange(11, 16))
    np.testing.assert_allclose(meta["daily_anytime_demand_kw"], np.arange(20, 25))
    assert problem.value == pytest.approx(.25 / 672 * (ON_RATE * 65 + ANY_RATE * 110))


def test_daily_floors_use_complete_earlier_blocks_and_survive_trace_restore():
    c = controller()
    start = pd.Timestamp("2025-12-01 10:00")
    history = np.full(600, 4.)
    history[120:150] = 40  # Anytime02:00 maximum.
    history[360:390] = 30  # On-peak06:00 maximum.
    history[-10:] = 100  # The complete09:30 block's mean is36, below40.
    observe(c, "2025-12-01", history)
    args = inputs(c, start)
    problem = solve(c, args, rest=10)
    meta = c.daily_demand_metadata(solved=True)
    np.testing.assert_allclose(meta["daily_demand_observed_on_peak_kw"], [30])
    np.testing.assert_allclose(meta["daily_demand_observed_anytime_kw"], [40])
    assert problem.value == pytest.approx(.25 / 32 * (ON_RATE * 30 + ANY_RATE * 40))
    assert c._observed_daily_demand_peaks(start + pd.Timedelta(hours=14)) == {
        "2025-12-02T00:00:00": {"on_peak_kw": 0., "anytime_kw": 0.},
    }
    restored = controller()
    restored._restored_daily_demand_peaks = meta["daily_demand_observed_peaks"]
    restored_problem = solve(restored, inputs(restored, start), rest=10)
    assert restored_problem.value == pytest.approx(problem.value)
    assert restored.daily_demand_metadata(solved=False)["daily_demand_objective_terms"] == {
        "on_peak": None, "anytime": None,
    }


def test_offpeak_midblock_uses_actual_half_for_anytime_max_and_signed_mean():
    c = controller()
    start = pd.Timestamp("2025-12-01 09:15")
    observe(c, "2025-12-01 09:00", [30.] * 15)
    profiles = np.zeros((35, 32))
    profiles[0, 0] = -10
    profiles[34, 0] = 10
    observed_half = c._observed_first_half_kw(start)
    problem = solve(c, inputs(c, start, profiles), observed_half=observed_half)
    meta = c.daily_demand_metadata(solved=True)
    np.testing.assert_allclose(np.asarray(meta["daily_demand_block_power_scenarios_kw"])[[0, 34], 0], [10, 20])
    np.testing.assert_allclose(meta["daily_anytime_demand_kw"], [20])
    np.testing.assert_allclose(meta["daily_on_peak_demand_kw"], [0])
    assert c._peak_demand_block_starts == ()  # Last17:00 block lacks its second half.
    assert problem.value == pytest.approx(.25 / 32 * ANY_RATE * 20)
    with pytest.raises(ValueError, match="observed first-half"):
        solve(c, inputs(c, start, profiles))


def test_weekend_zero_and_non_grid2_legacy_peak_remains_unchanged():
    c = controller()
    problem = solve(c, inputs(c, "2025-12-06", np.full((35, 32), 1000)))
    assert problem.value == 0
    assert c.daily_demand_metadata(solved=True)["daily_demand_days"] == []
    signed = np.zeros((35, 32))
    signed[:, :2] = [-20, 10]
    problem = solve(c, inputs(c, "2025-12-01 06:00", signed))
    assert problem.value == 0  # Positive part follows the signed30-minute mean.
    for rate, formulation in (("legacy", "weekday_daily_dt_over_n_v1"),
                              (WINTER_TOU, "weekday_daily_dt_over_n_v1"),
                              (WINTER_TOU_DEMAND, "weekday_daily_dt_over_n_v1"),
                              (WINTER_TOU_DEMAND_CPP, "legacy_horizon_peak")):
        legacy = controller(rate=rate, formulation=formulation)
        assert not legacy.uses_daily_grid2_demand
        problem = solve(legacy, inputs(legacy, "2025-12-01 06:00", np.full((35, 32), 22)))
        assert problem.value == pytest.approx(.71 * 22)
        assert legacy.daily_demand_metadata(solved=True) == {}

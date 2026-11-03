"""Analytic checks for bounded, scenario-specific EV trip-energy shortfall.

The approved cost is 10,000/N * sum_j(dt_j * epsilon_j / current_capacity).
Shortfall relaxes positive driving demand only; physical SOC bounds, event
timing, command sharing, and worst-complete-path comfort remain unchanged.
"""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from test.test_ev_v2h import vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.controllers.ev import (
    EVMPCController, EVHorizon, build_ev_problem,
)
from examples.single_home_hems.single_home_hems.controllers.ev_joint import (
    build_joint_ev_path_problem,
)
from examples.single_home_hems.single_home_hems.controllers.ev_scenario import (
    build_joint_ev_scenario_problem,
)


def shortfall_vehicle(formulation="convex", **options):
    settings = dict(
        formulation=formulation,
        solver="SCIPY" if formulation == "milp" else "CLARABEL",
        horizon_hours=1, health_weight=0, comfort_weight=0,
        electricity_weight=0, persistence_weight=0,
        enforce_terminal_soc=False, enforce_cycle_limit=False,
        enforce_dwell_constraints=False, enforce_discharge_budget=False,
        enable_degradation=False,
    )
    settings.update(options)
    return vehicle(**settings)


def build_path(formulation="convex", trips=(), *, timing="arrival", soc=.5,
               **options):
    ev = shortfall_vehicle(formulation, **options)
    mobility = schedule(trips)
    predicted = forecast(ev, mobility, price=0, load=0)
    controller = EVMPCController(ev)
    problem, plan = build_ev_problem(
        controller, mobility.start, soc, predicted, trip_timing=timing,
    )
    return controller, predicted, problem, plan


def solve(problem, formulation):
    solver = "SCIPY" if formulation == "milp" else "CLARABEL"
    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed")
    problem.solve(solver=solver)
    return problem.status


@pytest.mark.parametrize("formulation", ["convex", "milp"])
@pytest.mark.parametrize("timing,trips,event_minute", [
    ("arrival", [(0, 30, 100)], 15),
    ("departure", [(0, 180, 100)], 0),
    ("departure", [(-30, 180, 100)], 0),
    ("departure", [(30, 180, 100)], 30),
])
def test_long_trip_uses_only_unavoidable_shortfall_and_keeps_hard_soc_bounds(
        formulation, timing, trips, event_minute):
    _, predicted, problem, plan = build_path(formulation, trips, timing=timing)
    held = cp.Problem(problem.objective,
                      [*problem.constraints, plan["chemical"] == 0])
    assert solve(held, formulation) in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    h = plan["horizon"]
    event = int(h.t.get_loc(predicted.data.index[0] +
                          pd.Timedelta(minutes=event_minute)))
    assert h.trip_kwh.sum() == pytest.approx(20)
    np.testing.assert_array_equal(plan["trip_shortfall_indices"], [event])
    expected = np.zeros(len(h.dt))
    expected[event] = 15  # 20 kWh requested minus the measured 5 kWh inventory.
    np.testing.assert_allclose(plan["trip_shortfall_kwh"].value, expected, atol=1e-6)
    np.testing.assert_allclose(plan["served_trip_kwh"].value,
                               h.trip_kwh - expected, atol=1e-6)
    state = plan["state"].value
    assert state.min() >= -1e-7
    assert state.max() <= 1 + 1e-7
    assert state[-1] == pytest.approx(0, abs=1e-7)
    if timing == "arrival":
        assert state[0] == pytest.approx(.5, abs=1e-7)
        assert state[event] == pytest.approx(.5, abs=1e-7)
        assert state[event + 1] == pytest.approx(0, abs=1e-7)
    else:
        # Departure consumption stays at the departure instant, even when
        # the return is after the horizon or departure precedes the issue.
        assert state[event] == pytest.approx(0, abs=1e-7)
        if event:
            assert state[event - 1] == pytest.approx(.5, abs=1e-7)
    assert plan["trip_shortfall"].value == pytest.approx(937.5, abs=1e-4)
    assert held.value == pytest.approx(937.5, abs=1e-4)


@pytest.mark.parametrize("formulation", ["convex", "milp"])
@pytest.mark.parametrize("timing", ["arrival", "departure"])
def test_serviceable_trip_has_zero_shortfall(formulation, timing):
    _, _, problem, plan = build_path(formulation, [(0, 30, 10)], timing=timing)
    held = cp.Problem(problem.objective,
                      [*problem.constraints, plan["chemical"] == 0])
    assert solve(held, formulation) in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    np.testing.assert_allclose(plan["trip_shortfall_kwh"].value, 0, atol=1e-6)
    assert plan["served_trip_kwh"].value.sum() == pytest.approx(2, abs=1e-6)
    assert plan["state"].value[-1] == pytest.approx(.3, abs=1e-6)
    assert plan["trip_shortfall"].value == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize("timing", ["arrival", "departure"])
def test_cost_uses_event_dt_full_control_count_and_measured_faded_capacity(timing):
    controller, _, _, plan = build_path(
        trips=[(7, 12, 100), (35, 43, 100)], timing=timing, initial_soh=.8,
    )
    assert controller.options.trip_shortfall_weight == 10_000
    assert plan["capacity_kwh"] == 8
    assert plan["N"] == 4
    assert len(plan["horizon"].dt) > plan["N"]
    plan["trip_shortfall_variable"].value = np.array([2., 3.])
    # The event integration blocks are five and eight minutes. They remain
    # two events inside four decisions, rather than changing the denominator.
    expected = 10_000 / 4 * ((5 / 60) * 2 / 8 + (8 / 60) * 3 / 8)
    assert plan["trip_shortfall"].value == pytest.approx(expected)
    assert plan["objective_terms"]["trip_shortfall"].value == pytest.approx(expected)


def test_custom_shortfall_weight_is_honored():
    _, _, _, plan = build_path(trips=[(0, 30, 100)], trip_shortfall_weight=320)
    plan["trip_shortfall_variable"].value = np.array([2.])
    assert plan["trip_shortfall"].value == pytest.approx(320 * .25 / 4 * 2 / 10)


@pytest.mark.parametrize("formulation", ["convex", "milp"])
def test_shortfall_cannot_exceed_positive_demand_or_create_battery_energy(formulation):
    _, _, problem, plan = build_path(
        formulation, [(0, 180, 10)], timing="departure",
    )
    # A deliberately opposing objective tries to turn slack into stored
    # energy. Its cap allows cancelling this 2 kWh demand, never adding energy.
    maximize_soc = cp.Problem(cp.Maximize(plan["state"][-1]),
        [*problem.constraints, plan["chemical"] == 0])
    assert solve(maximize_soc, formulation) in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert plan["trip_shortfall_kwh"].value.sum() == pytest.approx(2, abs=1e-6)
    assert plan["state"].value[-1] == pytest.approx(.5, abs=1e-6)
    np.testing.assert_allclose(plan["served_trip_kwh"].value, 0, atol=1e-6)


@pytest.mark.parametrize("timing", ["arrival", "departure"])
def test_negative_replay_energy_is_preserved_and_has_no_slack(timing):
    ev = shortfall_vehicle(battery_model="power_loss")
    mobility = schedule([(0, 15, 0), (30, 45, 0)])
    trips = mobility.trips.assign(arrival_net_soc_depletion_pp=[20., -20.])
    mobility = replace(mobility, trips=trips)
    predicted = forecast(ev, mobility, price=0, load=0)
    problem, plan = build_ev_problem(
        EVMPCController(ev), mobility.start, .5, predicted, trip_timing=timing,
    )
    positive = np.flatnonzero(plan["horizon"].trip_kwh > 0)
    negative = np.flatnonzero(plan["horizon"].trip_kwh < 0)
    np.testing.assert_array_equal(plan["trip_shortfall_indices"], positive)
    assert len(positive) == len(negative) == 1
    maximize_soc = cp.Problem(cp.Maximize(plan["state"][-1]),
        [*problem.constraints, plan["chemical"] == 0])
    assert solve(maximize_soc, "convex") in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert plan["trip_shortfall_kwh"].value[negative[0]] == 0
    assert plan["served_trip_kwh"].value[negative[0]] == pytest.approx(-2)
    assert plan["trip_shortfall_kwh"].value.sum() == pytest.approx(2, abs=1e-6)
    assert plan["state"].value[-1] == pytest.approx(.7, abs=1e-6)


@pytest.mark.parametrize("formulation", ["convex", "milp"])
@pytest.mark.parametrize("timing", ["arrival", "departure"])
@pytest.mark.parametrize("depletion_pp", [100., -100.])
def test_trip_slack_does_not_relax_either_physical_soc_bound(
        formulation, timing, depletion_pp):
    ev = shortfall_vehicle(formulation, battery_model="power_loss")
    mobility = schedule([(0, 30, 0)])
    mobility = replace(mobility, trips=mobility.trips.assign(
        arrival_net_soc_depletion_pp=[depletion_pp]))
    predicted = forecast(ev, mobility, price=0, load=0)
    problem, plan = build_ev_problem(
        EVMPCController(ev), mobility.start, .5, predicted, trip_timing=timing,
    )
    constraints = [*problem.constraints, plan["chemical"] == 0]
    if depletion_pp > 0:
        # Serving all 10 kWh from a 5 kWh inventory must remain impossible.
        constraints.append(plan["trip_shortfall_kwh"] == 0)
    else:
        # Negative supplied depletion would overfill the battery. Slack exists
        # only for positive demand and cannot cancel that exogenous increment.
        assert plan["trip_shortfall_variable"] is None
    held = cp.Problem(problem.objective, constraints)
    assert solve(held, formulation) == cp.INFEASIBLE


@pytest.mark.parametrize("negative_trip", [False, True])
@pytest.mark.parametrize("timing", ["arrival", "departure"])
def test_no_positive_events_create_no_shortfall_variable(negative_trip, timing):
    ev = shortfall_vehicle(battery_model="power_loss")
    mobility = schedule([(0, 30, 0)] if negative_trip else [])
    mobility = replace(mobility, trips=mobility.trips.assign(
        arrival_net_soc_depletion_pp=[-20.] if negative_trip else []))
    predicted = forecast(ev, mobility, price=0, load=0)
    _, plan = build_ev_problem(EVMPCController(ev), mobility.start, .5,
                               predicted, trip_timing=timing)
    assert plan["trip_shortfall_variable"] is None
    assert len(plan["trip_shortfall_indices"]) == 0
    assert not plan["trip_shortfall_kwh"].variables()
    np.testing.assert_array_equal(plan["trip_shortfall_kwh"].value, 0)
    np.testing.assert_array_equal(plan["served_trip_kwh"].value,
                                  plan["horizon"].trip_kwh)
    assert plan["trip_shortfall"].value == 0


def scenario_forecast(ev, paths):
    predictions = tuple(forecast(ev, path, price=0, load=3) for path in paths)
    return EVScenarioSet(paths[0].start, tuple(paths),
        tuple(p.backed_load_kw for p in predictions), predictions,
        tuple(f"path_{i}" for i in range(len(paths))))


@pytest.mark.parametrize("formulation", ["convex", "milp"])
def test_scenarios_have_independent_slack_mean_trip_cost_and_max_path_comfort(formulation):
    ev = shortfall_vehicle(formulation, horizon_hours=2,
                            comfort_soc_min=.4, comfort_weight=1000)
    scenarios = scenario_forecast(ev, [schedule([(15, 30, 100)]),
                                        schedule([(15, 45, 100)])])
    _, plan = EVMPCController(ev).build_problem(scenarios.issued_at, .5, scenarios)
    first, second = plan["scenario_plans"]
    assert first["trip_shortfall_variable"].id != second["trip_shortfall_variable"].id
    first["trip_shortfall_variable"].value = np.array([2.])
    second["trip_shortfall_variable"].value = np.array([6.])
    first["state"].value = np.full(len(first["boundaries"]), .2)
    second["state"].value = np.full(len(second["boundaries"]), .2)
    assert first["trip_shortfall"].value == pytest.approx(62.5)
    assert second["trip_shortfall"].value == pytest.approx(187.5)
    assert plan["trip_shortfall"].value == pytest.approx(125)
    assert plan["trip_shortfall"].value != pytest.approx(187.5)
    assert first["comfort"].value == pytest.approx(18.75)
    assert second["comfort"].value == pytest.approx(12.5)
    assert plan["comfort"].value == pytest.approx(18.75)


@pytest.mark.parametrize("formulation", ["convex", "milp"])
def test_adding_slack_preserves_shared_first_command(formulation):
    ev = shortfall_vehicle(formulation)
    scenarios = scenario_forecast(ev, [schedule([(30, 180, 100)]),
                                        schedule([(45, 180, 50)])])
    problem, plan = EVMPCController(ev).build_problem(scenarios.issued_at, .5, scenarios)
    first, second = plan["scenario_plans"]
    key = "charge_mode" if formulation == "milp" else "chemical"
    a = first[key][first["command_index"]]
    b = second[key][second["command_index"]]
    incompatible = cp.Problem(problem.objective,
        [*problem.constraints, a == 1, b == 0])
    assert solve(incompatible, formulation) == cp.INFEASIBLE


def test_joint_path_applies_same_bounded_departure_shortfall_and_cost():
    ev = shortfall_vehicle("milp")
    mobility = schedule([(0, 180, 100)])
    predicted = forecast(ev, mobility, price=0, load=0)
    controller = EVMPCController(ev)
    h = EVHorizon.from_forecast(controller, mobility.start, predicted,
                                trip_timing="departure")
    problem, plan = build_joint_ev_path_problem(
        controller, mobility.start, .5, predicted, h,
        cp.Constant(np.zeros(len(h.dt))),
    )
    assert solve(problem, "milp") == cp.OPTIMAL
    np.testing.assert_allclose(plan["trip_shortfall_kwh"].value,
                               [15, 0, 0, 0], atol=1e-7)
    assert plan["served_trip_kwh"].value.sum() == pytest.approx(5)
    np.testing.assert_allclose(plan["state"].value, 0, atol=1e-7)
    assert plan["trip_shortfall"].value == pytest.approx(937.5)
    assert problem.value == pytest.approx(937.5)


def test_joint_water_by_ev_paths_keep_independent_slack_and_arithmetic_mean_cost():
    ev = shortfall_vehicle("milp")
    scenarios = scenario_forecast(ev, [schedule([(0, 180, 100)]) for _ in range(5)])
    _, plan = build_joint_ev_scenario_problem(
        EVMPCController(ev), scenarios.issued_at, .5, scenarios,
        np.zeros((7, 4)), np.zeros(4), pd.Timedelta(minutes=15),
        eligible_power_upper_kw=1,
    )
    paths = plan["scenario_plans"]
    assert len(paths) == 35
    assert len({p["trip_shortfall_variable"].id for p in paths}) == 35
    for path in paths:
        path["trip_shortfall_variable"].value = np.array([path["ev_scenario_index"] + 1.])
    # Each 15-minute impulse costs 62.5 times its shortfall kWh. Seven
    # water paths repeat each EV path, without summing or taking a maximum.
    assert plan["trip_shortfall"].value == pytest.approx(62.5 * 3)
    assert plan["objective_terms"]["trip_shortfall"].value == pytest.approx(187.5)

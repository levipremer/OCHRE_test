"""Approved EV comfort: home SOC below 40%, except the hour after each return.

The reference values below integrate eligible hours analytically. They use the
fixed number of control intervals, rather than the number of home blocks or
departures, and distinguish the maximum path total from per-step maxima.
"""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from test.test_ev_v2h import vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.controllers import ev as ev_controller
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.controllers.ev_joint import build_joint_ev_path_problem


def comfort_vehicle(formulation="convex", hours=2):
    return vehicle(
        formulation=formulation, solver="SCIPY" if formulation == "milp" else "CLARABEL",
        horizon_hours=hours, comfort_soc_min=.4, comfort_weight=1000,
        health_weight=0, persistence_weight=0, electricity_weight=0,
        enforce_terminal_soc=False, enforce_cycle_limit=False,
        enforce_dwell_constraints=False, enforce_discharge_budget=False,
        enable_degradation=False,
    )


def build(mobility=None, *, formulation="convex", hours=2, issue_minutes=0):
    ev = comfort_vehicle(formulation, hours)
    mobility = schedule() if mobility is None else mobility
    controller = EVMPCController(ev)
    issued = mobility.start + pd.Timedelta(minutes=issue_minutes)
    predicted = forecast(ev, mobility, price=0, load=0)
    problem, plan = controller.build_problem(issued, .2, predicted)
    return controller, predicted, problem, plan


def constant_soc_cost(plan, soc=.2):
    plan["state"].value = np.full(len(plan["boundaries"]), soc)
    return float(plan["comfort"].value)


@pytest.mark.parametrize("formulation", ["convex", "milp"])
def test_no_trips_still_penalizes_low_soc_during_all_home_hours(formulation):
    _, _, _, plan = build(formulation=formulation)
    h = plan["horizon"]
    assert len(h.departure_indices) == 0
    assert h.N == 8
    np.testing.assert_allclose(h.comfort_dt, h.dt)
    assert constant_soc_cost(plan) == pytest.approx(1000 * .2 * 2 / 8)
    assert constant_soc_cost(plan, .4) == 0
    assert constant_soc_cost(plan, .6) == 0


@pytest.mark.parametrize("use_metadata", [False, True])
def test_partially_elapsed_initial_grace_uses_actual_return_time(use_metadata):
    mobility = schedule() if use_metadata else schedule([(-90, -37, 0)])
    if use_metadata:
        mobility = replace(mobility, metadata={
            "last_observed_arrival": (mobility.start - pd.Timedelta(minutes=37)).isoformat(),
        })
    _, _, _, plan = build(mobility, hours=1)
    h = plan["horizon"]
    # The grace expires 23 minutes after issue, inside the second control block.
    # Only seven minutes of [15, 30) are eligible; no extra decision is created.
    np.testing.assert_allclose(h.comfort_dt, np.array([0, 7, 15, 15]) / 60)
    assert h.N == 4
    assert constant_soc_cost(plan) == pytest.approx(1000 * .2 * (37 / 60) / 4)

    # A subsequent issue must retain the same arrival clock rather than restart
    # an hour of grace. The next horizon has 52 eligible minutes.
    _, _, _, updated = build(mobility, hours=1, issue_minutes=15)
    np.testing.assert_allclose(updated["horizon"].comfort_dt, np.array([7, 15, 15, 15]) / 60)
    assert constant_soc_cost(updated) == pytest.approx(1000 * .2 * (52 / 60) / 4)


@pytest.mark.parametrize("arrival_minutes, eligible_minutes", [(0, 60), (-60, 120)])
def test_arrival_at_issue_and_exact_one_hour_boundary(arrival_minutes, eligible_minutes):
    mobility = schedule([(arrival_minutes - 30, arrival_minutes, 0)])
    _, _, _, plan = build(mobility)
    assert plan["horizon"].comfort_dt.sum() == pytest.approx(eligible_minutes / 60)
    assert constant_soc_cost(plan) == pytest.approx(1000 * .2 * (eligible_minutes / 60) / 8)


def test_away_low_soc_never_receives_comfort_penalty():
    _, _, _, plan = build(schedule([(-30, 180, 0)]))
    assert not plan["horizon"].z.any()
    np.testing.assert_array_equal(plan["horizon"].comfort_dt, 0)
    assert constant_soc_cost(plan, 0) == 0


def test_future_return_exempts_only_its_first_hour_and_terminal_is_not_a_stage():
    _, _, _, plan = build(schedule([(15, 30, 0)]))
    h = plan["horizon"]
    # Home [0,15), away [15,30), grace [30,90), eligible home [90,120).
    expected = np.array([15, 0, 0, 0, 0, 0, 15, 15]) / 60
    np.testing.assert_allclose(h.comfort_dt, expected)
    assert constant_soc_cost(plan) == pytest.approx(1000 * .2 * .75 / 8)

    state = np.full(len(h.t), .4)
    state[-1] = 0  # Terminal SOC is excluded, consistently with other stage costs.
    plan["state"].value = state
    assert plan["comfort"].value == 0
    state[np.flatnonzero(h.comfort_dt)[-1]] = .1
    plan["state"].value = state
    assert plan["comfort"].value == pytest.approx(1000 * .3 * .25 / 8)


def test_each_return_restarts_its_own_home_grace_period():
    _, _, _, plan = build(schedule([(0, 15, 0), (45, 60, 0)]))
    # The short home stay [15,45) is wholly in grace. The next return's
    # independent grace covers [60,120), so the whole horizon is exempt.
    np.testing.assert_array_equal(plan["horizon"].comfort_dt, 0)
    assert constant_soc_cost(plan, 0) == 0


def test_each_scenario_uses_its_own_return_and_home_mask():
    ev = comfort_vehicle()
    paths = (schedule([(0, 15, 0)]), schedule([(0, 45, 0)]))
    predictions = tuple(forecast(ev, path, price=0, load=0) for path in paths)
    scenarios = EVScenarioSet(paths[0].start, paths,
        tuple(prediction.backed_load_kw for prediction in predictions), predictions,
        ("early_return", "late_return"))
    _, plan = EVMPCController(ev).build_problem(paths[0].start, .2, scenarios)
    early, late = plan["scenario_plans"]
    assert early["horizon"].comfort_dt.sum() == pytest.approx(.75)
    assert late["horizon"].comfort_dt.sum() == pytest.approx(.25)
    assert constant_soc_cost(early) == pytest.approx(18.75)
    assert constant_soc_cost(late) == pytest.approx(6.25)
    assert plan["comfort"].value == pytest.approx(18.75)


def test_scenario_aggregation_is_maximum_total_path_cost_not_stepwise_max_or_mean():
    ev = comfort_vehicle()
    paths = (schedule(), schedule())
    predictions = tuple(forecast(ev, path, price=0, load=0) for path in paths)
    scenarios = EVScenarioSet(paths[0].start, paths,
        tuple(prediction.backed_load_kw for prediction in predictions), predictions, ("a", "b"))
    _, plan = EVMPCController(ev).build_problem(paths[0].start, .2, scenarios)
    first, second = plan["scenario_plans"]
    first["state"].value = np.array([.2] * 4 + [.4] * 5)
    second["state"].value = np.array([.4] * 4 + [.1] * 4 + [.4])
    assert first["comfort"].value == pytest.approx(25)
    assert second["comfort"].value == pytest.approx(37.5)
    assert plan["comfort"].value == pytest.approx(37.5)
    assert plan["comfort"].value != pytest.approx(62.5)  # Sum of per-step maxima.
    assert plan["comfort"].value != pytest.approx(31.25)  # Mean path total.


def test_fixed_control_interval_denominator_is_invariant_to_integration_splits():
    _, _, _, plan = build()
    h = plan["horizon"]
    base = ev_controller._ev_comfort_cost(h, cp.Constant(np.full(len(h.dt) + 1, .2)), .4)
    # Split every integration block in half while retaining the eight control
    # intervals. A split is not a new decision and cannot halve the objective.
    split_h = replace(h, dt=np.repeat(h.dt / 2, 2),
                      comfort_dt=np.repeat(h.comfort_dt / 2, 2))
    split = ev_controller._ev_comfort_cost(
        split_h, cp.Constant(np.full(len(split_h.dt) + 1, .2)), .4)
    assert split_h.N == h.N == 8
    assert base.value == pytest.approx(.2 * 2 / 8)
    assert split.value == pytest.approx(base.value)


def test_fixed_low_soc_solve_matches_convex_milp_and_joint_builders():
    values = []
    for formulation, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
        if solver not in cp.installed_solvers():
            pytest.skip(f"{solver} is not installed")
        c, predicted, _, _ = build(schedule([(15, 30, 0)]), formulation=formulation)
        issue = predicted.ev_schedules[c.ev.vehicle_id].start
        problem, plan = ev_controller.build_ev_problem(c, issue, .2, predicted,
                                                       trip_timing="departure")
        held = cp.Problem(problem.objective, [*problem.constraints, plan["chemical"] == 0])
        held.solve(solver=solver)
        assert held.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        values.append(float(plan["comfort"].value))
        assert values[-1] == pytest.approx(18.75, abs=1e-6)

        if formulation == "milp":
            joint, joint_plan = build_joint_ev_path_problem(
                c, issue, .2, predicted, plan["horizon"],
                cp.Constant(np.zeros(len(plan["horizon"].dt))),
            )
            held_joint = cp.Problem(joint.objective,
                [*joint.constraints, joint_plan["chemical"] == 0])
            held_joint.solve(solver=solver)
            assert held_joint.status == cp.OPTIMAL
            values.append(float(joint_plan["comfort"].value))
    np.testing.assert_allclose(values, 18.75, atol=1e-6)

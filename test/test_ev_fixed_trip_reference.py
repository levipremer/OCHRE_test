"""Fixed source-pack trip energy stays independent of replica capacity and SOH."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import json

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule, audi_schedule
from examples.single_home_hems.single_home_hems.control_config import (
    EV_CONTROL, load_unified_documents,
)
from examples.single_home_hems.single_home_hems.controllers.ev import (
    EVMPCController, EVHorizon, build_ev_problem,
)
from test.test_ev_v2h import CONFIG, ROOT, EVPlantTests, forecast, vehicle


def replica(*, reference=60.0, initial_soc=.8, **options):
    return replace(vehicle(battery_model="power_loss", initial_soh=.95,
                           soc_depletion_reference_capacity_kwh=reference,
                           standby_vehicle_kw=0, **options),
                   capacity_kwh=80, initial_soc_fraction=initial_soc)


def mobility(events=((1, 2, 10.0), (3, 4, 10.0), (5, 6, -2.0))):
    start = pd.Timestamp("2018-01-01")
    rows = [dict(departure=start+pd.Timedelta(minutes=departure),
                 arrival=start+pd.Timedelta(minutes=arrival), distance_km=999,
                 arrival_net_soc_depletion_pp=depletion)
            for departure, arrival, depletion in events]
    return EVSchedule(pd.DataFrame(rows), start, start+pd.Timedelta(hours=2), {})


def plant(ev, schedule):
    return BidirectionalEV(ev_config=ev, mobility=schedule,
                           start_time=dt.datetime(2018, 1, 1), duration=dt.timedelta(hours=1),
                           time_res=dt.timedelta(minutes=1), verbosity=7, save_results=False)


def test_reference_is_optional_physics_and_unified_documents_preserve_it():
    home_file = ROOT / "examples/single_home_hems/config/unified_home_sunday_ideal_audi.json"
    control_file = ROOT / "examples/single_home_hems/config/unified_control.json"
    homes, controls = json.loads(home_file.read_text()), json.loads(control_file.read_text())
    baseline = load_unified_documents(homes, controls, home_file, ROOT)
    assert baseline.neighborhood.homes[0].evs[0].hems.soc_depletion_reference_capacity_kwh is None
    assert "soc_depletion_reference_capacity_kwh" not in EV_CONTROL
    changed = deepcopy(homes)
    ev_doc = changed["homes"]["home_01"]["evs"]["01A"]
    ev_doc["capacity_kwh"] = 80
    ev_doc["physical"].update(initial_soh=.95, soc_depletion_reference_capacity_kwh=60)
    configured = load_unified_documents(changed, controls, home_file, ROOT)
    ev = configured.neighborhood.homes[0].evs[0]
    assert ev.capacity_kwh * ev.hems.initial_soh == 76
    assert ev.hems.soc_depletion_reference_capacity_kwh == 60
    again = load_unified_documents(configured.home_document, configured.control_document, home_file, ROOT)
    assert again.neighborhood.homes[0].evs[0].hems == ev.hems
    assert "soc_depletion_reference_capacity_kwh" not in homes["homes"]["home_01"]["evs"]["01A"]["physical"]


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True, "60"])
def test_reference_requires_finite_positive_capacity(invalid):
    with pytest.raises(ValueError, match="soc_depletion_reference_capacity_kwh"):
        replica(reference=invalid)


def test_reference_does_not_silently_change_distance_model():
    with pytest.raises(ValueError, match="power_loss"):
        vehicle(soc_depletion_reference_capacity_kwh=60)


def test_forecast_fixed_energy_and_departure_timing_survive_capacity_changes():
    ev = replica()
    schedule = mobility(((15, 30, 10.), (45, 80, -2.)))
    controller = EVMPCController(ev)
    issued = forecast(ev, schedule)
    for capacity in (76., 70., 51.):
        controller.current_capacity = capacity
        horizon = EVHorizon.from_forecast(controller, schedule.start, issued, trip_timing="departure")
        assert horizon.trip_kwh[horizon.t.get_loc(schedule.start+pd.Timedelta(minutes=15))] == 6
        assert horizon.trip_kwh[horizon.t.get_loc(schedule.start+pd.Timedelta(minutes=45))] == -1.2
        assert horizon.trip_kwh.sum() == pytest.approx(4.8)
        # The negative event is retained even though its return is after the horizon.
        assert horizon.trip_timing == "departure"


def test_ongoing_trip_fixed_energy_is_carried_to_issue():
    ev = replica()
    schedule = mobility(((-15, 30, 10.),))
    horizon = EVHorizon.from_forecast(EVMPCController(ev), schedule.start,
                                      forecast(ev, schedule), trip_timing="departure")
    assert horizon.trip_kwh[0] == 6
    assert horizon.trip_kwh.sum() == 6


def test_true_plant_applies_fixed_energy_at_return_after_fade_and_preserves_sign():
    ev = replica(reference_cycle_count=3)
    equipment = plant(ev, mobility())
    assert equipment.capacity == 76
    np.testing.assert_array_equal(equipment.trips.energy_kwh, [6., 6., -1.2])
    rows = [EVPlantTests().step(equipment, 0) for _ in range(7)]
    name = equipment.results_name
    # Departing does not consume true-plant energy. The arrival event does.
    assert rows[1][f"{name} Trip Energy (kWh)"] == 0
    assert rows[2][f"{name} Trip Energy (kWh)"] == pytest.approx(6)
    assert rows[2][f"{name} SOC (-)"] == pytest.approx(.8-6/76)
    assert rows[3][f"{name} Capacity (kWh)"] < 76
    assert rows[4][f"{name} Trip Energy (kWh)"] == pytest.approx(6)
    assert rows[6][f"{name} Trip Energy (kWh)"] == pytest.approx(-1.2)
    assert rows[6][f"{name} Trip Unserved Energy (kWh)"] == pytest.approx(0)
    for index in (2, 4, 6):
        prior_soc = rows[index-1][f"{name} End SOC (-)"]
        prior_capacity = rows[index-1][f"{name} End Capacity (kWh)"]
        assert rows[index][f"{name} SOC (-)"] == pytest.approx(
            prior_soc - rows[index][f"{name} Trip Energy (kWh)"]/prior_capacity)


def test_fixed_requested_energy_keeps_physical_bounds_and_departure_shortfall():
    equipment = plant(replica(initial_soc=.05), mobility(((1, 2, 10.),)))
    rows = [EVPlantTests().step(equipment, 0) for _ in range(3)]
    name = equipment.results_name
    assert rows[1][f"{name} Unmet Load (kWh)"] == pytest.approx(2.2)
    assert rows[2][f"{name} SOC (-)"] == 0
    assert rows[2][f"{name} Trip Energy (kWh)"] == pytest.approx(3.8)
    assert rows[2][f"{name} Trip Unserved Energy (kWh)"] == pytest.approx(2.2)


@pytest.mark.parametrize("formulation,solver", [("convex", "CLARABEL"), ("milp", "SCIPY")])
def test_predicted_shortfall_uses_fixed_trip_energy_and_current_soc_capacity(formulation, solver):
    ev = replica(initial_soc=.05, formulation=formulation, solver=solver)
    schedule = mobility(((0, 180, 10.),))
    # Availability covers this ongoing trip through the entire horizon.
    schedule = replace(schedule, end=schedule.start+pd.Timedelta(hours=4))
    controller = EVMPCController(ev)
    problem, plan = build_ev_problem(controller, schedule.start, .05,
        forecast(ev, schedule), trip_timing="departure")
    held = cp.Problem(problem.objective, [*problem.constraints, plan["chemical"] == 0])
    held.solve(solver=solver)
    assert held.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert plan["horizon"].trip_kwh[0] == 6
    assert plan["trip_shortfall_kwh"].value.sum() == pytest.approx(2.2, abs=1e-6)
    assert plan["served_trip_kwh"].value.sum() == pytest.approx(3.8, abs=1e-6)
    assert plan["state"].value.min() >= -1e-7
    assert plan["state"].value.max() <= 1+1e-7
    assert plan["trip_shortfall"].value == pytest.approx(10_000*.25*2.2/(76*4), abs=1e-5)
    # An opposing objective can cancel at most the same 6 kWh demand.
    cancel = cp.Problem(cp.Maximize(plan["state"][-1]),
                        [*problem.constraints, plan["chemical"] == 0])
    cancel.solve(solver=solver)
    assert cancel.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert plan["trip_shortfall_kwh"].value.sum() == pytest.approx(6, abs=1e-6)
    assert plan["state"].value[-1] == pytest.approx(.05, abs=1e-7)


def test_legacy_opt_out_still_uses_current_capacity():
    ev = replica(reference=None)
    schedule = mobility(((15, 30, 10.),))
    controller = EVMPCController(ev)
    controller.current_capacity = 70
    horizon = EVHorizon.from_forecast(controller, schedule.start, forecast(ev, schedule),
                                      trip_timing="departure")
    assert horizon.trip_kwh.sum() == 7
    equipment = plant(ev, mobility(((1, 2, 10.),)))
    rows = [EVPlantTests().step(equipment, 0) for _ in range(3)]
    assert rows[2][f"{equipment.results_name} Trip Energy (kWh)"] == pytest.approx(7.6)


def test_actual_audi_schedule_preserves_fixed_energy_for_all_history_and_future():
    config = replace(CONFIG, start_time=pd.Timestamp("2025-11-30", tz="America/Indiana/Indianapolis").to_pydatetime(),
                     duration=dt.timedelta(hours=176))
    ev = replica(source="audi", source_vehicle="CA_299a889dac408996", seed=43,
                 history_days=31, horizon_hours=8, forecast_margin_hours=8)
    source = ROOT / "examples/five_home_block/inputs/ev/audi_forecasting_15min_100vehicle.parquet"
    original = audi_schedule(source, ev, config)
    delayed = audi_schedule(source, replace(ev, hems=replace(ev.hems, return_delay_minutes=90,
        return_delay_collision_policy="cap_before_next_departure")), config)
    assert len(original.trips) == len(delayed.trips) == 31
    original_energy = original.demand(ev.hems.consumption_kwh_per_km,
        soc_depletion_capacity_kwh=ev.hems.soc_depletion_capacity_kwh(76)).energy_kwh
    delayed_energy = delayed.demand(ev.hems.consumption_kwh_per_km,
        soc_depletion_capacity_kwh=ev.hems.soc_depletion_capacity_kwh(75)).energy_kwh
    np.testing.assert_allclose(original_energy,
                               original.trips.arrival_net_soc_depletion_pp*.6)
    np.testing.assert_array_equal(original_energy, delayed_energy)
    pd.testing.assert_series_equal(original.trips.departure, delayed.trips.departure)
    assert original.metadata["soc_depletion_reference_capacity_kwh"] == 60
    assert delayed.metadata["soc_depletion_reference_capacity_kwh"] == 60


def test_checkpoint_rejects_different_trip_reference_but_reads_legacy_none():
    ev, schedule = replica(), mobility(((15, 30, 10.),))
    state = plant(ev, schedule).battery_checkpoint()
    plant(ev, schedule).restore_battery_checkpoint(state)
    with pytest.raises(ValueError, match="initialization"):
        plant(replace(ev, hems=replace(ev.hems, soc_depletion_reference_capacity_kwh=80)), schedule).restore_battery_checkpoint(state)
    old = plant(replica(reference=None), schedule)
    legacy = old.battery_checkpoint()
    legacy.pop("soc_depletion_reference_capacity_kwh")
    old.restore_battery_checkpoint(legacy)

"""The measured EV shutoff can restart before the next MPC issue."""

import pandas as pd
import pytest

from examples.single_home_hems.single_home_hems.ev_panel_guard import EVPanelGuard


def _command(power_kw: float) -> dict:
    return {"EV Charger": {"P Setpoint": power_kw}}


def _apply(guard: EVPanelGuard, minute: str, measured_kw: float, power_kw: float):
    return guard.apply(
        pd.Timestamp(minute, tz="America/Indiana/Indianapolis"),
        {"Total Electric Power (kW)": measured_kw},
        _command(power_kw),
        ("EV Charger",),
    )


def test_panel_exceedance_cuts_charge_and_latches_until_next_decision():
    guard = EVPanelGuard(38.0)
    controls, record = _apply(guard, "2025-12-03 19:07", 38.1, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 0
    assert record["ev_panel_guard_tripped"]
    assert record["ev_panel_guard_latched_until"] == "2025-12-03 19:15:00-05:00"

    controls, record = _apply(guard, "2025-12-03 19:14", 30.0, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 0
    assert record["ev_panel_guard_active"]

    controls, record = _apply(guard, "2025-12-03 19:15", 30.0, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 11.5
    assert not record["ev_panel_guard_active"]


def test_panel_guard_restarts_below_25_kw_with_headroom_then_retrips():
    guard = EVPanelGuard(38.0)
    _apply(guard, "2025-12-03 19:07", 38.1, 11.5)

    controls, record = _apply(guard, "2025-12-03 19:08", 25.0, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 0
    assert record["ev_panel_guard_active"]

    controls, record = _apply(guard, "2025-12-03 19:09", 24.9, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 11.5
    assert not record["ev_panel_guard_active"]
    assert record["ev_panel_guard_latched_until"] is None

    controls, record = _apply(guard, "2025-12-03 19:10", 38.1, 11.5)
    assert controls["EV Charger"]["P Setpoint"] == 0
    assert record["ev_panel_guard_tripped"]
    assert record["ev_panel_guard_latched_until"] == "2025-12-03 19:15:00-05:00"


def test_panel_guard_restart_requires_room_for_full_requested_charge():
    guard = EVPanelGuard(38.0)
    _apply(guard, "2025-12-03 19:07", 38.1, 14.0)

    controls, record = _apply(guard, "2025-12-03 19:08", 24.9, 14.0)
    assert controls["EV Charger"]["P Setpoint"] == 0
    assert record["ev_panel_guard_active"]

    controls, record = _apply(guard, "2025-12-03 19:09", 23.9, 14.0)
    assert controls["EV Charger"]["P Setpoint"] == 14.0
    assert not record["ev_panel_guard_active"]


def test_panel_guard_leaves_v2h_discharge_unchanged():
    guard = EVPanelGuard(38.0)
    controls, record = _apply(guard, "2025-12-03 19:07", 38.1, -5.0)
    assert controls["EV Charger"]["P Setpoint"] == -5.0
    assert not record["ev_panel_guard_tripped"]

    _apply(guard, "2025-12-03 19:08", 38.1, 11.5)
    controls, record = _apply(guard, "2025-12-03 19:09", 24.9, -5.0)
    assert controls["EV Charger"]["P Setpoint"] == -5.0
    assert record["ev_panel_guard_active"]


def test_panel_guard_requires_actual_total_home_meter_after_first_minute():
    guard = EVPanelGuard(38.0)
    with pytest.raises(ValueError, match="total-home power"):
        guard.apply(pd.Timestamp("2025-12-03 19:07"), {}, _command(11.5), ("EV Charger",))

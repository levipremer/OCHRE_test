"""Minute-level integration of MPC setpoint translation and real ASHP controls.

These tests need neither a forecast nor a solver.  They exercise the production
ASHP thermostat, resistance staging, compressor guard and defrost clock with a
fixed indoor temperature so the handoff itself determines the result.
"""

import datetime as dt
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.hvac import (
    HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY,
    HVACPredictiveController,
)
from ochre.Equipment.HVAC import ASHPHeater


def make_heater():
    heater = ASHPHeater.__new__(ASHPHeater)
    heater.use_ideal_capacity = False
    heater.use_staged_er_control = True
    heater.hvac_mult = 1
    heater.n_speeds = 1
    heater.speed_idx = 1
    heater.disable_speeds = np.array([False])
    heater.mode = "HP and ER On"
    heater.zone = SimpleNamespace(temperature=20.0)
    heater.envelope_model = None
    heater.is_heater = True
    heater.temp_setpoint = 21.0
    heater.prev_setpoint = 21.0
    heater.temp_indoor_prev = 20.0
    heater.temp_deadband = 1.5
    heater.deadband_offset = 0.5
    heater.time_res = dt.timedelta(minutes=1)
    heater.current_schedule = {
        "Ambient Dry Bulb (C)": 0.0,
        "HVAC Heating Setpoint (C)": 21.0,
        "Voltage (-)": 1.0,
    }
    heater.hp_lockout_temp = -17.78
    heater.compressor_min_on_time = dt.timedelta(minutes=15)
    heater.compressor_min_off_time = dt.timedelta(minutes=15)
    heater.compressor_time_in_state = dt.timedelta(minutes=30)
    heater.defrost_model = "Discrete"
    heater.defrost_control_type = "Timer"
    heater.defrost_finish_active_cycle = True
    heater.defrost_er_pause_requires_compressor = True
    heater.defrost_temp_threshold = 4.4445
    heater.defrost_remaining = dt.timedelta(minutes=3)
    heater.defrost_timer = dt.timedelta()
    heater.defrost_accumulation = dt.timedelta()
    heater.defrost_timer_interval = dt.timedelta(minutes=90)
    heater.defrost_duration = dt.timedelta(minutes=7)
    heater.defrost_cycle_count = 1
    heater.defrost_active_fraction = 1.0
    heater.defrost = True
    heater.defrost_er_strategy = "Aggressive"
    heater.er_capacity_list = [0, 10000, 20000]
    heater.er_defrost_capacity_list = heater.er_capacity_list.copy()
    heater.er_capacity_rated = 20000
    heater.er_ext_capacity = None
    heater.er_ext_capacity_frac = 1.0
    heater.er_stage_idx = 1
    heater.er_stage_timer = dt.timedelta(minutes=4)
    heater.er_stage_delay = dt.timedelta(minutes=10)
    heater.er_lockout_temp = 4.44
    heater.er_lockout_time = dt.timedelta()
    heater.er_hard_lockout_time = dt.timedelta()
    heater.er_soft_lockout_time = dt.timedelta()
    heater.er_setpoint_offset = 1.5
    heater.er_deadband_temp = 1.5
    return heater


def measured_status(heater, timestamp, er_power_kw=20.0):
    return {
        "Time": timestamp,
        HVAC_INDOOR_BOUNDARY_TEMPERATURE_KEY: heater.zone.temperature,
        "HVAC Heating Mode": heater.mode,
        "HVAC Heating ER Power (kW)": er_power_kw,
        "HVAC Heating Main Power (kW)": 2.0 if "HP" in heater.mode else 0.0,
        "HVAC Heating Defrost Active (-)": heater.defrost_active_fraction > 0,
        "HVAC Heating Defrost Active Fraction (-)": heater.defrost_active_fraction,
        "HVAC Heating Defrost Remaining (minutes)": (
            heater.defrost_remaining.total_seconds() / 60
        ),
    }


def advance_control(heater, command, timestamp):
    """Run real control and state transitions, omitting only thermal dynamics."""
    heater.mode = heater.update_external_control(command["HVAC Heating"])
    hp_running = "HP" in heater.mode
    fraction = heater._advance_discrete_defrost(hp_running, 0.1)
    heater.defrost_active_fraction = fraction
    heater.defrost = fraction > 0
    normal_er = heater.update_er_capacity(5000.0) if "ER" in heater.mode else 0.0
    er_w = ((1 - fraction) * normal_er
            + fraction * heater.get_defrost_backup_capacity())
    return measured_status(heater, timestamp, er_w / 1000)


class DefrostHandoffIntegrationTests(unittest.TestCase):
    def test_hp_only_request_waits_for_defrost_then_clears_normal_er(self):
        heater = make_heater()
        controller = HVACPredictiveController()
        controller.bind_heating_equipment(heater)
        start = pd.Timestamp("2025-12-02 19:48", tz="America/Indiana/Indianapolis")
        status = measured_status(heater, start - pd.Timedelta(minutes=1))
        controller._start_setpoint_interval(
            status, planned_heat_w=5000, planned_hp_heat_w=5000,
            planned_er_heat_w=0, training_setpoint_c=20.5,
            predicted_next_temperature_c=20.0,
        )

        for minute in range(3):
            command = controller._setpoint_control(status)
            self.assertEqual(command[controller.heating_name]["Setpoint"], 21.0)
            status = advance_control(heater, command, start + pd.Timedelta(minutes=minute))
            self.assertIn("HP", heater.mode)
            self.assertEqual(status["HVAC Heating ER Power (kW)"], 20.0)
            self.assertEqual(heater.er_stage_idx, 1)
            self.assertFalse(controller._hp_only_er_reset_attempted)

        # The last completed sample includes defrost but its boundary timer is
        # zero.  Normal control can respond in the next minute, without a lag.
        self.assertEqual(status["HVAC Heating Defrost Active Fraction (-)"], 1.0)
        self.assertEqual(status["HVAC Heating Defrost Remaining (minutes)"], 0.0)
        command = controller._setpoint_control(status)
        self.assertEqual(command[controller.heating_name]["Setpoint"], 20.0)
        self.assertTrue(controller._hp_only_er_off_pending)
        status = advance_control(heater, command, start + pd.Timedelta(minutes=3))
        self.assertEqual(heater.mode, "HP On")
        self.assertEqual(heater.er_stage_idx, 0)
        self.assertEqual(status["HVAC Heating ER Power (kW)"], 0.0)
        self.assertEqual(status["HVAC Heating Defrost Active Fraction (-)"], 0.0)

        command = controller._setpoint_control(status)
        self.assertEqual(command[controller.heating_name]["Setpoint"], 21.0)
        self.assertFalse(controller._hp_only_er_off_pending)
        status = advance_control(heater, command, start + pd.Timedelta(minutes=4))
        self.assertEqual(heater.mode, "HP On")
        self.assertEqual(status["HVAC Heating ER Power (kW)"], 0.0)

    def test_off_plan_finishes_active_defrost_before_stopping_compressor(self):
        heater = make_heater()
        controller = HVACPredictiveController()
        controller.bind_heating_equipment(heater)
        start = pd.Timestamp("2025-12-02 19:48", tz="America/Indiana/Indianapolis")
        status = measured_status(heater, start - pd.Timedelta(minutes=1))
        controller._start_setpoint_interval(
            status, planned_heat_w=0, planned_hp_heat_w=0,
            planned_er_heat_w=0, training_setpoint_c=20.5,
            predicted_next_temperature_c=20.0,
        )
        for minute in range(3):
            command = controller._setpoint_control(status)
            self.assertLess(command[controller.heating_name]["Setpoint"], 20.0)
            status = advance_control(heater, command, start + pd.Timedelta(minutes=minute))
            self.assertIn("HP", heater.mode)
            self.assertEqual(status["HVAC Heating ER Power (kW)"], 20.0)
        self.assertEqual(heater.defrost_remaining, dt.timedelta())
        status = advance_control(
            heater, controller._setpoint_control(status), start + pd.Timedelta(minutes=3)
        )
        self.assertEqual(heater.mode, "Off")
        self.assertEqual(status["HVAC Heating ER Power (kW)"], 0.0)


if __name__ == "__main__":
    unittest.main()

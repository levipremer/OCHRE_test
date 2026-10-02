"""Thermostat checks for the heat-pump-only water-heater mode."""

import unittest
from types import SimpleNamespace

import numpy as np

from ochre.Equipment import HeatPumpWaterHeater


class HeatPumpOnlyCutoffTests(unittest.TestCase):
    def setUp(self):
        self.heater = SimpleNamespace(
            hp_only_mode=True,
            er_only_mode=False,
            mode="Heat Pump On",
            setpoint_temp=51.0,
            deadband_temp=2.75,
            max_temp=60.0,
            t_upper_idx=2,
            t_lower_idx=9,
            model=SimpleNamespace(states=np.zeros(12)),
        )

    def thermostat(self, upper, lower):
        self.heater.model.states[2] = upper
        self.heater.model.states[9] = lower
        return HeatPumpWaterHeater.run_thermostat_control(self.heater)

    def test_upper_one_degree_above_setpoint_does_not_stop_hp_only(self):
        self.assertIsNone(self.thermostat(52.0, 45.0))

    def test_upper_at_maximum_stops_even_when_control_temperature_is_low(self):
        self.assertEqual(self.thermostat(60.0, 0.0), "Off")

    def test_weighted_control_temperature_still_stops_at_setpoint(self):
        self.assertEqual(self.thermostat(51.0, 51.0), "Off")

    def test_hybrid_retains_one_degree_upper_cutoff(self):
        self.heater.hp_only_mode = False
        self.assertEqual(self.thermostat(52.0, 45.0), "Off")


if __name__ == "__main__":
    unittest.main()

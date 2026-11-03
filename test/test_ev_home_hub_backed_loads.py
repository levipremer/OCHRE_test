"""V2H may serve the eligible Hub circuits, including fixed Hub-other power."""

from __future__ import annotations

import unittest

from examples.single_home_hems.single_home_hems.backed_loads import (
    HOME_HUB_BACKED_LOADS,
    backed_power,
)


class HomeHubBackedLoadsTests(unittest.TestCase):
    def test_cap_includes_hub_other_but_excludes_resistance_and_outside(self):
        plant = {
            "HVAC Heating Main Power (kW)": 1.5,
            "HVAC Heating Fan Power (kW)": 0.2,
            "HVAC Cooling Main Power (kW)": 0.0,
            "HVAC Cooling Fan Power (kW)": 0.0,
            "HVAC Heating ER Power (kW)": 3.0,
            "Water Heating Electric Power (kW)": 0.4,
            "DC House Hub Other Electric Power (kW)": 2.1,
            "DC House Outside Hub Electric Power (kW)": 9.0,
        }
        self.assertEqual(HOME_HUB_BACKED_LOADS,
                         ("heat_pump", "water_heater", "DC House Hub Other"))
        self.assertAlmostEqual(backed_power(plant, HOME_HUB_BACKED_LOADS), 4.2)


if __name__ == "__main__":
    unittest.main()

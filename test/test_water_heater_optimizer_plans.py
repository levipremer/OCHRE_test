"""Checks for water-heater open-loop optimizer-plan diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.water_heater import (
    WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems import plotting
from examples.single_home_hems.single_home_hems.plotting import (
    plot_water_heater_optimization_plans,
)


class WaterHeaterOptimizerPlanTests(unittest.TestCase):
    def test_plan_contains_temperatures_setpoint_power_price_and_draw(self) -> None:
        controller = WaterHeaterPredictiveController(
            model_kind="state-space",
            nominal_hp_capacity_kw=4.0,
            model_timestep=pd.Timedelta(minutes=15),
            cop=4.0,
            maximum_setpoint_c=55.0,
        )
        model = {
            "A": np.eye(2),
            "B": np.ones((2, 1)),
            "E": np.zeros((2, 1)),
            "output_matrix": np.eye(2),
            "output_names": (
                "water_heater_sensor_node_03_temperature_c",
                "water_heater_sensor_node_10_temperature_c",
            ),
            "disturbance_names": ("water_heater_draw_l_per_min",),
        }
        controller._record_optimization_plan(
            pd.Timestamp("2018-01-01 06:00"),
            np.array([48.0, 42.0]),
            np.array([0.0, 4.0]),
            np.array([0.15, 0.30]),
            np.array([[0.25, 1.0]]),
            model,
        )

        plan = controller.optimization_plans[0]
        self.assertEqual(len(plan), 3)
        np.testing.assert_allclose(
            plan["water_heater_sensor_node_03_temperature_c"],
            [48.0, 48.0, 52.0],
        )
        np.testing.assert_allclose(
            plan["water_heater_setpoint_temperature_c"],
            [45.0, 55.0, 55.0],
        )
        np.testing.assert_allclose(
            plan["water_heater_electric_power_kw"], [0.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            plan["hot_water_flow_rate_l_per_min"], [0.25, 1.0, 1.0]
        )

    def test_each_plan_is_saved_in_the_optimizer_plan_folder(self) -> None:
        index = pd.date_range("2018-01-01 06:00", periods=3, freq="15min")
        plan = pd.DataFrame(
            {
                "water_heater_sensor_node_03_temperature_c": [50, 49, 48],
                "water_heater_sensor_node_10_temperature_c": [45, 44, 43],
                "water_heater_setpoint_temperature_c": [45, 55, 55],
                "water_heater_electric_power_kw": [0, 0.5, 0.5],
                "electricity_price_usd_per_kwh": [0.15, 0.15, 0.30],
                "hot_water_flow_rate_l_per_min": [0, 1.0, 1.0],
            },
            index=index,
        )
        plan.attrs["solved_at"] = index[0]

        for scenario_count in (None, 3):
            with self.subTest(scenario_count=scenario_count), tempfile.TemporaryDirectory() as temporary_directory:
                if scenario_count is not None:
                    plan.attrs["scenario_count"] = scenario_count
                    # Percentile columns must not be mistaken for extra sensors.
                    plan["water_heater_sensor_node_03_temperature_c_p10"] = [48, 47, 46]
                output_directory = Path(temporary_directory) / "optimizer_plans"
                with patch.object(plotting.plt, "close", wraps=plotting.plt.close) as close:
                    files = plot_water_heater_optimization_plans(
                        (plan,), output_directory, dpi=50
                    )
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].is_file())
                self.assertEqual(files[0].name, "water_heater_plan_20180101_060000.png")
                figure = close.call_args.args[0]
                title = figure.axes[0].get_title()
                labels = [line.get_label() for axis in figure.axes for line in axis.lines]
                if scenario_count is None:
                    self.assertEqual(title, "2018-01-01 06:00")
                    self.assertIn("Tank setpoint", labels)
                    self.assertIn("Water-heater power", labels)
                else:
                    self.assertIn("scenario means (3 paths)", title)
                    self.assertIn("Mean scenario setpoint", labels)
                    self.assertIn("Mean planned power", labels)


if __name__ == "__main__":
    unittest.main()

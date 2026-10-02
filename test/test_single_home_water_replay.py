"""Exercise measured water overrides through the actual OCHRE plant boundary."""

from __future__ import annotations

import datetime as dt
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ochre import Dwelling
from examples.five_home_block.five_home_block.config import load_configuration
from examples.five_home_block.five_home_block.home_factory import create_dwelling
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.controllers import WaterHeaterPredictiveController
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation, select_home


REPO_ROOT = Path(__file__).resolve().parents[1]
DRAW_COLUMNS = ("Water Heating (L/min)", "Clothes Washer (L/min)", "Dishwasher (L/min)")


class WaterReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_configuration(REPO_ROOT / "examples/five_home_block/config", repo_root=REPO_ROOT)
        cls.home = replace(select_home(cls.config, "home_01"), evs=())

    def test_actual_plant_replays_identical_draws_and_full_forecast_for_both_runs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary)
            source = pd.Timestamp("2025-10-09 06:00", tz="America/New_York")
            # No measured water exists after the last plant minute. The real
            # scenario provider must still supply the entire 12-hour horizon.
            index = pd.date_range(pd.Timestamp("2025-10-01", tz=source.tz), source + pd.Timedelta(minutes=1), freq="min")
            flow = pd.Series(.1, index=index, name="water_flow_l_per_min")
            flow.loc[source] = 1.25
            flow.loc[source + pd.Timedelta(minutes=1)] = 2.5
            path = output / "water.csv"
            flow.rename_axis("timestamp").to_csv(path)
            config = replace(
                self.config, start_time=dt.datetime(2018, 1, 1, 6),
                duration=dt.timedelta(minutes=2), time_step=dt.timedelta(minutes=1),
                initialization_time=None, output_directory=output,
            )
            water_controller = WaterHeaterPredictiveController(
                forecast_type="scenario", scenario_method="daily_budget_high",
            )
            simulation = SingleHomeHEMSSimulation(
                config, self.home, HEMSController(device_controllers=(water_controller,)),
                water_draw_file=path, water_draw_start=source,
            )
            observed = []
            forecasts = []
            original_update_inputs = Dwelling.update_inputs

            def observe_inputs(dwelling, schedule_inputs=None):
                original_update_inputs(dwelling, schedule_inputs)
                tank = dwelling.get_equipment_by_end_use("Water Heating").model
                observed.append([tank.current_schedule[column] for column in DRAW_COLUMNS])

            def observe_controller(timestamp, previous_status, current_schedule, forecast):
                if not forecasts:
                    scenarios = forecast.water_scenarios.scenarios_at(timestamp)
                    self.assertEqual(scenarios.values.shape, (8, 48))
                    self.assertTrue(forecast.data.iloc[2:][DRAW_COLUMNS[0]].isna().all())
                    # Fabricating future water rows must not affect the causal
                    # provider, which uses only the historical source paths.
                    forecast.data.loc[timestamp:, DRAW_COLUMNS[0]] = 999.0
                    repeated = forecast.water_scenarios.scenarios_at(timestamp)
                    np.testing.assert_array_equal(scenarios.values, repeated.values)
                    np.testing.assert_allclose(scenarios.values, .1)
                    forecasts.append(forecast)
                return {}

            with patch.object(Dwelling, "update_inputs", autospec=True, side_effect=observe_inputs), \
                patch.object(water_controller, "controls_for_step", side_effect=observe_controller), \
                patch("examples.single_home_hems.single_home_hems.simulation.create_dwelling", wraps=create_dwelling) as factory:
                controlled = simulation._run_stepwise(apply_controls=True, raw_directory=output / "controlled")
                baseline = simulation._run_stepwise(apply_controls=False, raw_directory=output / "baseline")

            np.testing.assert_allclose(observed, [[1.25, 0, 0], [2.5, 0, 0]] * 2)
            for call in factory.call_args_list:
                self.assertEqual(call.args[0].duration, config.duration + dt.timedelta(hours=12))
            for result in (controlled, baseline):
                self.assertEqual(len(result[0]), 2)
                self.assertEqual(len(result[2]), 722)
                np.testing.assert_allclose(result[2].iloc[:2][DRAW_COLUMNS[0]], [1.25, 2.5])
                self.assertTrue(result[2].iloc[2:][DRAW_COLUMNS[0]].isna().all())
            # A no-op controller must produce the same thermal transition as baseline.
            pd.testing.assert_frame_equal(controlled[4], baseline[4])
            self.assertEqual(forecasts[0].water_scenarios.metadata["initial_complete_candidate_count"], 8)


if __name__ == "__main__":
    unittest.main()

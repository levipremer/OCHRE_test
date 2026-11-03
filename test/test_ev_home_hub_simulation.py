"""The split replay and live Hub cap work through the single-home step loop."""

from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.five_home_block.five_home_block.output_mapping import standardize_results
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.controllers.ev_home_hub import EVHomeHubController
from examples.single_home_hems.single_home_hems.ev_hub_forecast import EVHubIntervalForecast
from examples.single_home_hems.single_home_hems.forecast import PRICE_COLUMN
from examples.single_home_hems.single_home_hems.backed_loads import (
    HOME_HUB_BACKED_LOADS,
    backed_power,
)
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation
from examples.single_home_hems.single_home_hems.home_hub_observation import (
    project_ev_only_hub_observation,
)
from test.test_ev_v2h import schedule, vehicle


ROOT = Path(__file__).resolve().parents[1]


class DischargeEveryMinute:
    def __init__(self, equipment_name):
        self.equipment_name = equipment_name
        self.completed = []
        self.history_at_decision = []

    def observe_plant_step(self, timestamp, status):
        self.completed.append((timestamp + pd.Timedelta(minutes=1), status))

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        self.history_at_decision.append((timestamp, [time for time, _ in self.completed]))
        return {self.equipment_name: {"P Setpoint": -4.0, "Backed Load (kW)": 0.0}}


class RecordingHubProvider:
    def __init__(self):
        self.histories = []

    def issue(self, timestamp, completed_history, horizon, inverter_kw):
        self.histories.append(completed_history.copy())
        rows = pd.DataFrame({
            "eligible_load_mean_kw": 1.0,
            "eligible_mean_kw": 1.0,
            "eligible_min_kw": 1.0,
            "hub_mean_kw": 2.0,
        }, index=pd.date_range(timestamp, periods=int(horizon / pd.Timedelta(minutes=15)), freq="15min"))
        return EVHubIntervalForecast(timestamp, timestamp, rows, "observability_test")


class RecordingEVHomeHubController(EVHomeHubController):
    def __init__(self, ev, provider):
        super().__init__(ev, provider, outside_hub_forecast=object())
        self.observed = []
        self.controller_inputs = []

    def observe_plant_step(self, timestamp, status):
        self.observed.append((pd.Timestamp(timestamp), dict(status)))
        super().observe_plant_step(timestamp, status)

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        self.controller_inputs.append((
            pd.Timestamp(timestamp), dict(previous_status or {}),
            dict(current_schedule), tuple(forecast.data.columns),
        ))
        return super().controls_for_step(timestamp, previous_status, current_schedule, forecast)


class HomeHubSimulationTests(unittest.TestCase):
    def test_live_hub_meter_excludes_nonzero_resistance(self):
        plant = {
            "HVAC Heating Main Power (kW)": 1.0,
            "HVAC Heating Fan Power (kW)": 0.2,
            "HVAC Heating ER Power (kW)": 5.0,
            "HVAC Cooling Electric Power (kW)": 0.3,
            "Water Heating Electric Power (kW)": 0.4,
            "DC House Hub Other Electric Power (kW)": 0.5,
            "EV_01A Electric Power (kW)": -1.0,
            "EV_01A SOC (-)": 0.5,
            "EV_01A End SOC (-)": 0.5,
            "EV_01A SOH (-)": 1.0,
            "EV_01A Capacity (kWh)": 80.0,
            "EV_01A Parked": True,
        }
        observed = project_ev_only_hub_observation(plant, "EV_01A")
        self.assertEqual(set(observed), {
            "Home Hub Electric Power (kW)",
            "EV_01A Electric Power (kW)", "EV_01A SOC (-)",
            "EV_01A End SOC (-)", "EV_01A SOH (-)",
            "EV_01A Capacity (kWh)", "EV_01A Parked",
        })
        self.assertAlmostEqual(observed["Home Hub Electric Power (kW)"], 1.4)

    def test_ev_only_controller_receives_only_hub_aggregate_and_ev_meter(self):
        base = load_configuration(ROOT / "examples/five_home_block/config", repo_root=ROOT)
        ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=8,
                     control_minutes=15)
        home = replace(base.homes[0], evs=(ev,))
        start = dt.datetime(2018, 1, 1, 9, 0)
        split = pd.DataFrame({
            "hub_other_power_kw": 2.0,
            "outside_hub_power_kw": 20.0,
        }, index=pd.date_range(pd.Timestamp(start) - pd.Timedelta(minutes=1),
                               periods=3, freq="min"))
        provider = RecordingHubProvider()
        device = RecordingEVHomeHubController(ev, provider)

        def choose_discharge(timestamp, soc, forecast):
            device.held_mode = -1
            device.fallback = False
            device.next_solve = timestamp + pd.Timedelta(minutes=15)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = replace(base, start_time=start, duration=dt.timedelta(minutes=2),
                             time_step=dt.timedelta(minutes=1), initialization_time=dt.timedelta(0),
                             output_directory=Path(directory), homes=(home,), verbosity=7)
            simulation = SingleHomeHEMSSimulation(
                config, home, HEMSController(device_controllers=(device,)),
                hub_split_replay_kw=split, hvac_warmup_duration=pd.Timedelta(minutes=1),
                ev_mobility_schedules={ev.vehicle_id: schedule()},
                water_heater_hp_only=True,
            )
            with patch.object(device, "_solve", side_effect=choose_discharge):
                raw, _, _, _, _, _ = simulation._run_stepwise(
                    apply_controls=True, raw_directory=Path(directory) / "raw",
                )

        ev_prefix = f"EV_{ev.vehicle_id}"
        allowed = {"Home Hub Electric Power (kW)"} | {
            f"{ev_prefix} {suffix}" for suffix in (
                "Electric Power (kW)", "SOC (-)", "End SOC (-)", "SOH (-)",
                "Capacity (kWh)", "Parked",
            )
        }
        self.assertEqual(len(device.observed), 3)  # Warm-up and both evaluated minutes.
        self.assertEqual(len(device.controller_inputs), 2)
        self.assertEqual(len(provider.histories), 1)
        for _, status in device.observed:
            self.assertEqual(set(status), allowed)
        for _, status, current_schedule, forecast_columns in device.controller_inputs:
            self.assertEqual(set(status), allowed)
            self.assertEqual(set(current_schedule), {PRICE_COLUMN})
            self.assertEqual(forecast_columns, (PRICE_COLUMN,))

        issuance_history = provider.histories[0]
        self.assertEqual(issuance_history.index.tolist(), [pd.Timestamp(start)])
        self.assertEqual(set(issuance_history.columns), allowed)
        self.assertEqual(device.controller_inputs[0][1], device.observed[0][1])
        self.assertEqual(device.controller_inputs[1][1], device.observed[1][1])

        first_result = raw.loc[pd.Timestamp(start)]
        first_hub_observation = device.observed[1][1]
        mapped = standardize_results(raw, home.home_id, resistance_outside_hub=True).data
        np.testing.assert_allclose(
            mapped["total_electric_power_kw"],
            mapped["home_hub_non_ev_power_kw"] + mapped["ev_power_kw"]
            + mapped["outside_hub_with_resistance_power_kw"],
            rtol=0, atol=1e-7,
        )
        np.testing.assert_allclose(
            mapped["outside_hub_with_resistance_power_kw"],
            mapped["outside_hub_power_kw"] + mapped["hvac_resistance_power_kw"],
            rtol=0, atol=1e-7,
        )
        self.assertLess(first_result[f"{ev_prefix} Electric Power (kW)"], 0.0)
        self.assertAlmostEqual(
            first_hub_observation["Home Hub Electric Power (kW)"],
            first_result["Total Electric Power (kW)"]
            - first_result["DC House Outside Hub Electric Power (kW)"]
            - first_result["HVAC Heating ER Power (kW)"],
            places=6,
        )
        self.assertAlmostEqual(
            first_hub_observation[f"{ev_prefix} Electric Power (kW)"],
            first_result[f"{ev_prefix} Electric Power (kW)"],
            places=6,
        )

    def test_completed_warmup_is_visible_and_live_cap_excludes_outside(self):
        base = load_configuration(ROOT / "examples/five_home_block/config", repo_root=ROOT)
        ev = vehicle()
        home = replace(base.homes[0], evs=(ev,))
        start = dt.datetime(2018, 1, 1, 8, 59)
        replay_start = pd.Timestamp(start) - pd.Timedelta(minutes=1)
        split = pd.DataFrame({
            "hub_other_power_kw": [2.0, 2.0, 2.0],
            "outside_hub_power_kw": [20.0, 20.0, 20.0],
        }, index=pd.date_range(replay_start, periods=3, freq="min"))
        device = DischargeEveryMinute(f"EV_{ev.vehicle_id}")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = replace(base, start_time=start, duration=dt.timedelta(minutes=2),
                             time_step=dt.timedelta(minutes=1), initialization_time=dt.timedelta(0),
                             output_directory=Path(directory), homes=(home,), verbosity=7)
            simulation = SingleHomeHEMSSimulation(
                config, home, HEMSController(device_controllers=(device,)),
                hub_split_replay_kw=split, hvac_warmup_duration=pd.Timedelta(minutes=1),
                ev_mobility_schedules={ev.vehicle_id: schedule()},
                water_heater_hp_only=True,
            )
            raw, controls, _, _, _, hvac_training = simulation._run_stepwise(
                apply_controls=True, raw_directory=Path(directory) / "raw",
            )
            baseline_raw, baseline_controls, _, _, _, baseline_hvac_training = (
                simulation._run_stepwise(
                    apply_controls=False, raw_directory=Path(directory) / "baseline",
                )
            )

        self.assertEqual(len(device.history_at_decision), 2)
        first_time, first_completed = device.history_at_decision[0]
        self.assertEqual(first_time, pd.Timestamp(start))
        self.assertEqual(first_completed, [pd.Timestamp(start)])
        self.assertEqual(len(controls), 2)
        self.assertEqual(tuple(hvac_training["heating_setpoint_c"]), (22.5, 18.0))
        self.assertEqual(tuple(baseline_hvac_training["heating_setpoint_c"]), (22.5, 18.0))
        for recorded in (controls, baseline_controls):
            self.assertEqual(
                tuple(json.loads(command)["HVAC Heating"]["Setpoint"]
                      for command in recorded["control_signal_json"]),
                (22.5, 18.0),
            )
        self.assertEqual(tuple(raw["HVAC Heating Setpoint (C)"]), (22.5, 18.0))
        self.assertEqual(tuple(baseline_raw["HVAC Heating Setpoint (C)"]), (22.5, 18.0))
        for column in ("HVAC Heating Electric Power (kW)", "Water Heating Electric Power (kW)",
                       "Temperature - Indoor (C)"):
            np.testing.assert_allclose(raw[column], baseline_raw[column], atol=1e-12)
        actual_cap = backed_power(raw, HOME_HUB_BACKED_LOADS)
        discharge = -raw[f"EV_{ev.vehicle_id} Electric Power (kW)"].to_numpy(dtype=float)
        self.assertTrue(np.all(discharge >= 0))
        self.assertTrue(np.all(discharge <= actual_cap.to_numpy(dtype=float) + 1e-7))
        self.assertTrue(np.any(discharge > 0.5))
        self.assertTrue(np.all(discharge < split.loc[raw.index, "outside_hub_power_kw"].to_numpy()))


if __name__ == "__main__":
    unittest.main()

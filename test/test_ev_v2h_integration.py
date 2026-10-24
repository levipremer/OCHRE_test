"""Small real-OCHRE smoke runs for both EV sources/formulations."""
from contextlib import redirect_stdout
from dataclasses import replace
import datetime as dt
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation


class EVIntegrationTests(unittest.TestCase):
    def test_ev_and_hpwh_mpc_have_matched_thermal_forecasts(self):
        from examples.single_home_hems.single_home_hems.controllers.water_heater import (
            WaterHeaterPredictiveController, DEFAULT_RC_MODEL_PATH,
        )
        ev = vehicle(horizon_hours=0.5)
        home = replace(CONFIG.homes[0], evs=(ev,))
        water = WaterHeaterPredictiveController(model_kind="rc", model_path=DEFAULT_RC_MODEL_PATH,
            optimization_kind="convex", solver="CLARABEL", horizon=dt.timedelta(minutes=30))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,), duration=dt.timedelta(minutes=30), output_directory=Path(directory))
            result = SingleHomeHEMSSimulation(config, home, HEMSController(device_controllers=(water,)),
                ev_mobility_schedules={ev.vehicle_id: schedule()}, water_heater_hp_only=True).run()
        self.assertEqual(result.metrics.failed_solves.sum(), 0)
        np.testing.assert_allclose(result.standardized_results.water_heating_power_kw,
                                   result.uncontrolled_results.water_heating_power_kw)

    def test_neighborhood_opt_in_preserves_aggregation_and_weather_contract(self):
        from examples.five_home_block.five_home_block.simulation import NeighborhoodSimulation
        ev = vehicle(horizon_hours=0.25, forecast_margin_hours=0.25)
        home = replace(CONFIG.homes[0], evs=(ev,))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,)+CONFIG.homes[1:], duration=dt.timedelta(minutes=15), output_directory=Path(directory))
            result = NeighborhoodSimulation(config).run_independent()
        self.assertEqual(len(result.home_results[home.home_id]), 15)
        self.assertEqual(len(result.home_results), 5)
        self.assertEqual(len(result.aggregate_results), 15)

    def test_two_independent_controllers_and_single_v2h(self):
        first = replace(vehicle(horizon_hours=0.5), initial_soc_fraction=0.8)
        second = replace(vehicle(formulation="milp", solver="SCIPY", v2h=False, horizon_hours=0.5),
                         vehicle_id="second", initial_soc_fraction=0.1)
        mobility = {first.vehicle_id: schedule(), second.vehicle_id: schedule([(20, 45, 5)])}
        home = replace(CONFIG.homes[0], evs=(first, second))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,), duration=dt.timedelta(minutes=30), output_directory=Path(directory))
            result = SingleHomeHEMSSimulation(config, home, HEMSController(), ev_mobility_schedules=mobility).run()
        self.assertTrue(result.raw_results["EV_second Electric Power (kW)"].ge(0).all())
        self.assertTrue(result.raw_results[f"EV_{first.vehicle_id} Electric Power (kW)"].lt(0).any())
        self.assertEqual(result.metrics.failed_solves.sum(), 0)

    def test_opted_in_baseline_does_not_require_solver_or_discharge(self):
        ev = vehicle(controller="baseline", solver="GUROBI", horizon_hours=0.5)
        home = replace(CONFIG.homes[0], evs=(ev,))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,), duration=dt.timedelta(minutes=10), output_directory=Path(directory))
            result = SingleHomeHEMSSimulation(config, home, HEMSController(),
                        ev_mobility_schedules={ev.vehicle_id: schedule()}).run()
        self.assertEqual(result.metrics.solve_attempts.sum(), 0)
        self.assertTrue(result.raw_results[f"EV_{ev.vehicle_id} Electric Power (kW)"].ge(0).all())
        np.testing.assert_allclose(result.standardized_results, result.uncontrolled_results)

    def test_matched_runs_and_energy_conservation(self):
        for source, formulation, solver in (("audi", "convex", "CLARABEL"), ("nhts", "milp", "SCIPY")):
            with self.subTest(source=source), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                ev = vehicle(source=source, formulation=formulation, solver=solver, seed=42, initial_soh=.85,
                             horizon_hours=0.5, forecast_margin_hours=0.5, history_days=1,
                             source_file="examples/five_home_block/inputs/ev/audi_forecasting_15min_20vehicle.parquet" if source == "audi" else None)
                home = replace(CONFIG.homes[0], evs=(ev,))
                config = replace(CONFIG, duration=dt.timedelta(minutes=30), homes=(home,),
                                 output_directory=Path(directory))
                with redirect_stdout(io.StringIO()):
                    result = SingleHomeHEMSSimulation(config, home, HEMSController()).run()
                raw = result.raw_results
                name = f"EV_{ev.vehicle_id}"
                power = raw[f"{name} Electric Power (kW)"]
                final = raw[f"{name} End Stored Energy (kWh)"].iloc[-1]
                change = (power.clip(lower=0).sum()*ev.hems.charge_efficiency +
                          power.clip(upper=0).sum()/ev.hems.discharge_efficiency)/60
                trip = raw[f"{name} Trip Energy (kWh)"].sum()
                fade = raw[f"{name} Fade Energy Loss (kWh)"].sum()
                self.assertAlmostEqual(final, ev.initial_soc_fraction*ev.capacity_kwh*ev.hems.initial_soh+change-trip-fade, places=10)
                controlled = result.metrics.loc[result.metrics.scenario.eq("controlled")].iloc[0]
                self.assertEqual(controlled.failed_solves, 0)
                self.assertEqual(controlled.solve_attempts, 2)
                self.assertGreater(controlled.mean_solve_seconds, 0)
                self.assertTrue(raw[f"{name} End SOC (-)"].between(0, 1).all())
                backed = result.standardized_results.hvac_power_kw+result.standardized_results.water_heating_power_kw
                self.assertTrue(np.all(-power <= backed+1e-7))
                self.assertEqual(len(result.standardized_results), 30)


if __name__ == "__main__":
    unittest.main()

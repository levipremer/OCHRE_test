"""Controller-active pre-roll, boundary accounting and evaluation-only outputs."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule
from examples.single_home_hems.single_home_hems.control_config import load_unified, UnifiedConfiguration, resolve_control
from examples.single_home_hems.single_home_hems import unified as runner
from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics


class WarmupTests(unittest.TestCase):
    def load(self):
        directory = ROOT/"examples/single_home_hems/config"
        return load_unified(directory/"unified_home.json", directory/"unified_control.json", ROOT)

    def test_default_zero_and_invalid_windows(self):
        config = self.load()
        self.assertEqual(config.warmup_hours, 24)
        self.assertEqual(config.simulation_window().start_time, config.neighborhood.start_time-pd.Timedelta(days=1))
        self.assertEqual(config.cooldown_hours, 24)
        self.assertEqual(config.simulation_window().duration, config.neighborhood.duration+pd.Timedelta(hours=48))
        self.assertEqual(replace(config, warmup_hours=0, cooldown_hours=0).simulation_window(), config.neighborhood)
        for value in (-1, float("nan"), float("inf"), True, "24", .1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "warmup_hours"):
                replace(config, warmup_hours=value).simulation_window()
            with self.subTest(cooldown=value), self.assertRaisesRegex(ValueError, "cooldown_hours"):
                replace(config, cooldown_hours=value).simulation_window()
        home_doc = config.home_document.copy()
        home_doc["simulation"] = dict(home_doc["simulation"])
        home_doc["simulation"].pop("warmup_hours")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"home.json"
            path.write_text(json.dumps(home_doc), encoding="utf-8")
            restored = load_unified(path, ROOT/"examples/single_home_hems/config/unified_control.json", ROOT)
        self.assertEqual(restored.warmup_hours, 24)

    def test_short_real_run_keeps_state_and_excludes_warmup_and_tail(self):
        ev = replace(vehicle(horizon_hours=.5, comfort_soc_min=.9), initial_soc_fraction=.2)
        home = replace(CONFIG.homes[0], evs=(ev,))
        doc = {"defaults": {"ev": {"controller": "mpc", "horizon_hours": .5},
                            "hpwh": {"controller": "mpc", "horizon_hours": .5, "model": "rc"}}}
        controls = {d: resolve_control(doc, home.home_id, d) for d in ("hpwh", "hvac")}
        controls["evs"] = {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id)}
        evaluation = pd.Timestamp("2018-01-01")  # explicitly exercise annual weather/schedule wrapping
        mobility = replace(schedule(), start=evaluation-pd.Timedelta(hours=1))
        controllers = []
        original_factory = runner.make_controllers

        def capture(*args, **kwargs):
            devices, providers = original_factory(*args, **kwargs)
            controllers.extend(devices)
            return devices, providers

        with tempfile.TemporaryDirectory(dir=ROOT) as temp, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,), start_time=evaluation.to_pydatetime(),
                duration=pd.Timedelta(minutes=16), output_directory=Path(temp))
            unified = UnifiedConfiguration(config, {home.home_id: controls},
                {home.home_id: {"backed_loads": ["heat_pump", "water_heater"], "water_heater_hp_only": True}},
                {}, doc, warmup_hours=.25, cooldown_hours=.25)
            with patch.object(runner, "make_controllers", side_effect=capture):
                result = runner.run_home(unified, home, {ev.vehicle_id: mobility})
            for case in runner.CASES:
                data = result.cases[case]
                raw = data["raw"]
                self.assertEqual(len(raw), 16)
                self.assertEqual(raw.index[0], evaluation)
                prefix = Path(temp)/home.home_id/case/"data"
                mask = pd.read_csv(prefix/"evaluation_mask.csv", index_col=0, parse_dates=True)
                self.assertEqual(mask.in_comparison_window.sum(), 16)
                self.assertEqual((mask.index < evaluation).sum(), 15)
                self.assertEqual((mask.index >= evaluation+pd.Timedelta(minutes=16)).sum(),
                                 45 if case == "non_v2h_twin" else 15)
                self.assertEqual(mask.phase.eq("cooldown").sum(), 15)
                self.assertEqual(mask.phase.eq("warmup").sum(), 15)
                full = pd.read_csv(prefix/"raw_results_all_phases.csv", index_col=0)
                self.assertEqual(len(full), len(mask))
                row = result.metrics.loc[result.metrics.case.eq(case) & result.metrics.device.eq(f"EV_{ev.vehicle_id}")].iloc[0]
                self.assertAlmostEqual(row.initial_soc, raw[f"EV_{ev.vehicle_id} SOC (-)"].iloc[0])
                self.assertEqual(row.configured_initial_soc, .2)
                self.assertAlmostEqual(row.reference_degradation_cost_usd,
                    raw[f"EV_{ev.vehicle_id} Reference Degradation Cost ($)"].sum())
                self.assertAlmostEqual(row.soh_loss,
                    (1-ev.hems.calibration_end_of_life_soh)*row.damage_increment)
                self.assertAlmostEqual(row.final_stored_energy_kwh,
                    raw[f"EV_{ev.vehicle_id} End Stored Energy (kWh)"].iloc[-1])
                self.assertLessEqual(row.reference_degradation_cost_usd,
                    full[f"EV_{ev.vehicle_id} Reference Degradation Cost ($)"].sum())
                self.assertTrue((prefix/"ev_battery_checkpoints.json").exists())
                self.assertAlmostEqual(row.net_energy_kwh, raw[f"EV_{ev.vehicle_id} Electric Power (kW)"].sum()/60)
                for device, column in (("whole_home", "total_electric_power_kw"),
                                       ("hpwh", "water_heating_power_kw"), ("hvac", "hvac_power_kw")):
                    metric = result.metrics.loc[result.metrics.case.eq(case) & result.metrics.device.eq(device)].iloc[0]
                    power = data["mapped"][column]
                    self.assertAlmostEqual(metric.energy_kwh, power.sum()/60)
                    self.assertAlmostEqual(metric.cost_usd, (power*data["commands"].electricity_price_usd_per_kwh).sum()/60)
                if case == "all_baseline":
                    self.assertGreater(row.initial_soc, .2)
                    self.assertEqual(row.cycle_count, 0)  # charging continued across the boundary
                else:
                    for name in (f"EV_{ev.vehicle_id}", "Water Heating"):
                        history = pd.read_csv(prefix/f"{name.replace(' ', '_')}_solve_history.csv")
                        self.assertEqual(history.in_comparison_window.sum(), 2)
                        self.assertGreater(len(history), 2)
                        scored = result.solver_stats.loc[result.solver_stats.case.eq(case) & result.solver_stats.device.eq(name)].iloc[0]
                        self.assertEqual(scored.attempts, 2)
                        self.assertAlmostEqual(scored.solve_seconds_per_sim_timestep,
                            history.loc[history.in_comparison_window, "solve_seconds"].sum()/16)
            active_ev = [c for c in controllers if c.equipment_name.startswith("EV_") and c.solve_history]
            self.assertEqual(len(active_ev), 2)
            for controller in active_ev:
                self.assertEqual(controller.first_observation, evaluation-pd.Timedelta(minutes=15))
                self.assertEqual(controller.solve_history[1]["timestamp"], evaluation)
            runner.save_configuration(unified, [home])
            metadata = json.loads((Path(temp)/"resolved_configuration.json").read_text())
            self.assertEqual(metadata["evaluation_window"]["warmup_hours"], .25)
            self.assertEqual(metadata["evaluation_window"]["cooldown_hours"], .25)

    def test_metrics_filter_solves_overrides_and_boundary_cycles(self):
        from types import SimpleNamespace
        ev = vehicle()
        name = f"EV_{ev.vehicle_id}"
        index = pd.date_range("2018-01-01", periods=2, freq="min")
        raw = pd.DataFrame({f"{name} Electric Power (kW)": [-2., 2.],
            f"{name} SOC (-)": [.6, .59], f"{name} End SOC (-)": [.59, .60],
            f"{name} Unmet Load (kWh)": 0., f"{name} Safety Override": 0}, index=index)
        previous = pd.Series({f"{name} Electric Power (kW)": -2., f"{name} End SOC (-)": .6})
        times = [index[0]-pd.Timedelta(minutes=1), index[0], index[-1]+pd.Timedelta(minutes=1)]
        ctl = SimpleNamespace(equipment_name=name,
            solve_history=[dict(timestamp=t, solve_seconds=s, ocp_seconds=s, successful=True)
                           for t, s in zip(times, [100., 2., 200.])],
            action_history=[dict(timestamp=t, projection_override=True) for t in times])
        row = ev_metrics(raw, replace(CONFIG.homes[0], evs=(ev,)), pd.Series(.1, index=index),
                         1/60, [ctl], previous=previous).iloc[0]
        self.assertEqual((row.charge_starts, row.discharge_starts, row.cycle_count), (1, 0, 1))
        self.assertEqual((row.solve_attempts, row.mean_solve_seconds, row.max_solve_seconds), (1, 2, 2))
        self.assertEqual(row.projection_overrides, 1)
        self.assertEqual(row.initial_soc, .6)

    def test_prepared_cross_year_schedule_is_not_repeated(self):
        from ochre.utils.schedule import resample_and_reindex
        index = pd.date_range("2017-12-31 23:45", periods=60, freq="min")
        data = pd.DataFrame({"value": np.arange(60)}, index=index)
        result = resample_and_reindex(data, pd.Timedelta(minutes=1), index[0], pd.Timedelta(minutes=45))
        np.testing.assert_array_equal(result.value, np.arange(45))
        self.assertEqual(result.index[-1], pd.Timestamp("2018-01-01 00:29"))

    def test_annual_schedule_still_repeats_when_coverage_is_missing(self):
        from ochre.utils.schedule import resample_and_reindex
        index = pd.date_range("2017-01-01", periods=8760, freq="h", tz="Etc/GMT+5")
        data = pd.DataFrame({"value": np.arange(8760)}, index=index)
        with redirect_stdout(io.StringIO()):
            result = resample_and_reindex(data, pd.Timedelta(hours=1),
                pd.Timestamp("2017-12-31 23:00"), pd.Timedelta(hours=3))
        np.testing.assert_array_equal(result.value, [8759, 0, 1])

    def test_water_replay_anchor_and_training_origin_precede_evaluation(self):
        from examples.single_home_hems.single_home_hems.water_forecast import DCHouseWaterData
        from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation
        evaluation = pd.Timestamp("2018-01-01")
        source = pd.Timestamp("2020-01-31", tz="America/New_York")
        flow = pd.Series(np.arange(120), index=pd.date_range(source-pd.Timedelta(hours=1), periods=120, freq="min"))
        data = DCHouseWaterData(flow, evaluation, source)
        config = replace(CONFIG, start_time=(evaluation-pd.Timedelta(minutes=15)).to_pydatetime())
        with patch.object(DCHouseWaterData, "from_csv", return_value=data) as read:
            sim = SingleHomeHEMSSimulation(config, CONFIG.homes[0], None,
                water_draw_file=Path("fixture.csv"), water_draw_anchor_time=evaluation)
        self.assertEqual(read.call_args.kwargs["simulation_start"], evaluation)
        self.assertEqual(sim.water_draw_data.simulation_start, config.start_time)
        self.assertEqual(sim.water_draw_data.source_start, source-pd.Timedelta(minutes=15))
        self.assertEqual(sim.water_draw_data.source_time(evaluation), source)
        self.assertEqual(sim.water_draw_data.flow_at(pd.DatetimeIndex([evaluation])).iloc[0], 60)


if __name__ == "__main__":
    unittest.main()

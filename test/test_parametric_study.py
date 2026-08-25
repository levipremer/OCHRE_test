"""Tests for the isolated five-home seasonal electrification study."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.parametric_study.config import (
    load_study_configuration,
    resolve_all_seasons,
)
from examples.five_home_block.five_home_block.parametric_study.cooking import (
    FuelAwareCookingRange,
)
from examples.five_home_block.five_home_block.parametric_study.equipment import (
    _hpxml_backup_heating_capacity_w,
    build_daily_ev_events,
    build_scenario_equipment_overrides,
)
from examples.five_home_block.five_home_block.parametric_study.outputs import (
    AGGREGATABLE_COLUMNS,
    reconstruct_neighborhood_profile,
)
from examples.five_home_block.five_home_block.parametric_study.scenarios import (
    FIXED_HOME_STATE,
    generate_equipment_states,
    generate_scenario_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = (
    REPO_ROOT / "examples" / "five_home_block" / "parametric_config" / "study.json"
)


class StudyConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_study_configuration(STUDY_CONFIG, repo_root=REPO_ROOT)

    def test_study_references_untouched_base_configuration(self) -> None:
        self.assertEqual(self.config.base_config_directory.name, "config")
        self.assertEqual(self.config.variable_home_ids, tuple(f"home_{i:02d}" for i in range(2, 6)))
        self.assertEqual(self.config.maximum_erwh_homes, 1)
        self.assertEqual(self.config.home("home_01").ev_charger_power_kw, 11.5)

    def test_degree_hour_periods_are_deterministic(self) -> None:
        seasons = {season.season_id: season for season in resolve_all_seasons(self.config)}
        self.assertEqual(seasons["summer"].report_start, dt.datetime(2018, 7, 13))
        self.assertEqual(seasons["winter"].report_start, dt.datetime(2018, 12, 19))
        for season in seasons.values():
            self.assertEqual(season.report_end - season.report_start, dt.timedelta(days=7))
            self.assertEqual(season.report_start - season.run_start, dt.timedelta(days=2))

    def test_equipment_states_and_erwh_constrained_manifest(self) -> None:
        self.assertEqual(len(generate_equipment_states()), 12)
        manifest = generate_scenario_manifest(
            ("summer", "winter"),
            self.config.variable_home_ids,
            self.config.maximum_erwh_homes,
        )
        self.assertEqual(len(manifest), 24576)
        self.assertTrue((manifest.groupby("season_id").size() == 12288).all())
        self.assertLessEqual(manifest["erwh_homes"].max(), 1)
        self.assertTrue(manifest["home_01_profile_id"].str.endswith("fixed-electric").all())

    def test_home_one_uses_existing_controls_without_sizing_overrides(self) -> None:
        period = resolve_all_seasons(self.config)[0]
        home = self.config.home("home_01")
        overrides = build_scenario_equipment_overrides(
            self.config, home, period, FIXED_HOME_STATE
        )
        heater = overrides["ASHP Heater"]
        self.assertEqual(heater["Deadband Temperature (C)"], home.heating_deadband_c)
        self.assertEqual(
            heater["Compressor Minimum On Time (minutes)"],
            home.compressor_min_on_time_min,
        )
        self.assertNotIn("Capacity (W)", heater)
        self.assertNotIn("Backup Capacity (W)", heater)
        self.assertEqual(
            overrides["Water Heating"]["Setpoint Temperature (C)"],
            home.water_heater_setpoint_c,
        )

    def test_gas_furnace_uses_existing_hpxml_design_capacity(self) -> None:
        period = resolve_all_seasons(self.config)[1]
        home = self.config.home("home_02")
        state = next(
            state
            for state in generate_equipment_states()
            if state.heating == "gas"
            and state.water_heating == "gas"
            and state.cooking == "gas"
        )
        overrides = build_scenario_equipment_overrides(self.config, home, period, state)
        self.assertAlmostEqual(
            overrides["HVAC Heating"]["Capacity (W)"],
            _hpxml_backup_heating_capacity_w(home),
        )
        self.assertGreater(
            overrides["HVAC Heating"]["Capacity (W)"], 2 * 10960
        )

    def test_week_long_ev_schedule_repeats_existing_assumptions(self) -> None:
        period = resolve_all_seasons(self.config)[0]
        home = self.config.home("home_01")
        events = build_daily_ev_events(home, period.run_start, period.run_end)
        self.assertEqual(len(events), 9)
        self.assertTrue((events["start_soc"] == home.ev_arrival_soc_fraction).all())
        self.assertTrue((events["end_time"] > events["start_time"]).all())


class FuelAwareCookingTests(unittest.TestCase):
    def test_combined_event_retains_electric_and_gas_power(self) -> None:
        start = dt.datetime(2018, 1, 1)
        index = pd.date_range(start, periods=4, freq="1min")
        schedule = pd.DataFrame(
            {
                "Cooking Range (kW)": np.full(4, 0.05),
                "Cooking Range (therms/hour)": np.full(4, 0.02),
            },
            index=index,
        )
        cooking = FuelAwareCookingRange(
            name="Cooking Range",
            start_time=start,
            duration=dt.timedelta(minutes=4),
            time_res=dt.timedelta(minutes=1),
            schedule=schedule,
            save_results=False,
            verbosity=7,
        )
        results = cooking.simulate()
        self.assertTrue(cooking.is_electric)
        self.assertTrue(cooking.is_gas)
        self.assertGreater(results["Cooking Range Electric Power (kW)"].max(), 0)
        self.assertGreater(results["Cooking Range Gas Power (therms/hour)"].max(), 0)


class ReconstructionTests(unittest.TestCase):
    def test_manifest_row_reconstructs_exact_sum(self) -> None:
        index = pd.date_range("2018-01-01", periods=3, freq="1min", name="timestamp")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            scenario: dict[str, object] = {}
            for number in range(1, 6):
                profile_name = f"profile_{number}"
                scenario[f"home_{number:02d}_profile_id"] = profile_name
                data = pd.DataFrame(index=index)
                for column in AGGREGATABLE_COLUMNS:
                    data[column] = float(number)
                data["is_reporting_period"] = True
                data.to_parquet(directory / f"{profile_name}.parquet")
            aggregate = reconstruct_neighborhood_profile(scenario, directory)
            for column in AGGREGATABLE_COLUMNS:
                self.assertTrue((aggregate[column] == 15.0).all())


if __name__ == "__main__":
    unittest.main()

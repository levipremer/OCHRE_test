"""Lightweight tests for the modular five-home neighborhood example."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.aggregation import (
    aggregate_home_results,
    maximum_aggregate_error,
    maximum_power_balance_error,
)
from examples.five_home_block.five_home_block.config import (
    ConfigurationError,
    load_configuration,
)
from examples.five_home_block.five_home_block.output_mapping import (
    OutputMappingError,
    standardize_results,
)
from examples.five_home_block.five_home_block.home_factory import (
    _equipment_overrides,
    create_dwelling,
    resolve_noncommute_trip_overlaps,
)
from examples.five_home_block.five_home_block.plotting import (
    plot_home_power,
    plot_home_states,
    plot_neighborhood_power,
    plot_weather,
)
from examples.five_home_block.five_home_block.transformer import DistributionTransformer
from ochre.Analysis import calculate_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPO_ROOT / "examples" / "five_home_block" / "config"


def make_raw_results(scale: float = 1.0) -> pd.DataFrame:
    index = pd.date_range("2018-01-01", periods=12, freq="5min", name="timestamp")
    return pd.DataFrame(
        {
            "Total Electric Power (kW)": np.full(12, 5.0 * scale),
            "HVAC Heating Electric Power (kW)": np.full(12, 2.0 * scale),
            "HVAC Cooling Electric Power (kW)": np.zeros(12),
            "Water Heating Electric Power (kW)": np.full(12, 1.0 * scale),
            "EV Electric Power (kW)": np.full(12, 1.0 * scale),
            "Temperature - Indoor (C)": np.full(12, 21.0),
            "EV SOC (-)": np.linspace(0.4, 0.5, 12),
            "Hot Water Average Temperature (C)": np.full(12, 51.0),
            "Hot Water Minimum Temperature (C)": np.full(12, 49.0),
            "Hot Water Maximum Temperature (C)": np.full(12, 53.0),
            "HVAC Heating Setpoint (C)": np.full(12, 20.0),
            "HVAC Cooling Setpoint (C)": np.full(12, 25.0),
        },
        index=index,
    )


class EVTripPoolTests(unittest.TestCase):
    def test_longer_noncommute_trip_wins_overlap(self) -> None:
        trip_pool = pd.DataFrame(
            [
                {
                    "marker": "long",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 18:15").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 22:05").time(),
                },
                {
                    "marker": "short",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 19:00").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 19:15").time(),
                },
            ]
        )

        resolved = resolve_noncommute_trip_overlaps(trip_pool)

        self.assertEqual(resolved["marker"].tolist(), ["long"])
        dropped = resolved.attrs["dropped_overlapping_trips"]
        self.assertEqual(dropped["marker"].tolist(), ["short"])
        self.assertEqual(
            dropped.loc[0, "retained_trip_start_time"],
            pd.Timestamp("2018-01-01 18:15").time(),
        )

    def test_earlier_start_wins_equal_duration_overlap(self) -> None:
        trip_pool = pd.DataFrame(
            [
                {
                    "marker": "later",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 13:30").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 15:30").time(),
                },
                {
                    "marker": "earlier",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 13:00").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 15:00").time(),
                },
            ]
        )

        resolved = resolve_noncommute_trip_overlaps(trip_pool)

        self.assertEqual(resolved["marker"].tolist(), ["earlier"])
        dropped = resolved.attrs["dropped_overlapping_trips"]
        self.assertEqual(dropped["marker"].tolist(), ["later"])

    def test_nonoverlapping_and_commute_trips_are_unchanged(self) -> None:
        trip_pool = pd.DataFrame(
            [
                {
                    "marker": "morning",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 08:00").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 09:00").time(),
                },
                {
                    "marker": "evening",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "noncommute",
                    "trip_start_time": pd.Timestamp("2018-01-01 18:00").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 19:00").time(),
                },
                {
                    "marker": "commute",
                    "vehicle_case_id": "vehicle_1",
                    "day_of_week": 3,
                    "trip_type": "commute",
                    "trip_start_time": pd.Timestamp("2018-01-01 08:30").time(),
                    "trip_end_time": pd.Timestamp("2018-01-01 18:30").time(),
                },
            ]
        )

        resolved = resolve_noncommute_trip_overlaps(trip_pool)

        self.assertEqual(resolved["marker"].tolist(), trip_pool["marker"].tolist())
        self.assertTrue(resolved.attrs["dropped_overlapping_trips"].empty)


class ConfigurationTests(unittest.TestCase):
    def test_loads_five_valid_homes(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        self.assertEqual(len(config.homes), 5)
        self.assertEqual(config.time_step, pd.Timedelta(minutes=1))
        self.assertEqual(config.transformer.power_rating_kva, 50.0)
        self.assertEqual(config.transformer.overload_capacity_kva, 75.0)
        self.assertEqual(config.defrost_model, "Discrete")
        self.assertEqual(config.defrost_control_type, "Auto")
        self.assertEqual(config.defrost_er_strategy, "Aggressive")
        self.assertTrue(config.weather_file.is_file())
        self.assertTrue(all(home.hpxml_file.is_file() for home in config.homes))

    def test_duplicate_home_ids_fail(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        duplicate = replace(config, homes=(config.homes[0], config.homes[0]))
        with self.assertRaisesRegex(ConfigurationError, "unique"):
            duplicate.validate()

    def test_envelopes_have_two_stories_and_unconditioned_basements(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        for home in config.homes:
            root = ET.parse(home.hpxml_file).getroot()
            elements = list(root.iter())
            above_grade_floors = [
                element.text
                for element in elements
                if element.tag.rsplit("}", 1)[-1] == "NumberofConditionedFloorsAboveGrade"
            ]
            basement_conditioning = [
                child.text
                for foundation_type in elements
                if foundation_type.tag.rsplit("}", 1)[-1] == "FoundationType"
                for basement in foundation_type
                if basement.tag.rsplit("}", 1)[-1] == "Basement"
                for child in basement
                if child.tag.rsplit("}", 1)[-1] == "Conditioned"
            ]
            self.assertEqual(above_grade_floors, ["2.0"], home.home_id)
            self.assertEqual(basement_conditioning, ["false"], home.home_id)

    def test_equipment_overrides_preserve_hpxml_sizing(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        forbidden_sizing_fields = {
            "Capacity (W)",
            "Tank Volume (L)",
            "HPWH Capacity (W)",
            "HPWH Power (W)",
        }
        for home in config.homes:
            # Exercise the optional override with a fixed test input, independent
            # of the operating deadband selected in the example configuration.
            home = replace(
                home,
                water_heater_deadband_c=2.75 if home.home_id == "home_01" else None,
            )
            overrides = _equipment_overrides(
                home,
                config,
                np.random.default_rng(config.random_seed + home.seed_offset),
                evs=(),
                defrost_model=config.defrost_model,
                defrost_control_type=config.defrost_control_type,
            )
            for parameters in overrides.values():
                self.assertTrue(
                    forbidden_sizing_fields.isdisjoint(parameters),
                    home.home_id,
                )
            self.assertEqual(overrides["ASHP Heater"]["Deadband Temperature (C)"], 1.5)
            self.assertEqual(overrides["ASHP Heater"]["Deadband Offset (C)"], 0.5)
            self.assertEqual(overrides["ASHP Cooler"]["Deadband Temperature (C)"], 1.5)
            self.assertEqual(overrides["ASHP Cooler"]["Deadband Offset (C)"], 0.5)
            if home.home_id == "home_01":
                self.assertEqual(
                    overrides["Water Heating"]["Deadband Temperature (C)"], 2.75
                )
            else:
                self.assertNotIn("Deadband Temperature (C)", overrides["Water Heating"])
            self.assertEqual(
                overrides["ASHP Heater"]["Compressor Minimum On Time (minutes)"], 15
            )
            self.assertEqual(
                overrides["ASHP Heater"]["Compressor Minimum Off Time (minutes)"], 15
            )
            self.assertEqual(
                overrides["ASHP Heater"]["Backup Deadband Temperature (C)"], 1.5
            )
            self.assertEqual(
                overrides["ASHP Heater"]["Backup Setpoint Offset (C)"], 1.5
            )
            self.assertEqual(
                overrides["ASHP Heater"]["Backup Stage Escalation Delay (minutes)"],
                10,
            )
            if home.backup_number_of_stages is None:
                self.assertNotIn("Number of Backup Stages (-)", overrides["ASHP Heater"])
            else:
                self.assertEqual(
                    overrides["ASHP Heater"]["Number of Backup Stages (-)"],
                    home.backup_number_of_stages,
                )
            self.assertNotIn("Backup Stage Capacities (W)", overrides["ASHP Heater"])
            self.assertEqual(
                overrides["ASHP Heater"]["Backup Capacity (W)"],
                home.backup_stage_capacities_kw * 1000,
            )
            self.assertEqual(overrides["ASHP Heater"]["Defrost Model"], "Discrete")
            self.assertEqual(overrides["ASHP Heater"]["Defrost Control Type"], "Auto")
            self.assertEqual(
                overrides["ASHP Heater"]["Defrost ER Strategy"], "Aggressive"
            )

    def test_explicit_er_stages_flow_to_equipment_and_override_total(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        home = replace(
            config.homes[0],
            backup_number_of_stages=3,
            backup_stage_capacities_kw=(10, 5, 5),
        )
        overrides = _equipment_overrides(
            home,
            config,
            np.random.default_rng(config.random_seed + home.seed_offset),
            evs=(),
            defrost_model=config.defrost_model,
            defrost_control_type=config.defrost_control_type,
        )
        heater = overrides["ASHP Heater"]
        self.assertEqual(heater["Number of Backup Stages (-)"], 3)
        self.assertEqual(heater["Backup Stage Capacities (W)"], [10000, 5000, 5000])
        self.assertEqual(sum(heater["Backup Stage Capacities (W)"]), 20000)


class OutputMappingTests(unittest.TestCase):
    def test_mapping_and_power_balance(self) -> None:
        mapped = standardize_results(make_raw_results(), "test_home").data
        self.assertTrue((mapped["hvac_power_kw"] == 2.0).all())
        self.assertTrue((mapped["rest_of_home_power_kw"] == 1.0).all())
        self.assertLessEqual(maximum_power_balance_error(mapped), 1e-12)
        self.assertTrue(mapped["ev_soc_fraction"].between(0, 1).all())

    def test_missing_required_column_lists_available_columns(self) -> None:
        raw = make_raw_results().drop(columns="EV SOC (-)")
        with self.assertRaises(OutputMappingError) as context:
            standardize_results(raw, "test_home")
        self.assertIn("Available OCHRE columns", str(context.exception))
        self.assertIn("Cannot map EV SOC", str(context.exception))

    def test_maps_and_validates_multiple_ev_result_columns(self) -> None:
        raw = make_raw_results().drop(columns="EV SOC (-)")
        raw["EV_01A Electric Power (kW)"] = 0.4
        raw["EV_01B Electric Power (kW)"] = 0.6
        raw["EV_01A SOC (-)"] = 0.3
        raw["EV_01B SOC (-)"] = 0.7
        raw["EV_01A Unmet Load (kWh)"] = 0.0
        raw["EV_01B Unmet Load (kWh)"] = 0.0

        mapped = standardize_results(raw, "test_home").data

        self.assertTrue((mapped["ev_01a_power_kw"] == 0.4).all())
        self.assertTrue((mapped["ev_01b_power_kw"] == 0.6).all())
        self.assertTrue((mapped["ev_01a_soc_fraction"] == 0.3).all())
        self.assertTrue((mapped["ev_01b_soc_fraction"] == 0.7).all())
        self.assertNotIn("ev_soc_fraction", mapped)

        single_ev = raw.drop(
            columns=[
                "EV_01B Electric Power (kW)",
                "EV_01B SOC (-)",
                "EV_01B Unmet Load (kWh)",
            ]
        ).copy()
        single_ev["EV Electric Power (kW)"] = single_ev[
            "EV_01A Electric Power (kW)"
        ]
        single_mapped = standardize_results(single_ev, "test_home").data
        self.assertTrue(
            single_mapped["ev_soc_fraction"].equals(
                single_mapped["ev_01a_soc_fraction"]
            )
        )

        raw["EV Electric Power (kW)"] = 1.1
        with self.assertRaisesRegex(OutputMappingError, "aggregate EV power"):
            standardize_results(raw, "test_home")


class MultiEVOutputTests(unittest.TestCase):
    def test_two_ev_dwelling_preserves_aggregate_and_per_vehicle_results(self) -> None:
        config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            short_config = replace(
                config,
                duration=dt.timedelta(minutes=2),
                output_directory=output_directory,
            )
            dwelling = create_dwelling(
                short_config,
                short_config.homes[1],
                output_directory / "raw",
            )
            raw, _, _ = dwelling.simulate()

        per_ev_power_columns = [
            "EV_02_A Electric Power (kW)",
            "EV_02_B Electric Power (kW)",
        ]
        self.assertEqual(len(dwelling.equipment_by_end_use["EV"]), 2)
        self.assertTrue(
            np.allclose(
                raw["EV Electric Power (kW)"],
                raw[per_ev_power_columns].sum(axis=1),
            )
        )
        self.assertIn("EV_02_A SOC (-)", raw)
        self.assertIn("EV_02_B SOC (-)", raw)
        self.assertNotIn("EV SOC (-)", raw)

        standardized = standardize_results(raw, "home_02").data
        self.assertTrue(
            np.allclose(
                standardized["ev_power_kw"],
                standardized[["ev_02_a_power_kw", "ev_02_b_power_kw"]].sum(axis=1),
            )
        )
        self.assertLessEqual(maximum_power_balance_error(standardized), 1e-12)

    def test_two_ev_metrics_include_aggregate_and_per_vehicle_values(self) -> None:
        index = pd.date_range("2018-01-01", periods=2, freq="1h")
        results = pd.DataFrame(
            {
                "Total Electric Power (kW)": [3.0, 4.0],
                "EV Electric Power (kW)": [1.0, 2.0],
                "EV_01A SOC (-)": [0.4, 0.6],
                "EV_01B SOC (-)": [0.5, 0.7],
                "EV_01A Unmet Load (kWh)": [0.0, -0.2],
                "EV_01B Unmet Load (kWh)": [-0.1, 0.0],
            },
            index=index,
        )

        metrics = calculate_metrics(results, metrics_verbosity=4)

        self.assertNotIn("Average EV SOC (-)", metrics)
        self.assertAlmostEqual(metrics["Average EV_01A SOC (-)"], 0.5)
        self.assertAlmostEqual(metrics["Average EV_01B SOC (-)"], 0.6)
        self.assertAlmostEqual(metrics["Total EV Unmet Load (kWh)"], -0.3)
        self.assertAlmostEqual(metrics["Total EV_01A Unmet Load (kWh)"], -0.2)
        self.assertAlmostEqual(metrics["Total EV_01B Unmet Load (kWh)"], -0.1)


class AggregationAndPlotTests(unittest.TestCase):
    def test_transformer_adds_static_capacity_limits(self) -> None:
        transformer = DistributionTransformer(
            power_rating_kva=50,
            overload_capacity_percent=150,
        )
        aggregate = aggregate_home_results(
            {"home_01": standardize_results(make_raw_results(), "home_01").data}
        )
        aggregate_with_limits = transformer.add_capacity_columns(aggregate)
        self.assertTrue(
            (aggregate_with_limits["transformer_power_rating_kva"] == 50).all()
        )
        self.assertTrue(
            (aggregate_with_limits["transformer_overload_capacity_kva"] == 75).all()
        )

    def test_aggregation_matches_direct_sum(self) -> None:
        homes = {
            f"home_{number:02d}": standardize_results(
                make_raw_results(float(number)), f"home_{number:02d}"
            ).data
            for number in range(1, 6)
        }
        aggregate = DistributionTransformer(50, 150).add_capacity_columns(
            aggregate_home_results(homes)
        )
        self.assertLessEqual(maximum_aggregate_error(homes, aggregate), 1e-12)
        self.assertTrue((aggregate["aggregate_total_power_kw"] == 75.0).all())

    def test_required_figure_generation(self) -> None:
        homes = {
            f"home_{number:02d}": standardize_results(
                make_raw_results(float(number)), f"home_{number:02d}"
            ).data
            for number in range(1, 6)
        }
        aggregate = DistributionTransformer(50, 150).add_capacity_columns(
            aggregate_home_results(homes)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            files = [output / "neighborhood_aggregate_power.png", output / "shared_weather.png"]
            plot_neighborhood_power(aggregate, files[0], dpi=72)
            weather = pd.DataFrame(
                {
                    "Ambient Dry Bulb (C)": np.linspace(-2, 8, len(aggregate)),
                    "Ambient Relative Humidity (-)": np.full(len(aggregate), 0.7),
                    "GHI (W/m^2)": np.linspace(0, 400, len(aggregate)),
                    "DNI (W/m^2)": np.linspace(0, 500, len(aggregate)),
                    "DHI (W/m^2)": np.linspace(0, 100, len(aggregate)),
                    "Wind Speed (m/s)": np.full(len(aggregate), 2.0),
                },
                index=aggregate.index,
            )
            plot_weather(weather, files[1], dpi=72)
            for home_id, data in homes.items():
                power_file = output / f"{home_id}_power_breakdown.png"
                state_file = output / f"{home_id}_state_variables.png"
                plot_home_power(data, home_id, power_file, dpi=72)
                plot_home_states(data, home_id, state_file, dpi=72)
                files.extend((power_file, state_file))
            self.assertEqual(len(files), 12)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in files))


if __name__ == "__main__":
    unittest.main()

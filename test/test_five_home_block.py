"""Lightweight tests for the modular five-home neighborhood example."""

from __future__ import annotations

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
from examples.five_home_block.five_home_block.home_factory import _equipment_overrides
from examples.five_home_block.five_home_block.plotting import (
    plot_home_power,
    plot_home_states,
    plot_neighborhood_power,
    plot_weather,
)
from examples.five_home_block.five_home_block.transformer import DistributionTransformer


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
            "Backup Capacity (W)",
            "Tank Volume (L)",
            "HPWH Capacity (W)",
            "HPWH Power (W)",
        }
        for home in config.homes:
            overrides = _equipment_overrides(
                home,
                config.start_time,
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
            self.assertNotIn("Number of Backup Stages (-)", overrides["ASHP Heater"])
            self.assertNotIn("Backup Stage Capacities (W)", overrides["ASHP Heater"])
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
            config.start_time,
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
        self.assertIn("ev_soc_fraction", str(context.exception))


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

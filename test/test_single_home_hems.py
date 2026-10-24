"""Tests for the selectable single-home HEMS example."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.controllers import (
    CompositeController,
    HVACPredictiveController,
    WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems.forecast import (
    Forecast,
    TimeOfUsePrice,
)
from examples.single_home_hems.single_home_hems.training import (
    DELIVERED_HEAT_COLUMN,
    DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN,
    DIRECT_NORMAL_IRRADIANCE_COLUMN,
    DISTURBANCE_COLUMNS,
    HVACInternalGainRegime,
    HVACSplitWindows,
    HVACStateSpaceConfig,
    HVACTrainingWindow,
    WaterHeaterSensorLayout,
    WaterHeaterSplitWindows,
    WaterHeaterStateSpaceConfig,
    WaterHeaterTrainingWindow,
    add_projected_dni_candidates,
    aggregate_hvac_dataset,
    aggregate_water_heater_dataset,
    hvac_training_record,
    prepare_hvac_dataset_splits,
    prepare_hvac_training_dataset,
    prepare_water_heater_dataset_splits,
    prepare_water_heater_training_dataset,
    projected_dni_column,
    to_hvac_state_space_frame,
    to_state_space_frame,
    water_heater_training_record,
)
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    select_home,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPO_ROOT / "examples" / "five_home_block" / "config"


class SingleHomeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)

    @staticmethod
    def forecast() -> Forecast:
        index = pd.date_range("2018-01-01", periods=24, freq="h")
        schedule = pd.DataFrame(
            {
                "Ambient Dry Bulb (C)": range(24),
                "Water Heating (L/min)": 0.0,
            },
            index=index,
        )
        return Forecast.from_schedule(schedule, TimeOfUsePrice().price_at)

    def test_each_block_home_can_be_selected(self) -> None:
        for number in range(1, 6):
            home_id = f"home_{number:02d}"
            self.assertEqual(select_home(self.config, home_id).home_id, home_id)

    def test_unknown_home_lists_valid_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "home_01"):
            select_home(self.config, "home_99")

    def test_tou_price_has_three_to_six_pm_peak(self) -> None:
        price = TimeOfUsePrice()
        self.assertEqual(price.price_at(pd.Timestamp("2018-01-01 14:59")), 0.15)
        self.assertEqual(price.price_at(pd.Timestamp("2018-01-01 15:00")), 0.30)
        self.assertEqual(price.price_at(pd.Timestamp("2018-01-01 17:59")), 0.30)
        self.assertEqual(price.price_at(pd.Timestamp("2018-01-01 18:00")), 0.15)

    def test_default_hems_preserves_immediate_charge_baseline(self) -> None:
        controller = HEMSController()
        forecast = self.forecast()
        peak_controls = controller.controls_for_step(
            pd.Timestamp("2018-01-01 16:00"),
            previous_status=None,
            current_schedule={},
            forecast=forecast,
        )
        off_peak_controls = controller.controls_for_step(
            pd.Timestamp("2018-01-01 18:00"),
            previous_status=None,
            current_schedule={},
            forecast=forecast,
        )
        self.assertEqual(peak_controls, {})
        self.assertEqual(off_peak_controls, {})

    def test_forecast_returns_requested_horizon(self) -> None:
        window = self.forecast().window(
            pd.Timestamp("2018-01-01 12:00"),
            pd.Timedelta(hours=6),
        )
        self.assertEqual(len(window), 6)
        self.assertEqual(window.index[-1], pd.Timestamp("2018-01-01 17:00"))

    def test_unimplemented_device_controllers_are_isolated_pass_throughs(self) -> None:
        forecast = self.forecast()
        for controller in (
            WaterHeaterPredictiveController(),
            HVACPredictiveController(),
        ):
            controls = controller.controls_for_step(
                pd.Timestamp("2018-01-01 16:00"),
                previous_status=None,
                current_schedule={},
                forecast=forecast,
            )
            self.assertEqual(controls, {})

    def test_composite_rejects_overlapping_device_commands(self) -> None:
        from unittest.mock import Mock
        duplicate_ev_controllers = CompositeController(
            tuple(Mock(controls_for_step=Mock(return_value={"EV_01A": {"P Setpoint": 0}})) for _ in range(2))
        )
        with self.assertRaisesRegex(ValueError, "Multiple controllers"):
            duplicate_ev_controllers.controls_for_step(
                pd.Timestamp("2018-01-01 16:00"),
                previous_status=None,
                current_schedule={},
                forecast=self.forecast(),
            )

    def test_water_heater_sensor_layout_samples_configured_nodes(self) -> None:
        layout = WaterHeaterSensorLayout()
        readings = layout.measure(np.arange(1, 13, dtype=float))
        self.assertEqual(
            readings,
            {
                "water_heater_sensor_node_03_temperature_c": 3.0,
                "water_heater_sensor_node_10_temperature_c": 10.0,
            },
        )
        self.assertAlmostEqual(layout.height_fraction_from_bottom(3), 19 / 24)
        self.assertAlmostEqual(layout.height_fraction_from_bottom(10), 5 / 24)

    def test_water_heater_sensor_layout_rejects_invalid_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            WaterHeaterSensorLayout(model_nodes=12, sensor_nodes=(1, 13))

    def test_water_heater_training_record_has_outputs_and_driving_inputs(self) -> None:
        tank = SimpleNamespace(
            states=np.arange(1, 13, dtype=float),
            next_states=np.arange(21, 33, dtype=float),
            volume=170.34,
            draw_total=2.5,
            mains_temp=12.0,
            current_schedule={"Zone Temperature (C)": 20.0},
        )
        water_heater = SimpleNamespace(
            model=tank,
            setpoint_temp=52.0,
            electric_kw=0.7,
            delivered_heat=1800.0,
            mode="Heat Pump On",
            capacity_rated=4500.0,
            hp_capacity_nominal=1967.35,
            cop_nominal=3.9347,
        )

        record = water_heater_training_record(
            water_heater,
            WaterHeaterSensorLayout(),
            initial_states=np.arange(11, 23, dtype=float),
        )

        self.assertEqual(record["water_heater_draw_l_per_min"], 2.5)
        self.assertEqual(record["water_heater_delivered_heat_kw"], 1.8)
        self.assertEqual(record["water_heater_mode"], "Heat Pump On")
        self.assertEqual(record["water_heater_tank_volume_l"], 170.34)
        self.assertEqual(record["water_heater_sensor_node_10_temperature_c"], 20.0)
        self.assertEqual(record["next_water_heater_sensor_node_10_temperature_c"], 30.0)

    def test_water_heater_training_window_separates_features_and_targets(self) -> None:
        index = pd.date_range("2018-01-01", periods=4, freq="h")
        layout = WaterHeaterSensorLayout()
        data = pd.DataFrame(
            {
                "water_heater_sensor_node_03_temperature_c": [50, 49, 48, 47],
                "water_heater_sensor_node_10_temperature_c": [45, 44, 43, 42],
                "water_heater_setpoint_temperature_c": [51, 51, 52, 52],
                "water_heater_draw_l_per_min": [0, 1, 0, 2],
                "water_heater_mains_temperature_c": [10, 10, 10, 10],
                "water_heater_tank_ambient_temperature_c": [20, 20, 20, 20],
                "next_water_heater_sensor_node_03_temperature_c": [49, 48, 47, 46],
                "next_water_heater_sensor_node_10_temperature_c": [44, 43, 42, 41],
                "water_heater_electric_power_kw": [0, 0.5, 0.5, 0],
                "water_heater_delivered_heat_kw": [0, 1.5, 1.5, 0],
                "water_heater_mode": ["Off", "Heat Pump On", "Heat Pump On", "Off"],
                "water_heater_tank_volume_l": [170.34] * 4,
                "water_heater_resistance_capacity_w": [4500.0] * 4,
                "water_heater_nominal_heat_pump_capacity_w": [1967.35] * 4,
                "water_heater_nominal_heat_pump_cop": [3.9347] * 4,
            },
            index=index,
        )
        window = WaterHeaterTrainingWindow(index[1], index[3])

        dataset = prepare_water_heater_training_dataset(data, window, layout)

        self.assertEqual(dataset.sample_count, 2)
        self.assertEqual(dataset.features.index.tolist(), [index[1], index[2]])
        self.assertNotIn("water_heater_setpoint_temperature_c", dataset.features)
        self.assertIn(DELIVERED_HEAT_COLUMN, dataset.features)
        self.assertIn("water_heater_setpoint_temperature_c", dataset.diagnostics)
        self.assertIn("water_heater_electric_power_kw", dataset.diagnostics)
        self.assertEqual(
            dataset.targets.columns.tolist(),
            [
                "next_water_heater_sensor_node_03_temperature_c",
                "next_water_heater_sensor_node_10_temperature_c",
            ],
        )

        splits = prepare_water_heater_dataset_splits(
            data,
            WaterHeaterSplitWindows(
                training=WaterHeaterTrainingWindow(index[0], index[1]),
                validation=WaterHeaterTrainingWindow(index[1], index[2]),
                test=WaterHeaterTrainingWindow(
                    index[2], index[-1] + pd.Timedelta(hours=1)
                ),
            ),
            layout,
        )
        self.assertEqual(splits.training.sample_count, 1)
        self.assertEqual(splits.validation.sample_count, 1)
        self.assertEqual(splits.test.sample_count, 2)

        with self.assertRaisesRegex(ValueError, "overlap"):
            WaterHeaterSplitWindows(
                training=WaterHeaterTrainingWindow(index[0], index[2]),
                validation=WaterHeaterTrainingWindow(index[1], index[3]),
                test=WaterHeaterTrainingWindow(
                    index[3], index[-1] + pd.Timedelta(hours=1)
                ),
            )

    def test_water_heater_aggregation_and_state_space_adapter(self) -> None:
        index = pd.date_range("2018-01-01", periods=30, freq="min")
        layout = WaterHeaterSensorLayout()
        sequence = np.arange(len(index), dtype=float)
        data = pd.DataFrame(
            {
                "water_heater_sensor_node_03_temperature_c": 50 + sequence,
                "water_heater_sensor_node_10_temperature_c": 40 + sequence,
                "water_heater_setpoint_temperature_c": 52.0,
                "water_heater_draw_l_per_min": sequence,
                "water_heater_mains_temperature_c": 10 + sequence / 10,
                "water_heater_tank_ambient_temperature_c": 20 + sequence / 20,
                "next_water_heater_sensor_node_03_temperature_c": 50.5 + sequence,
                "next_water_heater_sensor_node_10_temperature_c": 40.5 + sequence,
                "water_heater_electric_power_kw": sequence / 10,
                DELIVERED_HEAT_COLUMN: sequence / 10,
                "water_heater_mode": "Heat Pump On",
                "water_heater_tank_volume_l": 170.34,
                "water_heater_resistance_capacity_w": 4500.0,
                "water_heater_nominal_heat_pump_capacity_w": 1967.35,
                "water_heater_nominal_heat_pump_cop": 3.9347,
            },
            index=index,
        )
        window = WaterHeaterTrainingWindow(
            index[0], index[0] + pd.Timedelta(minutes=30)
        )
        dataset = prepare_water_heater_training_dataset(data, window, layout)
        config = WaterHeaterStateSpaceConfig(
            prediction_horizon=pd.Timedelta(minutes=30),
            disturbance_columns=(DISTURBANCE_COLUMNS[0], DISTURBANCE_COLUMNS[1]),
        )

        aggregated = aggregate_water_heater_dataset(dataset, config, layout)
        frame = to_state_space_frame(aggregated, config, layout)

        self.assertEqual(
            aggregated.index.tolist(),
            [pd.Timestamp("2018-01-01 00:15"), pd.Timestamp("2018-01-01 00:30")],
        )
        self.assertEqual(aggregated.iloc[0][layout.sensor_columns[0]], 50.0)
        self.assertEqual(
            aggregated.iloc[0][f"next_{layout.sensor_columns[0]}"], 64.5
        )
        self.assertEqual(aggregated.iloc[0][DELIVERED_HEAT_COLUMN], 0.7)
        self.assertEqual(
            frame.columns.tolist(),
            [
                "Input_1",
                "Output_1",
                "Output_2",
                "Disturbance_1",
                "Disturbance_2",
            ],
        )

    def test_water_heater_state_space_config_validates_time_steps(self) -> None:
        no_selection = WaterHeaterStateSpaceConfig(
            feature_selection_method="none"
        )
        self.assertEqual(no_selection.scaling, "none")

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            WaterHeaterStateSpaceConfig(
                source_timestep=pd.Timedelta(minutes=2),
                model_timestep=pd.Timedelta(minutes=15),
            )
        with self.assertRaisesRegex(ValueError, "scaling must be 'none'"):
            WaterHeaterStateSpaceConfig(
                feature_selection_method="none",
                scaling="standard_deviation",
            )

    @staticmethod
    def hvac_record(timestamp: pd.Timestamp, indoor_temperature: float) -> dict:
        indoor = SimpleNamespace(
            t_idx=0,
            hvac_sens_gain=1200.0,
            internal_sens_gain=200.0,
        )
        schedule = {
            "Ambient Dry Bulb (C)": 5.0,
            "GHI (W/m^2)": 300.0,
            "Occupancy (Persons)": 2.0,
            "DNI (W/m^2)": 400.0,
            "DHI (W/m^2)": 100.0,
            "zenith": 45.0,
            "azimuth": 180.0,
            "Ground Temperature (C)": 10.0,
            "Wind Speed (m/s)": 3.0,
            "Ambient Humidity Ratio (-)": 0.006,
            "Sky Temperature (C)": 0.0,
        }
        envelope = SimpleNamespace(
            current_schedule=schedule,
            indoor_zone=indoor,
            next_outputs=np.array([indoor_temperature + 0.1]),
            occupancy_sensible_gain=80.0,
        )
        heating = SimpleNamespace(
            electric_kw=0.5,
            temp_setpoint=20.0,
            mode="On",
        )
        cooling = SimpleNamespace(
            electric_kw=0.0,
            temp_setpoint=24.0,
            mode="Off",
        )
        return hvac_training_record(
            timestamp,
            envelope,
            heating,
            cooling,
            indoor_temperature,
        )

    def test_hvac_record_uses_global_solar_and_day_night_regime(self) -> None:
        day = self.hvac_record(pd.Timestamp("2018-01-01 12:00"), 21.0)
        night = self.hvac_record(pd.Timestamp("2018-01-01 23:00"), 20.0)
        custom_regime = HVACInternalGainRegime(day_start_hour=8, day_end_hour=18)

        self.assertEqual(day["global_horizontal_irradiance_w_per_m2"], 300.0)
        self.assertEqual(day["internal_gain_day_indicator"], 1)
        self.assertEqual(night["internal_gain_day_indicator"], 0)
        self.assertEqual(day["hvac_indoor_sensible_heat_w"], 1200.0)
        self.assertEqual(day["next_indoor_temperature_c"], 21.1)
        self.assertEqual(day["actual_internal_sensible_gain_w"], 360.0)
        self.assertEqual(custom_regime.is_day(pd.Timestamp("2018-01-01 07:00")), 0)

    def test_hvac_training_dataset_keeps_dhi_as_diagnostic(self) -> None:
        index = pd.date_range("2018-01-01 11:00", periods=4, freq="h")
        data = pd.DataFrame(
            [
                self.hvac_record(timestamp, 20.0 + i)
                for i, timestamp in enumerate(index)
            ],
            index=index,
        )
        window = HVACTrainingWindow(index[1], index[3])

        dataset = prepare_hvac_training_dataset(data, window)

        self.assertEqual(dataset.sample_count, 2)
        self.assertIn("global_horizontal_irradiance_w_per_m2", dataset.features)
        self.assertNotIn("diffuse_horizontal_irradiance_w_per_m2", dataset.features)
        self.assertIn(
            "diffuse_horizontal_irradiance_w_per_m2",
            dataset.diagnostics,
        )
        self.assertEqual(
            dataset.targets.columns.tolist(),
            ["next_indoor_temperature_c"],
        )
        self.assertIn("hvac_electric_power_kw", dataset.diagnostics)

        splits = prepare_hvac_dataset_splits(
            data,
            HVACSplitWindows(
                training=HVACTrainingWindow(index[0], index[1]),
                validation=HVACTrainingWindow(index[1], index[2]),
                test=HVACTrainingWindow(index[2], index[-1] + pd.Timedelta(hours=1)),
            ),
        )
        self.assertEqual(splits.training.sample_count, 1)
        self.assertEqual(splits.validation.sample_count, 1)
        self.assertEqual(splits.test.sample_count, 2)

        with self.assertRaisesRegex(ValueError, "overlap"):
            HVACSplitWindows(
                training=HVACTrainingWindow(index[0], index[2]),
                validation=HVACTrainingWindow(index[1], index[3]),
                test=HVACTrainingWindow(index[3], index[-1] + pd.Timedelta(hours=1)),
            )

    def test_projected_dni_candidates_use_surface_azimuth(self) -> None:
        data = pd.DataFrame(
            {
                DIRECT_NORMAL_IRRADIANCE_COLUMN: [800.0, 800.0],
                "solar_zenith_degrees": [60.0, 100.0],
                "solar_azimuth_degrees": [180.0, 180.0],
            }
        )
        config = HVACStateSpaceConfig(
            prediction_horizon=pd.Timedelta(minutes=30),
        )

        projected = add_projected_dni_candidates(data, config)

        self.assertEqual(len(config.projected_dni_columns), 36)
        for surface_azimuth in config.solar_azimuths_degrees:
            expected = 800 * max(
                np.sin(np.deg2rad(60))
                * np.cos(np.deg2rad(180 - surface_azimuth)),
                0,
            )
            self.assertAlmostEqual(
                projected.loc[0, projected_dni_column(surface_azimuth)],
                expected,
            )
        self.assertTrue(
            (projected.loc[1, list(config.projected_dni_columns)] == 0).all()
        )

    def test_hvac_aggregation_and_state_space_adapter(self) -> None:
        index = pd.date_range("2018-01-01 11:00", periods=30, freq="min")
        data = pd.DataFrame(
            [
                self.hvac_record(timestamp, 20 + i / 100)
                for i, timestamp in enumerate(index)
            ],
            index=index,
        )
        window = HVACTrainingWindow(
            index[0], index[0] + pd.Timedelta(minutes=30)
        )
        dataset = prepare_hvac_training_dataset(data, window)
        config = HVACStateSpaceConfig(
            prediction_horizon=pd.Timedelta(minutes=30),
            solar_azimuth_step_degrees=90,
            include_ghi=False,
        )

        aggregated = aggregate_hvac_dataset(dataset, config)
        frame = to_hvac_state_space_frame(aggregated, config)

        self.assertEqual(
            aggregated.index.tolist(),
            [pd.Timestamp("2018-01-01 11:15"), pd.Timestamp("2018-01-01 11:30")],
        )
        self.assertEqual(aggregated.iloc[0]["indoor_temperature_c"], 20.0)
        self.assertAlmostEqual(
            aggregated.iloc[0]["next_indoor_temperature_c"], 20.24
        )
        self.assertEqual(aggregated.iloc[1]["indoor_temperature_c"], 20.15)
        self.assertAlmostEqual(
            aggregated.iloc[1]["next_indoor_temperature_c"], 20.39
        )
        self.assertEqual(aggregated.iloc[0]["hvac_indoor_sensible_heat_w"], 1200.0)
        self.assertEqual(
            frame.columns.tolist(),
            [
                "Input_1",
                "Output_1",
                *[
                    f"Disturbance_{i + 1}"
                    for i in range(len(config.disturbance_columns))
                ],
            ],
        )
        self.assertIn(
            DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN,
            config.disturbance_columns,
        )

    def test_control_shape_validation(self) -> None:
        timestamp = pd.Timestamp("2018-01-01")
        valid = {"EV": {"P Setpoint": 0.0}}
        self.assertIs(
            SingleHomeHEMSSimulation._validate_controls(valid, timestamp), valid
        )
        with self.assertRaises(TypeError):
            SingleHomeHEMSSimulation._validate_controls({"EV": 0.0}, timestamp)
        with self.assertRaisesRegex(ValueError, "supports Setpoint"):
            SingleHomeHEMSSimulation._validate_controls(
                {"Water Heating": {"Load Fraction": 0.0}},
                timestamp,
            )


if __name__ == "__main__":
    unittest.main()

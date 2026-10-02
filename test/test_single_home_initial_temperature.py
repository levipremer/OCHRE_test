"""Regression checks for deterministic single-home HEMS thermal initialization."""

from __future__ import annotations

import datetime as dt
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.run_simulation import parse_args
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    select_home,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPO_ROOT / "examples" / "five_home_block" / "config"


class SingleHomeInitialTemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)
        # EV trip generation is unrelated to the initial building temperature.
        cls.home = replace(select_home(cls.config, "home_01"), evs=())

    def _run_two_steps(
        self,
        *,
        apply_controls: bool = True,
        random_seed: int = 123,
        **simulation_options,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Use the real factory and OCHRE envelope, without rendering plots."""
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            raw_directory = Path(directory) / "raw"
            config = replace(
                self.config,
                # OCHRE's schedule resampler needs two timestamps.
                duration=dt.timedelta(minutes=2),
                time_step=dt.timedelta(minutes=1),
                initialization_time=None,
                random_seed=random_seed,
                output_directory=Path(directory),
            )
            simulation = SingleHomeHEMSSimulation(
                config,
                self.home,
                HEMSController(device_controllers=()),
                **simulation_options,
            )
            results = simulation._run_stepwise(
                apply_controls=apply_controls,
                raw_directory=raw_directory,
            )
            # OCHRE saves the source schedule before the factory applies the
            # configured thermostat overrides. This is the envelope's schedule.
            initial_schedule = pd.read_csv(
                raw_directory / f"{self.home.home_id}_schedule.csv"
            )
            return results[-1], initial_schedule

    def test_default_temperature_is_fixed_across_seeds_and_baseline(self) -> None:
        with patch(
            "numpy.random.uniform",
            side_effect=AssertionError("Unexpected thermal random draw"),
        ):
            first, _ = self._run_two_steps(random_seed=123)
            second, _ = self._run_two_steps(random_seed=987)
            baseline, _ = self._run_two_steps(random_seed=987, apply_controls=False)

        for training in (first, second, baseline):
            self.assertEqual(training.iloc[0]["indoor_temperature_c"], 21.0)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(second, baseline)

    def test_explicit_temperature_reaches_controlled_and_baseline_plant(self) -> None:
        with patch(
            "numpy.random.uniform",
            side_effect=AssertionError("Unexpected thermal random draw"),
        ):
            controlled, _ = self._run_two_steps(initial_temperature_c=23.5)
            baseline, _ = self._run_two_steps(
                initial_temperature_c=23.5, apply_controls=False
            )

        self.assertEqual(controlled.iloc[0]["indoor_temperature_c"], 23.5)
        self.assertEqual(baseline.iloc[0]["indoor_temperature_c"], 23.5)
        pd.testing.assert_frame_equal(controlled, baseline)

    def test_random_initialization_requires_explicit_none(self) -> None:
        with patch("numpy.random.uniform", return_value=0.25) as random_draw:
            training, schedule = self._run_two_steps(initial_temperature_c=None)

        random_draw.assert_called_once()
        mode = (
            "HVAC Cooling"
            if schedule.iloc[0]["Ambient Dry Bulb (C)"] > 12
            else "HVAC Heating"
        )
        expected = schedule.iloc[0][f"{mode} Setpoint (C)"] + 0.25
        self.assertAlmostEqual(training.iloc[0]["indoor_temperature_c"], expected)

    def test_invalid_explicit_temperatures_are_rejected_before_simulation(self) -> None:
        for temperature in (True, False, "21", np.nan, np.inf, -np.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaises((TypeError, ValueError)):
                    SingleHomeHEMSSimulation(
                        self.config,
                        self.home,
                        HEMSController(device_controllers=()),
                        initial_temperature_c=temperature,
                    )


class SingleHomeInitialTemperatureCLITests(unittest.TestCase):
    @staticmethod
    def _parse(*arguments: str):
        with patch.object(sys, "argv", ["run_simulation.py", *arguments]):
            return parse_args()

    def test_cli_defaults_to_fixed_21_c(self) -> None:
        self.assertEqual(self._parse().initial_temperature_c, 21.0)

    def test_cli_accepts_an_explicit_temperature(self) -> None:
        self.assertEqual(
            self._parse("--initial-temperature-c", "23.5").initial_temperature_c,
            23.5,
        )

    def test_cli_randomization_is_explicit(self) -> None:
        self.assertIsNone(
            self._parse("--randomize-initial-temperature").initial_temperature_c
        )

    def test_cli_rejects_both_initialization_choices(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                self._parse(
                    "--initial-temperature-c", "23.5", "--randomize-initial-temperature"
                )
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

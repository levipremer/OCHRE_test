"""Tests for the selectable single-home HEMS example."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    select_home,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPO_ROOT / "examples" / "five_home_block" / "config"


class SingleHomeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_configuration(CONFIG_DIRECTORY, repo_root=REPO_ROOT)

    def test_each_block_home_can_be_selected(self) -> None:
        for number in range(1, 6):
            home_id = f"home_{number:02d}"
            self.assertEqual(select_home(self.config, home_id).home_id, home_id)

    def test_unknown_home_lists_valid_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "home_01"):
            select_home(self.config, "home_99")

    def test_default_hems_is_pass_through(self) -> None:
        controls = HEMSController().controls_for_step(
            pd.Timestamp("2018-01-01"),
            previous_status=None,
            current_schedule={},
        )
        self.assertEqual(controls, {})

    def test_control_shape_validation(self) -> None:
        timestamp = pd.Timestamp("2018-01-01")
        valid = {"EV": {"P Setpoint": 0.0}}
        self.assertIs(
            SingleHomeHEMSSimulation._validate_controls(valid, timestamp), valid
        )
        with self.assertRaises(TypeError):
            SingleHomeHEMSSimulation._validate_controls({"EV": 0.0}, timestamp)


if __name__ == "__main__":
    unittest.main()

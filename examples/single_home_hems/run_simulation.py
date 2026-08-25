#!/usr/bin/env python3
"""Run one selectable five-home-block dwelling with a stepwise HEMS."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import replace
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
FIVE_HOME_CONFIG = REPO_ROOT / "examples" / "five_home_block" / "config"
HOME_IDS = tuple(f"home_{number:02d}" for number in range(1, 6))
for local_path in (REPO_ROOT, EXAMPLE_ROOT):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from examples.five_home_block.five_home_block.config import load_configuration
from hems import HEMSController
from single_home_hems import SingleHomeHEMSSimulation, select_home


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one five-home-block dwelling with a user-editable HEMS"
    )
    parser.add_argument(
        "--home-id",
        required=True,
        choices=HOME_IDS,
        help="Home from the five-home block to simulate",
    )
    parser.add_argument(
        "--source-config-directory",
        type=Path,
        default=FIVE_HOME_CONFIG,
        help="Five-home configuration directory used as the source",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Default: examples/single_home_hems/outputs/HOME_ID",
    )
    parser.add_argument(
        "--time-step-minutes",
        type=float,
        help="Override the timestep inherited from the five-home block",
    )
    parser.add_argument(
        "--heating-deadband-c",
        type=float,
        help="Override the selected home's compressor thermostat deadband",
    )
    parser.add_argument(
        "--heating-deadband-offset",
        type=float,
        help="Override the deadband offset fraction (0.5 centers it)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_configuration(args.source_config_directory, repo_root=REPO_ROOT)
    home = select_home(config, args.home_id)

    if args.time_step_minutes is not None:
        if args.time_step_minutes <= 0:
            raise ValueError("--time-step-minutes must be positive")
        config = replace(
            config,
            time_step=dt.timedelta(minutes=args.time_step_minutes),
        )
    if args.heating_deadband_c is not None:
        if args.heating_deadband_c <= 0:
            raise ValueError("--heating-deadband-c must be positive")
        home = replace(home, heating_deadband_c=args.heating_deadband_c)
    if args.heating_deadband_offset is not None:
        if not 0 <= args.heating_deadband_offset <= 1:
            raise ValueError("--heating-deadband-offset must be between zero and one")
        home = replace(home, heating_deadband_offset=args.heating_deadband_offset)

    output_directory = args.output_directory or EXAMPLE_ROOT / "outputs" / args.home_id
    if not output_directory.is_absolute():
        output_directory = REPO_ROOT / output_directory
    config = replace(
        config,
        output_directory=output_directory.resolve(),
        homes=(home,),
    )
    config.validate()

    SingleHomeHEMSSimulation(config, home, HEMSController()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

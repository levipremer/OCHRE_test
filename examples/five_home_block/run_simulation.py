#!/usr/bin/env python3
"""Command-line entry point for the five-home OCHRE example."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
for local_path in (REPO_ROOT, EXAMPLE_ROOT):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from five_home_block import NeighborhoodSimulation, load_configuration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular five-home OCHRE baseline")
    parser.add_argument(
        "--config-directory",
        type=Path,
        default=EXAMPLE_ROOT / "config",
        help="Directory containing neighborhood.json and homes.json",
    )
    parser.add_argument(
        "--heating-deadband-c",
        type=float,
        help="Override the compressor thermostat deadband for every home",
    )
    parser.add_argument(
        "--heating-deadband-offset",
        type=float,
        help="Override the compressor deadband offset fraction (0.5 centers it)",
    )
    parser.add_argument(
        "--compressor-min-on-minutes",
        type=float,
        help="Override compressor minimum on-time for every home",
    )
    parser.add_argument(
        "--compressor-min-off-minutes",
        type=float,
        help="Override compressor minimum off-time for every home",
    )
    parser.add_argument(
        "--defrost-model",
        choices=("Legacy", "Discrete"),
        help="Override the configured heat-pump defrost model",
    )
    parser.add_argument(
        "--defrost-control-type",
        choices=("Auto", "Timer", "Demand"),
        help="Override the configured discrete-defrost control type",
    )
    parser.add_argument(
        "--defrost-er-strategy",
        choices=("Aggressive", "Conservative"),
        help="Override full-bank versus first-stage ER during discrete defrost",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override the configured output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_configuration(args.config_directory, repo_root=REPO_ROOT)
    if args.heating_deadband_c is not None:
        if args.heating_deadband_c <= 0:
            raise ValueError("--heating-deadband-c must be positive")
        config = replace(
            config,
            homes=tuple(
                replace(home, heating_deadband_c=args.heating_deadband_c)
                for home in config.homes
            ),
        )
    if args.heating_deadband_offset is not None:
        if not 0 <= args.heating_deadband_offset <= 1:
            raise ValueError("--heating-deadband-offset must be between zero and one")
        config = replace(
            config,
            homes=tuple(
                replace(home, heating_deadband_offset=args.heating_deadband_offset)
                for home in config.homes
            ),
        )
    for argument, field, flag in (
        (
            args.compressor_min_on_minutes,
            "compressor_min_on_time_min",
            "--compressor-min-on-minutes",
        ),
        (
            args.compressor_min_off_minutes,
            "compressor_min_off_time_min",
            "--compressor-min-off-minutes",
        ),
    ):
        if argument is not None:
            if argument < 0:
                raise ValueError(f"{flag} cannot be negative")
            config = replace(
                config,
                homes=tuple(replace(home, **{field: argument}) for home in config.homes),
            )
    if args.defrost_model is not None:
        config = replace(config, defrost_model=args.defrost_model)
    if args.defrost_control_type is not None:
        config = replace(config, defrost_control_type=args.defrost_control_type)
    if args.defrost_er_strategy is not None:
        config = replace(config, defrost_er_strategy=args.defrost_er_strategy)
    if args.output_directory is not None:
        output_directory = args.output_directory
        if not output_directory.is_absolute():
            output_directory = REPO_ROOT / output_directory
        config = replace(config, output_directory=output_directory.resolve())
    NeighborhoodSimulation(config).run_independent()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

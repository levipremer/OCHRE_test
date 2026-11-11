#!/usr/bin/env python3
"""Run named controller variants from one experiment configuration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.single_home_hems.single_home_hems.experiment_config import load_experiment
from examples.single_home_hems.single_home_hems.experiment_runner import run_experiment
from examples.single_home_hems.single_home_hems.controller_suite import (
    load_controller_suite, run_controller_suite,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument("--experiment-config", type=Path)
    configuration.add_argument(
        "--controller-config", type=Path,
        help="Select existing controller runners using one JSON configuration",
    )
    parser.add_argument("--variants", nargs="+", help="Named variants to execute; default: every explicitly listed variant")
    parser.add_argument("--homes", nargs="+")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Check configurations and model artifacts without simulating")
    args = parser.parse_args(argv)
    if args.controller_config is not None:
        suite = load_controller_suite(
            args.controller_config, ROOT, case_names=args.variants,
            home_ids=args.homes, output_directory=args.output_directory,
        )
        run_controller_suite(suite, validate_only=args.validate_only)
        print(
            f"Validated controller cases: {', '.join(case.name for case in suite.cases)}"
            if args.validate_only else
            f"Controller cases saved to {suite.output_directory}"
        )
        return 0
    experiment_config = args.experiment_config or Path(__file__).parent/"config"/"experiment.json"
    experiment = load_experiment(experiment_config, ROOT, variant_names=args.variants,
        home_ids=args.homes, output_directory=args.output_directory.resolve() if args.output_directory else None)
    run_experiment(experiment, validate_only=args.validate_only)
    print("Experiment validated; no simulations executed." if args.validate_only else
          f"Experiment comparisons saved to {experiment.variants[0].unified.neighborhood.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

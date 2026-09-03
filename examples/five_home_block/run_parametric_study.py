#!/usr/bin/env python3
"""Run the isolated seasonal electrification parametric study."""

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

from five_home_block.parametric_study import (  # noqa: E402
    ParametricStudyRunner,
    load_study_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-home seasonal appliance-electrification study"
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=EXAMPLE_ROOT / "parametric_config" / "study.json",
        help="Study-only JSON configuration",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override the study output directory without changing its configuration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun profiles even when valid cached Parquet files exist",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run three representative six-hour profiles instead of the full study",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_study_configuration(args.study_config, repo_root=REPO_ROOT)
    if args.output_directory is not None:
        output = args.output_directory
        if not output.is_absolute():
            output = REPO_ROOT / output
        config = replace(config, output_directory=output.resolve())
    runner = ParametricStudyRunner(config)
    summary = (
        runner.run_smoke_test(force=True)
        if args.smoke_test
        else runner.run(force=args.force)
    )
    print("\nParametric study complete")
    print(f"  Smoke test: {summary.smoke_test}")
    print(f"  Profiles: {summary.profile_count}")
    print(f"  Cached profiles reused: {summary.cached_profile_count}")
    print(f"  Scenarios: {summary.scenario_count}")
    print(f"  Output: {summary.output_directory}")
    if summary.report_file is not None:
        print(f"  Report: {summary.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

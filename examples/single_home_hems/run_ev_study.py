#!/usr/bin/env python3
"""Stage an EV study by default; execution always requires an explicit flag."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.single_home_hems.single_home_hems.ev_study import stage_study, run_study, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, default=Path(__file__).parent/"config/ev_study_pilot.json")
    parser.add_argument("--output-directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly execute selected staged jobs")
    mode.add_argument("--smoke-test", action="store_true", help="Separate two-config validation: 15-minute warm-up + 30-minute evaluation")
    parser.add_argument("--configurations", nargs="+")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--no-progress", action="store_true", help="Suppress job/case/timestep progress (not OCHRE messages)")
    args = parser.parse_args(argv)
    manifest = stage_study(args.study_config, ROOT, output_directory=args.output_directory, smoke=args.smoke_test)
    doc = read_json(manifest)
    print(f"Staged {len(doc['cases'])} cases / {len(doc['jobs'])} configurations: {manifest}")
    if args.run or args.smoke_test:
        run_study(manifest, configuration_ids=args.configurations, case_ids=args.cases,
                  progress=not args.no_progress)
        print("Selected study jobs complete; reports and trajectories saved.")
    else:
        print("No simulations executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

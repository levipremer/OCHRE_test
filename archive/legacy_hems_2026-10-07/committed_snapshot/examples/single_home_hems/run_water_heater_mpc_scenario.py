#!/usr/bin/env python3
"""Run one scenario MPC variant with the standard simulation data and figures.

Defaults: MILP, daily_budget_high_rf_1, RC model, 72 hours, 12-hour horizon,
and the previously selected persistence weights. Every invocation creates a
new folder under outputs/water_heater_mpc_scenario; existing runs are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.single_home_hems import run_simulation


OUTPUT_ROOT = EXAMPLE_ROOT / "outputs" / "water_heater_mpc_scenario"
# Previously selected MILP/convex + persistence operating points.
DEFAULT_WEIGHTS = {
    "milp": {"comfort": 0.1, "legionella": 0.001, "persistence": 1.0},
    "convex": {"comfort": 0.01, "legionella": 0.01, "persistence": 0.001},
}


def run_title(value: str) -> str:
    """Keep custom titles as a single, readable directory name."""
    value = value.strip()
    if (
        value in {"", ".", ".."}
        or Path(value).name != value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise argparse.ArgumentTypeError("The title must be a nonempty folder name without path separators")
    return value


def create_run_directory(root: Path, title: str) -> Path:
    """Allocate title, title_2, ... atomically without overwriting a run."""
    title = run_title(title)
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = root / (title if number == 1 else f"{title}_{number}")
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
        else:
            return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = run_simulation.build_argument_parser()
    parser.description = __doc__
    parser.add_argument(
        "--optimization", dest="water_heater_optimization", choices=("milp", "convex"),
        default=argparse.SUPPRESS,
        help="Scenario MPC formulation (alias for --water-heater-optimization; default: milp)",
    )
    parser.add_argument(
        "--title", "--run-title", type=run_title,
        help="Run folder title (default: milp or convex; repeated titles gain _2, _3, ...)",
    )
    parser.set_defaults(
        controller="water-heater", water_heater_model="rc",
        forecast_type="scenario", water_heater_optimization="milp",
        water_heater_solver=None, duration_hours=72, time_step_minutes=1,
        water_draw_file=EXAMPLE_ROOT / "external_data" / "water_flow_dchouse_l_per_min.csv",
        water_draw_start=dt.datetime(2025, 11, 6),
        water_heater_cycle_limit="off", water_heater_dwell_constraints="off",
    )
    parser.epilog = (
        "Scenario runner defaults override the general runner: water-heater controller, "
        "RC model, scenario forecasts, 72 hours, cycle/dwell limits off; GUROBI for MILP "
        "or CLARABEL for convex. All standard figures are enabled. Use --title to name "
        "a run; its location is always outputs/water_heater_mpc_scenario/TITLE."
    )
    args = parser.parse_args(argv)
    if args.controller != "water-heater" or args.forecast_type != "scenario":
        parser.error("This runner requires --controller water-heater and --forecast-type scenario")
    if args.output_directory is not None:
        parser.error("Use --title to name a run under outputs/water_heater_mpc_scenario")
    if args.water_heater_solver is None:
        args.water_heater_solver = "GUROBI" if args.water_heater_optimization == "milp" else "CLARABEL"
    for name, value in DEFAULT_WEIGHTS[args.water_heater_optimization].items():
        attribute = f"water_heater_{name}_weight"
        if getattr(args, attribute) is None:
            setattr(args, attribute, value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    title = args.title or args.water_heater_optimization
    args.output_directory = create_run_directory(OUTPUT_ROOT, title)
    (args.output_directory / "run_settings.json").write_text(
        json.dumps(
            {**vars(args), "title": title, "comfort_scenario_aggregation": "max"},
            indent=2, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Scenario run: {title}\nOutput directory: {args.output_directory}", flush=True)
    return run_simulation.main(args)


if __name__ == "__main__":
    raise SystemExit(main())

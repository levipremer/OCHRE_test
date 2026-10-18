"""Run matched EV experiments. No overlay means unchanged native baseline EVs."""
import argparse
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.five_home_block.five_home_block.config import load_configuration
from examples.five_home_block.five_home_block.ev_config import EVOptions
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation, select_home


def apply_ev_overlay(home, path):
    """Overlay only named vehicles; never mutate existing home/configuration files."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - {ev.vehicle_id for ev in home.evs}
    if unknown:
        raise ValueError(f"EV overlay contains unknown vehicle IDs: {sorted(unknown)}")
    evs = []
    for ev in home.evs:
        if ev.vehicle_id in data:
            values = dict(data[ev.vehicle_id])
            if "hems" in values:
                values["hems"] = EVOptions(**values["hems"])
            ev = replace(ev, **values)
        evs.append(ev)
    return replace(home, evs=tuple(evs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-directory", type=Path, default=ROOT/"examples/five_home_block/config")
    parser.add_argument("--home-id", default="home_01")
    parser.add_argument("--ev-config", type=Path, help="Optional per-vehicle JSON overlay; explicit opt-in")
    parser.add_argument("--duration-hours", type=float, default=72)
    parser.add_argument("--start-time", type=dt.datetime.fromisoformat)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_configuration(args.config_directory, ROOT)
    home = select_home(config, args.home_id)
    if args.ev_config:
        home = apply_ev_overlay(home, args.ev_config)
    config = replace(config, homes=(home,), duration=dt.timedelta(hours=args.duration_hours),
                     start_time=args.start_time or config.start_time,
                     output_directory=args.output_directory.resolve())
    config.validate()
    SingleHomeHEMSSimulation(config, home, HEMSController()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Archived overlay CLI; use the unified runner in the active example instead."""


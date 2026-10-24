"""Read-only retrofit plotter: saved evaluation SOC, realized mode and tariff.

No simulation imports or reruns. PNG/SVG outputs default outside frozen attempt
directories. Small mode offsets separate coincident categorical trajectories;
they do not represent different physical modes. Prices must agree across cases.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASES = (
    ("advanced", "V2H", "#0072B2", "-", .08),
    ("non_v2h_twin", "Non-V2H", "#D55E00", "--", 0),
    ("all_baseline", "Baseline", "#6B7280", ":", -.08),
)


def read_series(path):
    """Require an ordered, unique timestamp index; never silently fill gaps."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or len(frame) < 2):
        raise ValueError(f"Expected at least two ordered, unique timestamps: {path}")
    return frame


def actual_mode(raw, equipment):
    """Charge/idle/discharge from measured AC power; away takes precedence."""
    power = pd.to_numeric(raw[f"{equipment} Electric Power (kW)"], errors="raise")
    parked = next((raw[c] for c in (f"{equipment} Parked", f"{equipment} Parked (-)") if c in raw), None)
    if parked is not None:
        parked = parked.astype(str).str.lower().map({"true": True, "false": False,
                                                   "1": True, "0": False, "1.0": True, "0.0": False})
    elif f"{equipment} Mode" in raw:
        modes = raw[f"{equipment} Mode"]
        if modes.isna().any():
            raise ValueError("Missing actual EV availability")
        parked = modes.str.lower().ne("away")
    else:
        raise ValueError("Actual mode requires saved EV availability or native Mode")
    if parked.isna().any() or not np.isfinite(power).all():
        raise ValueError("Invalid actual EV availability/power")
    state = np.select([~parked, power > 1e-7, power < -1e-7], [-2, 1, -1], default=0)
    return pd.Series(state, index=raw.index)


def load_cases(run_directory, home_id, vehicle_id):
    """Load evaluation-only trajectories and preserve their actual time grid."""
    cases = {}
    equipment = f"EV_{vehicle_id}"
    reference_index, reference_price = None, None
    for case, *_ in CASES:
        directory = Path(run_directory)/home_id/case/"data"
        if case == "non_v2h_twin" and not directory.exists():
            continue
        raw = read_series(directory/"raw_results.csv")
        commands = read_series(directory/"controls.csv")
        if not raw.index.equals(commands.index):
            raise ValueError(f"{case}: state and price timestamps do not match")
        differences = raw.index[1:]-raw.index[:-1]
        step = differences[0]
        if not (differences == step).all():
            raise ValueError(f"{case}: missing or nonuniform plant timesteps")
        price = pd.to_numeric(commands.electricity_price_usd_per_kwh, errors="raise")
        if not np.isfinite(price).all():
            raise ValueError(f"{case}: missing/nonfinite electricity prices")
        if reference_index is not None:
            if not raw.index.equals(reference_index):
                raise ValueError("Comparison cases must have identical evaluation timestamps")
            if not np.allclose(price, reference_price, rtol=0, atol=1e-10):
                raise ValueError("Cases have different tariffs; cannot use one shared price axis")
        reference_index, reference_price = raw.index, price
        start_col, end_col = f"{equipment} SOC (-)", f"{equipment} End SOC (-)"
        # Saved SOC is the interval-start state; append the final endpoint when
        # available rather than plotting end-of-step SOC one timestep too early.
        if start_col in raw:
            soc = raw[start_col].copy()
            if end_col in raw:
                soc.loc[raw.index[-1]+step] = raw[end_col].iloc[-1]
        elif end_col in raw:
            soc = raw[end_col].copy()
            soc.index = soc.index+step
        else:
            raise ValueError(f"{case}: no saved SOC for {equipment}")
        if not np.isfinite(soc).all():
            raise ValueError(f"{case}: nonfinite SOC")
        cases[case] = dict(soc=soc, mode=actual_mode(raw, equipment), price=price, step=step)
    return cases


def build_figure(cases, *, title="EV operation", start=None, end=None, price_alpha=.22):
    """Two shared-time panels; categorical modes are offset only for legibility."""
    if not 0 <= price_alpha <= 1:
        raise ValueError("price_alpha must be between 0 and 1")
    reference = cases["advanced"]
    first = reference["mode"].index[0]
    last = reference["mode"].index[-1]+reference["step"]
    left, right = pd.Timestamp(start) if start else first, pd.Timestamp(end) if end else last
    if not first <= left < right <= last:
        raise ValueError(f"Requested window must be within {first} to {last}")
    fig, (soc_ax, mode_ax) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                                         gridspec_kw={"height_ratios": [1.6, 1]}, layout="constrained")
    price_ax = mode_ax.twinx()
    price_ax.set_zorder(mode_ax.get_zorder()-1)
    mode_ax.patch.set_visible(False)
    for case, label, color, linestyle, offset in CASES:
        if case not in cases:
            continue
        data = cases[case]
        soc_ax.plot(data["soc"].index, 100*data["soc"], color=color, linestyle=linestyle,
                    linewidth=1.7, label=label, zorder=3-offset)
        mode = data["mode"]
        edges = mode.index.append(pd.DatetimeIndex([mode.index[-1]+data["step"]]))
        mode_ax.step(edges, np.r_[mode.values, mode.iloc[-1]]+offset, where="post",
                     color=color, linestyle=linestyle, linewidth=1.1)
    prices = reference["price"]
    edges = prices.index.append(pd.DatetimeIndex([last]))
    price_ax.step(edges, np.r_[prices.values, prices.iloc[-1]], where="post",
                  color="#292D32", alpha=price_alpha, linewidth=1.8, label="Electricity price")
    price_ax.set_ylabel("Electricity price ($/kWh)", color="#707070")
    price_ax.tick_params(axis="y", colors="#707070")
    price_ax.margins(y=.15)
    price_ax.legend(loc="upper right", frameon=False, fontsize=9)
    soc_ax.set(title=title, ylabel="SOC (%)", ylim=(-2, 102))
    soc_ax.legend(loc="upper right", ncol=3, frameon=False)
    mode_ax.set(yticks=[-2, -1, 0, 1], yticklabels=["Away", "Discharge", "Idle", "Charge"],
                ylim=(-2.32, 1.42), ylabel="Actual mode", xlabel="Simulation time", xlim=(left, right))
    mode_ax.text(.005, .015, "Small vertical offsets separate overlapping modes.",
                 transform=mode_ax.transAxes, fontsize=8, color="#707070")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    mode_ax.xaxis.set_major_locator(locator)
    mode_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in (soc_ax, mode_ax):
        axis.grid(axis="y", alpha=.18, linewidth=.7)
        axis.grid(axis="x", alpha=.10, linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    price_ax.spines[["top", "left"]].set_visible(False)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Completed unified run/attempt folder")
    parser.add_argument("--home-id", default="home_01")
    parser.add_argument("--vehicle-id", default="01A")
    parser.add_argument("--start", help="Optional visible window start (ISO timestamp)")
    parser.add_argument("--end", help="Optional visible window end (ISO timestamp)")
    parser.add_argument("--title", default="EV SOC and realized operation")
    parser.add_argument("--price-alpha", type=float, default=.22)
    parser.add_argument("--output-directory", type=Path, help="Must be outside the frozen run folder")
    args = parser.parse_args(argv)
    run = args.run_directory.resolve()
    if run.parent.parent.name == "r":
        default = run.parents[2]/"retrofit_plots"/f"{run.parent.name}_{run.name}"
    else:
        default = run.parent/"retrofit_plots"/run.name
    output = (args.output_directory or default).resolve()
    if output == run or run in output.parents:
        parser.error("Keep retrofit outputs outside the frozen run directory")
    paths = [output/f"{args.home_id}_EV_{args.vehicle_id}_clean.{ext}" for ext in ("png", "svg")]
    if any(path.exists() for path in paths):
        parser.error("Plot already exists; choose a new --output-directory")
    cases = load_cases(run, args.home_id, args.vehicle_id)
    fig = build_figure(cases, title=args.title, start=args.start, end=args.end, price_alpha=args.price_alpha)
    try:
        output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            fig.savefig(path, dpi=200)
            print(path)
    finally:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

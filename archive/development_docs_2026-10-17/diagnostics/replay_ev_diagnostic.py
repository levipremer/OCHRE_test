"""Replay saved EV OCPs without rerunning or modifying the original study.

Uses the pre-change controller snapshot and frozen inputs. Logs native Gurobi
progress, actual master candidates, and a rainflow supporting-plane slice.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Historical utility: requires the frozen study and source_before snapshots.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.single_home_hems.single_home_hems.control_config import load_unified
from examples.single_home_hems.single_home_hems.ev_study import restore_schedule
from examples.single_home_hems.single_home_hems.forecast import Forecast
from examples.single_home_hems.single_home_hems.backed_loads import backed_power

STUDY = ROOT / "examples/single_home_hems/outputs/ev30_janfeb_nmc_v2"
OUT = ROOT / "examples/single_home_hems/outputs/ev_diagnostic_20261015"


def read(path):
    return json.loads(path.read_text())


def replay(job, timestamp, label):
    # Local package alias permits original relative imports without replacing
    # the production controller or altering its on-disk formulation.
    name = "examples.single_home_hems.single_home_hems.controllers.ev_audit_original"
    spec = importlib.util.spec_from_file_location(name, OUT / "source_before/ev.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model_name = "ev_audit_original_degradation"
    model_spec = importlib.util.spec_from_file_location(model_name, OUT / "source_before/ev_degradation.py")
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules[model_name] = model_module
    model_spec.loader.exec_module(model_module)
    module.BatteryDegradation = model_module.BatteryDegradation
    case = job.split("__")[0]
    home_file = STUDY / "cases" / case / "home.json"
    home_doc = read(home_file)
    # Explicit in-memory schema migration ONLY for the frozen old experiment.
    # The old model used the same endpoint for calibration and economics.
    for home_doc_row in home_doc["homes"].values():
        for vehicle_doc in home_doc_row["evs"].values():
            physical = vehicle_doc["physical"]
            physical.setdefault("initial_soh", 1.0)  # Frozen experiment used static nominal capacity.
            physical.setdefault("calibration_end_of_life_soh", physical["end_of_life_soh"])
    original_read = Path.read_text
    def migrated_read(path, *args, **kwargs):
        return json.dumps(home_doc) if path == home_file else original_read(path, *args, **kwargs)
    with patch.object(Path, "read_text", migrated_read):
        config = load_unified(home_file, STUDY / "jobs" / job / "control.json", ROOT)
    home = config.neighborhood.homes[0]
    ev = home.evs[0]
    run = STUDY / read(STUDY / "jobs" / job / "status.json")["run_directory"] / home.home_id
    def frame(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = frame(run / "advanced/data/raw_results_all_phases.csv")
    twin = frame(run / "non_v2h_twin/data/raw_results_all_phases.csv")
    prices = frame(run / "non_v2h_twin/data/controls_all_phases.csv")
    mobility = {k: restore_schedule(v) for k, v in read(STUDY / "cases" / case / "mobility.json").items()}
    forecast = Forecast(prices[["electricity_price_usd_per_kwh"]], ev_schedules=mobility,
                        backed_load_kw=pd.Series(backed_power(twin, config.home_settings[home.home_id]["backed_loads"]), index=twin.index))
    c = module.EVMPCController(ev)
    t = pd.Timestamp(timestamp)
    step = raw.index[1] - raw.index[0]
    prefix = f"EV_{ev.vehicle_id}"
    previous_end = ev.initial_soc_fraction
    previous_power = 0.
    for moment, row in raw.loc[:t].iterrows():
        c._observe(moment, previous_power, step)
        c.soc_history.extend([previous_end, row[prefix + " SOC (-)"]])
        c.soc_history = np.asarray(c.soc_history)[module.turning_points(c.soc_history)].tolist()
        previous_end = row[prefix + " End SOC (-)"]
        previous_power = row[prefix + " Electric Power (kW)"]
    plans, captures = {}, []
    original_build = c.build_problem
    def build(*args):
        problem, plan = original_build(*args)
        plans.update(plan)
        return problem, plan
    c.build_problem = build
    def verbose_solve(problem, solver, options=None, *, time_limit_seconds=300):
        opts = dict(options or {})
        opts.update(TimeLimit=time_limit_seconds, OutputFlag=1, LogToConsole=0,
                    LogFile=str(OUT / f"{label}_master_{len(captures):02d}.log"))
        start = time.perf_counter()
        problem.solve(solver=solver, verbose=False, **opts)
        model = problem.solver_stats.extra_stats
        problem.hems_diagnostic = dict(solver=solver, limit_seconds=time_limit_seconds,
            solve_call_seconds=time.perf_counter()-start,
            backend_solve_seconds=model.Runtime, timed_out=model.Status == 9,
            limit_reached=problem.status == cp.USER_LIMIT)
        x = plans["state"].value
        capture = dict(status=problem.status, native_status=model.Status,
            objective=problem.value, bound=model.ObjBound,
            gap=model.MIPGap, nodes=model.NodeCount, seconds=model.Runtime,
            soc=x.tolist() if x is not None else None,
            theta=float(plans["rainflow_theta"].value) if plans["rainflow_theta"] is not None else None)
        captures.append(capture)
        print(label, "master", len(captures), capture["status"], capture["seconds"], flush=True)
    module.solve_ocp = verbose_solve
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c._solve(t, float(raw.loc[t, prefix + " SOC (-)"]), forecast)
    result = dict(job=job, timestamp=str(t), source="pre-change controller, frozen study configuration",
        warnings=[str(w.message) for w in caught], diagnostics=c.solve_history[-1], masters=captures)
    (OUT / f"{label}.json").write_text(json.dumps(result, indent=2, default=str))
    if plans["rainflow_theta"] is not None and captures and captures[0]["soc"] is not None:
        battery = plans["battery"]
        history = np.asarray(c.soc_history)
        past = battery.rainflow_cost_gradient(history)[0]
        x0 = np.asarray(captures[0]["soc"])
        cost, gradient = battery.rainflow_cost_gradient(np.r_[history[:-1], x0])
        cost -= past
        gradient = gradient[len(history)-1:]
        # A geometric SOC-space slice, not a claim that all points are feasible
        # MILP trajectories. Initial SOC is held fixed.
        direction = .08 * (.5-x0)
        direction[0] = 0
        alpha = np.linspace(-.2, 1, 201)
        paths = [x0+a*direction for a in alpha]
        valid = np.array([np.all((x >= -1e-8) & (x <= 1+1e-8)) for x in paths])
        alpha = alpha[valid]
        exact = [battery.rainflow_cost_gradient(np.r_[history[:-1], x])[0]-past
                 for x, ok in zip(paths, valid) if ok]
        cut = cost+alpha*(gradient @ direction)
        assert np.max(cut-np.asarray(exact)) < 1e-7
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        ax = axes[1]
        ax.plot(alpha, exact, label="Exact incremental rainflow cost", lw=2)
        ax.plot(alpha, cut, "--", label="First supporting cut")
        ax.scatter([0], [cost], color="black", zorder=4, label="First master SOC plan / cut contact")
        movement = max(np.max(np.abs(np.asarray(m["soc"])-x0)) for m in captures if m["soc"] is not None)
        ax.set(title="Supporting plane: local SOC-space slice",
               xlabel="Slice parameter (0 = first plan)",
               ylabel="Incremental degradation cost ($)")
        ax.legend(fontsize=8)
        hours = (plans["boundaries"]-t).total_seconds()/3600
        axes[0].plot(hours, x0, label="First master plan")
        if len(captures) > 1:
            axes[0].plot(hours, captures[1]["soc"], ":", alpha=.7, label="Second master candidate")
        axes[0].plot(hours, c.plans[-1]["soc"], "--", label="Accepted plan")
        axes[0].set(xlabel="Forecast hours", ylabel="SOC", title=f"{len(captures)} masters; max candidate change {movement:.2g}")
        axes[0].legend(fontsize=8)
        axes[2].plot(alpha, np.asarray(exact)-cut, label="Exact minus first cut")
        tolerance = c.options.rainflow_absolute_tolerance + c.options.rainflow_relative_tolerance*np.abs(exact)
        axes[2].plot(alpha, tolerance, "--", label="Rainflow dollar tolerance")
        axes[2].set(xlabel="Slice parameter", ylabel="Underestimation ($)", title="Cut is exact at its contact point")
        axes[2].legend(fontsize=8)
        fig.suptitle(f"{job}, {timestamp}\nActual replay plans; geometric slice points need not be MILP-feasible", fontsize=11)
        fig.savefig(OUT / f"{label}_supporting_cut.png", dpi=170)
        plt.close(fig)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("all", "rainflow", "rainflow_one_cut", "slow_milp"), default="all")
    selected = parser.parse_args().sample
    OUT.mkdir(parents=True, exist_ok=True)
    if selected in ("all", "rainflow"):
        replay("jan_41_soc_0p4__milp_rainflow", "2018-01-02 00:00:00", "rainflow")
    if selected in ("all", "rainflow_one_cut"):
        replay("jan_41_soc_0p4__milp_rainflow", "2018-01-02 00:15:00", "rainflow_one_cut")
    if selected in ("all", "slow_milp"):
        replay("feb_41_soc_0p4__milp_comfort", "2018-02-02 03:30:00", "slow_milp")

"""Evaluation-only cross-formulation plots, kept separate for each control case."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ev_metrics import ev_operating_state
from .unified_plots import _finish, plot_metric_bars, FORECAST_METRICS, SAVINGS_METRICS


def _factorial_scores(data, metrics, title, path):
    """Average situations ONLY: count is x, selector is line, objective is fixed."""
    metrics = [m for m in metrics if m in data]
    if not metrics:
        return
    fig, axes = plt.subplots((len(metrics)+2)//3, 3, figsize=(16, 3.8*((len(metrics)+2)//3)), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        for selector, rows in data.groupby("scenario_selection"):
            group = rows.groupby("scenario_count")[metric]
            mean, sd = group.mean(), group.std().fillna(0)
            valid = mean.notna()
            axis.errorbar(mean.index[valid], mean[valid], yerr=sd[valid], marker="o", capsize=3, label=selector)
        axis.set_xticks(sorted(data.scenario_count.unique()))
        axis.set_xlabel("Requested scenario count")
        axis.set_title(metric.replace("forecast_", "").replace("_", " "), fontsize="small")
        axis.grid(alpha=.2)
        if data[metric].notna().sum() == 0:
            axis.text(.5, .5, "N/A", transform=axis.transAxes, ha="center")
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize="small")
    fig.suptitle(title+"\nMean ± sample SD across paired situations only")
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _horizon_scores(data, metrics, title, path):
    """One cell per (bank, horizon); aggregate paired situations, never factors.

    Most cells show means of run metrics. Maximum solve time is the worst
    observed solve; failures/no-scenario counts are totals over paired runs.
    """
    metrics = [m for m in metrics if m in data]
    if not metrics:
        return
    fig, axes = plt.subplots((len(metrics)+2)//3, 3,
                            figsize=(16, 4*((len(metrics)+2)//3)), squeeze=False)
    horizons, banks = sorted(data.horizon_hours.unique()), sorted(data.lookback_days.unique())
    for axis, metric in zip(axes.flat, metrics):
        rule = {"max_solve_seconds": "max", "failed_solves": "sum",
                "forecast_no_scenarios_count": "sum"}.get(metric, "mean")
        groups = data.groupby(["lookback_days", "horizon_hours"])[metric]
        values = (groups.sum(min_count=1) if rule == "sum" else groups.agg(rule))
        grid = values.unstack("horizon_hours").reindex(index=banks, columns=horizons)
        mesh = axis.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)),
                           cmap="cividis", aspect="auto")
        for y, bank in enumerate(banks):
            for x, horizon in enumerate(horizons):
                value = grid.loc[bank, horizon]
                axis.text(x, y, "N/A" if pd.isna(value) else f"{value:.3g}",
                          ha="center", va="center", fontsize="small",
                          bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        axis.set_xticks(range(len(horizons)), horizons)
        axis.set_yticks(range(len(banks)), [f"{b/7:g}" for b in banks])
        axis.set_xlabel("Forecast horizon (hours)")
        axis.set_ylabel("Historical bank (weeks)")
        axis.set_title(metric.replace("forecast_", "").replace("_", " ")+f" ({rule})", fontsize="small")
        fig.colorbar(mesh, ax=axis, shrink=.8)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    fig.suptitle(title+"\nPaired situations only; forecast target coverage varies with horizon")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_study(output, manifest, statuses, paired_metrics):
    """Keep original three-case plots; add variant overlays and paired summaries.

    Read only already-scored CSVs. Missing jobs are omitted from overlays and
    explicitly counted in titles; savings bars use the common completed cases.
    """
    output = Path(output)
    if {"objective_variant", "lookback_days", "horizon_hours"} <= set(paired_metrics):
        # Keep any additional scenario factors separate, including future grids.
        group_keys = [k for k in ("objective_variant", "scenario_count", "scenario_selection") if k in paired_metrics]
        for labels, rows in paired_metrics.groupby(group_keys):
            labels = labels if isinstance(labels, tuple) else (labels,)
            directory = output/"figures"/"__".join(map(str, labels))
            directory.mkdir(parents=True, exist_ok=True)
            for control_case in ("non_v2h_twin", "advanced"):
                case_rows = rows.loc[rows.case.eq(control_case)]
                title = f"{' / '.join(map(str, labels))} / {control_case} ({rows.study_case.nunique()} paired situations)"
                whole = case_rows.loc[case_rows.device.eq("whole_home")]
                if not whole.empty:
                    _horizon_scores(whole, SAVINGS_METRICS, title, directory/f"{control_case}_horizon_savings.png")
                for device, ev in case_rows.loc[case_rows.device.str.startswith("EV_")].groupby("device"):
                    groups = {
                        "service_aging": ("mean_departure_comfort_shortfall", "trip_unserved_kwh",
                            "reference_degradation_cost_usd", "cycle_count", "final_soc"),
                        "forecast_accuracy": FORECAST_METRICS,
                        "computation": ("mean_solve_seconds", "max_solve_seconds",
                            "mean_ocp_build_and_solve_seconds", "failed_solves", "forecast_no_scenarios_count"),
                    }
                    for label, metrics in groups.items():
                        _horizon_scores(ev, metrics, title+f" / {device}",
                                        directory/f"{control_case}_{device}_horizon_{label}.png")
        return
    # Do not pool objective/count/selector combinations, or mix twin/advanced.
    # Factorial summaries have no situation-specific trajectory overlays;
    # each individual run still retains its normal three-case trajectories.
    if {"objective_variant", "scenario_count", "scenario_selection"} <= set(paired_metrics):
        for objective, rows in paired_metrics.groupby("objective_variant"):
            directory = output/"figures"/objective
            directory.mkdir(parents=True, exist_ok=True)
            coverage = rows.study_case.nunique()
            for control_case in ("non_v2h_twin", "advanced"):
                case_rows = rows.loc[rows.case.eq(control_case)]
                whole = case_rows.loc[case_rows.device.eq("whole_home")]
                title = f"{objective} / {control_case} ({coverage} paired situations)"
                if not whole.empty:
                    _factorial_scores(whole, SAVINGS_METRICS, title, directory/f"{control_case}_savings.png")
                for device, ev_rows in case_rows.loc[case_rows.device.str.startswith("EV_")].groupby("device"):
                    _factorial_scores(ev_rows, FORECAST_METRICS, title+f" / {device}",
                                      directory/f"{control_case}_{device}_forecast_accuracy.png")
                    _factorial_scores(ev_rows, ("cycle_count", "mean_departure_comfort_shortfall",
                        "trip_unserved_kwh", "max_predicted_trip_shortfall_kwh", "failed_solves",
                        "mean_solve_seconds", "max_solve_seconds"),
                        title+f" / {device}", directory/f"{control_case}_{device}_outcomes.png")
        return
    directory = output/"figures"
    directory.mkdir(parents=True, exist_ok=True)
    complete = {s["job_id"]: s for s in statuses if s["state"] == "complete"}
    variants = {j["configuration_id"] for j in manifest["jobs"].values()}
    colors = {name: plt.get_cmap("turbo")(i/max(1, len(variants)-1)) for i, name in enumerate(sorted(variants))}
    for case_id in manifest["cases"]:
        for control_case in ("non_v2h_twin", "advanced"):
            trajectories = []
            for job_id, job in manifest["jobs"].items():
                if job["case_id"] != case_id or job_id not in complete:
                    continue
                root = output/complete[job_id]["run_directory"]/job["home_id"]
                data = root/control_case/"data"
                files = [data/f"{name}.csv" for name in ("raw_results", "controls", "standardized")]
                if not all(p.exists() for p in files):
                    continue
                raw, commands, mapped = [pd.read_csv(p, index_col=0, parse_dates=True) for p in files]
                trajectories.append((job["configuration_id"], job["vehicle_id"], raw, commands, mapped, root))
            if not trajectories:
                continue
            fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
            for label, vehicle_id, raw, commands, mapped, _ in trajectories:
                name = f"EV_{vehicle_id}"
                mode = ev_operating_state(raw, name)
                color = colors[label]
                axes[0].step(raw.index, mode.ne("away").astype(int), where="post", color=color, label=label)
                axes[1].plot(raw.index, raw[f"{name} End SOC (-)"], color=color, label=label)
                axes[2].plot(commands.index, commands.electricity_price_usd_per_kwh, color=color, label=label)
                axes[3].step(raw.index, raw[f"{name} Electric Power (kW)"], where="post", color=color, label=label)
                axes[4].step(raw.index, mode.map({"away": -2, "discharging": -1, "idle": 0, "charging": 1}),
                             where="post", color=color, label=label)
                axes[5].plot(mapped.index, mapped.total_electric_power_kw, color=color, label=label)
                # Show inventory movement alongside the separately adjusted savings.
                capacity = paired_metrics.loc[(paired_metrics.study_case == case_id) &
                    (paired_metrics.configuration_id == label) &
                    (paired_metrics.device == name), "capacity_kwh"] if "capacity_kwh" in paired_metrics else pd.Series(dtype=float)
                if f"{name} End Stored Energy (kWh)" in raw:
                    energy = raw[f"{name} End Stored Energy (kWh)"]-raw[f"{name} Start Stored Energy (kWh)"].iloc[0]
                    axes[6].plot(raw.index, energy, color=color, label=label)
                elif not capacity.empty:  # Archived static-SOH results only.
                    energy = (raw[f"{name} End SOC (-)"]-raw[f"{name} SOC (-)"].iloc[0])*capacity.iloc[0]
                    axes[6].plot(raw.index, energy, color=color, label=label)
            # Baseline is an unaveraged paired reference from the first available job.
            _, vehicle_id, _, _, _, root = trajectories[0]
            baseline_file = root/"all_baseline/data/raw_results.csv"
            if baseline_file.exists():
                baseline = pd.read_csv(baseline_file, index_col=0, parse_dates=True)
                name = f"EV_{vehicle_id}"
                for axis, column in ((axes[1], f"{name} End SOC (-)"), (axes[3], f"{name} Electric Power (kW)")):
                    axis.plot(baseline.index, baseline[column], "--", color="black", alpha=.7, label="all_baseline")
            for axis, label in zip(axes, ("Home / away", "SOC", "Price (USD/kWh)", "EV AC power (kW)",
                                          "Actual mode", "Whole-home power (kW)", "Stored energy change (kWh)")):
                axis.set_ylabel(label)
            axes[0].set_yticks([0, 1], ["Away", "Home"])
            axes[4].set_yticks([-2, -1, 0, 1], ["Away", "Discharge", "Idle", "Charge"])
            _finish(fig, axes, f"{case_id} / {control_case}: {len(trajectories)}/{len(variants)} formulations (evaluation only)",
                    directory/f"{case_id}_{control_case}_formulations.png", 100)

    if paired_metrics.empty:
        return
    for control_case in ("non_v2h_twin", "advanced"):
        data = paired_metrics.loc[paired_metrics.case.eq(control_case) & paired_metrics.device.eq("whole_home")]
        if not data.empty:
            plot_metric_bars(data, SAVINGS_METRICS, "configuration_id",
                f"{control_case}: paired whole-home savings (mean ± sample SD; dots = cases)",
                directory/f"{control_case}_formulation_savings.png")
        ev_rows = paired_metrics.loc[paired_metrics.case.eq(control_case) & paired_metrics.device.str.startswith("EV_")]
        if "forecast_issuance_count" in ev_rows:
            for device, rows in ev_rows.groupby("device"):
                if rows.forecast_issuance_count.notna().any():
                    plot_metric_bars(rows, FORECAST_METRICS, "configuration_id",
                        f"{control_case} / {device}: paired final-forecast scores (mean ± SD; dots = cases)",
                        directory/f"{control_case}_{device}_formulation_forecast_accuracy.png")

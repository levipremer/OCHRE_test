"""Evaluation-only cross-formulation plots, kept separate for each control case."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ev_metrics import ev_operating_state
from .unified_plots import _finish


def plot_study(output, manifest, statuses, paired_metrics):
    """Keep original three-case plots; add variant overlays and paired summaries.

    Read only already-scored CSVs. Missing jobs are omitted from overlays and
    explicitly counted in titles; savings bars use the common completed cases.
    """
    output = Path(output)
    directory = output/"figures"
    directory.mkdir(exist_ok=True)
    complete = {s["job_id"]: s for s in statuses if s["state"] == "complete"}
    variants = {j["configuration_id"] for j in manifest["jobs"].values()}
    colors = dict(zip(sorted(variants), plt.get_cmap("tab10").colors))
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
                # Stored energy change exposes unpriced evaluation-boundary energy.
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
        if data.empty:
            continue
        metrics = ["operating_cost_savings_usd", "degradation_cost_savings_usd", "net_savings_usd"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        labels = sorted(data.configuration_id.unique())
        for axis, metric in zip(axes, metrics):
            group = data.groupby("configuration_id")[metric]
            means, stds = group.mean().reindex(labels), group.std().reindex(labels).fillna(0)
            x = np.arange(len(labels))
            axis.bar(x, means, yerr=stds, capsize=4, color=[colors[v] for v in labels], alpha=.65)
            for i, label in enumerate(labels):
                values = data.loc[data.configuration_id.eq(label), metric]
                axis.scatter(np.full(len(values), i), values, color="black", s=15, zorder=3)
            axis.axhline(0, color="grey", linewidth=.8)
            axis.set_ylabel(metric.replace("_", " "))
            axis.grid(axis="y", alpha=.2)
        axes[0].set_title(f"{control_case}: paired whole-home savings (mean ± sample SD; dots = cases)")
        axes[-1].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(directory/f"{control_case}_formulation_savings.png", dpi=100)
        plt.close(fig)

"""Export EV-only objective costs without changing the MPC formulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EV_TERMS = (
    "electricity", "health_soc", "degradation", "comfort", "trip_shortfall", "persistence",
    "terminal", "modeled_home_peak", "doe_panel_violation", "doe_grid_violation",
)


def predicted_trip_energy_fields(path: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a solved path's trip demand and slack, independently of plant replay.

    Old plans and lightweight test paths may predate the shortfall expression;
    those paths served their full modeled trip demand and therefore have zero
    predicted shortfall. Do not substitute realized plant unserved-trip metrics.
    """
    horizon = path["horizon"]
    requested = np.asarray(horizon.trip_kwh, dtype=float).reshape(-1)
    expression = path.get("trip_shortfall_kwh")
    shortfall = (np.zeros_like(requested) if expression is None else np.asarray(
        getattr(expression, "value", expression), dtype=float,
    ).reshape(-1))
    served_expression = path.get("served_trip_kwh")
    served = (requested - shortfall if served_expression is None else np.asarray(
        getattr(served_expression, "value", served_expression), dtype=float,
    ).reshape(-1))
    if (shortfall.shape != requested.shape or served.shape != requested.shape
            or not np.isfinite(shortfall).all() or not np.isfinite(served).all()):
        raise ValueError("Solved EV trip energy and predicted shortfall have incompatible trajectories")
    return {
        "requested_trip_energy_kwh": requested.tolist(),
        "predicted_trip_shortfall_kwh": shortfall.tolist(),
        "predicted_served_trip_energy_kwh": served.tolist(),
        "trip_shortfall_indices": np.flatnonzero(requested > 0).tolist(),
        "total_requested_trip_energy_kwh": float(requested.sum()),
        "total_predicted_trip_shortfall_kwh": float(shortfall.sum()),
        "total_predicted_served_trip_energy_kwh": float(served.sum()),
    }


def objective_component_frames(
    plans: Sequence[Mapping[str, Any]], solves: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one aggregate row per issue and one row per saved EV scenario.

    Individual scenario terms are the weighted path costs before the EV
    controller takes a mean (or the maximum for comfort). The aggregate rows
    contain the terms that actually enter the solved objective.
    """
    by_issue = {str(pd.Timestamp(plan["timestamp"])): plan for plan in plans}
    if len(by_issue) != len(plans):
        raise ValueError("EV plans contain duplicate solve timestamps")
    objective_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for solve in solves:
        issue = str(pd.Timestamp(solve["timestamp"]))
        plan = by_issue.get(issue)
        successful = str(solve.get("successful", "False")).lower() == "true"
        if successful and plan is None:
            raise ValueError(f"Accepted EV solve at {issue} has no saved plan")
        terms = dict(plan.get("objective_terms", {})) if plan is not None else {}
        row = {
            "issued_at": issue,
            "status": solve.get("status"),
            "successful": successful,
            "fallback": str(solve.get("fallback", not successful)).lower() == "true",
            "objective_cost": float(plan["objective"]) if plan is not None else np.nan,
            "modeled_home_peak_epigraph_kw": (
                float(plan["modeled_home_peak_epigraph_kw"])
                if plan is not None and "modeled_home_peak_epigraph_kw" in plan else np.nan
            ),
        }
        for name in EV_TERMS:
            row[f"{name}_objective_cost"] = (
                float(terms[name]) if name in terms else
                0.0 if name == "trip_shortfall" and plan is not None else np.nan
            )
        demand_parts = plan.get("daily_demand_objective_terms", {}) if plan is not None else {}
        row["grid2_demand_formulation"] = plan.get("grid2_demand_formulation") if plan is not None else None
        row["daily_demand_aggregation"] = plan.get("daily_demand_aggregation") if plan is not None else None
        row["daily_demand_worst_scenario_index"] = plan.get("daily_demand_worst_scenario_index") if plan is not None else None
        row["daily_demand_dt_over_n"] = plan.get("daily_demand_scale", np.nan) if plan is not None else np.nan
        for name in ("on_peak", "anytime"):
            row[f"daily_demand_{name}_objective_cost"] = (
                float(demand_parts[name]) if name in demand_parts else np.nan
            )
        electricity_parts = plan.get("electricity_components", {}) if plan is not None else {}
        for name in ("fixed_backed_load_cost", "ev_charge_cost", "ev_discharge_credit"):
            row[f"electricity_{name}"] = (
                float(electricity_parts[name]) if name in electricity_parts else np.nan
            )
        row["component_sum"] = float(sum(terms.values())) if terms else np.nan
        row["objective_balance_error"] = (
            row["component_sum"] - row["objective_cost"] if plan is not None else np.nan
        )
        objective_rows.append(row)
        if plan is None:
            continue
        paths = plan.get("scenarios", [])
        count = len(paths)
        for index, path in enumerate(paths):
            path_terms = path.get("objective_terms", {})
            entry = {
                "issued_at": issue,
                "scenario_index": index,
                "scenario_id": path.get("candidate_id", str(index)),
                "scenario_probability": 1 / count,
                "aggregation_note": (
                    "mean except comfort=max; modeled_home_peak=max of complete per-scenario demand costs across both components and all days"
                    if plan.get("daily_demand_aggregation") == "worst_complete_scenario_v1" else
                    "mean except comfort=max; modeled_home_peak is dt/N times the sum of weekday daily on-peak and anytime maxima, each separately cross-scenario"
                    if plan.get("grid2_demand_formulation") == "weekday_daily_dt_over_n_v1"
                    and bool(plan.get("daily_demand_objective_terms")) else
                    "mean except comfort=max; modeled_home_peak is cross-scenario max"
                ),
            }
            for name in EV_TERMS:
                entry[f"{name}_path_cost"] = (
                    float(path_terms[name]) if name in path_terms else
                    0.0 if name == "trip_shortfall" else np.nan
                )
            if plan.get("daily_demand_cost_scenarios") is not None:
                entry["modeled_home_peak_path_cost"] = plan["daily_demand_cost_scenarios"][index]
            requested = np.asarray(path.get(
                "requested_trip_energy_kwh", path.get("trip_energy_kwh", []),
            ), dtype=float)
            shortfall = np.asarray(path.get(
                "predicted_trip_shortfall_kwh", np.zeros_like(requested),
            ), dtype=float)
            served = np.asarray(path.get(
                "predicted_served_trip_energy_kwh", requested - shortfall,
            ), dtype=float)
            entry.update(
                total_requested_trip_energy_kwh=float(requested.sum()),
                total_predicted_trip_shortfall_kwh=float(shortfall.sum()),
                total_predicted_served_trip_energy_kwh=float(served.sum()),
            )
            electricity_parts = path.get("electricity_components", {})
            for name in ("fixed_backed_load_cost", "ev_charge_cost", "ev_discharge_credit"):
                entry[f"electricity_{name}"] = (
                    float(electricity_parts[name]) if name in electricity_parts else np.nan
                )
            scenario_rows.append(entry)
    if len({row["issued_at"] for row in objective_rows}) != len(objective_rows):
        raise ValueError("EV solve history contains duplicate timestamps")
    return pd.DataFrame(objective_rows), pd.DataFrame(scenario_rows)


def export_saved_run(run_directory: Path) -> tuple[Path, Path]:
    """Backfill objective CSVs from an existing run's accepted plans and solves."""
    run_directory = Path(run_directory)
    plans = json.loads((run_directory / "data/EV_01A_plans.json").read_text())
    solves = pd.read_csv(run_directory / "ev_solve_history.csv").to_dict("records")
    aggregate, scenarios = objective_component_frames(plans, solves)
    aggregate_path = run_directory / "ev_objective_components.csv"
    scenario_path = run_directory / "ev_scenario_objective_components.csv"
    aggregate.to_csv(aggregate_path, index=False)
    scenarios.to_csv(scenario_path, index=False)
    return aggregate_path, scenario_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    for path in export_saved_run(args.run_directory):
        print(path)


if __name__ == "__main__":
    main()

"""Post-run scores for final issued mobility ensembles; never used by the OCP.

Adapted from scenario-based-EV-prediction-analysis/evaluation: home Brier and
false-state rates, energy envelope and next-new-trip errors. Driving demand is
distance-derived chemical kWh, not served energy or raw Audi SOC depletion.
"""
import numpy as np
import pandas as pd


STEP_METRICS = ("brier_home", "false_home", "false_away", "arrival_energy_mae_kwh")
HORIZON_METRICS = ("horizon_energy_mae_kwh", "energy_coverage", "energy_sharpness_kwh",
                   "next_trip_energy_mae_kwh", "next_trip_missing_fraction",
                   "next_trip_false_prediction_fraction")


def _energy(path, left, right, consumption):
    arrivals = path.trips
    return float(arrivals.loc[(arrivals.arrival >= left) &
                             (arrivals.arrival < right), "distance_km"].sum()) * consumption


def _next_trip(path, left, right, consumption):
    # Ignore the ongoing trip when issuance occurs away; zero-distance trips count.
    trips = path.trips.loc[(path.trips.departure >= left) & (path.trips.arrival < right)]
    return float(trips.iloc[0].distance_km) * consumption if len(trips) else np.nan


def score_forecasts(issued, actual, consumption, control_minutes, start, end):
    """Return issuance/lead tables and an evaluation-only scalar summary.

Each issued item is (timestamp, forecast duration, final paths). Timestamp and
target bins must lie within evaluation. Horizon/next-trip metrics require full
coverage; partial tails contribute only timestep scores, weighted by duration.
Absent ensembles are flagged, never treated as accurate forecasts.
"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    step = pd.Timedelta(minutes=control_minutes)
    records, leads = [], []
    for timestamp, horizon, paths in issued:
        timestamp = pd.Timestamp(timestamp)
        if not start <= timestamp < end:
            continue
        finish = timestamp + horizon
        stop = min(finish, end)
        if actual.start > timestamp or actual.end < stop:
            raise ValueError("Actual mobility does not cover forecast evaluation")
        row = dict(timestamp=timestamp, horizon_end=finish, scenario_count=len(paths),
                   no_scenarios_selected=not paths, full_horizon=finish <= end,
                   scored_timesteps=0, next_trip_observed=False)
        row.update({name: np.nan for name in (*STEP_METRICS, *HORIZON_METRICS)})
        local = []
        if paths:
            for target in pd.date_range(timestamp, stop, freq=step, inclusive="left"):
                right = min(target + step, stop)
                truth = bool(actual.available([target])[0])
                home = np.array([p.available([target])[0] for p in paths], dtype=float)
                energy = _energy(actual, target, right, consumption)
                predicted = np.array([_energy(p, target, right, consumption) for p in paths])
                detail = dict(timestamp=timestamp, target_time=target,
                    lead_minutes=(target-timestamp).total_seconds()/60,
                    duration_hours=(right-target).total_seconds()/3600, scenario_count=len(paths),
                    actual_home=int(truth), predicted_home_probability=float(home.mean()),
                    actual_arrival_energy_kwh=energy, predicted_arrival_energy_mean_kwh=float(predicted.mean()),
                    brier_home=float((home.mean()-truth)**2),
                    false_home=float(np.mean((home == 1) & ~np.bool_(truth))),
                    false_away=float(np.mean((home == 0) & np.bool_(truth))),
                    arrival_energy_mae_kwh=float(np.mean(np.abs(predicted-energy))))
                local.append(detail)
            leads.extend(local)
            weights = [d["duration_hours"] for d in local]
            row.update({m: float(np.average([d[m] for d in local], weights=weights)) for m in STEP_METRICS})
            row["scored_timesteps"] = len(local)
            if row["full_horizon"]:
                energy = _energy(actual, timestamp, finish, consumption)
                predicted = np.array([_energy(p, timestamp, finish, consumption) for p in paths])
                next_actual = _next_trip(actual, timestamp, finish, consumption)
                next_pred = np.array([_next_trip(p, timestamp, finish, consumption) for p in paths])
                observed = bool(np.isfinite(next_actual))
                row.update(actual_horizon_energy_kwh=energy,
                    horizon_energy_mae_kwh=float(np.mean(np.abs(predicted-energy))),
                    energy_coverage=float(predicted.min() <= energy <= predicted.max()),
                    energy_sharpness_kwh=float(np.ptp(predicted)), next_trip_observed=observed,
                    next_trip_energy_mae_kwh=float(np.mean(np.abs(np.nan_to_num(next_pred)-next_actual))) if observed else np.nan,
                    next_trip_missing_fraction=float(np.mean(np.isnan(next_pred))) if observed else np.nan,
                    next_trip_false_prediction_fraction=float(np.mean(np.isfinite(next_pred))) if not observed else np.nan)
        records.append(row)
    issuance = pd.DataFrame(records, columns=["timestamp", "horizon_end", "scenario_count",
        "no_scenarios_selected", "full_horizon", "scored_timesteps", "next_trip_observed",
        *STEP_METRICS, *HORIZON_METRICS, "actual_horizon_energy_kwh"])
    details = pd.DataFrame(leads, columns=["timestamp", "target_time", "lead_minutes",
        "duration_hours", "scenario_count", "actual_home", "predicted_home_probability",
        "actual_arrival_energy_kwh", "predicted_arrival_energy_mean_kwh", *STEP_METRICS])
    summary = dict(forecast_issuance_count=len(issuance),
        forecast_no_scenarios_count=int(issuance.no_scenarios_selected.sum()),
        forecast_no_scenarios_fraction=float(issuance.no_scenarios_selected.mean()) if len(issuance) else np.nan,
        forecast_scored_issuance_count=int(issuance.scored_timesteps.gt(0).sum()),
        forecast_scored_timestep_count=len(details),
        forecast_full_horizon_count=int((issuance.full_horizon & ~issuance.no_scenarios_selected).sum()))
    for metric in STEP_METRICS:
        summary["forecast_"+metric] = float(np.average(details[metric], weights=details.duration_hours)) if len(details) else np.nan
    for metric in HORIZON_METRICS:
        summary["forecast_"+metric] = float(issuance[metric].mean())
        summary["forecast_"+metric+"_count"] = int(issuance[metric].count())
    return issuance, details, summary

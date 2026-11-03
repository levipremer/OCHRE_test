"""EV baseline, convex and MILP MPC: formulation, optional costs and dispatch."""
from __future__ import annotations

from copy import deepcopy
import time
import warnings
from typing import Any, Mapping
import numpy as np
import pandas as pd
import cvxpy as cp
from .solver import solve_ocp, usable_solution
from dataclasses import dataclass
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation, turning_points
from ..control_forecasts import EVScenarioSet
from ..ev_scenarios import NoEVScenarios
from ..ev_hub_forecast import INTERVAL as HUB_FORECAST_INTERVAL
from ..ev_home_hub_trace import predicted_trip_energy_fields
from ..forecast import Forecast
from ..rates import (
    LOCAL_TIMEZONE, MISO_RTP_DOE, WINTER_TOU, WINTER_TOU_DEMAND,
    WINTER_TOU_DEMAND_CPP, WinterTOURate,
)


def daily_terminal_energy_value(forecast, timestamp):
    """v = full-calendar-day time-average tariff, valued per chemical kWh.

    Use the average directly, without charge/discharge efficiency factors.
    This approved value also prices stored inventory in economic reporting.
    It is an opportunity-value proxy, not acquisition
    accounting. Use minute-resolution tariff intervals, independent of the
    MPC horizon, mobility and backed load; never inspect future realized rows.
    """
    if forecast.tariff_price_at is None:
        raise ValueError("EV terminal energy value requires a known full-day tariff; "
                         "build Forecast.from_schedule with a tariff price callable")
    start = pd.Timestamp(timestamp).normalize()
    end = start + pd.DateOffset(days=1)
    times = pd.date_range(start, end, freq="min", inclusive="left")
    prices = np.array([forecast.tariff_price_at(t) for t in times], dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("EV terminal energy value requires finite full-day tariff prices")
    return float(prices.mean())


HISTORICAL_HOME_TERMINAL_METHOD = "historical_home_dow_30d"


def historical_home_terminal_energy_value(forecast, timestamp, mobility):
    """Known issue-day price weighted by past same-weekday home availability.

    The pool is the preceding 30 *completed local calendar days*. Each eligible
    weekday contributes its observed home fraction at each local clock minute;
    current-day and future realized availability are never queried. A repeated
    DST clock minute first averages its two observations within its own day.
    Missing DST clock minutes have no observation in that day. If the selected
    days lack validated coverage, any current-day clock minute has no historical
    observations, or all historical availability is zero, retain the existing
    full-day mean tariff. The terminal energy expression and its N scaling are
    unchanged.
    """
    issue = pd.Timestamp(timestamp)
    start = issue.normalize()
    end = start + pd.DateOffset(days=1)
    times = pd.date_range(start, end, freq="min", inclusive="left")
    # Same current-day known tariff used by the original daily-mean value. No
    # historical prices or future-day rates enter the availability weighting.
    unweighted = daily_terminal_energy_value(forecast, issue)
    prices = np.asarray([forecast.tariff_price_at(t) for t in times], dtype=float)
    days = [start - pd.DateOffset(days=offset) for offset in range(1, 31)
            if (start - pd.DateOffset(days=offset)).dayofweek == start.dayofweek]
    provenance = dict(
        method=HISTORICAL_HOME_TERMINAL_METHOD,
        history_days=30, history_cutoff=start.isoformat(),
        issue_day=start.date().isoformat(),
        issue_day_timezone=str(start.tz) if start.tz is not None else None,
        selected_local_dates=[day.date().isoformat() for day in days],
        historical_day_count=len(days),
        history_source=(dict(source=mobility.metadata.get("source"),
                             source_vehicle=mobility.metadata.get("source_vehicle"),
                             source_start=mobility.metadata.get("source_start"))
                        if mobility is not None else None),
        unweighted_daily_mean_usd_per_kwh=unweighted,
        value_usd_per_kwh=unweighted, fallback=True, fallback_reason=None,
        historical_home_fraction_by_clock_minute=None,
        historical_observation_count_by_clock_minute=None,
        known_issue_day_prices_usd_per_kwh=prices.tolist(),
        issue_day_clock_minutes=(times.hour * 60 + times.minute).tolist(),
    )
    if mobility is None:
        provenance["fallback_reason"] = "missing_historical_availability"
        return unweighted, provenance
    if (pd.Timestamp(mobility.start).tz is None) != (start.tz is None):
        provenance["fallback_reason"] = "historical_availability_clock_mismatch"
        return unweighted, provenance
    # Convert the history bounds into the issue's local clock for comparison;
    # EVSchedule.available receives aware instants and preserves trip semantics.
    left, right = pd.Timestamp(mobility.start), pd.Timestamp(mobility.end)
    missing = [day.date().isoformat() for day in days
               if left > day or right < day + pd.DateOffset(days=1)]
    if missing:
        provenance.update(fallback_reason="incomplete_historical_day_coverage",
                          missing_local_dates=missing)
        return unweighted, provenance
    observations = []
    for day in days:
        index = pd.date_range(day, day + pd.DateOffset(days=1), freq="min", inclusive="left")
        # Strict causal guard independent of the schedule's future extent.
        if len(index) and index[-1] >= start:
            raise ValueError("Terminal availability history cannot include the current day")
        home = np.asarray(mobility.available(index), dtype=float)
        if home.shape != (len(index),) or not np.isfinite(home).all() or not np.isin(home, (0, 1)).all():
            provenance["fallback_reason"] = "invalid_historical_availability"
            return unweighted, provenance
        clock = index.hour * 60 + index.minute
        observations.append(pd.Series(home, index=clock).groupby(level=0).mean().reindex(range(1440)))
    history = pd.concat(observations, axis=1)
    counts = history.notna().sum(axis=1).to_numpy(dtype=int)
    profile = history.mean(axis=1).to_numpy(dtype=float)
    clock = times.hour * 60 + times.minute
    weights = profile[clock]
    provenance.update(
        historical_home_fraction_by_clock_minute=[float(v) if np.isfinite(v) else None for v in profile],
        historical_observation_count_by_clock_minute=counts.tolist(),
        historical_home_minutes_by_local_date={day.date().isoformat(): float(values.sum())
                                               for day, values in zip(days, observations)},
    )
    if not np.isfinite(weights).all():
        provenance["fallback_reason"] = "missing_historical_clock_minutes"
        return unweighted, provenance
    home_minutes = float(weights.sum())
    provenance["weighted_home_hours"] = home_minutes / 60
    if home_minutes <= 0:
        provenance["fallback_reason"] = "zero_historical_home_availability"
        return unweighted, provenance
    value = float(np.dot(prices, weights) / home_minutes)
    provenance.update(value_usd_per_kwh=value, fallback=False)
    return value, provenance


def controller_terminal_energy_value(controller, forecast, timestamp):
    """Opt-in historical terminal value; preserve every other controller default."""
    method = getattr(controller, "terminal_energy_value_method", "daily_mean")
    if method == "daily_mean":
        value = daily_terminal_energy_value(forecast, timestamp)
        return value, dict(method="daily_mean", value_usd_per_kwh=value,
                           fallback=False, fallback_reason=None)
    if method != HISTORICAL_HOME_TERMINAL_METHOD:
        raise ValueError(f"Unknown EV terminal energy value method: {method}")
    mobility = getattr(controller, "_terminal_availability_history", None)
    cache = getattr(controller, "_terminal_energy_value_cache", None)
    if cache is None:
        cache = controller._terminal_energy_value_cache = {}
    start = pd.Timestamp(timestamp).normalize()
    key = (start.isoformat(), id(mobility), id(forecast.tariff_price_at), method)
    if key not in cache:
        # Retain one issue-day entry. Thirty-five coupled paths reuse one value
        # and the same provenance without rereading their scenario mobility.
        cache.clear()
        cache[key] = historical_home_terminal_energy_value(forecast, timestamp, mobility)
    return cache[key]


def accept_ev_solution(problem, options):
    """EV-only MILP acceptance: feasible incumbent AND relative gap <= threshold."""
    accepted = usable_solution(problem)
    diagnostic = getattr(problem, "hems_diagnostic", {})
    if options.formulation == "milp":
        extra = getattr(getattr(problem, "solver_stats", None), "extra_stats", None)
        gap = extra.get("mip_gap") if isinstance(extra, dict) else getattr(extra, "MIPGap", None)
        dual = extra.get("mip_dual_bound") if isinstance(extra, dict) else getattr(extra, "ObjBound", None)
        primal = extra.get("mip_primal_bound") if isinstance(extra, dict) else getattr(extra, "ObjVal", None)
        # A proven optimum needs no incumbent-gap estimate. Limit/inaccurate
        # statuses with missing/nonfinite bounds must not be trusted.
        if gap is None and problem.status == cp.OPTIMAL:
            gap = 0.0
        gap_ok = gap is not None and np.isfinite(gap) and 0 <= gap <= options.mip_relative_gap + 1e-12
        accepted = accepted and gap_ok
        diagnostic.update(relative_mip_gap=gap, dual_bound=dual, backend_primal_bound=primal,
            model_objective=problem.value, mip_gap_threshold=options.mip_relative_gap,
            gap_accepted=bool(gap_ok))
    diagnostic.update(fallback=not accepted,
        accepted_incumbent=bool(accepted and problem.status == cp.USER_LIMIT))
    return bool(accepted)


def master_lower_bound(problem, solver):
    """Read a master bound in ORIGINAL objective units (include CVXPY offset).

    SCIPY's continuous extra_stats contains a dummy MIP bound: never use that.
    CLARABEL's dual value is accessed only after a proven OPTIMAL status. A
    missing/unverified certificate returns -inf, never a guessed zero gap.
    """
    try:
        _, _, inverse = problem.get_problem_data(solver)
        offset = float(np.asarray(inverse[-1]["offset"]).item())
        extra = problem.solver_stats.extra_stats
        if solver == "SCIPY":
            if problem.is_mixed_integer():
                bound = extra.get("mip_dual_bound")
            elif problem.status == cp.OPTIMAL:
                return float(problem.value) - 1e-8*max(1, abs(problem.value))
            else:
                return -np.inf
        elif solver == "GUROBI":
            bound = getattr(extra, "ObjBound", None)
        elif solver == "CLARABEL" and problem.status == cp.OPTIMAL:
            native = problem._solver_cache.get("CLARABEL")
            bound = native.get_solution().obj_val_dual if native is not None else None
        else:
            return -np.inf
        if bound is None or not np.isfinite(bound):
            return -np.inf
        value = float(bound)+offset
        return value - 1e-8*max(1, abs(value))
    except (AttributeError, KeyError, TypeError, ValueError):
        return -np.inf


def _comfort_home_durations(mobility, boundaries):
    """Eligible home hours after each path's one-hour return grace period.

    Completed observed returns can precede a truncated issued path. Their
    causal metadata preserves the remaining grace across controller updates.
    Future returns come from this path, including persistence adjustments.
    Duration overlap handles grace expiry without creating new decisions.
    """
    boundaries = pd.DatetimeIndex(boundaries)
    starts, ends = boundaries.asi8[:-1], boundaries.asi8[1:]
    home = mobility.available(boundaries[:-1])
    arrivals = [pd.Timestamp(t).value for t in mobility.trips.arrival]
    observed = mobility.metadata.get("last_observed_arrival")
    if observed is not None:
        observed = pd.Timestamp(observed)
        if pd.isna(observed) or (observed.tz is None) != (boundaries.tz is None):
            raise ValueError("EV last observed arrival must use the forecast clock")
        if observed > boundaries[0]:
            raise ValueError("EV last observed arrival cannot follow forecast issue")
        arrivals.append(observed.value)
    eligible_starts = starts.copy()
    if arrivals:
        arrivals = np.sort(np.asarray(arrivals, dtype=np.int64))
        latest = np.searchsorted(arrivals, starts, side="right") - 1
        has_return = latest >= 0
        eligible_starts[has_return] = np.maximum(
            starts[has_return], arrivals[latest[has_return]] + pd.Timedelta(hours=1).value)
    return np.where(home, np.maximum(0, ends - eligible_starts) / 3.6e12, 0.0)


@dataclass(frozen=True)
class ObservedPeakComfort:
    """Causally observed return exception, fixed until its recovery deadline."""

    arrived_at: pd.Timestamp
    arrival_soc: float
    arrival_capacity_kwh: float
    target_soc: float
    peak_end: pd.Timestamp
    deadline: pd.Timestamp


def _observed_peak_end(timestamp, forecast, rate_structure):
    """End of the actual arrival's peak block using prices known at arrival."""
    local = pd.Timestamp(timestamp)
    if local.tz is not None:
        local = local.tz_convert(LOCAL_TIMEZONE)
    if rate_structure in {WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP}:
        if not WinterTOURate.is_on_peak(local):
            return None
        end_hour = 8 if local.hour < 8 else 21
        return local.normalize() + pd.Timedelta(hours=end_hour)
    if rate_structure != MISO_RTP_DOE:
        raise ValueError("Observed peak comfort needs a Grid 1, 2, or 3 rate")
    day_start = local.normalize()
    day_end = day_start + pd.DateOffset(days=1)
    hours = pd.date_range(day_start, day_end, freq="h", inclusive="left")
    if forecast.tariff_price_as_known_at is not None:
        prices = np.asarray([
            forecast.tariff_price_as_known_at(timestamp, hour) for hour in hours
        ], dtype=float)
    elif forecast.tariff_price_at is not None:
        prices = np.asarray([forecast.tariff_price_at(hour) for hour in hours], dtype=float)
    else:
        raise ValueError("Grid 3 peak comfort needs a current-day hourly price estimate")
    if not np.isfinite(prices).all():
        raise ValueError("Grid 3 peak comfort needs finite current-day prices")
    # The 85th percentile identifies four hours on an ordinary day when
    # prices are distinct. Tied prices at the threshold may add peak hours.
    threshold = float(np.percentile(prices, 85))
    peak = set(np.flatnonzero(prices >= threshold).tolist())
    current = int(hours.searchsorted(local, side="right")) - 1
    if current not in peak:
        return None
    last = current
    while last + 1 in peak:
        last += 1
    return hours[last + 1] if last + 1 < len(hours) else day_end


def _ev_comfort_cost(horizon, state, target, observed_peak=None):
    """Time-step/N soft SOC shortfall, only at home outside return grace."""
    if not np.any(horizon.comfort_dt):
        return cp.Constant(0)
    if observed_peak is not None:
        cutoff = pd.Timestamp(observed_peak.deadline).value
        ends = pd.DatetimeIndex(horizon.t).asi8[1:]
        # Eligible comfort time is a terminal subinterval of each home block:
        # the mobility boundaries split returns and the grace expiry clips its
        # beginning. Divide only that eligible duration at the known deadline.
        eligible_starts = ends - np.rint(horizon.comfort_dt * 3.6e12).astype(np.int64)
        temporary_dt = np.maximum(0, np.minimum(ends, cutoff) - eligible_starts) / 3.6e12
        temporary_dt = np.minimum(temporary_dt, horizon.comfort_dt)
        ordinary_dt = np.maximum(0, horizon.comfort_dt - temporary_dt)
        temporary = np.flatnonzero(temporary_dt > 0)
        ordinary = np.flatnonzero(ordinary_dt > 0)
        costs = []
        if len(temporary):
            costs.append(cp.sum(cp.multiply(
                temporary_dt[temporary], cp.pos(observed_peak.target_soc - state[temporary]),
            )))
        if len(ordinary):
            costs.append(cp.sum(cp.multiply(
                ordinary_dt[ordinary], cp.pos(target - state[ordinary]),
            )))
        return sum(costs) / horizon.N
    return cp.sum(cp.multiply(horizon.comfort_dt, cp.pos(target - state[:-1]))) / horizon.N


@dataclass(frozen=True)
class EVHorizon:
    """Mobility-split arrays with fixed control-interval ownership."""
    t: pd.DatetimeIndex
    dt: np.ndarray
    z: np.ndarray
    price: np.ndarray
    backed_cost: np.ndarray
    discharge_cost: np.ndarray
    discharge_mean: np.ndarray
    discharge_min: np.ndarray
    trip_kwh: np.ndarray
    group: np.ndarray
    N: int
    future_hours: float
    history_hours: float
    budget_dt: np.ndarray
    departure_indices: np.ndarray
    trip_timing: str
    comfort_dt: np.ndarray

    @classmethod
    def from_forecast(cls, c, timestamp, forecast, *, trip_timing="arrival"):
        if trip_timing not in ("arrival", "departure"):
            raise ValueError("EV trip timing must be arrival or departure")
        o = c.options
        end = timestamp + c.horizon
        data = forecast.window(timestamp, c.horizon)
        step = forecast.data.index[1] - forecast.data.index[0]
        if data.empty or data.index[0] != timestamp or data.index[-1]+step < end:
            raise ValueError("EV forecast does not cover the complete horizon")
        hub_intervals = forecast.ev_hub_intervals
        if hub_intervals is None:
            if forecast.backed_load_kw is None:
                raise ValueError("EV forecast requires backed-load data")
            load = forecast.backed_load_kw.reindex(data.index)
            if load.isna().any() or not np.isfinite(load).all() or (load < -1e-7).any():
                raise ValueError("Invalid or incomplete backed-load forecast")
            hub_rows = None
        else:
            if forecast.backed_load_kw is not None:
                raise ValueError("EV forecast must select either interval Hub power or minute backed load")
            if step != pd.Timedelta(minutes=1):
                raise ValueError("EV interval Hub forecasts require one-minute tariff and mobility data")
            hub_rows = hub_intervals.window(timestamp, c.horizon, o.max_discharge_kw)
        mobility = forecast.ev_schedules[c.ev.vehicle_id]
        trips = mobility.demand(o.consumption_kwh_per_km,
            soc_depletion_capacity_kwh=o.soc_depletion_capacity_kwh(c.current_capacity))
        trips = trips.loc[(trips.departure < end) & (trips.arrival > timestamp)]
        boundaries = set(pd.date_range(timestamp, end, freq=pd.Timedelta(minutes=o.control_minutes)))
        boundaries.update((timestamp, end))
        F = min(o.horizon_hours, 12.0)
        budget_end = timestamp + pd.Timedelta(hours=F)
        boundaries.add(budget_end)  # integrate the budget boundary exactly, without another solve
        for trip in trips.itertuples():
            boundaries.update(t for t in (trip.departure, trip.arrival) if timestamp < t < end)
        t = pd.DatetimeIndex(sorted(boundaries))
        dt = np.diff(t.asi8)/3.6e12
        z = mobility.available(t[:-1]).astype(int)
        means, minima, prices, dc_cost, backed_cost = [], [], [], [], []
        for a, b in zip(t[:-1], t[1:]):
            selected = (data.index >= a) & (data.index < b)
            pi = data.loc[selected, "electricity_price_usd_per_kwh"].to_numpy()
            if hub_rows is None:
                L = load.loc[selected].to_numpy()
                if not len(L):
                    raise ValueError("EV trip boundaries require a one-minute forecast grid; "
                                     "normalize trips before constructing the horizon")
                cap = np.minimum(L, o.max_discharge_kw) if o.v2h else np.zeros(len(L))
                if not np.isfinite(pi).all():
                    raise ValueError("EV electricity prices must be finite")
                # Signed chemical power held over a block must satisfy every load cap.
                means.append(cap.mean()); minima.append(cap.min()); prices.append(pi.mean())
                dc_cost.append(np.mean(pi*cap)); backed_cost.append(np.mean(pi*L))
                if (o.formulation == "convex" and o.charge_efficiency*o.discharge_efficiency < 1
                        and z[len(means)-1] and cap.min() > 0 and (pi < 0).any()):
                    raise ValueError("Negative prices with bidirectional convex EV power are nonconvex; explicitly select MILP")
            else:
                parent = a.floor(HUB_FORECAST_INTERVAL)
                split = b - a != HUB_FORECAST_INTERVAL or a not in hub_rows.index
                # Default behavior requires complete bins. The explicit study
                # policy holds each *issued coefficient* on event subsegments;
                # it does not claim to reconstruct an unknown minute profile.
                if split and hub_intervals.mobility_split_policy != "hold_parent_interval_coefficients":
                    raise ValueError("EV mobility boundary splits a 15-minute Home Hub forecast interval")
                if parent not in hub_rows.index or b > parent + HUB_FORECAST_INTERVAL:
                    raise ValueError("EV mobility segment crosses a Home Hub forecast interval")
                minute_index = pd.date_range(a, b, freq="min", inclusive="left")
                if not data.index[selected].equals(minute_index):
                    raise ValueError("EV interval Hub forecast needs every minute of tariff data")
                # Retain the original constant-tariff condition over the whole
                # parent bin, even when mobility divides it into several pieces.
                if split:
                    parent_index = pd.date_range(parent, parent + HUB_FORECAST_INTERVAL,
                                                 freq="min", inclusive="left")
                    parent_prices = data.reindex(parent_index)["electricity_price_usd_per_kwh"].to_numpy()
                else:
                    parent_prices = pi
                if not np.isfinite(parent_prices).all():
                    raise ValueError("EV electricity prices must be finite")
                if not np.allclose(parent_prices, parent_prices[0], rtol=0, atol=1e-12):
                    raise ValueError("EV interval Hub forecast needs constant tariff within each 15-minute bin")
                row = hub_rows.loc[parent]
                cap_mean = float(row["eligible_mean_kw"]) if o.v2h else 0.0
                cap_min = float(row["eligible_min_kw"]) if o.v2h else 0.0
                price = float(pi.mean())
                means.append(cap_mean); minima.append(cap_min); prices.append(price)
                dc_cost.append(price * cap_mean)
                backed_cost.append(price * float(row["eligible_load_mean_kw"]))
                if (o.formulation == "convex" and o.charge_efficiency*o.discharge_efficiency < 1
                        and z[len(means)-1] and cap_min > 0 and (pi < 0).any()):
                    raise ValueError("Negative prices with bidirectional convex EV power are nonconvex; explicitly select MILP")
        trip_kwh = np.zeros(len(t)-1)
        for trip in trips.itertuples():
            if trip_timing == "departure":
                # A scenario's full away-trip demand is an impulse at its
                # departure. An already-away path carries that impulse into
                # the issue instant, since its departure is outside the solve.
                event = max(trip.departure, timestamp)
                trip_kwh[t.get_loc(event)] += trip.energy_kwh
            elif trip.arrival <= end:
                trip_kwh[t.get_loc(trip.arrival)-1] += trip.energy_kwh
        group = ((t[:-1]-timestamp)/pd.Timedelta(minutes=o.control_minutes)).astype(int).to_numpy()
        budget_dt = np.array([max(0, (min(b, budget_end)-a).total_seconds()/3600)
                              for a, b in zip(t[:-1], t[1:])])
        # Pre-departure SOC at exact mobility boundaries; use [timestamp, end).
        departures = trips.loc[trips.departure >= timestamp, "departure"].drop_duplicates()
        departure_indices = t.get_indexer(departures)
        comfort_dt = _comfort_home_durations(mobility, t)
        return cls(t, dt, z, np.array(prices), np.array(backed_cost), np.array(dc_cost),
                   np.array(means), np.array(minima), trip_kwh, group,
                   int(np.ceil(o.horizon_hours*60/o.control_minutes)), F, 24-F, budget_dt,
                   departure_indices, trip_timing, comfort_dt)


def _ev_trip_shortfall(h):
    """Bounded unmet kWh at positive trip events, with no slack elsewhere."""
    indices = np.flatnonzero(h.trip_kwh > 0)
    if not len(indices):
        return cp.Constant(np.zeros(len(h.dt))), None, indices, []
    variable = cp.Variable(len(indices), nonneg=True, name="trip_shortfall_kwh")
    selector = np.zeros((len(h.dt), len(indices)))
    selector[indices, np.arange(len(indices))] = 1.0
    return selector @ variable, variable, indices, [variable <= h.trip_kwh[indices]]


def _ev_soc_dynamics(h, x, chemical, initial_soc, capacity_kwh, trip_shortfall=None):
    """Return SOC constraints and the SOC immediately before each departure."""
    served_trip = h.trip_kwh if trip_shortfall is None else h.trip_kwh - trip_shortfall
    if h.trip_timing == "arrival":
        return [x[0] == initial_soc,
                x[1:] == x[:-1] + (cp.multiply(h.dt, chemical)-served_trip)/capacity_kwh], (
                    x[h.departure_indices] if len(h.departure_indices) else None)

    # In a scenario, x[j] is the state just AFTER any departure at t[j].
    # trip_kwh[j] is therefore timestamped at t[j], while predeparture
    # comfort and capacity checks use the state just before that impulse.
    events = (np.r_[h.trip_kwh, 0.0] if trip_shortfall is None
              else cp.hstack([served_trip, 0.0]))
    constraints = [x[0] == initial_soc-events[0]/capacity_kwh,
                   x[1:] == x[:-1] + (cp.multiply(h.dt, chemical)-events[1:])/capacity_kwh]
    predeparture = (x[h.departure_indices] + events[h.departure_indices]/capacity_kwh
                    if len(h.departure_indices) else None)
    if len(h.departure_indices):
        constraints += [predeparture >= 0, predeparture <= 1]
    return constraints, predeparture


def _rainflow_path_cost_gradient(battery, history, h, soc, capacity_kwh,
                                 trip_shortfall_kwh=None):
    """Evaluate aging over both sides of each scenario departure impulse.

    The measured issue SOC is fixed. If an event occurs at issue, retain it
    before the post-event x[0]. At later departures, insert the pre-event SOC
    before x[j] and add both gradient entries back onto that state variable.
    With trip shortfall, that inserted state is x[j]+(E[j]-epsilon[j])/C.
    Its derivative with respect to epsilon[j] must also enter each cut. The
    issue pre-event SOC remains measured data, independent of shortfall.

    The legacy call returns (cost, SOC gradient). Supplying shortfall returns
    (cost, SOC gradient, full trip-shortfall gradient in dollars per kWh).
    """
    soc = np.asarray(soc, dtype=float)
    shortfall = (np.zeros_like(h.trip_kwh, dtype=float) if trip_shortfall_kwh is None
                 else np.asarray(trip_shortfall_kwh, dtype=float))
    if shortfall.shape != h.trip_kwh.shape:
        raise ValueError("Trip shortfall must match the horizon trip-energy vector")
    issue_event = h.trip_timing == "departure" and h.trip_kwh[0] != 0
    fixed = np.asarray(history if issue_event else history[:-1], dtype=float)
    values = list(fixed)
    state_indices = []
    predeparture_positions = []
    for j, value in enumerate(soc):
        if h.trip_timing == "departure" and 0 < j < len(h.trip_kwh) and h.trip_kwh[j] != 0:
            predeparture_positions.append((j, len(values)))
            values.append(value + (h.trip_kwh[j]-shortfall[j])/capacity_kwh)
            state_indices.append(j)
        values.append(value)
        state_indices.append(j)
    total, gradient = battery.rainflow_cost_gradient(values)
    state_gradient = np.zeros_like(soc)
    np.add.at(state_gradient, state_indices, gradient[len(fixed):])
    if trip_shortfall_kwh is None:
        return total, state_gradient
    shortfall_gradient = np.zeros_like(shortfall)
    for j, position in predeparture_positions:
        shortfall_gradient[j] = -gradient[position]/capacity_kwh
    return total, state_gradient, shortfall_gradient


def _same_control_interval(constraints, q, h):
    """Mobility integration boundaries do not create extra decision opportunities."""
    shared, first = [], []
    for group in np.unique(h.group):
        active = np.flatnonzero((h.group == group) & (h.z == 1))
        if len(active) > 1:
            shared.extend(active[1:])
            first.extend([active[0]] * (len(active) - 1))
    if shared:
        constraints.append(q[np.asarray(shared)] == q[np.asarray(first)])


def _milp_transitions(c, h, b_c, b_d, constraints):
    """b_j-b_{j-1}=s_j-e_j; dwell and historical ends + planned starts budget."""
    o, n = c.options, len(h.dt)
    starts, ends = {}, {}
    for direction, b in ((1, b_c), (-1, b_d)):
        prior = int(c.previous_mode == direction)
        s = cp.Variable(n, boolean=True, name=f"b_start_{direction}")
        e = cp.Variable(n, boolean=True, name=f"b_end_{direction}")
        previous = cp.hstack([prior, b[:-1]]) if n > 1 else cp.Constant([prior])
        constraints += [b-previous == s-e, s+e <= 1]
        starts[direction], ends[direction] = s, e
        if o.enforce_cycle_limit:
            cap = c._cycle_cap(direction)
            if cap is not None:
                left = h.t[0]-pd.Timedelta(hours=h.history_hours)
                observed_ends = sum(left < t <= h.t[0] for t in c.ends[direction])
                planned = np.flatnonzero(h.budget_dt > 0)
                constraints.append(observed_ends + cp.sum(s[planned]) <= cap)
        if not o.enforce_dwell_constraints:
            continue
        can_run = h.z * (np.ones(n) if direction == 1 else (h.discharge_mean > 1e-9))
        remaining = c._remaining_dwell(h.t[0], direction)
        for j, t in enumerate(h.t[:-1]):
            if t < h.t[0]+remaining and (not prior or can_run[j]):
                constraints.append(b[j] == prior)
            for event, active, minutes in ((s, b, o.dwell_on_minutes), (e, 1-b, o.dwell_off_minutes)):
                for k in range(j, n):
                    if h.t[k] >= t+pd.Timedelta(minutes=minutes):
                        break
                    if event is s and not can_run[k]:
                        break  # forecast physical interruption takes precedence
                    constraints.append(active[k] >= event[j])
    return starts, ends


def build_ev_problem(c, timestamp, initial_soc, forecast, *, trip_timing="arrival"):
    """EV OCP, in the same data -> variables -> constraints -> cost order as HPWH.

    x = battery SOC; p = signed chemical kW; P = signed AC kW; z = availability.
    u_c/u_d = nonnegative AC power; b_c/b_d = exclusive MILP operating modes.
    N counts control intervals, while j indexes mobility-split integration blocks.

    min J_electricity + J_persistence + J_terminal + w_comfort*H_comfort
        + w_trip/N * sum_j(dt_j * epsilon_j / E)
        + w_health * (w_health_SOC*(H_low + H_high) + C_degradation/N).

    Health terms integrate using dt/N. Comfort uses eligible home dt/N, excluding
    the first hour after each return, with the same left-endpoint SOC sampling.
    C_degradation is already
    cumulative dollars: do not multiply it by dt again. With rainflow enabled,
    this builds a MASTER problem; _solve_rainflow supplies its supporting cuts
    and certifies the exact objective before an action can be accepted.
    """
    o, E = c.options, c.current_capacity
    h = EVHorizon.from_forecast(c, timestamp, forecast, trip_timing=trip_timing)
    n = len(h.dt)
    x = cp.Variable(n+1, name="x_ev")
    constraints = [x >= 0, x <= 1]
    mode_starts, mode_ends = {}, {}

    # (1) Admissible power. z is a constant, never a decision variable.
    if o.formulation == "convex":
        p = cp.Variable(n, name="p_chemical")
        constraints += [p <= o.charge_efficiency*c.ev.charger_power_kw*h.z,
                        p >= -h.discharge_min*h.z/o.discharge_efficiency]
        _same_control_interval(constraints, p, h)
        # Only bidirectional blocks require a convex piecewise AC expression.
        ac_blocks = [cp.maximum(p[j]/o.charge_efficiency, o.discharge_efficiency*p[j])
                       if h.z[j] and h.discharge_min[j] > 0 and o.charge_efficiency*o.discharge_efficiency < 1
                       else p[j]/o.charge_efficiency
                       for j in range(n)]
        P = cp.hstack(ac_blocks)
        u_c = cp.pos(p)/o.charge_efficiency
        u_d = o.discharge_efficiency*cp.pos(-p)
        energy_cost = sum(h.dt[j]*(h.backed_cost[j]+h.price[j]*ac_blocks[j]) for j in range(n)) / h.N
        dp = p-cp.hstack([c.previous_chemical_power, p[:-1]]) if n > 1 else p-c.previous_chemical_power
        J_pers = cp.sum(cp.multiply(h.dt, cp.abs(dp))) / h.N
        discharge_chemical = cp.pos(-p)
        fraction = None
    else:
        u_c = cp.Variable(n, nonneg=True, name="u_c")
        u_d = cp.Variable(n, nonneg=True, name="u_d")
        b_c = cp.Variable(n, boolean=True, name="b_c")
        b_d = cp.Variable(n, boolean=True, name="b_d")
        constraints += [b_c+b_d <= 1, b_c <= h.z, b_d <= h.z,
                        b_d <= (h.discharge_mean > 1e-9).astype(int),
                        u_c == c.ev.charger_power_kw*b_c,
                        u_d == cp.multiply(h.discharge_mean, b_d)]
        _same_control_interval(constraints, b_c, h)
        _same_control_interval(constraints, b_d, h)
        p = o.charge_efficiency*u_c-u_d/o.discharge_efficiency
        P, fraction = u_c-u_d, b_d
        mode_starts, mode_ends = _milp_transitions(c, h, b_c, b_d, constraints)
        energy_cost = cp.sum(cp.multiply(h.dt, h.backed_cost + cp.multiply(h.price, u_c)
                            - cp.multiply(h.discharge_cost, b_d))) / h.N
        # One-hot mode variation, not load-following changes within a mode.
        # Each switch changes two indicators; retain the existing dt/N scaling.
        b_idle = 1-b_c-b_d
        mode_changes = []
        for b, mode in ((b_c, 1), (b_d, -1), (b_idle, 0)):
            prior = int(c.previous_mode == mode)
            previous = cp.hstack([prior, b[:-1]]) if n > 1 else cp.Constant([prior])
            mode_changes.append(cp.abs(b-previous))
        J_pers = cp.sum(cp.multiply(h.dt, sum(mode_changes))) / h.N
        discharge_chemical = u_d/o.discharge_efficiency

    # (2) Trip shortfall can relax positive event demand, never SOC bounds.
    shortfall, shortfall_variable, shortfall_indices, shortfall_bounds = _ev_trip_shortfall(h)
    constraints.extend(shortfall_bounds)
    served_trip = h.trip_kwh - shortfall
    dynamics, predeparture_soc = _ev_soc_dynamics(h, x, p, initial_soc, E, shortfall)
    constraints.extend(dynamics)

    # (3) Cumulative V2H battery discharge (not driving or SOC excursion).
    D_past = c.discharge_history(h.t[0]-pd.Timedelta(hours=h.history_hours), h.t[0], normalized=True)
    D_future = cp.sum(cp.multiply(h.budget_dt, discharge_chemical)) / E
    if o.enforce_discharge_budget:
        constraints.append(D_past+D_future <= o.discharge_budget_soc)

    # (4) Soft health band throughout; home comfort after the one-hour return grace.
    health_low = cp.sum(cp.multiply(h.dt, cp.pos(o.health_soc_min-x[:-1]))) / h.N
    health_high = cp.sum(cp.multiply(h.dt, cp.pos(x[:-1]-o.health_soc_max))) / h.N
    comfort = (_ev_comfort_cost(h, x, o.comfort_soc_min,
                                getattr(c, "observed_peak_comfort", None))
               if o.comfort_weight else cp.Constant(0))

    # (5) Optional monetary degradation, with the same physical calibration used
    #     for realized-trajectory metrics. Health and comfort weights are separate.
    battery = BatteryDegradation(c.ev.capacity_kwh, o, current_capacity_kwh=E)
    degradation = cp.Constant(0)
    theta = None
    if o.enable_degradation and o.degradation_method != "none" and o.health_weight > 0 and battery.depreciable_usd > 0:
        if o.degradation_method == "linear_throughput":
            # Battery-side throughput includes served driving energy once.
            throughput = cp.sum(cp.multiply(h.dt, cp.abs(p))) + cp.sum(served_trip)
            degradation = battery.throughput_usd_per_kwh * throughput / h.N
        else:
            # Exact rainflow is evaluated by the outer loop, not a CVXPY atom.
            theta = cp.Variable(nonneg=True, name="rainflow_increment_usd")
            degradation = theta / h.N
    # (6) Terminal energy value: v/N * (e_0-e_N), e_j = C_k*x_j.
    # C_k is measured faded capacity, fixed for this solve. This affine term
    # prices depletion AND credits retained energy; it is not a moving SOC floor.
    v, terminal_provenance = (controller_terminal_energy_value(c, forecast, timestamp)
                               if o.enforce_terminal_soc else (0.0, None))
    J_terminal = v * E * (initial_soc-x[-1]) / h.N
    J = dict(electricity=o.electricity_weight*energy_cost,
             health_soc=o.health_weight*o.health_soc_weight*(health_low+health_high),
             degradation=o.health_weight*degradation,
             comfort=o.comfort_weight*comfort,
             trip_shortfall=o.trip_shortfall_weight*cp.sum(cp.multiply(h.dt, shortfall))/(E*h.N),
             persistence=o.persistence_weight*J_pers, terminal=J_terminal)
    problem = cp.Problem(cp.Minimize(sum(J.values())), constraints)
    if not problem.is_dcp():
        raise ValueError("EV formulation is not DCP; check bidirectional negative-price inputs")
    first_active = np.flatnonzero((h.group == 0) & (h.z == 1))
    return problem, dict(state=x, chemical=p, ac=P, charge=u_c, discharge=u_d, fraction=fraction,
        charge_mode=b_c if o.formulation == "milp" else None,
        discharge_mode=b_d if o.formulation == "milp" else None,
        boundaries=h.t, command_index=int(first_active[0]) if len(first_active) else 0,
        N=h.N, horizon=h, starts=mode_starts, ends=mode_ends, discharge_past=D_past,
        measured_soh=c.current_soh, capacity_kwh=E,
        trip_shortfall_kwh=shortfall, trip_shortfall_variable=shortfall_variable,
        trip_shortfall_indices=shortfall_indices, served_trip_kwh=served_trip,
        terminal_value_usd_per_kwh=v, terminal_value_provenance=terminal_provenance,
        initial_soc=initial_soc, rainflow_theta=theta, battery=battery, health_low=health_low, health_high=health_high,
        discharge_future=D_future, objective_terms=J, **J)


class EVMPCController:
    """Fixed-clock MPC with measured SOH, observed history and safe actuation."""
    def __init__(self, ev):
        self.ev, self.options = ev, ev.hems
        # Explicit per-run opt-in: the isolated Grid 3 coordinator selects the
        # historical method. Generic EV and all other grid controllers retain
        # the original daily mean unless that run deliberately opts in.
        self.terminal_energy_value_method = "daily_mean"
        self._terminal_availability_history = None
        self._terminal_energy_value_cache = {}
        if self.options.enforce_terminal_soc and self.options.terminal_soc_weight != 0:
            warnings.warn("EV terminal_soc_weight is retired and ignored: terminal cost is "
                          "daily average tariff * (initial-final chemical kWh) / N. "
                          "Remove terminal_soc_weight; enforce_terminal_soc remains the on/off switch.",
                          UserWarning, stacklevel=2)
        self.equipment_name = f"EV_{ev.vehicle_id}"
        self.horizon = pd.Timedelta(hours=self.options.horizon_hours)
        self.current_soh = self.options.initial_soh
        self.current_capacity = ev.capacity_kwh*self.current_soh
        self.solve_history, self.plans, self.action_history = [], [], []
        self.starts, self.ends = {1: [], -1: []}, {1: [], -1: []}
        self.last_change = {1: None, -1: None}
        self.discharge_samples = []  # (start, end, battery-side V2H kWh, capacity at action)
        self.soc_history = [float(ev.initial_soc_fraction)]
        self.previous_mode, self.previous_power, self.previous_chemical_power = 0, 0.0, 0.0
        self.next_solve, self.first_observation = None, None
        self.target, self.held_mode, self.fallback = 0.0, 0, False
        self.peak_arrival_comfort_enabled = False
        self.peak_arrival_rate_structure: str | None = None
        self.observed_peak_comfort: ObservedPeakComfort | None = None
        self._last_processed_comfort_arrival: pd.Timestamp | None = None
        self._peak_comfort_history_seeded = False
        self.peak_comfort_history: list[dict[str, Any]] = []
        if self.options.solver not in cp.installed_solvers():
            raise ValueError(f"EV {ev.vehicle_id}: solver {self.options.solver} not installed; "
                             "install its dependency/license or select another backend")

    def configure_observed_peak_comfort(self, rate_structure: str) -> None:
        """Opt in to the 30% observed-return rule for a predictive controller."""
        if rate_structure not in {
            WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE,
        }:
            raise ValueError("Observed peak comfort needs a Grid 1, 2, or 3 rate")
        if not np.isclose(self.options.comfort_soc_min, 0.30, rtol=0, atol=1e-12):
            raise ValueError("Observed peak comfort requires a 30% base comfort target")
        self.peak_arrival_rate_structure = rate_structure
        self.peak_arrival_comfort_enabled = True

    def comfort_target_at(self, timestamp: pd.Timestamp) -> float:
        """Current measured-state target, including a still-active return latch."""
        active = self.observed_peak_comfort
        if active is not None and pd.Timestamp(timestamp) < active.deadline:
            return active.target_soc
        return float(self.options.comfort_soc_min)

    def observe_realized_peak_comfort(
        self, timestamp: pd.Timestamp, mobility, soc: float, forecast: Forecast,
        *, arrival_capacity_kwh: float | None = None, parked: bool | None = None,
    ) -> None:
        """Latch only a realized peak return, at its measured arrival minute."""
        if not self.peak_arrival_comfort_enabled:
            return
        timestamp = pd.Timestamp(timestamp)
        active = self.observed_peak_comfort
        if active is not None and timestamp >= active.deadline:
            self.peak_comfort_history.append(dict(
                event="deadline_expired", timestamp=timestamp,
                arrived_at=active.arrived_at, deadline=active.deadline,
            ))
            self.observed_peak_comfort = None
        elif (active is not None and (mobility.trips.departure == timestamp).any()
              and (parked is None or not bool(parked))):
            self.peak_comfort_history.append(dict(
                event="actual_departure_cancel", timestamp=timestamp,
                reason="actual_departure_before_deadline",
                arrived_at=active.arrived_at, deadline=active.deadline,
            ))
            self.observed_peak_comfort = None
        arrivals = mobility.trips.loc[mobility.trips.arrival == timestamp, "arrival"]
        if arrivals.empty or self._last_processed_comfort_arrival == timestamp:
            return
        self._last_processed_comfort_arrival = timestamp
        if ((parked is not None and not bool(parked))
                or not bool(mobility.available([timestamp])[0])
                or soc >= self.options.comfort_soc_min):
            return
        peak_end = _observed_peak_end(timestamp, forecast, self.peak_arrival_rate_structure)
        if peak_end is None:
            return
        target = max(0.20, float(soc))
        charging_kw = self.ev.charger_power_kw * self.options.charge_efficiency
        if not np.isfinite(charging_kw) or charging_kw <= 0:
            raise ValueError("Observed peak comfort needs a positive estimated charging rate")
        capacity = (self.current_capacity if arrival_capacity_kwh is None
                    else float(arrival_capacity_kwh))
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("Observed peak comfort needs positive measured arrival capacity")
        recovery_minutes = (self.options.comfort_soc_min - target) * capacity / charging_kw * 60
        recovery_minutes = 15 * int(np.ceil(max(0.0, recovery_minutes) / 15 - 1e-12))
        latch = ObservedPeakComfort(
            arrived_at=timestamp, arrival_soc=float(soc),
            arrival_capacity_kwh=capacity, target_soc=target,
            peak_end=peak_end, deadline=peak_end + pd.Timedelta(minutes=recovery_minutes),
        )
        self.observed_peak_comfort = latch
        self.peak_comfort_history.append(dict(
            event="qualifying_actual_arrival", timestamp=timestamp,
            arrival_soc=latch.arrival_soc, arrival_capacity_kwh=latch.arrival_capacity_kwh,
            target_soc=latch.target_soc, peak_end=latch.peak_end,
            estimated_chemical_charge_kw=charging_kw,
            recovery_minutes=recovery_minutes, deadline=latch.deadline,
        ))

    def seed_observed_peak_comfort(
        self, timestamp: pd.Timestamp, mobility, forecast: Forecast,
        completed_states: list[tuple[pd.Timestamp, Mapping[str, Any]]],
    ) -> None:
        """Carry a causally measured warm-up return into the first MPC issue."""
        if not self.peak_arrival_comfort_enabled or self._peak_comfort_history_seeded:
            return
        self._peak_comfort_history_seeded = True
        issue = pd.Timestamp(timestamp)
        arrivals = set(pd.DatetimeIndex(mobility.trips.arrival).asi8.tolist())
        departures = set(pd.DatetimeIndex(mobility.trips.departure).asi8.tolist())
        events = arrivals | departures
        soc_key = f"{self.equipment_name} SOC (-)"
        capacity_key = f"{self.equipment_name} Capacity (kWh)"
        parked_key = f"{self.equipment_name} Parked"
        preceding_time = None
        preceding_parked = None
        for completed_at, status in sorted(completed_states, key=lambda item: pd.Timestamp(item[0])):
            completed_at = pd.Timestamp(completed_at)
            event_time = completed_at - pd.Timedelta(minutes=1)
            if event_time >= issue:
                break
            if parked_key not in status:
                raise ValueError("Warm-up EV observation lacks measured parked state")
            parked = bool(status[parked_key])
            prior = (preceding_parked if preceding_time == completed_at - pd.Timedelta(minutes=1)
                     else None)
            preceding_time, preceding_parked = completed_at, parked
            if event_time.value not in events:
                continue
            actual_arrival = event_time.value in arrivals and prior is False and parked
            actual_departure = event_time.value in departures and prior is True and not parked
            if not (actual_arrival or actual_departure):
                continue
            if soc_key not in status or capacity_key not in status:
                raise ValueError("Warm-up EV event lacks measured SOC or usable capacity")
            self.observe_realized_peak_comfort(
                event_time, mobility, float(status[soc_key]), forecast,
                arrival_capacity_kwh=float(status[capacity_key]), parked=parked,
            )

    def set_terminal_availability_history(self, mobility):
        """Register this vehicle's validated observed schedule, never a scenario.

        The terminal helper queries only completed historical days even if this
        immutable replay also carries the future plant schedule.
        """
        if mobility is not self._terminal_availability_history:
            self._terminal_availability_history = mobility
            self._terminal_energy_value_cache.clear()

    def _observe(self, timestamp, power, plant_step):
        mode = 1 if power > 1e-7 else -1 if power < -1e-7 else 0
        observed_at = timestamp-plant_step
        if self.first_observation is None:
            self.first_observation = timestamp
            observed_at = timestamp  # no invented pre-simulation operating history
        else:
            self.discharge_samples.append((observed_at, timestamp,
                max(-power, 0)/self.options.discharge_efficiency*plant_step.total_seconds()/3600,
                self.current_capacity))
        if mode != self.previous_mode:
            for direction in (1, -1):
                if (mode == direction) != (self.previous_mode == direction):
                    self.last_change[direction] = observed_at
                if mode == direction:
                    self.starts[direction].append(observed_at)
                if self.previous_mode == direction:
                    self.ends[direction].append(observed_at)
        self.previous_mode, self.previous_power = mode, power
        self.previous_chemical_power = (power*self.options.charge_efficiency if power >= 0
                                       else power/self.options.discharge_efficiency)
        cutoff = timestamp-pd.Timedelta(hours=24)
        for direction in (1, -1):
            self.starts[direction] = [t for t in self.starts[direction] if t > cutoff]
            self.ends[direction] = [t for t in self.ends[direction] if t > cutoff]
        self.discharge_samples = [s for s in self.discharge_samples if s[1] > cutoff]

    def discharge_history(self, left, right, *, normalized=False):
        """Partial boundary intervals count by overlap; driving is excluded."""
        total = 0.
        for sample in self.discharge_samples:
            a, b, kwh = sample[:3]
            capacity = sample[3] if len(sample) == 4 else self.ev.capacity_kwh
            if b > a and b > left and a < right:
                total += kwh*max(0, (min(b, right)-max(a, left)).total_seconds())/(b-a).total_seconds() / (
                    capacity if normalized else 1)
        return total

    def _remaining_dwell(self, timestamp, direction):
        changed = self.last_change[direction]
        if not self.options.enforce_dwell_constraints or changed is None:
            return pd.Timedelta(0)
        minutes = (self.options.dwell_on_minutes if self.previous_mode == direction
                   else self.options.dwell_off_minutes)
        return max(pd.Timedelta(0), changed+pd.Timedelta(minutes=minutes)-timestamp)

    def _cycle_cap(self, direction):
        if not self.options.enforce_cycle_limit:
            return None
        return (self.options.max_charge_starts_24h if direction == 1
                else self.options.max_discharge_starts_24h)

    def build_problem(self, timestamp, soc, forecast):
        """Dispatch to a named formulation, mirroring the HPWH public builders."""
        if isinstance(forecast, EVScenarioSet):
            from .ev_scenario import build_scenario_problem
            return build_scenario_problem(self, pd.Timestamp(timestamp), soc, forecast)
        builder = (self.optimization_problem_convex if self.options.formulation == "convex"
                   else self.optimization_problem_milp)
        return builder(timestamp, soc, forecast)

    def optimization_problem_convex(self, timestamp, soc, forecast):
        """Signed chemical power, fixed availability and convex optional costs."""
        if self.options.formulation != "convex":
            raise ValueError("Convex builder requires formulation='convex'")
        return build_ev_problem(self, pd.Timestamp(timestamp), soc, forecast)

    def optimization_problem_milp(self, timestamp, soc, forecast):
        """Exclusive modes with optional dwell/caps and the same health costs."""
        if self.options.formulation != "milp":
            raise ValueError("MILP builder requires formulation='milp'")
        return build_ev_problem(self, pd.Timestamp(timestamp), soc, forecast)

    def _project(self, timestamp, target, soc, available, backed):
        """Discrete conversion and physical safety ONLY; never enforce dwell/caps."""
        o = self.options
        candidates = {0: 0.0}
        if available and soc < 1-1e-9:
            candidates[1] = self.ev.charger_power_kw
        if available and soc > 1e-9 and o.v2h and backed > 1e-9:
            candidates[-1] = -min(backed, o.max_discharge_kw)
        if o.projection == "threshold":
            reference = self.ev.charger_power_kw if target >= 0 else min(backed, o.max_discharge_kw)
            target = np.sign(target)*reference if reference and abs(target) >= o.projection_threshold*reference else 0
        def score(mode):
            power = candidates[mode]
            if o.formulation == "milp":
                return (power-target)**2
            chemical = power*o.charge_efficiency if power >= 0 else power/o.discharge_efficiency
            return (o.projection_tracking_weight*(power-target)**2 +
                    o.projection_switch_weight*(mode != self.previous_mode) +
                    o.persistence_weight*abs(chemical-self.previous_chemical_power))
        mode = min(candidates, key=score)
        override = bool(self.previous_mode and self.previous_mode not in candidates)
        return mode, candidates[mode], override

    def _solve_rainflow(self, problem, plan, options, deadline):
        """Supporting-cut outer approximation of the EXACT convex rainflow cost.

        Each feasible SOC and trip shortfall gives an exact upper bound. Each
        master dual bound is a lower bound. Native MILP gap alone cannot certify
        this problem. History is fixed; its already-accounted cost is subtracted.
        """
        if "scenario_plans" in plan:
            from .ev_scenario import solve_rainflow
            return solve_rainflow(self, problem, plan, options, deadline)
        o = self.options
        x, theta, battery = plan["state"], plan["rainflow_theta"], plan["battery"]
        shortfall = plan.get("trip_shortfall_kwh",
                             cp.Constant(np.zeros(len(plan["horizon"].dt))))
        initial_soc = float(plan["initial_soc"])
        history = list(self.soc_history)
        if not history or abs(history[-1]-initial_soc) > 1e-10:
            history.append(initial_soc)
        historical_cost = battery.rainflow_cost_gradient(history)[0]
        scale = o.health_weight/plan["N"]
        base_objective = sum(v for k, v in plan["objective_terms"].items() if k != "degradation")
        constraints = list(problem.constraints)
        lower, upper = -np.inf, np.inf
        best, best_values = None, None
        cut_planes = []
        best_soc, best_shortfall, best_incremental = None, None, None
        iterations, cuts = 0, 0
        elapsed_calls, backend_seconds = 0.0, 0.0
        backend_known, converged, timed_out = False, False, False
        approximation_error, last_error = np.nan, None
        tolerance = o.mip_relative_gap if o.formulation == "milp" else o.rainflow_relative_tolerance
        absolute = o.rainflow_absolute_tolerance
        for _ in range(o.rainflow_max_iterations):
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            master = cp.Problem(cp.Minimize(base_objective+scale*theta), constraints)
            iterations += 1
            try:
                solve_ocp(master, o.solver, options,
                          time_limit_seconds=None if o.allow_unlimited_solver_time else remaining,
                          allow_unlimited_time=o.allow_unlimited_solver_time)
            except cp.error.SolverError as error:
                last_error = str(error)
            diagnostic = getattr(master, "hems_diagnostic", {})
            elapsed_calls += diagnostic.get("solve_call_seconds", 0)
            duration = diagnostic.get("backend_solve_seconds")
            if duration is not None:
                backend_seconds += duration
                backend_known = True
            timed_out |= diagnostic.get("timed_out", False)
            if last_error or not usable_solution(master):
                problem = master
                break
            lower = max(lower, master_lower_bound(master, o.solver))
            trial_soc = x.value.copy()
            trial_shortfall = np.array(shortfall.value, copy=True)
            exact_total, gradient, shortfall_gradient = _rainflow_path_cost_gradient(
                battery, history, plan["horizon"], trial_soc, plan["capacity_kwh"],
                trial_shortfall)
            incremental = exact_total-historical_cost
            if incremental < -absolute:
                raise RuntimeError("Rainflow incremental cost became negative")
            incremental = max(0.0, incremental)
            trial_objective = float(base_objective.value)+scale*incremental
            if trial_objective < upper:
                upper = trial_objective
                best = master
                best_soc, best_incremental = trial_soc.copy(), incremental
                best_shortfall = trial_shortfall.copy()
                best_values = {v.id: np.array(v.value, copy=True) for v in master.variables()}
                best_values[theta.id] = np.asarray(incremental)
            if np.isfinite(lower) and lower > upper+absolute+1e-7*max(1, abs(upper)):
                raise RuntimeError("Rainflow lower bound exceeds exact feasible objective")
            gap = max(0.0, upper-lower)
            # Separate whole-objective optimality from the dollar accuracy of
            # the rainflow envelope at the incumbent we would actually accept.
            # Large comfort constants must not hide an omitted aging term.
            envelope = max([0.0, *[cost+g@(best_soc-soc)+eg@(best_shortfall-eps)
                                   for cost, g, soc, eg, eps in cut_planes]])
            approximation_error = max(0.0, best_incremental-envelope)
            model_tolerance = absolute+o.rainflow_relative_tolerance*abs(best_incremental)
            objective_certified = gap <= absolute+tolerance*max(abs(upper), absolute)
            if objective_certified and approximation_error <= model_tolerance:
                converged = True
                break
            if objective_certified:
                # Refine the retained incumbent, which need not be this master's
                # latest candidate, before accepting the full-problem MIP gap.
                trial_soc, incremental = best_soc.copy(), best_incremental
                trial_shortfall = best_shortfall.copy()
                _, gradient, shortfall_gradient = _rainflow_path_cost_gradient(
                    battery, history, plan["horizon"], trial_soc, plan["capacity_kwh"],
                    trial_shortfall)
            # Support the joint affine SOC/shortfall path; history is fixed.
            constraints.append(theta >= incremental + gradient @ (x-trial_soc)
                               + shortfall_gradient @ (shortfall-trial_shortfall))
            cut_planes.append((incremental, gradient.copy(), trial_soc,
                               shortfall_gradient.copy(), trial_shortfall))
            cuts += 1
            problem = master
        timed_out |= time.perf_counter() >= deadline
        if best is not None:
            problem = best
            for v in best.variables():
                v.save_value(best_values[v.id])
            envelope = max([0.0, *[cost+g@(best_soc-soc)+eg@(best_shortfall-eps)
                                   for cost, g, soc, eg, eps in cut_planes]])
            approximation_error = max(0.0, best_incremental-envelope)
        finite_bounds = np.isfinite(upper) and np.isfinite(lower)
        relative_gap = max(0.0, upper-lower)/max(abs(upper), absolute, 1e-12) if finite_bounds else np.inf
        info = getattr(problem, "hems_diagnostic", {})
        info.update(rainflow_iterations=iterations, rainflow_cuts=cuts,
            rainflow_upper_bound=upper, rainflow_lower_bound=lower,
            rainflow_relative_gap=relative_gap, rainflow_converged=converged,
            rainflow_approximation_error_usd=approximation_error,
            rainflow_model_tolerance_usd=(absolute+o.rainflow_relative_tolerance*abs(best_incremental)
                                          if best_incremental is not None else np.nan),
            rainflow_model_certified=bool(best_incremental is not None and
                approximation_error <= absolute+o.rainflow_relative_tolerance*abs(best_incremental)),
            rainflow_absolute_gap=max(0.0, upper-lower) if finite_bounds else np.inf,
            historical_rainflow_cost_usd=historical_cost, exact_objective=upper,
            solve_call_seconds=elapsed_calls, backend_solve_seconds=backend_seconds if backend_known else None,
            timed_out=bool(timed_out), limit_reached=bool(not converged),
            limit_seconds=o.solver_time_limit_seconds, fallback=not converged,
            accepted_incumbent=bool(converged and problem.status == cp.USER_LIMIT),
            outer_status="rainflow_converged" if converged else (last_error or "rainflow_not_converged"))
        problem.hems_diagnostic = info
        return problem, converged

    def _solve_problem(self, problem, options, *, time_limit_seconds, allow_unlimited_time):
        """Ordinary solve dispatch; rainflow retains its certified master loop."""
        solve_ocp(problem, self.options.solver, options,
                  time_limit_seconds=time_limit_seconds,
                  allow_unlimited_time=allow_unlimited_time)

    def _solve(self, timestamp, soc, forecast):
        started, solve_seconds, successful, status_text, problem = time.perf_counter(), 0., False, "error", None
        budget = self.options.solver_time_limit_seconds
        unlimited = self.options.allow_unlimited_solver_time
        deadline = np.inf if unlimited else started + budget
        try:
            prediction = forecast.for_device(self.equipment_name, timestamp, self.horizon)
            problem, plan = self.build_problem(timestamp, soc, prediction)
            options = deepcopy(self.options.solver_options)
            if self.options.formulation == "milp":
                if self.options.solver == "SCIPY":
                    options.setdefault("scipy_options", {})["mip_rel_gap"] = self.options.mip_relative_gap
                elif self.options.solver == "GUROBI":
                    options["MIPGap"] = self.options.mip_relative_gap
            solving = time.perf_counter()
            try:
                if plan["rainflow_theta"] is not None:
                    problem, successful = self._solve_rainflow(problem, plan, options, deadline)
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        self._solve_problem(problem, options,
                                            time_limit_seconds=None if unlimited else remaining,
                                            allow_unlimited_time=unlimited)
                        successful = accept_ev_solution(problem, self.options)
                    else:
                        problem.hems_diagnostic = dict(timed_out=True, limit_reached=True,
                            fallback=True, accepted_incumbent=False,
                            outer_status="time_budget_exhausted_before_solve")
            finally:
                solve_seconds = time.perf_counter()-solving
            diagnostic = getattr(problem, "hems_diagnostic", {})
            diagnostic.setdefault("backend_limit_seconds", diagnostic.get("limit_seconds"))
            diagnostic["limit_seconds"] = budget
            problem.hems_diagnostic = diagnostic
            if "scenario_plans" in plan:
                diagnostic.update(forecast_type="scenario", scenario_count=len(plan["scenario_plans"]),
                                  scenario_ids=list(plan["scenario_ids"]))
                problem.hems_diagnostic = diagnostic
            solve_seconds = diagnostic.get("solve_call_seconds", solve_seconds)
            status_text = diagnostic.get("outer_status", str(problem.status))
            if successful:
                if "shared_first_command" in plan:
                    command = plan["shared_first_command"]
                    if command is None:  # Every scenario is away throughout this control interval.
                        self.target, self.held_mode = 0., 0
                    elif self.options.formulation == "milp":
                        charge, discharge = (float(v.value) > .5 for v in command)
                        self.held_mode = int(charge) - int(discharge)
                        self.target = (self.ev.charger_power_kw if charge else
                                       -self.options.max_discharge_kw if discharge else 0.)
                    else:
                        chemical = float(command[0].value)
                        self.target = (chemical / self.options.charge_efficiency if chemical >= 0
                                       else chemical * self.options.discharge_efficiency)
                        self.held_mode = 1 if self.target > 1e-6 else -1 if self.target < -1e-6 else 0
                else:
                    j = plan["command_index"]
                    self.target = float(plan["ac"].value[j])
                    self.held_mode = 1 if self.target > 1e-6 else -1 if self.target < -1e-6 else 0
                self.plans.append(dict(timestamp=timestamp, boundaries=plan["boundaries"].astype(str).tolist(),
                    soc=plan["state"].value.tolist(), chemical_kw=plan["chemical"].value.tolist(),
                    charge_kw=plan["charge"].value.tolist(), discharge_kw=plan["discharge"].value.tolist(),
                    objective=float(diagnostic.get("exact_objective", problem.value)),
                    objective_terms={k: float(v.value) for k, v in plan["objective_terms"].items()},
                    degradation_method=self.options.degradation_method,
                    measured_soh=plan["measured_soh"], capacity_kwh=plan["capacity_kwh"],
                    terminal_value_usd_per_kwh=plan["terminal_value_usd_per_kwh"],
                    terminal_value_provenance=plan.get("terminal_value_provenance"),
                    initial_energy_kwh=plan["capacity_kwh"]*soc,
                    terminal_energy_kwh=plan["capacity_kwh"]*float(plan["state"].value[-1]),
                    dispatched_target_kw=self.target, dispatched_mode=self.held_mode,
                    normalization_steps=plan["N"], cycle_history_hours=plan["horizon"].history_hours,
                    cycle_future_hours=plan["horizon"].future_hours,
                    observed_history_hours=max(0, (timestamp-(self.first_observation or timestamp)).total_seconds()/3600),
                    discharge_past_soc=plan["discharge_past"], discharge_future_soc=float(plan["discharge_future"].value),
                    **predicted_trip_energy_fields(plan)))
                if "scenario_plans" in plan:
                    self.plans[-1].update(scenario_metadata=plan["scenario_metadata"],
                        representative_scenario_id=plan["scenario_ids"][0],
                        scenarios=[dict(candidate_id=identifier, boundaries=p["boundaries"].astype(str).tolist(),
                            soc=p["state"].value.tolist(), chemical_kw=p["chemical"].value.tolist(),
                            charge_kw=p["charge"].value.tolist(), discharge_kw=p["discharge"].value.tolist(),
                            available=p["horizon"].z.tolist(), trip_energy_kwh=p["horizon"].trip_kwh.tolist(),
                            **predicted_trip_energy_fields(p),
                            departure_indices=p["horizon"].departure_indices.tolist(),
                            objective_terms={k: float(v.value) for k, v in p["objective_terms"].items()},
                            terminal_energy_kwh=p["capacity_kwh"]*float(p["state"].value[-1]))
                            for identifier, p in zip(plan["scenario_ids"], plan["scenario_plans"])])
        except NoEVScenarios as error:
            status_text = str(error)
        except cp.error.SolverError as error:
            status_text = str(error)
        self.fallback = not successful
        diagnostic = getattr(problem, "hems_diagnostic", {})
        diagnostic.setdefault("limit_seconds", budget)
        self.solve_history.append(dict(timestamp=timestamp, status=status_text, successful=successful,
            measured_soh=self.current_soh, capacity_kwh=self.current_capacity,
            solve_seconds=solve_seconds, ocp_seconds=time.perf_counter()-started,
            **diagnostic))
        self.solve_history[-1]["fallback"] = self.fallback
        self.next_solve = timestamp+pd.Timedelta(minutes=self.options.control_minutes)

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        timestamp = pd.Timestamp(timestamp)
        status = dict(previous_status or {})
        status.update(getattr(forecast, "ev_states", {}))
        soc = float(np.clip(status[f"{self.equipment_name} SOC (-)"], 0, 1))
        mobility = forecast.ev_schedules[self.ev.vehicle_id]
        available = bool(mobility.available([timestamp])[0])
        backed = float(forecast.backed_load_kw.at[timestamp])
        step = forecast.data.index[1]-forecast.data.index[0]
        self._observe(timestamp, float(status.get(f"{self.equipment_name} Electric Power (kW)", 0)), step)
        # Plant measurements refresh every step; build_problem freezes these
        # coefficients only for the current OCP horizon, never across solves.
        soh_key = f"{self.equipment_name} SOH (-)"
        if soh_key not in status:
            raise ValueError(f"{self.equipment_name}: current plant SOH is required")
        self.current_soh = float(status[soh_key])
        if not np.isfinite(self.current_soh) or not 0 < self.current_soh <= 1:
            raise ValueError("EV measured SOH must be in (0,1]")
        self.current_capacity = self.ev.capacity_kwh*self.current_soh
        reported_capacity = status.get(f"{self.equipment_name} Capacity (kWh)", self.current_capacity)
        if not np.isclose(reported_capacity, self.current_capacity, rtol=1e-9, atol=1e-10):
            raise ValueError("EV measured SOH and capacity are inconsistent")
        # Previous end-SOC precedes this boundary's trip impulse. Preserve both
        # extrema so driving cannot cancel an immediately preceding charge.
        self.soc_history.extend([float(status.get(f"{self.equipment_name} End SOC (-)", soc)), soc])
        self.soc_history = np.asarray(self.soc_history)[turning_points(self.soc_history)].tolist()
        if self.next_solve is None or timestamp >= self.next_solve:
            self._solve(timestamp, soc, forecast)
        target = self.target
        if self.fallback:
            # Baseline fallback is target tracking, not a stale optimized mode.
            target = min(self.ev.charger_power_kw, max(0, self.options.comfort_soc_min-soc)*
                         self.current_capacity/(step.total_seconds()/3600*self.options.charge_efficiency)) if available else 0
            mode, power, override = int(target > 0), target, False
        else:
            if self.options.formulation == "milp":
                target = (self.ev.charger_power_kw if self.held_mode == 1 else
                          -min(backed, self.options.max_discharge_kw) if self.held_mode == -1 else 0)
            mode, power, override = self._project(timestamp, target, soc, available, backed)
        self.action_history.append(dict(timestamp=timestamp, requested_kw=target, projected_kw=power,
            mode=mode, projection_override=override, fallback=self.fallback))
        return {self.equipment_name: {"P Setpoint": power, "Backed Load (kW)": backed}}


class EVBaselineController:
    """Native EVs retain native behavior; opted-in EVs charge to comfort minimum."""
    def __init__(self, ev_names=(), **kwargs):
        self.ev_names = ev_names

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        return {}


class EVScenarioController(EVMPCController):
    """Named scenario variant; shares EV options, solve clock and physical projector.

    ScenarioEVForecast supplies paths through Forecast.for_device. No separate
    simulation loop or alternative EV physics is introduced.
    """


@dataclass
class EVPredictiveController:
    """Starter EV controller that blocks charging during the TOU peak."""

    ev_names: tuple[str, ...] = ("EV_01A",)
    horizon: pd.Timedelta = pd.Timedelta(hours=18)
    no_charge_price: float = 0.30

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Block peak charging; replace this with predictive scheduling logic."""
        # TODO: forecast.window(timestamp, self.horizon) provides prices through
        # the planning horizon. previous_status contains SOC and home/away status.
        if forecast.price_at(timestamp) >= self.no_charge_price:
            return {name: {"Load Fraction": 0} for name in self.ev_names}
        return {}

"""EV baseline, convex and MILP MPC: formulation, optional costs and dispatch."""
from copy import deepcopy
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from .solver import LIMIT_SECONDS, solve_ocp, usable_solution
from dataclasses import dataclass
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation, turning_points
from ..control_forecasts import EVScenarioSet
from ..ev_scenarios import NoEVScenarios


def daily_terminal_energy_value(forecast, timestamp):
    """v = full-calendar-day minimum known tariff, valued per chemical kWh.

    User reconfirmed preserving the existing minimum on 2026-10-17. The older
    terminal tests still assert the previous mean; see docs/validation/README.md.
    Use the minimum DIRECTLY, not minimum/eta_charge or minimum*eta_discharge.
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
    return float(prices.min())


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

    @classmethod
    def from_forecast(cls, c, timestamp, forecast):
        o = c.options
        end = timestamp + c.horizon
        data = forecast.window(timestamp, c.horizon)
        step = forecast.data.index[1] - forecast.data.index[0]
        if data.empty or data.index[0] != timestamp or data.index[-1]+step < end:
            raise ValueError("EV forecast does not cover the complete horizon")
        if forecast.backed_load_kw is None:
            raise ValueError("EV forecast requires backed-load data")
        load = forecast.backed_load_kw.reindex(data.index)
        if load.isna().any() or not np.isfinite(load).all() or (load < -1e-7).any():
            raise ValueError("Invalid or incomplete backed-load forecast")
        mobility = forecast.ev_schedules[c.ev.vehicle_id]
        trips = mobility.demand(o.consumption_kwh_per_km)
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
        # Sorted forecast rows: slice the identical half-open blocks directly.
        # Preserve per-block reductions and exact mobility boundaries; no rebinning.
        if not data.index.is_monotonic_increasing:
            raise ValueError("EV forecast timestamps must be sorted")
        offsets = data.index.searchsorted(t, side="left")
        price_values = data["electricity_price_usd_per_kwh"].to_numpy()
        load_values = load.to_numpy()
        for left, right in zip(offsets[:-1], offsets[1:]):
            pi = price_values[left:right]
            L = load_values[left:right]
            cap = np.minimum(L, o.max_discharge_kw) if o.v2h else np.zeros(len(L))
            if not np.isfinite(pi).all():
                raise ValueError("EV electricity prices must be finite")
            # Signed chemical power held over a block must satisfy every load cap.
            means.append(cap.mean()); minima.append(cap.min()); prices.append(pi.mean())
            dc_cost.append(np.mean(pi*cap)); backed_cost.append(np.mean(pi*L))
            if (o.formulation == "convex" and o.charge_efficiency*o.discharge_efficiency < 1
                    and z[len(means)-1] and cap.min() > 0 and (pi < 0).any()):
                raise ValueError("Negative prices with bidirectional convex EV power are nonconvex; explicitly select MILP")
        trip_kwh = np.zeros(len(t)-1)
        for trip in trips.itertuples():
            # x[0] is measured AFTER this boundary's plant events. Never debit
            # an ongoing/current-boundary trip again; horizon end is exclusive.
            if timestamp < trip.departure < end:
                trip_kwh[t.get_loc(trip.departure)-1] += trip.energy_kwh
        group = ((t[:-1]-timestamp)/pd.Timedelta(minutes=o.control_minutes)).astype(int).to_numpy()
        budget_dt = np.array([max(0, (min(b, budget_end)-a).total_seconds()/3600)
                              for a, b in zip(t[:-1], t[1:])])
        # Pre-departure SOC at exact mobility boundaries; use [timestamp, end).
        departures = trips.loc[trips.departure >= timestamp, "departure"].drop_duplicates()
        departure_indices = t.get_indexer(departures)
        return cls(t, dt, z, np.array(prices), np.array(backed_cost), np.array(dc_cost),
                   np.array(means), np.array(minima), trip_kwh, group,
                   int(np.ceil(o.horizon_hours*60/o.control_minutes)), F, 24-F, budget_dt,
                   departure_indices)


def _same_control_interval(constraints, q, h):
    """Mobility integration boundaries do not create extra decision opportunities."""
    for group in np.unique(h.group):
        active = np.flatnonzero((h.group == group) & (h.z == 1))
        for j in active[1:]:
            constraints.append(q[j] == q[active[0]])


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


def build_ev_problem(c, timestamp, initial_soc, forecast):
    """EV OCP, in the same data -> variables -> constraints -> cost order as HPWH.

    x = post-boundary battery SOC; x_pre = SOC before a departure impulse.
    p = signed chemical kW; P = signed AC kW; z = availability.
    u_c/u_d = nonnegative AC power; b_c/b_d = exclusive MILP operating modes.
    N counts control intervals, while j indexes mobility-split integration blocks.

    min J_electricity + J_persistence + J_terminal + J_trip
        + w_comfort*H_comfort
        + w_health * (w_health_SOC*(H_low + H_high) + C_degradation/N).

    Health terms integrate using dt/N; comfort averages pre-departure shortfalls.
    Trip cost averages unmet_kWh/E over new departures, not control intervals.
    C_degradation is already
    cumulative dollars: do not multiply it by dt again. With rainflow enabled,
    this builds a MASTER problem; _solve_rainflow supplies its supporting cuts
    and certifies the exact objective before an action can be accepted.
    """
    o, E = c.options, c.current_capacity
    h = EVHorizon.from_forecast(c, timestamp, forecast)
    n = len(h.dt)
    x = cp.Variable(n+1, name="x_ev")
    constraints = [x[0] == initial_soc, x >= 0, x <= 1]
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

    # (2) SOC dynamics. Trip demand stays distance-derived, but infeasible
    #     forecast paths receive a bounded, explicitly penalized unmet-energy
    #     slack. The simulated plant still attempts the complete realized trip,
    #     clips physical SOC at zero, and reports its actual unmet energy.
    trip_blocks = np.flatnonzero(h.trip_kwh > 0)
    if len(trip_blocks):
        trip_shortfall = cp.Variable(len(trip_blocks), nonneg=True, name="trip_shortfall_kwh")
        selector = np.zeros((n, len(trip_blocks)))
        selector[trip_blocks, np.arange(len(trip_blocks))] = 1
        constraints.append(trip_shortfall <= h.trip_kwh[trip_blocks])
        shortfall_by_block = selector @ trip_shortfall
        served_trip_kwh = h.trip_kwh-shortfall_by_block
        trip_shortfall_total = cp.sum(trip_shortfall)
    else:
        trip_shortfall = None
        shortfall_by_block = cp.Constant(np.zeros(n))
        served_trip_kwh = h.trip_kwh
        trip_shortfall_total = cp.Constant(0)
    constraints.append(x[1:] == x[:-1] + (cp.multiply(h.dt, p)-served_trip_kwh)/E)
    # At a departure, x is POST-debit; x_pre restores the served debit only.
    # Bounding both prevents charging above capacity immediately before a trip.
    x_pre = cp.hstack([x[0], x[1:]+served_trip_kwh/E])
    constraints += [x_pre >= 0, x_pre <= 1]
    # Include both sides of each impulse in rainflow: otherwise charging just
    # before departure can disappear when netted with the driving debit.
    event_states = [x[0]]
    for j in range(1, n+1):
        if h.trip_kwh[j-1] > 0:
            event_states.append(x_pre[j])
        event_states.append(x[j])
    rainflow_state = cp.hstack(event_states)

    # Average over new trips in this horizon (including zero-demand trips),
    # retaining capacity normalization. Already-debited trips are excluded.
    trip_count = int(np.count_nonzero(h.departure_indices > 0))
    J_trip = o.trip_weight * trip_shortfall_total / (E*max(1, trip_count))

    # (3) Cumulative V2H battery discharge (not driving or SOC excursion).
    D_past = c.discharge_history(h.t[0]-pd.Timedelta(hours=h.history_hours), h.t[0], normalized=True)
    D_future = cp.sum(cp.multiply(h.budget_dt, discharge_chemical)) / E
    if o.enforce_discharge_budget:
        constraints.append(D_past+D_future <= o.discharge_budget_soc)

    # (4) Soft health band throughout; comfort only immediately before departure.
    health_low = cp.sum(cp.multiply(h.dt, cp.pos(o.health_soc_min-x[:-1]))) / h.N
    health_high = cp.sum(cp.multiply(h.dt, cp.pos(x[:-1]-o.health_soc_max))) / h.N
    departures = h.departure_indices
    comfort_soc = x_pre
    if 0 in departures and np.isfinite(getattr(c, "current_departure_soc", np.nan)):
        comfort_soc = cp.hstack([c.current_departure_soc, x_pre[1:]])
    #comfort = (cp.sum(cp.pos(o.comfort_soc_min-comfort_soc[departures])) / len(departures)
    #           if len(departures) else cp.Constant(0))

    comfort = cp.sum(cp.multiply(h.dt, cp.pos(o.comfort_soc_min-x[:-1]))) / h.N

    # (5) Optional monetary degradation, with the same physical calibration used
    #     for realized-trajectory metrics. Health and comfort weights are separate.
    battery = BatteryDegradation(c.ev.capacity_kwh, o, current_capacity_kwh=E)
    degradation = cp.Constant(0)
    theta = None
    if o.enable_degradation and o.degradation_method != "none" and o.health_weight > 0 and battery.depreciable_usd > 0:
        if o.degradation_method == "linear_throughput":
            # Battery-side throughput; driving demand is an additional fixed term.
            throughput = cp.sum(cp.multiply(h.dt, cp.abs(p))) + h.trip_kwh.sum()
            degradation = battery.throughput_usd_per_kwh * throughput / h.N
        else:
            # Exact rainflow is evaluated by the outer loop, not a CVXPY atom.
            theta = cp.Variable(nonneg=True, name="rainflow_increment_usd")
            degradation = theta / h.N
    # (6) Terminal energy value: v/N * (e_0-e_N), e_j = C_k*x_j.
    # C_k is measured faded capacity, fixed for this solve. This affine term
    # prices depletion AND credits retained energy; it is not a moving SOC floor.
    v = daily_terminal_energy_value(forecast, timestamp) if o.enforce_terminal_soc else 0.0
    J_terminal = v * E * (initial_soc-x[-1]) / h.N
    J = dict(electricity=o.electricity_weight*energy_cost,
             trip_shortfall=J_trip,
             health_soc=o.health_weight*o.health_soc_weight*(health_low+health_high),
             degradation=o.health_weight*degradation,
             comfort=o.comfort_weight*comfort,
             persistence=o.persistence_weight*J_pers, terminal=J_terminal)
    problem = cp.Problem(cp.Minimize(sum(J.values())), constraints)
    if not problem.is_dcp():
        raise ValueError("EV formulation is not DCP; check bidirectional negative-price inputs")
    first_active = np.flatnonzero((h.group == 0) & (h.z == 1))
    return problem, dict(state=x, pre_departure_state=comfort_soc, rainflow_state=rainflow_state,
        chemical=p, ac=P, charge=u_c, discharge=u_d, fraction=fraction,
        charge_mode=b_c if o.formulation == "milp" else None,
        discharge_mode=b_d if o.formulation == "milp" else None,
        boundaries=h.t, command_index=int(first_active[0]) if len(first_active) else 0,
        N=h.N, horizon=h, starts=mode_starts, ends=mode_ends, discharge_past=D_past,
        measured_soh=c.current_soh, capacity_kwh=E,
        trip_shortfall_variable=trip_shortfall, trip_shortfall_by_block=shortfall_by_block,
        trip_departure_blocks=trip_blocks, trip_count=trip_count,
        terminal_value_usd_per_kwh=v,
        initial_soc=initial_soc, rainflow_theta=theta, battery=battery, health_low=health_low, health_high=health_high,
        discharge_future=D_future, objective_terms=J, **J)


class EVMPCController:
    """Fixed-clock MPC with measured SOH, observed history and safe actuation."""
    def __init__(self, ev):
        self.ev, self.options = ev, ev.hems
        if self.options.enforce_terminal_soc and self.options.terminal_soc_weight != 0:
            warnings.warn("EV terminal_soc_weight is retired and ignored: terminal cost is "
                          "daily minimum tariff * (initial-final chemical kWh) / N. "
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
        if self.options.solver not in cp.installed_solvers():
            raise ValueError(f"EV {ev.vehicle_id}: solver {self.options.solver} not installed; "
                             "install its dependency/license or select another backend")

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

        Each feasible SOC gives an upper bound using exact cycle counting. Each
        master dual bound is a lower bound. Native MILP gap alone cannot certify
        this problem. History is fixed; its already-accounted cost is subtracted.
        """
        if "scenario_plans" in plan:
            from .ev_scenario import solve_rainflow
            return solve_rainflow(self, problem, plan, options, deadline)
        o = self.options
        x, theta, battery = plan["rainflow_state"], plan["rainflow_theta"], plan["battery"]
        initial_soc = float(plan["initial_soc"])
        history = list(self.soc_history)
        if not history or abs(history[-1]-initial_soc) > 1e-10:
            history.append(initial_soc)
        prefix = np.asarray(history[:-1])
        historical_cost = battery.rainflow_cost_gradient(history)[0]
        scale = o.health_weight/plan["N"]
        base_objective = sum(v for k, v in plan["objective_terms"].items() if k != "degradation")
        constraints = list(problem.constraints)
        lower, upper = -np.inf, np.inf
        best, best_values = None, None
        cut_planes = []
        best_soc, best_incremental = None, None
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
                solve_ocp(master, o.solver, options, time_limit_seconds=remaining)
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
            exact_total, gradient = battery.rainflow_cost_gradient(np.r_[prefix, trial_soc])
            incremental = exact_total-historical_cost
            if incremental < -absolute:
                raise RuntimeError("Rainflow incremental cost became negative")
            incremental = max(0.0, incremental)
            trial_objective = float(base_objective.value)+scale*incremental
            if trial_objective < upper:
                upper = trial_objective
                best = master
                best_soc, best_incremental = trial_soc.copy(), incremental
                best_values = {v.id: np.array(v.value, copy=True) for v in master.variables()}
                best_values[theta.id] = np.asarray(incremental)
            if np.isfinite(lower) and lower > upper+absolute+1e-7*max(1, abs(upper)):
                raise RuntimeError("Rainflow lower bound exceeds exact feasible objective")
            gap = max(0.0, upper-lower)
            # Separate whole-objective optimality from the dollar accuracy of
            # the rainflow envelope at the incumbent we would actually accept.
            # Large comfort constants must not hide an omitted aging term.
            envelope = max([0.0, *[cost+g@(best_soc-soc) for cost, g, soc in cut_planes]])
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
                _, gradient = battery.rainflow_cost_gradient(np.r_[prefix, trial_soc])
            # F(s) >= F(s_k) + g_k' (s-s_k); past coordinates are constants.
            g_future = gradient[len(prefix):]
            constraints.append(theta >= incremental + g_future @ (x-trial_soc))
            cut_planes.append((incremental, g_future.copy(), trial_soc))
            cuts += 1
            problem = master
        timed_out |= time.perf_counter() >= deadline
        if best is not None:
            problem = best
            for v in best.variables():
                v.save_value(best_values[v.id])
            envelope = max([0.0, *[cost+g@(best_soc-soc) for cost, g, soc in cut_planes]])
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
            limit_seconds=LIMIT_SECONDS, fallback=not converged,
            accepted_incumbent=bool(converged and problem.status == cp.USER_LIMIT),
            outer_status="rainflow_converged" if converged else (last_error or "rainflow_not_converged"))
        problem.hems_diagnostic = info
        return problem, converged

    def _solve(self, timestamp, soc, forecast):
        started, solve_seconds, successful, status_text, problem = time.perf_counter(), 0., False, "error", None
        try:
            prediction = forecast.for_device(self.equipment_name, timestamp, self.horizon)
            problem, plan = self.build_problem(timestamp, soc, prediction)
            options = deepcopy(self.options.solver_options)
            if self.options.solver == "GUROBI":
                # Classify infeasible versus unbounded in one backend call;
                # an automatic retry could consume a second native time budget.
                options.setdefault("DualReductions", 0)
            if self.options.formulation == "milp":
                if self.options.solver == "SCIPY":
                    options.setdefault("scipy_options", {})["mip_rel_gap"] = self.options.mip_relative_gap
                elif self.options.solver == "GUROBI":
                    options["MIPGap"] = self.options.mip_relative_gap
            solving = time.perf_counter()
            try:
                if plan["rainflow_theta"] is not None:
                    problem, successful = self._solve_rainflow(problem, plan, options, started+LIMIT_SECONDS)
                else:
                    solve_ocp(problem, self.options.solver, options,
                              time_limit_seconds=max(1e-6, started+LIMIT_SECONDS-time.perf_counter()))
                    successful = accept_ev_solution(problem, self.options)
            finally:
                solve_seconds = time.perf_counter()-solving
            diagnostic = getattr(problem, "hems_diagnostic", {})
            if "scenario_plans" in plan:
                diagnostic.update(forecast_type="scenario", scenario_count=len(plan["scenario_plans"]),
                                  scenario_ids=list(plan["scenario_ids"]))
                problem.hems_diagnostic = diagnostic
            solve_seconds = diagnostic.get("solve_call_seconds", solve_seconds)
            status_text = diagnostic.get("outer_status", str(problem.status))
            if successful:
                path_plans = plan.get("scenario_plans", [plan])
                predicted_shortfalls = [float(np.sum(p["trip_shortfall_by_block"].value)) for p in path_plans]
                diagnostic.update(predicted_trip_shortfall_kwh_mean=float(np.mean(predicted_shortfalls)),
                    predicted_trip_shortfall_kwh_max=float(np.max(predicted_shortfalls)),
                    predicted_scenarios_with_trip_shortfall=sum(v > 1e-7 for v in predicted_shortfalls))
                j = plan["command_index"]
                self.target = float(plan["ac"].value[j])
                self.held_mode = 1 if self.target > 1e-6 else -1 if self.target < -1e-6 else 0
                self.plans.append(dict(timestamp=timestamp, boundaries=plan["boundaries"].astype(str).tolist(),
                    soc=plan["state"].value.tolist(), rainflow_soc=plan["rainflow_state"].value.tolist(),
                    chemical_kw=plan["chemical"].value.tolist(),
                    charge_kw=plan["charge"].value.tolist(), discharge_kw=plan["discharge"].value.tolist(),
                    objective=float(diagnostic.get("exact_objective", problem.value)),
                    objective_terms={k: float(v.value) for k, v in plan["objective_terms"].items()},
                    degradation_method=self.options.degradation_method,
                    measured_soh=plan["measured_soh"], capacity_kwh=plan["capacity_kwh"],
                    terminal_value_usd_per_kwh=plan["terminal_value_usd_per_kwh"],
                    predicted_trip_shortfall_kwh=predicted_shortfalls[0],
                    trip_energy_timing="departure", trip_count=plan["trip_count"],
                    trip_departure_blocks=plan["trip_departure_blocks"].tolist(),
                    pre_departure_soc=plan["pre_departure_state"].value.tolist(),
                    trip_energy_kwh=plan["horizon"].trip_kwh.tolist(),
                    trip_shortfall_kwh=plan["trip_shortfall_by_block"].value.tolist(),
                    initial_energy_kwh=plan["capacity_kwh"]*soc,
                    terminal_energy_kwh=plan["capacity_kwh"]*float(plan["state"].value[-1]),
                    normalization_steps=plan["N"], cycle_history_hours=plan["horizon"].history_hours,
                    cycle_future_hours=plan["horizon"].future_hours,
                    observed_history_hours=max(0, (timestamp-(self.first_observation or timestamp)).total_seconds()/3600),
                    discharge_past_soc=plan["discharge_past"], discharge_future_soc=float(plan["discharge_future"].value)))
                if "scenario_plans" in plan:
                    self.plans[-1].update(scenario_metadata=plan["scenario_metadata"],
                        representative_scenario_id=plan["scenario_ids"][0],
                        scenarios=[dict(candidate_id=identifier, boundaries=p["boundaries"].astype(str).tolist(),
                            soc=p["state"].value.tolist(), rainflow_soc=p["rainflow_state"].value.tolist(),
                            chemical_kw=p["chemical"].value.tolist(),
                            charge_kw=p["charge"].value.tolist(), discharge_kw=p["discharge"].value.tolist(),
                            available=p["horizon"].z.tolist(), trip_energy_kwh=p["horizon"].trip_kwh.tolist(),
                            predicted_trip_shortfall_kwh=float(np.sum(p["trip_shortfall_by_block"].value)),
                            trip_shortfall_kwh=p["trip_shortfall_by_block"].value.tolist(),
                            departure_indices=p["horizon"].departure_indices.tolist(),
                            pre_departure_soc=p["pre_departure_state"].value.tolist(),
                            trip_count=p["trip_count"], trip_energy_timing="departure",
                            objective_terms={k: float(v.value) for k, v in p["objective_terms"].items()},
                            terminal_energy_kwh=p["capacity_kwh"]*float(p["state"].value[-1]))
                            for identifier, p in zip(plan["scenario_ids"], plan["scenario_plans"])])
        except NoEVScenarios as error:
            status_text = str(error)
        except cp.error.SolverError as error:
            status_text = str(error)
        self.fallback = not successful
        self.solve_history.append(dict(timestamp=timestamp, status=status_text, successful=successful,
            measured_soh=self.current_soh, capacity_kwh=self.current_capacity,
            solve_seconds=solve_seconds, ocp_seconds=time.perf_counter()-started,
            **getattr(problem, "hems_diagnostic", {})))
        self.solve_history[-1]["fallback"] = self.fallback
        self.next_solve = timestamp+pd.Timedelta(minutes=self.options.control_minutes)

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        timestamp = pd.Timestamp(timestamp)
        status = previous_status or {}
        soc = float(np.clip(status[f"{self.equipment_name} SOC (-)"], 0, 1))
        self.current_departure_soc = float(status.get(f"{self.equipment_name} Pre-departure SOC (-)", np.nan))
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
        self.soc_history.extend([float(status.get(f"{self.equipment_name} End SOC (-)", soc))])
        if np.isfinite(self.current_departure_soc):
            self.soc_history.append(self.current_departure_soc)
        self.soc_history.append(soc)
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

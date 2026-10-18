"""Independent EV OCPs with explicit convex relaxation and actuator projection.

Trips use battery-side kWh inferred from distance. Return saturation is exact
in MILP and relaxed in convex mode; the simulated equipment is always exact.
"""
import time
import numpy as np
import pandas as pd
import cvxpy as cp


class EVMPCController:
    def __init__(self, ev):
        self.ev = ev
        self.options = ev.hems
        self.equipment_name = f"EV_{ev.vehicle_id}"
        self.horizon = pd.Timedelta(hours=self.options.horizon_hours)
        self.solve_history = []
        self.plans = []
        self.action_history = []
        self.starts = {1: [], -1: []}
        self.last_change = {1: None, -1: None}
        self.previous_mode = 0
        self.previous_power = 0.0
        self.next_solve = None
        self.target = 0.0
        self.held_mode = 0
        if self.options.solver not in cp.installed_solvers():
            raise ValueError(f"EV {ev.vehicle_id}: solver {self.options.solver} not installed; "
                             "install its dependency/license or select another backend")

    def _observe(self, timestamp, power, plant_step):
        mode = 1 if power > 1e-7 else -1 if power < -1e-7 else 0
        observed_at = timestamp - plant_step
        if mode != self.previous_mode:
            for direction in (1, -1):
                if (mode == direction) != (self.previous_mode == direction):
                    self.last_change[direction] = observed_at
                if mode == direction:
                    self.starts[direction].append(observed_at)
        self.previous_mode, self.previous_power = mode, power
        for direction in (1, -1):
            self.starts[direction] = [t for t in self.starts[direction]
                                      if t > timestamp - pd.Timedelta(hours=24)]

    def _remaining_dwell(self, timestamp, direction):
        changed = self.last_change[direction]
        if self.options.cycling != "dwell" or changed is None:
            return pd.Timedelta(0)
        minutes = (self.options.dwell_on_minutes if self.previous_mode == direction
                   else self.options.dwell_off_minutes)
        return max(pd.Timedelta(0), changed + pd.Timedelta(minutes=minutes) - timestamp)

    def _cycle_cap(self, direction):
        return (self.options.max_charge_starts_24h if direction == 1
                else self.options.max_discharge_starts_24h)

    def build_problem(self, timestamp, soc, forecast):
        """Build event-aligned blocks; integrate capped backed load at plant resolution."""
        o, ev = self.options, self.ev
        mobility = forecast.ev_schedules[ev.vehicle_id]
        end = timestamp + self.horizon
        data = forecast.window(timestamp, self.horizon)
        if len(data) < 1 or forecast.backed_load_kw is None:
            raise ValueError("EV MPC requires full prices, mobility, and backed-load forecast")
        plant_step = forecast.data.index[1] - forecast.data.index[0]
        if data.index[0] != timestamp or data.index[-1] + plant_step < end:
            raise ValueError("EV forecast does not cover the complete horizon")
        trips = mobility.demand(o.consumption_kwh_per_km)
        trips = trips.loc[(trips.departure < end) & (trips.arrival > timestamp)]
        boundaries = set(pd.date_range(timestamp, end, freq=pd.Timedelta(minutes=o.control_minutes)))
        boundaries.update([timestamp, end])
        for trip in trips.itertuples():
            boundaries.update(t for t in (trip.departure, trip.arrival) if timestamp < t < end)
        boundaries = pd.DatetimeIndex(sorted(boundaries))
        n = len(boundaries)-1
        hours = np.diff(boundaries.asi8)/3.6e12
        available = mobility.available(boundaries[:-1]).astype(float)
        load = forecast.backed_load_kw.reindex(data.index)
        if load.isna().any() or not np.isfinite(load).all() or (load < -1e-7).any():
            raise ValueError("Invalid or incomplete backed-load forecast")
        cap, charge_cost, discharge_cost, backed_cost = [], [], [], []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            selected = (data.index >= a) & (data.index < b)
            price = data.loc[selected, "electricity_price_usd_per_kwh"].to_numpy()
            limited = np.minimum(load.loc[selected].to_numpy(), o.max_discharge_kw) if o.v2h else np.zeros(len(price))
            cap.append(float(limited.mean()))
            charge_cost.append(float(price.mean()))
            discharge_cost.append(float(np.mean(price*limited)))
            backed_cost.append(float(np.mean(price*load.loc[selected].to_numpy())))
        cap = np.array(cap)
        state = cp.Variable(n+1)
        charge = cp.Variable(n, nonneg=True)
        discharge_fraction = cp.Variable(n, nonneg=True)
        discharge = cp.multiply(cap, discharge_fraction)
        constraints = [state[0] == soc, state >= 0, state <= 1,
                       charge <= ev.charger_power_kw*available,
                       discharge_fraction <= available]
        if not o.v2h:
            constraints.append(discharge_fraction == 0)
        modes = {}
        if o.formulation == "milp":
            modes = {1: cp.Variable(n, boolean=True), -1: cp.Variable(n, boolean=True)}
            constraints += [charge == ev.charger_power_kw*modes[1],
                            discharge_fraction == modes[-1], modes[1]+modes[-1] <= 1]
            for direction, binary in modes.items():
                can_run = available * (np.ones(n) if direction == 1 else (cap > 1e-9))
                constraints += [binary <= can_run]
                on, off = cp.Variable(n, boolean=True), cp.Variable(n, boolean=True)
                prior = int(self.previous_mode == direction)
                constraints += [binary - cp.hstack([prior, binary[:-1]]) == on-off, on+off <= 1]
                remaining = self._remaining_dwell(timestamp, direction)
                for j, t in enumerate(boundaries[:-1]):
                    if t < timestamp+remaining and (not prior or can_run[j]):
                        constraints.append(binary[j] == prior)
                    cycle_cap = self._cycle_cap(direction)
                    if cycle_cap is not None:
                        cutoff = t-pd.Timedelta(hours=24)
                        past = sum(s > cutoff for s in self.starts[direction])
                        indices = np.flatnonzero((boundaries[:-1] > cutoff) & (boundaries[:-1] <= t))
                        constraints.append(cp.sum(on[indices])+past <= cycle_cap)
                    if o.cycling == "dwell":
                        for transition, active, minutes in ((on, binary, o.dwell_on_minutes),
                                                            (off, 1-binary, o.dwell_off_minutes)):
                            for k in range(j, n):
                                if boundaries[k] >= t+pd.Timedelta(minutes=minutes):
                                    break
                                if transition is on and not can_run[k]:
                                    break  # known physical interruption overrides on dwell
                                constraints.append(active[k] >= transition[j])
        shortfalls, departure_indices, returns = [], [], {}
        for trip in trips.itertuples():
            j = int(boundaries.searchsorted(max(timestamp, trip.departure)))
            demand = float(trip.energy_kwh)
            q = cp.Variable(nonneg=True)
            deficit = demand - ev.capacity_kwh*state[j]
            constraints += [q >= deficit, q <= demand]
            if o.formulation == "milp":
                short = cp.Variable(boolean=True)
                constraints += [q <= demand*short,
                                q <= deficit + ev.capacity_kwh*(1-short)]
            if trip.arrival <= end:
                k = int(boundaries.searchsorted(trip.arrival))
                returns[k] = (demand-q)/ev.capacity_kwh
            shortfalls.append(q)
            departure_indices.append((j, demand))
        for j in range(n):
            constraints.append(state[j+1] == state[j] + hours[j]/ev.capacity_kwh *
                               (o.charge_efficiency*charge[j] - discharge[j]/o.discharge_efficiency)
                               - returns.get(j+1, 0))
        electricity = o.electricity_weight/n * cp.sum(cp.multiply(hours,
            np.array(backed_cost) + cp.multiply(charge_cost, charge) - cp.multiply(discharge_cost, discharge_fraction)))
        health = (o.health_low_weight*cp.sum(cp.pos(o.health_min_soc-state[1:])) +
                  o.health_high_weight*cp.sum(cp.pos(state[1:]-o.health_max_soc)))
        trip_cost = o.trip_weight*sum(shortfalls, cp.Constant(0))/ev.capacity_kwh
        persistence = cp.Constant(0)
        if o.cycling == "persistence":
            dc = cp.hstack([charge[0]-max(self.previous_power, 0), charge[1:]-charge[:-1]])
            dd = cp.hstack([discharge[0]-max(-self.previous_power, 0), discharge[1:]-discharge[:-1]])
            persistence = o.persistence_weight/n * cp.sum(cp.multiply(hours, cp.abs(dc)+cp.abs(dd)))
        problem = cp.Problem(cp.Minimize(electricity+health+trip_cost+persistence), constraints)
        return problem, dict(state=state, charge=charge, discharge=discharge,
                             fraction=discharge_fraction, shortfalls=shortfalls,
                             departures=departure_indices, boundaries=boundaries,
                             electricity=electricity, health=health, trip_cost=trip_cost,
                             persistence=persistence)

    def _project(self, timestamp, target, soc, available, backed):
        """Nearest/threshold projection over discrete modes; honor realized history.

        Nearest minimizes tracking_weight*(P-target)^2 plus switch_weight and,
        for persistence, persistence_weight*|P-previous P|. Threshold first maps
        the target to idle/rated charging/load-following discharge, then applies
        the same feasibility/selection rule. Hardware/SOC saturation is in plant.
        """
        o = self.options
        candidates = {0: 0.0}
        if available and soc < 1-1e-9:
            candidates[1] = self.ev.charger_power_kw
        if available and soc > 1e-9 and o.v2h and backed > 1e-9:
            candidates[-1] = -min(backed, o.max_discharge_kw)
        if o.projection == "threshold":
            reference = self.ev.charger_power_kw if target >= 0 else min(backed, o.max_discharge_kw)
            target = (np.sign(target)*reference if reference and abs(target) >= o.projection_threshold*reference else 0)
        feasible = {}
        for mode, power in candidates.items():
            valid = True
            for direction in (1, -1):
                if (mode == direction) != (self.previous_mode == direction):
                    if self._remaining_dwell(timestamp, direction) > pd.Timedelta(0):
                        # Availability/SOC/load can make the held mode impossible.
                        if not (self.previous_mode == direction and direction not in candidates):
                            valid = False
                if mode == direction and self.previous_mode != direction:
                    cap = self._cycle_cap(direction)
                    if cap is not None and len(self.starts[direction]) >= cap:
                        valid = False
            if valid:
                feasible[mode] = power
        override = (not feasible or (self.previous_mode != 0 and self.previous_mode not in candidates
                                    and self._remaining_dwell(timestamp, self.previous_mode) > pd.Timedelta(0)))
        if not feasible:
            feasible = {0: 0.0}
        def cost(mode):
            p = feasible[mode]
            if o.formulation == "milp":
                return (p-target)**2  # projection weights must not replace an optimized binary mode
            return (o.projection_tracking_weight*(p-target)**2 +
                    o.projection_switch_weight*(mode != self.previous_mode) +
                    (o.persistence_weight*abs(p-self.previous_power) if o.cycling == "persistence" else 0))
        mode = min(feasible, key=cost)
        return mode, feasible[mode], override

    def controls_for_step(self, timestamp, previous_status, current_schedule, forecast):
        timestamp = pd.Timestamp(timestamp)
        status = previous_status or {}
        soc = float(status[f"{self.equipment_name} SOC (-)"])
        mobility = forecast.ev_schedules[self.ev.vehicle_id]
        available = bool(mobility.available([timestamp])[0])
        backed = float(forecast.backed_load_kw.at[timestamp])
        step = forecast.data.index[1]-forecast.data.index[0]
        self._observe(timestamp, float(status.get(f"{self.equipment_name} Electric Power (kW)", 0)), step)
        if self.next_solve is None or timestamp >= self.next_solve:
            started = time.perf_counter()
            solve_seconds = 0.0
            successful = False
            status_text = "error"
            try:
                problem, plan = self.build_problem(timestamp, soc, forecast)
                solving = time.perf_counter()
                try:
                    problem.solve(solver=self.options.solver, **self.options.solver_options)
                finally:
                    solve_seconds = time.perf_counter()-solving
                status_text = problem.status
                successful = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                if successful:
                    self.target = float(plan["charge"].value[0]-plan["discharge"].value[0])
                    self.held_mode = (1 if plan["charge"].value[0] > 1e-6 else
                                      -1 if plan["fraction"].value[0] > 0.5 else 0)
                    q = [float(v.value) for v in plan["shortfalls"]]
                    gap = [v-max(d-self.ev.capacity_kwh*plan["state"].value[j], 0)
                           for v, (j, d) in zip(q, plan["departures"])]
                    self.plans.append(dict(timestamp=timestamp, boundaries=plan["boundaries"].astype(str).tolist(),
                        soc=plan["state"].value.tolist(), charge_kw=plan["charge"].value.tolist(),
                        discharge_kw=plan["discharge"].value.tolist(), shortfall_kwh=q,
                        relaxation_gap_kwh=gap, objective=float(problem.value)))
            except cp.error.SolverError as error:
                status_text = str(error)
            if not successful:
                self.target = self.ev.charger_power_kw  # explicit, logged baseline fallback
                self.held_mode = 1
            self.solve_history.append(dict(timestamp=timestamp, status=status_text, successful=successful,
                                           solve_seconds=solve_seconds, ocp_seconds=time.perf_counter()-started))
            self.next_solve = timestamp+pd.Timedelta(minutes=self.options.control_minutes)
            # The OCP splits at mobility events; execute a new first action at the
            # same boundary instead of holding a pre-trip action after a return.
            for field in ("departure", "arrival"):
                future = mobility.trips.loc[mobility.trips[field] > timestamp, field]
                if not future.empty:
                    self.next_solve = min(self.next_solve, future.min())
        target = self.target
        if self.options.formulation == "milp":
            target = (self.ev.charger_power_kw if self.held_mode == 1 else
                      -min(backed, self.options.max_discharge_kw) if self.held_mode == -1 else 0)
        mode, power, override = self._project(timestamp, target, soc, available, backed)
        self.action_history.append(dict(timestamp=timestamp, requested_kw=target, projected_kw=power,
                                        mode=mode, projection_override=override))
        return {self.equipment_name: {"P Setpoint": power, "Backed Load (kW)": backed}}

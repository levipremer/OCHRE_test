# Scenario-based EV MPC

Implemented October 17, 2026. Use the existing unified runner and three cases;
no separate scenario simulation entry point. HPWH equations and behavior are
unchanged. HVAC remains pass-through. Existing perfect/deterministic configurations
remain selected until explicitly changed.

## Configuration

Set these fields in `defaults.ev`, `homes.HOME.ev_defaults`, or
`homes.HOME.evs.VEHICLE` of the control document (normal override precedence):

```json
{
  "controller": "mpc",
  "forecast": {
    "mode": "scenario",
    "scenario_method": "knn_state",
    "scenario_count": 5,
    "lookback_days": 30,
    "knn_history_hours": 24,
    "scenario_minutes": 15,
    "prefix_control_intervals": 4
  }
}
```

All existing per-EV formulation, solver/options, horizon/forecast length, weights,
degradation, terminal energy value, discharge budget, cycling/dwell and projection
preferences still apply. Convex still has no OCP cycle caps/dwell; neither EV
projector enforces them. Physical constraints and actual-load discharge caps remain.
Terminal value preserves the existing daily **minimum** tariff (user reconfirmed
October 17), not the mean still expected by the older terminal-value tests.
No terminal equation or numerical valuation was changed for this revision.

`scenario_minutes` divides a day and the control interval, and aligns with the
plant step. Solve origins must align with that grid. Prefix length is measured
in **control intervals**, not plant or mobility-event steps, and must fit inside
the MPC horizon. Forecast coverage is `max(horizon, forecast_hours)`.

## Historical selection

- Default kNN ranks historical origins by mean absolute home/away difference
  over the preceding 24 hours. Ties prefer newer origins. No random sampling.
- Alternatives: `tod_only`, `tod_dow`, `tod_state`, `tod_dow_state`,
  `tod_dow_state_hier`, `tod_dow_hier`, `tod_state_hier`, matching the prediction
  analysis project's filters and hierarchical relaxation. Filter methods keep
  distinct source days; kNN may use distinct non-overlapping windows in one day.
- The pool is a rolling **30 days**, not a calendar-month boundary. Every selected
  source trajectory must end by issuance. Bouts used for distance must have a
  historically completed return. There is no access to actual future trip energy.
- Rank first, then greedily retain temporally non-overlapping forecast windows.
  Adjacent windows may touch at their boundary. Their preceding kNN matching
  histories may overlap. This reduces redundancy; it does not establish statistical
  independence or guarantee the largest possible scenario set.
- Use fewer than requested if necessary, with uniform probability `1/S`.
  Zero candidates triggers logged baseline charging fallback until the next solve.
  Infeasible scenarios are NOT silently discarded.
- Unified loading increases effective mobility `history_days` as needed before
  replay-start selection: 31 days for default kNN (30-day pool + 24-hour matching
  history). The full history precedes **warmup start**, not evaluation start.
  Audi retains seeded interior selection without wrapping; NHTS retains annual
  replay. Insufficient dataset coverage raises an explicit error.
- Uses existing canonical trip distance and configured vehicle consumption.
  Raw Audi net-SOC depletion is not substituted for driving energy.

## Persistent short-term forecast

All scenarios hold measured home/away state for the first four control intervals
(one hour at 15 minutes), then resume the selected historical mobility path.
Arrival distance overwritten at or before the prefix boundary is accumulated
at the next arrival strictly **after** that boundary. Crossing trips are stitched
into coherent home-to-home bouts; an already-away EV is not assigned a new
departure at issuance. If no later arrival is in the forecast, displaced distance
is recorded as deferred beyond it; no artificial arrival or free charging gap is
inserted. Each solve constructs a fresh forecast, not a persistent ledger of
unrealized hypothetical trips. The plant always applies actual trip demand.

At prefix length zero, the measured state still anchors the first plant interval.
This is a persistence assumption, **not perfect short-term mobility**. It does not
trigger event-driven solves or alter actual availability.

## Formulation

For equally probable paths `s=1,...,S`:

`min max_s(J_comfort,s) + mean_s(J_electricity,s + J_health,s + J_degradation,s + J_persistence,s + J_terminal,s)`.

Each term already contains its existing weight. Comfort averages violations over
that scenario's departure steps (zero with no departures), then takes the maximum
**whole-path cost**, never the sum of per-step maxima. Other normalization is
unchanged. Disabled terms remain zero.

Only the first control interval's command is shared: MILP mode or convex chemical
power. Mobility splits do not create extra first-stage decisions. Subsequent
commands are scenario-specific recourse, as in HPWH. This is a two-stage model,
not a full multistage scenario tree: it optimistically assumes uncertainty resolves
after the first interval, even when the persistence prefix is longer.

Each path has its own SOC, full trip demand and optional constraints. All start
from measured SOC/SOH and common realized controller history. Faded capacity is
frozen within the solve. Only actual plant operation updates SOH/history.

Known tariff and non-V2H-twin backed-load trajectories are shared across paths.
The paired forecast interface is intentionally ready for the future aggregated
HEMS OCP to supply forecast/planned controlled-subsystem loads; that joint
optimization is **not implemented here**. No independent load or price sampling.

Linear throughput is directly solvable. Rainflow uses a separate epigraph and
supporting cuts for each path in one joint master, mean exact degradation cost,
and independent dollar-accuracy certification of **every** path. The overall
objective gap policy is unchanged. All master iterations share one 300-s budget;
canonicalization is not protected by a process-level watchdog. Unaccepted/infeasible
plans use the existing baseline fallback and fixed next-solve clock.

### Local Gurobi license limitation

The installed runtime currently uses a size-limited license. A five-path, 12-hour
MILP fixture had 2,165 scalar variables and exceeded that license (native error
10010); SCIPY solved the same fixture optimally. Small Gurobi scenario fixtures
passed for both formulations and all aging options. Use an unrestricted Gurobi
license for larger models, or explicitly select SCIPY for MILP / CLARABEL for
convex. The pipeline never silently changes the configured backend. This is a
license limit, not evidence that the scenario formulation is infeasible.

## Outputs and validation

Existing plots, metrics and phase masks remain. Per case/EV:

- `EV_ID_forecast_issuance.csv`: selected origins, kNN distances/filter tiers,
  counts, exclusions, uniform probability, prefix and deferred-distance metadata.
- `EV_ID_plans.json`: aggregate objective plus all scenario states/powers/costs.
  Legacy top-level trajectory fields represent the explicitly labeled first
  scenario; they are not ensemble averages or actual outcomes.
- `EV_ID_solve_history.csv`: scenario count, IDs, global solve diagnostics, and
  rainflow per-scenario approximation errors/tolerances when enabled.
- Existing comparison/report files keep baseline, twin and advanced separate;
  warmup/cooldown remain outside scoring. No research-duration study is launched.

Regression tests: `test/test_ev_scenario.py`. See [scenario validation](validation/ev_scenario_2026-10-17.md) for
the executed validation and any existing repository-wide limitations.

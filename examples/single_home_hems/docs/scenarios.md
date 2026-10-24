# Scenario-based EV MPC

Use the existing unified runner with baseline/advanced and an optional twin;
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

Backed-load paths use the realized non-V2H twin, or all-baseline when the control
document sets `run_non_v2h_twin: false`. The source is recorded in issuance diagnostics.

Only scenarios whose state at issuance differs from measured home/away state
hold measured state for the first four control intervals (one hour at 15 minutes).
Matching scenarios retain all original transitions and trip distances, including
departures/returns inside that interval. Comparison uses departure-inclusive,
arrival-exclusive away intervals. Candidate diagnostics record `prefix_applied`
and `prefix_original_state`; issuance diagnostics include `prefix_applied_count`.
For mismatched scenarios, apply the following energy rules:
When home, discard forecast trips returning at or before the prefix boundary.
For an overwritten departure whose return is later, place departure at the
boundary and retain its full distance and original return. Never compound
discarded distance into later trips. Diagnostics record ignored distance/count,
shifted departures, and distance shifted outside the exclusive forecast end.
When already away, extend that absence through the prefix with zero new debit;
absorb intervening departures until a genuine home gap, without duplicating the
ongoing trip's already-paid energy. Each solve creates a fresh forecast, not a
ledger of unrealized trips. Actual plant departures always consume actual demand.

At prefix length zero, a mismatch still anchors the first plant interval.
This is a persistence assumption, **not perfect short-term mobility**. It does not
trigger event-driven solves or alter actual availability.

## Formulation

For equally probable paths `s=1,...,S`:

`min max_s(J_comfort,s) + max_s(J_trip,s) + mean_s(J_electricity,s + J_health,s + J_degradation,s + J_persistence,s + J_terminal,s)`.

Each term already contains its existing weight. Comfort integrates SOC-floor
violations throughout each path (dt/N), including away periods, then takes the maximum
**whole-path cost**, never the sum of per-step maxima. Trip cost is
`trip_weight * sum(unmet_kWh) / (current_capacity_kWh * new_trip_count)` per path,
zero without new trips. Comfort and trip maxima can come from different paths.
Other normalization is unchanged. Disabled terms remain zero.

Only the first control interval's command is shared: MILP mode or convex chemical
power. Mobility splits do not create extra first-stage decisions. Subsequent
commands are scenario-specific recourse, as in HPWH. This is a two-stage model,
not a full multistage scenario tree: it optimistically assumes uncertainty resolves
after the first interval, even when the persistence prefix is longer.

Each path has its own SOC, distance-derived trip demand, bounded penalized unmet
trip-energy slack, and optional constraints. The slack prevents one physically
unreachable path from invalidating every scenario; it does not alter actual plant
trip demand. All paths start from measured SOC/SOH and common realized controller
history. Faded capacity is frozen within the solve. Only actual plant operation
updates SOH/history.

Known tariff and the selected baseline/twin backed-load trajectory are shared across paths.
The paired forecast interface is intentionally ready for the future aggregated
HEMS OCP to supply forecast/planned controlled-subsystem loads; that joint
optimization is **not implemented here**. No independent load or price sampling.

Linear throughput is directly solvable. Rainflow uses a separate epigraph and
supporting cuts for each path in one joint master, mean exact degradation cost,
and independent dollar-accuracy certification of **every** path. The overall
objective gap policy is unchanged. All master iterations share one 300-s budget;
canonicalization is not protected by a process-level watchdog. Unaccepted/infeasible
plans use the existing baseline fallback and fixed next-solve clock.

Gurobi requires a valid license for the machine and model size. Set
`GRB_LICENSE_FILE` to your valid license if discovery selects an obsolete file.
The pipeline never silently changes the configured backend.
EV Gurobi solves default to `DualReductions=0` for unambiguous classification
within a single backend call.

## Outputs and validation

Existing plots, metrics and phase masks remain. Per case/EV:

- `EV_ID_forecast_issuance.csv`: selected origins, kNN distances/filter tiers,
  counts, exclusions, uniform probability, conditional-prefix application and
  ignored/shifted-distance metadata;
  `no_scenarios_selected` explicitly flags empty selection/fallback.
- `EV_ID_forecast_errors.csv`: per-issuance scores and eligibility/count diagnostics.
  `EV_ID_forecast_errors_by_lead.csv`: detailed control-bin targets and errors.
  Accuracy summaries and lead plots compare cases using the **final forecast,
  including its persistence prefix**, never separate prefix/tail scores.
- `EV_ID_plans.json`: aggregate objective plus all scenario states/powers/costs.
  Legacy top-level trajectory fields represent the explicitly labeled first
  scenario; they are not ensemble averages or actual outcomes.
- `EV_ID_solve_history.csv`: scenario count, IDs, global solve diagnostics, and
  rainflow per-scenario approximation errors/tolerances when enabled. Mean/max
  predicted trip shortfall and the number of affected scenarios are included.
- Existing comparison/report files keep baseline, twin and advanced separate;
  warmup/cooldown remain outside scoring. No research-duration study is launched.

Scoring runs after simulation against actual distance-derived driving kWh; it
cannot influence scenario selection. Home Brier uses ensemble home probability;
false-home/false-away rates average mismatches over all scenario-timestep pairs.
Departure-energy MAE averages absolute scenario errors, not error of ensemble mean.
State is sampled at each control-bin start; energy sums departures within that bin.
New `departure_*` scores include horizon energy MAE and envelope coverage/width.
Existing arrival-named/legacy horizon scores remain arrival-based for historical
comparisons; plots retain both. Next-trip scoring includes a new departure even
if its return is beyond the horizon.
Timestep summaries are duration-weighted and require issuance/targets inside
evaluation. Horizon energy MAE, min/max-envelope coverage/width, and next-new-trip
energy MAE require the entire forecast inside evaluation. These average across
eligible issuances; counts are retained. The ongoing trip at issuance is excluded
from next-new-trip scoring. No observed next trip gives unavailable MAE; missing
predicted trips count as zero energy when truth exists, with missing/false-prediction
fractions reported. Empty ensembles have a flag/count, never zero-error scores.
Baseline/non-scenario controllers have unavailable ensemble metrics. Warmup and
cooldown are excluded. Run/study summaries keep twin and advanced separate.

Regression tests: `test/test_ev_scenario.py`, `test/test_ev_reporting.py`. See [validation](validation/README.md) for
the executed validation and any existing repository-wide limitations.

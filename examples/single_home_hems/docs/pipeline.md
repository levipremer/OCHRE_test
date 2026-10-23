# Unified pipeline

[Guide index](../README.md) · [Extension map](extending.md)

## Execution and comparison

`run_simulation.py` loads standalone home/control JSON, validates settings, freezes
mobility, and runs the selected cases with fresh controller and plant state:

| Case | Behavior |
| --- | --- |
| all_baseline | Native HPWH/HVAC; EV charging to comfort minimum; no MPC or V2H |
| non_v2h_twin | Configured controllers, but V2H disabled |
| advanced | Configured controllers, including V2H if selected |

Supplied control/study documents set top-level `run_non_v2h_twin` to `false`;
omitting the key preserves the legacy `true` default.
Set it to `false` to run only all-baseline and advanced; twin outputs are omitted.
The twin supplies advanced backed-load forecasts when enabled; otherwise the
all-baseline realized selected end uses supply them. The selected reference
extends beyond cooldown to cover final horizons; this tail is not scored.
The all-baseline case always remains the savings reference. All cases share weather,
mobility, physical configuration and configured initial states.

Studies may reuse an integrity-checked all-baseline trajectory with identical
plant inputs and comfort target; metrics/plots are regenerated per job. The
optional `baseline_cache` runner argument is study-scoped and never reuses the
non-V2H twin or advanced state. Baseline-cache keys distinguish reference-tail durations.

At most one V2H EV per home receives the full backed-load reference. The usual
selection is `["heat_pump", "water_heater"]`; home configuration must state it.
Heat pump means compressor/main plus fan, excluding resistance backup.
Water heater includes resistance electricity if physically enabled. Other supported
native electrical end uses may be selected; EVs/storage/generation cannot back
themselves. Whole-home power includes all end uses and signed EV power.

The plant updates non-EV loads first and caps discharge by actual same-step
selected load, inverter rating, availability and battery energy. Forecasts do not
override these safeguards. Missing backed-load components raise an error.

## Configuration

Copy the [home](../config/unified_home.template.json) and
[control](../config/unified_control.template.json) templates. Their root
`_comments` objects document units, feasible values and inactive fields.
Example/template tuning is independent; neither is an optimized recommendation.

- **Home:** simulation windows, native input paths, HVAC/HPWH physical settings,
  backed loads, EV nominal capacity, initial SOC/SOH, charger/inverter limits,
  efficiencies, mobility dataset/selection and battery calibration/economics.
- **Control:** controller, formulation, solver/options, horizon, forecast length,
  control interval, forecast mode, objective weights, cycling and projection.
- Control precedence: `defaults.DEVICE` → home device settings
  (`ev_defaults` for EVs) → `homes.HOME.evs.ID`.
- Missing required physics is an error. Unified loading never consults legacy
  home/neighborhood JSON; native OCHRE and shared `home_factory` remain dependencies.
- Paths inside JSON resolve against the repository root; CLI config paths resolve
  against the working directory. Explicit absolute paths are supported.
- Configure complete home entries before selecting multiple homes with `--homes`.
  There is no joint multi-home or HVAC optimization.

EV options remain independent per vehicle. Formulations are convex or MILP;
backends are CLARABEL, SCIPY and GUROBI. CLARABEL cannot solve MILP.
Baseline is the default unless overridden, and does not require an OCP solver.

HPWH selects `rc`, `rc-old`, `rc-stratified`, or `state-space` predictors,
with optional `model_path` and independent `parameters`. Saved predictors require
15-minute model/control steps; scenario mode requires a 12-hour horizon,
quarter-hour alignment and an explicit water-draw replay file.
Existing scenario costs are mean electricity/legionella/persistence and maximum
whole-path comfort. HPWH equations are unchanged by this cleanup.

## Time and mobility

`start_time` and `duration_hours` define the **evaluation** window.
`warmup_hours` and `cooldown_hours` default to 24; zero disables either.
Both are controller-active, unscored, and aligned with plant/control intervals.
Initial states apply at warmup start; controller memory and battery cycle residue
are never reset at evaluation boundaries. Native `initialization_hours` is zero.
Legacy compatibility calls retain zero padding.

OCPs run at start and fixed control intervals, never on mobility events.
Plant availability, trip energy, projection and safety update every plant step.
Prediction blocks may split at events without introducing extra control decisions.

Canonical trips contain departure, arrival and distance_km. Vehicle consumption
converts trip distance to energy debited at departure; SOC uses current faded capacity.
NHTS annual replay may wrap. Audi uses seeded interior replay with sufficient
history/future and no wrapping; Audi SOC/energy fields infer distance using the
Audi reference vehicle, not the simulated vehicle's driving demand.

Forecast coverage includes warmup/cooldown, history and final prediction tails.
Missing coverage is an error, not invented data. Cooldown does not force recharge,
equalize ending SOC or remove finite-horizon incentives.

## Forecasts

- **Perfect:** realized exogenous schedules and the selected baseline/twin reference.
- **Deterministic:** explicit external JSON snapshots, latest `issued_at <=`
  solve time, complete finite plant-grid coverage and declared units.
- **EV scenario:** implemented historical paths; see [scenarios](scenarios.md).
- **HPWH scenario:** existing water-use scenario provider; methods
  `daily_budget_high_rf_1` and `daily_budget_high`.

External JSON has `units` and `snapshots`. Each snapshot has `issued_at` and
`data` rows with `timestamp`. EV rows require `electricity_price_usd_per_kwh`
(USD/kWh), `backed_load_kw` (kW), plus `trips` containing departure/arrival/
distance_km and declared distance units km. HPWH rows require price, water-heater/
clothes-washer/dishwasher draw (L/min), and mains temperature (C).
No future realized columns fill missing deterministic predictions. The known
tariff is retained separately for terminal valuation.

## EV constraints and costs

Both formulations keep hard SOC [0,1] bounds before and after departure. At forecast departures, a bounded
unmet-driving-energy slack is limited by that trip's distance-derived demand and
penalized as `trip_weight * sum(unmet_kWh) / (current_capacity_kWh * trip_count)`.
Count new departures strictly after the measured initial state and before horizon
end, including zero-distance trips; zero trips gives zero cost. There is no dt/N
factor. Scenario MPC takes the maximum path trip cost separately from maximum
comfort cost. Predicted slack is recorded in plans/solve diagnostics.
The physical plant still attempts the full realized trip, saturates finite SOC,
and independently records actual clipping/unserved energy at departure. Arrivals
only restore availability. Initial SOC is already net of any ongoing trip.
The plant blocks both charge and discharge while away and logs rejected commands.
Pre-departure SOC is retained for departure-comfort metrics and rainflow peaks.
The default `trip_weight` is 10,000 (also explicit in the supplied templates).
This remains a soft preference, not lexicographic priority. Large unavoidable
shortfall costs can inflate the absolute error permitted by a relative MIP gap.
Trip-count averaging reduces each individual trip's marginal penalty in busy
horizons; worst-case scenario aggregation does not make this a hard guarantee.
Use a new study/output directory for departure accounting: old checkpoints and
arrival-accounted cached baselines are not compatible. Do not pool old/new cohorts.

Convex uses one signed battery-side power with efficiency-aware piecewise AC power;
MILP uses exclusive charge/discharge/idle modes. Lossy bidirectional convex
intervals reject negative prices rather than silently clip prices or switch solver.

Comfort currently penalizes `max(comfort_soc_min - x_j, 0)` throughout the horizon,
including away periods, using dt/N. It is a soft reserve, not departure-only.
Departure-comfort metrics still use the measured pre-debit SOC. Health limits
penalize SOC-band violations throughout. Electricity, comfort, health and
persistence use dt/N; cumulative degradation uses /N.
MILP persistence penalizes changes in mode indicators; convex uses chemical-power
variation. Details and calibration are in [degradation](degradation.md).

EV MILP supports independent cycle caps and dwell constraints in the OCP.
Convex has neither; **neither EV projector enforces them**.
Caps use historical ends plus planned starts, with future=min(horizon,12h) and
history=24h-future. The optional V2H discharge budget uses that same window,
excluding driving. Projection enforces physical limits, not a realized-budget
guarantee. HPWH retains its separate cap/dwell OCP/projector behavior.

Optional `enforce_terminal_soc` enables `v*(e0-eN)/N`, where energy uses measured
faded capacity frozen within the horizon. **v is the minimum known tariff over
the solve calendar day**, sampled at one-minute intervals and assigned directly
in USD/battery-side-kWh without efficiency adjustment. It is independent of
availability/horizon. Default tariff gives v=$0.15/kWh. This heuristic prices
depletion and credits surplus; `terminal_soc_weight` is retired.
Old mean-price tests remain a documented discrepancy, not silently rewritten.

## Solver and fallback policy

Quiet native backend limits are 300 seconds. Rainflow master solves share one
elapsed OCP budget and receive its remaining time. Construction/canonicalization
and backend overhead mean this is **not a process-level watchdog**.
Finite feasibility and integrality checks remain; EV MILP additionally checks
the configured gap (default 5%). Rainflow requires whole-objective certification
and a separate dollar envelope-accuracy check on every scenario.
See [degradation](degradation.md).

EV Gurobi solves default to `DualReductions=0` so diagnostics distinguish
infeasible from unbounded models without a second backend solve. Explicit
per-EV solver options can override this; HPWH solver settings are unchanged.
Unaccepted EV plans use immediate charging toward comfort minimum without V2H
until the next scheduled solve. HPWH retains its prior target through existing
actuator logic. Failures, limits, physical overrides and projection are recorded.

## Outputs

Each output root contains:

- `resolved_configuration.json`: input/resolved settings and windows.
- `reports/run_report.md`: consolidated performance **and solver** report.
- `data/comparison.csv`, `performance.csv`, `solver_stats.csv`,
  `solve_history.csv`, and case-wise aggregate power.
- Per-home, per-case data/figures, controller plans and issuance/solve/action histories.
- EV forecast-error CSVs, final-ensemble accuracy/lead plots, and EV/whole-home
  savings comparison plots (see [scenario scores](scenarios.md#outputs-and-validation)).
- `raw_results_all_phases.csv`, `controls_all_phases.csv`, and
  `evaluation_mask.csv` retaining warmup/evaluation/cooldown/forecast-tail context.

Comparison metrics/plots score only evaluation. Native raw diagnostics may include
all phases. Aging scores continuous-ledger increments, carrying earlier cycle
context without billing warmup/cooldown increments. Checkpoints save battery state,
not a complete dwelling/controller restart.

Net savings = baseline operating cost - case operating cost + baseline reference
aging cost - case reference aging cost. Twin and advanced remain separate.
`net_savings_usd` retains that definition. Separately,
`energy_adjusted_net_savings_usd = net_savings_usd + energy_inventory_adjustment_usd`,
where the correction is `vT*(ET_case-ET_baseline) - v0*(E0_case-E0_baseline)`.
Energies are actual stored chemical kWh at evaluation boundaries, using faded
capacity. Common `v0`/`vT` are daily-minimum tariff values at the first/last scored
timestep, independent of whether the terminal objective is enabled. Different
boundary values include inventory revaluation. Whole-home sums EV inventories;
thermal storage is not valued. Missing archived boundary data gives unavailable
adjusted savings, not a fabricated zero. Compare trip shortfalls alongside savings.
Cycle starts and equivalent full cycles are distinct.

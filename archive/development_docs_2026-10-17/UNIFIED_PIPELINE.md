# Historical unified pipeline guide (superseded)

Archived October 17, 2026 before consolidation. Relative paths refer to the
original `examples/single_home_hems/` location. Later headings can contradict
earlier revisions; use the current pipeline guide for supported behavior.

October 17: scenario EV MPC is available; see [EV_SCENARIO.md](EV_SCENARIO.md).
User reconfirmed retaining the current **daily-minimum** terminal energy value.
This supersedes earlier mean-price wording below; older mean-price tests are
left unchanged and their failures are recorded in EV_SCENARIO_VALIDATION.md.

Updated 2026-10-15; paired 30-job study, unscored cooldown and independent rainflow accuracy.
This guide supersedes the two-case and
TOU-controller instructions in the earlier EV/README documents. Existing source
home files are not rewritten. No research comparison studies were launched for
this revision; validation uses unit tests and short integration fixtures.

## Run

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --home-config examples/single_home_hems/config/unified_home.json --control-config examples/single_home_hems/config/unified_control.json --home-id home_01 --output-directory examples/single_home_hems/outputs/my_unified_run
```

This is a command for a future user-selected run, not a validation command.
Use a fresh output directory: the canonical runner rejects nonempty directories.
The example explicitly configures home_01. Add complete home entries before using
`--homes home_01 home_02` for multiple independent homes.
There is no coordinated neighborhood optimization or HVAC MPC.

Every pipeline invocation retains:

| Case | Controls |
| --- | --- |
| `all_baseline` | Native HVAC/HPWH; EVs charge to comfort minimum; no MPC, no V2H |
| `non_v2h_twin` | Exactly the configured controllers and preferences, with V2H disabled; EV MPC remains charge-only MPC |
| `advanced` | Fully configured controls, including V2H if selected |

Weather, initial physical states, realized mobility, and water-use inputs are
shared. The twin runs past the comparison end to supply complete future demand
trajectories. Comparison CSVs/plots and solver summary denominators use the requested
duration; native raw twin output and detailed solver history retain the extension.
The all-baseline case is not a forecast provider for advanced control.

### Controller-active warm-up and cooldown

Set `simulation.warmup_hours` in the home JSON; default **24**, or **0** to disable.
Set `simulation.cooldown_hours` similarly; default **24**, or **0** to disable.
Cooldown continues normal control after evaluation, without forcing recharge or
equal terminal SOC. All evaluation-end horizons remain fully forecasted; the twin
forecast tail comes after cooldown. Cooldown alone does not remove finite-horizon
MPC effects or credit final stored energy in savings.
`start_time` and `duration_hours` describe evaluation, not warm-up. Controllers and
plant start that many hours earlier, with the configured initial SOC/temperature.
All three cases retain their own evolved state, fixed solve clock, cycling/dwell
memory and degradation history without a boundary reset. Warm-up does not guarantee
steady state; lengthen it if the modeled system needs more settling time.

Performance, degradation, cycle and solver metrics use only the evaluation window.
EV SOH evolves throughout all phases from explicit `physical.initial_soh` (.85
in the supplied documents). Damage is accumulated with reference SOC-aware
rainflow; scored aging sums ledger increments without resetting cycle residue.
SOC/energy dynamics use faded capacity. MPC reads fresh SOH each solve and freezes
it within that horizon; nominal pack capacity remains the economic valuation basis.
See EV_DEGRADATION.md for damage units, energy conservation and checkpoint scope.
Ongoing warm-up cycles are not new evaluation starts. Aging metrics use the
evaluation SOC path including its boundary state and residue half cycles; they
do not attribute earlier warm-up cycling to evaluation. EV `initial_soc` reports
that boundary state; `configured_initial_soc` records the warm-up-start setting.
Comparison plots/CSVs exclude warm-up AND cooldown, while native raw results,
`raw_results_all_phases.csv`, `controls_all_phases.csv` and solve/action histories
retain them. `evaluation_mask.csv` labels warmup/evaluation/cooldown/forecast_tail;
`resolved_configuration.json`
identify the windows. The twin also retains its forecast tail.
Native diagnostic metrics under `data/raw/` cover that full trajectory; the unified
`reports/run_report.md` and comparison/performance CSVs are the scored results.

Both padding periods must be finite, nonnegative and aligned with the plant and active control
intervals. Audi selection includes the longer window and never wraps. External
forecasts and water replay must cover warm-up, cooldown and required history/future; missing
data is not fabricated. `water_draw_start` remains anchored to evaluation start.
Native `initialization_hours` remains zero; it is not controller-active warm-up.
The legacy compatibility entry point keeps its previous zero-padding behavior.

## Configuration separation

The home document is self-contained. It never reads legacy `homes.json` or
`neighborhood.json`; `source_config_directory` is rejected. Native OCHRE,
`home_factory`, shared configuration/equipment/schedule utilities, and referenced
HPXML/weather/dataset/model files remain normal dependencies.
All paths inside the documents resolve from the repository root (absolute paths
also work). The two `--*-config` paths resolve from the current working directory.

Home document fields:

- `simulation`: evaluation start/duration/timestep, warm-up, initialization, weather/NHTS paths,
  output directory, random seed, output/validation settings, defrost and transformer
  parameters. All fields are explicit; initialization must be zero.
- `homes.HOME`: explicit HPXML/schedule paths, description/seed offset, HVAC
  setpoints/deadbands/compressor timings/backup staging, water-heater settings.
- `homes.HOME.backed_loads`: explicit list; template uses `["heat_pump", "water_heater"]`.
- `homes.HOME.water_heater_hp_only`: explicit physical operating restriction,
  applied to all three cases. The example enables it for the supplied RC model.
- `homes.HOME.initial_temperature_c`, `water_draw_file`, `water_draw_start`:
  initialization and optional DC House physical replay.
- `homes.HOME.evs.ID`: required `capacity_kwh`, `charger_power_kw` (AC charging
  limit), `initial_soc_fraction`, vehicle type and NHTS sampling fields. Optional
  sampling values use explicit null; nothing inherits from legacy files.
- `evs.ID.physical`: required `consumption_kwh_per_km`, `charge_efficiency`,
  `discharge_efficiency`, `max_discharge_kw`, `v2h`, and the explicit battery-aging
  calibration fields described in [EV_DEGRADATION.md](EV_DEGRADATION.md).
  Health/comfort SOC preferences belong in the control document, not physics.
- `evs.ID.mobility`: `source` (`nhts`/`audi`), `source_file`, `source_vehicle`,
  `seed`, `history_days`, all explicit. The NHTS sampling parameters are stored
  alongside EV capacity/charger settings, even when unused for Audi.

HPWH/HVAC envelope and equipment details not overridden in this document remain
in the explicitly named native HPXML, not in a legacy simulator configuration.
Missing required fields produce a path-specific error. Optional native values
must be explicit null. No solver/weights belong in EV physics.

Two directly loadable, documented templates are provided:

- [unified_home.template.json](config/unified_home.template.json)
- [unified_control.template.json](config/unified_control.template.json)

Standard JSON has no comment syntax, so each template has an ignored `_comments`
object describing variables, units, feasible values, and inactive fields. The
plain example JSONs remain available. Copy templates and edit values; no comment
stripping or separate parser is needed. Example and template tuning are independent;
read the selected file rather than assuming they match. These are not optimized values.

Control resolution: `defaults.DEVICE` -> per-home device settings -> per-EV settings.
Devices are `hpwh`, `hvac`, and `ev`; each home supports `hpwh`, `hvac`,
`ev_defaults`, and `evs.ID`. Defaults remain baseline. Common preferences:

| Field | Meaning / default |
| --- | --- |
| `controller` | `baseline` or `mpc`; baseline default |
| `formulation`, `solver` | `convex`, `CLARABEL`; alternatives MILP/SCIPY/GUROBI |
| `solver_options` | Backend-specific CVXPY options, default `{}` |
| `horizon_hours` | Optimization horizon, default 12 |
| `forecast_hours` | Required supplied forecast coverage; defaults to horizon |
| `control_minutes` | Fixed solve interval, default 15 |
| `forecast.mode` | `perfect`, `deterministic`, or `scenario`; perfect default |

Each EV independently accepts `enforce_cycle_limit`, `enforce_dwell_constraints`,
`persistence_weight`, `dwell_on_minutes`,
`dwell_off_minutes`, `max_charge_starts_24h`, `max_discharge_starts_24h`,
`electricity_weight`, `health_soc_min`, `health_soc_max`, `health_soc_weight`,
`health_weight`, `comfort_soc_min`, `comfort_weight`, `enable_degradation`,
`degradation_method`, `rainflow_relative_tolerance`, `rainflow_absolute_tolerance`,
`rainflow_max_iterations`,
`enforce_terminal_soc`, `enforce_discharge_budget`,
`discharge_budget_soc`, `mip_relative_gap`,
`projection`, `projection_tracking_weight`, `projection_switch_weight`, and
`projection_threshold`. Units/defaults are in the commented template. The old
mutually exclusive `cycling` enum is a compatibility input only; explicit switches
win. Positive persistence is active independently of dwell/caps.

HPWH controls additionally select `model` (`rc`, `rc-old`, `rc-stratified`, or
`state-space`), optional `model_path`, `model_minutes`, and `parameters` containing
existing controller preferences such as independent `electricity_weight` (default 1),
`comfort_weight`, `legionella_weight`,
`persistence_weight`, `control_variation_weight`, `enforce_cycle_limit`,
`max_cycles`, `enforce_dwell_constraints`, and `dwell_time_on`/`dwell_time_off`
(duration strings such as `"60min"`). `control_variation_weight` and
`minimum_lower_temperature_c` are inactive compatibility fields (keep 0/null);
use `persistence_weight` for the active variation term. Model/solver/horizon fields must not be
duplicated inside `parameters`.

Saved HPWH predictors currently require 15-minute model/control steps; scenario
mode additionally requires a 12-hour horizon and a quarter-hour-aligned start.
Unsupported settings raise errors, not silent resampling. HPWH scenario aggregation
is unchanged: mean electricity/legionella/persistence; maximum per-scenario total
comfort; shared first action. Scenario method is `forecast.scenario_method`,
`daily_budget_high_rf_1` (default) or `daily_budget_high`.

HVAC remains native/pass-through; requesting `controller: mpc` raises an explicit
error. GUROBI is optional and requires its package/license; short convex and MILP
EV solves passed locally on October 12. CLARABEL cannot solve MILP.

## Cycling and solver policy

Cycle caps, dwell, and positive persistence penalties can coexist in any
combination. EV MILP applies enabled caps/dwell inside the OCP; convex does not add
discrete constraints. Neither EV projector enforces caps/dwell. HPWH still applies
its enabled caps/dwell through its projector and retains its original scaling and
scenario aggregation.

EV caps count direction-specific **observed ends plus planned starts**. Future
window F=min(horizon,12h); history=24h-F. Continuing an active cycle consumes no new
start. No pre-simulation operating history is invented. Legacy `max_*_starts_24h`
key names are retained. Optional discharge budget uses the same window and counts
battery-side V2H throughput/capacity, excluding driving; 1 means one capacity's
worth of discharge, not max-minus-min SOC. HPWH keeps
its existing different budget: predicted horizon starts plus min(1, stops in the
preceding hour) <= max_cycles. Its projector checks the corresponding first-step
budget; it is not a rolling daily/hourly start cap. Dwell uses realized mode history.
HPWH projector mode commands use native binary duty control when cap/dwell is
enabled; this prevents thermostat starts bypassing a low-setpoint request. With
both flags disabled, the existing high/low setpoint mapping remains. Native
maximum-temperature safety and EV availability/SOC/load constraints take priority;
overrides are recorded. Projection does not reoptimize the horizon.

All EV/HPWH OCPs use a quiet 300-second native backend limit: SCIPY
`scipy_options.time_limit`, CLARABEL `time_limit`, GUROBI `TimeLimit`. User backend
options cannot increase that limit or enable solver verbosity. Construction and
CVXPY canonicalization are outside the backend cap; this is not a hard process
wall-clock watchdog. Timing separates OCP/build+solve, solve-call and available
backend time. Limit exits, failures and fallbacks remain visible in histories.
Finite feasible incumbents, including integer checks, may be accepted without
labeling them optimal. EV MILP also requires backend relative MIP gap <=
`mip_relative_gap` (default 0.05); missing/nonfinite gap on a nonconverged solve is
rejected. Otherwise EV uses baseline comfort-min charging until the next solve;
HPWH retains its prior target through its unchanged projector. Backend gaps/bounds
exclude canonicalization objective offsets, so `model_objective` may differ from
backend primal/dual values. Native backend gap normalization is used.
Rainflow additionally requires an exact-cost/master-bound certificate across
all internal solves, under one shared 300-second elapsed OCP budget. Its bound
includes objective offsets. See [EV_DEGRADATION.md](EV_DEGRADATION.md) for tolerances
and why native master optimality alone is insufficient.

## Fixed clock and physical safeguards

EV OCP solves occur at start and every configured control interval, never on a
departure/return. Mobility still updates plant availability and trip energy each
step. Prediction integration can split at mobility boundaries, but available
pieces of the same control interval share a command. Safety, discrete projection,
and availability filters remain active between solves. Failed solves are logged;
fallback does not schedule an extra solve.

HPWH also starts from measured initial tank states on the fixed clock; identified
models use the existing repeated-measured-state history initializer. No future
realized measurements initialize an identified state. Failed attempts do not cause
off-clock retries. Native controllers run on the ordinary plant timestep.

One EV at most may have V2H capability in a home. `heat_pump` means reported space-
conditioning compressor/main plus fan electricity, excluding the separately
reported resistance backup. `water_heater` means total HPWH electricity, including
its resistance element if enabled. Other named OCHRE electrical end uses may be
selected; EVs, generation/storage, and whole-home totals cannot back themselves.
Unknown/missing component data cause an error, not substitution of total HVAC load.

The realized **non-V2H twin** supplies the reference demand forecast. At execution,
non-EV equipment is updated first, then EV discharge is clipped against actual
same-step selected demand, inverter rating, availability, and battery energy.
EVs add no indoor heat gains, so this ordering preserves native thermal dependencies.
`Actual Backed Load (kW)` and `Safety Override` are recorded per EV. Whole-home
power remains all end uses, including signed EV power and miscellaneous loads.

Trip-distance inference is unchanged. The OCP subtracts full trip demand on return,
with hard SOC [0,1] and **no trip-shortfall variable or penalty**. An unserviceable
forecast trip can make the OCP infeasible; baseline fallback is intentional.
Trips returning beyond the horizon have no arrival impulse inside it.
At execution, finite SOC saturates at 0 or 1; clipping and unserved trip energy
are recorded. Invalid configuration/nonfinite state still raises.
`SOC Clipped Energy (kWh)` includes rejected requested energy and arrival deficits;
it is diagnostic, not additional consumption. Arrival `Trip Unserved Energy`
and departure deficit are separate diagnostics.

## EV mathematical formulation

`controllers/ev.py` contains horizon data, power, dynamics, optional budgets,
objective terms and controllers; `ev_formulation.py` is only an import shim.
Let x be SOC, p signed battery-side kW (positive
charging), z fixed home/away data, E capacity, and D arrival-trip energy.

```
x[j+1] = x[j] + (dt[j]*p[j] - D[j])/E;  0 <= x <= 1
Convex: P[j] = max(p[j]/eta_c, eta_d*p[j])
         -z[j]*min(backed_load,inverter)/eta_d <= p[j] <= z[j]*eta_c*charger
MILP:    p = eta_c*u_c - u_d/eta_d; exclusive b_c,b_d; b_c+b_d <= z
```

Convex has one power decision: simultaneous charge/discharge is impossible.
Held chemical power respects the minimum backed load within each integration
block; MILP discharge follows backed load up to inverter rating. No relaxed
home/away indicator exists.

EV electricity, health SOC and persistence terms use `dt[j]/N`; N counts control
intervals, not event blocks. Electricity uses AC net demand. The lower/upper
health band shares `health_soc_weight`; comfort shortfall has `comfort_weight`
and is averaged only over expected departure steps in [solve time, horizon end).
No departure means zero comfort cost; it is not penalized after a trip.
`health_weight` multiplies both weighted health-band violations and optional
degradation dollars/N. Cumulative degradation is NOT multiplied by dt again.
Convex persistence uses `abs(delta p)`;
MILP uses `abs(delta b_charge)+abs(delta b_discharge)+abs(delta b_idle)`;
each mode switch contributes two indicator changes, regardless of power rating.
Optional terminal energy cost (`enforce_terminal_soc=true`) is
`v/N * (e0-eN)`, with `e0=C_k*x[0]`, `eN=C_k*x[-1]` and measured faded
capacity `C_k` frozen within the horizon. It prices depletion and credits surplus;
there is no positive-part SOC-floor penalty or extra multiplier.
`v` is the full solve-calendar-day time-average known tariff, assigned directly
as dollars per battery-side kWh (no efficiency conversion). The daily tariff is
sampled at one-minute intervals, independently of horizon length and availability;
no extended load/mobility forecast or future realized data are used. With the
default 21 off-peak hours at $0.15 and 3 peak hours at $0.30, `v=0.16875`.
External device forecasts retain this known tariff separately from their finite
prediction rows. Custom callers must supply it through `Forecast.from_schedule`;
missing/nonfinite daily tariffs fail explicitly when the term is enabled.
The old `terminal_soc_weight` is retired: accepted for compatibility, ignored,
and warned about if nonzero while active. Plans record `terminal_value_usd_per_kwh`,
`initial_energy_kwh`, `terminal_energy_kwh`, and the normalized terminal cost.
This valuation is a heuristic, not a guaranteed conservative opportunity value.
It changes controller decisions only, not reported operating/degradation savings.
No trip-shortfall penalty or slack is used. Baseline/fallback charges immediately
to `comfort_soc_min`. Optional `rainflow` or `linear_throughput` degradation uses
the shared calibration and safeguards in [EV_DEGRADATION.md](EV_DEGRADATION.md).
The physical calibration remains 3000 reference cycles to 80% SOH; the economic
70% SOH endpoint gives an extrapolated 4500-cycle budget. New/residual values are
$119/$0 per kWh. Both endpoints are explicit, independent home-config fields.

Lossy bidirectional convex intervals reject negative-price forecasts because the
piecewise electricity cost would be nonconvex. Explicitly select MILP for them.
Charge-only and lossless convex intervals accept negative prices. No price clipping
or automatic solver switching occurs.

Projection selects charge/idle/discharge and enforces physical limits, **not**
dwell/caps. Convex discrete execution can differ from planned discharge budget,
terminal SOC and targets; OCP budget feasibility is not a realized-budget guarantee.
MILP remains subject to actual-load safety and forecast errors. Compare realized
metrics with plan JSON, not only objective values.

## Deterministic and placeholder forecasts

Perfect mode uses realized exogenous schedules and the no-V2H twin reference load.
Deterministic mode requires an explicit `forecast.file`; there is no implicit
perfect-data fallback or invented forecast method.

The external file is JSON with `units` and `snapshots`. Each snapshot contains
`issued_at` and ordered `data` rows with `timestamp`. Select the latest issuance
no later than the decision. Every native plant step through `forecast_hours` must
be covered. Local timestamps use the same timezone convention as the simulation.

- EV data columns: `electricity_price_usd_per_kwh` (`USD/kWh`) and `backed_load_kw`
  (`kW`). Each snapshot also has `trips`: complete `departure`, `arrival`,
  `distance_km` (`km`) records, including returns beyond the prediction horizon.
- HPWH data columns: `electricity_price_usd_per_kwh` (`USD/kWh`),
  `Water Heating (L/min)`, `Clothes Washer (L/min)`, `Dishwasher (L/min)` (all
  `L/min`), and `Mains Temperature (C)` (`C`). Current measured tank ambient is
  retained as in the existing predictor.
- `units` maps each column name to the unit string above. Missing units/coverage,
  duplicate targets, or unavailable causal issuances raise errors. Future realized
  columns are not copied into the external prediction. Actual tariff and demand
  remain separate from forecast errors.

The twin's forecast tail also needs valid external issuances/coverage; mobility
adapters reserve up to two forecast horizons beyond the requested duration for
its final solves. The realized source mapping is frozen across all three cases.

EV scenario MPC is implemented: default five non-overlapping kNN paths from a
causal 30-day pool, with four control intervals of home/away persistence. Worst
per-path departure comfort and mean other costs; shared first command, later
recourse. All EV options remain available. See [EV_SCENARIO.md](EV_SCENARIO.md)
for configuration, selection safeguards and trip-energy boundary handling.
HPWH scenario MPC is unchanged.

## Output locations

```text
OUTPUT/
  resolved_configuration.json
  data/solver_stats.csv
  data/solve_history.csv
  data/comparison.csv
  data/performance.csv
  data/{CASE}_aggregate.csv
  reports/run_report.md          # single consolidated human-readable report
  HOME_ID/
    data/                       # home-wide comparisons, timing, trips/replay metadata
    figures/                    # three-case EV, HPWH, HVAC, whole-home overlays
    all_baseline/
    non_v2h_twin/
    advanced/
      data/                     # same layout in every case
        raw/                    # OCHRE files, including twin forecast tail
        raw_results.csv         # comparison-window raw results
        standardized.csv
        controls.csv
        water_heater_sensors.csv
        water_heater_training.csv
        hvac_training.csv
        backed_loads.csv
        ev_metrics.csv
        ev_states.csv
        *_solve_history.csv     # MPC devices only; detailed tail flag included
        EV_*_actions.csv
        EV_*_plans.json
        hpwh_*                  # objective/scenario diagnostics and plan CSVs
      figures/                  # case-local HPWH, HVAC, whole-home, and per-EV plots
```

EV plots show home/away, SOC, electricity price, EV power, and actual
charging/idle/discharging/away state. Thermal plots retain
legacy formatting, with extra three-case overlays. Solver summaries include each
home/device/case, zero attempts for baseline devices, failures, mean/max solver-call
time, and total solver time divided by comparison-window plant timesteps. CSVs
also report mean/max construction-plus-solve time, OCP seconds per plant timestep, available
backend time, limits/timeouts, accepted incumbents and fallbacks. The single
`reports/run_report.md` consolidates these with all homes/devices/cases' energy,
cost, peaks, cycle starts, EV final SOC/health and comfort violations/trip deficits,
estimated/reference degradation dollars, operating/degradation/net savings versus all-baseline, and thermal comfort.
Machine-readable tables and detailed histories are retained.

Cycle counts mean observed starts, not battery equivalent-full cycles. An active
first sample counts as a start; safety interruptions and subsequent restarts also
count. EV charging/discharging starts are separate; HPWH counts compressor starts;
HVAC counts heating/cooling compressor starts. Comfort metrics are realized
degree-hours (HPWH average-tank shortfall; HVAC thermostat deviations), not OCP
weighted costs. Whole-window SOC health/comfort diagnostics use SOC-fraction-hours;
departure comfort also reports event counts and mean/max SOC-fraction shortfalls.
Native reference-tail attempts are
marked in detailed histories and excluded from comparison-window summaries.

Plots retain all three trajectories; reported net savings always uses
all-baseline as the comparator, never the twin. Different
final SOCs can appear as savings; inspect stored energy and actual trip shortfalls.
Net savings includes the change in NMC SOC-aware reference degradation cost
relative to baseline. Aging uses new minus residual battery value; see
[EV_DEGRADATION.md](EV_DEGRADATION.md) for the additive NMC stress model and
required home-config migration from the obsolete B2 parameters.
The old profit and ambiguous savings columns are removed from new outputs.
Cost optimization does not guarantee peak reduction.

Legacy runners/APIs adapt their inputs into this pipeline and warn; they no longer
own separate EV simulation loops. Legacy CLI flags remain compatibility inputs;
they cannot override the new control file. The old TOU-avoidant EV controller is
removed, including under its compatibility import name.

## Focused validation

```powershell
.\.venv\Scripts\python.exe -B -m unittest test.test_ev_degradation test.test_ev_formulation test.test_unified_controls test.test_unified_hems test.test_ev_v2h test.test_ev_v2h_integration -q
```

The pre-existing repository-wide legacy API/fixture errors are documented in
[EV_V2H_VALIDATION.md](EV_V2H_VALIDATION.md); its counts describe the earlier
revision, not a new validation result. Unified validation results are recorded in
[UNIFIED_VALIDATION.md](UNIFIED_VALIDATION.md). Legacy recovery copies and their
completeness limits are documented in the
[archive](../../archive/legacy_hems_2026-10-07/README.md).

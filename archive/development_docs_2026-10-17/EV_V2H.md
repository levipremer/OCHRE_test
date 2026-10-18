# EV smart charging and V2H

> Historical guide, superseded October 17, 2026. Relative paths refer to its
> former examples/single_home_hems location; use the current example README.

Historical October 6 interface/formulation notes. For current behavior, use
[UNIFIED_PIPELINE.md](UNIFIED_PIPELINE.md), including the October 12 EV equations.
The old full-charge baseline, health/trip penalties, convex shortfall relaxation
and projector cycling rules below are superseded, not current instructions.

## Opt-in and execution

The default is immediate charging to full, without V2H or optimization.
Existing `homes.json` EV entries without `hems` use native OCHRE EV equipment.
`HEMSController()` and the general runner default to no external controls.
The old TOU rule remains available through explicit `--controller ev` selection.

Add a `hems` object inside each EV entry to select the new continuous-SOC model.
Its `controller` is independently `baseline` (default) or `mpc`. The new model's
baseline also charges immediately to full. `v2h: true` permits discharge but does
not enable MPC by itself. Only one EV per home may have V2H capability; that EV
can offset the entire HVAC-plus-water-heater load, capped by its inverter rating.
Other EVs cannot discharge. Export, outage operation, and away charging are excluded.

The example overlays are illustrative, **not tuned research results**. They do
not modify the existing home files. From the repository root, for example:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_mpc.py --ev-config examples/single_home_hems/config/ev_audi_convex.json --duration-hours 72 --output-directory examples/single_home_hems/outputs/my_audi_experiment
```

Use `config/ev_nhts_milp.json` for the NHTS/MILP example. `--start-time`,
`--home-id`, and `--config-directory` select the simulation context. Without an
overlay or `hems` entries, this runner does not enable optimization. Use a new
output directory for each experiment; OCHRE replaces existing files in that run.

For paired EV and HPWH controllers, the regular `run_simulation.py` can load
`hems` entries from its source configuration and select the thermal controller
normally. Thermal controller policies are identical across the EV baseline and
controlled comparison. Existing HPWH-only workflows and objectives are unchanged.
Independent neighborhood runs dispatch opted-in homes through the same EV workflow.
The separate legacy fuel-switching `parametric_study` package is not repaired here;
its existing configuration/API incompatibilities remain.

## Configuration

Each EV retains its existing `capacity_kwh`, `charger_power_kw`, and explicit
`initial_soc_fraction`. Every new-model EV must supply the six physical/health
parameters below in `hems`; other parameters have the listed defaults.

| Parameters | Meaning / default |
| --- | --- |
| `consumption_kwh_per_km` | Required **battery-side** driving consumption, positive |
| `charge_efficiency`, `discharge_efficiency` | Required AC-to-battery / battery-to-AC efficiencies, in (0,1] |
| `max_discharge_kw` | Required nonnegative AC inverter rating; positive for V2H |
| `health_min_soc`, `health_max_soc` | Required soft health band, fractions in [0,1] |
| `controller`, `v2h` | `baseline`, `false` |
| `formulation`, `solver`, `solver_options` | `convex`, `CLARABEL`, `{}` |
| `horizon_hours`, `control_minutes` | 12, 15; aligned with the plant timestep |
| `cycling` | `none`, `persistence`, or `dwell`; default `none` |
| `electricity_weight`, `trip_weight` | 1, 100 |
| `health_low_weight`, `health_high_weight` | 1, 1; independent penalties |
| `persistence_weight` | 0; active only with `cycling: persistence` |
| `dwell_on_minutes`, `dwell_off_minutes` | 30, 30; active only with `cycling: dwell` |
| `max_charge_starts_24h`, `max_discharge_starts_24h` | `null` disables each rolling start cap |
| `projection` | `nearest` (default) or `threshold` |
| `projection_tracking_weight`, `projection_switch_weight` | 1, 0 |
| `projection_threshold` | 0.5, fraction of available directional rated power |
| `source`, `source_file`, `source_vehicle` | `nhts`; neighborhood EV file; seeded selection unless an Audi vehicle key is given |
| `seed`, `history_days`, `forecast_margin_hours` | 0, 0, 12 |

Consumption is not charging efficiency. Trip demand is
`distance_km * consumption_kwh_per_km`; SOC depletion is demand divided by
`capacity_kwh`. Do not apply charging efficiency a second time to trip energy.
Capacity and consumption changes do not resample mobility.

Supported numerical backends: SCIPY for convex/LP or MILP, CLARABEL for convex
only, and optional GUROBI for either formulation. GUROBI requires a working local
installation/license. There is no automatic solver substitution. Invalid or
unavailable backends fail explicitly; runtime solver failures are logged and use
an immediate-charge target through the same actuator safety/cycling filter.

The seven planned variants are convex basic, MILP basic, MILP persistence, MILP
dwell, MILP cycle caps, MILP caps+persistence, and MILP caps+dwell. Convex mode can
also use persistence and actuator-level dwell/cycle caps. Discrete cycling
constraints remain in the projection, not in the convex OCP.

## Mobility and forecast semantics

- NHTS retains the existing cleaning/sampling functions. The adapter generates a
  fixed, seeded 365-day reference realization, joins overlapping away trips, and
  wraps it for history and forecast coverage. Its legacy distance-to-energy
  calculation is ignored; the new configured battery-side consumption is used.
- Audi accepts the supplied 20- and 100-vehicle `.parquet` or `.csv.gz` files.
  `arrival_distance_km` alone supplies distance. Signed SOC fields remain
  diagnostics, never driving energy or availability. No negative-SOC special case.
- Audi uses conservative contiguous away-bin bouts. Multiple episodes in a bin
  are combined; exact sub-15-minute mobility cannot be recovered from these files.
  Arrival distance can occur in the last away bin or first home bin.
- A seeded source vehicle and interior source start preserve local clock time
  and weekday. The central half of eligible starts is sampled. Entire requested
  history + simulation + forecast coverage is checked. Incomplete/missing-distance
  bouts and DST-crossing windows are not invented; invalid windows are excluded.
  Automatically selected vehicles are tried in seeded order; explicit vehicles
  fail if no complete window exists. Exclusions and chosen source dates are saved.
- For horizon sweeps, set `forecast_margin_hours` to the largest tested horizon
  and `history_days` to the largest required history, keeping seed and duration
  fixed. This prevents source-window selection changing with the horizon.
- All home EVs get coverage through the largest configured EV horizon. Home means
  plugged in. Initial SOC is applied at the simulation start; trips debit at return.
  Zero OCHRE initialization time is required.

The forecast container carries complete per-EV trips/availability plus a plant-grid
HVAC/WH electrical-load forecast. An advance reference run uses the same weather,
thermal state, schedules, and thermal controls. Its first simulation-length portion
is the no-MPC EV baseline. Thermal load equality is checked after the controlled run.
No stochastic EV forecast or scenario probabilities are introduced.

## OCP and actuator

Positive AC power charges; negative power supplies the backed load. Charging adds
`dt * eta_charge * P_charge` battery kWh; discharging removes
`dt * P_discharge / eta_discharge`. Availability, hardware limits, and SOC [0,1]
are hard. The health band and departure energy deficit are soft costs. There is
no health reserve added to the departure target and no hard departure-sufficiency
constraint. A returning trip consumes `min(demand, available battery energy)`;
unserved trip energy is reported without making SOC negative.

MILP modes are mutually exclusive: idle, rated charge, and load-following discharge
capped by the inverter. For trip demand D, pre-trip battery energy B, and shortfall q,
the MILP enforces `q = max(D-B, 0)`. Convex mode uses the approved relaxation
`q >= D-B`, `0 <= q <= D`. It may preserve predicted energy by overstating q;
`max_planned_relaxation_gap_kwh` reports this limitation. The plant always applies
exact trip accounting. Convex simultaneous charging/discharging can also occur in
the plan; projection selects one physical mode.

Control intervals are split at departures/returns. The OCP integrates capped
backed load and its price product at plant resolution. Reoptimization happens at
the configured interval or the next mobility event, whichever comes first. If N
is the number of resulting blocks, the objective is:

- Electricity: `electricity_weight/N * sum(dt * price * (backed + charge - discharge))`.
- Trips: `trip_weight * sum(q/capacity)`.
- Health: separate weighted sums of lower/upper SOC hinge penalties.
- Persistence: `persistence_weight/N * sum(dt * (abs(delta charge) + abs(delta discharge)))`,
  including variation from the last realized power; zero outside persistence mode.

Projection candidates are idle, rated charge, and load/inverter-limited discharge.
`nearest` minimizes weighted squared power-tracking error plus a configurable
switch penalty and, in persistence mode, weighted absolute power change.
`threshold` first maps the requested power using the directional threshold, then
uses the same feasibility filter. MILP actions are not reweighted by convex
projection penalties. Realized charging/discharging starts have separate rolling
24-hour histories. Dwell is carried between solves. Availability, zero backed load,
and SOC protection can interrupt dwell; overrides are reported. Hardware clipping
may also produce partial power at a SOC boundary.

No terminal-SOC constraint/value or degradation cost is added. **Always compare
final SOC alongside cost/energy**: lower final stored energy can look like savings.
Projection can degrade planned departure readiness; actual deficits, not planned
relaxation variables, determine reported insufficient-SOC events.

## Outputs and checks

`data/ev_comparison.csv` contains per-EV baseline/controlled energy, cost, power,
final SOC, departure shortfalls, health violations, starts, and overrides.
`household_comparison.csv` adds matched whole-home cost and peak metrics.
Raw output separates beginning/end SOC, served trip energy, and unserved energy.

Timing includes failed attempts: `mean_solve_seconds` is solver-call wall time
divided by OCP invocations; `solve_seconds_per_sim_timestep` divides total solver
time by plant timesteps (held-action steps count in the denominator).
`mean_ocp_build_and_solve_seconds` additionally includes OCP construction.
Baseline timing is zero. Per-controller solve histories, projected actions, plans,
trip tables, replay metadata, and paired power/SOC figures are saved.

Run focused tests with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest test.test_ev_v2h test.test_ev_v2h_integration -q
```

These cover energy conservation, exact/relaxed trip shortfalls, both Audi formats
and cohorts, NHTS wrapping, projection/history, all planned variants, independent
EV configurations, and real-OCHRE matched runs. GUROBI execution requires the optional
local dependency/license and is not established by SCIPY/CLARABEL tests.


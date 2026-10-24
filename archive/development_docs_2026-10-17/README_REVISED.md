# OCHRE HEMS simulator: HPWH MPC, EV charging, and V2H

> Historical development guide, superseded October 17, 2026. Relative paths
> describe its original examples/single_home_hems location. Use the current
> example README instead; later passages preserve earlier development decisions.

October 17: scenario EV MPC is available through the unified runner, preserving
all per-EV preferences. See [EV_SCENARIO.md](EV_SCENARIO.md) for five-path kNN,
causal non-overlapping history, persistence prefix and per-path rainflow checks.
The current daily-minimum terminal value is preserved by explicit user direction;
it supersedes the earlier mean-price description below.

For the current three-case, home/control-config pipeline, use
[UNIFIED_PIPELINE.md](UNIFIED_PIPELINE.md). The guide below describes the earlier
two-case interface; its TOU rule and mobility-triggered solving are superseded.
October 12 EV revision also supersedes its charge-to-full baseline, health/trip
penalties, split convex powers, shortfall relaxation and projector cycling rules.
October 14 adds comfort-min baseline charging, control-config health/comfort bands,
selectable rainflow/linear degradation, shared NMC battery calibration and SOC-aware reference
evaluation. See [EV_DEGRADATION.md](EV_DEGRADATION.md). Signed chemical-power convex
MPC, end/start budgets and a 5% MILP gap policy remain. October 16 replaces the
optional moving SOC-floor cost with `v*(e0-eN)/N`: measured faded capacity,
full-day average tariff as USD per chemical kWh, and no terminal weight.
`enforce_terminal_soc` remains the switch; the old `terminal_soc_weight` is retired.

October 14 study revision: [EV_STUDY.md](EV_STUDY.md) documents Audi-only,
initial-SOC-varying MILP comparisons, frozen schedules and explicit execution.
The original 30-job study is complete; revised controllers have not been rerun across it.
New outputs report operating/degradation/net
savings versus baseline (including baseline NMC reference aging), replacing profit.
The NMC revision retains additive fade, uses signed SOC stress and the Mathews
depth power law, and prices degradation at new minus residual battery value
($119 minus $0 per kWh by default). Physical fade retains the 3000-cycle/80%-SOH
calibration; monetary aging uses the 70%-SOH budget, extrapolating 4500 cycles.
EV SOH now evolves from explicit `physical.initial_soh` (supplied value .85).
Reference rainflow ages every plant, regardless of OCP cost choice. SOC uses faded
capacity; MPC freezes the latest measured SOH only within each horizon. Scored
aging sums continuous damage increments, excluding warmup/cooldown while retaining
cycle history. New health plots and battery checkpoints are saved alongside trajectories.
Comfort now averages only departure shortfalls. MILP persistence penalizes
charge/discharge/idle state changes; convex persistence uses power variation.
No nonlinear early-life SEI model is used.
Smoothed/hh are deferred. The revised January/February study stages 30 jobs:
three paired Audi traces, two months, one convex and four MILP cost formulations.
Comfort/active health weights are 1000. Cross-formulation plots and statistics
keep advanced and non-V2H cases separate. Both normal runs and studies support
controller-active `warmup_hours` and `cooldown_hours` (each default 24), excluded
from metrics; `start_time` begins evaluation. Cooldown is not forced recharge.
Rainflow additionally requires its own dollar-error check independent of the
5% MILP objective gap. See EV_STUDY.md and EV_DEGRADATION.md for details.

Current configuration templates: [home](config/unified_home.template.json) and
[control](config/unified_control.template.json). Both are valid JSON with an
ignored `_comments` documentation object. Unified configs are self-contained;
native OCHRE, `home_factory`, HPXML and other shared utilities remain dependencies.
Independent cap/dwell/persistence, bounded quiet solvers, three-case state plots,
and `reports/run_report.md` are documented in the current unified guide. Do not
use the historical overlay/CLI/report instructions below for new studies.

User guide for the improved simulator, verified against the local implementation
on 2026-10-06. This is a separate revision; [the original README](README.md) is
preserved. Run all commands from the `OCHRE_test_grant` repository root.

Contents: [Capabilities](#1-capabilities-and-operating-boundaries) ·
[Setup](#2-installation-and-preflight) · [Quick starts](#3-choose-an-entry-point) ·
[HPWH](#4-hpwh-configuration-and-forecasts) · [EV/V2H](#5-ev-configuration) ·
[Combined/neighborhood](#6-combined-controls-and-five-home-runs) ·
[Outputs](#7-outputs-and-interpretation) · [Tests](#8-tests-and-troubleshooting) ·
[Architecture](#9-architecture-and-extension-points).

## 1. Capabilities and operating boundaries

- Simulate any of five configured homes using native OCHRE building, HVAC, and
  water-heater physics, weather, occupancy, and equipment schedules.
- Compare HPWH MPC against native water-heater control, using four saved
  predictors, convex or MILP optimization, and perfect or historical-scenario
  water-use forecasts, optionally with a first-step Random Forest correction.
- Compare immediate EV charging against per-EV smart charging/V2H, using NHTS or
  Audi mobility, continuous SOC, separate charging/discharging efficiencies,
  soft health/departure penalties, and configurable cycling/projection settings.
- Combine HPWH and EV controllers. EVs have independent solver/formulation
  settings; only one EV per home may have V2H capability.
- Run the five homes independently, aggregate their power, and report static
  transformer loading. This is not coordinated neighborhood optimization.
- Save raw/standardized results, baseline comparisons, controller diagnostics,
  solve timing, and plots.

Default EV behavior remains immediate charging to full: no V2H and no optimizer.
New EV functionality is explicitly enabled by a per-vehicle `hems` configuration.
V2H offsets only HVAC plus water-heater electrical demand, capped by the inverter;
it does not model islanded outages, grid export, or away charging. HVAC uses native
controls; the HEMS HVAC controller is a pass-through template, not HVAC MPC.

## 2. Installation and preflight

Use Python 3.12 for the documented environment. Reuse the existing `.venv` if it
is already configured; do not recreate it unnecessarily. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -r examples/single_home_hems/requirements.txt
.\.venv\Scripts\python.exe -c "import cvxpy as cp; print(cp.installed_solvers())"
```

Commands below use the environment's executable directly, so activation and a
PowerShell execution-policy change are unnecessary. On Linux/macOS substitute
`.venv/bin/python` and create the environment with `python3.12 -m venv .venv`.

| Backend | EV formulations | HPWH use | Dependency |
| --- | --- | --- | --- |
| `CLARABEL` | Convex | Convex | CVXPY environment |
| `SCIPY` | Convex/LP and MILP | MILP examples; also selectable for convex | SciPy/HiGHS |
| `GUROBI` | Convex and MILP | Convex and MILP | Optional package and working license |

To enable the optional backend locally:

```powershell
.\.venv\Scripts\python.exe -m pip install gurobipy
```

An installed package does not prove that a usable license exists. GUROBI execution
has not been validated in the recorded EV test environment. CLARABEL cannot solve
MILP. The general HPWH runner defaults to GUROBI even if only its formulation is
changed; specify the backend explicitly. EV defaults are independently convex /
CLARABEL, and unavailable requested EV solvers raise an error rather than switch.

Keep the full checkout and these inputs together:

- `examples/five_home_block/config/{neighborhood,homes}.json`: shared simulation
  settings and per-home/EV settings. Input paths are relative to the repository
  root, not to the configuration directory; absolute paths are also accepted.
- `examples/five_home_block/inputs/`: configured HPXMLs, occupancy schedules,
  weather, `ev/NHTS_tripv2pub.csv`, and Audi `.parquet` / `.csv.gz` files.
- `examples/single_home_hems/models/`: saved HPWH predictor artifacts.
- `examples/single_home_hems/external_data/water_flow_dchouse_l_per_min.csv`
  and `water_usage_forecasting/`: required for the supplied DC House replay and
  historical/RF forecast workflows.

Configuration loading checks all listed homes' files before selecting one home.
Check `neighborhood.json` points to `NHTS_tripv2pub.csv`, not its old filename.
Parquet support (`pyarrow`) is included in the repository dependencies.

## 3. Choose an entry point

| Entry point, relative to repository root | Purpose | Important options |
| --- | --- | --- |
| `examples/single_home_hems/run_simulation.py` | Native baseline, HPWH MPC, or combined HPWH+configured EV MPC | `--source-config-directory`, `--controller`, `--output-directory` |
| `examples/single_home_hems/run_water_heater_mpc_scenario.py` | Preset HPWH scenario experiments | `--optimization`, `--scenario-method`, `--title` |
| `examples/single_home_hems/run_ev_mpc.py` | EV experiments using optional vehicle overlays | `--config-directory`, `--ev-config`, required `--output-directory` |
| `examples/five_home_block/run_simulation.py` | Independent five-home simulation and aggregation | `--config-directory`, `--output-directory` |

All support `--help`. `--ev-config` belongs only to the EV runner; `--optimization`
is the scenario-runner alias for `--water-heater-optimization`. The configuration
directory option has different names in the general and EV runners.

### Native baseline

With the supplied home configuration **without `hems` entries**:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --controller none --duration-hours 1 --output-directory examples/single_home_hems/outputs/baseline_smoke
```

The general runner defaults to home `home_01`, controller `none`, and the configured
start/duration/timestep (currently 2018-01-01, 18 hours, one minute). Select
`--home-id home_02` through `home_05` as needed. Indoor temperature defaults to
21 C; use `--initial-temperature-c` or `--randomize-initial-temperature` to change
it. Keep zero initialization time for new-model EV and DC House replay runs.

`--controller ev` selects the legacy TOU blocking rule, **not the new EV MPC**.
`--controller all` composes HPWH, pass-through HVAC, and that legacy rule. If `hems`
entries exist, their EV settings supersede the legacy rule and activate the EV
comparison workflow even with `--controller none`. To disable optimization on
new-model EVs, set each `hems.controller` to `baseline`.

### HPWH quick starts

One-hour RC/convex check, followed by a longer MILP example:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --controller water-heater --water-heater-model rc --water-heater-optimization convex --water-heater-solver CLARABEL --duration-hours 1 --skip-optimizer-plan-figures --output-directory examples/single_home_hems/outputs/hpwh_convex_smoke
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --controller water-heater --water-heater-model rc --water-heater-optimization milp --water-heater-solver SCIPY --duration-hours 72 --output-directory examples/single_home_hems/outputs/hpwh_milp_72h
```

Historical/RF scenario MPC using its dedicated presets:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_water_heater_mpc_scenario.py --optimization convex --duration-hours 1 --title hpwh_scenario_smoke --skip-optimizer-plan-figures
.\.venv\Scripts\python.exe examples/single_home_hems/run_water_heater_mpc_scenario.py --optimization milp --water-heater-solver SCIPY --scenario-method daily_budget_high --title hpwh_historical_milp
```

The scenario runner defaults to 72 hours. It creates a fresh directory under
`outputs/water_heater_mpc_scenario/TITLE`, suffixing duplicate titles, and records
`run_settings.json`. It rejects `--output-directory`; use `--title` instead.

### EV quick starts

These overlays target vehicle `01A` in `home_01`, leave source configurations
unchanged, and automatically generate matched immediate-charge baselines:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_mpc.py --ev-config examples/single_home_hems/config/ev_audi_convex.json --duration-hours 72 --output-directory examples/single_home_hems/outputs/ev_audi_72h
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_mpc.py --ev-config examples/single_home_hems/config/ev_nhts_milp.json --duration-hours 72 --output-directory examples/single_home_hems/outputs/ev_nhts_72h
```

Use `--duration-hours 1` and a different output directory for a short smoke run.
The EV runner defaults to 72 hours and inherits the configuration's start/timestep;
`--start-time` changes the simulation date.
A smoke run need not include a trip or discharge event; inspect a multiday run
before drawing behavioral conclusions. Example weights are illustrative, not
tuned research settings. No overlay means no additional EV opt-in; any `hems`
settings already in the loaded home configuration still apply.

## 4. HPWH configuration and forecasts

Model, optimizer, and water forecast are independent choices:

| Setting | Choices / behavior |
| --- | --- |
| `--water-heater-model` | `state-space` (general default), `rc`, `rc-old`, `rc-stratified` |
| `--water-heater-optimization` | `convex` or `milp` (general default) |
| `--forecast-type` | `perfect` (general default) or `scenario` |
| `--water-heater-horizon-hours` | Default 12; scenario mode requires 12 |
| `--water-heater-cycle-limit` | `on`/`off`; MILP predicted-start cap, general default on |
| `--water-heater-max-cycles` | Default cap 3 when enabled |
| `--water-heater-dwell-constraints` | `on`/`off`; MILP 60-minute on / 15-minute off, default off |
| `--water-heater-hp-only` | Suppress resistance heating; RC selections automatically enable HP-only plant/baseline |
| `--water-heater-legionella-cooldown` | Optional RC six-hour / 6 C allowance, default off |

Saved models live in `models/water_heater_model`, `water_heater_rc_model`,
`water_heater_rc_model_old`, and `water_heater_rc_model_stratified`. No refitting is
needed for the supplied experiments. They are calibrated to Home 1's tank and a
12-node plant with sensors at nodes 3 and 10, numbered top to bottom. Selecting a
different home does not recalibrate its predictor. Changing `--water-heater-nodes`
or `--water-heater-sensor-nodes` requires compatible artifacts.

The two runners intentionally have different presets:

| Parameter | General runner | Scenario MILP | Scenario convex |
| --- | ---: | ---: | ---: |
| Model | state-space | rc | rc |
| Solver | GUROBI | GUROBI | CLARABEL |
| Comfort weight | 100 | 0.1 | 0.01 |
| Legionella weight | 0.001 | 0.001 | 0.01 |
| Persistence weight | 0.001 | 1 | 0.001 |
| Cycle / dwell enabled | yes / no | no / no | no / no |

Override with `--water-heater-comfort-weight`, `--water-heater-legionella-weight`,
and `--water-heater-persistence-weight`. The general CLI also exposes
`--water-heater-switching-weight` (MILP, default 0.003). Zero persistence weight
disables that penalty. Changing formulation alone does not import scenario presets.

### DC House replay and scenario construction

Use the converted one-minute `timestamp,water_flow_l_per_min` CSV, not the original
source CSV. Scenario mode selects it automatically. For perfect replay, add:

```text
--forecast-type perfect --water-draw-file examples/single_home_hems/external_data/water_flow_dchouse_l_per_min.csv --water-draw-start 2025-11-06T00:00:00
```

Append these arguments to an HPWH command. `--water-draw-start` selects the source
date; `--start-time` selects the OCHRE date. Their local times of day must match.
Replay requires a one-minute plant and zero warm-up, replaces fixture demand, and
zeroes washer/dishwasher draw channels in both runs. Fixture demand is tempered
water; hot-tank draw can differ because of mains-water mixing.

Scenario MPC uses 48 quarter-hour steps, historical paths from the preceding 30
local days, and at least seven complete paths. Solves occur on local quarter hours
after the initial measurement, retaining full forecast coverage near the run end.
Missing/ambiguous paths are excluded.

- `daily_budget_high`: historical paths with morning-budget construction weights.
- `daily_budget_high_rf_1`: additionally adjusts the first forecast step using an
  RF fitted only on data preceding replay; later steps remain historical.

Morning conditioning operates 06:00–12:00 and reduces weights on excessive
remaining demand once observed use reaches 75% of the historical median. These
construction weights affect RF correction, **not optimization probabilities**.
Both formulations use equally probable paths: electricity, legionella, and
persistence costs are arithmetic means; comfort is the maximum of per-scenario
total-horizon comfort costs. Only the first action is shared across scenarios.
This is HPWH scenario MPC; the EV controller currently uses perfect deterministic
mobility/load forecasts, not stochastic EV scenarios.

## 5. EV configuration

An EV without `hems` keeps native equipment. Adding `hems` opts into the new model;
`controller: baseline` remains immediate charging, while `controller: mpc` enables
optimization. `v2h: true` grants discharge capability but does not enable MPC alone.

Copy an example overlay and edit it for a study. Overlay keys must match vehicle
IDs in the selected home. Omitted vehicle properties are inherited; a supplied
`hems` object is rebuilt from its contents/defaults, **not recursively merged**.
For another home, update the overlay keys as well as `--home-id`.

Minimal complete overlay for the new baseline model (illustrative values):

```json
{
  "01A": {
    "initial_soc_fraction": 0.5,
    "hems": {
      "controller": "baseline",
      "source": "nhts",
      "consumption_kwh_per_km": 0.25,
      "charge_efficiency": 0.9,
      "discharge_efficiency": 0.9,
      "max_discharge_kw": 7.2,
      "health_min_soc": 0.2,
      "health_max_soc": 0.8,
      "v2h": false
    }
  }
}
```

Keep `capacity_kwh`, `charger_power_kw`, and `initial_soc_fraction` explicit in
the underlying home configuration or overlay. The first two are inherited above.
Consumption is battery-side kWh/km, **not** charging efficiency. For distance d,
trip demand is `d * consumption_kwh_per_km`; ideal SOC depletion is demand divided
by capacity. Do not apply charging efficiency again to driving demand.

All following settings are independent per EV; none inherits HPWH solver weights:

| `hems` field(s) | Default / meaning |
| --- | --- |
| `consumption_kwh_per_km` | Required, positive battery-side driving consumption |
| `charge_efficiency`, `discharge_efficiency` | Required, each in (0,1] |
| `max_discharge_kw` | Required, nonnegative AC inverter rating; positive for V2H |
| `health_min_soc`, `health_max_soc` | Required soft band within [0,1] |
| `controller`, `v2h` | `baseline`, `false` |
| `formulation`, `solver`, `solver_options` | `convex`, `CLARABEL`, `{}`; solver options use the selected CVXPY backend's syntax |
| `horizon_hours`, `control_minutes` | 12, 15; align with plant timestep; horizon covers at least one interval |
| `electricity_weight`, `trip_weight` | 1, 100 |
| `health_low_weight`, `health_high_weight` | 1, 1 |
| `cycling`, `persistence_weight` | `none`, 0; alternatives `persistence` or `dwell` |
| `dwell_on_minutes`, `dwell_off_minutes` | 30, 30; active for dwell mode |
| `max_charge_starts_24h`, `max_discharge_starts_24h` | `null` disables each rolling cap; nonnegative integers enable |
| `projection` | `nearest` or `threshold`; default nearest |
| `projection_tracking_weight`, `projection_switch_weight` | 1, 0 |
| `projection_threshold` | 0.5 of available directional rated power |
| `source`, `source_file` | `nhts`; defaults to neighborhood EV file; set Audi file explicitly |
| `source_vehicle`, `seed` | `null` for seeded Audi vehicle selection; seed 0 |
| `history_days`, `forecast_margin_hours` | 0, 12 |

For charge-only MPC use `controller: mpc, v2h: false`. To permit V2H, enable it on
exactly one EV at most per home; that EV receives the entire backed-load forecast.
The restriction also counts an EV configured as baseline with `v2h: true`.
Other vehicles may optimize charging with different settings but cannot discharge.

The seven experiment variants are convex basic; MILP basic, persistence, dwell,
cycle caps, caps+persistence, and caps+dwell. Configure the two caps independently;
`null` disables them and zero forbids new starts. Convex may also use persistence
and actuator-level dwell/caps; binary dwell/cap constraints are not added to its OCP.

### Mobility adapters and time mapping

| Dataset | Processing | Time handling |
| --- | --- | --- |
| NHTS | Reuses existing cleaning and commute/trip sampling; retains distances, replaces legacy energy inference | Seeded 365-day reference; annual wrapping allowed |
| Audi | Reads pre-cleaned 20- or 100-vehicle Parquet/CSV.GZ; uses `arrival_distance_km` | Seeded interior window with enough history, duration, and forecast; no wrapping |

Audi signed SOC fields are retained as diagnostics without a negative-value special
case. They do not determine driving energy. Capacity/consumption changes reuse the
same distance schedule. Contiguous away bins define trips, with arrival distance
included from the last away/first home bin. Trips are conservatively aligned to
the plant grid; exact sub-15-minute Audi timing cannot be recovered.

Audi mapping preserves local weekday/clock, samples the central half of feasible
starts, and rejects missing-distance, incomplete, or DST-crossing windows. Automatic
vehicle selection retries in seeded order and records exclusions; an explicit
vehicle fails if its requested coverage is infeasible. If coverage fails, inspect
the saved/available data and reduce windows or choose another vehicle; data are not
silently wrapped. For horizon sweeps, keep seed/duration/history fixed and set
`forecast_margin_hours` to the largest tested horizon. All EVs receive coverage
through the largest home EV horizon. History coverage is available for experiments;
it does not make EV forecasts stochastic.

### Physical accounting, OCP, and actuator

Positive AC power charges; negative power supplies backed load. Over dt hours,
charging adds `dt * eta_charge * P_charge`; discharging consumes
`dt * P_discharge / eta_discharge` from the battery. At home means plugged in;
away means neither charging nor V2H. Initial SOC applies at simulation start.

SOC [0,1], availability, and hardware limits are hard constraints. The health band
and insufficient departure energy are soft costs; there is no hard trip-readiness
requirement or added health reserve. Complete trip demand is known at departure,
even if return lies beyond the horizon. At return, demand is deducted once up to
available battery energy, with unserved energy reported rather than negative SOC.

MILP has mutually exclusive idle/rated-charge/load-following-discharge modes and
exact `shortfall = max(trip demand - departure battery energy, 0)`. Convex uses the
approved lower-bound shortfall relaxation; it can overstate shortfall or plan
simultaneous charging/discharging. Neither is silently treated as a physical action.

Projection selects one feasible mode. Nearest projection balances squared tracking
error, switch penalty, and enabled persistence; threshold projection first maps the
request by a directional fraction. Realized dwell/start history carries between
solves. Safety/availability/SOC can override dwell, and clipping can yield partial
power at a SOC boundary. MILP modes are not reweighted by convex projection costs.
Solves occur at control intervals or earlier mobility events.

No terminal-SOC value/constraint or degradation model is added; health costs are
SOC-band penalties, not a calibrated aging model. See [EV_V2H.md](EV_V2H.md) for
the precise objective normalization, projection semantics, and relaxation caveats.

## 6. Combined controls and five-home runs

For combined HPWH+EV control, prepare a separate configuration directory containing
copies of both JSON configuration files. In its `homes.json`, put the overlay's
`hems` object inside the corresponding existing `evs` entry and apply any desired
initial-SOC/capacity updates. Retain all required home/EV fields and original
repository-relative input paths. The following assumes that prepared directory is
`examples/single_home_hems/config/combined`:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --source-config-directory examples/single_home_hems/config/combined --controller water-heater --water-heater-model rc --water-heater-optimization convex --water-heater-solver CLARABEL --duration-hours 72 --output-directory examples/single_home_hems/outputs/combined_72h
```

This directory is a user-prepared example, not a supplied configuration. Add HPWH
scenario/replay flags if desired; EV settings remain in JSON. The EV-only runner
does not expose HPWH CLI options. Keep distinct output directories for experiments.

The EV workflow freezes mobility, runs an extended thermal reference through the
last EV horizon, then runs EV control with that exact HVAC+WH electrical forecast.
It verifies matched thermal loads. In combined runs, **the baseline retains the
same HPWH policy** and changes only EV control. It is not an all-native baseline
or joint centralized HPWH/EV optimization. Use a separate HPWH-only experiment
to isolate HPWH-control savings. Provide input coverage for the reference extension.

For the five-home configuration (retain all five homes):

```powershell
.\.venv\Scripts\python.exe examples/five_home_block/run_simulation.py --config-directory examples/five_home_block/config --output-directory examples/five_home_block/outputs/my_block_run
```

To enable V2H in this workflow, use a prepared five-home configuration directory
with `hems` entries. Opted-in homes dispatch through the EV comparison; others keep
their legacy path. EV details are under `OUTPUT/ev/HOME_ID/`; top-level data contain
per-home standardized results, `neighborhood_aggregate.csv`, and
`output_mapping.json`, with 12 standard neighborhood/home figures.

The neighborhood CLI exposes compressor deadband/timing and defrost options; use
`--help` for those overrides. Transformer loading is reporting, not a controller
constraint; voltage, losses, temperature, aging, and islanding are not modeled.
Synchronized coordination is an unimplemented extension point. The separate legacy
fuel-switching `parametric_study` has known API/import incompatibilities and is not
part of this verified EV workflow.

## 7. Outputs and interpretation

General and EV runners can reuse/overwrite files in a chosen output directory;
choose a new directory for each run. Generated outputs are Git-ignored. Save a copy
of the input configurations and CLI invocation with each research experiment;
only the scenario wrapper automatically saves its CLI settings JSON.

### Common and HPWH-only results

| Path within run output | Content |
| --- | --- |
| `data/raw/` | Native OCHRE results and resolved schedules; reference under `uncontrolled/` |
| `data/HOME_ID_standardized.csv` | Controlled whole-home/end-use series |
| `data/HOME_ID_uncontrolled_standardized.csv` | Matched baseline on the simulation clock |
| `data/HOME_ID_hems_controls.csv` | Applied commands and prices |
| `data/HOME_ID_water_heater_operating_costs.csv` | HPWH-only workflow's baseline/controlled cost, energy, average power |
| `data/*water_heater_sensors.csv`, `data/*training_data.csv` | HPWH-only workflow's sensor and identification records |
| `data/water_heater_sensor_layout.json` | HPWH-only sensor layout |
| `data/optimization_solve_history.csv` | HPWH-only attempted solve status/timing |
| `data/optimization_objective_component_history.csv` | HPWH-only successful objective terms |
| `data/optimization_scenario_diagnostics_history.csv` | HPWH-only scenario/RF diagnostics |
| `data/water_draw_metadata.json` | Replay metadata when saved by the HPWH-only workflow |
| `figures/` | Workflow-dependent plots; HPWH optimizer plans in `water_heater_optimizer_plans/` |

`--skip-optimizer-plan-figures` suppresses HPWH per-solve images, not ordinary
summary plots. Scenario plan means are not a realized physical trajectory.

### EV workflow results, including combined runs

| Path within run output | Content |
| --- | --- |
| `data/ev_comparison.csv` | Per-vehicle baseline/controlled charge/V2H/net energy, cost, peaks, final SOC, deficits, starts, overrides, timing |
| `data/household_comparison.csv` | Whole-home energy, cost, mean power, peak power |
| `data/EV_ID_solve_history.csv` | Attempted OCP status, solver time, construction+solve time |
| `data/EV_ID_actions.csv` | Projected actions and override diagnostics |
| `data/EV_ID_plans.json` | Optimized trajectories, objective, trip-shortfall relaxation gaps |
| `data/EV_ID_trips.csv` | Frozen mobility table |
| `data/ev_replay_metadata.json` | Source selection, time mapping, coverage/exclusions |
| `figures/ev_comparison.png`, `figures/whole_home_comparison.png` | Paired EV power/SOC and household comparison |

`EV_ID` means the equipment name, e.g. `EV_01A`. Solver/action/plan files are
written for MPC EVs only. Baseline timing is zero. Raw EV results include beginning
and end SOC, departure deficits, served/unserved trip energy, and safety overrides.
The raw thermal reference extends beyond the comparison duration; use the aligned
standardized/comparison files when calculating baseline metrics.

The EV dispatch path does **not** emit the full HPWH-only diagnostic/plot bundle,
even when a HPWH controller is present. Do not expect the HPWH optimizer-history,
training, or per-plan files listed above from that combined path.

Timing definitions in `ev_comparison.csv`:

- `mean_solve_seconds`: solver-call wall time divided by attempted OCP invocations.
- `solve_seconds_per_sim_timestep`: total solver-call time divided by all plant
  steps, including steps that hold an action without solving.
- `mean_ocp_build_and_solve_seconds`: mean OCP construction plus solution time.

Failed attempts count in timing. These are not total simulation runtime; the
reference run, plotting, and other overhead are additional.

Before claiming a benefit, check successful solves, physical SOC, departure
shortfalls, cycling, and final SOC. The supplied tariff is $0.15/kWh off-peak and
$0.30/kWh from 15:00 to 18:00. It is illustrative, not a utility tariff. To change
it programmatically, supply a `PriceSchedule` to `HEMSController`; there is no
tariff CLI flag. EV energy/cost uses signed AC power, while household metrics
include all loads. Lower final battery energy can appear as cost savings; these
are not equal-terminal-energy comparisons. Cost minimization does not guarantee
a lower household peak. Inspect `max_planned_relaxation_gap_kwh` and actual
departure deficits rather than treating convex predictions as realized behavior.

## 8. Tests and troubleshooting

Focused EV tests and full discovery:

```powershell
.\.venv\Scripts\python.exe -B -m unittest test.test_ev_v2h test.test_ev_v2h_integration -q
.\.venv\Scripts\python.exe -B -m unittest discover -s test
```

Recorded validation: all 29 new EV tests passed, including real-OCHRE, EV+HPWH,
and five-home integration. Full discovery ran 284 tests: 158 passed and 126 legacy
API/fixture errors; zero assertion failures. **The full suite is not green.** Both
72-hour Audi/CLARABEL and NHTS/SCIPY runs completed with no solver failures and
battery energy-balance error below 1.3e-13 kWh. See
[EV_V2H_VALIDATION.md](EV_V2H_VALIDATION.md) for evidence and accepted limitations.

| Symptom | Check / action |
| --- | --- |
| No EV optimization or discharge | Check `hems.controller: mpc`, `v2h`, availability, SOC, backed load, and solve history; `--controller ev` is the old rule |
| Unknown vehicle in overlay | Match keys to the selected home's `vehicle_id` values |
| Multiple-V2H validation error | Set all but one EV's `v2h` false, even for baseline EVs |
| Missing solver / license | Inspect `installed_solvers()`; install/license requested backend or explicitly select an available compatible one |
| EV runtime solver failure | Inspect per-EV history; fallback is an immediate-charge target passed through actuator safety/cycling filters |
| HPWH simulation completes without optimization | Inspect `optimization_solve_history.csv`; missing/incompatible artifacts can leave native control active; solver failure retains the previous setpoint |
| Audi window unavailable | Check vehicle, history, duration, future margin, missing data, and DST; do not extrapolate or wrap the short record |
| Forecast/alignment error | Keep horizon/control periods aligned with the plant step, zero EV warm-up, and complete future input coverage |
| DC House replay error | Use converted data, one-minute plant, matching source/simulation local time, and sufficient prior/future records |
| HPWH sensor mismatch | Restore 12 tank nodes and sensors 3/10 or provide compatible recalibrated artifacts |
| Perfect thermal forecast mismatch | Check thermal policies, initial states, schedules, and randomness; do not ignore the failed equality check |
| Apparently large savings / occasional health-band violation | Compare final stored energy; health band and departure deficits are intentionally soft, physical SOC bounds are hard |
| Legacy test failures | Compare against the recorded failure categories before attributing them to a new change |

## 9. Architecture and extension points

Configuration -> schedules/forecast -> device controllers -> OCHRE plant step ->
measured state -> next control decision -> standardized results/comparison.

| Module | Responsibility |
| --- | --- |
| `hems.py`, `single_home_hems/controllers/{base,composite}.py` | `HEMSController`, `DeviceController`, and composite dispatch |
| `single_home_hems/simulation.py` | Stepwise plant loop, state exposure, workflow selection |
| `single_home_hems/forecast.py` | Shared schedule/price, water scenarios, EV mobility, backed load |
| `single_home_hems/controllers/ev_mpc.py` | Independent EV OCPs, projection, realized cycling history |
| `single_home_hems/ev_simulation.py` | Paired EV runs, oracle checks, metrics, persistence, plots |
| `../five_home_block/five_home_block/ev_config.py` | Validated per-EV options |
| `../five_home_block/five_home_block/ev_schedule.py` | Common trips, Audi/NHTS adapters and coverage |
| `../five_home_block/five_home_block/ev_equipment.py` | Physical bidirectional SOC/power and trip accounting |
| `../five_home_block/five_home_block/home_factory.py` | Opt-in equipment integration with native OCHRE |

Reusable APIs include `SingleHomeHEMSSimulation`, `HEMSController`, and
`apply_ev_overlay` in `run_ev_mpc.py`. Freeze schedules and use fresh controller
state for fair comparisons. Do not change objective normalization, forecast
probabilities, physical constraints, or HPWH mathematics without the explicit
formulation approval required by the repository's `AGENTS.md`.

Further references: [detailed EV formulation/configuration](EV_V2H.md),
[original HPWH guide](README.md), [validation report](EV_V2H_VALIDATION.md), and
[change log](../../CUSTOM_CHANGES.md).


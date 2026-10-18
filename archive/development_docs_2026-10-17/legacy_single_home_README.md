# Historical single-home HEMS guide (superseded)

Archived October 17, 2026. Relative paths describe the original
`examples/single_home_hems/` location; use the current guide there instead.

Run one OCHRE dwelling with water-heater model predictive control (MPC), a
matched uncontrolled baseline, and saved data and figures. Water-heater control
supports convex and mixed-integer linear (MILP) optimization, four saved
predictors, and either perfect water-use forecasts or historical scenarios.
DC House water usage is included for replay and scenario generation.

Run the commands below **from the repository root**. This example uses the local
OCHRE source and the shared inputs in `examples/five_home_block`; keep the full
repository checkout together.

## Install

Python 3.12 is suitable for this example. The repository declares Python
`>=3.9,<3.13` support.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r examples/single_home_hems/requirements.txt
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The runtime requirements add CVXPY and the Random Forest dependencies to OCHRE.
CLARABEL handles convex problems; SCIPY provides a MILP solver without a
commercial license. GUROBI is also supported and requires its Python package
and a working license appropriate for the problem size:

```bash
python -m pip install gurobipy
```

| Optimization | `--water-heater-solver` | Setup |
| --- | --- | --- |
| `convex` | `CLARABEL` | Included with CVXPY |
| `milp` | `SCIPY` | Uses SciPy/HiGHS from the installed dependencies |
| `convex` or `milp` | `GUROBI` | Install `gurobipy` and configure a license |

The general runner defaults to GUROBI, including when only the optimization
flag is changed. Select a solver explicitly as in the examples below. CLARABEL
cannot solve MILP problems. Multi-scenario MILP runs can take substantially
longer than convex runs.

## Quick starts

Start with a one-hour RC/convex run using the original household water schedule:

```bash
python examples/single_home_hems/run_simulation.py \
  --controller water-heater \
  --water-heater-model rc \
  --water-heater-optimization convex \
  --water-heater-solver CLARABEL \
  --duration-hours 1 \
  --skip-optimizer-plan-figures
```

For a full three-day MILP run with perfect water-use forecasts:

```bash
python examples/single_home_hems/run_simulation.py \
  --controller water-heater \
  --water-heater-model rc \
  --water-heater-optimization milp \
  --water-heater-solver SCIPY \
  --duration-hours 72 \
  --output-directory examples/single_home_hems/outputs/home_01/rc_milp_72h
```

Run a one-hour scenario smoke check on the included DC House water usage:

```bash
python examples/single_home_hems/run_water_heater_mpc_scenario.py \
  --optimization convex \
  --duration-hours 1 \
  --title convex_smoke \
  --skip-optimizer-plan-figures
```

Run the complete three-day scenario simulations:

```bash
# Convex, CLARABEL, first-step Random Forest adjustment
python examples/single_home_hems/run_water_heater_mpc_scenario.py \
  --optimization convex --title convex_rf

# MILP, GUROBI; a working license is required
python examples/single_home_hems/run_water_heater_mpc_scenario.py \
  --optimization milp --title milp_rf

# MILP using SCIPY and historical scenarios without the RF adjustment
python examples/single_home_hems/run_water_heater_mpc_scenario.py \
  --optimization milp --water-heater-solver SCIPY \
  --scenario-method daily_budget_high --title milp_historical
```

Every scenario-runner invocation creates a new directory under
`outputs/water_heater_mpc_scenario/`. Repeated titles gain `_2`, `_3`, and so on.
Its `run_settings.json` records the resolved command-line settings. Use
`--title` or `--run-title` with this runner; `--output-directory` is available
only in the general runner.

Both runners support `--help`:

```bash
python examples/single_home_hems/run_simulation.py --help
python examples/single_home_hems/run_water_heater_mpc_scenario.py --help
```

## Choose a water-heater MPC variant

The model, optimization, and water-use forecast are separate choices. All four
model selections are available with `convex` or `milp` and with `perfect` or
`scenario` forecasts. Use the general runner to specify these explicitly:

| Option | Accepted values | General-runner default |
| --- | --- | --- |
| `--water-heater-model` | `state-space`, `rc`, `rc-old`, `rc-stratified` | `state-space` |
| `--water-heater-optimization` | `convex`, `milp` | `milp` |
| `--forecast-type` | `perfect`, `scenario` | `perfect` |
| `--scenario-method` | `daily_budget_high_rf_1`, `daily_budget_high` | `daily_budget_high_rf_1` |
| `--water-heater-horizon-hours` | Positive hours; scenario mode requires `12` | `12` |
| `--water-heater-cycle-limit` | `on`, `off` | `on` |
| `--water-heater-dwell-constraints` | `on`, `off` | `off` |

The saved predictors are ready to load; fitting a model is not a prerequisite:

| Model | Artifact under `models/` | Description |
| --- | --- | --- |
| `state-space` | `water_heater_model/metadata.json` and `model_arrays.npz` | Identified state-space predictor |
| `rc` | `water_heater_rc_model/metadata.json` | Two-node RC predictor fitted using measured electric power |
| `rc-old` | `water_heater_rc_model_old/metadata.json` | Earlier thermostat-grid RC fit |
| `rc-stratified` | `water_heater_rc_model_stratified/metadata.json` | RC fit with a searched stratification thickness |

These artifacts use Home 1's tank and the default 12-node plant with virtual
sensors at nodes 3 and 10 (top-to-bottom numbering). They are not separately
calibrated for every selectable home. Keep the default tank/sensor layout when
using the included artifacts. RC selections automatically run the plant and
baseline in heat-pump-only mode; `--water-heater-hp-only` also enables this for
the state-space selection.

For example, select the state-space predictor with convex scenario MPC:

```bash
python examples/single_home_hems/run_simulation.py \
  --controller water-heater \
  --water-heater-model state-space \
  --water-heater-optimization convex \
  --water-heater-solver CLARABEL \
  --water-heater-hp-only \
  --forecast-type scenario \
  --scenario-method daily_budget_high_rf_1 \
  --water-draw-start 2025-11-06T00:00:00 \
  --duration-hours 72
```

Change `--water-heater-model` in this command to run the other saved predictors.
The scenario runner also accepts this option, although its default is `rc`.

### Weights and operational options

The following table reports the **current code defaults**, which differ between
the two entry points. Selecting convex in the general runner does not load the
scenario runner's weight preset.

| Setting | General runner | Scenario runner: MILP | Scenario runner: convex |
| --- | ---: | ---: | ---: |
| Comfort weight | `100` | `0.1` | `0.01` |
| Legionella weight | `0.001` | `0.001` | `0.01` |
| Persistence weight | `0.001` | `1` | `0.001` |
| Cycle limit | on | off | off |
| Dwell constraints | off | off | off |
| Solver | GUROBI | GUROBI | CLARABEL |

Use `--water-heater-comfort-weight`, `--water-heater-legionella-weight`, and
`--water-heater-persistence-weight` to supply explicit nonnegative values.
A zero persistence weight disables that penalty. The current implementation's
weights and time normalization are preserved.

Cycle and dwell constraints apply to MILP. `--water-heater-max-cycles` changes
the default cap of three when cycle limits are enabled. Enabling dwell
constraints retains the existing 60-minute minimum-on and 15-minute minimum-off
times. For the developed MILP persistence operating point without those caps:

```bash
python examples/single_home_hems/run_simulation.py \
  --controller water-heater --water-heater-model rc \
  --water-heater-optimization milp --water-heater-solver SCIPY \
  --water-heater-cycle-limit off --water-heater-dwell-constraints off \
  --water-heater-comfort-weight 0.1 \
  --water-heater-legionella-weight 0.001 \
  --water-heater-persistence-weight 1 \
  --duration-hours 72
```

For an RC predictor, `--water-heater-legionella-cooldown` enables the existing
optional six-hour / 6 C cooldown allowance. It is off by default. Attainment
memory comes from measured weighted tank temperature; a separate zero-heating,
zero-draw RC forecast supplies the cooldown mask for the existing temperature
penalty.

## DC House replay and scenario behavior

The simulation reads
`external_data/water_flow_dchouse_l_per_min.csv`, with columns `timestamp` and
`water_flow_l_per_min`. This is one-minute, interval-start, timezone-aware data
in L/min. The original `water_flow_dchouse.csv` is retained as source data;
pass the converted file to the simulation.

Scenario mode automatically selects the converted file. To use it with perfect
forecasts, provide `--water-draw-file` explicitly:

```bash
python examples/single_home_hems/run_simulation.py \
  --controller water-heater --water-heater-model rc \
  --water-heater-optimization convex --water-heater-solver CLARABEL \
  --forecast-type perfect \
  --water-draw-file examples/single_home_hems/external_data/water_flow_dchouse_l_per_min.csv \
  --water-draw-start 2025-11-06T00:00:00 \
  --duration-hours 72
```

`--water-draw-start` selects a date in the source water data. `--start-time`
selects the OCHRE simulation date; it retains the configured 2018 start unless
overridden. Replay offsets the dates while preserving local clock time, so the
two starts must have the same time of day. The scenario runner defaults to
November 6, 2025 for the water source. If the general runner receives no water
source start, the loader chooses 30 local days after the first source date.

Replay replaces fixture use and zeroes the washer/dishwasher draw channels in
both the controlled home and its baseline. It is interpreted as tempered
fixture demand; actual hot-tank withdrawal can differ because OCHRE mixes hot
and mains water. Replay requires a one-minute plant timestep and zero warm-up,
which are the supplied configuration's defaults. Missing readings, unavailable
simulation windows, and ambiguous daylight-saving mappings raise errors.

Scenario MPC solves on local quarter hours after the initial measurement. Each
solve retains a full 12-hour horizon of 48 fifteen-minute steps, even near the
simulation end. The forecast keeps complete historical paths from the previous
**30 local days** at the same clock time. Missing paths and ambiguous clock
transitions are excluded; the scenario count can therefore vary. Initialization
requires at least seven complete historical paths with the default lookback.

| Scenario method | Behavior |
| --- | --- |
| `daily_budget_high` | Historical paths with morning-budget construction weights; no RF correction |
| `daily_budget_high_rf_1` | Same paths, with an RF correction to only the first 15-minute step |

The RF fits once using data completed before replay starts. Its first-step
adjustment matches the construction-weighted scenario mean to the RF prediction;
steps 2–48 remain historical. Morning conditioning covers 06:00–12:00 and
reduces construction weights on excessive remaining morning demand after
observed volume reaches 75% of the historical median.

**Both convex and MILP scenario objectives assign equal probability to every
retained path.** Electricity, legionella, and persistence costs are arithmetic
means across paths. Comfort is the maximum of the per-scenario **total horizon**
comfort costs, with the existing weights and per-path time normalization. Only
the first action is shared across scenarios; subsequent decisions are
scenario-specific recourse.

Construction weights affect the RF adjustment, not the optimization objective.
Without RF they are diagnostic only. Consequently, the optimizer's arithmetic
first-step mean can differ from the RF prediction. Forecast diagnostics record
both means. Weather, mains temperature, and prices retain their scheduled
forecast treatment.

## Home, controller, and initialization settings

`--home-id` accepts `home_01` through `home_05`, defaulting to `home_01`.
The general runner defaults to `--controller none` (native immediate EV charging); specify
`--controller water-heater` for MPC. It inherits the supplied configuration's
2018-01-01 start, 18-hour duration, and one-minute timestep unless overridden.

| `--controller` | Behavior |
| --- | --- |
| `none` | Native OCHRE controls |
| `water-heater` | Water-heater MPC |
| `hvac` | Pass-through HVAC template; native OCHRE control |
| `ev` | Starter time-of-use charging rule |
| `all` | Water-heater MPC, HVAC template, and EV rule combined |

For example, `--controller all --duration-hours 48` in a water-heater command
also runs the EV rule. The EV rule blocks charging from 15:00 to 18:00 and
otherwise permits native charging. The illustrative electricity price is
$0.30/kWh during those hours and $0.15/kWh otherwise.

Per-EV HEMS settings explicitly opt into the separate matched EV workflow and
supersede the starter EV rule. See [EV/V2H configuration and formulation](EV_V2H.md).
Without those settings, existing native EV equipment remains unchanged.

Indoor temperature starts at 21 C for the controlled and baseline simulations.
Use `--initial-temperature-c 20` for another fixed value, or
`--randomize-initial-temperature` to opt into OCHRE's native random initialization.
Those options are mutually exclusive. `--source-config-directory` selects
another directory containing `neighborhood.json` and `homes.json`.

## Files needed to run

The simulation package includes:

- `run_simulation.py`, `run_water_heater_mpc_scenario.py`, and `hems.py` for
  launching and composing controllers;
- `single_home_hems/` for simulation, controllers, forecasts, plotting, and
  runtime training-record/predictor support;
- `models/` for the saved predictor metadata and state-space arrays;
- `external_data/` for DC House water usage;
- the forecast helper modules in `water_usage_forecasting/` used to construct
  historical paths and fit the short-lead RF at simulation startup.

The required shared files outside this directory are the local `ochre/` package,
`examples/five_home_block/five_home_block/`, its two configuration JSON files,
and the configured weather, EV trip data, HPXMLs, and schedules under
`examples/five_home_block/inputs/`. Configuration loading checks every listed
home's inputs before selecting a home. Standalone forecasting experiments,
model-fitting entry points, and controller-comparison studies are not needed to
run these simulations.

## Results and checking that MPC ran

General-runner outputs default to
`outputs/HOME_ID/CONTROLLER_OPTIMIZATION_MODEL/`, with an additional scenario
method suffix in scenario mode. `--output-directory` overrides the location.
Use a new output directory for each general-runner experiment; its default
location is reused on subsequent invocations. The scenario runner always
allocates a fresh titled directory as described above.

Each output directory contains:

| Location | Contents |
| --- | --- |
| `data/raw/` | Raw OCHRE results and resolved schedules; baseline under `uncontrolled/` |
| `data/HOME_ID_standardized.csv` | Controlled home results |
| `data/HOME_ID_uncontrolled_standardized.csv` | Matched baseline results |
| `data/HOME_ID_hems_controls.csv` | Applied equipment commands and prices |
| `data/HOME_ID_water_heater_operating_costs.csv` | Controlled/baseline energy, average power, and cost |
| `data/*water_heater_sensors.csv` | Controlled and baseline virtual tank sensors |
| `data/*training_data.csv` | Aligned water-heater and HVAC transition records |
| `data/water_heater_sensor_layout.json` | Sensor nodes and heights |
| `data/optimization_solve_history.csv` | Solver status, success, and timing for attempted optimizations |
| `data/optimization_objective_component_history.csv` | Objective terms for successful solves |
| `data/optimization_scenario_diagnostics_history.csv` | Scenario counts, candidate IDs, probabilities, and forecast diagnostics |
| `data/water_draw_metadata.json` | Replay mapping and scenario/RF metadata when water replay is used |
| `figures/` | HVAC, water heater, EV, whole-home, power comparison, and water-use plots |
| `figures/water_heater_optimizer_plans/` | One figure per successful MPC solve |

The water-heater-only selection also saves a baseline temperature/heat plot,
for seven standard figures in total. `--skip-optimizer-plan-figures` skips only
the per-solve figures; standard summaries and CSV diagnostics are retained.
Scenario plan figures summarize an ensemble, so future means are not a single
realized control trajectory. Generated outputs stay out of version control.

**Check `optimization_solve_history.csv` for successful solves.** A completed
plant simulation alone does not establish that optimization succeeded. Solver
failures retain the previous setpoint; unavailable state-space artifacts can
leave native control active. If the file is absent, check model artifacts,
sensor compatibility, run duration, and solver installation. A scenario run
must extend past its first quarter-hour control opportunity. For a GUROBI
license failure, configure the license or rerun MILP with `SCIPY` / convex with
`CLARABEL`.

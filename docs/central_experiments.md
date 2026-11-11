# Central controller experiments

`run_experiment.py` supports a controller suite for selecting established
controller families from one JSON file. It also retains the earlier experiment
configuration workflow described below. Both paths use the existing OCHRE
simulation loop and controller implementations; this launcher does not change
their objectives, constraints, forecasts, or plant dynamics. The individual
runners remain available with their existing arguments and outputs.

## Controller suite quickstart

Install the [single-home dependencies](../examples/single_home_hems/README.md#install)
and configure a Gurobi license for the supplied MILP cases. The example JSON
uses the versioned weather, forecast, model, and seed inputs under
`examples/single_home_hems/inputs/predictive/`; each example runs for one hour.

From the repository root, inspect and copy
`examples/single_home_hems/config/controller_suite.json`. Set `selected` to the
case names you want to run, edit those cases' runner arguments (home, source
inputs, dates, rate/grid scenario, controller settings) and output directory,
then validate:

```sh
.venv/bin/python examples/single_home_hems/run_experiment.py \
  --controller-config examples/single_home_hems/config/controller_suite.json \
  --validate-only
```

Run a selected case in a new or empty output directory:

```sh
.venv/bin/python examples/single_home_hems/run_experiment.py \
  --controller-config examples/single_home_hems/config/controller_suite.json \
  --variants hvac_only \
  --output-directory /tmp/ochre-hvac-study
```

Omit `--variants` to run the names listed in the config's `selected` array.
`--variants` selects named cases from the config without editing the file;
cases run sequentially. `--homes` can override the configured `home_id` for one
home at a time. The top-level `schema_version` is `1`; `variants` maps each
user-defined case name to a `controller` and an `arguments` object. `controller`
must be one of the five values below.

| `controller` | Existing runner | Predictive devices |
| --- | --- | --- |
| `hvac_only` | `run_simulation.py --controller hvac` | HVAC |
| `water_heater_only` | `run_simulation.py --controller water-heater` | Water heater |
| `joint_thermal` | `run_joint_hems.py` | HVAC and water heater in one thermal optimization |
| `whole_home_joint` | `run_whole_home_joint.py` | HVAC, water heater, and bidirectional EV in one whole-home optimization |
| `bidirectional_charger_only` | `run_ev_home_hub.py` | Bidirectional EV charger; HVAC and water heater remain under native control |

All five cases dispatch to their existing runners. Each `arguments` key is an
argument name from that runner's `--help` output, written with underscores
rather than hyphens: `duration_hours`, for example, maps to
`--duration-hours`. For HVAC-only and water-heater-only, the suite supplies
the required `--controller` value; do not put `controller` in `arguments`.
The suite also supplies an isolated `--output-directory` for each case.
Boolean flags use JSON `true` or `false`. Explicit relative input paths in
the JSON resolve relative to the JSON file. The JSON `output_directory` is
relative to that file, while `--output-directory` on the command line is
relative to the current working directory.

Keep the selected formulation and forecast explicit in each case. The supplied
`hvac_only` example specifies the eight-hour MILP HVAC formulation used in the
recent joint thermal runs, including a 16-hour native warmup. Set
`hvac_only.arguments.hvac_horizon_hours` to `20` for a 20-hour forecast horizon;
the decision interval remains 15 minutes. The example does **not**
reproduce the January 2026 study's 20-hour horizon and custom daily heating
setpoint schedule. An arbitrary daily HVAC setpoint schedule is not currently
configurable through the controller suite without a simulation input hook;
use the study-specific runner for that schedule until the input is exposed.
The other example cases have their own source windows and declared arguments;
their names alone do not guarantee a particular formulation. Read the saved
resolved arguments before interpreting results.

The suite output saves `requested_controller_config.json`,
`resolved_controller_config.json` (including the exact dispatched command and
effective parsed arguments), `run_status.json`, a per-case log under
`runner_logs/`, and each existing runner's own outputs under its case name.
Those outputs retain the runner's available commands, solve histories, plans,
measured trajectories, and comparisons. Use that runner's documentation to
interpret its files.
The cases can have different input data and evaluation windows; running them
together does not by itself create a matched comparison. Align the home,
mobility realization, weather and load sources, tariff, and evaluation window
before comparing cases.

`--validate-only` checks suite structure, selected input paths, and the
translated runner arguments. It also calls a runner's `_validated_args` check
where available. It does not fit forecast models, acquire a solver license, or
prove source-data coverage and complete simulation success. Those checks
remain in the underlying runner at runtime; make a short real run before a
long study with new inputs.

## Existing named experiment workflow

The earlier `--experiment-config` workflow selects variants from one experiment
JSON and uses the existing OCHRE simulation loop and controllers. It remains
available independently of `--controller-config`.

### Run an experiment

From the repository root:

```sh
.venv/bin/python examples/single_home_hems/run_experiment.py --validate-only

.venv/bin/python examples/single_home_hems/run_experiment.py \
  --experiment-config examples/single_home_hems/config/experiment.json \
  --variants hvac_milp all_independent \
  --homes home_01 \
  --output-directory /tmp/ochre-my-experiment
```

Use a new or empty output directory. Omit `--variants` to run every variant
explicitly listed in the experiment; omit `--homes` to run all configured homes.
The CLI selects the experiment and outputs. Controller settings belong in JSON.
`--validate-only` checks configuration, controller compatibility, and model
artifacts without creating outputs or simulating. It does not exercise solver
licenses, fit every runtime forecast, or replace a simulation check.

The supplied example has four variants:

| Variant | Predictive controllers |
| --- | --- |
| `hvac_milp` | HVAC |
| `water_heater_scenario` | Scenario water heater |
| `all_independent` | Separate HVAC, water-heater, and EV controllers |
| `joint_thermal_plus_ev` | Existing joint HVAC–water-heater controller, plus separate EV MPC |

The joint thermal variant uses the existing eight-hour horizon, fifteen-minute
interval, and Gurobi formulation. It does not jointly optimize all three devices.
Independent variants retain each selected controller's supported intervals and
horizons. All variants use one-minute plant steps.

The joint thermal controller also accepts a named HPWH choice:

```json
{
  "controls": {
    "joint": {
      "controller": "mpc",
      "hpwh_variant": "rolling-starts"
    }
  }
}
```

`persistence` remains the default and preserves the existing HPWH weights
(comfort 0.1, legionella 0.001, persistence 1.0). `rolling-starts` selects
comfort 100, legionella 0.001, persistence 0.1, a soft four-start rolling
24-hour limit with slack weight 100, and terminal cycle-reserve weight 0.1.
Its terminal target retains at least one unused start allowance. Explicit
`hpwh_*_weight` settings override the selected preset. Forecast settings,
scenario count, HVAC, EV and tariffs retain their independently resolved
values. The rolling choice requires a complete 24-hour native prelude for
observed starts, compared with the existing 16-hour thermal prelude.
Its path-specific rolling counts and endpoint reserves are saved under
`diagnostics/joint_thermal/water_heater_rolling_cycle_diagnostics.csv` for each
case, alongside the joint objective components and solve history.

The example specifies 24 hours of continuous EV/controller warmup followed by
24 hours of evaluation and no cooldown. It uses the existing reference household,
Audi mobility input, DC-house water replay anchored at November 6, 2025, and
DC-house rest-load replay anchored at September 20, 2026. These are explicit
example choices in the new `experiment_home.json`; existing household input
files were not replaced.

## Configuration and precedence

`experiment.json` references `experiment_home.json` through `home_config`.
The household file specifies building physics, equipment, initial conditions,
weather, schedules, and mobility. The experiment specifies controller selections,
shared settings, named variants, and optional simulation-window overrides.

Settings resolve in this order, with later settings overriding earlier ones:

1. Existing controller defaults.
2. Experiment `defaults`.
3. The selected variant's `controls`.
4. Home device and vehicle overrides from experiment `homes` and variant `homes`.

Within each level, device settings override `shared` settings. Variant home
overrides take precedence over experiment home overrides; individual `evs`
settings override `ev_defaults`. For example:

```json
{
  "home_config": "experiment_home.json",
  "defaults": {
    "ev": {
      "controller": "baseline",
      "solver_time_limit_seconds": 300
    }
  },
  "variants": {
    "ev_short_budget": {
      "controls": {
        "ev": {"controller": "mpc", "solver_time_limit_seconds": 60}
      },
      "homes": {
        "home_01": {
          "evs": {"01A": {"solver_time_limit_seconds": 30}}
        }
      },
      "options": {"ev_only_comparison": true}
    }
  }
}
```

Here EV `01A` receives a 30-second solver budget. Thermal solver options retain
their current interfaces and semantics. Unsupported settings are errors rather
than silent substitutions. The runner rejects joint thermal control combined
with an independent controller for either of those same devices. It also rejects
incompatible model timesteps, forecast modes, and solver selections. Parameter
sweeps require explicitly listed variants; the wrapper does not generate them.

Experiment `home_config`, `output_directory`, and forecast/model paths are
resolved relative to the experiment file where applicable. Physical household
paths retain the existing loader's repository/config-directory rules; water
replay paths first check the household file's directory, then the repository.
The saved resolved JSON contains absolute resolved paths and effective settings.

## Baselines and initialization

Each variant runs a fresh matched whole-home baseline and a controlled household.
The primary baseline disables every predictive controller: HVAC follows its
scheduled thermostat, the water heater uses native control, and EVs immediately
charge to `comfort_soc_min` with V2H disabled. Thermal controller objects remain
available as initialization context so disabling their actions does not remove
the existing matched thermal preconditioning or thermostat schedule.

All selected variants share one frozen actual mobility realization per home,
with sufficient history and forecast margin for every selected variant. Changing
the EV forecast method or horizon does not resample the trips. Baselines are
paired within each variant; a baseline is not cached across variants with
potentially different thermal initialization requirements.

`options.ev_only_comparison: true` adds a separate EV reference. It retains the
variant's thermal control policies and uses native EV charging. Thermal actions
may respond to the resulting household trajectory, so this comparison holds
policies, not an identical sequence of thermal actions, fixed. This reference
does not replace the primary whole-home baseline.

The internal `non_v2h_twin` is a different run: it supplies the existing backed-load
forecast needed by V2H EV MPC. It uses EV predictive control with V2H disabled.
It is retained when required even if the optional EV reference is off, and it is
excluded from primary comparison figures and savings tables.

Controller-active warmup runs continuously into evaluation. SOC, SOH, stored
energy, and aging/controller memory are not reset at the evaluation boundary.
Both accounting windows are saved:

- `including_warmup`: configured simulation start through evaluation end.
- `excluding_warmup`: evaluation start through evaluation end.

Both exclude cooldown and the internal twin's extra forecast tail. Existing
thermal preconditioning before the configured simulation start remains separate
from this EV/controller warmup. Inventory-adjusted EV savings use actual boundary
energy and the approved daily-average tariff valuations in both windows.

## Results

```text
output/
  requested_experiment.json
  resolved_experiment.json
  inputs/<home>/EV_<id>_trips.csv
  comparison_metrics.csv             # primary whole-home/EV accounting, both windows
  all_case_metrics.csv               # includes clearly labeled reference/internal roles
  solver_summary.csv
  shared_plot_limits.json
  run_status.json
  <variant>/
    resolved_configuration.json
    <home>/
      cases/whole_home_baseline/
      cases/controlled/
      internal/non_v2h_twin/          # when required for the EV forecast
      references/ev_only_reference/  # optional
      whole_home/
      hvac/                          # when selected
      water_heater/                  # when selected, including joint control
      ev_<id>/                       # when selected
      ev_only_comparison/            # optional comparison and paired economics
```

Each selected-device folder compares the device's trajectories from the same
baseline and controlled whole-home runs. They are not isolated-device reruns.
Comparison folders contain trajectory CSVs, metrics, and figures. Thermal plots
include temperatures, setpoints/shifts, electrical power, and available operating
states. EV plots include available SOC, SOH, capacity, and power telemetry.
`requested_commands.csv` records issued commands separately from observed plant
values and optimizer predictions.

Trajectory plots and the per-device comparison-folder metrics cover evaluation.
The root `comparison_metrics.csv` covers both accounting windows and includes
whole-home and EV economics. The optional EV reference's
`economics_both_windows.csv` recomputes savings against that reference, including
battery-inventory adjustments. Device electricity costs are tariff-weighted
attributions, not separate household bills.

Comparable figures share axis limits across the selected variants. Water-heater
baseline and controlled panels both use electrical kW. The same correction and
combined/joint water-heater baseline figure support also apply to the existing
individual simulation reporting.

Each case saves raw data, evaluation masks, controller solve histories, available
objective components, and optimizer plans. Solver summaries count failures and
fallbacks where the existing controller exposes them; an empty fallback field
means that controller does not expose a fallback counter. The wrapper preserves
existing solver/fallback policies. Exceptions stop the experiment and record a
failed run in `run_status.json`; partial outputs remain available for diagnosis.

## Verification scope

The implementation passed 349 tests: 194 EV/central-wrapper tests and 155 existing
household/thermal regression tests. A short real OCHRE run completed
all four example variants with actual Gurobi/SCIPY optimization and no recorded
solver failures, including the optional EV reference. This bounded check used
15 minutes of warmup and 15 minutes of evaluation; it was not a full-day savings
study. The unchanged joint thermal horizon remained eight hours in that check.

# Module map and extension points

[Guide index](../README.md)

Paths below are relative to examples/single_home_hems unless otherwise stated.

| Responsibility | Source |
| --- | --- |
| Unified CLI / study CLI | run_simulation.py / run_ev_study.py |
| Strict loading, precedence, phase windows | single_home_hems/control_config.py |
| Three-case orchestration and reports | single_home_hems/unified.py |
| Native dwelling timestep loop | single_home_hems/simulation.py |
| Forecast routing / external snapshots | single_home_hems/forecast.py / control_forecasts.py |
| Historical EV scenario selection | single_home_hems/ev_scenarios.py |
| EV single-path formulation, controllers, projection | single_home_hems/controllers/ev.py |
| EV scenario assembly and joint rainflow certification | single_home_hems/controllers/ev_scenario.py |
| HPWH formulations / scenario assembly | single_home_hems/controllers/water_heater.py / water_heater_scenario.py |
| Quiet backend solve and feasibility checks | single_home_hems/controllers/solver.py |
| Backed-load definition and actual-load cap | single_home_hems/backed_loads.py |
| Realized EV metrics | single_home_hems/ev_metrics.py |
| Unified / study plots | single_home_hems/unified_plots.py / ev_study_plots.py |
| Frozen paired studies and checkpoints | single_home_hems/ev_study.py |
| Legacy caller adaptation | single_home_hems/legacy_adapter.py |

Shared physical EV code remains in examples/five_home_block/five_home_block:
ev_config.py validates internal options, ev_schedule.py adapts mobility,
ev_equipment.py owns the plant/SOH state, ev_degradation.py counts/prices damage,
and home_factory.py builds native equipment. These are active shared dependencies,
not legacy files to delete. Native ochre/ remains the physical simulator.

## Safe changes

1. **New preference:** add validation/defaults to EVOptions/control loading as
   appropriate; document both templates, resolution precedence and units.
2. **New forecast:** implement the existing provider interface; preserve issuance
   causality, complete coverage and explicit trip distances. Add leakage/grid tests.
3. **New controller:** implement DeviceController, register construction in unified,
   preserve fixed solve timing and baseline fallback. Do not create another plant loop.
4. **New cost:** obtain specific formulation approval first; preserve units,
   normalization, per-scenario aggregation and disabled-term behavior. Reference
   plant aging remains independent of the controller's approximate cost.
5. **New metric:** compute from realized evaluation-window state/ledger, not plans.
   Preserve baseline-relative economics and separate twin/advanced groups.
6. **New study variant:** append a new ID without modifying frozen existing inputs.
   Source edits, including cleanup/docstrings, change recorded source fingerprints:
   use a new study revision or the original checkout for an existing study.

Prefer focused tests in test/test_ev_*.py and test/test_unified_*.py.
Retain native and HPWH regression checks when touching shared interfaces.
Do not change test expectations to conceal a formulation discrepancy.

The obsolete ev_mpc.py, ev_formulation.py and ev_simulation.py modules have been
removed. Import controllers from controllers.ev and metrics from ev_metrics.
The old EV overlay runner/configs are archived; use unified home/control JSON.
Existing HPWH CLI callers still use legacy_adapter; new features belong in the
unified pipeline.

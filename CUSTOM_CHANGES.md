# Changes from base OCHRE

## 2026-10-17: Repository cleanup and documentation consolidation

- Current user guides are indexed by examples/single_home_hems/README.md and
  live in its docs/ directory; root README now explains the unified workflow.
- Superseded guides, EV overlay runner/configs and the frozen-study replay tool
  are preserved in archive/development_docs_2026-10-17. The older legacy archive
  is unchanged. Historical entries below retain their original dated claims.
- EV metrics now live in ev_metrics.py. Removed ev_formulation.py, ev_mpc.py,
  ev_simulation.py, unused summary objects and the misleading EVPredictiveController
  alias; callers use controllers.ev, EVBaselineController and ev_metrics.
- Removed confirmed unused imports; added concise interface docstrings.
  No objective, physics, active configuration values or test expectations changed.
- Generated caches are ignored; orphan tmpx5j91pta outputs moved to ignored
  .test-tmp/cleanup_recovery_2026-10-17 for recovery. Datasets/results are retained.
- Source hashes change even for cleanup; frozen-study checks are not bypassed.
  Validation and known limitations: [cleanup report](examples/single_home_hems/docs/validation/cleanup_2026-10-17.md).

## 2026-10-17: Scenario-based EV MPC — finalized

- Added causal per-EV historical selection: default five kNN paths, 30-day pool,
  24-hour matching history; all existing prediction-analysis filter alternatives.
  Greedy non-overlapping completed trajectories, equal weights, logged fallback
  if none. Reserve full pool/history before seeded Audi replay selection.
- Added configurable four-control-interval home/away persistence prefix;
  overwritten arrival distance is delayed/compounded at the next later arrival,
  or recorded as deferred beyond forecast coverage. No future-realization override.
- Added two-stage scenario wrapper reusing existing EV formulation: maximum
  whole-path departure comfort, mean remaining costs; shared first interval,
  scenario recourse afterward. Preserves all EV options, frozen measured capacity,
  actual-load cap, solve clock and plant aging; HPWH/native physics untouched.
- Added joint multicut rainflow with per-path dollar-accuracy checks and shared
  time budget. Extended configuration/templates, forecast/plan/solver diagnostics,
  unified provider wiring, and documentation (EV_SCENARIO.md).
- Existing perfect forecast mode and user control/physical tuning retained.
  Existing minimum-price terminal value explicitly reconfirmed by user; older
  mean-price test discrepancy documented, not hidden or changed to force a pass.
- Final scenario tests: 21 passed. Full discovery: 410 tests, 126 previously
  documented legacy errors and 8 old mean-terminal-price assertion failures.
  Five-path/12-hour MILP solved with SCIPY; installed restricted Gurobi exceeds
  license size (2,165 variables), although small backend/aging fixtures passed.
  Details: examples/single_home_hems/EV_SCENARIO_VALIDATION.md.
- User approved finalization October 17 with the documented test/license limitations.
  Finalization updated records only. No full study, commit, archive edits or data deletions.

## 2026-10-16: EV daily-tariff terminal energy valuation

- User-directed replacement of the moving SOC-floor penalty with
  `v*(e0-eN)/N` for convex and MILP EV MPC. Both energies use measured faded
  capacity frozen within the horizon; energy surplus receives a credit.
- `v` directly equals the full solve-calendar-day known tariff time average
  (minute intervals), valued in USD per chemical kWh, with no efficiency
  conversion. Current TOU gives $0.16875. No extended mobility/load forecasts.
- Kept `enforce_terminal_soc` switch; retired/ignored `terminal_soc_weight`
  with an explicit warning for active nonzero legacy values. Removed the old
  100000 multiplier from active EV config and the retired field from templates.
- Added tariff metadata to the shared forecast container without changing
  HPWH behavior; external device forecasts retain the known tariff separately.
- Saved plans expose v, initial/terminal chemical energy and terminal cost.
  Reporting metrics, physical battery settings and study inputs/results unchanged.
- Validation: 138 EV/unified regression tests passed in 299.708 seconds,
  including terminal-value checks with CLARABEL, SCIPY and GUROBI, none/throughput/
  rainflow costs, measured-capacity refresh, credits and normalized units.
  HPWH controller SHA256 remains DFDAE01C6E79B150BB613A22468E61A7B55D2943598AAE8773E8353C9C2EE927.
  No full repository suite or parametric study was launched.
- Finalized with explicit user approval on October 16. Finalization updated
  records only; no additional code changes, simulations or Git commit.

## 2026-10-16: Dynamic EV SOH and incremental damage accounting

Finalized with explicit user approval on October 16. Finalization updated records
only; no additional simulations, code changes or Git commit.
HPWH control code, settings and equations are untouched.

- Added required unified physical `initial_soh` (.85 in supplied home/templates).
  Beginning-of-life capacity remains the pack-valuation basis; current usable
  capacity is nominal capacity times measured SOH.
- Added streaming reference/depth/throughput damage ledgers, retaining open
  residue. Every actual charge, V2H action and served trip ages the plant,
  regardless of baseline/fallback or objective degradation settings.
- Calibration-normalized D maps to SOH via 1-(1-SOH_cal)*D. Economic budget
  (1-SOH_econ)/(1-SOH_cal) prices new damage only; initial damage is not rebilled.
  Existing 80% calibration, 70% economics and $119/$0 assumptions are retained.
- SOC-preserving fade records inaccessible-energy loss separately. OCHRE model
  evaluations preview damage; plant steps commit once. EOL is a flag, not a
  forced replacement. Physical exhaustion and unsupported stress settings fail explicitly.
- MPC uses fresh plant SOH/current capacity, frozen within each horizon. Original
  pack value stays fixed; throughput pricing and historical discharge budgets
  use the corresponding current/action-time capacity. Solve clock unchanged.
- Metrics sum scored continuous-ledger increments, excluding warmup/cooldown
  without losing spanning-cycle history. Added SOH/capacity/damage/energy outputs,
  health plots, plan diagnostics and battery-only checkpoints (not whole-run resume).
- Added test_ev_soh and updated EV conservation/measurement/warmup regressions.
  Old saved studies remain unchanged. New study destination is ev30_dynamic_soh_v4;
  full study is not run. Validation:142 focused tests pass; two Gurobi driver
  jobs/six cases have zero failed solves and resume without rerunning. Full
  discovery still encounters126 legacy/native API/import errors. See
  examples/single_home_hems/EV_SOH_VALIDATION.md for details.


## 2026-10-15: EV departure comfort, mode persistence, economics and diagnostics

Finalized with explicit user approval on October 15. Finalization updated records
only; no additional simulations, HPWH edits, or Git commits.

- EV ONLY: no HPWH controller, objective, settings, or native OCHRE changes.
  Shared runner now supplies mobility to EV reporting; HPWH tests were run only
  as compatibility checks.
- EV comfort is mean positive pre-departure SOC shortfall over departures in
  [solve time, horizon end); zero when none. Health remains a horizon-wide soft
  band. Hard SOC bounds, fixed solve clock and baseline behavior are unchanged.
- MILP persistence uses charge/discharge/idle one-hot L1 changes (2 per switch);
  convex retains chemical-power variation. Existing dt/N scaling and weights
  remain; changing backed load within discharge mode is not a mode switch.
- New/residual battery assumptions: $119/$0 per kWh. Added explicit required
  calibration_end_of_life_soh=0.8, separate from economic end_of_life_soh=0.7.
  Physical fade remains 20% per 3000 reference cycles. All monetary methods
  use the consistent extrapolated 4500-reference-cycle/30%-fade budget.
  This is constant-slope extrapolation, not a validated aging-knee model.
- Added scored departure-shortfall metrics and both SOH endpoints to EV outputs;
  retained whole-window comfort diagnostics. Updated active JSON/templates/docs.
  Next full study defaults to a new directory, ev30_departure_nmc_v3; NOT run.
- Preserved all 30 completed original study jobs. Saved pre-change EV/solver
  snapshots, Gurobi diagnostic logs, real-plan rainflow cuts and analysis under
  examples/single_home_hems/outputs/ev_diagnostic_20261015. Diagnostic replay
  explicitly migrates frozen schema in memory; normal loading remains strict.
- Validation: 126 focused EV/unified/study/HPWH regression tests plus one added
  departure-preparation test passed (127 total). Two short real-OCHRE driver
  jobs (GUROBI convex/MILP; six case simulations) passed, zero scored failures;
  warm-up/cooldown excluded. Results: outputs/ev_departure_eol_validation_20261015.
  Unrelated legacy test_parametric_study cannot import its missing analysis
  module; no legacy fix attempted. Windows temporary-folder permission errors
  were avoided using a workspace-local test temporary directory.
- Gurobi warning reproduced as a 300-second timeout, 15.5256% native gap and
  baseline fallback, not demonstrated numerical instability. No solver gap
  policy changed. $119/kWh implemented as requested; exact IEA attribution
  remains unverified. Full details are in the diagnostic report.

## 2026-10-15: Paired 30-job study, cooldown, plots and rainflow accuracy

- Staged only: Jan 3-8 and Feb 3-8, 2018; Audi seeds 41/42/43; initial SOC 0.4;
  five cost formulations (one convex, four MILP), GUROBI, perfect forecasts,
  15-minute control and 12-hour horizon. Comfort/active SOC-health weights 1000.
  HVAC/HPWH baseline, V2H and physical battery/load settings unchanged.
- Added explicit pair_mobility_by_seed: freeze one trace per seed and shift its
  entire relative timeline across months, accepting weekday mismatch. Existing
  archive/output/staging directories were not overwritten.
- Added configurable controller-active cooldown_hours (default 24, zero disables)
  to regular and parametric runs, in addition to 24-hour warm-up. No forced
  recharge, terminal constraint or state reset. Only the original evaluation
  window contributes to costs, aging, shortfalls, starts and solver metrics.
  All-phase raw/control outputs and phase masks retain pre/post-roll and twin tail.
- Summaries explicitly group by configuration, case and device: advanced and
  non-V2H are never pooled. Added numeric std/min/max/count reporting, solver
  diagnostics, equivalent full cycles, energy change and fallback exposure.
- Added separate advanced/non-V2H cross-formulation trajectory overlays and
  paired savings plots with individual cases/sample SD; original plots retained.
- User explicitly approved independent rainflow envelope accuracy at the retained
  incumbent: error <= $1e-6 + 0.001*incremental aging dollars, in addition to the
  unchanged overall 5% MILP gap. No arbitrary minimum cut count. Added regression
  with a large unavoidable objective offset that previously hid omitted aging.
- Validation: 110 focused tests passed across EV degradation/formulation/plant,
  unified controls/pipeline, study driver/plots, and warm-up/cooldown accounting.
  Small integration tests only; no pilot job or full study was executed.
  Initial temporary-folder permission errors were avoided using workspace temp
  directories. Tight analytic test uses tighter CLARABEL backend precision;
  production/study backend defaults are unchanged.
- Staging: outputs/ev30_janfeb_nmc_20261015, 6 paired cases / 30 jobs. Actual
  January/February trip offsets/distances and forecast coverage checked. This
  supersedes the older staged pilot; execution still requires explicit --run.

## 2026-10-14: NMC stress calibration and residual-value economics

- Per explicit approval, retained additive fade accumulation (no nonlinear SEI
  transformation). Replaced LMO/B2 depth stress with Mathews NMC power stress:
  exponent 2.03, reference depth 0.8, mean SOC 0.6, 3000 reference cycles, EOL 0.8.
- Reference SOC stress is exp(k*(mean-reference)), not an absolute deviation;
  coefficient 1.04 retained as a modeling assumption. Depth-only OCP rainflow and
  linear throughput share the same reference life. No solver/constraint changes.
- Gross new cost 209 and user-assumed residual 171 $/kWh produce depreciable value
  38 $/kWh for every aging cost. Gross/residual/depreciable totals are reported.
- Migrated active home JSON/template, strict physical schema, report labels and
  documentation; obsolete B2 keys are rejected. Prior outputs, archives and staged
  study snapshots remain untouched. New studies require a new staging directory.
- Updated targeted test expectations for the approved math and residual costs.
  Tests/simulations were NOT run, as explicitly requested by the user.

## 2026-10-14: EV study, net savings and controller-active warm-up (finalized)

- Added `run_ev_study.py`, `single_home_hems/ev_study.py`, documented pilot JSON
  and EV_STUDY.md. Explicit stage/run/smoke modes, frozen Audi schedules, input
  and environment fingerprints, append-only configuration IDs, verified resume,
  preserved failed attempts, and common-case aggregate results.
- Current staged pilot: 4 Audi/home-load combinations x 3 initial SOCs (20/50/80%)
  x 4 MILP objectives = 48 jobs, with 24-hour warm-up and 48-hour evaluation.
  Full pilot was NOT run. Smoothed/hh are deferred by the user's latest instruction
  and removed from this pilot; the earlier 72-job staged snapshot is preserved.
- Replaced old electricity-only savings/profit output with operating-cost savings,
  B2 degradation-cost savings, and net savings, all relative to all-baseline.
  Added initial SOC/final stored energy and p95 solve time diagnostics. Old saved
  outputs and prior archives are untouched; no terminal-energy credit added.
- Added explicit `none` and `rainflow_multicut` alias without changing existing
  rainflow equations, fixed solve clock, gap acceptance, physical guards or HPWH.
- Unified runner optionally consumes study-frozen mobility. Home/study configs now
  accept warmup_hours (default 24; 0 disables). Start/duration describe evaluation,
  configured SOC applies at warm-up start. Plant and controller history persist;
  no extra boundary OCP solve, state reset, or formulation change.
- All evaluated energy, aging, comfort, cycle and solver metrics exclude warm-up
  and twin tail. Boundary SOC and continuing cycles are handled explicitly.
  Detailed histories/native outputs retain pre-roll, with evaluation masks and
  resolved time-window metadata. Water replay retains its evaluation-date mapping
  while forecast training remains anchored before the actual simulation start.
- Two targeted native schedule guards fix already-covered cross-year windows and
  preserve normalized timezone when annual data is repeated. No native physics changes.
  Legacy entry points retain zero warm-up for compatibility.
- Latest driver validation: two Audi/MILP configurations, 15-minute warm-up plus
  30-minute evaluation, all six subsimulations; 4 warm-up + 8 scored + 4 tail OCPs,
  zero failed solves. Resume performs no additional runs. Results checked for
  evaluation slicing, boundary SOC, safety, paired baselines, figures and economics.
- Corrected new-driver Windows path-length failure with compact hashed run folders;
  original failed validation attempt retained. Successful validation outputs are in
  examples/single_home_hems/outputs/ev_driver_validation. Latest warm-up validation:
  examples/single_home_hems/outputs/ev_warmup_validation. 105 focused tests passed.
  Current full pilot: outputs/audi_milp_degradation_pilot_warmup (staged only).
- Final full discovery:365 tests,0 assertion failures,126 pre-existing legacy
  errors,0 skips. Same error count/categories as before; full suite remains
  non-green. See UNIFIED_VALIDATION.md. Finalized with user approval on 2026-10-14;
  no additional simulations, Git staging or commits during finalization.


This repository is a work fork of the Object-oriented Controllable
High-resolution Residential Energy Model (OCHRE). The custom work starts after
base commit `458b9b2`. That commit is the most useful comparison point for the
work in this fork; the fork has not yet been rebased onto newer OCHRE releases.

## HVAC model extensions

The main OCHRE engine changes are in `ochre/Equipment/HVAC.py`.

- Added an optional discrete reverse-cycle defrost model while retaining the
  original time-averaged behavior as the default `Legacy` model.
- Added timer-based and demand-based discrete defrost controls, including
  configurable duration, outdoor-temperature threshold, initial accumulation,
  heat-extraction multiplier, and compressor-power multiplier.
- Added aggressive and conservative electric-resistance backup strategies
  during discrete defrost.
- Added automatic or explicitly configured 5/10 kW resistance-backup stages.
- Added thermostat-based stage escalation and release behavior.
- Added compressor minimum-on and minimum-off timing that remains independent
  of resistance-backup dispatch.
- Added detailed defrost, compressor, resistance-capacity, and stage outputs at
  higher verbosity levels.

The new configuration fields and model behavior are described in
`docs/source/InputsAndArguments.rst` and `docs/source/ModelingApproach.rst`.

## System identification and predictive control

The notebooks under `bin/` contain the earlier exploration and control work:

- dwelling and RC/HVAC model exploration;
- HVAC system identification, including Hankel and ARMAX experiments;
- uncontrolled baseline simulation;
- model predictive control (MPC);
- time-of-use MPC; and
- data-enabled predictive control (DeePC).

The control notebooks use `cvxpy`, which is not required by the base OCHRE
package and must be installed separately.

## Five-home neighborhood example

`examples/five_home_block/` adds a reproducible five-home simulation using five
distinct Indiana ResStock HPXML/schedule inputs and shared weather.

- Runs all homes on a shared one-minute clock.
- Standardizes version-specific OCHRE output names.
- Checks individual-home and neighborhood power balances.
- Aggregates electric load and produces per-home and neighborhood figures.
- Includes a deliberately simple static transformer-capacity model.
- Includes a seasonal electrification parametric study that caches individual
  home profiles and reconstructs valid neighborhood scenarios.

Only the active HPXML files, schedules, and weather file are included. Generated
simulation outputs are intentionally ignored.

## Single-home HEMS

`examples/single_home_hems/` provides a stepwise control interface for any home
from the five-home cohort. It includes water-heater MPC and opt-in EV MPC/V2H;
`hems.py` composes device controllers. Native controls remain the default.

Legacy EV configurations remain supported. New EV configurations use continuous
SOC and distance-based NHTS or Audi mobility; see the dated implementation log below.

## 2026-10-06: EV smart charging and V2H finalized

User approved implementation finalization with the documented legacy-test and
optional-GUROBI limitations. This entry records working-folder changes, not a
Git commit or upstream release.

### Added

- `examples/five_home_block/five_home_block/ev_config.py`: independent per-EV
  physical, health, solver, objective, cycling, projection, and mobility settings;
  validation permits at most one V2H-capable EV per home.
- `ev_schedule.py` in the same package: common trip/availability format, existing
  NHTS cleaning/sampling with annual wrapping, and seeded interior Audi windows
  with history/forecast coverage. Audi accepts both supplied cohorts and both
  file formats; arrival distance determines demand using configured consumption.
  Signed SOC fields remain diagnostics, not driving-energy inputs.
- `ev_equipment.py` in the same package: opt-in bidirectional OCHRE equipment,
  efficiency-aware SOC, load/inverter discharge limits, exact return-trip energy
  accounting, unserved-trip reporting, and immediate-charge baseline.
- `examples/single_home_hems/single_home_hems/controllers/ev_mpc.py`: MILP and
  convex OCPs with SCIPY, CLARABEL (convex only), and optional GUROBI; soft health
  and departure-deficit costs, hard physical SOC bounds, persistence/dwell/start
  caps, discrete projection, and approved convex trip-shortfall relaxation.
- `examples/single_home_hems/single_home_hems/ev_simulation.py`: matched baseline
  comparison, advance HVAC/WH load forecast with equality checks, full backed-load
  allocation to the single V2H EV, timing/energy/cost/peak/SOC diagnostics, and plots.
- `examples/single_home_hems/run_ev_mpc.py` and Audi/convex and NHTS/MILP JSON
  overlays under `config/`: opt-in execution without modifying source home files.
- `test/test_ev_v2h.py` and `test/test_ev_v2h_integration.py`: 29 tests covering
  adapters, formulations, physical accounting, independent EV settings, projection,
  failure handling, real OCHRE, EV+HPWH, and five-home integration.
- [Archived EV/V2H guide](archive/development_docs_2026-10-17/EV_V2H.md) and
  [validation report](examples/single_home_hems/docs/validation/ev_v2h.md).

### Integrated and preserved

- Shared configuration, dwelling factory, neighborhood dispatch, single-home
  forecast container, simulation loop, and controller exports support opt-in EVs.
- Default HEMS/general-runner behavior is native immediate charging without V2H
  or optimization. The old TOU controller remains an explicit selection.
- Reporting includes mean OCP solve time per invocation and solver time amortized
  over every plant timestep, including failed attempts.
- Native `ochre/` code and HPWH controller mathematics were not changed.
- Pre-existing user changes to home/neighborhood configurations and EV dataset
  additions/replacement were preserved; they are not implementation-log changes.

### Validation and limitations

- Full discovery: 284 tests, 158 passed, 126 legacy API/fixture errors, zero
  assertion failures. All 29 new EV tests passed; the full suite is not green.
- Audi/CLARABEL and NHTS/SCIPY 72-hour runs completed with zero solver failures;
  cumulative battery energy-balance error was below 1.3e-13 kWh.
- GUROBI execution remains unverified pending local installation/license.
- Convex relaxation/projection limitations are documented. Different final SOCs
  prevent interpreting raw cost differences as equal-terminal-energy savings.
- Existing legacy parametric-study incompatibilities remain outside this scope.
- Finalization updates documentation only; no formulation changes, staging, or
  commits were performed at this checkpoint.

## Tests

### 2026-10-07: cycling, standalone configuration and consolidated reporting

- Implemented approved independent EV cap/dwell switches and persistence;
  MILP constraints coexist, convex remains continuous with actuator-only caps/dwell.
- Added HPWH post-solve cap/dwell guard through the native binary duty interface,
  configuring its external-control timestep without changing native OCHRE source.
  Retained existing HPWH cycle-budget semantics and scenario mean/max aggregation.
- Added independent HPWH electricity weight (default 1); retained existing term
  normalization and separate device/EV weights, including user-tuned values.
- Shared quiet solver wrapper enforces 300-second native backend limits for all
  EV/HPWH OCP calls; validates feasible/integral incumbents and records limits,
  fallbacks, and timing. GUROBI runtime remains unverified (not installed).
- Unified home JSON is self-contained; missing required fields identify their
  configuration path. Native OCHRE, home_factory, shared config/equipment/schedules,
  HPXML, datasets and models remain dependencies, not legacy config fallbacks.
- Added valid JSON home/control templates with `_comments` descriptions, units
  and feasible values; explicit inactive fields are included. Current home_01
  tuning preserved; additional homes must be explicitly configured.
- Added EV operating-state plots/CSV, realized cycle/start and comfort metrics,
  a single `reports/run_report.md`, and retained machine-readable diagnostic files.
  Canonical runs reject nonempty output directories; resolved inputs/seeds remain saved.
- Added 12 focused tests. Full discovery: 311 tests, 185 passes, 126 pre-existing
  legacy errors, zero assertion failures; permitted rerun removed sandbox-only
  temporary-directory errors. See the current unified validation report.
- Updated guides and WORKFLOW_CHECKPOINT.md. No research simulations, source
  deletions, staging or commits. This revision awaits finalization approval.

### 2026-10-07: unified HEMS revision finalized

- Added separate physical/mobility home overlays and per-home/per-EV control
  configuration, example JSON files, and a canonical `run_simulation.py` mode.
- Consolidated EV variants in `controllers/ev.py`; removed TOU avoidance and
  retained import/runner compatibility adapters to the shared simulation loop.
- Replaced the separate two-case result writers with three explicit retained
  cases: all-baseline, identical-controller no-V2H twin, and advanced control.
- Switched EV solves to the fixed control clock. Mobility updates accounting
  without new solves; available prediction blocks within an interval share a command.
- Added baseline-twin component forecasts and same-step actual-load discharge caps;
  default space-heat-pump demand excludes resistance backup and includes the fan.
- Preserved HPWH objectives; added configured fixed-clock initialization/reporting,
  causal external deterministic forecasts, and explicit EV-scenario/HVAC-MPC
  placeholders/rejections rather than undocumented fallback behavior.
- Added case-local and three-case plots, common solver CSV/Markdown reports,
  resolved configuration capture, and retained thermal diagnostics in combined runs.
- Added 15 unified tests. Full discovery: 173 passes, 126 existing legacy errors,
  zero assertion failures. No research comparison studies were launched.
- Added [unified guide (archived)](archive/development_docs_2026-10-17/UNIFIED_PIPELINE.md) and
  [validation checkpoint](examples/single_home_hems/docs/validation/unified.md).
- User approved finalization; nothing staged or committed.
- Finished the [legacy archive](archive/legacy_hems_2026-10-07/README.md):
  55 committed source/config/document/model files verified against the export
  and source commit, plus the separately captured pre-unification EV controller.
  SHA-256 manifest records provenance and Windows line-ending normalization.
  This is not a complete snapshot of intervening uncommitted legacy versions;
  no active legacy files were deleted or replaced.
- The next cycling/configuration/reporting revision is in design; its OCP
  formulation changes require specific approval before implementation.

The added tests cover:

- timestep-independent discrete-defrost state behavior;
- electric-resistance stage configuration and dispatch;
- five-home configuration, output mapping, aggregation, and plotting;
- seasonal parametric-study scenario generation and reconstruction; and
- single-home HEMS selection and control validation.

Run the focused tests from the repository root with:

```bash
.venv/bin/python -m pytest \
  test/test_equipment/test_defrost.py \
  test/test_equipment/test_staged_er.py \
  test/test_five_home_block.py \
  test/test_parametric_study.py \
  test/test_single_home_hems.py
```

## 2026-10-12: EV mathematical formulation revision finalized

- Added `controllers/ev_formulation.py`: explicit horizon data, signed chemical
  power, affine SOC dynamics, optional budgets and normalized objective terms.
  Convex has one power decision; MILP retains exclusive charge/discharge modes.
- Baseline tracks configurable target-min SOC without overshoot. Replaced health
  penalties with soft target-band costs; removed OCP trip-shortfall relaxation and
  cost. Full forecast demand can cause infeasibility and logged baseline fallback.
  Plant saturation/unserved-energy diagnostics preserve physical [0,1] SOC.
- EV-only dt/N normalization, chemical-power convex persistence, optional soft
  terminal SOC, historical-ends/planned-starts cap and matching battery-throughput
  discharge budget. Removed all EV projector cap/dwell enforcement; MILP OCP keeps
  enabled cap/dwell constraints. HPWH formulation, normalization and projection unchanged.
- EV MILP checks feasible/integral incumbents and configurable 5% relative backend
  gap; nonconverged missing/nonfinite gaps fall back. Fixed solve clock retained.
  Negative lossy bidirectional convex tariffs error explicitly; affine charge-only
  and lossless cases accept them. No silent clipping/backend changes.
- Added max solve/OCP times and target/saturation metrics to the single run report.
  Migrated unified JSONs and descriptive templates; existing tuning retained.
  Prior Python aliases remain supported, not legacy-file fallback.
- Added 17 formulation/acceptance/plant tests and revised prior EV assertions to
  the approved semantics. GUROBI convex/MILP and infeasible fallback smoke solves
  passed locally. Final counts are in `UNIFIED_VALIDATION.md`.
- Source LaTeX, native OCHRE, archived files, unrelated inputs and HPWH mathematics
  were not changed. No research runs, staging or commits. User approved finalization
  with the documented 126 pre-existing legacy test errors and zero assertion failures.

## 2026-10-14: EV health/degradation revision — finalized with user approval

- Consolidated EV horizon/formulation/controller code in `controllers/ev.py`,
  with named convex/MILP builders, mathematical notation and ordered constraints/
  cost sections. Old `ev_formulation.py` is an import shim only.
- Moved SOC preferences into control JSON: soft `health_soc_min/max`, one common
  `health_soc_weight`, overall `health_weight`, and independently weighted
  `comfort_soc_min`. Baseline/fallback charges to comfort minimum.
- Added configurable depth-only convex rainflow or battery-side linear-throughput
  OCP degradation. Shared B2-derived reference cycle life, replacement cost and
  calibration apply to both methods and the B2 reference metric; driving included.
  Only the second and third supplied papers are used, not pay-for-performance.
- Rainflow uses exact numerical cycle costs/subgradients, retained observed
  history and supporting-cut master solves. Full-objective upper/lower bounds,
  common 300-second budget and conservative fallback protect both formulations.
  Native master gap alone is insufficient. Added iteration/gap/timing diagnostics.
- All three cases report unweighted estimated and B2 reference degradation.
  Savings is versus all-baseline only. Net profit subtracts TOTAL case reference
  degradation, not its difference from baseline. No twin-relative money metrics.
- Migrated four JSON examples/templates without changing unrelated home/HPWH/
  EV tuning. Added `EV_DEGRADATION.md`; updated unified guide and README pointer.
  New degradation is opt-in; comfort weight defaults to zero pending study tuning.
- Added 21 degradation/calibration/certificate/reporting tests; retained and
  extended the mixed EV/HPWH three-case fixture. Validation and known legacy
  errors are recorded in `examples/single_home_hems/UNIFIED_VALIDATION.md`.
- Native OCHRE, HPWH equations, source papers/LaTeX, mobility inputs and archive
  untouched. No research-duration experiment, staging or commit. User approved
  finalization with the documented 126 legacy errors and zero assertion failures.

## Known limitations

- These extensions have not been accepted or validated by the upstream OCHRE
  maintainers.
- The discrete-defrost and backup-staging models are research-oriented and
  should be checked against the intended equipment assumptions before being
  treated as calibrated equipment models.
- The neighborhood transformer is static; it does not model temperature,
  voltage, losses, aging, or time-dependent overload behavior.
- Notebook outputs reflect exploratory development and are not a formal
  benchmark suite.
- The older repository-wide test suite contains pre-existing errors from outdated
  APIs and fixtures, including the legacy parametric-study import. Current counts
  and verified EV coverage are recorded in the dated entry and validation report.
- Compatibility with OCHRE versions newer than the pinned base has not yet
  been evaluated.

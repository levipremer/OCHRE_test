# HEMS workflow checkpoint — 2026-10-17

## Latest: approved repository cleanup

Canonical guides: examples/single_home_hems/README.md and docs/.
EV formulation/controller imports now use controllers.ev; metrics use ev_metrics.
Obsolete EV shims/overlay configs/runner are removed from the active layout;
historical documentation and overlay sources are archived. Native OCHRE,
home_factory, active HPWH/multi-home adapters and prior studies remain intact.
See docs/validation/cleanup_2026-10-17.md under the single-home example for checks.
Cleanup completed: 152 focused tests and 17 cleanup/HPWH CLI tests passed.
Full discovery: 417 tests, same 126 errors/eight assertions as the prior log
(failure identifiers compared, none added). Three final layout tests passed.
HPWH file hash and EV executable formulation were preserved.
No formulation edits, study runs, commits or pushes are authorized by cleanup.
The earlier dated checkpoint sections below are historical, not current paths.

## Current revision: EV scenario MPC — finalized

User approved finalization on October 17. Finalization updated records only;
no additional code changes, simulations or Git commit. Await the next directive;
do not launch a full study. See examples/single_home_hems/EV_SCENARIO.md and
EV_SCENARIO_VALIDATION.md. Code: ev_scenarios.py (causal selection/prefix),
controllers/ev_scenario.py (two-stage aggregation/multicut), existing ev.py
(dispatch/reporting), control_forecasts/config and unified provider wiring.

- Default five kNN scenarios, 30-day pool, configurable 24-hour history,
  non-overlapping completed paths; alternatives preserve filter hierarchy.
- Four control intervals of current home/away persistence; displaced arrival
  distance carried to next later arrival, or diagnosed as deferred beyond horizon.
- Maximum normalized per-path departure comfort; mean every other weighted cost.
  First interval command shared, future recourse. All physical/control options
  retained; each path uses true current capacity frozen in its horizon.
- Separate rainflow epigraph/cuts/accuracy checks per path; one joint time budget.
- Startup reserves full mobility history, including before warmup; no event solves.
  All three cases and their outputs preserved. HPWH unchanged (hash below).
- 21 new scenario tests passed, including OCHRE integration and all solver/aging
  combinations in small fixtures. Existing focused group: 168 tests with only
  8 known terminal-value assertions failing. Full discovery: 410 tests, the same
  8 assertions and 126 legacy errors. Log: .test-tmp/ev_scenario_full_suite.log.
- IMPORTANT: existing terminal function uses prices.min(), not mean. User
  explicitly chose to preserve the minimum and document old-test disagreement.
  No numerical change made; old tests remain. This supersedes earlier wording.
- IMPORTANT: installed Gurobi is size-limited (native error 10010). Default-sized
  5-path/12-hour MILP had 2,165 variables and failed that license. SCIPY solved the
  same fixture optimally. No automatic backend substitution or config switch.
- Existing active perfect mode retained. No study configs/results, datasets,
  archives or HPWH physical/controller files modified; no Git commit.

## Previous revision: EV terminal energy value — finalized

User approved finalization on October 16. Finalization updated records only;
no additional code changes, simulations or Git commit. Await next directive.

User explicitly requested `J_terminal=v*(e0-eN)/N`, replacing the moving SOC-floor
penalty. Implemented for convex and MILP in controllers/ev.py. Measured faded
capacity is frozen per horizon for both energies. Full solve-calendar-day known
tariff mean supplies v directly in USD/chemical-kWh, with no efficiency conversion
or extra multiplier; current TOU gives $0.16875. Forecast retains the known tariff
callable separately, so no extended mobility/load forecast or future realized
price rows are needed. Minute sampling exactly captures the current hourly tariff.

`enforce_terminal_soc` remains the optional switch. Removed the old 100000
multiplier from active unified_control.json; retired terminal_soc_weight remains
accepted/ignored with an active-nonzero warning. Templates/docs updated. Plans
save v and initial/terminal chemical energy alongside the existing objective terms.
Study configs/results and physical/HPWH settings unchanged; no study launched.

Validation: 138 EV/unified tests passed (299.708 s), including new
test_ev_terminal_value and existing EV math, SOH, degradation, plant/integration,
config, warmup/cooldown and study-driver tests. New checks exercise CLARABEL,
SCIPY and GUROBI with none/throughput/rainflow costs. Full repository suite not
rerun. HPWH controller hash unchanged (see below); tracked diff whitespace checks
pass. Finalization approved; no commit.

## Previous revision: EV dynamic SOH — finalized

User approved finalization on October 16. Implementation and validation are
recorded in examples/single_home_hems/EV_SOH_VALIDATION.md. Finalization updates
records only: no additional simulations, code changes or Git commit. Await the
next directive; do not launch the full study without authorization.

- Initial SOH .85 explicit in home/template; unchanged initial SOC/HPWH settings.
- Normalized initial D=(1-SOH0)/.2; economic budget1.5; $/D=nominal pack value/1.5.
- Actual reference rainflow ages every plant, including baseline/disabled objective.
  Streaming open residue; current capacity; SOC-preserving fade with energy loss.
- Fresh measured SOH frozen per MPC horizon; nominal capacity still prices pack.
- Scored continuous damage increments retain warmup context, exclude warmup/cooldown.
- Health outputs/plots, battery-only state checkpoints and new regression tests.
- Final focused suite:142 tests passed. Full discovery attempted384 tests and
  found126 legacy/native API/import errors outside this revision; one idle-EV
  test assertion was corrected and its rerun/final focused suite passed.
- Final smoke: outputs/ev_soh_validation_20261016_final, 2 Gurobi jobs/6 cases,
  zero failed solves, energy residual <=7.96e-15 kWh. Resume skips both jobs.
- Full study config now targets outputs/ev30_dynamic_soh_v4; NOT staged or run.
  Old study inputs/results and prior diagnostics are preserved.
- HPWH controller SHA256 remains
  DFDAE01C6E79B150BB613A22468E61A7B55D2943598AAE8773E8353C9C2EE927.

Everything below describes historical revisions.

## Historical revision: EV-only departure comfort and economics — finalized

User approved finalization on October 15 after confirming HPWH must remain
untouched. No HPWH control code, equations, weights or settings were changed.

- EV comfort averages only expected pre-departure SOC shortfalls; zero without
  departures. MILP persistence penalizes charge/discharge/idle indicator changes;
  convex retains chemical-power variation. Hard SOC bounds remain unchanged.
- New/residual battery costs are $119/$0 per kWh. Physical fade remains calibrated
  to 3000 reference cycles at 80% SOH. Separate economic EOL of 70% yields the
  consistent, extrapolated 4500-cycle budget for all degradation cost methods.
- Departure metrics, explicit calibration/economic endpoints, templates and
  documentation are updated. Existing whole-window comfort diagnostics remain.
- Validation: 127 relevant tests passed; two short GUROBI driver jobs completed
  all six case simulations with zero scored solve failures. Existing legacy
  test_parametric_study still cannot import its missing analysis module.
- Original ev30_janfeb_nmc_v2 study is complete (30/30 jobs) and preserved.
  Revised full study ev30_departure_nmc_v3 is NOT staged or run.
- Diagnostics reproduced the CVXPY warning as a 300-second Gurobi timeout with
  15.5256% native gap and baseline fallback. Solver acceptance policy unchanged.
  Real-plan supporting-cut plots explain one-cut/two-master acceptance.
- Details: examples/single_home_hems/outputs/ev_diagnostic_20261015/report.md.
  Smoke results: examples/single_home_hems/outputs/ev_departure_eol_validation_20261015.
- Exact IEA attribution for $119/kWh remains unverified; treated as the requested
  assumption. Constant 25 C is an idealized publication assumption, not a new
  temperature model. Economic EOL extrapolation does not model an aging knee.

Finalization records approval only: no additional simulation, Git commit,
HPWH changes, or rewriting of original study results. Await the next directive.
The sections below are historical checkpoints, not the current implementation.

## Historical revision: Audi EV study + warm-up — finalized with user approval

User approved net-savings accounting and reproducible study driver, then requested
Audi only, varied initial SOC, full pilot staged but NOT executed, and only minimal
real driver validation. Latest instruction explicitly defers hh and smoothed
rainflow. Only four implemented MILP objectives remain in the current pilot.

Implemented this turn:
- New run_ev_study.py / single_home_hems/ev_study.py / config/ev_study_pilot.json.
- Four paired Audi/load cases, initial SOC .2/.5/.8, four configurations = 48 jobs
  staged under outputs/audi_milp_degradation_pilot_warmup; no r/ folder or status
  files, no simulations. Earlier 72-job staged snapshot remains untouched.
- Objectives: electricity, +health, +throughput, +multicut. hh/smoothed deferred;
  do NOT implement them. Exact B2 stays evaluation-only.
- UnifiedConfiguration.warmup_hours and home/study JSON fields default24, zero
  disables; start/duration are evaluation, initial SOC is at warm-up start.
  simulation_window() shifts runtime start earlier, extends duration. Native
  initialization remains0. All cases carry controller/plant state into evaluation.
- Evaluation masks exclude warm-up/twin tail from energy/aging/comfort/cycles/solver
  metrics; boundary SOC and continuing modes handled. Water replay mapping remains
  evaluation-anchored; RF training actual-start-anchored. Native schedule.py needed
  two guards for prepared cross-year data and annual timezone normalization.
- Legacy adapter explicitly warmup0; short tests0 or15minutes. No HPWH/EV math changes.
- Baseline-relative operating, degradation and net savings; profit removed in new
  outputs. none/multicut method aliases, final stored energy and p95 timing.
- Frozen mobility input to unified runner; input/source/package fingerprints,
  append-only configuration IDs, attempt preservation, integrity-checked resume.
- Initial long-path smoke failure preserved; compact run folders fix Windows paths.
- Successful real smoke outputs: outputs/ev_driver_validation, two configs x
  30-minute comparisons, 30-minute horizons, six cases; 8 scored OCPs, no failures,
  identical all-baseline trajectories, energy/safety/output checks pass. Resume
  repeats zero jobs. No performance conclusions from this brief check.
- Latest smoke: outputs/ev_warmup_validation, two configs x (15min warm-up +30min
  evaluation), six subsimulations, 4warm-up/8scored/4tail OCPs, zero failures.
  Verified boundary SOC, energy/safety, evaluation masks, identical baselines,
  figures/net-savings; actual resume skips both completed jobs (attempt1).
- 105 focused tests pass (6 new warm-up tests). Final full discovery:365 tests,
  126 known legacy errors,0 assertion failures,0skips. Same error categories/count
  as pre-warm-up359-test discovery. No new EV/study/unified errors. Full suite not green.
- Documentation: EV_STUDY.md; appended CUSTOM_CHANGES.md. Nothing git-staged or
  committed. Existing dirty files, archives and earlier result folders preserved.

User approved finalization on 2026-10-14 with the documented legacy-test limitations.
Changelog, EV_STUDY, UNIFIED_PIPELINE, UNIFIED_VALIDATION and templates are updated.
No optimizer additions remain in current scope. Await the next directive.
Finalization changed documentation only; no additional simulations or Git commits.
Do not run the full pilot without a new explicit request. No subagents authorized.

The older finalized checkpoint below describes the PRECEDING revision only.

## Status: finalized with user approval — 2026-10-14

User approved the EV health/degradation design and corrected economics to use
only all-baseline savings and profit. Implementation and validation are complete.
User approved finalization with the documented legacy-test limitations.
Nothing staged or committed. Await the next directive.

## Current implementation

- All EV OCP construction/controller variants now in controllers/ev.py;
  ev_formulation.py is an import shim. Numeric rainflow/calibration is shared in
  five_home_block/ev_degradation.py, not duplicated in reporting.
- Control config: health_soc_min/max, ONE health_soc_weight, overall health_weight,
  comfort_soc_min/comfort_weight. Baseline and fallback charge to comfort minimum.
- Optional degradation_method = rainflow or linear_throughput. Shared B2 reference
  cycle life, replacement cost, reference depth/mean and EOL calibration.
  Both methods include battery-side driving. Calendar/temperature/C-rate excluded.
- Rainflow uses only the SECOND supplied convex degradation paper. Exact depth-only
  cycles/subgradients, historical constant offset, supporting-cut outer loop,
  verified full-objective upper/lower bounds, one 300-second OCP budget and fallback.
  Third paper supplies B2 evaluation and linear baseline. Pay-for-performance
  paper excluded entirely.
- Realized metrics: selected estimated degradation and B2 reference dollars.
  savings = all_baseline electricity cost - case electricity cost.
  profit = savings - TOTAL case reference degradation (not incremental).
  No twin-relative money metrics. Baseline profit can be negative.
- Physical [0,1] OCP bounds, full arrival-distance demand, negative-price guard,
  MILP-only dwell/caps, signed convex power, fixed solve clock and actual-load cap
  unchanged. No EV projector dwell/caps. Optional discharge budget excludes driving.
- HPWH equations/normalization/scenario aggregation/projector unchanged.
  HVAC remains pass-through; EV scenario remains an explicit placeholder.
- Four unified JSON examples/templates migrated without changing unrelated tuning.
  New degradation defaults off; comfort weight defaults zero for explicit tuning.
- See EV_DEGRADATION.md and UNIFIED_PIPELINE.md for equations, assumptions and usage.
  CUSTOM_CHANGES.md has a dated October 14 entry.

## Completed validation

- 21 new degradation tests pass: shared calibration, half cycles, mean-SOC B2,
  supporting planes including ties, history, all three solver backends, analytic
  convex optimum, exhaustive MILP optimum, missing-bound/iteration/time fallback,
  config requirements, realized trip extrema and exact baseline-relative profit.
- 62-test focused group passed (before three additional degradation safeguards).
  Includes mixed HPWH/EV OCHRE three-case fixture with rainflow, figures/data,
  energy conservation, fixed solve counts and zero failed solves.
- Final full discovery: 349 tests, zero assertion failures, 126 pre-existing
  legacy API/fixture errors, no skips. Error categories match the October 12
  checkpoint. No errors in EV degradation/formulation/V2H or unified tests.
- 41-test affected-module rerun with workspace-local temporary directories passed.
- Final full run also used workspace-local temporary directories and removed
  the 12 sandbox permission error entries seen in the first full attempt.
- Validation details: examples/single_home_hems/UNIFIED_VALIDATION.md. The complete
  legacy suite is NOT green; fixing its 126 unrelated errors is outside this scope.
- Native 300-second wiring tested, not a real five-minute timeout stress test.
  No research-duration experiment. Gurobi restricted license used successfully.

## Operational notes

Use apply_patch; no subagents authorized. Preserve unrelated dirty files and
earlier uncommitted work. Native OCHRE/home_factory remain normal dependencies.
Archive/legacy_hems_2026-10-07 contains 55 committed October 1 source/model files
plus a separate EV snapshot, NOT every intermediate uncommitted version.
No archive/input deletion or changes to source papers/LaTeX for this revision.

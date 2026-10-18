# Unified pipeline validation checkpoint

Date: 2026-10-14. Study/net-savings/warm-up revision tested and finalized with user
approval. Nothing git-staged or committed. Historical entries retain their scope.

## October 14 study, net-savings and warm-up revision

- **105 focused tests pass. Final full discovery: 365 tests, zero assertion
  failures, 126 known legacy errors, zero skips.** The errors remain in legacy
  equipment/model/utility fixtures and five old import failures, matching the
  pre-change error count/categories; no new EV/study/unified errors. Full suite
  is not green, and unrelated legacy repairs are outside this revision.
- Six new warm-up tests cover default24/zero/invalid alignment, real EV+HPWH
  control state carryover, evaluation-only metrics and histories, boundary SOC,
  boundary cycle counting, cross-year schedule coverage/timezone, and water
  replay/training anchors. Study tests verify warm-up is included in frozen replay.
- Minimal real driver validation: two Audi/GUROBI MILP objectives, 15-minute
  warm-up +30-minute evaluation, 30-minute forecast. All six subsimulations
  completed; 4 warm-up, 8 evaluated and 4 twin-tail OCPs; zero failures.
  Evaluation CSVs have30 rows; native trajectories retain45/75/45 rows.
  Checked savings algebra, baseline zero savings, initial SOC continuity,
  SOC bounds/actual-load discharge caps, paired identical baselines and figures.
- Repeating the actual smoke command resumed both completed jobs without another
  simulation (attempt1 retained, output hashes verified). Outputs:
  `outputs/ev_warmup_validation`. Earlier validations/failure artifacts are preserved.
- **Full pilot not run:** `outputs/audi_milp_degradation_pilot_warmup` contains
  12 frozen cases x4 implemented MILP objectives =48 jobs, all with24-hour warm-up.
  No run folders/status files exist. hh and smoothed rainflow are deferred by
  user request and excluded from this pilot. Exact B2 remains evaluation-only.
- No OCP equations, weights, gap acceptance or control-clock rules changed.
  Two small native schedule fixes were necessary for January1 pre-roll: avoid
  re-duplicating data already covering both years and retain normalized timezone
  when annual data is repeated. Regression tests exercise both cases.
- Short checks validate mechanics, not long-horizon convergence, degradation
  accuracy or comparative controller performance. A24-hour production warm-up
  was staged but not simulated. Existing native weather-station/radiation warnings
  still appear in short runs. Native raw diagnostic metrics include the full
  trajectory; use the unified report/comparison files for scored metrics.

Tests used the repository virtual environment and workspace-local TEMP/TMP to
avoid sandbox temporary-directory permission errors. `git diff --check` passed.

## October 14 EV health/degradation revision

- **Final full discovery: 349 tests, 126 known legacy API/fixture errors,
  zero assertion failures, no skips.** Error categories and counts match the
  October 12 baseline; no errors in EV degradation/formulation/V2H or unified tests.
  The full repository suite remains non-green; unrelated legacy repairs were not made.
- All **21 new degradation tests pass**: common B2-derived lifetime/replacement
  calibration, full/half/nested cycles, B2 mean-SOC effects, randomized supporting
  planes including plateaus/ties, retained history, all backend/formulation pairs,
  exact analytic convex and exhaustive MILP objective checks, missing certificates,
  iteration/remaining-time safeguards, disabled costs, driving normalization,
  strict config requirements, actual trip extrema, baseline comfort charging,
  and total-reference-cost profit versus all-baseline only.
- The 62-test focused group passed before three final certificate/analytic checks
  were added. The final 21-test degradation group and final full discovery include
  those additional checks. The three-case OCHRE fixture now enables rainflow EV
  MPC alongside HPWH MPC, verifying energy/SOC, fixed solve counts, actual-load
  limits, matched thermal trajectories, retained plots/CSVs/report and zero failed solves.
- Both methods execute with SCIPY convex/MILP, CLARABEL convex, and GUROBI
  convex/MILP under the installed restricted license. A first-use license banner
  can appear. Rainflow solves log full-objective bound certificates; native master
  optimality alone does not establish acceptance.
- Initial full discovery (before three test-only additions) had 346 tests and
  138 error entries, including 12 sandbox temporary-folder permission entries.
  A 41-test affected-module rerun passed with workspace-local temporary files.
  Final full discovery used that same temporary-directory workaround and retained
  only the original 126 legacy errors. One redundant quiet rerun was stopped after
  it ceased returning progress; its incomplete result is not used as evidence.
- Numerical lower-bound guards can prevent impractically tight tolerances from
  certifying; conservative fallback is intentional. Default tolerances and a
  stricter analytic-objective check pass. Native remaining-budget/quiet options
  and expired-budget behavior are tested; no real five-minute stress solve was run.
- No research-duration experiments, native OCHRE changes, HPWH mathematical
  changes, mobility-loader changes, archive changes, staging or commits.

The final audit used the existing environment's unittest discovery, with
`tempfile.tempdir = str(Path.cwd())` before collecting `test`. Detailed definitions,
Current calibration and schema migration are in [degradation](../degradation.md).

## October 12 EV formulation revision

- 17 new tests pass: signed chemical power/efficiencies, no simultaneous modes,
  fixed availability, full trip demand, soft SOC/terminal targets, dt/N costs,
  negative-price exceptions, one-interval horizons, end/start cycle budgets,
  shortened forecast windows, active-cycle continuation, OCP-only dwell,
  discharge history overlap/exclusion of driving, infeasible-trip fixed-clock
  fallback, 5% gap feasibility/integrality checks, plant saturation and max timing.
- The existing 56-test EV/unified group passes, including small actual-OCHRE
  Audi/NHTS runs, five-home aggregation, independent controllers and three-case
  outputs. A final 32-test math/unified group passes after report assertions were
  expanded to maximum timing and target/saturation fields.
- Full discovery: **328 tests; 202 passed, 126 pre-existing legacy errors,
  zero assertion failures**. Error categories match the October 7 breakdown below.
  Sandbox added 12 Windows temporary-folder errors; approved outside-sandbox rerun
  removed them. The full legacy suite is not green.
- SCIPY/CLARABEL execute successfully. GUROBI is now installed: actual convex,
  MILP and infeasible-fallback smoke solves passed with a local restricted license.
  Native 300-second/quiet options remain tested; no five-minute stress solve or
  research-length experiment was run. A first-use GUROBI license banner may appear.
- HPWH equations, scenario mean/max aggregation, scaling and actuator policy are
  unchanged. Native OCHRE, source LaTeX, archive and unrelated input edits preserved.
- Canonical configs/templates and guide updated; old Python field aliases remain
  compatible, while unified JSON rejects old health/trip fields explicitly.

## October 7 historical validation

### Cycling/configuration/reporting revision

- Added 12 focused tests: all eight cap/dwell/persistence combinations for EV
  and HPWH in both formulations; MILP versus convex enforcement; actuator safety;
  independent weights; backend timeout/quiet options; feasible integral incumbents;
  standalone documented templates; missing-field diagnostics; output isolation;
  actual states and start counts. All passed.
- The 54-test unified/single-home/scenario/legionella group passed. Short real
  OCHRE integration verifies three-case outputs and fixed solve clocks. A final
  report-fixture rerun verifies the single report includes cycle, SOC, timing,
  backend, timeout and fallback metrics.
- Full discovery: **311 tests; 185 passed, 126 existing legacy errors, zero
  assertion failures**. The error categories match the preceding revision:
  missing native imports (2), Envelope zones (16), dict/index fixtures (27),
  native EV ambient schedule (7), PV orientation (2), short schedules (5), tank
  volume (37), humidity signature (2), Simulator time arguments (17), old
  parametric-study import (1), undefined legacy fixture names (10).
- Initial sandbox execution added 12 Windows temporary-directory permission
  errors; the approved outside-sandbox rerun removed those environmental errors.
- SCIPY and CLARABEL execute successfully. GUROBI options are unit-tested but
  its package/license is unavailable, so actual GUROBI execution remains unverified.
  No five-minute stress solve or research-length simulation was launched.
- Native OCHRE source is unchanged. `home_factory`, shared config/equipment/
  schedule utilities and external physics/input files remain dependencies;
  only legacy simulator configuration fallback was removed from the new loader.
- Both `.template.json` files load directly; their ignored `_comments` objects
  document fields, units, choices and inactive values. User tuning is preserved.

## Previous unified revision evidence

### Scope and results

- Added 15 unified tests; all passed. They cover home/control loading and precedence,
  independent solver settings, canonical CLI delegation, explicit unsupported-mode
  rejection, causal deterministic forecasts, fixed EV solve timing, held-command
  prediction constraints, HP versus resistance demand, physical clipping, and
  three-case output persistence.
- A 16-minute real-OCHRE fixture exercises all-baseline, charge-only twin, and
  advanced EV+HPWH control with an off-grid departure/return. It verifies SOC and
  energy conservation, actual-load limits, retained data/plots, two scheduled
  solves per active controller in the comparison window, and zero failed solves.
- The 58-test EV/HEMS group passed before four additional unified checks were
  added; the resulting 15-test unified group also passed separately.
- All 48 focused HPWH CLI, replay, scenario-optimizer, scenario-provider, and
  legionella regression tests passed. Existing HPWH optimization mathematics
  remain unchanged.
- Full repository discovery: **299 tests; 173 passed, 126 errors, zero assertion
  failures**. The errors remain the same legacy API/fixture categories recorded
  in [EV/V2H validation](ev_v2h.md). The full suite is not green.
- Sandbox temporary-folder restrictions affected an initial test attempt;
  permitted execution outside the sandbox removed those environmental failures.

Validation used unit tests and short integration fixtures, including existing
Audi/NHTS and five-home tests. No new 72-hour or research comparison experiments
were run. The earlier 72-hour results are historical evidence for the preceding
revision, not new unified-pipeline performance claims.

## Output contract

Every future run saves `all_baseline`, `non_v2h_twin`, and `advanced` case folders,
each with raw/standardized data, controls, EV metrics, thermal training/sensors,
backed-load reference/actual series, and plots. Home-level overlays compare all
three. Root-level `data/solver_stats.csv`, `data/solve_history.csv`, and
`reports/run_report.md` expose per-home/device/case timing and failures alongside
performance metrics. `data/performance.csv` and `ev_states.csv` supplement the
existing machine-readable comparisons and case outputs.

The twin keeps the selected controllers, including EV charging MPC, but disables
V2H. Its forecast-tail records are retained separately from comparison-window
timing denominators. The all-baseline case has no optimization. Discharge safety
uses contemporaneous physical compressor/fan and other selected load powers, not
the reference trajectory or previous-step demand.

## Known boundaries

- GUROBI backend selection and small EV execution are verified locally; license
  capacity and large-instance performance remain environment-dependent.
- HVAC remains pass-through; requesting HVAC MPC raises an explicit error.
- EV scenario MPC is structure-only; requesting execution raises an explicit
  error before simulation. No scenario equations/probabilities were invented.
- HPWH scenario control retains its 12-hour horizon and 15-minute model/control
  restriction. Other saved HPWH predictors retain 15-minute model/control steps.
- Deterministic mode currently requires an external, causally issued forecast
  JSON; missing inputs never fall back to perfect data.
- Final SOC differences and convex discrete-projection limitations remain;
  energy/cost differences are not automatically equal-terminal-energy savings.
- Cycle count means realized operating starts, not equivalent-full battery
  cycles. EV MILP uses historical ends plus planned starts over the split 24h
  window; EV projector no longer enforces caps/dwell. HPWH retains its horizon
  budget with one-bit preceding-hour stop carry-in and convex actuator enforcement.
- Full trip demand plus hard OCP SOC can be infeasible; fallback is intentional.
  Convex discrete projection does not guarantee realized discharge-budget use.
  Missing/nonfinite nonconverged EV MILP gaps trigger fallback; default threshold 5%.
- Native 300-second backend limits exclude model build/canonicalization, not a
  hard wall-clock process deadline. Feasibility/integrality acceptance tolerance
  is 1e-4; physical safety remains enforced by the plant.
- No native `ochre/` source was changed. User-owned configurations and dataset
  additions/replacement were preserved. Changes are not staged or committed.

See the [pipeline guide](../pipeline.md) for current commands and file schemas.
Legacy recovery copies and their completeness limits are documented in the
[archive](../../../../archive/legacy_hems_2026-10-07/README.md).
<!-- Moved during the October 17 documentation cleanup; paths in historical
     validation records refer to the repository or original example directory. -->


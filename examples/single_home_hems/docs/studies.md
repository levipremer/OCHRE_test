# Reproducible EV control study

## Status

The driver and eight objective variants are implemented. Smoothed rainflow and hh
are deferred by user request and excluded from the current pilot. NMC SOC-aware
additive rainflow is the common evaluation model, not an OCP option. Earlier
staged snapshots retain their old calibration; use a new output directory and
stage again for revised objectives/economics. The original 30-job study in
`outputs/ev30_janfeb_nmc_v2` and 48-job `outputs/ev30_dynamic_soh_v4` are complete
and preserved. Current scenario selection and horizon/bank definitions use
separate output directories; their manifests/status files are the source of
truth for completion, not this guide.

## Pilot definition

The [pilot configuration](../config/ev_study_pilot.json) is editable JSON with
`_comments` metadata. Paths below refer to the repository or example root.
It references the standalone unified home and control documents. One home and one
Audi EV use January 3-8 and February 3-8, 2018 (exclusive end), seeds 41/42/43,
and initial SOC 0.4. Six paired cases times eight formulations = **48 unified jobs**,
each saving all-baseline and advanced, plus the twin when enabled. HPWH/HVAC stay baseline;
EV V2H remains enabled and forecasts are perfect. Evaluation is 120 hours, preceded
by 24-hour warm-up and followed by 24-hour cooldown. The horizon is 12 hours,
control interval 15 minutes, and plant interval remains one minute.

Optional top-level study `run_non_v2h_twin: false` overrides the control document
and stages baseline + advanced only. When omitted, inherit the control document
(supplied configs: false; omitted everywhere: true). With no twin, baseline provides the extended backed-load forecast;
reports and plots omit twin results. Use a new output directory when changing
this setting; never combine two-case and three-case cohorts as equivalent.

`pair_mobility_by_seed=true` selects each Audi source trace once at the first
month's anchor and shifts its entire timeline to the other month, including
pre/post-roll and forecast tails. Vehicle, trip distances, relative event times,
and availability are identical. February deliberately sacrifices weekday alignment.
All eight formulations reuse these frozen traces; there is no Audi wrapping.

The original staged scenario study expanded three MILP objectives × four counts ×
five selectors into 60 combinations and 360 jobs. The current edited
[scenario definition](../config/ev_study_scenarios.json) retains the health objective
but narrows counts to 5/7/10 and selectors to `tod_only`, `tod_dow_state_hier`,
and 12-hour-history kNN: nine combinations and **54 jobs** over six situations.
It retains the same Audi vehicles,
physics, SOC, weights and windows, but uses corrected, later mobility windows.
All selectors draw completed non-overlapping paths from the preceding **28 days**;
the source window also includes kNN history before that bank. A four-interval
state-persistence prefix corrects only current-state mismatches. Matching paths
retain their near-term transitions. The number actually selected is diagnostic;
it can be smaller than the requested count when candidates are insufficient.

The [horizon/bank definition](../config/ev_study_horizons.json) fixes MILP electricity
+ comfort + health + terminal, five `tod_dow_state_hier` scenarios and the one-hour
persistence prefix. Banks of 14/21/28 days cross horizons of 6/12/24 hours: nine
combinations and **54 jobs** over the same six situations. Physical settings,
weights and evaluation padding are unchanged. Its top-level 24-hour horizon
reserves a 48-hour twin forecast tail; per-job prediction and forecast lengths
both follow the horizon axis. All combinations share frozen mobility, with
enough past/future data for the largest windows; anchors need not match older
studies. Choose a fresh output directory; the supplied definitions may name
existing historical outputs, which must not be overwritten or silently restaged.

Replay version 2 anchors **simulation start**, not the beginning of saved history.
Version 1 accidentally shifted history into the evaluation timeline. The old
realized traces lack sufficient preceding Audi history for this scenario study;
they are preserved, not silently repaired. New paired windows retain their real
past data, and February shifts only by the January-to-February anchor difference.
New code rejects restaging old frozen manifests into the same output directory.

Both convex and MILP compare electricity + comfort; adding health SOC; then
adding throughput or multicut rainflow degradation to the health variant.
Comfort weight is 1000 in every variant; current code integrates SOC-floor
shortfall throughout the horizon (dt/N), including away time. Health SOC weight is zero for the first two,
1000 for the remaining six. Overall health/electricity weights are 1. Health bounds
are 0.2/0.8, comfort floor 0.4. Persistence, dwell/caps and discharge-budget terms
are off. Terminal energy valuation is enabled: `v*(e0-eN)/N`, using the
daily-minimum tariff with no separate weight. Baseline/fallback charges to 0.4.
The current definition selects GUROBI; install a suitable license separately.
Seed=0 and Threads=1 improve reproducibility. Set `GRB_LICENSE_FILE` to
your valid license if discovery is ambiguous. SCIPY/HiGHS remains a license-free MILP backend.
Forecast trip infeasibility is handled by bounded unmet-energy slack with weight
10000; actual trip shortfall remains a separate plant metric. Use a new output
directory when staging this revision; existing frozen studies retain their
original formulation and weights.
The study retains a 300-second OCP budget and 5% MILP objective gap,
with the approved separate rainflow envelope accuracy of $1e-6 + 0.1% of
incremental aging dollars, maximum 100 master solves.

Baseline cache I/O uses Windows extended paths so long OneDrive roots and the
full cache key do not make equipment artifacts unreadable after publication.
Missing or modified cache contents still fail integrity checks; they are not
silently reused. Failed attempts are preserved and retried on the next run;
verified completed jobs are skipped.
Publication retries temporary directory-access locks six times (6.2 seconds of
total backoff). If access remains denied, the valid baseline simulation is used
without caching; a warning and `baseline_reuse.json` record explain why. This
does not excuse missing/corrupt entries that were previously published.

`warmup_hours` and `cooldown_hours` default to 24 (0 disables). Each `start_time` is the evaluation
start; initial SOC applies at warm-up start. All controllers run throughout warm-up,
retaining plant state, solve clocks, cycle/dwell and aging history. The 120-hour
evaluation excludes all warm-up/cooldown costs, aging, starts and solver timing. Evaluation
SOC may differ across controllers; the driver reports both `study_initial_soc`
and realized EV `initial_soc`. Changing either padding period requires a new study revision.
Cooldown continues normal controls without forcing recharge or equal final SOC;
it is not a substitute for a terminal constraint. Forecasts remain full-length
through evaluation and cooldown. Final stored energy is measured at evaluation end.
The perfect-forecast config still names `outputs/ev30_dynamic_soh_v4`; choose a new
directory if restaging it with corrected replay. Previous outputs are preserved. Economics
use $119/kWh new, zero residual, and a 70%-SOH economic endpoint while retaining
3000 cycles to the 80%-SOH calibration endpoint. Initial SOH is explicitly .85;
capacity fades under reference rainflow in every case, even without an OCP aging
term. Scored degradation sums continuous-ledger increments, retaining warmup
cycle context but excluding warmup/cooldown increments. All methods use the same effective
4500-cycle economic life. Existing SOC-fraction-hour comfort metrics remain
whole-window diagnostics; departure-comfort events are reported separately.

## Commands (repository root)

Stage ONLY; freeze schedules and resolved input documents, without simulation:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_study.py --study-config examples/single_home_hems/config/ev_study_scenarios.json
```

For the horizon/bank study, substitute `config/ev_study_horizons.json` in that
path. Add `--run` only when ready to execute the staged study.
Both deterministic and scenario studies print elapsed minutes and job number/ID,
with job completion/skip messages. Current `StudyProgress` suppresses per-step and
phase callbacks; it does not provide a percentage or wall-time ETA.
Use `--no-progress` to suppress these messages; OCHRE messages remain unchanged.
Already-running Python processes will not gain this indicator. The existing
source/environment resume guard remains enforced for older study revisions.

Run minimal driver validation for a selected config in a fresh, separate folder:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_study.py --study-config examples/single_home_hems/config/ev_study_scenarios.json --smoke-test --output-directory examples/single_home_hems/outputs/ev_scenario_smoke
```

This runs only the first two configurations on one Audi case at initial SOC 0.5,
with 15-minute warm-up/cooldown, 30-minute evaluation and 30-minute prediction horizon. It exercises the
actual OCHRE pipeline with configured cases, not a performance study. It does not validate
long-horizon aging accuracy, trips absent from that brief window, or new methods.
The smoke prefix is capped at two control intervals to fit its 30-minute horizon.

Later, explicitly execute selected implemented configurations with `--run` and
`--configurations convex_comfort milp_comfort`. `--cases` accepts exact IDs
from the manifest. For the scenario study, IDs look like `rainflow__n10__knn24`.
An unfiltered `--run` executes every job in the selected config and is therefore
a full study, not a smoke test. The CLI rejects unsupported methods without substitution.

## Reproducibility and extension

- Each schedule case is sampled once and serialized as departure/arrival/distance
  records with source metadata. Initial SOC and control choices cannot resample it.
- Plant input files, OCHRE defaults, case snapshots and controls carry SHA-256
  fingerprints. Source code and Python/solver package versions are recorded on
  first execution; changed inputs or execution environments require a new study
  output directory. Keep the original input files available and unchanged.
- Append new configurations with new IDs. Existing IDs, schedules, weights and
  baseline-defining data cannot silently change. New horizons or physical cases
  require a new study revision so coverage cannot drift.
- Re-running resumes only verified complete jobs. Failed/interrupted attempts are
  preserved; a retry gets another attempt directory. Do not edit the manifest or
  generated files; edit the study definition and stage a new revision as needed.
- For GUROBI, fixed seed and one thread improve reproducibility but do not
  promise bitwise-identical numerical results across machines. Those options
  do not apply to SCIPY.
- `reuse_all_baseline=true` enables study-local, hash-verified trajectory caching.
  Identical plant inputs, mobility, comfort targets and code/runtime fingerprints
  share one all-baseline run per situation. Per-configuration metrics/figures are
  regenerated, including estimated degradation under the selected cost model.
  Every job still saves all three outputs and `all_baseline/data/baseline_reuse.json`.
  Twins cannot be shared: their configured control changes the backed-load reference.
  Reuse requires identical forecast extensions too, so different horizon levels
  may retain separate baseline cache entries. Corrupt/incomplete
  cache entries are not accepted. Cache snapshots use Parquet, not pickle.

## Outputs and interpretation

`manifest.json` describes every case/job and warm-up setting.
`cases/` holds immutable inputs and mobility, `jobs/` holds controls and status
checkpoints, and `r/<job-hash>/a001/` holds attempts (short paths for Windows).
Each status records its attempt path. Attempt directories use the existing unified
layout, including trajectories, figures, solve history and consolidated run report.

`study_runs.csv` includes failures and jobs not run. `study_metrics.csv` contains
completed results. `paired_summary.csv` aggregates only cases completed by EVERY
staged configuration, so it is intentionally empty until those comparisons exist.
`study_report.md` describes coverage. Individual trajectories remain available.
`paired_metrics.csv` has **one row per combination** (nine in each current scenario grid),
factor labels and `paired_case_count`. After common cases complete, columns such
as `advanced_whole_home_energy_adjusted_net_savings_usd_mean` and their
`non_v2h_twin_...` counterparts keep outcomes separate. Staged rows have zero
paired-case coverage and no invented performance values. `paired_summary.csv`
retains the long configuration/case/device format.
Summary statistics include count/mean/sample-standard-deviation/min/max for all
numeric metrics, grouped by configuration, case and device. **Advanced and
non-V2H are never averaged together.** `max_solve_seconds_max` is the worst
individual solve; the mean of per-run maxima is also retained but is not that maximum.
Charge/discharge starts are distinct from throughput equivalent full cycles.

`figures/` adds per-mobility-case cross-formulation overlays (availability, SOC,
price, EV power/mode, whole-home power, stored-energy change) and paired savings
bar charts with per-case dots and sample standard deviations, including inventory
correction and energy-adjusted net savings. Final-ensemble forecast accuracy bars
compare formulations when scenario controls are present (prefix included).
Advanced and non-V2H have separate figures when enabled; omitted cases produce no rows/figures.
Overlays for the original formulation study show completion coverage; savings
plots use the common completed case intersection.
Run-level `raw_results_all_phases.csv`, `controls_all_phases.csv`, solve/action
histories and `evaluation_mask.csv` retain padding for audit; comparison files
contain evaluation only. No plots imply that unexecuted staged jobs have results.

The factorial study instead produces one figure set per objective and control
case, averaging **only situational cases**. Requested scenario count is the
horizontal axis and selection methods remain separate curves (mean ± sample SD).
Savings, forecast accuracy and EV outcome/solver panels never pool counts,
selectors or twin/advanced. Per-run selected-case trajectory plots remain available;
the study-level factorial figures do not duplicate situation-specific overlays.

For cleaner saved-run trajectories, use the standalone retrofit plotter from
the repository root (no simulation is run):

```powershell
.venv\Scripts\python.exe tools/plot_ev_trajectories.py examples/single_home_hems/outputs/ev30_dynamic_soh_v4/r/0cbecce4eb26/a001 --title "MILP comfort + health | February, seed 42"
```

The upper panel compares V2H/non-V2H/baseline SOC; the lower shows actual
charge/idle/discharge/away states with a faint right-axis tariff. Mode is inferred
from realized power and availability, not requested commands; tiny vertical
offsets expose coincident states. Optional `--start`/`--end` zoom the evaluation
window; `--price-alpha` changes tariff opacity. Set `--home-id`/`--vehicle-id`
for other equipment. PNG and vector SVG go under the study's `retrofit_plots/`,
outside frozen attempts; existing plots are never overwritten. Supply a new
`--output-directory` for another view. The standalone `tools/` utility is outside
simulation source fingerprints and does not invalidate study resumes.

The horizon/bank grid instead adds annotated heatmaps: savings (including the
inventory correction), comfort/trip service and aging, forecast errors, and
computation. Columns are forecast hours; rows are historical weeks. Each cell
averages paired situations only, separately for twin and advanced. Worst solve
time uses the maximum; failures/no-scenario counts use sums, labeled explicitly.
Missing values stay N/A. Forecast scores include the prefix; changing horizons
changes target coverage, so inspect saved score counts before comparing accuracy.
Figures are generated only after paired completed results exist, not at staging.

Net savings = (baseline operating cost - case operating cost) +
(baseline NMC reference degradation cost - case NMC reference degradation cost). Baseline net savings
is zero. Reported degradation is unweighted. Final SOC/stored energy, unserved
trips, health/comfort violations, starts, mean/p95/max solve time, detailed gaps,
timeouts and fallback history remain available. The separate
`energy_adjusted_net_savings_usd` adds `vT*(ET_case-ET_baseline) -
v0*(E0_case-E0_baseline)`, using actual faded-capacity chemical energy at evaluation
boundaries and common first/last daily-minimum tariff values. It is independent of
the terminal objective; archived missing boundary data remains unavailable.
Forecast errors/coverage and no-scenario counts also propagate to study summaries;
see [score definitions](scenarios.md#outputs-and-validation). Run-level forecast
error CSVs retain issuance/lead detail. There is no silent exclusion of failed
solves from completed simulations.

## Performance and resumption

Forecast scoring uses batched queries; horizon preparation uses array slices.
For normal resumption use the original matching checkout/configuration; for changed
formulations, prefixes or case selection, stage a new cohort. Never edit manifest
hashes to bypass compatibility checks. Keep pre/post-maintenance timing cohorts separate.
Parallel study execution is not implemented.

## Deferred optimizer work

Smoothed rainflow and hh are not implemented or validated. They are outside this
revision's remaining scope unless requested again.

# Reproducible EV control study

## Status

The driver and five objective variants are implemented. Smoothed rainflow and hh
are deferred by user request and excluded from the current pilot. NMC SOC-aware
additive rainflow is the common evaluation model, not an OCP option. Earlier
staged snapshots retain their old calibration; use a new output directory and
stage again for revised objectives/economics. The original 30-job study in
`outputs/ev30_janfeb_nmc_v2` is complete and preserved. No full study has been run
with the departure-only comfort or revised economics described below.

## Pilot definition

The [pilot configuration](../config/ev_study_pilot.json) is editable JSON with
`_comments` metadata. Paths below refer to the repository or example root.
It references the standalone unified home and control documents. One home and one
Audi EV use January 3-8 and February 3-8, 2018 (exclusive end), seeds 41/42/43,
and initial SOC 0.4. Six paired cases times five formulations = **30 unified jobs**,
each saving all-baseline, non-V2H twin and advanced results. HPWH/HVAC stay baseline;
EV V2H remains enabled and forecasts are perfect. Evaluation is 120 hours, preceded
by 24-hour warm-up and followed by 24-hour cooldown. The horizon is 12 hours,
control interval 15 minutes, and plant interval remains one minute.

`pair_mobility_by_seed=true` selects each Audi source trace once at the first
month's anchor and shifts its entire timeline to the other month, including
pre/post-roll and forecast tails. Vehicle, trip distances, relative event times,
and availability are identical. February deliberately sacrifices weekday alignment.
All five formulations reuse these frozen traces; there is no Audi wrapping.

The objectives are convex electricity + comfort; MILP electricity + comfort;
MILP adding health SOC; then adding throughput or multicut rainflow to that MILP.
Comfort weight is 1000 in every variant; its cost now averages pre-departure
shortfalls only (zero when no departure is forecast). Health SOC weight is zero for the first two,
1000 for the last three. Overall health/electricity weights are 1. Health bounds
are 0.2/0.8, comfort floor 0.4. Persistence, dwell/caps, discharge-budget and
terminal terms are off. Baseline/fallback charges to 0.4. GUROBI uses fixed Seed=0
and Threads=1. The study retains a 300-second OCP budget and 5% MILP objective gap,
with the approved separate rainflow envelope accuracy of $1e-6 + 0.1% of
incremental aging dollars, maximum 100 master solves.

`warmup_hours` and `cooldown_hours` default to 24 (0 disables). Each `start_time` is the evaluation
start; initial SOC applies at warm-up start. All controllers run throughout warm-up,
retaining plant state, solve clocks, cycle/dwell and aging history. The 120-hour
evaluation excludes all warm-up/cooldown costs, aging, starts and solver timing. Evaluation
SOC may differ across controllers; the driver reports both `study_initial_soc`
and realized EV `initial_soc`. Changing either padding period requires a new study revision.
Cooldown continues normal controls without forcing recharge or equal final SOC;
it is not a substitute for a terminal constraint. Forecasts remain full-length
through evaluation and cooldown. Final stored energy is measured at evaluation end.
The next study default is `outputs/ev30_dynamic_soh_v4`; it is not staged or run
by this revision. Previous frozen inputs and outputs are preserved. New economics
use $119/kWh new, zero residual, and a 70%-SOH economic endpoint while retaining
3000 cycles to the 80%-SOH calibration endpoint. Initial SOH is explicitly .85;
capacity fades under reference rainflow in every case, even without an OCP aging
term. Scored degradation sums continuous-ledger increments, retaining warmup
cycle context but excluding warmup/cooldown increments. All methods use the same effective
4500-cycle economic life. Existing SOC-fraction-hour comfort metrics remain
whole-window diagnostics, not the new departure-only objective value.

## Commands (repository root)

Stage ONLY; freeze schedules and resolved input documents, without simulation:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_study.py
```

Run the minimal driver validation, in a separate `_driver_smoke` output folder:

```powershell
.\.venv\Scripts\python.exe examples/single_home_hems/run_ev_study.py --smoke-test
```

This runs only the first two configurations on one Audi case at initial SOC 0.5,
with 15-minute warm-up/cooldown, 30-minute evaluation and 30-minute prediction horizon. It exercises the
actual three-case OCHRE pipeline, not a performance study. It does not validate
long-horizon aging accuracy, trips absent from that brief window, or new methods.

Later, explicitly execute selected implemented configurations with `--run` and
`--configurations convex_comfort milp_comfort`. `--cases` accepts exact IDs
from the manifest. An unfiltered `--run` executes all five variants and is therefore
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
- GUROBI's seed is fixed and one thread is requested. This improves reproducibility
  but does not promise bitwise-identical numerical results across machines.
- No cross-configuration baseline caching is implemented yet. Repeating baselines
  costs time but avoids unsafe sharing. Every job saves all three case outputs.

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
Summary statistics include count/mean/sample-standard-deviation/min/max for all
numeric metrics, grouped by configuration, case and device. **Advanced and
non-V2H are never averaged together.** `max_solve_seconds_max` is the worst
individual solve; the mean of per-run maxima is also retained but is not that maximum.
Charge/discharge starts are distinct from throughput equivalent full cycles.

`figures/` adds per-mobility-case cross-formulation overlays (availability, SOC,
price, EV power/mode, whole-home power, stored-energy change) and paired savings
bar charts with per-case dots and sample standard deviations. Advanced and
non-V2H have separate figures. Existing three-case plots are unchanged. Overlays
show completion coverage; savings plots use the common completed case intersection.
Run-level `raw_results_all_phases.csv`, `controls_all_phases.csv`, solve/action
histories and `evaluation_mask.csv` retain padding for audit; comparison files
contain evaluation only. No plots imply that unexecuted staged jobs have results.

Net savings = (baseline operating cost - case operating cost) +
(baseline NMC reference degradation cost - case NMC reference degradation cost). Baseline net savings
is zero. Reported degradation is unweighted. Final SOC/stored energy, unserved
trips, health/comfort violations, starts, mean/p95/max solve time, detailed gaps,
timeouts and fallback history remain available. There is no terminal-energy
credit or silent exclusion of failed solves from completed simulations.

## Deferred optimizer work

Smoothed rainflow and hh are not implemented or validated. They are outside this
revision's remaining scope unless requested again.

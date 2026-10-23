# Validation and known limitations

[Guide index](../../README.md)

The repository-wide suite is **not currently green**. This index replaces five
overlapping historical validation reports; their committed versions remain in Git.
Current formulations are documented in the pipeline, scenario and battery guides.

## Recorded checks and open discrepancies

- October 23 conditional-prefix revision: 31 scenario/reporting tests passed,
  including the three-case simulation and convex/MILP solver fixtures.
- October 23 optional-twin revision: 50/51 broader checks passed, including
  two-/three-case simulations, config loading, baseline caching and plots.
  Two additional two-case study aggregation/retrofit checks passed.
- Six supplied control/study JSON documents loaded and validated after setting
  `run_non_v2h_twin=false`. Older frozen snapshots were not rewritten.
- Historical October 17 full discovery recorded 126 legacy errors and eight
  terminal assertions. That is historical evidence, not a current full-suite count.

Known mismatches retained rather than silently changed/skipped:

1. Older terminal tests expect daily mean tariff; runtime intentionally uses minimum.
2. A departure-energy test expects departure-only comfort; current user-edited OCP
   penalizes the soft SOC floor throughout the horizon. Departure metrics remain pre-debit.
3. The progress test expects phase/final callbacks to print; current StudyProgress
   deliberately suppresses per-step/phase messages.

This push-preparation pass changes documentation/comments and removes redundant
notes, not formulations or test expectations. No full study or full discovery is
rerun as part of this budget-conscious cleanup. Four baseline-cache and six
retrofit-plot tests passed. Lightweight check results are in
[PUSH_README.md](../../../../PUSH_README.md).

## Reproduce

From the repository root, using the existing environment:

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) '.test-tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest test.test_ev_scenario test.test_ev_departure_energy test.test_ev_reporting test.test_ev_soh test.test_ev_degradation test.test_unified_hems test.test_unified_controls test.test_unified_warmup test.test_ev_study test.test_ev_baseline_cache test.test_ev_retrofit_plot -q
.\.venv\Scripts\python.exe -m unittest discover -s test -q
```

Tests use synthetic data and short fixtures; passing them is not a full study
benchmark or measured validation of the battery aging assumptions.
Solver availability/licensing and referenced local inputs affect integration checks.
Preserve failure logs when reporting counts. Do not classify native size-limited
Gurobi error 10010 as mathematical infeasibility.

No HVAC MPC, coordinated multi-home OCP, scenario tree beyond first-action sharing,
hh/smoothed-rainflow optimizer, temperature/calendar aging, or complete simulation
restart is claimed. Battery checkpoints alone cannot resume the whole dwelling.

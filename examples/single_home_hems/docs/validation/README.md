# Validation and known limitations

[Guide index](../../README.md)

The repository-wide suite is **not currently green**. Historical reports describe
what was actually run at their dates; they are not fresh claims about this checkout.

- [Scenario validation, October 17](ev_scenario_2026-10-17.md): 21 scenario tests
  passed; full discovery recorded 126 legacy errors and eight terminal assertions
  across three tests. Full discovery preceded four final scenario additions.
- [Dynamic SOH validation](ev_soh.md)
- [Earlier EV/V2H validation](ev_v2h.md)
- [Earlier unified validation](unified.md)
- [Cleanup validation](cleanup_2026-10-17.md)

Relative paths inside historical reports refer to their original example folder
or the repository root. The legacy terminal tests expect daily mean tariff;
the approved runtime behavior is daily minimum. Those expectations are deliberately
not changed or skipped in this cleanup. Old reports can describe superseded economics.

## Reproduce

From the repository root, using the existing environment:

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) '.test-tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest test.test_ev_scenario test.test_ev_soh test.test_ev_degradation test.test_unified_hems test.test_unified_controls test.test_unified_warmup test.test_ev_study -q
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

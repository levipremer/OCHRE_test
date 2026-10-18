# EV scenario validation — October 17, 2026

User approved finalization October 17 with the documented test/license limitations.
Finalization updated records only; no additional code changes or simulations.
No research-duration study or Git commit. Existing datasets, physical battery settings, active forecast
mode, HPWH controller and native OCHRE physics were not changed.

## Executed checks

| Check | Result |
| --- | --- |
| Final `python -m unittest test.test_ev_scenario -q` | **21 tests passed**, 14.574 s |
| Existing EV/unified + HPWH scenario/forecast group | 168 tests; 8 terminal-value assertion failures across 3 test methods; no other failures/errors |
| Full `python -m unittest discover -s test -q` | 410 tests, 174.725 s; 126 legacy errors and the same 8 terminal-value assertions |
| Five-path, 12-hour Audi MILP, 60-kWh fixture | SCIPY optimal under a 5-s requested solver limit |
| Same full-size model, installed Gurobi | Restricted-license failure; native error 10010 reproduced |
| Compile checks and tracked diff whitespace checks | Passed |

Full discovery included the first 17 scenario tests. Four additional tests and
the final reporting/preflight refinements were subsequently covered by the final
21-test scenario run. Full suite log: `.test-tmp/ev_scenario_full_suite.log`.
An ad hoc subset rerun without workspace-local TEMP hit a sandbox temporary-file
permission error. The new fixtures now explicitly use repository-local temporary
directories; the final complete scenario rerun passed afterward.

## Scenario coverage

- Default five-path kNN, all seven filter/hierarchical alternatives, repeatability,
  full historical coverage, completed paths, non-overlap and forecast-length handling.
- Actual Audi data with 31-day reserve and seeded interior replay; canonical
  distance-based inputs remain shared with NHTS. Existing Audi/NHTS adapter tests
  also ran in the broader regression group.
- No future-trip-distance leakage; prefix home/away behavior; delayed/compounded
  arrivals, exact prefix boundaries and beyond-horizon deferred distance.
- Fewer valid paths, no-path fallback, invalid settings, history reservation.
- Single-scenario equivalence; mean non-comfort terms versus maximum whole-path
  comfort; shared first command across mobility-event splits.
- MILP and convex, optional caps/dwell/persistence/discharge budget/terminal cost,
  measured faded capacity, and no hypothetical updates of realized aging history.
- CLARABEL/SCIPY plus small Gurobi fixtures; Gurobi exercised both formulations
  with none, linear-throughput and rainflow costs.
- Per-path rainflow certificates; iteration-limit and infeasibility fallback.
- Real 16-minute OCHRE three-case fixture: baseline/twin/advanced outputs,
  per-path plans/forecasts, two scheduled EV solves in each scored MPC case,
  no mobility-triggered solves, SOC bounds and actual-load discharge caps.

## Explicitly retained discrepancies/limits

1. The terminal function already returned `prices.min()`, while its older tests
   and documentation expected the daily mean. User explicitly requested preserving
   the minimum and documenting the discrepancy. The calculation is unchanged;
   its docstring/warning now describe the minimum. Old tests were not rewritten
   or suppressed. A new scenario test verifies the preserved minimum. Eight
   failed assertions arise from mean-price values and a charging-credit expectation
   that is no longer guaranteed when off-peak tariff equals terminal value.
2. Full discovery still has 126 legacy API/fixture errors, matching the previously
   documented error count/categories in the workflow checkpoint. No errors were
   reported in the EV scenario or unified modules. This repository-wide suite is
   **not green**; unrelated legacy repairs are outside this revision.
3. Local Gurobi is size-limited. The default-size fixture had 2,165 scalar variables,
   1,213 equalities and 1,930 inequalities. The same model solved with SCIPY;
   larger Gurobi runs need a suitable license. No backend was silently substituted
   in pipeline configuration. Direct diagnostic solves do not benchmark full MPC.
4. The model is two-stage (first command shared), not a full scenario tree.
   Non-overlap reduces redundancy, not statistical dependence. Known prices and
   twin backed loads remain shared pending future aggregated-HEMS optimization.
5. Native solver budgets do not constitute a process-level canonicalization
   watchdog. No real 300-second timeout stress test or long scenario study was run.

HPWH controller SHA256 remains:
`DFDAE01C6E79B150BB613A22468E61A7B55D2943598AAE8773E8353C9C2EE927`.
<!-- Moved during the October 17 documentation cleanup; paths in historical
     validation records refer to the repository or original example directory. -->


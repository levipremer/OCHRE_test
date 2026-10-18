# Dynamic EV SOH — implementation and validation, 2026-10-16

Status: finalized with explicit user approval on October 16. Approval recorded;
no additional simulations, code changes or Git commit during finalization.
No full parametric study has been launched. HPWH controls/settings are unchanged.

## Audit and resolution

| Location | Previous issue | Implemented correction |
| --- | --- | --- |
| `five_home_block/ev_equipment.py` | Constant capacity; no health state or damage feedback | Initial SOH, continuous damage ledger, dynamic capacity, transactional preview/commit aging |
| Plant charging, V2H, trips and fallback | Energy/SOC conversion used nominal capacity | Start-of-action faded capacity, bounded SOC, explicit fade-energy loss |
| `five_home_block/ev_degradation.py` | Batch damage priced but not propagated | Streaming four-point residue/closed damage; normalized D, physical SOH and dollar mapping |
| `controllers/ev.py` | Nominal prediction capacity; no SOH measurement | Fresh measured SOH, frozen per horizon; current-capacity throughput coefficient and recorded action-time discharge capacity |
| `simulation.py` | Controller status omitted health | Expose post-arrival SOH/capacity; save battery-only checkpoint |
| `ev_simulation.py` | Static stored energy; scored path re-count lost pre-window cycle context | Dynamic energy and scored ledger increments; SOH/damage/capacity/EOL metrics |
| Plotting/configuration | No physical initial SOH or health trajectories | Explicit home field; health figures and dynamic stored-energy overlays |
| Economic conversion | Already algebraically consistent after prior calibration/EOL split | Preserve mapping; make damage units explicit; do not reprice faded pack or bill initial damage |

The reference rainflow ledger always controls physical aging. Turning the OCP aging
term off does not turn physical aging off. Depth-only rainflow/throughput remain
reduced-order prediction costs. No temperature, knee, calendar or HPWH aging added.

## Approved conventions

- Nominal capacity C0; current usable capacity C=C0*SOH.
- alpha=1-SOH_cal=.2; D0=(1-SOH_initial)/alpha=.75 for initial SOH .85.
- SOH=1-alpha*D; budget A=(1-SOH_econ)/alpha=1.5 at economic SOH .7.
- R=C0*(119-0); dollar increment=(R/A)*delta_D. No initial-damage charge.
- After energy exchange, fade preserves SOC and removes accessible energy
  SOC_end*(C_before-C_after), recorded separately without another dollar charge.
- Economic EOL flags only; nonpositive physical SOH raises an error.
- Warmup/cooldown age normally. Scored costs sum only scored increments with
  continuous pre-window residue. No isolated-window rainflow recount.
- The checkpoint restores battery state/residue, not dwelling or MPC restart state.

## Tests

New `test/test_ev_soh.py` covers calibration versus economics, batch/streaming
agreement at every random-trace prefix, checkpoint continuation, stress/health
validation, exact-once plant aging, trip/action energy conservation, disabled-cost
aging, idle/EOL behavior, warmup residue accounting, both MPC formulations and
all three cost options, live SOH refresh, historical discharge capacity, and
physical exhaustion. Existing tests now supply required measured SOH and check
dynamic stored energy rather than nominal-capacity energy.

Focused regression run: **142 tests passed** in 178.223 seconds, covering EV
formulation/degradation/plant/SOH, unified configuration/control/warmup, study
driver, HPWH cutoff/scenario compatibility, both mobility sources and multi-home
integration. Log: `.test-tmp/dynamic_soh_focused_suite.log`.

Full repository discovery: 384 tests attempted; 126 errors in legacy/native suites
(64 equipment, 47 model, 10 utility errors and 5 module-import errors). Examples:
removed `ochre.FileIO` / `add_equipment_from_properties`, obsolete constructor
arguments and dictionary schedules, missing legacy `parametric_study.analysis`.
These are outside the EV changes and were not repaired. One additional warmup
assertion required strict positive excluded damage even for an idle EV; corrected
to allow zero and rerun successfully. The full-suite process had loaded the older
assertion before its correction. Full log: `.test-tmp/dynamic_soh_full_suite.log`.

## Real driver validation

`outputs/ev_soh_validation_20261016_final`: two Gurobi configurations (convex/MILP),
each with all-baseline, non-V2H twin and advanced cases; 15-minute warmup,
30-minute scoring, 15-minute cooldown, plus twin forecast tail.

- Both jobs and all six case simulations completed; zero failed solves across all phases.
- Initial SOH .85 and capacity 51 kWh for the 60-kWh pack in every case.
- Fresh solve-history SOH matches plant measurements at each solve timestamp.
- Scored dollar costs equal the scored ledger increments; masks exclude pre/post-roll.
- Per-step conservation error, including fade loss, <= 7.96e-15 kWh.
- Checkpoints and health plots exist for every case; comparison plot inspected.
- Repeating the driver command reused both completed jobs (no new simulations).
- `reports/run_report.md` within each job consolidates all metrics/solver diagnostics.

An earlier smoke snapshot without the final report-wording update is retained in
`outputs/ev_soh_validation_20261016`; it is not the final validation artifact.
Original study results and prior diagnostic artifacts remain untouched.

HPWH controller SHA256 before/after:
`DFDAE01C6E79B150BB613A22468E61A7B55D2943598AAE8773E8353C9C2EE927`.

## Next run

The supplied pilot config targets fresh `outputs/ev30_dynamic_soh_v4`, with initial
SOH inherited from the home document. It has NOT been staged or run. Do not resume
an old static-SOH study under this changed model. Existing source/input fingerprints
continue to protect against mixing incompatible experiments.
<!-- Moved during the October 17 documentation cleanup; paths in historical
     validation records refer to the repository or original example directory. -->


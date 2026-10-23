# HEMS push notes — October 23, 2026

Baseline: `0dd3cfa` (local `HEAD` and `origin/main` at preparation),
"Added metrics and parametric study config". No remote fetch, commit or push was
performed. These notes describe the pending working-tree changes, not new study results.

## Pertinent changes since that push

- **Departure energy:** plant and EV OCP debit distance-derived demand at departure,
  not arrival. Ongoing trips are already reflected in measured SOC; arrival restores
  availability only. Preserve pre-debit SOC for departure metrics and both sides
  of the impulse for rainflow. Plant rejects charging/discharging while away.
- **Trip protection:** bounded unmet-energy slack with cost
  `trip_weight * sum(unmet_kWh)/(current_capacity_kWh * new_trip_count)`;
  zero with no new trips. Scenario MPC minimizes separate worst-path comfort and
  trip costs, averaging all other costs. Physical unmet driving is reported separately.
- **Current comfort:** all-time soft SOC floor, including away periods, with dt/N
  normalization. This differs from older departure-only studies. Terminal valuation
  remains `v*(e0-eN)/N`, using the daily-minimum tariff directly per chemical kWh.
- **Conditional prefix:** correct only selected scenarios whose current availability
  disagrees with reality. Matching paths keep all near-term events. On corrected
  home paths discard enclosed trips and shift crossing departures with full distance;
  corrected away paths add no second debit. Diagnostics expose applied/ignored/shifted events.
- **Optional twin:** supplied controls/template/study definitions explicitly set
  `run_non_v2h_twin=false`. Run baseline + advanced; baseline realized end uses supply
  the extended backed-load forecast. True enables the configured non-V2H twin instead.
  Omission preserves legacy true behavior. Skipped cases have no invented output rows.
- **Studies and runtime:** Cartesian scenario-count/selector and horizon/bank studies,
  paired frozen mobility, optional integrity-checked baseline reuse, Windows long-path
  handling and bounded publication retries. Batched scoring and horizon preparation.
  Job-level progress; no detailed step-percentage display or parallel execution.
- **Reporting:** departure-energy forecast scores, predicted/actual trip shortfalls,
  timing/gap/fallback diagnostics, horizon/bank plots and a read-only retrofit SOC/mode
  plotter. Baseline-relative net savings and energy-adjusted savings remain separate;
  twin and advanced aggregates stay separate. Max solve time is an absolute maximum.
- **Solver safeguards:** quiet 300-second native limits; scenario rainflow masters
  share one OCP budget. EV acceptance checks feasibility/integrality and MIP gap;
  rainflow additionally certifies each path's aging approximation. Not a process watchdog.

Dynamic reference SOH, faded-capacity SOC, shared NMC calibration/economic damage
pricing, and unscored warmup/cooldown continue unchanged. No HVAC MPC or coordinated
HEMS optimization is added. HPWH equations and native OCHRE physics are unchanged
by this cleanup.

## Navigate / run

- [Quick start](examples/single_home_hems/README.md), [pipeline](examples/single_home_hems/docs/pipeline.md),
  [formulations](examples/single_home_hems/docs/degradation.md), [scenarios](examples/single_home_hems/docs/scenarios.md).
- [Studies](examples/single_home_hems/docs/studies.md): stage without `--run`;
  choose a fresh output directory for these changed inputs/formulations.
- `tools/plot_ev_trajectories.py`: cleaner SOC and actual-mode plot with price overlay;
  supports runs with or without the twin and writes outside frozen output directories.

## Cleanup and release caveats

- Consolidated five dated validation reports into the [validation index](examples/single_home_hems/docs/validation/README.md).
  Removed obsolete `WORKFLOW_CHECKPOINT.md`. Local recovery copies, including
  uncommitted report edits, are in ignored `.test-tmp/push_cleanup_recovery_20261023/`.
- Retained regression tests, model/input files, archives, reusable tools and
  parametric-study outputs. Removed one-off diagnostic/validation outputs and
  revision-locked maintenance artifacts. Generated outputs are ignored push content.
- Documentation now distinguishes current all-time comfort, conditional persistence,
  two-case example defaults and job-only progress from historical behavior.
- Tests have known terminal/comfort/progress expectation mismatches and legacy
  failures; do not claim the full suite is green. No test was deleted to hide failures.
- Let the active study finish before commit/push. Do not pool results across source
  revisions or change frozen manifests. Existing processes do not reload edited code.
- Supplied study output paths may already exist locally: specify a **new** output
  directory for restaging. Preserve the original checkout/configs to resume old cohorts.
- Do not commit solver licenses, secrets, virtual environments, generated results
  or private/restricted inputs. Confirm dataset redistribution permissions separately.

## Checks for this preparation pass

- All single-home JSON configurations parsed successfully.
- Updated Python sources compiled; documentation links and diff whitespace passed.
- Four baseline-cache and six retrofit-plot tests passed.
- The focused departure-energy group retains its documented all-time-comfort
  expectation mismatch; no formulation was changed to make an old assertion pass.

No production simulation, full-suite claim, commit or push was performed.

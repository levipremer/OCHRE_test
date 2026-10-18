# EV/V2H validation checkpoint

Validation date: 2026-10-06. Implementation finalized with user approval on
2026-10-06, accepting the limitations below.
Historical formulation: [archived EV/V2H guide](../../../../archive/development_docs_2026-10-17/EV_V2H.md).
Change log: [CUSTOM_CHANGES.md](../../../../CUSTOM_CHANGES.md).

## Regression tests

Repository discovery (`python -B -m unittest discover -s test`) executed 284
tests: **158 passed, 126 errors, zero assertion failures**. All 29 new tests in
`test_ev_v2h.py` and `test_ev_v2h_integration.py` passed, including the five-home
integration and combined EV/HPWH MPC checks. The full suite is **not green**.

The remaining errors are existing legacy API/fixture incompatibilities:

| Error group | Count |
| --- | ---: |
| Missing `add_equipment_from_properties` / `FileIO` imports | 2 |
| Envelope fixtures omit `zones` | 16 |
| Equipment schedule fixtures use dictionaries instead of indexed data | 27 |
| Native EV fixtures omit ambient temperature | 7 |
| PV fixtures omit tilt/azimuth | 2 |
| Scheduled-load input shorter than required duration | 5 |
| Water-heater fixtures omit `Tank Volume (L)` | 37 |
| Humidity fixtures use obsolete `t_indoor` argument | 2 |
| Simulator fixtures omit start time/duration | 17 |
| Missing legacy `parametric_study.analysis` module | 1 |
| Undefined legacy test names (`FileIO`, `properties`, `resample_args1`) | 10 |

Native `ochre/` implementation and HPWH controller mathematics were not modified.
Legacy repairs are outside this EV implementation's scope.

## Real-OCHRE 72-hour checks

Both runs use 4,320 one-minute plant steps, 12-hour forecast horizons, and paired
immediate-charge baselines. The advance thermal forecasts match the controlled
HVAC/WH loads. These are functional checks, not tuned research comparisons.

| Metric | Audi / convex / CLARABEL | NHTS / MILP / SCIPY |
| --- | ---: | ---: |
| OCP invocations | 288 | 291 |
| Solver failures | 0 | 0 |
| Mean solver-call time (s/invocation) | 0.048795 | 0.255937 |
| Total solver time / plant steps (s/step) | 0.003253 | 0.017240 |
| Mean OCP construction + solve time (s) | 0.070230 | 0.290724 |
| V2H energy delivered (kWh) | 23.621827 | 26.840812 |
| Maximum cumulative battery energy-balance error (kWh) | 1.21e-13 | 1.03e-13 |
| Actual insufficient-energy departures | 0 | 0 |
| Final controlled SOC | 0.209925 | 0.200032 |
| Final baseline SOC | 1.000000 | 1.000000 |

The NHTS run respects the configured four-start rolling 24-hour caps (observed
maxima: one charging start, four discharging starts). The Audi example has no
start caps. Its minimum SOC, 0.197837, is below the soft 0.2 health threshold but
within the hard physical bounds. The maximum observed convex shortfall relaxation
gap was 2.78e-11 kWh; a separate unit test exercises a nonzero relaxation gap.

Saved run directories, relative to `examples/single_home_hems/`:

- `outputs/ev_v2h_validation/audi_convex_reviewed`
- `outputs/ev_v2h_validation/nhts_milp_reviewed`

Each contains comparison metrics, raw results, solver histories, trip/replay
metadata, and figures. Outputs are Git-ignored.

## Accepted limitations and finalization

- GUROBI configuration is supported, but execution remains unverified because
  the optional backend/license is not installed in this environment.
- Baseline and controlled final SOC differ substantially. Raw cost/energy
  differences are not equal-terminal-energy savings estimates.
- Electricity cost is optimized, not peak demand directly; the Audi run's
  household peak increased slightly despite lower energy/cost.
- Convex trip-shortfall relaxation and discrete projection retain the documented
  approximation limitations. Physical SOC and trip accounting remain exact.
- User approved finalization with these limitations. No changes have been
  committed or staged; finalization changed documentation only.
<!-- Moved during the October 17 documentation cleanup; paths in historical
     validation records refer to the repository or original example directory. -->


# Changes from base OCHRE

This repository is a work fork of the Object-oriented Controllable
High-resolution Residential Energy Model (OCHRE). The custom work starts after
base commit `458b9b2`. That commit is the most useful comparison point for the
work in this fork; the fork has not yet been rebased onto newer OCHRE releases.

## HVAC model extensions

The main OCHRE engine changes are in `ochre/Equipment/HVAC.py`.

- Added an optional discrete reverse-cycle defrost model while retaining the
  original time-averaged behavior as the default `Legacy` model.
- Added timer-based and demand-based discrete defrost controls, including
  configurable duration, outdoor-temperature threshold, initial accumulation,
  heat-extraction multiplier, and compressor-power multiplier.
- Added aggressive and conservative electric-resistance backup strategies
  during discrete defrost.
- Added automatic or explicitly configured 5/10 kW resistance-backup stages.
- Added thermostat-based stage escalation and release behavior.
- Added compressor minimum-on and minimum-off timing that remains independent
  of resistance-backup dispatch.
- Added detailed defrost, compressor, resistance-capacity, and stage outputs at
  higher verbosity levels.

The new configuration fields and model behavior are described in
`docs/source/InputsAndArguments.rst` and `docs/source/ModelingApproach.rst`.

## System identification and predictive control

The notebooks under `bin/` contain the earlier exploration and control work:

- dwelling and RC/HVAC model exploration;
- HVAC system identification, including Hankel and ARMAX experiments;
- uncontrolled baseline simulation;
- model predictive control (MPC);
- time-of-use MPC; and
- data-enabled predictive control (DeePC).

The control notebooks use `cvxpy`, which is not required by the base OCHRE
package and must be installed separately.

## Five-home neighborhood example

`examples/five_home_block/` adds a reproducible five-home simulation using five
distinct Indiana ResStock HPXML/schedule inputs and shared weather.

- Runs all homes on a shared one-minute clock.
- Standardizes version-specific OCHRE output names.
- Checks individual-home and neighborhood power balances.
- Aggregates electric load and produces per-home and neighborhood figures.
- Includes a deliberately simple static transformer-capacity model.
- Includes a seasonal electrification parametric study that caches individual
  home profiles and reconstructs valid neighborhood scenarios.

Only the active HPXML files, schedules, and weather file are included. Generated
simulation outputs are intentionally ignored.

## Single-home HEMS and planned EV work

`examples/single_home_hems/` provides a stepwise control interface for any home
from the five-home cohort. The included controller is a pass-through baseline;
`hems.py` is the intended extension point for HVAC, water-heater, and EV control.

The current EV assumptions use fixed battery size, charger power, arrival and
departure time, and arrival state of charge for each home. The proposed next
steps for probabilistic battery sizing and travel schedules are captured in
`docs/decisions/EV_simulation_decision_outline.docx`.

## Tests

The added tests cover:

- timestep-independent discrete-defrost state behavior;
- electric-resistance stage configuration and dispatch;
- five-home configuration, output mapping, aggregation, and plotting;
- seasonal parametric-study scenario generation and reconstruction; and
- single-home HEMS selection and control validation.

Run the focused tests from the repository root with:

```bash
.venv/bin/python -m pytest \
  test/test_equipment/test_defrost.py \
  test/test_equipment/test_staged_er.py \
  test/test_five_home_block.py \
  test/test_parametric_study.py \
  test/test_single_home_hems.py
```

## Known limitations

- These extensions have not been accepted or validated by the upstream OCHRE
  maintainers.
- The discrete-defrost and backup-staging models are research-oriented and
  should be checked against the intended equipment assumptions before being
  treated as calibrated equipment models.
- The neighborhood transformer is static; it does not model temperature,
  voltage, losses, aging, or time-dependent overload behavior.
- Notebook outputs reflect exploratory development and are not a formal
  benchmark suite.
- The focused tests listed above pass. The older repository-wide test suite
  contains pre-existing failures from outdated APIs and fixtures and is not the
  verification target for this handoff.
- Compatibility with OCHRE versions newer than the pinned base has not yet
  been evaluated.

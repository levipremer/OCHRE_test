# Predictive bidirectional EV and joint HEMS

These examples run the bidirectional EV charger by itself or coordinate the EV,
HVAC, and heat-pump water heater in one single-home OCHRE simulation. Run commands
from the repository root. The two controller entry points are
`examples/single_home_hems/run_ev_home_hub.py` and
`examples/single_home_hems/run_whole_home_joint.py`.

The input bundle at `examples/single_home_hems/inputs/predictive/` contains the
prepared weather, frozen load forecasts, RF artifacts, the active HVAC
state-space predictor, and pretest seeds used by these runners. `MANIFEST.json`
records their provenance and SHA-256 hashes. The raw DC House meters,
rest-of-home load, water draw, Audi mobility, EV battery reference, MISO price
extract, active HPXML/schedules, and EPW weather file stay at their existing
`external_data/`, `models/`, or `examples/five_home_block/inputs/` paths. A
clean checkout needs every file named in the manifest.

## Environment

Python 3.12, the local OCHRE package, the example requirements, Gurobi's Python
package, and a working Gurobi license are required for these predictive MILPs.
The solver package is required even to import the current fast MILP module.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . -r examples/single_home_hems/requirements.txt
.venv/bin/python -m pip install gurobipy
.venv/bin/python -c 'import cvxpy, gurobipy; print(cvxpy.installed_solvers())'
```

The original OCHRE checkout supplies the plant model. Keep the complete
repository together so the selected home configuration can resolve its
HPXML, schedule, weather, mobility, and model files. Verify every packaged
input against the manifest before running a controller:

```bash
.venv/bin/python examples/single_home_hems/verify_predictive_inputs.py
```

## EV-only input check

The EV-only runner controls the charger while native OCHRE controls HVAC and
water heating. The Home Hub RF uses the completed aggregate and EV measurements;
the Outside-Hub RF uses completed total-home minus corrected Home Hub
measurements, including HVAC electric resistance. The following short input
check uses the 80 kWh ideal vehicle configuration but does not run the
twelve-case comparison:

```bash
.venv/bin/python examples/single_home_hems/run_ev_home_hub.py \
  --start-time 2025-11-30T00:00:00 \
  --source-start 2026-10-04T00:00:00 \
  --duration-hours 0.25 \
  --native-warmup-hours 24 \
  --mobility-source-window-hours 168 \
  --rate-selection-duration-hours 168 \
  --water-draw-start 2025-11-16T00:00:00 \
  --source-config-directory examples/single_home_hems/config/oikolab_2025_2026 \
  --ev-home-config examples/single_home_hems/config/unified_home_sunday_ideal_80kwh_audi.json \
  --hub-forecast-mode corrected_rf \
  --hub-rf-artifact examples/single_home_hems/inputs/predictive/models/eligible_hub_rf.joblib \
  --hub-seed-csv examples/single_home_hems/inputs/predictive/seeds/home_hub_meter_15min.csv \
  --outside-forecast-mode rf_multires_weather_persistence \
  --outside-rf-artifact examples/single_home_hems/inputs/predictive/models/outside_hub_er_multires_rf.joblib \
  --ev-scenario-count 5 \
  --ev-scenario-method tod_dow_state_hier_limit_persistence \
  --hub-mobility-split-policy hold_parent_interval_coefficients \
  --rate-structure tou-winter \
  --validate-only
```

For a one-decision simulation, remove `--validate-only` and add
`--output-directory` with a new empty directory under
`examples/single_home_hems/outputs/`. Check `ev_solve_history.csv` and
`ev_home_hub_settings.json` in that output before interpreting the result.

## Joint HEMS one-decision check

This direct runner uses seven HPWH paths. The historical 80 kWh comparison
runner used an explicit five-path override and is outside this package. The
example below selects the existing rolling-starts HPWH preset, five EV paths,
causal weather, split-Hub load forecasts, and Grid Scenario 3. October 5 gives
the direct runner a complete measured 24-hour warm-up. December 1 keeps
the MISO source available through the preceding warm-up day.

```bash
.venv/bin/python examples/single_home_hems/run_whole_home_joint.py \
  --start-time 2025-12-01T00:00:00 \
  --duration-hours 0.25 \
  --rest-source-start 2026-10-05T00:00:00 \
  --water-draw-start 2025-11-16T00:00:00 \
  --source-config-directory examples/single_home_hems/config/oikolab_2025_2026 \
  --ev-home-config examples/single_home_hems/config/unified_home_sunday_ideal_80kwh_audi.json \
  --split-hub-forecast \
  --hub-other-source examples/single_home_hems/inputs/predictive/forecasts/hub_other_forecast_source.csv \
  --outside-hub-source examples/single_home_hems/inputs/predictive/forecasts/outside_hub_forecast_source.csv \
  --ev-scenario-count 5 \
  --ev-scenario-method tod_dow_state_hier_limit_persistence \
  --hpwh-variant rolling-starts \
  --weather-forecast-mode predict \
  --rate-structure rtp-miso+doe \
  --time-limit-seconds 300 \
  --mip-gap 0.05 \
  --solver-threads 1 \
  --output-directory examples/single_home_hems/outputs/joint_predictive_one_decision
```

Use a fresh output directory for each run. Check
`data/whole_home_joint_solve_history.csv` and
`whole_home_joint_settings.json` for solve success and effective settings.

For a new weekly study starting Sunday 2025-11-30 at 00:00 local time, the
approved evaluation endpoint is Saturday 2025-12-06 at 12:00, exclusive:
`--duration-hours 156`. Keep the full eight-hour forecast horizon and its
source-data tail at the last issue. Short checks such as those above retain
their explicitly requested duration.

# Standalone HVAC MPC: Grid Scenario 1, 20-hour horizon

This is the November 30–December 2, 2025, `home_01` study. Run it from the
repository root with the code on this branch. Only HVAC heating and cooling
receive predictive commands; water heating, EV charging, and other equipment
retain native operation. The simulation also runs a native HVAC reference over
the same 48-hour window. `run.py` writes a manifest, minute-level time series,
optimization traces, and figures. `analyze.py` computes the performance report.

The controller uses the trained regularized state-space predictor in `model/`,
mixed-integer control with Gurobi, 15-minute decisions, the winter Grid 1 TOU
tariff, and an archived 00Z GFS weather forecast. The 20-hour forecast has 80
intervals at every issue, including the last one, whose tail extends beyond
the evaluation endpoint. The controller has a 2 °C·h comfort budget over the
previous four measured plus 20 forecast hours, scheduled setpoint ±1.5 °C
comfort bounds, a 1 °C per-step deviation cap, persistence weight 0.1,
degree-hour slack weight 100, and deviation-cap slack weight 10,000. The L2
comfort term, cycle limit, and dwell constraints are off. The original
scheduled setpoints are 20.5 °C (21:00–06:00), 22.5 °C (06:00–09:00),
18 °C (09:00–17:00), and 22.5 °C (17:00–21:00). The later requested
21/22/20/22/21 °C schedule and ±0.5 °C thermostat deadband are a separate,
unfinished study and are not used here.

## Install and supply inputs

Use Python 3.12 from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . -r examples/single_home_hems/requirements.txt
.venv/bin/python -m pip install gurobipy
```

Configure a working Gurobi license locally; do not put license credentials in
Git. The exact study also needs these prepared input files, which are normally
kept outside Git:

| Input | Repository-relative location |
| --- | --- |
| Realized one-minute weather | `examples/single_home_hems/outputs/oikolab_weather/prepared/realized_1min.csv` |
| Archived 00Z GFS forecasts | `examples/single_home_hems/outputs/oikolab_weather/prepared/gfs_00z_forecasts.parquet` |
| DC House rest-of-home power | `examples/single_home_hems/external_data/dc_house_power/rest_of_home_load.csv` |

For weather, use `examples/single_home_hems/oikolab_weather_pipeline.py` with
your own `OIKOLAB_API_KEY`. A local range from 2025-11-28 through 2025-12-04
exclusive covers the 16-hour native warm-up, the 48-hour evaluation, and the
full forecast tail. See `examples/single_home_hems/OIKOLAB_WEATHER_PIPELINE.md`
for the `plan`, `fetch-realized`, `fetch-forecast`, and `prepare` commands.
Alternatively place already prepared files with the listed schemas at the paths
above. The model files are included in this directory; no training step is
needed for this benchmark.

For the DC House load, place the five meter inputs named in
`examples/single_home_hems/build_dc_house_rest_of_home.py` under
`examples/single_home_hems/external_data/dc_house_power/`, then run that script
to build `rest_of_home_load.csv`. Using a different load source or weather
changes the resulting performance.

## Run and verify

```bash
.venv/bin/python benchmarks/hvac_grid1_2day_20h_20261110/run.py
.venv/bin/python benchmarks/hvac_grid1_2day_20h_20261110/analyze.py
```

The runner refuses to overwrite an existing `results/` directory. Inspect
`results/study_manifest.json` for the evaluation window and completion status,
`results/data/hvac_run_settings.json` for the effective controller settings,
`results/data/hvac_optimization_solve_history.csv` for all 192 decisions, and
`results/performance.json` for the measured comparison. The final optimizer
plan ends on December 2 at 19:45 local time, confirming the 20-hour tail at
the final 23:45 issue.

## Integrate the controller into another run

Keep the controller, simulation, forecast, training, OCHRE plant, and
`five_home_block` changes from this branch together. The preexisting GitHub
`main` version of `controllers/hvac.py` is only a pass-through template. The
essential wiring is in `run.py`:

```python
controller = HVACPredictiveController(
    model_path=MODEL,
    horizon=pd.Timedelta(hours=20),
    model_timestep=pd.Timedelta(minutes=15),
    optimization_kind="mixed integer",
    solver="GUROBI",
    comfort_band_c=1.5,
    comfort_weight=0.0,
    persistence_weight=0.1,
    enforce_degree_hour_budget=True,
    enforce_instantaneous_cap=True,
    degree_hour_limit_c_hours=2.0,
    max_deviation_c=1.0,
    degree_hour_slack_weight=100.0,
    max_deviation_slack_weight=10_000.0,
    enforce_cycle_limit=False,
    enforce_dwell_constraints=False,
)
hems = HEMSController(
    device_controllers=(controller,), price_schedule=rate.schedule
)
simulation = SingleHomeHEMSSimulation(
    config, home, hems, weather_archive=weather,
    initial_temperature_c=21.0,
    hvac_warmup_duration=pd.Timedelta(hours=16),
)
summary = simulation.run()
```

`weather` must be an `ArchivedWeatherForecast` with complete coverage through
the last 20-hour plan. Keep `device_controllers` to this single HVAC controller
for HVAC-only operation. The original scheduled heating setpoint function in
`single_home_hems/training/hvac.py` supplies the endpoint comfort bounds and
the matched native reference. When adapting the run, confirm the effective
settings file before comparing bills or comfort results.

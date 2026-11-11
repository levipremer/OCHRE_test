# OikoLab weather acquisition for the single-home HEMS

`oikolab_weather_pipeline.py` acquires OikoLab ERA5 realized weather and one
archived GFS 00Z run for each UTC day overlapping a Lafayette simulation. It
prepares atmospheric weather for OCHRE and a separate, run-indexed forecast
table for HVAC MPC. OCHRE's EPW-derived ground and water-mains temperatures
are supplied by the hybrid plant integration.

Install the example environment and a Parquet engine (`pyarrow` is used in the
project's `.venv`). Set `OIKOLAB_API_KEY` as an environment variable. The key is
sent in the `api-key` HTTP header and is never saved in raw records, the
prepared data, or the manifest.

```bash
PY=.venv/bin/python
SCRIPT=examples/single_home_hems/oikolab_weather_pipeline.py
WEATHER_START_LOCAL=2025-10-30
WEATHER_END_LOCAL=2026-04-03
$PY "$SCRIPT" plan --start-local "$WEATHER_START_LOCAL" --end-local-exclusive "$WEATHER_END_LOCAL"
$PY "$SCRIPT" account
$PY "$SCRIPT" fetch-realized --start-local "$WEATHER_START_LOCAL" --end-local-exclusive "$WEATHER_END_LOCAL" --execute
$PY "$SCRIPT" fetch-forecast --start-local "$WEATHER_START_LOCAL" --end-local-exclusive "$WEATHER_END_LOCAL" --max-new-runs 1 --execute
$PY "$SCRIPT" account
# Continue the archive and record cycles that OikoLab explicitly marks unavailable.
$PY "$SCRIPT" fetch-forecast --start-local "$WEATHER_START_LOCAL" --end-local-exclusive "$WEATHER_END_LOCAL" --skip-unavailable --execute
# Prepare accepts only gaps explicitly recorded as unavailable by that fetch.
$PY "$SCRIPT" prepare --start-local "$WEATHER_START_LOCAL" --end-local-exclusive "$WEATHER_END_LOCAL"
```

Fetch commands only show a plan unless `--execute` is present. They skip
already saved responses; `--force` explicitly re-fetches and may consume more
data units. The evaluation interval of November 1 through March 31 needs
weather on both sides: the example starts October 30 and ends April 3 to
cover the single-home HVAC history, a 24-hour unified warm-up and cooldown,
and forecast/twin tails of up to 12 hours. Increase these buffers if your
configured horizons are longer. This padded interval requires **156** distinct
00Z runs. Five forecast variables give a base estimate of **780 units**;
month-crossing runs can cost extra units. Check the account endpoint after
staged requests. The realized request has eight variables and spans six
calendar months.
By default, any unavailable archived cycle stops fetching. With
`--skip-unavailable`, only HTTP 400 responses whose title is exactly
`UTC cycle not available` are skipped. The tool writes
`reports/gfs_00z_unavailable_cycles.json`, listing those UTC runs and any runs
not attempted because of `--max-new-runs`. Other API errors still stop the
fetch. `prepare` accepts only missing runs that the report identifies as
unavailable for the same interval and location. It rejects any run that was
not attempted or is missing without that API response. On November 23 and
December 6, 2025, OikoLab reported the 00Z cycle unavailable. HVAC decisions
on those UTC dates use the latest earlier available OikoLab 00Z run; the
selected run supplies the entire prediction horizon. The prepared manifest
records these substitutions. The controller switches to a new run when its
00Z cycle becomes available, and never splices runs within a horizon.

To select the realized weather explicitly, put these fields in the source
`neighborhood.json` used by `--source-config-directory` (alongside its existing
home and simulation settings):

```json
{
  "weather_file": "examples/single_home_hems/outputs/oikolab_weather/prepared/realized_1min.csv",
  "weather_metadata": {
    "latitude": 40.430930,
    "longitude": -86.911617,
    "timezone": "America/Indiana/Indianapolis"
  },
  "ground_mains_epw_file": "examples/five_home_block/inputs/weather/G1801570.epw",
  "weather_interpolation": "linear",
  "weather_forecast_file": "examples/single_home_hems/outputs/oikolab_weather/prepared/gfs_00z_forecasts.parquet"
}
```

The plant reads `weather_file` in both modes. The EPW supplies OCHRE's
existing ground and mains temperature calculations. The CLI defaults to
`--weather-forecast-mode perfect`; use `--weather-forecast-mode predict` for
HVAC MPC to read the specified Parquet archive. You can override its path with
`--weather-forecast-file`. The water-use `--forecast-type` option is separate.

The default data directory is
`examples/single_home_hems/outputs/oikolab_weather` (ignored by Git). Override
it with `--data-dir` before the subcommand if desired. Layout:

```text
raw/realized/era5_START_END.json
raw/gfs_00z/gfs_YYYYMMDDT0000Z.json
prepared/realized_1min.csv
prepared/gfs_00z_forecasts.parquet
prepared/manifest.json
reports/gfs_00z_unavailable_cycles.json  # only with --skip-unavailable
```

Raw JSON records contain the original OikoLab response and nonsecret request
metadata. `realized_1min.csv` has `Time` plus OCHRE's raw input names
`temp_air`, `relative_humidity`, `pressure`, `ghi`, `dni`, `dhi`, `wind_speed`,
and `sky_temperature`. `Time` is a local ISO timestamp with its actual UTC
offset, so the repeated autumn hour remains unambiguous. Minute values are
linearly interpolated on the UTC timeline. OCHRE must parse `Time` as UTC first
and then convert to `America/Indiana/Indianapolis` to preserve both DST hours.

The Parquet file has one row per `(run_time_utc, valid_time_utc)` with
`valid_time_local` and normalized `temp_air`, `relative_humidity`, `ghi`, `dni`,
and `dhi`. Both UTC columns are timezone-aware. OikoLab's first valid value is
at run +1 hour; the archive contains hourly values through +120 hours, then
three-hourly values through +384 hours. No forecast is spliced between runs by
this preparation step. The controller chooses its run at decision time.

Units in both prepared files are °C, relative-humidity fraction, surface
pressure kPa, irradiance W/m², and wind m/s. Sky temperature is derived from
downward longwave radiation with OCHRE's existing Stefan–Boltzmann constant.
The manifest records the local interval, UTC interval, location, source files,
units, row counts, and any unavailable 00Z cycles. `prepare` checks complete
hourly realized coverage and the full +1 through +384 hour cadence of every
saved forecast run before writing the manifest.

An opt-in source configuration is in
`config/oikolab_2025_2026/neighborhood.json`. It selects the prepared realized
weather, EPW ground and mains source, and prepared forecast archive. For a
full November 2025 through March 2026 HVAC run from the repository root, use
`home_02`, which has an OCHRE rest-of-home source:

```bash
.venv/bin/python examples/single_home_hems/run_simulation.py \
  --source-config-directory examples/single_home_hems/config/oikolab_2025_2026 \
  --home-id home_02 --controller hvac \
  --start-time 2025-11-01T00:00:00 --duration-hours 3624 \
  --weather-forecast-mode predict --skip-optimizer-plan-figures \
  --output-directory examples/single_home_hems/outputs/home_02/oikolab_2025nov_2026mar_predict
```

Change the mode to `perfect` and choose a different output directory for the
comparison. The 3,624 elapsed hours include both DST changes and end at local
April 1, 2026 midnight. A full run performs about 14,496 HVAC decisions and
two minute-resolution plant simulations, so the short DST smoke tests are a
faster integration check. `home_01` uses a short measured rest-of-home replay;
its source does not cover the full November–March interval. The bundled water
draw replay also does not cover the full interval for joint HEMS runs.

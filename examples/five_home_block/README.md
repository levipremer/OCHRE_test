# Five-home all-electric OCHRE block

This example runs five all-electric OCHRE dwellings on a shared one-minute
clock for one day, maps version-specific OCHRE outputs to stable internal
names, aggregates neighborhood power, validates the results, and creates 12
headless Matplotlib figures.

## Layout

```text
examples/five_home_block/
├── config/
│   ├── neighborhood.json       # shared clock, weather, outputs, and tolerances
│   └── homes.json              # envelope, equipment, thermostat, and EV inputs
├── inputs/README.md             # location for future block-specific inputs
├── outputs/
│   ├── data/                    # standardized, aggregate, mapping, and raw OCHRE data
│   └── figures/                 # 12 required PNG files
├── run_simulation.py            # command-line entry point
└── five_home_block/
    ├── config.py                # typed configuration loader and validation
    ├── home_factory.py          # OCHRE dwelling creation and equipment checks
    ├── output_mapping.py        # OCHRE-to-standard output mapping
    ├── aggregation.py           # power aggregation and balance checks
    ├── plotting.py              # headless figures
    ├── transformer/             # static distribution-transformer package
    └── simulation.py            # reusable neighborhood runner
```

## Requirements and command

Use the repository's existing environment from the repository root:

```bash
./.venv/bin/python examples/five_home_block/run_simulation.py
```

## Seasonal electrification parametric study

The separate parametric-study entry point leaves `run_simulation.py` and its
two configuration files unchanged. It reads the existing configuration first,
then layers the study-only assumptions from
`parametric_config/study.json`:

```bash
./.venv/bin/python examples/five_home_block/run_parametric_study.py
```

Home 1 stays fixed at its existing all-electric configuration. Homes 2-5 vary
between gas and electric space heating, gas/HPWH/ERWH water heating, and gas or
electric cooking. At most one of those four homes may use an ERWH. Electric
HVAC sizing remains inherited from each existing HPXML, and all configured
deadbands, cycling rules, backup staging, defrost settings, setpoints, EV
assumptions, and seeds come from the existing base configuration.

The runner selects seven-day summer and winter degree-hour windows, prepends a
48-hour warm-up, and caches 98 one-minute OCHRE profiles in compressed Parquet.
It also writes a 24,576-row scenario manifest that maps every valid seasonal
neighborhood scenario to five cached profiles. The profiles retain electric,
reactive, and gas power by relevant end use; no transformer size or transformer
constraint is applied. Neighborhood time series can therefore be reconstructed
later for arbitrary transformer post-processing.

Run a short integration check with three representative equipment states using:

```bash
./.venv/bin/python examples/five_home_block/run_parametric_study.py \
  --smoke-test \
  --output-directory /tmp/ochre_parametric_smoke
```

Gas and electric cooking use the same combined range/oven representation as
OCHRE's HPXML parser. A study-local event class preserves the gas schedule that
the current general OCHRE `EventBasedLoad` drops; it does not claim separate
cooktop and oven profiles or dual-fuel appliance support.

The workflow uses the existing OCHRE, pandas, NumPy, and Matplotlib
dependencies; it adds no package dependency.

## Envelope cohort and five-home diversity

The five distinct envelopes come from the Indiana subset of the ResStock
2024.2 TMY3 upgrade 11 dataset. The cohort is centered on the approximately
2,200 ft2 field home and deliberately matches its geometry: all five are
single-family detached homes with two conditioned floors above grade and a
separate unconditioned basement. They also keep 1960s-1970s construction,
wood-frame walls, uninsulated concrete basement walls, and vented attics
correlated. The records are drawn from Indiana counties with this matched
archetype, then simulated with the common Lafayette/Tippecanoe County
`G1801570.epw` so all five loads are weather-aligned.

| Home | ResStock up11 ID | Actual area | Requested band | Deviation |
|---|---:|---:|---:|---|
| `home_01` reference | `bldg0247499` | 2,179 ft2 | about 2,200 ft2 | -21 ft2 |
| `home_02` smaller | `bldg0164561` | 1,698 ft2 | 1,800-2,000 ft2 | 102 ft2 below preferred band; 2 ft2 below overall target |
| `home_03` similar | `bldg0220050` | 2,179 ft2 | 2,100-2,300 ft2 | within band |
| `home_04` larger | `bldg0293608` | 2,678 ft2 | 2,400-2,600 ft2 | 78 ft2 above preferred band |
| `home_05` outlier | `bldg0218050` | 2,678 ft2 | 2,600-2,800 ft2 | within band |

The deviations for Homes 2 and 4 are caused by the discrete ResStock geometry
sizes available in this matched cohort: 1,698, 2,179, and 2,678 ft2. Home 5 is
the larger, leakier neighborhood outlier, but retains the same two-story and
unconditioned-basement archetype.

These are post-upgrade values read from the downloaded HPXML files. Upgrade 11
applies R-60 ceiling insulation and a 30% infiltration reduction to all five,
while retaining realistic differences in the underlying envelopes.

| Home | Vintage / stories | Post-upgrade ACH50 | Main wall insulation | Windows U / SHGC | Foundation insulation | Ducts | Thermal mass |
|---|---|---:|---|---|---|---|---|
| `home_01` | 1970s / 2 | 14.0 | Effective R-9.3 | 0.47 / 0.54 | Uninsulated | Unconditioned basement, 30% outside leakage | Light, 40% of floor area |
| `home_02` | 1960s / 2 | 10.5 | Effective R-4.1 | 0.38 / 0.44 | Uninsulated | Unconditioned basement, 20% outside leakage | Light, 40% of floor area |
| `home_03` | 1960s / 2 | 17.5 | Effective R-4.1 | 0.38 / 0.44 | Uninsulated | Unconditioned basement, 10% outside leakage | Light, 40% of floor area |
| `home_04` | 1970s / 2 | 10.5 | Effective R-9.4 | 0.38 / 0.44 | Uninsulated | Unconditioned basement, 20% outside leakage | Light, 40% of floor area |
| `home_05` | 1960s / 2 | 14.0 | Effective R-4.1 | 0.34 / 0.49 | Uninsulated | Unconditioned basement, 30% outside leakage | Light, 40% of floor area |

Most supply and return ducts are in the unconditioned basement, with smaller
fractions in conditioned space. This produces a realistic 10%-30% leakage-to-
outside spread without changing the common foundation archetype. Absolute
furniture thermal mass scales with conditioned floor area even though the
common mass category and area fraction are held correlated.

HVAC and water-heater equipment is retained from each ResStock record except
for the intentionally modified Home 1 equipment and occupancy inputs below.
The capacities are read from the HPXML inputs, not neighborhood overrides:

| Home | ASHP compressor thermal capacity | Nominal tonnage | Emergency backup | Water-heater type | Rated tank / UEF | Other distinguishing inputs |
|---|---:|---:|---:|---|---:|---|
| `home_01` | 17.58 kW | 5.00 tons, single-speed | 20.00 kW | Hybrid/auto HPWH | 50 gal / 3.35 | 3 HPXML residents, 45 kWh EV, 3.6 kW charger |
| `home_02` | 10.96 kW | 3.12 tons | 24.05 kW | Hybrid/auto HPWH | 50 gal / 3.45 | 2 HPXML residents, 60 kWh EV, 7.2 kW charger |
| `home_03` | 13.89 kW | 3.95 tons | 31.29 kW | Hybrid/auto HPWH | 66 gal / 3.35 | 5 HPXML residents, 75 kWh EV, 7.2 kW charger |
| `home_04` | 11.10 kW | 3.16 tons | 24.83 kW | Hybrid/auto HPWH | 66 gal / 3.35 | 5 HPXML residents, 50 kWh EV, 3.6 kW charger |
| `home_05` | 17.65 kW | 5.02 tons | 38.30 kW | Hybrid/auto HPWH | 50 gal / 3.45 | 4 HPXML residents, 82 kWh EV, 11.5 kW charger |

Nominal tonnage converts the HPXML thermal capacity using
1 refrigeration ton = 3.5169 kW thermal (12,000 Btu/h).

Occupancy is read from each HPXML rather than overridden by the neighborhood
configuration. Home 1 is intentionally configured for three residents and a
50-gallon tank; Homes 2 and 5 also have 50-gallon tanks, while Homes 3 and 4
retain their 66-gallon ResStock tanks.

ResStock upgrade 11 sizes the original backup as emergency heat, so the backup
capacities equal the large design loads rather than only the compressor
shortfall. Home 1 is intentionally set to 20 kW. For the active one-minute
cycling simulation, OCHRE rounds ER totals to the nearest 5 kW and
automatically divides them into 5/10 kW stages: 20, 25, 30, 25, and 40 kW for
Homes 1 through 5, respectively. The HPXMLs do not specify HPWH element
wattage; OCHRE therefore uses
its 4.5 kW HPWH resistance-element default for all five instead of the former
custom 4.5-5.5 kW progression. OCHRE models usable electric tank volume as 90%
of rated volume, yielding 224.9 L for 66-gallon tanks and 170.3 L for 50-gallon
tanks.

Thermostat and water-heater setpoints, EV arrival/departure times, arrival SOC,
and deterministic OCHRE seeds differ. The fixed base seed is
`20260805`; each home adds its configured `seed_offset`. OCHRE uses those seeds
when sampling appliance events, which prevents perfectly synchronized loads.

EV `ev_arrival_soc_fraction` is the SOC when the vehicle arrives home, matching
OCHRE's `start_soc` event field. Before arrival the vehicle is away and OCHRE
reports its retained SOC state. Charging is positive household demand; the
current EV implementation does not discharge.

`initialization_hours` is zero in this baseline. Current OCHRE initialization
advances event-based equipment over the initialization interval; using it with
a single explicit one-day EV event would consume or alter that event before the
recorded simulation. Thermal states are still initialized by OCHRE from the
first schedule state.

## Changing the simulation

- Change duration, timestep, start time, weather, seed, or output location in
  `config/neighborhood.json`. At the configured one-minute timestep, this
  OCHRE version represents explicit equipment cycling rather than automatically
  selecting ideal-capacity HVAC and water-heater logic.
- Add or remove objects in `config/homes.json`; home IDs must be unique.
- Replace `hpxml_file` and `schedule_file` with repository-relative or absolute
  paths for another ResStock/HPXML model. Both files are checked before a run.
- HVAC type, compressor capacity, backup-heat capacity, water-heater type,
  tank volume, efficiency, and element capacity are inherited from each home's
  ResStock HPXML. Heating, cooling, and water-heater setpoints remain explicit
  neighborhood assumptions in `homes.json`.
- `heating_deadband_c` controls the compressor thermostat band. The backup band
  and its offset remain independently configurable so compressor experiments do
  not silently change resistance-backup thresholds. The active baseline uses a
  centered 1.5 C band, from setpoint -0.75 C to setpoint +0.75 C.
- `heating_deadband_offset` locates the setpoint within that compressor band.
  The baseline value is 0.5, which centers the band equally above and below the
  setpoint.
- `compressor_min_on_time_min` and `compressor_min_off_time_min` independently
  constrain the compressor without preventing resistance backup from running.
  Both are set to 15 minutes in the active baseline.
- `backup_setpoint_offset_c` and `backup_deadband_c` are both 1.5 C. ER stage
  1 therefore starts 1.5 C below setpoint; additional stages are added after
  unsuccessful 10-minute recovery intervals, and all stages release at the
  setpoint. Change the timer with `backup_stage_escalation_delay_min`.
  Set `backup_number_of_stages` and `backup_stage_capacities_kw` to override
  automatic staging; the explicit capacity-list sum also overrides total ER
  capacity.
- `defrost_model` and `defrost_control_type` in `config/neighborhood.json`
  configure every heat pump in the block. The active baseline uses `Discrete`
  with `Auto`: single- and two-speed units select timer control, while
  variable-speed units select demand control.
- `defrost_er_strategy` selects `Aggressive` full-bank ER or `Conservative`
  first-stage-only ER during discrete defrost. The baseline remains
  `Aggressive` for backward compatibility.
- Modify EV capacity, charger power, arrival/departure clocks, and arrival SOC
  with the `ev_*` fields.
- Set the block transformer nameplate and overload percentage with the
  `transformer` object in `config/neighborhood.json`. The baseline is a 50 kVA
  transformer with a 150% overload capacity, or 75 kVA.

## Transformer model

The initial `transformer` package is deliberately static. It does not model
temperature, losses, voltage, power factor, aging, or time-dependent overload
behavior. It adds constant 50 kVA nameplate and 75 kVA overload-capacity columns
to the neighborhood aggregate data. The aggregate figure plots the 75 kVA
overload capacity as a horizontal reference line. Comparing that apparent-power
limit directly with home real power assumes unity power factor for this first
iteration.

## Output mapping

`output_mapping.py` inspects actual result columns and records its choices in
`outputs/data/output_mapping.json`. The initial OCHRE 0.9.1 mapping is:

| Standard field | OCHRE source |
|---|---|
| `total_electric_power_kw` | `Total Electric Power (kW)` |
| `hvac_power_kw` | heating plus cooling electric-power columns |
| `water_heating_power_kw` | `Water Heating Electric Power (kW)` |
| `ev_power_kw` | `EV Electric Power (kW)` |
| `rest_of_home_power_kw` | total minus HVAC, water heating, and EV |
| `indoor_temperature_c` | `Temperature - Indoor (C)` |
| `ev_soc_fraction` | `EV SOC (-)` |
| `water_heater_temperature_c` | `Hot Water Average Temperature (C)` |

Tank minimum/maximum temperatures and heating/cooling setpoints are retained
when available. Missing required fields produce an error listing every raw
column. Only negative rest-of-home values within `numerical_tolerance_kw` are
clipped; material negative values remain visible and emit a warning.

## Outputs

Each home receives `outputs/data/home_XX_standardized.csv`. The aggregate is
`outputs/data/neighborhood_aggregate.csv`, including the two constant transformer
capacity columns. Complete OCHRE CSV, metrics, hourly,
event, schedule, and JSON files remain under `outputs/data/raw/home_XX/`.

Figures are:

- `neighborhood_aggregate_power.png`
- `shared_weather.png`
- `home_01_power_breakdown.png` through `home_05_power_breakdown.png`
- `home_01_state_variables.png` through `home_05_state_variables.png`

Figures use a noninteractive backend and can be generated on a headless host.

## Validation and tests

The runner checks exact timestamp alignment, missing/duplicate timesteps,
finite and plausible power/temperature values, per-home and neighborhood power
balances, all-electric equipment and zero gas use, EV SOC bounds, water-tank
temperature bounds, and all 12 figure files.

Run lightweight unit tests with:

```bash
./.venv/bin/python -m unittest test.test_five_home_block -v
```

Override the active compressor minimum times for a separate scenario with:

```bash
./.venv/bin/python examples/five_home_block/run_simulation.py \
  --compressor-min-on-minutes 5 \
  --compressor-min-off-minutes 5 \
  --output-directory examples/five_home_block/outputs_compressor_min_5m
```

The defrost model and discrete control can also be overridden without editing
the configuration file:

```bash
./.venv/bin/python examples/five_home_block/run_simulation.py \
  --defrost-model Discrete \
  --defrost-control-type Demand
```

Compare conservative defrost ER dispatch with:

```bash
./.venv/bin/python examples/five_home_block/run_simulation.py \
  --defrost-er-strategy Conservative \
  --output-directory examples/five_home_block/outputs_conservative_defrost
```

## Known limitations and future coordination

- The five records are a deliberately coherent engineering sample, not a
  statistically representative sample of Lafayette housing. They are selected
  statewide for envelope compatibility and all use Lafayette weather.
- The common ResStock TMY3 weather is a typical-year realization, not a forecast
  or measured 2018 weather series.
- Upgrade 11 makes the ceiling and air-sealing measures common across the cohort;
  compare with baseline packages if pre-retrofit envelope performance is needed.
- The one-minute timestep increases runtime and output size relative to the
  original five-minute baseline, but preserves explicit HVAC and water-heater
  cycling in the exported profiles.
- All five ResStock heat pumps are single-stage in OCHRE. The active baseline
  enforces 15-minute compressor minimum on-time and off-time independently of
  resistance-backup operation.
- The baseline runs homes independently. `NeighborhoodSimulation.run_synchronized`
  and the `NeighborhoodController` protocol in `simulation.py` are the explicit
  extension point for enforcing transformer limits, dynamic operating envelopes,
  thermostat/EV/water-heater controls, or coordinated HEMS operation.

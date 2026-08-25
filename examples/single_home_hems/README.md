# Single-home HEMS OCHRE example

This example runs one selectable dwelling from the five-home block through a
stepwise Home Energy Management System (HEMS) interface. It reuses the same
HPXML, schedules, weather, one-minute timestep, deadbands, equipment settings,
and deterministic seed as the selected block home by default.

## Select and run a home

From the repository root, choose `home_01` through `home_05`:

```bash
./.venv/bin/python examples/single_home_hems/run_simulation.py --home-id home_01
```

Each selection writes to its own directory under
`examples/single_home_hems/outputs/HOME_ID`, so it does not overwrite the
five-home block outputs.

Optional simulation overrides include:

```bash
./.venv/bin/python examples/single_home_hems/run_simulation.py \
  --home-id home_03 \
  --time-step-minutes 1 \
  --heating-deadband-c 1.5 \
  --heating-deadband-offset 0.5
```

## Add your HEMS

Edit `hems.py`, specifically `HEMSController.controls_for_step`. It is called
once per timestep before OCHRE updates the home and receives:

- `timestamp`: the timestep being controlled;
- `previous_status`: OCHRE results from the preceding timestep, or `None` on
  the first step;
- `current_schedule`: current weather, setpoints, occupancy schedules, and
  other exogenous OCHRE inputs.

Return an OCHRE control dictionary keyed by equipment or end use. The supplied
controller returns `{}`, which is an uncontrolled baseline. For example, this
would lower the heating setpoint by 1 C during hours 17 through 19:

```python
def controls_for_step(self, timestamp, previous_status, current_schedule):
    if 17 <= timestamp.hour < 20:
        base = current_schedule["HVAC Heating Setpoint (C)"]
        return {"HVAC Heating": {"Setpoint": base - 1.0}}
    return {}
```

Common native OCHRE controls include HVAC `Setpoint`, `Deadband`, and
`Load Fraction`; water-heater `Setpoint`, `Deadband`, and `Load Fraction`; and
EV `P Setpoint`, `Max Power`, `SOC Rate`, and `Delay`.

## Outputs

The example saves:

- raw OCHRE outputs under `data/raw/`;
- standardized home data under `data/HOME_ID_standardized.csv`;
- every HEMS command under `data/HOME_ID_hems_controls.csv`;
- home power and state-variable figures under `figures/`.

The transformer is intentionally not included here because this example models
one home behind its meter. Transformer-level coordination remains in the
five-home block example.

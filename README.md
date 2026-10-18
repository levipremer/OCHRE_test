![OCHRE](https://github.com/NREL/OCHRE/blob/main/docs/source/images/OCHRE-Logo-Horiz-2Color.png)

# OCHRE: The Object-oriented Controllable High-resolution Residential Energy Model

## Unified HEMS research fork

Built on NREL OCHRE (base commit `458b9b2`), this fork adds EV/V2H and HPWH
control, dynamic battery health, and reproducible paired studies. HVAC remains
native/pass-through. Upstream attribution and native usage are retained below.

### Added and changed

- Standalone home physics/mobility and control preferences, independently
  configurable per home/EV: solver, formulation, horizon, forecast and weights.
- One runner saves all-baseline, non-V2H twin and advanced trajectories.
  Twin end-use demand supplies backed-load forecasts; actual-load caps protect execution.
- EV convex/MILP and historical-scenario MPC, fixed solve clocks, departure
  comfort, optional health/degradation/terminal costs, and baseline fallback.
- Continuous SOH/damage accounting, faded-capacity SOC, baseline-relative net
  savings, unscored warmup/cooldown and consolidated performance/solver reports.
- Frozen paired studies and resumable attempts. Existing HVAC defrost,
  backup heat, system identification and multi-home functionality remain.

### Install and run

Use Python **3.12** for custom HEMS (study hashing requires 3.11+). From this
checkout, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -r examples/single_home_hems/requirements.txt
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --home-config examples/single_home_hems/config/unified_home.json --control-config examples/single_home_hems/config/unified_control.json --home-id home_01 --output-directory examples/single_home_hems/outputs/my_unified_run
```

On Linux/macOS use `.venv/bin/python`. Review the selected JSONs before running;
use a fresh output directory. CLARABEL supports convex problems, SCIPY supports
MILP, and optional GUROBI needs its package **and a suitable license**.
Installing gurobipy alone does not remove license size limits.

### Navigate and extend

| Guide | Purpose |
| --- | --- |
| [HEMS quick start](examples/single_home_hems/README.md) | Templates, dependencies and commands |
| [Pipeline](examples/single_home_hems/docs/pipeline.md) | Three-case flow, constraints and outputs |
| [Extension map](examples/single_home_hems/docs/extending.md) | Module responsibilities and safe changes |
| [Battery model](examples/single_home_hems/docs/degradation.md) | SOH, calibration and damage pricing |
| [Scenarios](examples/single_home_hems/docs/scenarios.md) | Historical selection and joint scenario OCP |
| [Studies](examples/single_home_hems/docs/studies.md) | Stage, execute, resume and compare variants |
| [Five-home example](examples/five_home_block/README.md) | Shared factory, inputs and neighborhood experiments |
| [Validation](examples/single_home_hems/docs/validation/README.md) | Commands and known failures |
| [Change history](CUSTOM_CHANGES.md) / [archive](archive/development_docs_2026-10-17/README.md) | Research changes and superseded guides |

Native physics live in `ochre/`; shared EV equipment, mobility and aging models
live in `examples/five_home_block/five_home_block/`. Controllers, forecasts,
orchestration and reporting live in `examples/single_home_hems/single_home_hems/`.
Tests are in `test/`; predictors remain under the example's `models/`.
Keep referenced datasets available; review redistribution rights before publishing.

Extend these interfaces instead of creating another simulation loop.
[AGENTS.md](AGENTS.md) requires explicit approval for formulation changes.
The full suite has known legacy failures and terminal mean/minimum discrepancies.
Source cleanup also changes study fingerprints: use a new study revision or
the original checkout for older experiments; do not bypass environment checks.

## Upstream OCHRE overview

OCHRE&trade; is a Python-based energy modeling tool designed to model end-use
loads and distributed energy resources in residential buildings. It can model
flexible devices---including HVAC equipment, water heaters, electric vehicles,
solar PV, and batteries---and the thermal and electrical interactions between
them. OCHRE has been used to generate diverse and high-resolution load
profiles, examine the impacts of advanced control strategies on energy costs
and occupant comfort, and assess grid reliability and resilience through
building-to-grid co-simulation.

More information about OCHRE can be found in [our
documentation](https://ochre-nrel.readthedocs.io/), on [NREL's
website](https://www.nrel.gov/grid/ochre.html), and from the [Powered By
OCHRE](https://www.youtube.com/watch?v=B5elLVtYDbI) webinar recording.

If you use OCHRE for your research or other projects, please fill out our [user survey](https://forms.office.com/g/U4xYhaWEvs).

## Native upstream installation

These commands install upstream OCHRE, not this fork's HEMS additions. For the
unified pipeline, use the editable-checkout installation above.

OCHRE can be installed using `pip` from the command line:

```
pip install ochre-nrel
```

Alternatively, you can install a specific branch, for example:

```
pip install git+https://github.com/NREL/OCHRE@dev
```

Note that OCHRE requires Python version >=3.9 and <3.13.

## Usage

OCHRE can be used to simulate a residential dwelling or an individual piece of
equipment. In either case, a python object is instantiated and then simulated.
A set of input parameters and/or input files must be defined. 

Below is a simple example of simulating a dwelling:
```
import os
import datetime as dt
from ochre import Dwelling
from ochre.utils import default_input_path # for using sample files
dwelling = Dwelling(
    start_time=dt.datetime(2018, 1, 1, 0, 0),
    time_res=dt.timedelta(minutes=10),       
    duration=dt.timedelta(days=3),
    hpxml_file=os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml"),
    hpxml_schedule_file=os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
    weather_file=os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
)

df, metrics, hourly = dwelling.simulate()
```

This will return 3 variables:
 * `df`: a Pandas DataFrame with 10 minute resolution
 * `metrics`: a dictionary of energy metrics
 * `hourly`: a Pandas DataFrame with 1 hour resolution (verbosity >= 3 only)

For more examples, see:
* The [OCHRE User
  Tutorial](https://colab.research.google.com/github/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb)
  Jupyter notebook 
* Python example scripts to:
  * Run a [single dwelling](https://github.com/NREL/OCHRE/blob/main/bin/run_dwelling.py)
  * Run a [single piece of equipment](https://github.com/NREL/OCHRE/blob/main/bin/run_equipment.py)
  * Run a [fleet of equipment](https://github.com/NREL/OCHRE/blob/main/bin/run_fleet.py)
  * Run [multiple dwellings](https://github.com/NREL/OCHRE/blob/main/bin/run_multiple.py)
  * Run a [OCHRE with an external controller](https://github.com/NREL/OCHRE/blob/main/bin/run_external_control.py)
  * Run a [OCHRE in co-simulation using HELICS](https://github.com/NREL/OCHRE/blob/main/bin/run_cosimulation.py)

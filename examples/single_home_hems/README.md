# Unified single-home HEMS

One OCHRE-backed runner compares **all-baseline** and **advanced**, with an optional
**non-V2H twin**, for each selected home. EV and HPWH support independent
baseline/MPC choices; HVAC remains native/pass-through. Multiple homes are
independent simulations, not a coordinated neighborhood OCP.

Set top-level `"run_non_v2h_twin": false` in the control JSON (or study JSON)
to omit the twin and use baseline backed-load forecasts. Supplied configs/templates
set `false`; omission retains the backward-compatible `true` behavior.

## Install and run

Use Python **3.12** for this HEMS workflow. Native OCHRE declares a broader
Python range; the study driver uses Python 3.11+ hashing APIs.
From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -r examples/single_home_hems/requirements.txt
.\.venv\Scripts\python.exe -c "import cvxpy as cp; print(cp.installed_solvers())"
.\.venv\Scripts\python.exe examples/single_home_hems/run_simulation.py --home-config examples/single_home_hems/config/unified_home.json --control-config examples/single_home_hems/config/unified_control.json --home-id home_01 --output-directory examples/single_home_hems/outputs/my_unified_run
```

On Linux/macOS, use `.venv/bin/python`. Keep the complete checkout and referenced
HPXML, weather, mobility and predictor inputs. Choose a **new output directory**;
the runner refuses to overwrite existing results. Review the selected JSONs
before running: home overrides can enable MPC even when defaults say baseline.

- **CLARABEL:** convex problems; **SCIPY:** MILP through HiGHS.
- **GUROBI:** optional (`python -m pip install gurobipy`), with a suitable license.
  Package installation alone does not provide an unrestricted academic license.
  Point `GRB_LICENSE_FILE` at your valid license if automatic discovery selects
  an obsolete/restricted one; the pipeline does not silently switch solvers.
- Templates use valid JSON `_comments` metadata rather than JSON comments:
  [home](config/unified_home.template.json), [control](config/unified_control.template.json).

## Guides

| Guide | Contents |
| --- | --- |
| [Pipeline](docs/pipeline.md) | Configuration, clocks, physical safeguards, outputs |
| [Battery degradation](docs/degradation.md) | SOH evolution, calibration, economics and rainflow certification |
| [EV scenarios](docs/scenarios.md) | Historical sampling, persistence prefix and scenario objective |
| [EV studies](docs/studies.md) | Stage, run, resume and compare paired experiments |
| [Extension map](docs/extending.md) | Module responsibilities and safe extension points |
| [Validation](docs/validation/README.md) | Commands, recent checks and known failures |

Start with `run_ev_study.py` **without** `--run` to stage a study only.
`--smoke-test` executes a small validation with the configured cases; `--run` executes selected
jobs (all jobs if unfiltered). See the study guide before resuming frozen inputs.

Legacy HPWH CLI entry points remain compatibility interfaces. The EV overlay
runner/configs and obsolete EV import shims are removed from the active layout.
New work uses the two-file unified configuration. Historical instructions
are preserved in the [development archive](../../archive/development_docs_2026-10-17/README.md).

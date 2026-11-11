"""standalone HVAC MPC with a 20-hour horizon."""

from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "single_home_hems"))

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.controllers.hvac import HVACPredictiveController
from examples.single_home_hems.single_home_hems.rate_setup import select_rate_for_simulation
from examples.single_home_hems.single_home_hems.simulation import (
    SingleHomeHEMSSimulation,
    select_home,
)
from examples.single_home_hems.single_home_hems.weather_forecast import ArchivedWeatherForecast


OUTPUT = Path(__file__).resolve().parent / "results"
MODEL = Path(__file__).resolve().parent / "model"
START = dt.datetime(2025, 11, 30)
DURATION = dt.timedelta(hours=48)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Study output already exists: {OUTPUT}")
    config = load_configuration(
        ROOT / "examples" / "single_home_hems" / "config" / "oikolab_2025_2026",
        repo_root=ROOT,
    )
    home = select_home(config, "home_01")
    config = replace(
        config,
        start_time=pd.Timestamp(START).tz_localize(
            str(config.weather_metadata["timezone"]),
            ambiguous="raise", nonexistent="raise",
        ),
        duration=DURATION,
        time_step=dt.timedelta(minutes=1),
        output_directory=OUTPUT,
        homes=(home,),
    )
    config.validate()
    rate = select_rate_for_simulation("tou-winter", config)
    if config.weather_forecast_file is None:
        raise ValueError("The HVAC run requires weather_forecast_file in the source config")
    weather = ArchivedWeatherForecast.from_parquet(
        config.weather_forecast_file,
        local_timezone=str(config.weather_metadata["timezone"]),
    )
    controller = HVACPredictiveController(
        model_path=MODEL,
        horizon=pd.Timedelta(hours=20),
        model_timestep=pd.Timedelta(minutes=15),
        optimization_kind="mixed integer",
        solver="GUROBI",
        solver_options={"TimeLimit": 300.0, "MIPGap": 0.05, "Threads": 1},
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
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "study_manifest.json").write_text(json.dumps({
        "study": "standalone HVAC predictive control, Grid Scenario 1",
        "evaluation_start_local": config.start_time.isoformat(),
        "evaluation_end_exclusive_local": (config.start_time + DURATION).isoformat(),
        "evaluation_duration_hours": 48,
        "prediction_horizon_hours": 20,
        "prediction_decision_minutes": 15,
        "comfort_history_rule": "previous 4 measured hours plus 20 forecast hours",
        "weather_forecast_mode": "archived 00Z GFS predictive",
        "rate_structure": rate.metadata,
        "controlled_devices": ["HVAC Heating", "HVAC Cooling"],
        "native_reference": True,
        "solver_options": dict(controller.solver_options),
        "status": "running",
    }, indent=2) + "\n", encoding="utf-8")
    summary = SingleHomeHEMSSimulation(
        config,
        home,
        HEMSController(device_controllers=(controller,), price_schedule=rate.schedule),
        initial_temperature_c=21.0,
        hvac_warmup_duration=pd.Timedelta(hours=16),
        weather_archive=weather,
        save_optimizer_plan_figures=False,
    ).run()
    price = summary.controls["electricity_price_usd_per_kwh"]
    bills = {}
    for name, results in (
        ("controlled", summary.standardized_results),
        ("native_reference", summary.uncontrolled_results),
    ):
        power = results["total_electric_power_kw"].astype(float)
        bills[name] = {
            "energy_kwh": float(power.sum() / 60),
            "energy_charge_usd": float((power * price.reindex(power.index)).sum() / 60),
            "peak_kw": float(power.max()),
        }
    (OUTPUT / "grid1_bill.json").write_text(
        json.dumps(bills, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = OUTPUT / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

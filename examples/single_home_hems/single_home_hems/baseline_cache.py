"""Study-scoped, integrity-checked reuse of all-baseline plant trajectories.

Only the plant run is cached: scoring and figures are regenerated per job.
No controller state or non-V2H twin is reused. Parquet preserves frame dtypes;
there is no pickle loading or cross-study reuse.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import pandas as pd


# These preferences do not affect autonomous baseline equipment. Everything
# else (including comfort target, physical aging, mobility and initialization)
# stays in the key. Forecast extension is separately included in the context.
INERT = {"controller", "formulation", "solver", "solver_options", "horizon_hours",
         "control_minutes", "enable_degradation", "degradation_method",
         "electricity_weight", "health_weight", "health_soc_weight", "comfort_weight", "trip_weight",
         "persistence_weight", "mip_relative_gap", "rainflow_relative_tolerance",
         "rainflow_absolute_tolerance", "rainflow_max_iterations", "enforce_terminal_soc",
         "terminal_soc_weight"}
FRAMES = ("raw", "commands", "schedule", "sensors", "water_training", "hvac_training")


def _cache_path(path):
    """Use extended Windows paths even when system long-path support is off.

    Publishing replaces a short temporary name with a 64-character key; files
    readable during staging can otherwise exceed MAX_PATH after that rename.
    """
    path = Path(path).absolute()
    if os.name != "nt" or str(path).startswith("\\\\?\\"):
        return path
    value = str(path)
    return Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline_key(sim, scope):
    """Conservative identity; scope fingerprints study inputs and source/runtime."""
    config = asdict(sim.config)
    config.pop("output_directory")
    config.pop("homes")
    home = asdict(sim.home)
    for ev in home["evs"]:
        if ev["hems"]:
            ev["hems"] = {k: v for k, v in ev["hems"].items() if k not in INERT}
    mobility = {key: dict(start=str(path.start), end=str(path.end),
        trips=path.trips.to_json(orient="split", date_format="iso"), metadata=path.metadata)
        for key, path in sim.ev_mobility_schedules.items()}
    payload = dict(version=2, trip_energy_timing="departure", scope=scope, config=config, home=home, mobility=mobility,
        water_draw=sim.water_draw_metadata, water_draw_file=sim.water_draw_file,
        water_heater_hp_only=sim.water_heater_hp_only,
        initial_temperature_c=sim.initial_temperature_c,
        sensor_layout=sim.water_heater_sensor_layout.metadata(),
        gain_regime=asdict(sim.hvac_gain_regime), context=sim.unified_context,
        tariff=type(sim.controller).__qualname__)
    return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()


def _publish_cache(staging, entry):
    """Retry Windows sharing/access locks; optional cache failure is nonfatal.

    Never replace an existing entry. A completed plant trajectory is usable
    even if OneDrive/antivirus prevents publishing its cache directory.
    """
    for attempt in range(6):
        if entry.exists():
            return "Cache destination already exists; not replaced"
        try:
            staging.rename(entry)
            return None
        except PermissionError as error:
            if attempt == 5:
                return str(error)
            time.sleep(.2 * 2**attempt)  # 6.2 seconds total; no unbounded wait.


def baseline_result(sim, cache, *, progress=None):
    """Load a complete verified baseline, or atomically publish a fresh one."""
    if sim.controller.device_controllers or sim.unified_context.get("providers"):
        raise ValueError("Baseline cache cannot reuse advanced controllers/providers")
    root = _cache_path(cache["directory"])
    root.mkdir(parents=True, exist_ok=True)
    key = baseline_key(sim, cache["scope"])
    entry = root/key
    reused = entry.exists()
    publication_error = None
    if reused:
        if progress:
            progress(phase="verifying cached baseline")
        info = json.loads((entry/"manifest.json").read_text(encoding="utf-8"))
        if info["key"] != key:
            raise ValueError("Baseline cache identity mismatch")
        for name, expected in info["hashes"].items():
            path = (entry/name).resolve()
            if not path.is_relative_to(entry.resolve()) or _hash(path) != expected:
                raise ValueError(f"Baseline cache changed: {name}")
        result = tuple(pd.read_parquet(entry/f"{name}.parquet") for name in FRAMES)
        shutil.copytree(entry/"artifacts", sim.data_directory, dirs_exist_ok=True)
        if progress:
            progress(phase="cached baseline reused")
    else:
        result = sim._run_stepwise(apply_controls=True, raw_directory=sim.raw_directory, progress=progress)
        with tempfile.TemporaryDirectory(prefix="building_", dir=root, ignore_cleanup_errors=True) as temporary:
            staging = Path(temporary)
            for name, frame in zip(FRAMES, result):
                frame.to_parquet(staging/f"{name}.parquet")
            shutil.copytree(sim.data_directory, staging/"artifacts")
            info = dict(key=key, hashes={str(p.relative_to(staging)): _hash(p)
                for p in staging.rglob("*") if p.is_file()})
            (staging/"manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            publication_error = _publish_cache(staging, entry)
        if publication_error:
            warnings.warn("Baseline cache publication failed; using the completed simulation without caching. "
                          + publication_error, RuntimeWarning)
            if progress:
                progress(phase="cache publication unavailable; continuing uncached")
    sim.data_directory.mkdir(parents=True, exist_ok=True)
    record = dict(reused=reused, key=key, cache_directory=str(entry))
    if not reused:
        record.update(published=publication_error is None)
        if publication_error:
            record["publication_error"] = publication_error
    (sim.data_directory/"baseline_reuse.json").write_text(json.dumps(
        record, indent=2), encoding="utf-8")
    return result

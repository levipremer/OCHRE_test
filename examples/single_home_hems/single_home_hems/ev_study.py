"""Frozen, resumable EV comparisons using the unified three-case runner.

Staging never simulates a home. Unsupported planned formulations are retained
in the manifest and rejected explicitly when selected for execution.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import EVSchedule, build_ev_schedules
from .control_config import EV_CONTROL, keys, load_unified, merge
from .unified import run_unified

METHODS = {"none", "linear_throughput", "rainflow", "rainflow_multicut"}
PLANNED = {"rainflow_smoothed", "rainflow_hh"}


def digest(value):
    """Hash canonical finite JSON for frozen-study identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """Hash file bytes without loading the full dataset into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    """Read a UTF-8 study document or checkpoint."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Atomic checkpoints; never overwrite a simulation directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identifier(value):
    """Validate portable nonempty IDs for study directories."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError(f"Unsafe or empty study identifier: {value!r}")
    return value


def resolve(root, value):
    """Resolve an input path against the explicit repository root."""
    path = Path(value)
    return (path if path.is_absolute() else root/path).resolve()


def environment(root):
    """Record actual source, including uncommitted changes, and solver versions."""
    sources = {}
    for folder in ("ochre", "examples/single_home_hems", "examples/five_home_block/five_home_block"):
        for path in sorted((root/folder).rglob("*.py")):
            if not any(part.startswith("outputs") for part in path.parts):
                sources[str(path.relative_to(root))] = file_hash(path)
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "gurobipy", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_sha256": digest(sources), "versions": versions}


def frozen_schedule(schedule):
    """Serialize trips and provenance without resampling."""
    rows = json.loads(schedule.trips.to_json(orient="records", date_format="iso"))
    return {"trips": rows, "start": str(schedule.start), "end": str(schedule.end),
            "metadata": schedule.metadata}


def restore_schedule(data):
    """Rebuild canonical mobility from a frozen snapshot."""
    trips = pd.DataFrame(data["trips"])
    if trips.empty:
        trips = pd.DataFrame(columns=["departure", "arrival", "distance_km"])
    for column in ("departure", "arrival"):
        trips[column] = pd.to_datetime(trips[column])
    return EVSchedule(trips, pd.Timestamp(data["start"]), pd.Timestamp(data["end"]), data["metadata"])


def shift_schedule(data, start):
    """Replay one frozen source trace at another calendar date, without resampling."""
    schedule = restore_schedule(data)
    delta = pd.Timestamp(start)-schedule.start
    trips = schedule.trips.copy()
    for column in ("departure", "arrival"):
        trips[column] += delta
    metadata = dict(schedule.metadata, replay_anchor=str(schedule.start),
                    replay_shift_seconds=delta.total_seconds(), weekday_alignment_preserved=delta.days % 7 == 0)
    return frozen_schedule(replace(schedule, trips=trips, start=schedule.start+delta,
                                   end=schedule.end+delta, metadata=metadata))


def validate_study(study):
    """Check IDs, windows, paired cases and supported/planned methods."""
    keys(study, {"schema_version", "name", "home_config", "control_config", "output_directory",
                 "home_id", "vehicle_id", "duration_hours", "horizon_hours", "control_minutes",
                 "warmup_hours", "cooldown_hours", "pair_mobility_by_seed", "initial_socs",
                 "schedule_cases", "common_ev_controls", "configurations", "_comments"}, "EV study")
    if study["schema_version"] != 1:
        raise ValueError("Unsupported EV study schema")
    identifier(study["name"])
    socs = study["initial_socs"]
    if not socs or len(set(socs)) != len(socs) or any(not isinstance(s, (int, float)) or not 0 <= s <= 1 for s in socs):
        raise ValueError("initial_socs must be unique finite fractions in [0,1]")
    for field in ("duration_hours", "horizon_hours", "control_minutes"):
        if not isinstance(study[field], (int, float)) or not 0 < study[field] < float("inf"):
            raise ValueError(f"Study {field} must be finite and positive")
    for field in ("warmup_hours", "cooldown_hours"):
        value = study.get(field, 24.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
            raise ValueError(f"Study {field} must be finite and nonnegative")
    if type(study.get("pair_mobility_by_seed", False)) is not bool:
        raise ValueError("pair_mobility_by_seed must be boolean")
    keys(study["common_ev_controls"], EV_CONTROL | {"forecast", "forecast_hours"}, "common EV controls")
    for collection in ("schedule_cases", "configurations"):
        entries = study[collection]
        ids = [identifier(v["id"]) for v in entries]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{collection} requires unique nonempty IDs")
    for case in study["schedule_cases"]:
        keys(case, {"id", "start_time", "mobility_seed", "home_seed"}, "schedule case")
        pd.Timestamp(case["start_time"])
        for name in ("mobility_seed", "home_seed"):
            if type(case[name]) is not int or case[name] < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
    for variant in study["configurations"]:
        keys(variant, {"id", "description", "ev_controls"}, "study configuration")
        keys(variant["ev_controls"], EV_CONTROL, "variant EV controls")
        method = variant["ev_controls"].get("degradation_method", "none")
        if method not in METHODS | PLANNED:
            raise ValueError(f"Unknown study degradation method: {method}")


def stage_study(study_file, repo_root, *, output_directory=None, smoke=False):
    """Freeze paired inputs and materialize every requested configuration.

    Existing case definitions cannot change. New configurations may be appended;
    saved definitions and mobility are verified, never silently regenerated.
    Smoke mode is separate: 15-minute warm-up + 30-minute evaluation, two configs.
    """
    root = Path(repo_root).resolve()
    study = read_json(study_file)
    validate_study(study)
    if smoke:
        study = deepcopy(study)
        study.update(name=study["name"] + "_driver_smoke", duration_hours=.5, horizon_hours=.5, warmup_hours=.25, cooldown_hours=.25,
                     initial_socs=[.5], schedule_cases=study["schedule_cases"][:1])
        study["configurations"] = study["configurations"][:2]
    output = resolve(root, output_directory or (study["output_directory"] + ("_driver_smoke" if smoke else "")))
    home_doc = read_json(resolve(root, study["home_config"]))
    control_doc = read_json(resolve(root, study["control_config"]))
    home_id, vehicle_id = study["home_id"], study["vehicle_id"]
    home = home_doc["homes"][home_id]
    if set(home["evs"]) != {vehicle_id}:
        raise ValueError("Initial EV study requires exactly one explicitly selected EV")
    if home["evs"][vehicle_id]["mobility"]["source"] != "audi":
        raise ValueError("This pilot requires Audi mobility")
    home_doc["homes"] = {home_id: home}
    control_doc["homes"] = {home_id: control_doc.get("homes", {}).get(home_id, {})}
    # Files influencing plant inputs are immutable within a study, including
    # OCHRE defaults; do not hash irrelevant large NHTS data for an Audi study.
    inputs = [resolve(root, home_doc["simulation"]["weather_file"]),
              resolve(root, home["hpxml_file"]), resolve(root, home["schedule_file"]),
              resolve(root, home["evs"][vehicle_id]["mobility"]["source_file"])]
    if home.get("water_draw_file"):
        inputs.append(resolve(root, home["water_draw_file"]))
    inputs.extend(p for p in (root/"ochre/defaults").rglob("*") if p.is_file())
    hashes = {str(p): file_hash(p) for p in sorted(set(inputs))}
    fixed = {k: v for k, v in study.items() if k not in {"configurations", "_comments", "output_directory"}}
    context = digest({"study": fixed, "home": home_doc, "control": control_doc, "input_hashes": hashes})
    manifest_file = output/"manifest.json"
    manifest = read_json(manifest_file) if manifest_file.exists() else {
        "schema_version": 1, "name": study["name"], "repo_root": str(root),
        "context_sha256": context, "input_hashes": hashes, "cases": {}, "jobs": {},
        "smoke": smoke, "staged_only": True}
    if manifest["context_sha256"] != context:
        raise ValueError("Frozen study inputs changed; use a new output directory")
    schedule_cache = {}
    # Reload already frozen anchors when appending/resuming staging, rather than
    # selecting a fresh source window for a newly added month.
    paired_mobility = study.get("pair_mobility_by_seed", False)
    for case in study["schedule_cases"]:
        if paired_mobility:
            for saved_id, saved in manifest["cases"].items():
                if saved["schedule_case"] == case["id"]:
                    schedule_cache.setdefault(case["mobility_seed"], read_json(output/"cases"/saved_id/"mobility.json"))
                    break
    for case in study["schedule_cases"]:
        for soc in study["initial_socs"]:
            case_id = f"{case['id']}_soc_{str(soc).replace('.', 'p')}"
            hdoc, cdoc = deepcopy(home_doc), deepcopy(control_doc)
            hdoc["simulation"].update(start_time=case["start_time"], duration_hours=study["duration_hours"],
                warmup_hours=study.get("warmup_hours", 24.0), cooldown_hours=study.get("cooldown_hours", 24.0),
                random_seed=case["home_seed"],
                output_directory=str(output/"unused"), plot_dpi=80)
            entry = hdoc["homes"][home_id]["evs"][vehicle_id]
            entry["initial_soc_fraction"] = soc
            entry["mobility"].update(seed=case["mobility_seed"])
            common = merge(cdoc.get("defaults", {}).get("ev", {}),
                           cdoc["homes"][home_id].get("evs", {}).get(vehicle_id, {}))
            common = merge(common, study["common_ev_controls"])
            common.update(controller="mpc", formulation="milp", horizon_hours=study["horizon_hours"],
                          forecast_hours=study["horizon_hours"], control_minutes=study["control_minutes"],
                          forecast={"mode": "perfect"}, enable_degradation=False, degradation_method="none")
            ctl = cdoc["homes"][home_id]
            ctl.update(evs={vehicle_id: common})
            for device in ("hpwh", "hvac"):
                ctl[device] = merge(ctl.get(device, {}), {"controller": "baseline"})
            case_dir = output/"cases"/case_id
            if case_id not in manifest["cases"]:
                write_json(case_dir/"home.json", hdoc)
                write_json(case_dir/"control.json", cdoc)
                config = load_unified(case_dir/"home.json", case_dir/"control.json", root)
                mobility_key = case["mobility_seed"] if paired_mobility else case["id"]
                if mobility_key not in schedule_cache:
                    dwelling = config.neighborhood.homes[0]
                    dwelling = replace(dwelling, evs=tuple(replace(e, hems=replace(e.hems,
                        forecast_margin_hours=2*study["horizon_hours"])) for e in dwelling.evs))
                    schedule_cache[mobility_key] = {key: frozen_schedule(value) for key, value in
                        build_ev_schedules(config.simulation_window(), dwelling).items()}
                mobility = schedule_cache[mobility_key]
                if paired_mobility:
                    mobility = {key: shift_schedule(value, config.simulation_window().start_time)
                                for key, value in mobility.items()}
                write_json(case_dir/"mobility.json", mobility)
                manifest["cases"][case_id] = {"initial_soc": soc, "schedule_case": case["id"],
                    "warmup_hours": config.warmup_hours,
                    "cooldown_hours": config.cooldown_hours,
                    "mobility_seed": case["mobility_seed"],
                    "files": {str(p.relative_to(output)): file_hash(p) for p in
                              (case_dir/"home.json", case_dir/"control.json", case_dir/"mobility.json")}}
            for variant in study["configurations"]:
                job_id = f"{case_id}__{variant['id']}"
                variant_doc = deepcopy(cdoc)
                variant_doc["homes"][home_id]["evs"][vehicle_id] = merge(common, variant["ev_controls"])
                job_dir = output/"jobs"/job_id
                definition = digest({"context": context, "case_id": case_id, "variant": variant})
                if job_id in manifest["jobs"]:
                    if manifest["jobs"][job_id]["definition_sha256"] != definition:
                        raise ValueError(f"Configuration {variant['id']} changed; give it a new ID")
                    continue
                write_json(job_dir/"control.json", variant_doc)
                method = variant_doc["homes"][home_id]["evs"][vehicle_id]["degradation_method"]
                # Validate implemented variants now; planned methods are explicitly blocked.
                if method in METHODS:
                    load_unified(case_dir/"home.json", job_dir/"control.json", root)
                manifest["jobs"][job_id] = {"case_id": case_id, "configuration_id": variant["id"],
                    "definition_sha256": definition, "control_sha256": file_hash(job_dir/"control.json"),
                    "method": method, "availability": "ready" if method in METHODS else "pending_formulation",
                    "home_id": home_id, "vehicle_id": vehicle_id}
    # Verify previous staged snapshots even when no simulation is requested.
    verify_files(output, manifest)
    write_json(manifest_file, manifest)
    return manifest_file


def verify_files(output, manifest):
    """Verify frozen hashes before resuming or extending a study."""
    for path, expected in manifest["input_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    for case in manifest["cases"].values():
        for path, expected in case["files"].items():
            if file_hash(output/path) != expected:
                raise ValueError(f"Frozen case changed: {path}")
    for job_id, job in manifest["jobs"].items():
        if file_hash(output/"jobs"/job_id/"control.json") != job["control_sha256"]:
            raise ValueError(f"Staged controls changed: {job_id}")


def summarize(output, manifest):
    """Long-form paired results; failures are never treated as zero savings."""
    frames, statuses = [], []
    for job_id, job in manifest["jobs"].items():
        status_file = output/"jobs"/job_id/"status.json"
        status = read_json(status_file) if status_file.exists() else {"state": "not_run"}
        statuses.append({"job_id": job_id, **job, **status})
        if status["state"] == "complete":
            frame = pd.read_csv(output/status["run_directory"]/"data/performance.csv")
            solver_file = output/status["run_directory"]/"data/solver_stats.csv"
            if solver_file.exists():
                solver = pd.read_csv(solver_file)
                solver["device"] = solver.device.replace({"Water Heating": "hpwh", "HVAC": "hvac"})
                fields = [c for c in solver if c not in frame or c in {"home_id", "case", "device"}]
                frame = frame.merge(solver[fields], on=["home_id", "case", "device"], how="left", validate="one_to_one")
            frame["configuration_id"], frame["study_case"] = job["configuration_id"], job["case_id"]
            frame["study_initial_soc"] = manifest["cases"][job["case_id"]]["initial_soc"]
            frame["warmup_hours"] = manifest["cases"][job["case_id"]].get("warmup_hours", 0)
            frame["cooldown_hours"] = manifest["cases"][job["case_id"]].get("cooldown_hours", 0)
            frames.append(frame)
    pd.DataFrame(statuses).to_csv(output/"study_runs.csv", index=False)
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data.to_csv(output/"study_metrics.csv", index=False)
        # Aggregate only the common completed case intersection across ALL variants.
        variants = {j["configuration_id"] for j in manifest["jobs"].values()}
        coverage = {v: set(data.loc[data.configuration_id.eq(v), "study_case"]) for v in variants}
        paired = set.intersection(*coverage.values())
        values = data.loc[data.study_case.isin(paired)]
        numeric = list(values.select_dtypes(include="number").columns)
        # Case is an essential grouping key: never pool advanced and non-V2H.
        aggregate = values.groupby(["configuration_id", "case", "device"])[numeric].agg(["count", "mean", "std", "min", "max"])
        aggregate.columns = ["_".join(column) for column in aggregate.columns]
        aggregate.reset_index().to_csv(output/"paired_summary.csv", index=False)
        from .ev_study_plots import plot_study
        plot_study(output, manifest, statuses, values)
    complete = sum(s["state"] == "complete" for s in statuses)
    text = (f"# EV study status\n\nCompleted {complete}/{len(statuses)} jobs.\n\n"
            "Each job retains all-baseline, non-V2H twin, and advanced data/figures. "
            "Net savings includes baseline NMC reference degradation at new minus residual battery value. No terminal-energy credit. "
            "Warm-up and cooldown are controller-active and excluded from all evaluation metrics; study_initial_soc is the warm-up-start setting, "
            "whereas initial_soc in EV metrics is the realized evaluation-start SOC. "
            "All SOC/comfort violations, unmet trips, and solver fallbacks remain in the metrics.\n\n"
            "study_runs.csv includes every job and failure. study_metrics.csv contains completed jobs only. "
            "paired_summary.csv uses only cases completed by every staged configuration; it may be empty. "
            "Every summary groups by configuration, case and device: advanced and non-V2H are never averaged together. "
            "Statistics include mean/std/min/max and counts; max_solve_seconds_max is the worst observed solve. "
            "figures/ contains separate advanced/non-V2H cross-controller trajectories and paired savings comparisons. "
            "No cross-configuration baseline cache is used in this first driver.\n")
    (output/"study_report.md").write_text(text, encoding="utf-8")


def run_study(manifest_file, *, configuration_ids=None, case_ids=None):
    """Execute selected jobs, resume completed ones, preserve failed attempts."""
    manifest_file = Path(manifest_file).resolve()
    output, manifest = manifest_file.parent, read_json(manifest_file)
    verify_files(output, manifest)
    root = Path(manifest["repo_root"])
    for requested, known in ((configuration_ids, {j["configuration_id"] for j in manifest["jobs"].values()}),
                             (case_ids, set(manifest["cases"]))):
        if requested is not None and (not requested or set(requested)-known):
            raise ValueError("Unknown or empty study selection")
    selected = {key: job for key, job in manifest["jobs"].items()
                if (configuration_ids is None or job["configuration_id"] in configuration_ids)
                and (case_ids is None or job["case_id"] in case_ids)}
    if not selected:
        raise ValueError("Study selection contains no jobs")
    blocked = [key for key, job in selected.items() if job["method"] not in METHODS]
    if blocked:
        raise NotImplementedError("Selected formulations are staged, not implemented; no runs started: " + ", ".join(blocked))
    runtime = environment(root)
    if "execution_environment" in manifest and manifest["execution_environment"] != runtime:
        raise ValueError("Code or solver environment changed; start a new study revision")
    manifest.update(execution_environment=runtime, staged_only=False)
    write_json(manifest_file, manifest)
    for job_id, job in selected.items():
        job_dir, case_dir = output/"jobs"/job_id, output/"cases"/job["case_id"]
        status_file = job_dir/"status.json"
        status = read_json(status_file) if status_file.exists() else {}
        if status.get("state") == "complete":
            for path, expected in status["output_hashes"].items():
                if file_hash(output/path) != expected:
                    raise ValueError(f"Completed output changed: {path}")
            continue
        attempt = int(status.get("attempt", 0)) + 1
        # OCHRE emits long equipment filenames; compact run paths avoid Windows
        # MAX_PATH failures under long OneDrive workspace roots.
        run_dir = output/"r"/digest(job_id)[:12]/f"a{attempt:03d}"
        config = load_unified(case_dir/"home.json", job_dir/"control.json", root)
        config.neighborhood = replace(config.neighborhood, output_directory=run_dir)
        frozen = {key: restore_schedule(value) for key, value in read_json(case_dir/"mobility.json").items()}
        status = {"state": "running", "attempt": attempt, "run_directory": str(run_dir.relative_to(output))}
        write_json(status_file, status)
        started = time.perf_counter()
        try:
            run_unified(config, [job["home_id"]], mobility_by_home={job["home_id"]: frozen})
            status.update(state="complete", elapsed_seconds=time.perf_counter()-started,
                          output_hashes={str(p.relative_to(output)): file_hash(p) for p in run_dir.rglob("*") if p.is_file()})
        except Exception as error:
            status.update(state="failed", elapsed_seconds=time.perf_counter()-started,
                          error=f"{type(error).__name__}: {error}")
            write_json(status_file, status)
            summarize(output, manifest)
            raise
        write_json(status_file, status)
    summarize(output, manifest)
    return output

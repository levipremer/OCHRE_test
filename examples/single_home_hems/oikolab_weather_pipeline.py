"""Acquire and prepare OikoLab weather for the single-home HEMS.

The prepared plant CSV contains only atmospheric weather. OCHRE's separate EPW
path remains the source of ground and water-mains temperatures. Forecast rows
retain both run and valid times; no controller forecast assumption is imposed
by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


LOCAL_TIMEZONE = "America/Indiana/Indianapolis"
LOCAL_TZ = ZoneInfo(LOCAL_TIMEZONE)
BASE_URL = "https://api.oikolab.com"
DEFAULT_LATITUDE = 40.430930
DEFAULT_LONGITUDE = -86.911617
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "outputs" / "oikolab_weather"
STEFAN_BOLTZMANN_OCHRE = 5.6697e-8

# OikoLab's column labels include their units. Use surface pressure, rather
# than sea-level pressure, for OCHRE's psychrometric calculations.
PLANT_PARAMETERS = (
    "temperature",
    "relative_humidity",
    "surface_pressure",
    "surface_solar_radiation",
    "direct_normal_solar_radiation",
    "surface_diffuse_solar_radiation",
    "wind_speed",
    "surface_thermal_radiation",
)
FORECAST_PARAMETERS = (
    "temperature",
    "relative_humidity",
    "surface_solar_radiation",
    "direct_normal_solar_radiation",
    "surface_diffuse_solar_radiation",
)
OIKO_COLUMNS = {
    "temperature": ("temperature (degC)", "temp_air"),
    "relative_humidity": ("relative_humidity (0-1)", "relative_humidity"),
    "surface_pressure": ("surface_pressure (Pa)", "pressure"),
    "surface_solar_radiation": ("surface_solar_radiation (W/m^2)", "ghi"),
    "direct_normal_solar_radiation": ("direct_normal_solar_radiation (W/m^2)", "dni"),
    "surface_diffuse_solar_radiation": ("surface_diffuse_solar_radiation (W/m^2)", "dhi"),
    "wind_speed": ("wind_speed (m/s)", "wind_speed"),
    "surface_thermal_radiation": ("surface_thermal_radiation (W/m^2)", "sky_temperature"),
}
PLANT_COLUMNS = (
    "temp_air", "relative_humidity", "pressure", "ghi", "dni", "dhi",
    "wind_speed", "sky_temperature",
)
FORECAST_COLUMNS = ("temp_air", "relative_humidity", "ghi", "dni", "dhi")


class ArchivedCycleUnavailable(RuntimeError):
    """OikoLab explicitly reports that a requested archived 00Z cycle is absent."""


def _local_window(start_date: str, end_date_exclusive: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date.fromisoformat(start_date)).tz_localize(LOCAL_TZ)
    end = pd.Timestamp(date.fromisoformat(end_date_exclusive)).tz_localize(LOCAL_TZ)
    if end <= start:
        raise ValueError("end-local-exclusive must be later than start-local")
    return start, end


def required_utc_runs(start_local: pd.Timestamp, end_local: pd.Timestamp) -> pd.DatetimeIndex:
    """00Z runs that can be selected during a local simulation interval."""
    first_day = start_local.tz_convert("UTC").normalize()
    last_day = (end_local.tz_convert("UTC") - pd.Timedelta(minutes=1)).normalize()
    return pd.date_range(first_day, last_day, freq="D", tz="UTC")


def realized_request_dates(start_local: pd.Timestamp, end_local: pd.Timestamp) -> tuple[str, str]:
    """UTC API dates including both hourly anchors needed for interpolation."""
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    # OikoLab returns both start 00Z and end 00Z for date-only requests.
    # The ceiling ensures a sample at/after a non-midnight local end.
    return start_utc.floor("D").strftime("%Y-%m-%d"), end_utc.ceil("D").strftime("%Y-%m-%d")


def _api_key() -> str:
    key = os.environ.get("OIKOLAB_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OIKOLAB_API_KEY in the environment before fetching weather")
    return key


def _get_json(endpoint: str, params: dict[str, object]) -> dict:
    query: list[tuple[str, object]] = []
    for name, value in params.items():
        if isinstance(value, (tuple, list)):
            query.extend((name, item) for item in value)
        else:
            query.append((name, value))
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=query,
        headers={"api-key": _api_key()},
        timeout=120,
    )
    if endpoint == "archivedforecast" and response.status_code == 400:
        try:
            error = response.json()
        except ValueError:
            error = None
        if isinstance(error, dict) and error.get("title") == "UTC cycle not available":
            raise ArchivedCycleUnavailable("OikoLab reports UTC cycle not available")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected {endpoint} response type")
    return payload


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def _read_record(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        record = json.load(file)
    if not isinstance(record, dict) or "response" not in record or "request" not in record:
        raise ValueError(f"Invalid OikoLab raw record: {path}")
    return record


def _make_record(endpoint: str, request: dict[str, object], response: dict) -> dict:
    # The credential is always in the HTTP header and is never serialized.
    return {
        "schema_version": 1,
        "endpoint": endpoint,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "response": response,
    }


def _parse_index(values: list) -> pd.DatetimeIndex:
    if not values:
        raise ValueError("OikoLab returned no weather rows")
    if isinstance(values[0], (int, float)):
        index = pd.to_datetime(values, unit="s", utc=True)
    else:
        index = pd.to_datetime(values, utc=True)
    if index.isna().any() or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("OikoLab UTC timestamps are missing, duplicated, or unsorted")
    return pd.DatetimeIndex(index, name="valid_time_utc")


def decode_response(
    response: dict, parameters: tuple[str, ...], expected_model: str | None = None,
) -> pd.DataFrame:
    """Return normalized values indexed by the actual UTC valid time."""
    data = response.get("data")
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict) or not {"index", "columns", "data"} <= data.keys():
        raise ValueError("OikoLab response has no tabular data")
    frame = pd.DataFrame(data["data"], columns=data["columns"], index=_parse_index(data["index"]))
    if expected_model is not None and "model (name)" in frame:
        models = set(frame["model (name)"].dropna().astype(str))
        if models != {expected_model}:
            raise ValueError(f"Expected {expected_model} source rows, got {sorted(models)}")
    column_names = [OIKO_COLUMNS[parameter][0] for parameter in parameters]
    missing = sorted(set(column_names) - set(frame.columns))
    if missing:
        raise ValueError(f"OikoLab response is missing requested columns: {missing}")
    normalized = frame.loc[:, column_names].rename(
        columns={OIKO_COLUMNS[parameter][0]: OIKO_COLUMNS[parameter][1] for parameter in parameters}
    )
    normalized = normalized.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(normalized.to_numpy()).all():
        raise ValueError("OikoLab response contains missing or non-finite weather values")
    if "pressure" in normalized:
        normalized["pressure"] /= 1000.0  # Pa to OCHRE kPa
    if "sky_temperature" in normalized:
        if (normalized["sky_temperature"] <= 0).any():
            raise ValueError("Downward longwave radiation must be positive")
        normalized["sky_temperature"] = (
            (normalized["sky_temperature"] / STEFAN_BOLTZMANN_OCHRE) ** 0.25 - 273.15
        )
    _validate_weather(normalized)
    return normalized


def _validate_weather(frame: pd.DataFrame) -> None:
    if "relative_humidity" in frame and not frame["relative_humidity"].between(0, 1).all():
        raise ValueError("Relative humidity must be a fraction in [0, 1]")
    if "pressure" in frame and not frame["pressure"].between(50, 120).all():
        raise ValueError("Surface pressure in kPa is outside the expected range [50, 120]")
    for column in ("ghi", "dni", "dhi", "wind_speed"):
        if column in frame and (frame[column] < 0).any():
            raise ValueError(f"{column} contains negative values")


def _check_raw_request(record: dict, endpoint: str, latitude: float, longitude: float) -> None:
    request = record["request"]
    if record["endpoint"] != endpoint:
        raise ValueError(f"Expected {endpoint} raw record")
    if not np.isclose(float(request["lat"]), latitude) or not np.isclose(float(request["lon"]), longitude):
        raise ValueError("Raw weather coordinates do not match the requested location")


def fetch_realized(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float, force: bool = False,
) -> Path:
    request_start, request_end = realized_request_dates(start_local, end_local)
    path = data_dir / "raw" / "realized" / f"era5_{request_start}_{request_end}.json"
    if path.exists() and not force:
        _check_raw_request(_read_record(path), "weather", latitude, longitude)
        return path
    request = {
        "param": list(PLANT_PARAMETERS), "lat": latitude, "lon": longitude,
        "start": request_start, "end": request_end, "model": "era5", "freq": "H",
    }
    response = _get_json("weather", request)
    decode_response(response, PLANT_PARAMETERS, expected_model="era5")
    _write_json(path, _make_record("weather", request, response))
    return path


def fetch_forecasts(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float, force: bool = False,
    max_new_runs: int | None = None,
    skip_unavailable: bool = False,
) -> tuple[list[Path], int, list[pd.Timestamp]]:
    files: list[Path] = []
    new_count = 0
    unavailable: list[pd.Timestamp] = []
    saved_runs: set[pd.Timestamp] = set()
    for run_time in required_utc_runs(start_local, end_local):
        path = data_dir / "raw" / "gfs_00z" / f"gfs_{run_time:%Y%m%dT0000Z}.json"
        if path.exists() and not force:
            _check_raw_request(_read_record(path), "archivedforecast", latitude, longitude)
            files.append(path)
            saved_runs.add(run_time)
            continue
        if max_new_runs is not None and new_count >= max_new_runs:
            break
        request = {
            "param": list(FORECAST_PARAMETERS), "lat": latitude, "lon": longitude,
            "utc_cycle": run_time.strftime("%Y-%m-%dT00:00:00"), "model": "gfs",
        }
        try:
            response = _get_json("archivedforecast", request)
        except ArchivedCycleUnavailable:
            if not skip_unavailable:
                raise
            unavailable.append(run_time)
            continue
        decoded = decode_response(response, FORECAST_PARAMETERS, expected_model="gfs")
        _validate_gfs_horizon(decoded.index, run_time)
        reported_run = response.get("attributes", {}).get("gfs_reference_time")
        if reported_run is not None:
            parsed_run = pd.Timestamp(datetime.strptime(reported_run, "%Y-%m-%d %HZ"), tz="UTC")
            if parsed_run != run_time:
                raise ValueError(f"GFS response run time does not match request {run_time}")
        _write_json(path, _make_record("archivedforecast", request, response))
        files.append(path)
        saved_runs.add(run_time)
        new_count += 1
    if skip_unavailable:
        required = required_utc_runs(start_local, end_local)
        unavailable_set = set(unavailable)
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/archivedforecast",
            "reason": "UTC cycle not available",
            "location": {"latitude": latitude, "longitude": longitude},
            "local_interval": {
                "start_inclusive": start_local.isoformat(),
                "end_exclusive": end_local.isoformat(),
            },
            "required_run_count": len(required),
            "saved_run_count": len(saved_runs),
            "unavailable_run_times_utc": [run.isoformat() for run in unavailable],
            "not_attempted_run_times_utc": [
                run.isoformat() for run in required if run not in saved_runs and run not in unavailable_set
            ],
        }
        _write_json(data_dir / "reports" / "gfs_00z_unavailable_cycles.json", report)
    return files, new_count, unavailable


def _validate_gfs_horizon(valid_times: pd.DatetimeIndex, run_time: pd.Timestamp) -> None:
    expected_leads = list(range(1, 121)) + list(range(123, 385, 3))
    expected = run_time + pd.to_timedelta(expected_leads, unit="h")
    if not valid_times.equals(pd.DatetimeIndex(expected, name=valid_times.name)):
        raise ValueError(
            f"GFS run {run_time} does not contain the expected +1 to +384 hour valid-time cadence"
        )


def _deduplicate_realized(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames).sort_index()
    duplicates = frame.index[frame.index.duplicated()].unique()
    for timestamp in duplicates:
        rows = frame.loc[[timestamp]].to_numpy(dtype=float)
        if not np.allclose(rows, rows[0], rtol=0, atol=1e-8):
            raise ValueError(f"Conflicting realized weather at {timestamp}")
    return frame.loc[~frame.index.duplicated(keep="first")]


def _load_realized(
    data_dir: Path, latitude: float, longitude: float,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((data_dir / "raw" / "realized").glob("*.json"))
    if not paths:
        raise FileNotFoundError("No raw realized OikoLab files found")
    frames = []
    for path in paths:
        record = _read_record(path)
        _check_raw_request(record, "weather", latitude, longitude)
        if record["request"].get("model") != "era5":
            raise ValueError(f"Unexpected realized model in {path}")
        frames.append(decode_response(record["response"], PLANT_PARAMETERS, expected_model="era5"))
    return _deduplicate_realized(frames), paths


def prepare_realized(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float,
) -> tuple[Path, pd.DataFrame, list[Path]]:
    hourly, paths = _load_realized(data_dir, latitude, longitude)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    if hourly.index[0] > start_utc or hourly.index[-1] < end_utc:
        raise ValueError(
            f"Realized hourly coverage {hourly.index[0]} to {hourly.index[-1]} "
            f"does not bracket {start_utc} to {end_utc}"
        )
    needed_hours = pd.date_range(start_utc.floor("h"), end_utc.ceil("h"), freq="h", tz="UTC")
    missing_hours = needed_hours.difference(hourly.index)
    if len(missing_hours):
        raise ValueError(f"Realized weather is missing {len(missing_hours)} hourly samples; first: {missing_hours[0]}")
    # Interpolate in UTC so both fall-back local 01:00 hours are distinct and
    # spring-forward local times are never fabricated.
    minutes_utc = pd.date_range(start_utc, end_utc, freq="min", inclusive="left")
    joined = hourly.reindex(hourly.index.union(minutes_utc)).sort_index()
    minute = joined.interpolate(method="time", limit_area="inside").reindex(minutes_utc)
    if minute.isna().any().any():
        raise ValueError("Realized minute interpolation left missing weather values")
    minute = minute.loc[:, PLANT_COLUMNS]
    minute.index = minute.index.tz_convert(LOCAL_TZ)
    minute.index.name = "Time"
    path = data_dir / "prepared" / "realized_1min.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    output = minute.reset_index()
    output["Time"] = [timestamp.isoformat() for timestamp in minute.index]
    temporary = path.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    return path, minute, paths


def prepare_forecasts(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float,
) -> tuple[Path, pd.DataFrame, list[Path], list[dict[str, str]]]:
    run_times = required_utc_runs(start_local, end_local)
    paths = [data_dir / "raw" / "gfs_00z" / f"gfs_{run:%Y%m%dT0000Z}.json" for run in run_times]
    missing_runs = [run for run, path in zip(run_times, paths) if not path.is_file()]
    available = [(run, path) for run, path in zip(run_times, paths) if path.is_file()]
    substitutions: list[dict[str, str]] = []
    if missing_runs:
        _validate_unavailable_report(
            data_dir, start_local, end_local, latitude, longitude,
            run_times, available, missing_runs,
        )
        for missing_run in missing_runs:
            earlier = [run for run, _ in available if run < missing_run]
            if not earlier:
                raise ValueError(f"No earlier archived GFS run can substitute for {missing_run}")
            substitutions.append({
                "missing_run_time_utc": missing_run.isoformat(),
                "selected_run_time_utc": earlier[-1].isoformat(),
            })
    frames = []
    for run_time, path in available:
        record = _read_record(path)
        _check_raw_request(record, "archivedforecast", latitude, longitude)
        if record["request"].get("utc_cycle") != run_time.strftime("%Y-%m-%dT00:00:00"):
            raise ValueError(f"Forecast raw run timestamp mismatch in {path}")
        frame = decode_response(record["response"], FORECAST_PARAMETERS, expected_model="gfs")
        _validate_gfs_horizon(frame.index, run_time)
        rows = frame.reset_index()
        rows.insert(0, "run_time_utc", run_time)
        rows.insert(2, "valid_time_local", rows["valid_time_utc"].dt.tz_convert(LOCAL_TZ))
        frames.append(rows)
    forecast = pd.concat(frames, ignore_index=True)
    forecast = forecast.loc[:, ("run_time_utc", "valid_time_utc", "valid_time_local", *FORECAST_COLUMNS)]
    if forecast.duplicated(["run_time_utc", "valid_time_utc"]).any():
        raise ValueError("Forecast archive has duplicate run/valid-time pairs")
    path = data_dir / "prepared" / "gfs_00z_forecasts.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    forecast.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path, forecast, [path for _, path in available], substitutions


def _validate_unavailable_report(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float, required: pd.DatetimeIndex,
    available: list[tuple[pd.Timestamp, Path]], missing: list[pd.Timestamp],
) -> None:
    report_path = data_dir / "reports" / "gfs_00z_unavailable_cycles.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing {len(missing)} archived 00Z runs and no unavailable-cycle report: {report_path}"
        )
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError("Unavailable-cycle report must be a JSON object")
    interval = report.get("local_interval", {})
    location = report.get("location", {})
    if (
        report.get("schema_version") != 1
        or report.get("endpoint") != "/archivedforecast"
        or report.get("reason") != "UTC cycle not available"
        or not isinstance(interval, dict)
        or interval.get("start_inclusive") != start_local.isoformat()
        or interval.get("end_exclusive") != end_local.isoformat()
        or not isinstance(location, dict)
        or not np.isclose(float(location.get("latitude", np.nan)), latitude)
        or not np.isclose(float(location.get("longitude", np.nan)), longitude)
        or report.get("required_run_count") != len(required)
        or report.get("saved_run_count") != len(available)
        or report.get("not_attempted_run_times_utc") != []
    ):
        raise ValueError("Unavailable-cycle report does not match this interval, location, or saved archive")
    reported = report.get("unavailable_run_times_utc")
    expected = [run.isoformat() for run in missing]
    if (
        not isinstance(reported, list)
        or not all(isinstance(run, str) for run in reported)
        or sorted(reported) != sorted(expected)
    ):
        raise ValueError("Unavailable-cycle report does not exactly match missing 00Z runs")


def prepare(
    data_dir: Path, start_local: pd.Timestamp, end_local: pd.Timestamp,
    latitude: float, longitude: float,
) -> dict:
    realized_path, minute, realized_raw = prepare_realized(
        data_dir, start_local, end_local, latitude, longitude
    )
    forecast_path, forecast, forecast_raw, substitutions = prepare_forecasts(
        data_dir, start_local, end_local, latitude, longitude
    )
    manifest = {
        "schema_version": 1,
        "location": {"latitude": latitude, "longitude": longitude, "timezone": LOCAL_TIMEZONE},
        "interval": {
            "start_local_inclusive": start_local.isoformat(),
            "end_local_exclusive": end_local.isoformat(),
            "start_utc_inclusive": start_local.tz_convert("UTC").isoformat(),
            "end_utc_exclusive": end_local.tz_convert("UTC").isoformat(),
        },
        "realized": {
            "source": "OikoLab /weather model=era5",
            "path": str(realized_path.relative_to(data_dir)),
            "rows": len(minute),
            "frequency": "1 minute, linearly interpolated in UTC",
            "columns": list(PLANT_COLUMNS),
            "raw_files": [str(path.relative_to(data_dir)) for path in realized_raw],
        },
        "forecast": {
            "source": "OikoLab /archivedforecast model=gfs, 00Z runs",
            "path": str(forecast_path.relative_to(data_dir)),
            "rows": len(forecast),
            "runs": int(forecast["run_time_utc"].nunique()),
            "required_runs": len(required_utc_runs(start_local, end_local)),
            "unavailable_runs": [entry["missing_run_time_utc"] for entry in substitutions],
            "substitutions": substitutions,
            "selection_policy": "Use the latest available 00Z run at or before each decision for its entire horizon",
            "unavailable_report": (
                "reports/gfs_00z_unavailable_cycles.json" if substitutions else None
            ),
            "columns": list(FORECAST_COLUMNS),
            "raw_files": [str(path.relative_to(data_dir)) for path in forecast_raw],
        },
        "units": {
            "temp_air": "degC", "relative_humidity": "fraction", "pressure": "kPa",
            "ghi": "W/m^2", "dni": "W/m^2", "dhi": "W/m^2",
            "wind_speed": "m/s", "sky_temperature": "degC",
        },
        "sky_temperature_method": "(downward_longwave_W_m2 / 5.6697e-8) ** 0.25 - 273.15",
        "epw_hybrid": "EPW-derived mains and ground temperatures are supplied by OCHRE integration",
    }
    _write_json(data_dir / "prepared" / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-realized", "fetch-forecast", "prepare", "plan"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--start-local", required=True, help="First local simulation date YYYY-MM-DD")
        sub.add_argument("--end-local-exclusive", required=True, help="First local date after simulation")
        if name.startswith("fetch"):
            sub.add_argument("--execute", action="store_true", help="Actually make billable API requests")
            sub.add_argument("--force", action="store_true", help="Re-fetch existing raw responses")
        if name == "fetch-forecast":
            sub.add_argument("--max-new-runs", type=int, help="Limit new daily archived requests")
            sub.add_argument(
                "--skip-unavailable", action="store_true",
                help="Record and continue past exact 'UTC cycle not available' API responses",
            )
    subparsers.add_parser("account", help="Show current OikoLab usage (no data units)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "account":
        account = _get_json("account", {})
        print(json.dumps({key: account.get(key) for key in (
            "incl_units_per_billing_period", "current_usage", "daily_usage"
        )}, indent=2))
        return
    start_local, end_local = _local_window(args.start_local, args.end_local_exclusive)
    run_count = len(required_utc_runs(start_local, end_local))
    request_start, request_end = realized_request_dates(start_local, end_local)
    if args.command == "plan" or (args.command.startswith("fetch") and not args.execute):
        print(json.dumps({
            "local_start": start_local.isoformat(),
            "local_end_exclusive": end_local.isoformat(),
            "realized_request_utc_dates": [request_start, request_end],
            "realized_parameters": list(PLANT_PARAMETERS),
            "forecast_00z_runs": run_count,
            "forecast_parameters": list(FORECAST_PARAMETERS),
            "forecast_unit_estimate": run_count * len(FORECAST_PARAMETERS),
            "note": "The unit estimate is based on one parameter-month per archived run; check /account after staging.",
        }, indent=2))
        return
    if args.command == "fetch-realized":
        path = fetch_realized(
            args.data_dir, start_local, end_local, args.latitude, args.longitude, args.force
        )
        print(f"Realized raw response: {path}")
    elif args.command == "fetch-forecast":
        if args.max_new_runs is not None and args.max_new_runs < 0:
            raise ValueError("max-new-runs must be non-negative")
        paths, new_count, unavailable = fetch_forecasts(
            args.data_dir, start_local, end_local, args.latitude, args.longitude,
            args.force, args.max_new_runs, args.skip_unavailable,
        )
        print(
            f"Archived 00Z runs saved: {len(paths)}/{run_count}; "
            f"newly fetched: {new_count}; unavailable: {len(unavailable)}"
        )
        if args.skip_unavailable:
            print(f"Missing-cycle report: {args.data_dir / 'reports' / 'gfs_00z_unavailable_cycles.json'}")
    elif args.command == "prepare":
        manifest = prepare(args.data_dir, start_local, end_local, args.latitude, args.longitude)
        print(json.dumps({
            "realized_path": manifest["realized"]["path"],
            "realized_rows": manifest["realized"]["rows"],
            "forecast_path": manifest["forecast"]["path"],
            "forecast_runs": manifest["forecast"]["runs"],
        }, indent=2))


if __name__ == "__main__":
    main()

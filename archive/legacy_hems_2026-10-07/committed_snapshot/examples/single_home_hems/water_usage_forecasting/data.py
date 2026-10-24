"""Unit conversion, chronological splits, and an immutable test-data package."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


US_GALLON_TO_LITERS = 3.785411784
FLOW_COLUMN = "water_flow_l_per_min"
SOURCE_INTERVAL = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class PreparedData:
    training: pd.Series
    validation: pd.Series
    test_archive: Path
    manifest: dict


def thanksgiving(year: int) -> pd.Timestamp:
    """US Thanksgiving: the fourth Thursday in November."""
    first = pd.Timestamp(year=year, month=11, day=1)
    return first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 21)


def load_flow(path: Path, timezone: str, timestamp_position: str) -> pd.Series:
    """Convert US gpm to L/min and index readings by interval start.

    Preserve source order so the repeated fall-back hour can be disambiguated.
    Blank readings remain missing; they are never interpreted as zero usage.
    """
    if timestamp_position not in {"start", "end"}:
        raise ValueError("timestamp_position must be 'start' or 'end'")
    raw = pd.read_csv(path)
    if not {"Time", "Value"}.issubset(raw.columns):
        raise ValueError("Source CSV must contain Time and Value (US gpm)")
    if raw.empty:
        raise ValueError("Source CSV must contain at least one reading")
    index = pd.DatetimeIndex(pd.to_datetime(raw["Time"], errors="raise"))
    if index.hasnans:
        raise ValueError("Source timestamps cannot be missing")
    if index.tz is None:
        index = index.tz_localize(timezone, ambiguous="infer", nonexistent="raise")
    else:
        index = index.tz_convert(timezone)
    if timestamp_position == "end":
        index -= SOURCE_INTERVAL
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and increasing after timezone conversion")
    if (index.second != 0).any() or (index.microsecond != 0).any() or (index.nanosecond != 0).any():
        raise ValueError("Source readings must be aligned to whole minutes")
    values = pd.to_numeric(raw["Value"], errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Flow values must be nonnegative and finite, or blank")
    flow = pd.Series(values * US_GALLON_TO_LITERS, index=index, name=FLOW_COLUMN)
    flow.index.name = "timestamp"
    # Missing timestamps and missing measurements have the same explicit NaN representation.
    return flow.reindex(pd.date_range(index[0], index[-1], freq="min", name="timestamp"))


def aggregate_flow(flow: pd.Series, interval_minutes: int) -> pd.Series:
    """Average flow only over complete intervals, preserving volume in liters."""
    if interval_minutes <= 0 or 60 % interval_minutes:
        raise ValueError("interval_minutes must be positive and divide 60, keeping bins aligned across DST")
    bins = flow.resample(f"{interval_minutes}min", origin="start_day")
    return bins.mean().where(bins.count() == interval_minutes).rename(FLOW_COLUMN)


def _csv_bytes(flow: pd.Series) -> bytes:
    return flow.to_csv().encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_if_changed(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def prepare_data(
    source: Path,
    output: Path,
    converted_output: Path,
    *,
    interval_minutes: int = 15,
    validation_days: int = 7,
    test_year: int | None = None,
    timezone: str = "America/New_York",
    timestamp_position: str = "end",
) -> PreparedData:
    """Save converted data, pretest splits, and a checksummed ZIP holdout.

    The seal is a reproducibility and workflow boundary, not encryption. The
    converted source still contains all dates, but comparison code receives
    training and validation only until test evaluation is explicitly requested.
    """
    if source.resolve() == converted_output.resolve():
        raise ValueError("Converted output must not overwrite the original gpm source")
    if validation_days <= 0:
        raise ValueError("validation_days must be positive")
    if interval_minutes <= 0 or 60 % interval_minutes:
        raise ValueError("interval_minutes must be positive and divide 60, keeping bins aligned across DST")
    flow = load_flow(source, timezone, timestamp_position)
    available_end = flow.index[-1] + SOURCE_INTERVAL
    if test_year is None:
        candidates = [
            int(year) for year in sorted(set(flow.index.year))
            if flow.index[0] <= thanksgiving(int(year)).tz_localize(timezone) - pd.DateOffset(days=14)
            and available_end >= thanksgiving(int(year)).tz_localize(timezone)
        ]
        if len(candidates) != 1:
            raise ValueError("Specify --test-year; source must cover a complete pre-Thanksgiving fortnight")
        test_year = candidates[0]
    test_end = thanksgiving(test_year).tz_localize(timezone)
    test_start = test_end - pd.DateOffset(days=14)
    validation_start = test_start - pd.DateOffset(days=validation_days)
    if flow.index[0] >= validation_start or available_end < test_end:
        raise ValueError("Source must cover training, validation, and the complete test window")

    windows = {
        "training": (flow.index[0], validation_start),
        "validation": (validation_start, test_start),
        "test": (test_start, test_end),
    }
    # Aggregate each split independently; no bin may mix two dataset partitions.
    minute_splits = {
        name: flow.loc[(flow.index >= start) & (flow.index < end)]
        for name, (start, end) in windows.items()
    }
    splits = {
        name: aggregate_flow(part, interval_minutes)
        for name, part in minute_splits.items()
    }
    for name in ("training", "validation"):
        if not splits[name].notna().any():
            raise ValueError(f"No complete, observed {name} intervals")
    manifest = {
        "schema_version": 1,
        "source_name": source.name,
        "source_sha256": _sha256(source.read_bytes()),
        "source_units": "US gallons/minute",
        "units": "liters/minute",
        "gpm_to_l_per_min": US_GALLON_TO_LITERS,
        "timezone": timezone,
        "source_timestamp_position": timestamp_position,
        "output_timestamp_position": "interval start",
        "source_interval_minutes": 1,
        "model_interval_minutes": interval_minutes,
        "aggregation": "mean of all minutes; incomplete intervals remain missing",
        "test_year": test_year,
        "windows": {
            name: {
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat(),
                "minute_rows": len(minute_splits[name]),
                "missing_minutes": int(minute_splits[name].isna().sum()),
                "model_rows": len(splits[name]),
                "missing_model_intervals": int(splits[name].isna().sum()),
            }
            for name, (start, end) in windows.items()
        },
        "excluded_after_test_minutes": int((flow.index >= test_end).sum()),
        "seal_policy": "Test excluded from fitting and selection; open only for explicit final evaluation.",
    }
    members = {
        "test_minutes_l_per_min.csv": _csv_bytes(minute_splits["test"]),
        "test_model_intervals_l_per_min.csv": _csv_bytes(splits["test"]),
    }
    manifest["test_members_sha256"] = {name: _sha256(payload) for name, payload in members.items()}
    members["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    archive_path = output / "data" / "test_sealed.zip"
    if archive_path.exists() and archive_path.read_bytes() != buffer.getvalue():
        raise ValueError("Existing test seal differs; use a new output directory to preserve it")
    _write_if_changed(archive_path, buffer.getvalue())
    _write_if_changed(archive_path.with_suffix(".zip.sha256"),
                      f"{_sha256(buffer.getvalue())}  {archive_path.name}\n".encode("utf-8"))
    _write_if_changed(converted_output, _csv_bytes(flow))
    for name in ("training", "validation"):
        _write_if_changed(output / "data" / f"{name}_l_per_min.csv", _csv_bytes(splits[name]))
    _write_if_changed(output / "data" / "manifest.json", members["manifest.json"])
    return PreparedData(splits["training"], splits["validation"], archive_path, manifest)


def open_test_archive(path: Path) -> pd.Series:
    """Verify the seal and return test intervals only for final evaluation."""
    checksum = path.with_suffix(".zip.sha256").read_text().split()[0]
    if _sha256(path.read_bytes()) != checksum:
        raise ValueError("Test archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, checksum in manifest["test_members_sha256"].items():
            if _sha256(archive.read(name)) != checksum:
                raise ValueError(f"Test member checksum mismatch: {name}")
        frame = pd.read_csv(io.BytesIO(archive.read("test_model_intervals_l_per_min.csv")))
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_convert(manifest["timezone"])
    index.name = "timestamp"
    return pd.Series(frame[FLOW_COLUMN].to_numpy(), index=index, name=FLOW_COLUMN)


def load_prepared_data(
    output: Path, *, interval_minutes: int = 15, validation_days: int = 7,
    test_year: int | None = None, timezone: str = "America/New_York",
    timestamp_position: str = "end", source_name: str | None = None,
) -> PreparedData:
    """Load only pretest artifacts and verify the archive's outer checksum.

    A comparison rerun never parses the full source or opens/recreates the
    holdout. Existing preparation settings must match its immutable manifest.
    """
    directory = output / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        "model_interval_minutes": interval_minutes, "timezone": timezone,
        "source_timestamp_position": timestamp_position,
    }
    if test_year is not None:
        expected["test_year"] = test_year
    if source_name is not None:
        expected["source_name"] = source_name
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"Prepared {key} differs; preserve the seal and use a new output directory")
    windows = manifest["windows"]
    validation_start = pd.Timestamp(windows["validation"]["start_inclusive"]).tz_convert(timezone)
    test_start = pd.Timestamp(windows["test"]["start_inclusive"]).tz_convert(timezone)
    if validation_start + pd.DateOffset(days=validation_days) != test_start:
        raise ValueError("Prepared validation window differs; preserve the seal and use a new output directory")
    archive = directory / "test_sealed.zip"
    checksum = archive.with_suffix(".zip.sha256").read_text().split()[0]
    if _sha256(archive.read_bytes()) != checksum:
        raise ValueError("Test archive checksum mismatch")
    splits = {}
    for name in ("training", "validation"):
        frame = pd.read_csv(directory / f"{name}_l_per_min.csv")
        index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).tz_convert(timezone)
        index.name = "timestamp"
        start = pd.Timestamp(windows[name]["start_inclusive"]).tz_convert(timezone)
        end = pd.Timestamp(windows[name]["end_exclusive"]).tz_convert(timezone)
        # resample labels a partial first source interval by its bin start.
        start = start.tz_convert("UTC").floor(f"{interval_minutes}min").tz_convert(timezone)
        expected_index = pd.date_range(start, end, inclusive="left", freq=f"{interval_minutes}min", name="timestamp")
        if not index.equals(expected_index) or len(frame) != windows[name]["model_rows"]:
            raise ValueError(f"Prepared {name} timestamps disagree with the sealed manifest")
        values = pd.to_numeric(frame[FLOW_COLUMN], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"Prepared {name} values must be nonnegative and finite, or missing")
        if int(np.isnan(values).sum()) != windows[name]["missing_model_intervals"]:
            raise ValueError(f"Prepared {name} missing count disagrees with the sealed manifest")
        splits[name] = pd.Series(values, index=index, name=FLOW_COLUMN)
    return PreparedData(splits["training"], splits["validation"], archive, manifest)


def resample_pretest(
    prepared: PreparedData, interval_minutes: int,
) -> tuple[pd.Series, pd.Series, dict]:
    """Coarsen pretest data without reading source files or the sealed test.

    A fixed elapsed-time grid is anchored at validation start so both partitions
    meet exactly even across DST. A partial first training bin stays missing.
    Each complete bin preserves volume; any missing constituent invalidates it.
    """
    base = prepared.manifest["model_interval_minutes"]
    if (isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int)
            or interval_minutes < base or interval_minutes % base
            or 1440 % interval_minutes
            or (60 % interval_minutes and interval_minutes % 60)):
        raise ValueError("Comparison interval must be a prepared-interval multiple dividing 60 minutes or whole hours dividing a day")
    count = interval_minutes // base
    anchor = prepared.validation.index[0]
    parts = {}
    for name, series in (("training", prepared.training), ("validation", prepared.validation)):
        bins = series.resample(f"{interval_minutes}min", origin=anchor)
        parts[name] = bins.mean().where(bins.count() == count).rename(FLOW_COLUMN)
        parts[name].index.name = "timestamp"
    interval = pd.Timedelta(minutes=interval_minutes)
    if parts["training"].index[-1] + interval != parts["validation"].index[0]:
        raise ValueError("Coarse training and validation grids must meet at the sealed boundary")
    if parts["validation"].index[-1] + interval != pd.Timestamp(
        prepared.manifest["windows"]["test"]["start_inclusive"]
    ):
        raise ValueError("Comparison interval must align with validation end; test remains sealed")
    metadata = {
        "prepared_interval_minutes": base,
        "comparison_interval_minutes": interval_minutes,
        "aggregation": "mean of all constituent intervals; incomplete bins stay missing",
        "grid_anchor": anchor.isoformat(),
        "grid_clock": "fixed elapsed time; local phase can change across DST",
        "windows": {
            name: {"start_inclusive": series.index[0].isoformat(),
                   "end_exclusive": (series.index[-1] + interval).isoformat(),
                   "model_rows": len(series),
                   "missing_model_intervals": int(series.isna().sum())}
            for name, series in parts.items()
        },
        "test_status": "original archive unchanged and unopened; only pretest series aggregated",
    }
    return parts["training"], parts["validation"], metadata

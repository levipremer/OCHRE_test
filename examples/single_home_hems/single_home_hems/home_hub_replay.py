"""Align, audit, and replay measured electricity by Home Hub wiring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from examples.single_home_hems.build_dc_house_rest_of_home import (
    DATA_DIR,
    INPUT_FILES,
    fill_short_stable_gaps,
    read_meter,
)
from examples.five_home_block.five_home_block.rest_of_home import (
    HOME_HUB_TEST_END, HOME_HUB_TEST_START, TEST_END, TEST_START,
)


MINUTE = pd.Timedelta(minutes=1)
HOME_HUB_REPLAY_START = HOME_HUB_TEST_START - pd.Timedelta(hours=16)
HOME_HUB_REPLAY_END = HOME_HUB_TEST_END + pd.Timedelta(days=1)
# Physical replay only: one negative Hub residual is reallocated to Outside-
# Hub after the approved retrospective meter fills. Forecast issuance does not
# receive this future-dependent correction until its readings are complete.
APPROVED_REALLOCATION_TOTAL_END = pd.Timestamp("2026-10-08 00:15:00")
METER_NAMES = (
    "total_home",
    "home_hub",
    "ev_charger",
    "ahu_fan",
    "ac_unit",
    "water_heater",
)
EQUIPMENT_NAMES = ("ev_charger", "ahu_fan", "ac_unit", "water_heater")
SOURCE_FILES = {**INPUT_FILES, "home_hub": "GM Home Hubreturn.csv"}
STAGE_ONE_QUALITY = (
    "observed",
    "interpolated_source_meter",
    "unresolved_long_or_edge_gap",
    "unresolved_endpoint_change",
)
FINAL_METER_QUALITY = (*STAGE_ONE_QUALITY, "interpolated_replay_meter", "unavailable")


@dataclass(frozen=True)
class StageOneHomeHubMeters:
    """Shifted meters after the existing per-source fill, before replay fill."""

    aligned_w: pd.DataFrame
    quality: pd.DataFrame
    quality_counts: pd.DataFrame


@dataclass(frozen=True)
class PreparedHomeHubReplay:
    """Complete interval-start replay and its meter-level provenance."""

    power_kw: pd.DataFrame
    meter_quality: pd.DataFrame
    meter_quality_counts: pd.DataFrame
    split_audit_w: pd.DataFrame


def load_stage_one_home_hub_meters(
    data_dir: str | Path = DATA_DIR,
) -> StageOneHomeHubMeters:
    """Read six DC House CSVs and apply the established source-meter rule.

    Equipment and Home Hub meters at source time ``t-1`` are shifted to the
    total-home meter clock ``t`` before interpolation. The imported central
    rule only fills gaps of at most 60 minutes with endpoint difference below
    100 W. Remaining missing readings stay missing for a separately approved
    replay policy. This function never derives or interpolates split loads.
    """
    directory = Path(data_dir)
    meters: dict[str, pd.Series] = {}
    for name in METER_NAMES:
        table = read_meter(directory / SOURCE_FILES[name])
        meters[name] = pd.Series(
            table["Value"].to_numpy(),
            index=pd.DatetimeIndex(table["Time"], name="Time"),
            name=name,
        )
    source = _meter_frame(meters)
    aligned = source.copy()
    aligned.loc[:, ("home_hub", *EQUIPMENT_NAMES)] = source.loc[
        :, ("home_hub", *EQUIPMENT_NAMES)
    ].shift(1)

    filled = pd.DataFrame(index=source.index)
    quality = pd.DataFrame(index=source.index)
    for name in METER_NAMES:
        values, long_or_edge, endpoint_change = fill_short_stable_gaps(aligned[name])
        filled[name] = values
        status = pd.Series("observed", index=source.index, dtype="object")
        status.loc[aligned[name].isna() & values.notna()] = "interpolated_source_meter"
        status.loc[long_or_edge] = "unresolved_long_or_edge_gap"
        status.loc[endpoint_change] = "unresolved_endpoint_change"
        if (status.str.startswith("unresolved") != values.isna()).any():
            raise AssertionError(f"Source-meter quality does not match {name} gaps")
        quality[name] = status

    counts = pd.DataFrame(
        {
            name: quality[name].value_counts().reindex(STAGE_ONE_QUALITY, fill_value=0)
            for name in METER_NAMES
        }
    ).T
    counts.index.name = "meter"
    return StageOneHomeHubMeters(filled, quality, counts)


def prepare_home_hub_replay(
    source_start: str | pd.Timestamp,
    source_end: str | pd.Timestamp,
    *,
    data_dir: str | Path = DATA_DIR,
) -> PreparedHomeHubReplay:
    """Prepare a complete half-open ``[source_start, source_end)`` replay.

    Input times are naive local *interval starts*. First apply the established
    <=60-minute/<100-W source-meter fill on the total-meter clock. Then apply
    the existing replay policy to each remaining test-split source-meter gap:
    interpolate only complete, bounded gaps of at most four minutes. The
    second stage uses neither training nor future replay endpoints and never
    extrapolates or partially fills longer gaps. Split the Hub only after both
    meter stages, so Hub-other plus Outside-Hub equals the legacy residual.
    Requested minutes with unresolved meters or negative balances are errors.
    """
    start, end = pd.Timestamp(source_start), pd.Timestamp(source_end)
    aligned = (
        pd.isna(start) or pd.isna(end)
        or start.tz is not None or end.tz is not None
        or start != start.floor("min") or end != end.floor("min")
    )
    if aligned:
        raise ValueError("Home Hub replay bounds must be naive whole-minute timestamps")
    if TEST_START <= start < end <= TEST_END:
        split_start, split_end = TEST_START, TEST_END
    elif HOME_HUB_REPLAY_START <= start < end <= HOME_HUB_REPLAY_END:
        split_start, split_end = HOME_HUB_REPLAY_START, HOME_HUB_REPLAY_END
    else:
        raise ValueError(
            "Home Hub replay requires a positive whole-minute source interval "
            "within the approved September or October test replay window"
        )

    stage_one = load_stage_one_home_hub_meters(data_dir)
    test_ends = pd.date_range(split_start + MINUTE, split_end, freq=MINUTE)
    aligned_test = stage_one.aligned_w.reindex(test_ends)
    quality_test = stage_one.quality.reindex(test_ends).fillna("unavailable")
    replay_filled = aligned_test.copy()
    for name in METER_NAMES:
        series = aligned_test[name]
        missing = series.isna()
        run_lengths = missing.groupby((~missing).cumsum()).transform("sum")
        interpolated = series.interpolate(method="time", limit_area="inside")
        use_fill = missing & run_lengths.le(4) & interpolated.notna()
        replay_filled.loc[use_fill, name] = interpolated.loc[use_fill]
        quality_test.loc[use_fill, name] = "interpolated_replay_meter"

    split = split_aligned_home_hub_loads(replay_filled)
    if APPROVED_REALLOCATION_TOTAL_END in split.index:
        issue = APPROVED_REALLOCATION_TOTAL_END
        row = split.loc[issue]
        if (not np.isclose(row["hub_other_raw_w"], -55.599741, atol=0.001)
                or not np.isfinite(row["rest_of_home_raw_w"])
                or row["rest_of_home_raw_w"] < 0
                or row["outside_hub_raw_w"] < 0):
            raise ValueError("Approved October 8 Hub reallocation no longer matches source balances")
        split.loc[issue, "hub_other_w"] = 0.0
        split.loc[issue, "outside_hub_w"] = row["rest_of_home_raw_w"]
        split.loc[issue, "rest_of_home_w"] = row["rest_of_home_raw_w"]
        split.loc[issue, "quality"] = "approved_negative_hub_reallocation"
        split.loc[issue, "valid"] = True
    requested_ends = pd.date_range(start + MINUTE, end, freq=MINUTE)
    selected = split.reindex(requested_ends)
    selected_quality = quality_test.reindex(requested_ends)
    if not selected["valid"].all():
        bad = selected.loc[~selected["valid"]]
        first = bad.index[0]
        unresolved = selected_quality.loc[first]
        missing_meters = unresolved.index[replay_filled.loc[first].isna()].tolist()
        raise ValueError(
            f"Home Hub replay has {len(bad)} unresolved or negative-balance "
            f"minute(s); first total-meter time {first}, "
            f"quality {bad.iloc[0]['quality']}, missing meters {missing_meters}"
        )

    error_w = (
        selected["hub_other_w"] + selected["outside_hub_w"]
        - selected["rest_of_home_w"]
    )
    if not np.allclose(error_w.to_numpy(), 0.0, rtol=0, atol=1e-9):
        raise AssertionError("Home Hub split does not equal the original rest residual")

    interval_starts = pd.DatetimeIndex(
        selected["source_interval_start"], name="source_interval_start"
    )
    power_kw = pd.DataFrame(
        {
            "hub_other_power_kw": selected["hub_other_w"].to_numpy() / 1000,
            "outside_hub_power_kw": selected["outside_hub_w"].to_numpy() / 1000,
        },
        index=interval_starts,
    )
    selected_quality.index = interval_starts
    audit = selected.copy()
    audit.index = interval_starts
    counts = pd.DataFrame(
        {
            name: selected_quality[name].value_counts().reindex(
                FINAL_METER_QUALITY, fill_value=0
            )
            for name in METER_NAMES
        }
    ).T
    counts.index.name = "meter"
    return PreparedHomeHubReplay(power_kw, selected_quality, counts, audit)


def _meter_frame(meters: pd.DataFrame | Mapping[str, pd.Series]) -> pd.DataFrame:
    """Copy six source-meter series on one regular, one-minute clock."""
    if isinstance(meters, pd.DataFrame):
        if meters.columns.has_duplicates:
            raise ValueError("Source meter columns must be unique")
        missing = set(METER_NAMES) - set(meters.columns)
        if missing:
            raise ValueError(f"Missing source meters: {sorted(missing)}")
        frame = meters.loc[:, METER_NAMES].copy()
    elif isinstance(meters, Mapping):
        missing = set(METER_NAMES) - set(meters)
        if missing:
            raise ValueError(f"Missing source meters: {sorted(missing)}")
        series = {name: meters[name] for name in METER_NAMES}
        if any(not isinstance(value, pd.Series) for value in series.values()):
            raise TypeError("Each source meter must be a pandas Series")
        first_index = series[METER_NAMES[0]].index
        if any(not value.index.equals(first_index) for value in series.values()):
            raise ValueError("Source meter timestamp grids must match exactly")
        frame = pd.DataFrame({name: value.copy() for name, value in series.items()})
    else:
        raise TypeError("Source meters must be a DataFrame or mapping of Series")

    index = frame.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or not index.equals(index.floor("min"))
        or not index.equals(pd.date_range(index[0], periods=len(index), freq=MINUTE))
    ):
        raise ValueError("Source meters require one shared, regular one-minute timestamp grid")

    for name in METER_NAMES:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        if np.isinf(values).any():
            raise ValueError(f"Source meter {name} must be finite or missing")
        frame[name] = values
    return frame


def split_home_hub_loads(
    meters: pd.DataFrame | Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Return Hub-other and Outside-Hub power in source watts.

    At the total-home meter timestamp ``t``, Home Hub and equipment readings
    at ``t - 1 minute`` refer to the same physical interval ``[t-1, t)``.
    The output remains indexed by ``t`` and supplies ``source_interval_start``
    for downstream replay. A preceding source row is needed for the first
    usable result.

    ``hub_other + outside_hub`` equals the original total-minus-equipment
    rest-of-home residual on each valid row. Invalid balances remain visible
    in the raw columns; approved power columns are missing on those rows.
    No missing-value interpolation or clipping is performed here.
    """
    source = _meter_frame(meters)
    aligned = source.copy()
    aligned.loc[:, ("home_hub", *EQUIPMENT_NAMES)] = source.loc[
        :, ("home_hub", *EQUIPMENT_NAMES)
    ].shift(1)

    return split_aligned_home_hub_loads(aligned)


def split_aligned_home_hub_loads(
    aligned_meters: pd.DataFrame | Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Audit the split when input meters already share the total-meter clock.

    This accepts output from ``load_stage_one_home_hub_meters`` and from any
    later approved replay-fill step without applying another one-minute shift.
    """
    aligned = _meter_frame(aligned_meters)
    equipment = aligned.loc[:, EQUIPMENT_NAMES].sum(
        axis=1, min_count=len(EQUIPMENT_NAMES)
    )
    hub_other_raw = aligned["home_hub"] - equipment
    outside_raw = aligned["total_home"] - aligned["home_hub"]
    rest_raw = aligned["total_home"] - equipment

    missing_meter = aligned.isna().any(axis=1)
    negative_hub_other = hub_other_raw.lt(0)
    negative_outside_hub = outside_raw.lt(0)
    valid = ~(missing_meter | negative_hub_other | negative_outside_hub)

    quality = pd.Series("observed", index=aligned.index, dtype="object")
    quality.loc[missing_meter] = "excluded_missing_meter"
    quality.loc[~missing_meter &
                (negative_hub_other | negative_outside_hub)] = "excluded_negative_balance"

    return pd.DataFrame(
        {
            "source_interval_start": aligned.index - MINUTE,
            "hub_other_raw_w": hub_other_raw,
            "outside_hub_raw_w": outside_raw,
            "rest_of_home_raw_w": rest_raw,
            "hub_other_w": hub_other_raw.where(valid),
            "outside_hub_w": outside_raw.where(valid),
            "rest_of_home_w": rest_raw.where(valid),
            "missing_meter": missing_meter,
            "negative_hub_other": negative_hub_other,
            "negative_outside_hub": negative_outside_hub,
            "quality": quality,
            "valid": valid,
        },
        index=aligned.index,
    )

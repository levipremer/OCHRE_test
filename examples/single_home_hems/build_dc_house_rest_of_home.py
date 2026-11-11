"""Build a quality-flagged, one-minute DC House rest-of-home power series.

The source CSVs provide no unit or timezone metadata. Their magnitudes suggest
watts, so the output retains the source values without a unit conversion.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "external_data" / "dc_house_power"
INPUT_FILES = {
    "total_home": "total_home_demand_hhreturn.csv",
    "ev_charger": "EV Chargerreturn.csv",
    "ahu_fan": "AHU_fanreturn.csv",
    "ac_unit": "AC_unitoutreturn.csv",
    "water_heater": "HPWHreturn.csv",
}
EQUIPMENT_METERS = ("ev_charger", "ahu_fan", "ac_unit", "water_heater")
MAX_FILL_GAP_MINUTES = 60
MAX_ENDPOINT_CHANGE_W = 100.0
# Outside the known August 21 meter mismatch, only one of about 100,000
# complete minutes has a residual below 500 W. Treat these rare near-zero
# balances as uncertain instead of calling them representative base load.
MIN_PLAUSIBLE_REST_OF_HOME_W = 500.0
# The total-home meter falls below the separately measured equipment load in
# these sustained windows. Nearby small positive residuals are affected too.
METER_MISMATCH_INTERVALS = (
    ("2026-08-21 10:33:00", "2026-08-21 11:39:00"),
    ("2026-08-21 17:52:00", "2026-08-21 18:27:00"),
)


def read_meter(path: Path) -> pd.DataFrame:
    meter = pd.read_csv(path)
    if list(meter.columns) != ["Time", "Value"]:
        raise ValueError(f"Unexpected columns in {path}: {list(meter.columns)}")
    meter["Time"] = pd.to_datetime(
        meter["Time"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    meter["Value"] = pd.to_numeric(meter["Value"], errors="raise")
    if meter["Time"].isna().any() or meter["Time"].duplicated().any():
        raise ValueError(f"Invalid or duplicate timestamps in {path}")
    return meter


def fill_short_stable_gaps(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Interpolate bounded gaps up to an hour if endpoint change is <100 W."""
    missing = series.isna()
    groups = missing.ne(missing.shift()).cumsum()
    gap_length = missing.groupby(groups).transform("sum")
    before = series.ffill()
    after = series.bfill()
    endpoint_change = (after - before).abs()
    bounded = before.notna() & after.notna()
    short = gap_length.le(MAX_FILL_GAP_MINUTES)
    stable = endpoint_change.lt(MAX_ENDPOINT_CHANGE_W)
    fillable = missing & bounded & short & stable
    long_or_edge = missing & (~bounded | ~short)
    endpoint_change_gap = missing & bounded & short & ~stable
    interpolated = series.interpolate(method="linear", limit_area="inside")
    return series.where(~fillable, interpolated), long_or_edge, endpoint_change_gap


def contiguous_intervals(mask: pd.Series, times: pd.Series) -> pd.DataFrame:
    groups = mask.ne(mask.shift()).cumsum()
    intervals = (
        pd.DataFrame({"Time": times[mask], "group": groups[mask]})
        .groupby("group", sort=False)
        .agg(Start=("Time", "first"), End=("Time", "last"), Minutes=("Time", "size"))
        .reset_index(drop=True)
    )
    return intervals


def main() -> None:
    meters = {name: read_meter(DATA_DIR / filename) for name, filename in INPUT_FILES.items()}
    times = meters["total_home"]["Time"]
    expected_step = pd.Timedelta(minutes=1)
    if not times.diff().iloc[1:].eq(expected_step).all():
        raise ValueError("Total-home timestamps are not a regular one-minute grid")
    for name, meter in meters.items():
        if not meter["Time"].equals(times):
            raise ValueError(f"Timestamp grid differs for {name}")

    # Equipment readings at source time t match the total-home reading at t+1.
    # All 1,545 EV blank minutes match total-home blanks after this shift.
    aligned = pd.DataFrame({"Time": times})
    aligned["total_home"] = meters["total_home"]["Value"]
    for name in EQUIPMENT_METERS:
        aligned[name] = meters[name]["Value"].shift(1)

    filled = pd.DataFrame({"Time": times})
    long_gap = pd.DataFrame(index=aligned.index)
    endpoint_change_gap = pd.DataFrame(index=aligned.index)
    for name in INPUT_FILES:
        filled[name], long_gap[name], endpoint_change_gap[name] = fill_short_stable_gaps(
            aligned[name]
        )

    source_columns = list(INPUT_FILES)
    observed = aligned[source_columns].notna().all(axis=1)
    complete = filled[source_columns].notna().all(axis=1)
    residual = filled["total_home"] - filled[list(EQUIPMENT_METERS)].sum(
        axis=1, min_count=len(EQUIPMENT_METERS)
    )
    negative = complete & residual.lt(0)
    meter_mismatch = pd.Series(False, index=aligned.index)
    for start, end in METER_MISMATCH_INTERVALS:
        meter_mismatch |= times.between(pd.Timestamp(start), pd.Timestamp(end))
    low_balance = complete & residual.ge(0) & residual.lt(MIN_PLAUSIBLE_REST_OF_HOME_W)
    valid = complete & ~negative & ~meter_mismatch & ~low_balance

    quality = pd.Series("observed", index=aligned.index)
    quality[~complete] = "excluded_long_gap"
    quality[~complete & ~long_gap.any(axis=1)] = "excluded_endpoint_change"
    quality[complete & ~observed] = "interpolated"
    quality[complete & meter_mismatch] = "excluded_meter_mismatch"
    quality[low_balance & ~meter_mismatch] = "excluded_low_balance"
    quality[negative] = "excluded_negative_balance"
    if (quality.eq("observed") & ~observed).any():
        raise AssertionError("An incomplete record was marked observed")
    if (quality.str.startswith("excluded") != ~valid).any():
        raise AssertionError("Quality flags do not match output validity")

    imputed_sources = pd.Series("", index=aligned.index, dtype="object")
    for name in source_columns:
        used_fill = valid & aligned[name].isna() & filled[name].notna()
        separator = imputed_sources.ne("").map({True: ";", False: ""})
        imputed_sources[used_fill] = (
            imputed_sources[used_fill] + separator[used_fill] + name
        )

    output = pd.DataFrame(
        {
            "Time": times.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Value": residual.where(valid),
            "Quality": quality,
            "ImputedSources": imputed_sources,
        }
    )
    output_path = DATA_DIR / "rest_of_home_load.csv"
    output.to_csv(output_path, index=False, float_format="%.9f")

    # Give each meter's unfilled source gaps separately for an auditable inventory.
    gap_tables = []
    for name in source_columns:
        for reason, mask in (
            ("over_60_min", long_gap[name]),
            ("endpoint_change_ge_100_w", endpoint_change_gap[name]),
        ):
            meter_gaps = contiguous_intervals(mask, times)
            meter_gaps.insert(0, "Meter", name)
            meter_gaps["Reason"] = reason
            if reason == "over_60_min":
                meter_gaps.loc[
                    meter_gaps["Minutes"].le(MAX_FILL_GAP_MINUTES), "Reason"
                ] = "unbounded"
            endpoint_differences = []
            for row in meter_gaps.itertuples(index=False):
                first = times.searchsorted(row.Start)
                last = times.searchsorted(row.End)
                before = aligned[name].iloc[first - 1] if first > 0 else float("nan")
                after = (
                    aligned[name].iloc[last + 1]
                    if last + 1 < len(aligned)
                    else float("nan")
                )
                endpoint_differences.append(abs(after - before))
            meter_gaps["EndpointDifferenceW"] = endpoint_differences
            gap_tables.append(meter_gaps)
    gaps = pd.concat(gap_tables, ignore_index=True).sort_values(
        ["Start", "Meter", "Reason"]
    )
    gaps = gaps[
        ["Meter", "Start", "End", "Minutes", "EndpointDifferenceW", "Reason"]
    ]
    gap_path = DATA_DIR / "rest_of_home_excluded_gaps.csv"
    gaps.to_csv(gap_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"Output: {output_path}")
    print(f"Gap intervals: {gap_path}")
    print("Timestamp offset: equipment +1 minute; fill <=60-minute gaps only when endpoint change <100 W")
    print(quality.value_counts().to_string())


if __name__ == "__main__":
    main()

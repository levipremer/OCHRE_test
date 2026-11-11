"""Figures for a heating-only HVAC predictive-control comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting as _plotting


plt = _plotting.plt
mdates = _plotting.mdates
PRICE_COLUMN = "electricity_price_usd_per_kwh"
_BASELINE = "Without predictive control"
_PREDICTIVE = "With predictive control"
_COLORS = {
    "indoor": "#202735",
    "setpoint": "#cf4f36",
    "comfort": "#197a75",
    "baseline": "#68788b",
    "predictive": "#2366a8",
    "price": "#bc762a",
}
_STYLE = {
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 120,
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} needs a DatetimeIndex")


def _aligned_series(series: pd.Series, index: pd.DatetimeIndex, name: str) -> pd.Series:
    values = pd.to_numeric(series.reindex(index), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{name} must have finite values at every simulation timestamp")
    return values


def _format_hvac_time_axis(axis: plt.Axes) -> None:
    """Keep short HVAC runs readable as well as multiday runs."""
    start, end = axis.get_xlim()
    if end - start <= 0.25:
        axis.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.grid(True, alpha=0.25)
    else:
        _plotting._format_time_axis(axis)


def _thermostat_thresholds(
    setpoint: pd.Series,
    deadband_c: float,
    deadband_offset: float,
) -> tuple[pd.Series, pd.Series]:
    """Use OCHRE's heating on/off thresholds around the actual setpoint."""
    if not np.isfinite([deadband_c, deadband_offset]).all():
        raise ValueError("Thermostat deadband and offset must be finite")
    if deadband_c < 0 or not 0 <= deadband_offset <= 1:
        raise ValueError("Thermostat deadband must be nonnegative and offset in [0, 1]")
    return (
        setpoint - deadband_c * (1 - deadband_offset),
        setpoint + deadband_c * deadband_offset,
    )


def hvac_temperature_limits(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    comfort_setpoint_c: pd.Series,
    *,
    comfort_band_c: float,
    thermostat_deadband_c: float,
    thermostat_deadband_offset: float,
) -> tuple[float, float]:
    """Share one temperature scale across the baseline and controlled plots."""
    if not np.isfinite(comfort_band_c) or comfort_band_c < 0:
        raise ValueError("Comfort band must be finite and nonnegative")
    values: list[np.ndarray] = []
    for name, frame in (("Controlled HVAC", controlled), ("Baseline HVAC", uncontrolled)):
        _require_columns(frame, {"indoor_temperature_c", "heating_setpoint_c"}, name)
        setpoint = _aligned_series(frame["heating_setpoint_c"], frame.index, "Heating setpoint")
        indoor = _aligned_series(frame["indoor_temperature_c"], frame.index, "Indoor temperature")
        thermostat_lower, thermostat_upper = _thermostat_thresholds(
            setpoint, thermostat_deadband_c, thermostat_deadband_offset
        )
        comfort = _aligned_series(comfort_setpoint_c, frame.index, "Comfort setpoint")
        values.extend(
            array.to_numpy(float)
            for array in (indoor, setpoint, thermostat_lower, thermostat_upper,
                          comfort - comfort_band_c, comfort + comfort_band_c)
        )
    all_values = np.concatenate(values)
    low, high = float(all_values.min()), float(all_values.max())
    padding = max(0.5, 0.04 * (high - low))
    return low - padding, high + padding


def hvac_power_limits(
    controlled: pd.DataFrame, uncontrolled: pd.DataFrame
) -> tuple[float, float]:
    """Share one electric-power scale across the two operation plots."""
    values = []
    for name, frame in (("Controlled HVAC", controlled), ("Baseline HVAC", uncontrolled)):
        _require_columns(frame, {"hvac_power_kw"}, name)
        values.append(_aligned_series(frame["hvac_power_kw"], frame.index, name))
    upper = max(float(series.max()) for series in values)
    return 0.0, max(1.0, upper * 1.04)


def plot_hvac_operation(
    results: pd.DataFrame,
    comfort_setpoint_c: pd.Series,
    home_id: str,
    operation: str,
    output_file: Path,
    dpi: int,
    *,
    comfort_band_c: float,
    thermostat_deadband_c: float,
    thermostat_deadband_offset: float,
    temperature_limits: tuple[float, float] | None = None,
    power_limits: tuple[float, float] | None = None,
    price: pd.Series | None = None,
) -> None:
    """Plot actual heating behavior, thermostat thresholds, and comfort bounds."""
    _require_columns(
        results,
        {"indoor_temperature_c", "heating_setpoint_c", "hvac_power_kw"},
        "HVAC operation",
    )
    if not np.isfinite(comfort_band_c) or comfort_band_c < 0:
        raise ValueError("Comfort band must be finite and nonnegative")
    index = results.index
    indoor = _aligned_series(results["indoor_temperature_c"], index, "Indoor temperature")
    thermostat_setpoint = _aligned_series(results["heating_setpoint_c"], index, "Heating setpoint")
    comfort_setpoint = _aligned_series(comfort_setpoint_c, index, "Comfort setpoint")
    power = _aligned_series(results["hvac_power_kw"], index, "HVAC electric power")
    thermostat_lower, thermostat_upper = _thermostat_thresholds(
        thermostat_setpoint, thermostat_deadband_c, thermostat_deadband_offset
    )
    comfort_lower = comfort_setpoint - comfort_band_c
    comfort_upper = comfort_setpoint + comfort_band_c

    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            2, 1, figsize=(14, 8.5), sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1]},
        )
        temperature_axis, power_axis = axes
        temperature_axis.fill_between(
            index, comfort_lower, comfort_upper, step="post",
            color=_COLORS["comfort"], alpha=0.11, label="Comfort range", zorder=1,
        )
        temperature_axis.step(
            index, comfort_lower, where="post", color=_COLORS["comfort"],
            linestyle=":", linewidth=1.35, label="Comfort bounds", zorder=2,
        )
        temperature_axis.step(
            index, comfort_upper, where="post", color=_COLORS["comfort"],
            linestyle=":", linewidth=1.35, zorder=2,
        )
        temperature_axis.fill_between(
            index, thermostat_lower, thermostat_upper, step="post",
            color=_COLORS["setpoint"], alpha=0.14,
            label="Thermostat deadband", zorder=3,
        )
        temperature_axis.step(
            index, comfort_setpoint, where="post", color=_COLORS["comfort"],
            linewidth=1.5, alpha=0.85, label="Comfort setpoint", zorder=4,
        )
        temperature_axis.step(
            index, thermostat_setpoint, where="post", color=_COLORS["setpoint"],
            linestyle="--", linewidth=1.8, label="Heating thermostat setpoint", zorder=5,
        )
        temperature_axis.plot(
            index, indoor, color=_COLORS["indoor"], linewidth=1.65,
            label="Indoor temperature", zorder=6,
        )
        temperature_axis.set_ylabel("Temperature (°C)")
        temperature_axis.set_title(f"{home_id}: HVAC {operation}", loc="left", pad=10)
        if temperature_limits is not None:
            temperature_axis.set_ylim(*temperature_limits)
        temperature_axis.legend(loc="upper right", ncol=3, framealpha=0.95)

        power_axis.step(
            index, power, where="post", color=_COLORS["predictive"],
            linewidth=1.45, label="HVAC electric power",
        )
        power_axis.set_ylim(*(power_limits if power_limits is not None else (0, None)))
        power_axis.set_ylabel("Electric power (kW)")
        if price is not None:
            aligned_price = _aligned_series(price, index, "Electricity price")
            price_axis = _plotting._add_price_axis(power_axis, aligned_price)
            _plotting._combined_legend(power_axis, price_axis, loc="upper right")
        else:
            power_axis.legend(loc="upper right")
        for axis in axes:
            _format_hvac_time_axis(axis)
        power_axis.set_xlabel("Time")
        figure.tight_layout(pad=1.5)
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def plot_hvac_weather(
    schedule: pd.DataFrame,
    index: pd.DatetimeIndex,
    home_id: str,
    output_file: Path,
    dpi: int,
) -> None:
    """Show outdoor temperature, humidity, and irradiance over the run."""
    required = {
        "Ambient Dry Bulb (C)", "Ambient Relative Humidity (-)",
        "GHI (W/m^2)", "DNI (W/m^2)", "DHI (W/m^2)",
    }
    _require_columns(schedule, required, "Weather schedule")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Weather figure needs a DatetimeIndex")
    weather = schedule.reindex(index)
    for column in required:
        weather[column] = _aligned_series(weather[column], index, column)

    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(
            index, weather["Ambient Dry Bulb (C)"], color="#376d9e",
            linewidth=1.55, label="Outdoor air temperature",
        )
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title(f"{home_id}: weather during HVAC run", loc="left", pad=10)
        axes[0].legend(loc="upper right")

        axes[1].plot(
            index, weather["Ambient Relative Humidity (-)"] * 100,
            color="#8a5794", linewidth=1.45, label="Relative humidity",
        )
        axes[1].set_ylabel("Relative humidity (%)")
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc="upper right")

        for column, label, color in (
            ("GHI (W/m^2)", "Global horizontal", "#cb8b30"),
            ("DNI (W/m^2)", "Direct normal", "#b44e43"),
            ("DHI (W/m^2)", "Diffuse horizontal", "#5f8e70"),
        ):
            axes[2].plot(index, weather[column], label=label, color=color, linewidth=1.35)
        axes[2].set_ylabel("Solar irradiance (W/m²)")
        axes[2].set_ylim(bottom=0)
        axes[2].legend(loc="upper right", ncol=3)
        for axis in axes:
            _format_hvac_time_axis(axis)
        axes[-1].set_xlabel("Time")
        figure.tight_layout(pad=1.5)
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("HVAC cost comparison needs at least two timestamps")
    deltas = np.diff(index.asi8)
    if (deltas <= 0).any() or not np.equal(deltas, deltas[0]).all():
        raise ValueError("HVAC cost comparison needs a regular time index")
    return float(pd.Timedelta(int(deltas[0]), unit="ns") / pd.Timedelta(hours=1))


def plot_hvac_power_comparison(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
    operating_summary: pd.DataFrame | None = None,
) -> None:
    """Compare HVAC power and the cumulative electricity cost of each run."""
    for name, frame in (("Controlled HVAC", controlled), ("Baseline HVAC", uncontrolled)):
        _require_columns(frame, {"hvac_power_kw"}, name)
    if not controlled.index.equals(uncontrolled.index):
        raise ValueError("HVAC comparison runs must use the same timestamps")
    index = controlled.index
    aligned_price = _aligned_series(price, index, "Electricity price")
    timestep_hours = _timestep_hours(index)
    baseline_power = _aligned_series(uncontrolled["hvac_power_kw"], index, "Baseline HVAC power")
    predictive_power = _aligned_series(controlled["hvac_power_kw"], index, "Predictive HVAC power")
    baseline_cost = (baseline_power * aligned_price * timestep_hours).cumsum()
    predictive_cost = (predictive_power * aligned_price * timestep_hours).cumsum()
    if operating_summary is not None:
        baseline_total = float(operating_summary.loc[_BASELINE, "operating_cost_usd"])
        predictive_total = float(operating_summary.loc[_PREDICTIVE, "operating_cost_usd"])
    else:
        baseline_total = float(baseline_cost.iloc[-1])
        predictive_total = float(predictive_cost.iloc[-1])

    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [1.8, 1]},
        )
        axes[0].step(
            index, baseline_power, where="post", color=_COLORS["baseline"],
            linewidth=1.5, label="Baseline",
        )
        axes[0].step(
            index, predictive_power, where="post", color=_COLORS["predictive"],
            linewidth=1.5, label="Predictive controller",
        )
        axes[0].set_title(f"{home_id}: HVAC electricity comparison", loc="left", pad=10)
        axes[0].set_ylabel("Electric power (kW)")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc="upper right", ncol=2)

        axes[1].plot(
            index, baseline_cost, color=_COLORS["baseline"], linewidth=1.7,
            label=f"Baseline: ${baseline_total:,.2f}",
        )
        axes[1].plot(
            index, predictive_cost, color=_COLORS["predictive"], linewidth=1.7,
            label=f"Predictive controller: ${predictive_total:,.2f}",
        )
        axes[1].set_ylabel("Cumulative cost ($)")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="upper left", ncol=2)
        for axis in axes:
            _format_hvac_time_axis(axis)
        axes[-1].set_xlabel("Time")
        figure.tight_layout(pad=1.5)
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def plot_hvac_optimization_plan(plan: pd.DataFrame, output_file: Path, dpi: int) -> None:
    """Plot one open-loop HVAC solve using predicted temperature and power."""
    required = {
        "indoor_temperature_c", "comfort_lower_c", "comfort_upper_c",
        "electric_power_kw", "hp_heat_w", "er_heat_w", "hp_cop", PRICE_COLUMN,
    }
    _require_columns(plan, required, "HVAC optimization plan")
    solved_at = pd.Timestamp(plan.attrs.get("solved_at", plan.index[0]))
    hp_electric_kw = plan["hp_heat_w"] / plan["hp_cop"] / 1000
    er_electric_kw = plan["er_heat_w"] / 1000

    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            2, 1, figsize=(13, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [1.6, 1]},
        )
        axes[0].fill_between(
            plan.index, plan["comfort_lower_c"], plan["comfort_upper_c"],
            step="post", color=_COLORS["comfort"], alpha=0.12,
            label="Comfort range",
        )
        axes[0].step(
            plan.index, plan["comfort_lower_c"], where="post",
            color=_COLORS["comfort"], linestyle=":", linewidth=1.2,
            label="Comfort bounds",
        )
        axes[0].step(
            plan.index, plan["comfort_upper_c"], where="post",
            color=_COLORS["comfort"], linestyle=":", linewidth=1.2,
        )
        axes[0].plot(
            plan.index, plan["indoor_temperature_c"], color=_COLORS["indoor"],
            linewidth=1.75, label="Predicted indoor temperature",
        )
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title(
            f"HVAC plan solved {solved_at:%Y-%m-%d %H:%M}", loc="left", pad=10
        )
        axes[0].legend(loc="upper right", ncol=3)

        axes[1].step(
            plan.index, plan["electric_power_kw"], where="post",
            color=_COLORS["predictive"], linewidth=1.7, label="Total electric power",
        )
        axes[1].step(
            plan.index, hp_electric_kw, where="post", color="#5f9a60",
            linewidth=1.2, label="Heat pump",
        )
        axes[1].step(
            plan.index, er_electric_kw, where="post", color="#c7784b",
            linewidth=1.2, label="Resistance heat",
        )
        axes[1].set_ylabel("Electric power (kW)")
        axes[1].set_ylim(bottom=0)
        price_axis = _plotting._add_price_axis(axes[1], plan[PRICE_COLUMN])
        _plotting._combined_legend(axes[1], price_axis, loc="upper right", ncol=2)
        for axis in axes:
            _format_hvac_time_axis(axis)
        axes[-1].set_xlabel("Time")
        figure.tight_layout(pad=1.5)
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def plot_hvac_optimization_plans(
    plans: tuple[pd.DataFrame, ...], output_directory: Path, dpi: int
) -> tuple[Path, ...]:
    """Save one horizon figure for every successful HVAC optimization solve."""
    output_directory.mkdir(parents=True, exist_ok=True)
    files = []
    for position, plan in enumerate(plans, start=1):
        solved_at = pd.Timestamp(plan.attrs.get("solved_at", plan.index[0]))
        output_file = output_directory / (
            f"hvac_plan_{position:04d}_{solved_at:%Y%m%d_%H%M%S}.png"
        )
        plot_hvac_optimization_plan(plan, output_file, dpi)
        files.append(output_file)
    return tuple(files)

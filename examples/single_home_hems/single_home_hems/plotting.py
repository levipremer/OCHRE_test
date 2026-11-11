"""Summary figures for the single-home HEMS example."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "ochre_single_home_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


WATER_HEATER_COMFORT_THRESHOLD_C = 38.0
WATER_USAGE_COLOR = "#9ecae1"


def _add_water_heater_comfort_threshold(axis: plt.Axes) -> None:
    axis.axhline(
        WATER_HEATER_COMFORT_THRESHOLD_C,
        color="0.5",
        linestyle=":",
        linewidth=1.5,
        label="Comfort threshold (38 °C)",
    )


def water_heater_temperature_limits(
    *temperatures: pd.Series | pd.DataFrame,
) -> tuple[float, float]:
    """Return padded limits shared by controlled and baseline temperature plots."""
    flattened = [
        temperature.stack() if isinstance(temperature, pd.DataFrame) else temperature
        for temperature in temperatures
    ]
    values = (
        pd.to_numeric(pd.concat(flattened), errors="coerce")
        if flattened
        else pd.Series(dtype=float)
    )
    values = values.replace([float("inf"), float("-inf")], pd.NA).dropna()
    lower = (
        min(WATER_HEATER_COMFORT_THRESHOLD_C, float(values.min()))
        if not values.empty
        else WATER_HEATER_COMFORT_THRESHOLD_C
    )
    upper = (
        max(WATER_HEATER_COMFORT_THRESHOLD_C, float(values.max()))
        if not values.empty
        else WATER_HEATER_COMFORT_THRESHOLD_C
    )
    padding = max(1.0, 0.05 * (upper - lower))
    return lower - padding, upper + padding


def _add_water_usage_axis(axis: plt.Axes, water_usage: pd.Series) -> plt.Axes:
    """Add a subdued draw profile that keeps routine usage low in the panel."""
    usage_axis = axis.twinx()
    usage_axis.step(
        water_usage.index,
        water_usage,
        where="post",
        color=WATER_USAGE_COLOR,
        alpha=0.4,
        linewidth=0.8,
        zorder=1,
    )
    usage_axis.set_ylabel("Water usage (L/min)", color=WATER_USAGE_COLOR)
    usage_axis.tick_params(axis="y", colors=WATER_USAGE_COLOR)

    finite_usage = pd.to_numeric(water_usage, errors="coerce").dropna().clip(lower=0)
    if finite_usage.empty or float(finite_usage.max()) == 0:
        usage_upper = 1.0
    else:
        # Routine draws occupy at most the lower quarter of the panel, while an
        # unusually large draw may extend through most of it.
        usage_upper = max(
            1.0,
            4.0 * float(finite_usage.quantile(0.95)),
            1.1 * float(finite_usage.max()),
        )
    usage_axis.set_ylim(0, usage_upper)

    return usage_axis


def _format_time_axis(axis: plt.Axes) -> None:
    start, end = axis.get_xlim()
    if end - start > 2:  # Matplotlib dates are measured in days.
        axis.xaxis.set_major_locator(mdates.DayLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.set_xlabel("Date")
    else:
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(True, alpha=0.25)


def _add_price_axis(axis: plt.Axes, price: pd.Series) -> plt.Axes:
    price_axis = axis.twinx()
    price_axis.step(
        price.index,
        price,
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="Electricity price",
    )
    price_axis.set_ylabel("Electricity price ($/kWh)", color="tab:orange")
    price_axis.tick_params(axis="y", colors="tab:orange")
    price_axis.set_ylim(bottom=0)
    return price_axis


def _combined_legend(axis: plt.Axes, second_axis: plt.Axes, **kwargs) -> None:
    handles, labels = axis.get_legend_handles_labels()
    second_handles, second_labels = second_axis.get_legend_handles_labels()
    axis.legend(handles + second_handles, labels + second_labels, **kwargs)


def plot_hvac(
    results: pd.DataFrame,
    schedule: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
) -> None:
    """Plot weather, indoor conditions, and HVAC power."""
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(
        results.index,
        schedule.loc[results.index, "Ambient Dry Bulb (C)"],
        color="tab:blue",
        label="Outdoor air temperature",
    )
    axes[0].set_ylabel("Temperature (°C)")
    solar_axis = axes[0].twinx()
    solar_axis.plot(
        results.index,
        schedule.loc[results.index, "GHI (W/m^2)"],
        color="tab:orange",
        alpha=0.8,
        label="Global horizontal irradiance",
    )
    solar_axis.set_ylabel("Solar irradiance (W/m²)")
    _combined_legend(axes[0], solar_axis, loc="upper left")

    axes[1].plot(
        results.index,
        results["indoor_temperature_c"],
        color="black",
        linewidth=1.8,
        label="Indoor air temperature",
    )
    for column, label, color in (
        ("heating_setpoint_c", "Heating setpoint", "tab:red"),
        ("cooling_setpoint_c", "Cooling setpoint", "tab:blue"),
    ):
        if column in results:
            axes[1].plot(results.index, results[column], "--", color=color, label=label)
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend(ncol=3)

    axes[2].plot(
        results.index,
        results["hvac_power_kw"],
        color="tab:red",
        label="HVAC power",
    )
    axes[2].set_ylabel("Power (kW)")
    axes[2].set_xlabel("Time of day")
    price_axis = _add_price_axis(axes[2], price)
    _combined_legend(axes[2], price_axis, loc="upper left")

    axes[0].set_title(f"{home_id}: HVAC")
    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_water_heater(
    results: pd.DataFrame,
    raw: pd.DataFrame,
    sensors: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
    water_usage: pd.Series | None = None,
    temperature_limits: tuple[float, float] | None = None,
    power_limits: tuple[float, float] | None = None,
) -> None:
    """Plot water-heater temperatures and power."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for column in sensors:
        node = int(column.removeprefix("water_heater_sensor_node_").split("_")[0])
        axes[0].plot(
            sensors.index,
            sensors[column],
            label=f"Sensor at node {node}",
        )
    axes[0].plot(
        results.index,
        raw.loc[results.index, "Water Heating Deadband Upper Limit (C)"],
        color="tab:red",
        linestyle="--",
        label="Tank setpoint",
    )
    _add_water_heater_comfort_threshold(axes[0])
    if temperature_limits is not None:
        axes[0].set_ylim(temperature_limits)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"{home_id}: water heater")
    if water_usage is None:
        water_usage = raw.loc[results.index, "Water Heating (L/min)"]
    _add_water_usage_axis(axes[0], water_usage)
    axes[0].legend(loc="lower left", ncol=2)

    axes[1].plot(
        results.index,
        results["water_heating_power_kw"],
        color="tab:blue",
        label="Water-heater power",
    )
    axes[1].set_ylabel("Power (kW)")
    if power_limits is not None:
        axes[1].set_ylim(power_limits)
    axes[1].set_xlabel("Time of day")
    price_axis = _add_price_axis(axes[1], price)
    _combined_legend(axes[1], price_axis, loc="upper left")

    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def _water_heater_sensor_label(column: str, position: int) -> str:
    prefix = "water_heater_sensor_node_"
    if column.startswith(prefix):
        node = int(column.removeprefix(prefix).split("_")[0])
        return f"Sensor at node {node}"
    return "Upper-tank temperature" if position == 0 else "Lower-tank temperature"


def plot_water_heater_optimization_plan(
    plan: pd.DataFrame,
    output_file: Path,
    dpi: int,
) -> None:
    """Plot one solved open-loop water-heater plan over its full horizon."""
    required = {
        "water_heater_setpoint_temperature_c",
        "water_heater_electric_power_kw",
        "electricity_price_usd_per_kwh",
        "hot_water_flow_rate_l_per_min",
    }
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"Water-heater optimization plan is missing {sorted(missing)}")
    temperature_columns = [
        column
        for column in plan.columns
        if column.startswith("water_heater_sensor_node_")
        and column.endswith("_temperature_c")
    ]
    if len(temperature_columns) != 2:
        raise ValueError("Water-heater optimization plan requires two sensor temperatures")

    scenario_count = plan.attrs.get("scenario_count")
    scenario_mean = scenario_count is not None or plan.attrs.get("forecast_type") == "scenario"
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for position, column in enumerate(temperature_columns):
        label = _water_heater_sensor_label(column, position)
        axes[0].plot(
            plan.index,
            plan[column],
            label=f"{label} (mean)" if scenario_mean else label,
        )
    axes[0].step(
        plan.index,
        plan["water_heater_setpoint_temperature_c"],
        where="post",
        color="tab:red",
        linestyle="--",
        label="Mean scenario setpoint" if scenario_mean else "Tank setpoint",
    )
    axes[0].set_ylabel("Temperature (°C)")
    solved_at = pd.Timestamp(plan.attrs.get("solved_at", plan.index[0]))
    title = solved_at.strftime("%Y-%m-%d %H:%M")
    if scenario_mean:
        title += " — scenario means"
        if scenario_count is not None:
            title += f" ({scenario_count} paths)"
    axes[0].set_title(title)

    flow_axis = axes[0].twinx()
    flow = plan["hot_water_flow_rate_l_per_min"]
    flow_axis.step(
        plan.index,
        flow,
        where="post",
        color="#8ab6d6",
        alpha=0.45,
        linewidth=0.9,
        label="Mean hot water flow rate" if scenario_mean else "Hot water flow rate",
        zorder=1,
    )
    flow_axis.set_ylabel("Hot water flow rate (L/min)", color="#6f9fbe")
    flow_axis.tick_params(axis="y", colors="#6f9fbe")
    finite_flow = flow.dropna()
    flow_max = max(float(finite_flow.max()), 0.0) if not finite_flow.empty else 0.0
    flow_axis.set_ylim(0, 4 * flow_max if flow_max > 0 else 1)
    _combined_legend(axes[0], flow_axis, loc="lower left")

    axes[1].step(
        plan.index,
        plan["water_heater_electric_power_kw"],
        where="post",
        color="tab:blue",
        label="Mean planned power" if scenario_mean else "Water-heater power",
    )
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_ylim(bottom=0)
    price_axis = _add_price_axis(
        axes[1], plan["electricity_price_usd_per_kwh"]
    )
    _combined_legend(axes[1], price_axis, loc="upper left")

    for axis in axes:
        _format_time_axis(axis)
        axis.set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_water_heater_optimization_plans(
    plans: tuple[pd.DataFrame, ...],
    output_directory: Path,
    dpi: int,
) -> tuple[Path, ...]:
    """Save one horizon figure for every successful optimization solve."""
    output_directory.mkdir(parents=True, exist_ok=True)
    files = []
    for plan in plans:
        solved_at = pd.Timestamp(plan.attrs.get("solved_at", plan.index[0]))
        output_file = output_directory / (
            f"water_heater_plan_{solved_at.strftime('%Y%m%d_%H%M%S')}.png"
        )
        plot_water_heater_optimization_plan(plan, output_file, dpi)
        files.append(output_file)
    return tuple(files)


def plot_water_heater_baseline(
    training_data: pd.DataFrame,
    sensor_columns: tuple[str, ...],
    home_id: str,
    output_file: Path,
    dpi: int,
    water_usage: pd.Series | None = None,
    temperature_limits: tuple[float, float] | None = None,
    power_limits: tuple[float, float] | None = None,
) -> None:
    """Plot tank temperatures and electrical consumption for the baseline."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    sensor_positions = ((sensor_columns[0], "Upper tank", "tab:blue"),)
    if len(sensor_columns) > 1:
        sensor_positions += ((sensor_columns[-1], "Lower tank", "tab:green"),)
    for column, label, color in sensor_positions:
        axes[0].plot(training_data.index, training_data[column], color=color, label=label)
    axes[0].plot(
        training_data.index,
        training_data["water_heater_setpoint_temperature_c"],
        color="tab:red",
        linestyle="--",
        label="Tank setpoint",
    )
    _add_water_heater_comfort_threshold(axes[0])
    if temperature_limits is not None:
        axes[0].set_ylim(temperature_limits)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"{home_id}: water-heater baseline")
    if water_usage is None:
        water_usage = training_data["water_heater_draw_l_per_min"]
    _add_water_usage_axis(axes[0], water_usage)
    axes[0].legend(loc="lower left", ncol=2)

    axes[1].plot(
        training_data.index,
        training_data["water_heater_electric_power_kw"],
        color="tab:blue",
        label="Water-heater electrical power",
    )
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_xlabel("Time of day")
    if power_limits is not None:
        axes[1].set_ylim(power_limits)
    else:
        axes[1].set_ylim(bottom=0)
    axes[1].legend()

    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_water_heater_power_comparison(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
    operating_summary: pd.DataFrame | None = None,
) -> None:
    """Compare water-heater power with and without predictive control."""
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        uncontrolled.index,
        uncontrolled["water_heating_power_kw"],
        color="0.55",
        linewidth=1.5,
        label="Without predictive control",
    )
    axis.plot(
        controlled.index,
        controlled["water_heating_power_kw"],
        color="tab:blue",
        linewidth=1.5,
        label="With predictive control",
    )
    axis.set_title(f"{home_id}: water-heater power comparison")
    axis.set_ylabel("Water-heater power (kW)")
    axis.set_xlabel("Time of day")
    price_axis = _add_price_axis(axis, price)
    _combined_legend(axis, price_axis, loc="upper left")
    _format_time_axis(axis)
    if operating_summary is not None:
        without = operating_summary.loc["Without predictive control"]
        with_control = operating_summary.loc["With predictive control"]
        figure.text(
            0.5,
            0.025,
            "Water heater only  |  "
            f"Without: {without['energy_kwh']:.2f} kWh, "
            f"USD {without['operating_cost_usd']:.2f}  |  "
            f"With: {with_control['energy_kwh']:.2f} kWh, "
            f"USD {with_control['operating_cost_usd']:.2f}",
            ha="center",
        )
        figure.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_water_usage_profile(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    home_id: str,
    output_file: Path,
    dpi: int,
) -> None:
    """Plot the water drawn from the tank in each simulation."""
    column = "water_heater_draw_l_per_min"
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(
        uncontrolled.index,
        uncontrolled[column],
        color="0.55",
        linewidth=1,
        label="Without predictive control",
    )
    axis.plot(
        controlled.index,
        controlled[column],
        color="tab:blue",
        linewidth=1,
        alpha=0.8,
        label="With predictive control",
    )
    axis.set_title(f"{home_id}: water drawn from tank")
    axis.set_ylabel("Draw (L/min)")
    axis.set_xlabel("Time of day")
    axis.set_ylim(bottom=0)
    axis.legend()
    _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_ev(
    results: pd.DataFrame,
    raw: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
) -> None:
    """Plot EV state of charge, home/away status, and charging power."""
    soc_columns = [
        column
        for column in results
        if column.startswith("ev_") and column.endswith("_soc_fraction")
        and column != "ev_soc_fraction"
    ]
    parked_columns = [
        column
        for column in raw
        if column.startswith("EV_") and column.endswith(" Parked")
    ]

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for column in soc_columns:
        vehicle = column.removeprefix("ev_").removesuffix("_soc_fraction")
        axes[0].plot(results.index, results[column], label=f"EV {vehicle} SOC")
    axes[0].set_ylabel("SOC (fraction)")
    axes[0].set_ylim(-0.02, 1.02)

    status_axis = axes[0].twinx()
    for column in parked_columns:
        vehicle = column.removeprefix("EV_").removesuffix(" Parked")
        status_axis.step(
            results.index,
            raw.loc[results.index, column].astype(int),
            where="post",
            linestyle="--",
            alpha=0.75,
            label=f"EV {vehicle} location",
        )
    status_axis.set_ylabel("Vehicle location")
    status_axis.set_yticks([0, 1], labels=["Away", "Home"])
    status_axis.set_ylim(-0.1, 1.1)
    _combined_legend(axes[0], status_axis, loc="lower left")
    axes[0].set_title(f"{home_id}: electric vehicle")

    axes[1].plot(
        results.index,
        results["ev_power_kw"],
        color="tab:green",
        label="EV charging power",
    )
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_xlabel("Time of day")
    price_axis = _add_price_axis(axes[1], price)
    _combined_legend(axes[1], price_axis, loc="upper left")

    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_ev_operation(
    results: pd.DataFrame,
    raw: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    operation: str,
    output_file: Path,
    dpi: int,
    *,
    power_limits: tuple[float, float] | None = None,
    health_soc_bounds: tuple[float, float] | None = None,
    comfort_soc_min: float | None = None,
) -> None:
    """Show one EV case with matched SOC and signed AC-power axes.

    ``results`` defines the evaluation window. The raw plant output may also
    contain warmup rows, which are deliberately excluded from these figures.
    """
    if results.empty or not isinstance(results.index, pd.DatetimeIndex):
        raise ValueError("EV operation needs nonempty timestamped results")
    if not results.index.isin(raw.index).all() or not results.index.isin(price.index).all():
        raise ValueError("EV operation raw state and tariff must cover every plotted minute")
    index = results.index
    plotted_raw = raw.loc[index]
    plotted_price = pd.to_numeric(price.loc[index], errors="coerce")
    if plotted_price.isna().any():
        raise ValueError("EV operation tariff must be finite over the plotted window")

    # Prefer the end-of-step SOC recorded by the opt-in EV plant. Legacy EVs
    # report only the current SOC, so use that when the end state is absent.
    # Exactly one trace is drawn for each vehicle.
    vehicles = sorted({
        column.removeprefix("EV_").removesuffix(" Electric Power (kW)")
        for column in plotted_raw
        if column.startswith("EV_") and column.endswith(" Electric Power (kW)")
    })
    if not vehicles:
        raise ValueError("EV operation requires per-vehicle power for at least one EV")
    if power_limits is not None and (len(power_limits) != 2 or power_limits[0] >= power_limits[1]):
        raise ValueError("EV power limits must be an increasing pair")
    if health_soc_bounds is not None and (
        len(health_soc_bounds) != 2 or not 0 <= health_soc_bounds[0] <= health_soc_bounds[1] <= 1
    ):
        raise ValueError("EV healthy SOC bounds must lie in [0,1] and be ordered")
    if comfort_soc_min is not None and not 0 <= comfort_soc_min <= 1:
        raise ValueError("EV departure comfort target must lie in [0,1]")

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    soc_axis, power_axis = axes
    if health_soc_bounds is not None:
        soc_axis.axhspan(*health_soc_bounds, color="tab:green", alpha=0.1,
                         label="Healthy SOC band (soft)")
    if comfort_soc_min is not None:
        soc_axis.axhline(comfort_soc_min, color="tab:orange", linestyle=":",
                         linewidth=1.4, label="Departure comfort target")
    location_axis = soc_axis.twinx()
    for vehicle in vehicles:
        name = f"EV_{vehicle}"
        soc_column = next((column for column in (
            f"{name} End SOC (-)", f"{name} SOC (-)"
        ) if column in plotted_raw), None)
        if soc_column is None:
            raise ValueError(f"EV operation requires SOC for {name}")
        soc_axis.plot(index, plotted_raw[soc_column], label=f"EV {vehicle} SOC")
        parked = plotted_raw.get(f"{name} Parked")
        if parked is None:
            mode = plotted_raw.get(f"{name} Mode")
            if mode is None:
                raise ValueError(f"EV operation requires parked state or mode for {name}")
            parked = mode.ne("Away")
        location_axis.step(index, parked.astype(int), where="post", linestyle="--",
                           alpha=0.65, label=f"EV {vehicle} home")
        power_column = f"{name} Electric Power (kW)"
        if power_column not in plotted_raw:
            raise ValueError(f"EV operation requires electric power for {name}")
        power_axis.step(index, plotted_raw[power_column], where="post",
                        label=f"EV {vehicle} AC power")
    soc_axis.set_title(f"{home_id}: EV {operation}")
    soc_axis.set_ylabel("SOC (fraction)")
    soc_axis.set_ylim(-0.02, 1.02)
    location_axis.set_ylabel("Vehicle location")
    location_axis.set_yticks([0, 1], labels=["Away", "Home"])
    location_axis.set_ylim(-0.1, 1.1)
    _combined_legend(soc_axis, location_axis, loc="lower left")

    power_axis.axhline(0, color="0.4", linewidth=0.9)
    power_axis.set_ylabel("Signed AC power (kW)")
    power_axis.set_xlabel("Time of day")
    if power_limits is not None:
        power_axis.set_ylim(*power_limits)
    price_axis = _add_price_axis(power_axis, plotted_price)
    _combined_legend(power_axis, price_axis, loc="upper left")
    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_whole_home_comparison(
    controlled: pd.DataFrame,
    uncontrolled: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    output_file: Path,
    dpi: int,
    *,
    power_limits: tuple[float, float] | None = None,
) -> None:
    """Compare measured one-minute power with and without HEMS control."""
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        uncontrolled.index,
        uncontrolled["total_electric_power_kw"],
        color="0.45",
        linewidth=1.5,
        label="Baseline (1 minute)",
    )
    axis.plot(
        controlled.index,
        controlled["total_electric_power_kw"],
        color="tab:blue",
        linewidth=1.5,
        label="Predictive (1 minute)",
    )
    axis.set_title(f"{home_id}: measured one-minute whole-home power")
    axis.set_ylabel("Net electric power (kW)")
    axis.set_xlabel("Time of day")
    if power_limits is not None:
        axis.set_ylim(*power_limits)
    price_axis = _add_price_axis(axis, price)
    price_axis.set_ylim(0, max(0.01, float(price.max()) * 1.15))
    _combined_legend(axis, price_axis, loc="upper left", ncol=3)
    _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_whole_home_operation(
    results: pd.DataFrame,
    price: pd.Series,
    home_id: str,
    operation: str,
    output_file: Path,
    dpi: int,
    *,
    power_limits: tuple[float, float],
) -> None:
    """Show measured grid power and the aligned averages used for peak reporting."""
    power = results["total_electric_power_kw"].astype(float)
    aligned_power = power.resample("15min").mean()
    night_bins = aligned_power.loc[
        (aligned_power.index.hour >= 3) & (aligned_power.index.hour < 7)
    ]
    night_peak_label = f"{night_bins.max():.2f} kW" if not night_bins.empty else "n/a"
    plotted_price = price.reindex(power.index)
    if plotted_price.isna().any():
        raise ValueError("Tariff series does not cover the whole-home power plot")

    figure, (power_axis, tariff_axis) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    color = "0.35" if operation == "Baseline" else "tab:blue"
    power_axis.plot(
        power.index, power, color=color, linewidth=0.65, alpha=0.45,
        label="Measured 1-minute power",
    )
    power_axis.step(
        aligned_power.index, aligned_power, where="post", color=color,
        linewidth=1.6, label="Aligned 15-minute mean",
    )
    first_peak_window = True
    for day in pd.date_range(
        power.index.min().normalize(), power.index.max().normalize(), freq="D"
    ):
        window_start = day + pd.Timedelta(hours=3)
        window_end = day + pd.Timedelta(hours=7)
        if window_start <= power.index.max() and window_end > power.index.min():
            for axis in (power_axis, tariff_axis):
                axis.axvspan(
                    window_start, window_end, color="tab:green", alpha=0.09,
                    label=(
                        "03:00–07:00 peak window"
                        if axis is power_axis and first_peak_window else None
                    ),
                )
            first_peak_window = False
    power_axis.set_title(f"{home_id}: whole-home grid power — {operation.lower()}")
    power_axis.set_ylabel("Net electric power (kW)")
    power_axis.set_ylim(*power_limits)
    power_axis.legend(loc="upper left", ncol=3, fontsize="small")
    power_axis.text(
        0.99, 0.97,
        f"All-hour 15-min peak: {aligned_power.max():.2f} kW\n"
        f"03:00–07:00 peak: {night_peak_label}",
        transform=power_axis.transAxes, ha="right", va="top", fontsize="small",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    tariff_axis.step(
        plotted_price.index, plotted_price, where="post",
        color="tab:orange", linewidth=1.2,
    )
    tariff_axis.set_ylabel("Price\n($/kWh)")
    tariff_axis.set_ylim(0, max(0.01, float(plotted_price.max()) * 1.15))
    tariff_axis.set_xlim(power.index.min(), power.index.max())
    for axis in (power_axis, tariff_axis):
        _format_time_axis(axis)
    power_axis.set_xlabel("")
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)

"""Headless Matplotlib plots for standardized home and neighborhood data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "ochre_five_home_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _format_time_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(True, alpha=0.25)


def plot_neighborhood_power(data: pd.DataFrame, output_file: Path, dpi: int) -> None:
    """Save aggregate total and end-use neighborhood power."""
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        data.index,
        data["aggregate_total_power_kw"],
        color="black",
        linewidth=2.4,
        label="Aggregate total",
        zorder=5,
    )
    component_styles = {
        "aggregate_hvac_power_kw": ("HVAC", "tab:red"),
        "aggregate_water_heating_power_kw": ("Water heating", "tab:blue"),
        "aggregate_ev_power_kw": ("EV charging", "tab:green"),
        "aggregate_rest_of_home_power_kw": ("Rest of home", "tab:purple"),
    }
    for column, (label, color) in component_styles.items():
        axis.plot(data.index, data[column], linestyle="--", alpha=0.75, label=label, color=color)
    overload_column = "transformer_overload_capacity_kva"
    if overload_column in data:
        rating = data["transformer_power_rating_kva"].iloc[0]
        overload = data[overload_column].iloc[0]
        axis.plot(
            data.index,
            data[overload_column],
            color="tab:orange",
            linestyle="-.",
            linewidth=2,
            label=f"Transformer limit ({overload:g} kVA; {rating:g} kVA rated)",
            zorder=4,
        )
    axis.set_title("Five-home block: aggregate electric demand")
    axis.set_ylabel("Home demand (kW); transformer capacity (kVA)")
    axis.set_xlabel("Time of day")
    axis.legend(ncol=3)
    _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_weather(data: pd.DataFrame, output_file: Path, dpi: int) -> None:
    """Save shared outdoor temperature, humidity, solar, and wind conditions."""
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(data.index, data["Ambient Dry Bulb (C)"], color="tab:red", linewidth=1.8)
    axes[0].set_ylabel("Outdoor temperature (°C)")
    humidity_axis = axes[0].twinx()
    humidity_axis.plot(
        data.index,
        data["Ambient Relative Humidity (-)"] * 100,
        color="tab:blue",
        linewidth=1.3,
        alpha=0.8,
    )
    humidity_axis.set_ylabel("Relative humidity (%)")
    axes[0].set_title("Lafayette weather used by all five homes")

    for column, label, style in (
        ("GHI (W/m^2)", "Global horizontal", "-"),
        ("DNI (W/m^2)", "Direct normal", "--"),
        ("DHI (W/m^2)", "Diffuse horizontal", ":"),
    ):
        axes[1].plot(data.index, data[column], linestyle=style, label=label)
    axes[1].set_ylabel("Solar irradiance (W/m²)")
    axes[1].legend(ncol=3)

    axes[2].plot(data.index, data["Wind Speed (m/s)"], color="tab:green", linewidth=1.8)
    axes[2].set_ylabel("Wind speed (m/s)")
    axes[2].set_xlabel("Time of day")

    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_home_power(data: pd.DataFrame, home_id: str, output_file: Path, dpi: int) -> None:
    """Save four-line power breakdown for one home."""
    figure, axis = plt.subplots(figsize=(11, 5))
    columns = {
        "hvac_power_kw": "HVAC",
        "water_heating_power_kw": "Water heating",
        "ev_power_kw": "EV charging",
        "rest_of_home_power_kw": "Rest of home",
    }
    for column, label in columns.items():
        axis.plot(data.index, data[column], label=label, linewidth=1.6)
    axis.set_title(f"{home_id}: electric power by end use")
    axis.set_ylabel("Electric power (kW)")
    axis.set_xlabel("Time of day")
    axis.legend(ncol=2)
    _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)


def plot_home_states(data: pd.DataFrame, home_id: str, output_file: Path, dpi: int) -> None:
    """Save indoor temperature, EV SOC, and water-tank temperatures for one home."""
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(data.index, data["indoor_temperature_c"], label="Indoor temperature", linewidth=1.8)
    if "heating_setpoint_c" in data:
        axes[0].plot(data.index, data["heating_setpoint_c"], "--", label="Heating setpoint")
    if "cooling_setpoint_c" in data:
        axes[0].plot(data.index, data["cooling_setpoint_c"], "--", label="Cooling setpoint")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"{home_id}: state variables")
    axes[0].legend(ncol=3)

    axes[1].plot(data.index, data["ev_soc_fraction"], color="tab:green", label="EV SOC")
    axes[1].set_ylabel("EV SOC (fraction)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()

    axes[2].plot(
        data.index,
        data["water_heater_temperature_c"],
        color="tab:blue",
        linewidth=1.8,
        label="Tank average",
    )
    if "water_heater_min_temperature_c" in data:
        axes[2].plot(
            data.index,
            data["water_heater_min_temperature_c"],
            linestyle=":",
            label="Tank minimum",
        )
    if "water_heater_max_temperature_c" in data:
        axes[2].plot(
            data.index,
            data["water_heater_max_temperature_c"],
            linestyle=":",
            label="Tank maximum",
        )
    axes[2].set_ylabel("Water temperature (°C)")
    axes[2].set_xlabel("Time of day")
    axes[2].legend(ncol=3)

    for axis in axes:
        _format_time_axis(axis)
    figure.tight_layout()
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)

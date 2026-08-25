"""Reusable independent-run neighborhood simulation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .aggregation import (
    aggregate_home_results,
    maximum_aggregate_error,
    maximum_power_balance_error,
)
from .config import HomeConfig, NeighborhoodConfig
from .home_factory import create_dwelling
from .output_mapping import standardize_results
from .plotting import plot_home_power, plot_home_states, plot_neighborhood_power, plot_weather


class NeighborhoodController(Protocol):
    """Future extension point for synchronized transformer-level controls."""

    def controls_for_step(
        self, timestamp: pd.Timestamp, status_by_home: dict[str, dict[str, float]]
    ) -> dict[str, dict]:
        """Return OCHRE control signals keyed by home ID."""
        ...


@dataclass(frozen=True)
class SimulationSummary:
    """Paths and high-level values from a completed neighborhood run."""

    home_results: dict[str, pd.DataFrame]
    aggregate_results: pd.DataFrame
    data_directory: Path
    figure_directory: Path
    figure_files: tuple[Path, ...]
    peak_aggregate_power_kw: float


class NeighborhoodSimulation:
    """Run independent OCHRE homes and compile synchronized neighborhood data."""

    def __init__(self, config: NeighborhoodConfig):
        self.config = config
        self.data_directory = config.output_directory / "data"
        self.raw_directory = self.data_directory / "raw"
        self.figure_directory = config.output_directory / "figures"

    def _prepare_directories(self) -> None:
        for directory in (self.data_directory, self.raw_directory, self.figure_directory):
            directory.mkdir(parents=True, exist_ok=True)

    def _expected_index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.config.start_time,
            self.config.start_time + self.config.duration,
            freq=self.config.time_step,
            inclusive="left",
            name="timestamp",
        )

    def _validate_home(
        self, home: HomeConfig, raw: pd.DataFrame, standardized: pd.DataFrame
    ) -> None:
        expected = self._expected_index()
        if not standardized.index.equals(expected):
            raise RuntimeError(
                f"{home.home_id}: expected {len(expected)} aligned timesteps, got "
                f"{len(standardized)} from {standardized.index.min()} to {standardized.index.max()}"
            )
        if not np.isfinite(standardized.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{home.home_id}: standardized results contain nonfinite values")

        tolerance = self.config.numerical_tolerance_kw
        charging_load_fields = (
            "total_electric_power_kw",
            "hvac_power_kw",
            "water_heating_power_kw",
            "ev_power_kw",
        )
        for field in charging_load_fields:
            minimum = standardized[field].min()
            if minimum < -tolerance:
                raise RuntimeError(
                    f"{home.home_id}: negative {field} found in charging-only case: {minimum} kW"
                )
        total = standardized["total_electric_power_kw"]
        if total.max() > self.config.maximum_home_power_kw:
            raise RuntimeError(
                f"{home.home_id}: power {total.max()} kW exceeds configured plausibility limit "
                f"{self.config.maximum_home_power_kw} kW"
            )
        balance_error = maximum_power_balance_error(standardized)
        if balance_error > self.config.power_balance_tolerance_kw:
            raise RuntimeError(
                f"{home.home_id}: maximum power-balance error is {balance_error} kW"
            )

        indoor = standardized["indoor_temperature_c"]
        if not indoor.between(5, 40).all():
            raise RuntimeError(
                f"{home.home_id}: implausible indoor temperature range "
                f"{indoor.min():.2f} to {indoor.max():.2f} C"
            )
        soc = standardized["ev_soc_fraction"]
        if not soc.between(-1e-6, 1 + 1e-6).all():
            raise RuntimeError(f"{home.home_id}: EV SOC is outside [0, 1]")
        water = standardized["water_heater_temperature_c"]
        if not water.between(0, 100).all():
            raise RuntimeError(
                f"{home.home_id}: implausible water temperature range "
                f"{water.min():.2f} to {water.max():.2f} C"
            )
        gas_column = "Total Gas Power (therms/hour)"
        if gas_column not in raw:
            raise RuntimeError(f"{home.home_id}: raw results are missing {gas_column}")
        if raw[gas_column].abs().max() > 1e-9:
            raise RuntimeError(f"{home.home_id}: nonzero fossil-fuel use found in all-electric case")

    def _run_home(self, home: HomeConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        print(f"Running {home.home_id}: {home.description}")
        dwelling = create_dwelling(
            self.config,
            home,
            self.raw_directory / home.home_id,
        )
        try:
            raw, _, _ = dwelling.simulate()
        except Exception as exc:
            raise RuntimeError(f"{home.home_id}: OCHRE simulation failed") from exc
        if raw is None:
            raise RuntimeError(f"{home.home_id}: OCHRE returned no time-series results")

        mapped = standardize_results(
            raw,
            home.home_id,
            numerical_tolerance_kw=self.config.numerical_tolerance_kw,
        )
        self._validate_home(home, raw, mapped.data)
        mapped.data.to_csv(self.data_directory / f"{home.home_id}_standardized.csv")
        return mapped.data, mapped.selected_columns

    def _make_figures(
        self, home_results: dict[str, pd.DataFrame], aggregate: pd.DataFrame
    ) -> tuple[Path, ...]:
        aggregate_file = self.figure_directory / "neighborhood_aggregate_power.png"
        weather_file = self.figure_directory / "shared_weather.png"
        figures = [aggregate_file, weather_file]
        plot_neighborhood_power(aggregate, aggregate_file, self.config.plot_dpi)
        plot_weather(self._load_shared_weather(), weather_file, self.config.plot_dpi)
        for home_id, data in home_results.items():
            power_file = self.figure_directory / f"{home_id}_power_breakdown.png"
            state_file = self.figure_directory / f"{home_id}_state_variables.png"
            plot_home_power(data, home_id, power_file, self.config.plot_dpi)
            plot_home_states(data, home_id, state_file, self.config.plot_dpi)
            figures.extend((power_file, state_file))
        return tuple(figures)

    def _load_shared_weather(self) -> pd.DataFrame:
        """Load the aligned weather columns from the first home's saved schedule."""
        first_home = self.config.homes[0]
        schedule_file = (
            self.raw_directory / first_home.home_id / f"{first_home.home_id}_schedule.csv"
        )
        weather = pd.read_csv(schedule_file, index_col="Time", parse_dates=True)
        weather.index.name = "timestamp"
        required = [
            "Ambient Dry Bulb (C)",
            "Ambient Relative Humidity (-)",
            "GHI (W/m^2)",
            "DNI (W/m^2)",
            "DHI (W/m^2)",
            "Wind Speed (m/s)",
        ]
        missing = [column for column in required if column not in weather]
        if missing:
            raise RuntimeError(f"Shared weather schedule is missing columns: {missing}")
        weather = weather.loc[:, required]
        if not weather.index.equals(self._expected_index()):
            raise RuntimeError("Shared weather schedule is not aligned with the simulation clock")
        return weather

    def run_independent(self) -> SimulationSummary:
        """Run all homes independently on the shared simulation clock."""
        self._prepare_directories()
        home_results: dict[str, pd.DataFrame] = {}
        mapping: dict[str, dict[str, list[str]]] = {}
        for home in self.config.homes:
            home_results[home.home_id], mapping[home.home_id] = self._run_home(home)

        aggregate = aggregate_home_results(home_results)
        aggregate_error = maximum_aggregate_error(home_results, aggregate)
        if aggregate_error > self.config.power_balance_tolerance_kw:
            raise RuntimeError(f"Neighborhood aggregation error is {aggregate_error} kW")
        aggregate = self.config.transformer.add_capacity_columns(aggregate)
        aggregate.to_csv(self.data_directory / "neighborhood_aggregate.csv")
        with (self.data_directory / "output_mapping.json").open("w", encoding="utf-8") as stream:
            json.dump(mapping, stream, indent=2)

        figure_files = self._make_figures(home_results, aggregate)
        missing_figures = [path for path in figure_files if not path.is_file()]
        if len(figure_files) != 12 or missing_figures:
            raise RuntimeError(
                f"Expected 12 figures; created {len(figure_files)} and missing {missing_figures}"
            )

        summary = SimulationSummary(
            home_results=home_results,
            aggregate_results=aggregate,
            data_directory=self.data_directory,
            figure_directory=self.figure_directory,
            figure_files=figure_files,
            peak_aggregate_power_kw=float(aggregate["aggregate_total_power_kw"].max()),
        )
        self._print_summary(summary)
        return summary

    def run_synchronized(self, controller: NeighborhoodController) -> None:
        """Reserved for future stepwise transformer or HEMS coordination."""
        raise NotImplementedError(
            "Synchronized stepping is the transformer-control extension point; "
            "the initial baseline intentionally runs homes independently."
        )

    def _print_summary(self, summary: SimulationSummary) -> None:
        energy_hours = self.config.time_step.total_seconds() / 3600
        aggregate_energy = (
            summary.aggregate_results["aggregate_total_power_kw"].sum() * energy_hours
        )
        print("\nFive-home block simulation complete")
        print(f"  Homes: {len(summary.home_results)}")
        print(f"  Timesteps per home: {len(summary.aggregate_results)}")
        print(f"  Aggregate energy: {aggregate_energy:.2f} kWh")
        print(f"  Peak aggregate power: {summary.peak_aggregate_power_kw:.2f} kW")
        print(
            f"  Transformer: {self.config.transformer.power_rating_kva:g} kVA rated, "
            f"{self.config.transformer.overload_capacity_kva:g} kVA overload capacity"
        )
        print(f"  Standardized data: {summary.data_directory}")
        print(f"  Figures: {summary.figure_directory}")

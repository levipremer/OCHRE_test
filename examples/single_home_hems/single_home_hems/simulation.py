"""Stepwise single-home simulation runner for HEMS development."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.aggregation import (
    maximum_power_balance_error,
)
from examples.five_home_block.five_home_block.config import (
    HomeConfig,
    NeighborhoodConfig,
)
from examples.five_home_block.five_home_block.home_factory import create_dwelling
from examples.five_home_block.five_home_block.output_mapping import standardize_results
from examples.five_home_block.five_home_block.plotting import (
    plot_home_power,
    plot_home_states,
)

from .controller import HEMSControllerProtocol


def select_home(config: NeighborhoodConfig, home_id: str) -> HomeConfig:
    """Select one configured home by ID, with a useful error on failure."""
    homes = {home.home_id: home for home in config.homes}
    try:
        return homes[home_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown home ID {home_id!r}; choose one of {sorted(homes)}"
        ) from exc


@dataclass(frozen=True)
class SingleHomeSimulationSummary:
    """Outputs from a completed single-home HEMS simulation."""

    home_id: str
    standardized_results: pd.DataFrame
    controls: pd.DataFrame
    data_directory: Path
    figure_directory: Path
    peak_power_kw: float


class SingleHomeHEMSSimulation:
    """Run one selected five-home-block dwelling with stepwise HEMS controls."""

    def __init__(
        self,
        config: NeighborhoodConfig,
        home: HomeConfig,
        controller: HEMSControllerProtocol,
    ):
        self.config = config
        self.home = home
        self.controller = controller
        self.data_directory = config.output_directory / "data"
        self.raw_directory = self.data_directory / "raw"
        self.figure_directory = config.output_directory / "figures"

    def _expected_index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.config.start_time,
            self.config.start_time + self.config.duration,
            freq=self.config.time_step,
            inclusive="left",
            name="timestamp",
        )

    @staticmethod
    def _validate_controls(
        controls: object, timestamp: pd.Timestamp
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(controls, dict):
            raise TypeError(
                f"HEMS controls at {timestamp} must be a dictionary, got "
                f"{type(controls).__name__}"
            )
        invalid = [name for name, values in controls.items() if not isinstance(values, dict)]
        if invalid:
            raise TypeError(
                f"HEMS equipment controls at {timestamp} must be dictionaries: {invalid}"
            )
        return controls

    def _run_stepwise(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dwelling = create_dwelling(self.config, self.home, self.raw_directory)
        previous_status: Mapping[str, Any] | None = None
        control_records: list[dict[str, Any]] = []
        try:
            for timestamp in dwelling.sim_times:
                if dwelling.current_time != timestamp:
                    raise RuntimeError(
                        f"OCHRE clock mismatch: expected {timestamp}, got "
                        f"{dwelling.current_time}"
                    )

                # Split OCHRE's normal update into its public phases so the HEMS
                # can observe current exogenous schedule inputs before deciding.
                dwelling.update_inputs()
                controls = self.controller.controls_for_step(
                    pd.Timestamp(timestamp),
                    previous_status,
                    dict(dwelling.current_schedule),
                )
                controls = self._validate_controls(controls, pd.Timestamp(timestamp))
                dwelling.update_model(control_signal=controls)
                previous_status = dwelling.update_results()
                control_records.append(
                    {
                        "timestamp": timestamp,
                        "control_signal_json": json.dumps(controls, sort_keys=True),
                    }
                )
        except Exception:
            dwelling.finalize(failed=True)
            raise

        raw, _, _ = dwelling.finalize()
        if raw is None:
            raise RuntimeError(f"{self.home.home_id}: OCHRE returned no results")
        controls = pd.DataFrame(control_records).set_index("timestamp")
        controls.index = pd.DatetimeIndex(controls.index, name="timestamp")
        return raw, controls

    def _validate_results(self, raw: pd.DataFrame, standardized: pd.DataFrame) -> None:
        expected = self._expected_index()
        if not standardized.index.equals(expected):
            raise RuntimeError(
                f"{self.home.home_id}: expected {len(expected)} aligned timesteps, "
                f"got {len(standardized)}"
            )
        if not np.isfinite(standardized.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{self.home.home_id}: results contain nonfinite values")
        balance_error = maximum_power_balance_error(standardized)
        if balance_error > self.config.power_balance_tolerance_kw:
            raise RuntimeError(
                f"{self.home.home_id}: maximum power-balance error is "
                f"{balance_error} kW"
            )
        gas_column = "Total Gas Power (therms/hour)"
        if gas_column not in raw or raw[gas_column].abs().max() > 1e-9:
            raise RuntimeError(
                f"{self.home.home_id}: expected zero gas use in the all-electric home"
            )

    def run(self) -> SingleHomeSimulationSummary:
        """Execute the selected home and save data, controls, and figures."""
        for directory in (self.data_directory, self.raw_directory, self.figure_directory):
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Running {self.home.home_id} with {type(self.controller).__name__}")
        print(f"  {self.home.description}")
        raw, controls = self._run_stepwise()
        mapped = standardize_results(
            raw,
            self.home.home_id,
            numerical_tolerance_kw=self.config.numerical_tolerance_kw,
        )
        self._validate_results(raw, mapped.data)

        results_file = self.data_directory / f"{self.home.home_id}_standardized.csv"
        controls_file = self.data_directory / f"{self.home.home_id}_hems_controls.csv"
        mapping_file = self.data_directory / "output_mapping.json"
        mapped.data.to_csv(results_file)
        controls.to_csv(controls_file)
        with mapping_file.open("w", encoding="utf-8") as stream:
            json.dump(mapped.selected_columns, stream, indent=2)

        plot_home_power(
            mapped.data,
            self.home.home_id,
            self.figure_directory / f"{self.home.home_id}_power_breakdown.png",
            self.config.plot_dpi,
        )
        plot_home_states(
            mapped.data,
            self.home.home_id,
            self.figure_directory / f"{self.home.home_id}_state_variables.png",
            self.config.plot_dpi,
        )

        summary = SingleHomeSimulationSummary(
            home_id=self.home.home_id,
            standardized_results=mapped.data,
            controls=controls,
            data_directory=self.data_directory,
            figure_directory=self.figure_directory,
            peak_power_kw=float(mapped.data["total_electric_power_kw"].max()),
        )
        print("\nSingle-home HEMS simulation complete")
        print(f"  Home: {summary.home_id}")
        print(f"  Timesteps: {len(summary.standardized_results)}")
        print(f"  Peak power: {summary.peak_power_kw:.2f} kW")
        print(f"  Data and HEMS controls: {summary.data_directory}")
        print(f"  Figures: {summary.figure_directory}")
        return summary

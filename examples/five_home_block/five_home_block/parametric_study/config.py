"""Configuration and weather-period selection for the parametric study."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ConfigurationError, NeighborhoodConfig, load_configuration


@dataclass(frozen=True)
class SeasonConfig:
    """Rule used to select and prepare one seasonal reporting period."""

    season_id: str
    selection: str
    base_temperature_c: float
    reporting_hours: int
    warmup_hours: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeasonConfig":
        config = cls(
            season_id=str(data["season_id"]),
            selection=str(data["selection"]),
            base_temperature_c=float(data["base_temperature_c"]),
            reporting_hours=int(data["reporting_hours"]),
            warmup_hours=int(data["warmup_hours"]),
        )
        if config.selection not in {
            "max_cooling_degree_hours",
            "max_heating_degree_hours",
        }:
            raise ConfigurationError(
                f"{config.season_id}: unsupported seasonal selection {config.selection!r}"
            )
        if config.reporting_hours <= 0 or config.warmup_hours < 0:
            raise ConfigurationError(
                f"{config.season_id}: reporting hours must be positive and warmup nonnegative"
            )
        return config


@dataclass(frozen=True)
class ResolvedSeason:
    """Concrete run and reporting clocks for one selected weather period."""

    season_id: str
    selection: str
    base_temperature_c: float
    run_start: dt.datetime
    report_start: dt.datetime
    report_end: dt.datetime
    run_end: dt.datetime

    @property
    def duration(self) -> dt.timedelta:
        return self.run_end - self.run_start

    def as_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field in ("run_start", "report_start", "report_end", "run_end"):
            data[field] = data[field].isoformat()
        data["duration_hours"] = self.duration.total_seconds() / 3600
        return data


@dataclass(frozen=True)
class GasFurnaceConfig:
    afue: float
    capacity_source: str
    auxiliary_power_source: str


@dataclass(frozen=True)
class GasWaterHeaterConfig:
    energy_factor: float
    burner_efficiency: float
    capacity_w: float
    ua_w_per_k: float
    tank_volume_source: str


@dataclass(frozen=True)
class ElectricResistanceWaterHeaterConfig:
    efficiency: float
    capacity_w: float
    tank_volume_source: str


@dataclass(frozen=True)
class ProfileStorageConfig:
    format: str
    compression: str


@dataclass(frozen=True)
class ParametricStudyConfig:
    """Fully loaded study settings layered over the untouched base configuration."""

    repo_root: Path
    source_file: Path
    base_config_directory: Path
    output_directory: Path
    variable_home_ids: tuple[str, ...]
    maximum_erwh_homes: int
    seasons: tuple[SeasonConfig, ...]
    gas_furnace: GasFurnaceConfig
    gas_water_heater: GasWaterHeaterConfig
    electric_resistance_water_heater: ElectricResistanceWaterHeaterConfig
    profile_storage: ProfileStorageConfig
    base: NeighborhoodConfig

    def validate(self) -> None:
        base_ids = tuple(home.home_id for home in self.base.homes)
        if "home_01" not in base_ids:
            raise ConfigurationError("The base configuration must contain home_01")
        if "home_01" in self.variable_home_ids:
            raise ConfigurationError("home_01 is fixed and cannot be a variable home")
        if len(self.variable_home_ids) != 4 or len(set(self.variable_home_ids)) != 4:
            raise ConfigurationError("Exactly four unique variable home IDs are required")
        missing = set(self.variable_home_ids) - set(base_ids)
        if missing:
            raise ConfigurationError(f"Variable homes are absent from base configuration: {missing}")
        if not 0 <= self.maximum_erwh_homes <= len(self.variable_home_ids):
            raise ConfigurationError("maximum_erwh_homes is outside the valid block range")
        season_ids = [season.season_id for season in self.seasons]
        if len(season_ids) != len(set(season_ids)) or not season_ids:
            raise ConfigurationError("Season IDs must be nonempty and unique")
        if not 0 < self.gas_furnace.afue <= 1:
            raise ConfigurationError("Gas-furnace AFUE must be in (0, 1]")
        if self.gas_furnace.capacity_source != "existing_hpxml_backup_capacity":
            raise ConfigurationError(
                "Gas-furnace capacity_source must be existing_hpxml_backup_capacity"
            )
        if (
            self.gas_furnace.auxiliary_power_source
            != "existing_hpxml_heating_equipment"
        ):
            raise ConfigurationError(
                "Unsupported gas-furnace auxiliary_power_source"
            )
        if not 0 < self.gas_water_heater.burner_efficiency <= 1:
            raise ConfigurationError("Gas-water-heater efficiency must be in (0, 1]")
        if self.gas_water_heater.energy_factor <= 0:
            raise ConfigurationError("Gas-water-heater energy factor must be positive")
        if self.gas_water_heater.capacity_w <= 0 or self.gas_water_heater.ua_w_per_k <= 0:
            raise ConfigurationError("Gas-water-heater capacity and UA must be positive")
        if self.gas_water_heater.tank_volume_source != "existing_hpxml_water_heater":
            raise ConfigurationError("Unsupported gas-water-heater tank volume source")
        if not 0 < self.electric_resistance_water_heater.efficiency <= 1:
            raise ConfigurationError("ERWH efficiency must be in (0, 1]")
        if self.electric_resistance_water_heater.capacity_w <= 0:
            raise ConfigurationError("ERWH capacity must be positive")
        if (
            self.electric_resistance_water_heater.tank_volume_source
            != "existing_hpxml_water_heater"
        ):
            raise ConfigurationError("Unsupported ERWH tank volume source")
        if self.profile_storage.format != "parquet":
            raise ConfigurationError("The study currently supports parquet profile storage only")

    def home(self, home_id: str):
        return next(home for home in self.base.homes if home.home_id == home_id)


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def load_study_configuration(
    source_file: Path, repo_root: Path | None = None
) -> ParametricStudyConfig:
    """Load study settings and their referenced existing five-home configuration."""
    source_file = source_file.resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    repo_root = repo_root.resolve()
    with source_file.open(encoding="utf-8") as stream:
        data = json.load(stream)

    base_directory = _resolve_path(repo_root, data["base_config_directory"])
    base = load_configuration(base_directory, repo_root=repo_root)
    storage = data["profile_storage"]
    config = ParametricStudyConfig(
        repo_root=repo_root,
        source_file=source_file,
        base_config_directory=base_directory,
        output_directory=_resolve_path(repo_root, data["output_directory"]),
        variable_home_ids=tuple(str(value) for value in data["variable_home_ids"]),
        maximum_erwh_homes=int(data["maximum_erwh_homes"]),
        seasons=tuple(SeasonConfig.from_dict(item) for item in data["seasons"]),
        gas_furnace=GasFurnaceConfig(**data["gas_furnace"]),
        gas_water_heater=GasWaterHeaterConfig(**data["gas_water_heater"]),
        electric_resistance_water_heater=ElectricResistanceWaterHeaterConfig(
            **data["electric_resistance_water_heater"]
        ),
        profile_storage=ProfileStorageConfig(**storage),
        base=base,
    )
    config.validate()
    return config


def _read_epw_dry_bulb(weather_file: Path, year: int) -> pd.Series:
    """Read hourly EPW dry-bulb values onto the base simulation year."""
    data = pd.read_csv(
        weather_file,
        skiprows=8,
        header=None,
        usecols=[1, 2, 3, 6],
        names=["month", "day", "hour", "dry_bulb_c"],
    )
    index = pd.to_datetime(
        {
            "year": [year] * len(data),
            "month": data["month"],
            "day": data["day"],
            "hour": data["hour"] - 1,
        }
    )
    return pd.Series(data["dry_bulb_c"].to_numpy(float), index=index)


def resolve_season(
    season: SeasonConfig, weather_file: Path, year: int
) -> ResolvedSeason:
    """Select the midnight-starting weather window with maximum degree-hours."""
    temperature = _read_epw_dry_bulb(weather_file, year)
    reporting_delta = dt.timedelta(hours=season.reporting_hours)
    candidates: list[tuple[float, pd.Timestamp]] = []
    for start in temperature.index[temperature.index.hour == 0]:
        end = start + reporting_delta - dt.timedelta(hours=1)
        window = temperature.loc[start:end]
        if len(window) != season.reporting_hours:
            continue
        if season.selection == "max_cooling_degree_hours":
            score = (window - season.base_temperature_c).clip(lower=0).sum()
        else:
            score = (season.base_temperature_c - window).clip(lower=0).sum()
        candidates.append((float(score), start))
    if not candidates:
        raise ConfigurationError(f"Could not resolve weather period for {season.season_id}")

    _, report_start_timestamp = max(candidates, key=lambda item: (item[0], -item[1].dayofyear))
    report_start = report_start_timestamp.to_pydatetime()
    report_end = report_start + reporting_delta
    run_start = report_start - dt.timedelta(hours=season.warmup_hours)
    return ResolvedSeason(
        season_id=season.season_id,
        selection=season.selection,
        base_temperature_c=season.base_temperature_c,
        run_start=run_start,
        report_start=report_start,
        report_end=report_end,
        run_end=report_end,
    )


def resolve_all_seasons(config: ParametricStudyConfig) -> tuple[ResolvedSeason, ...]:
    return tuple(
        resolve_season(season, config.base.weather_file, config.base.start_time.year)
        for season in config.seasons
    )

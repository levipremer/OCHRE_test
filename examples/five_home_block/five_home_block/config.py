"""Configuration loading and validation for the five-home example."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transformer import DistributionTransformer


class ConfigurationError(ValueError):
    """Raised when a neighborhood or home configuration is invalid."""


def _resolve_file(repo_root: Path, value: str, label: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist or is not a file: {path}")
    return path


def _parse_clock(value: str, label: str) -> dt.time:
    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ConfigurationError(f"{label} must use HH:MM format, got {value!r}") from exc

@dataclass(frozen=True)
class EVConfig:
    vehicle_id: str
    vehicle_type: str
    capacity_kwh: float
    charger_power_kw: float
    nom_commute_arrival_time: dt.time | None
    nom_commute_departure_time: dt.time | None
    nom_commute_miles: float | None
    work_from_home_probability: float
    initial_soc_fraction: float
    max_commute_distance_var_miles: float | None
    max_commute_times_var_hr: float | None
    max_roundtrip_miles: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EVConfig":
        return cls(
            vehicle_id=str(data["vehicle_id"]),
            vehicle_type=str(data["vehicle_type"]),
            capacity_kwh=float(data["capacity_kwh"]),
            charger_power_kw=float(data["charger_power_kw"]),

            nom_commute_arrival_time=(
                _parse_clock(data["nom_commute_arrival_time"], "Typical EV arrival time")
                if "nom_commute_arrival_time" in data
                else None
            ),
            nom_commute_departure_time=(
                _parse_clock(data["nom_commute_departure_time"], "Typical EV departure time")
                if "nom_commute_departure_time" in data
                else None
            ),
            nom_commute_miles=(
                float(data["nom_commute_miles"])
                if "nom_commute_miles" in data
                else None
            ),

            work_from_home_probability=float(data["work_from_home_probability"]),
            initial_soc_fraction=float(data["initial_soc_fraction"]),

            # NEW OPTIONAL FIELDS
            max_commute_distance_var_miles=(
                float(data["max_commute_distance_var_miles"])
                if "max_commute_distance_var_miles" in data
                else None
            ),
            max_commute_times_var_hr=(
                float(data["max_commute_times_var_hr"])
                if "max_commute_times_var_hr" in data
                else None
            ),
            max_roundtrip_miles=(
                float(data["max_roundtrip_miles"])
                if "max_roundtrip_miles" in data
                else None
            ),
        )

@dataclass(frozen=True)
class HomeConfig:
    """Supported per-home inputs used to create one OCHRE dwelling."""

    home_id: str
    description: str
    hpxml_file: Path
    schedule_file: Path
    seed_offset: int
    heating_setpoint_c: float
    cooling_setpoint_c: float
    heating_deadband_c: float
    heating_deadband_offset: float
    compressor_min_on_time_min: float
    compressor_min_off_time_min: float
    backup_deadband_c: float
    backup_setpoint_offset_c: float
    backup_stage_escalation_delay_min: float
    backup_number_of_stages: int | None
    backup_stage_capacities_kw: tuple[float, ...] | None
    water_heater_setpoint_c: float
    evs: tuple[EVConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], repo_root: Path) -> "HomeConfig":
        """Build and validate a home configuration from decoded JSON."""
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - set(data)
        if missing:
            raise ConfigurationError(f"Home config is missing fields: {sorted(missing)}")

        stage_capacities = data["backup_stage_capacities_kw"]
        evs_data = data["evs"]
        evs = tuple(EVConfig.from_dict(ev) for ev in evs_data)
        config = cls(
            home_id=str(data["home_id"]),
            description=str(data["description"]),
            hpxml_file=_resolve_file(repo_root, data["hpxml_file"], "HPXML file"),
            schedule_file=_resolve_file(repo_root, data["schedule_file"], "Schedule file"),
            seed_offset=int(data["seed_offset"]),
            heating_setpoint_c=float(data["heating_setpoint_c"]),
            cooling_setpoint_c=float(data["cooling_setpoint_c"]),
            heating_deadband_c=float(data["heating_deadband_c"]),
            heating_deadband_offset=float(data["heating_deadband_offset"]),
            compressor_min_on_time_min=float(data["compressor_min_on_time_min"]),
            compressor_min_off_time_min=float(data["compressor_min_off_time_min"]),
            backup_deadband_c=float(data["backup_deadband_c"]),
            backup_setpoint_offset_c=float(data["backup_setpoint_offset_c"]),
            backup_stage_escalation_delay_min=float(
                data["backup_stage_escalation_delay_min"]
            ),
            backup_number_of_stages=(
                int(data["backup_number_of_stages"])
                if data["backup_number_of_stages"] is not None
                else None
            ),
            backup_stage_capacities_kw=(
                tuple(float(value) for value in stage_capacities)
                if stage_capacities is not None
                else None
            ),
            water_heater_setpoint_c=float(data["water_heater_setpoint_c"]),
            evs = evs,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate physical ranges and supported equipment choices."""
        if not self.home_id or any(char.isspace() for char in self.home_id):
            raise ConfigurationError("home_id must be nonempty and contain no whitespace")
        positive = {
            "heating_deadband_c": self.heating_deadband_c,
            "backup_deadband_c": self.backup_deadband_c,
            "backup_setpoint_offset_c": self.backup_setpoint_offset_c,
            "backup_stage_escalation_delay_min": self.backup_stage_escalation_delay_min,
        }
        for label, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"{self.home_id}: {label} must be positive")

        for ev in self.evs:
            # Capacity must be positive
            if ev.capacity_kwh <= 0:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} capacity_kwh must be positive"
                )

            # Charger power must be positive
            if ev.charger_power_kw <= 0:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} charger_power_kw must be positive"
                )

            # Initial SOC must be between 0 and 1
            if not 0 <= ev.initial_soc_fraction <= 1:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} initial SOC must be between zero and one"
                )

            # Work-from-home probability must be between 0 and 1
            if not 0 <= ev.work_from_home_probability <= 1:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} work-from-home probability must be between zero and one"
                )

            # Vehicle type must be one of the supported types
            if ev.vehicle_type.lower() not in {"sedan", "suv", "truck"}:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} vehicle_type must be sedan, suv, or truck"
                )
            if ev.max_commute_distance_var_miles is not None and ev.max_commute_distance_var_miles < 0:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} max_commute_distance_var_miles cannot be negative"
                )

            if ev.max_commute_times_var_hr is not None and ev.max_commute_times_var_hr < 0:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} max_commute_times_var_hr cannot be negative"
                )

            if ev.max_roundtrip_miles is not None and ev.max_roundtrip_miles <= 0:
                raise ConfigurationError(
                    f"{self.home_id}: EV {ev.vehicle_id} max_roundtrip_miles must be positive"
                )

        if self.backup_number_of_stages is not None and self.backup_number_of_stages <= 0:
            raise ConfigurationError(
                f"{self.home_id}: backup_number_of_stages must be a positive integer"
            )
        if self.backup_stage_capacities_kw is not None:
            if any(value not in (5, 10) for value in self.backup_stage_capacities_kw):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage capacities must be 5 or 10 kW"
                )
            if tuple(sorted(self.backup_stage_capacities_kw, reverse=True)) != (
                self.backup_stage_capacities_kw
            ):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage capacities must list larger stages first"
                )
            if (
                self.backup_number_of_stages is not None
                and len(self.backup_stage_capacities_kw) != self.backup_number_of_stages
            ):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage count must match the capacity list length"
                )
        if not 0 <= self.heating_deadband_offset <= 1:
            raise ConfigurationError(
                f"{self.home_id}: heating_deadband_offset must be between zero and one"
            )
        nonnegative = {
            "compressor_min_on_time_min": self.compressor_min_on_time_min,
            "compressor_min_off_time_min": self.compressor_min_off_time_min,
        }
        for label, value in nonnegative.items():
            if value < 0:
                raise ConfigurationError(f"{self.home_id}: {label} cannot be negative")
        if self.cooling_setpoint_c - self.heating_setpoint_c < 1:
            raise ConfigurationError(
                f"{self.home_id}: cooling setpoint must be at least 1 C above heating setpoint"
            )


@dataclass(frozen=True)
class NeighborhoodConfig:
    """Shared simulation inputs and output validation settings."""

    repo_root: Path
    config_directory: Path
    start_time: dt.datetime
    duration: dt.timedelta
    time_step: dt.timedelta
    initialization_time: dt.timedelta | None
    weather_file: Path
    ev_trip_file: Path
    NHTS_location_CDIVMSAR_ID: int | None
    output_directory: Path
    random_seed: int
    verbosity: int
    plot_dpi: int
    numerical_tolerance_kw: float
    power_balance_tolerance_kw: float
    maximum_home_power_kw: float
    defrost_model: str
    defrost_control_type: str
    defrost_er_strategy: str
    transformer: DistributionTransformer
    homes: tuple[HomeConfig, ...]

    def validate(self) -> None:
        """Validate neighborhood-wide settings and home identifiers."""
        if self.duration <= dt.timedelta(0):
            raise ConfigurationError("duration must be positive")
        if self.time_step <= dt.timedelta(0) or self.time_step > self.duration:
            raise ConfigurationError("time step must be positive and no longer than duration")
        if self.duration % self.time_step != dt.timedelta(0):
            raise ConfigurationError("duration must be an integer multiple of the time step")
        if self.verbosity < 7:
            raise ConfigurationError("verbosity must be at least 7 to expose water-tank states")
        if self.defrost_model not in {"Legacy", "Discrete"}:
            raise ConfigurationError(
                "defrost_model must be either 'Legacy' or 'Discrete'"
            )
        if self.defrost_control_type not in {"Auto", "Timer", "Demand"}:
            raise ConfigurationError(
                "defrost_control_type must be one of 'Auto', 'Timer', or 'Demand'"
            )
        if self.defrost_er_strategy not in {"Aggressive", "Conservative"}:
            raise ConfigurationError(
                "defrost_er_strategy must be either 'Aggressive' or 'Conservative'"
            )
        if not self.homes:
            raise ConfigurationError("at least one home must be configured")
        ids = [home.home_id for home in self.homes]
        if len(ids) != len(set(ids)):
            raise ConfigurationError(f"home IDs must be unique, got {ids}")


def load_configuration(config_directory: Path, repo_root: Path | None = None) -> NeighborhoodConfig:
    """Load neighborhood and home JSON configuration files."""
    config_directory = config_directory.resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = repo_root.resolve()

    neighborhood_file = config_directory / "neighborhood.json"
    homes_file = config_directory / "homes.json"
    if not neighborhood_file.is_file() or not homes_file.is_file():
        raise ConfigurationError(
            f"Expected neighborhood.json and homes.json in {config_directory}"
        )

    with neighborhood_file.open(encoding="utf-8") as stream:
        data = json.load(stream)
    with homes_file.open(encoding="utf-8") as stream:
        home_data = json.load(stream)

    homes = tuple(HomeConfig.from_dict(item, repo_root) for item in home_data)
    output_path = Path(data["output_directory"])
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    initialization_hours = float(data.get("initialization_hours", 0))
    try:
        transformer_data = data["transformer"]
        transformer = DistributionTransformer(
            power_rating_kva=float(transformer_data["power_rating_kva"]),
            overload_capacity_percent=float(
                transformer_data["overload_capacity_percent"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid transformer configuration: {exc}") from exc

    config = NeighborhoodConfig(
        repo_root=repo_root,
        config_directory=config_directory,
        start_time=dt.datetime.fromisoformat(data["start_time"]),
        duration=dt.timedelta(hours=float(data["duration_hours"])),
        time_step=dt.timedelta(minutes=float(data["time_step_minutes"])),
        initialization_time=(
            dt.timedelta(hours=initialization_hours) if initialization_hours > 0 else None
        ),
        weather_file=_resolve_file(repo_root, data["weather_file"], "Weather file"),
        output_directory=output_path.resolve(),
        random_seed=int(data["random_seed"]),
        verbosity=int(data["verbosity"]),
        plot_dpi=int(data["plot_dpi"]),
        numerical_tolerance_kw=float(data["numerical_tolerance_kw"]),
        power_balance_tolerance_kw=float(data["power_balance_tolerance_kw"]),
        maximum_home_power_kw=float(data["maximum_home_power_kw"]),
        defrost_model=str(data.get("defrost_model", "Legacy")).strip().title(),
        defrost_control_type=str(data.get("defrost_control_type", "Auto")).strip().title(),
        defrost_er_strategy=str(
            data.get("defrost_er_strategy", "Aggressive")
        ).strip().title(),
        transformer=transformer,
        homes=homes,
        ev_trip_file=_resolve_file(repo_root, data["ev_trip_file"], "EV trip file"),
        NHTS_location_CDIVMSAR_ID=int(data.get("NHTS_location_CDIVMSAR_ID", 0)),
    )
    config.validate()
    return config

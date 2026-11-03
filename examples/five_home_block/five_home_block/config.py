"""Configuration loading and validation for the five-home example."""

from __future__ import annotations

import datetime as dt
import json
import pandas as pd
from pytz.exceptions import InvalidTimeError
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .transformer import DistributionTransformer
from .ev_config import EVOptions, validate_home_evs


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
    hems: EVOptions | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EVConfig":
        return cls(
            hems=EVOptions(**data["hems"]) if data.get("hems") is not None else None,
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
class RestOfHomeConfig:
    """Select native OCHRE electricity or the DC House forecast test measurements."""

    source: str = "ochre"
    data_file: Path | None = None
    source_start: str = "2026-09-15"
    missing_data_policy: str = "error"

    @classmethod
    def from_dict(cls, data: dict[str, Any], repo_root: Path) -> "RestOfHomeConfig":
        if not isinstance(data, dict):
            raise ConfigurationError("rest_of_home must be a JSON object")
        source = data.get("source", "ochre")
        if source not in {"ochre", "dc_house_test"}:
            raise ConfigurationError("rest_of_home.source must be ochre or dc_house_test")
        policy = data.get("missing_data_policy", "error")
        if policy not in {"error", "linear_interpolate"}:
            raise ConfigurationError(
                "rest_of_home.missing_data_policy must be error or linear_interpolate"
            )
        source_start = str(data.get("source_start", "2026-09-15"))
        try:
            dt.datetime.fromisoformat(source_start)
        except ValueError as exc:
            raise ConfigurationError("rest_of_home.source_start must be an ISO date or datetime") from exc
        data_file = data.get(
            "data_file",
            "examples/single_home_hems/external_data/dc_house_power/rest_of_home_load.csv",
        )
        return cls(
            source=source,
            data_file=(
                _resolve_file(repo_root, data_file, "DC House rest-of-home data")
                if source == "dc_house_test" else None
            ),
            source_start=source_start,
            missing_data_policy=policy,
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
    cooling_deadband_c: float
    cooling_deadband_offset: float
    compressor_min_on_time_min: float
    compressor_min_off_time_min: float
    backup_deadband_c: float
    backup_setpoint_offset_c: float
    backup_stage_escalation_delay_min: float
    backup_number_of_stages: int | None
    backup_stage_capacities_kw: float | tuple[float, ...] | None
    water_heater_setpoint_c: float
    evs: tuple[EVConfig, ...]
    water_heater_deadband_c: float | None = None
    rest_of_home: RestOfHomeConfig = RestOfHomeConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any], repo_root: Path) -> "HomeConfig":
        """Build and validate a home configuration from decoded JSON."""
        required = {field.name for field in cls.__dataclass_fields__.values()} - {
            "water_heater_deadband_c", "rest_of_home"
        }
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
            cooling_deadband_c=float(data["cooling_deadband_c"]),
            cooling_deadband_offset=float(data["cooling_deadband_offset"]),
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
                if isinstance(stage_capacities, (list, tuple))
                else (
                    float(stage_capacities)
                    if stage_capacities is not None
                    else None
                )
            ),
            water_heater_setpoint_c=float(data["water_heater_setpoint_c"]),
            evs = evs,
            water_heater_deadband_c=(
                float(data["water_heater_deadband_c"])
                if data.get("water_heater_deadband_c") is not None
                else None
            ),
            rest_of_home=RestOfHomeConfig.from_dict(data.get("rest_of_home", {}), repo_root),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate physical ranges and supported equipment choices."""
        if not self.home_id or any(char.isspace() for char in self.home_id):
            raise ConfigurationError("home_id must be nonempty and contain no whitespace")
        positive = {
            "heating_deadband_c": self.heating_deadband_c,
            "cooling_deadband_c": self.cooling_deadband_c,
            "backup_deadband_c": self.backup_deadband_c,
            "backup_setpoint_offset_c": self.backup_setpoint_offset_c,
            "backup_stage_escalation_delay_min": self.backup_stage_escalation_delay_min,
        }
        for label, value in positive.items():
            if value <= 0:
                raise ConfigurationError(f"{self.home_id}: {label} must be positive")
        if self.water_heater_deadband_c is not None and self.water_heater_deadband_c <= 0:
            raise ConfigurationError(f"{self.home_id}: water_heater_deadband_c must be positive")

        vehicle_ids = [ev.vehicle_id for ev in self.evs]
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ConfigurationError(
                f"{self.home_id}: EV vehicle IDs must be unique, got {vehicle_ids}"
            )

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
            if isinstance(self.backup_stage_capacities_kw, float):
                if self.backup_stage_capacities_kw <= 0:
                    raise ConfigurationError(
                        f"{self.home_id}: backup capacity must be positive"
                    )
            elif any(value not in (5, 10) for value in self.backup_stage_capacities_kw):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage capacities must be 5 or 10 kW"
                )
            elif tuple(sorted(self.backup_stage_capacities_kw, reverse=True)) != (
                self.backup_stage_capacities_kw
            ):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage capacities must list larger stages first"
                )
            if (
                not isinstance(self.backup_stage_capacities_kw, float)
                and self.backup_number_of_stages is not None
                and len(self.backup_stage_capacities_kw) != self.backup_number_of_stages
            ):
                raise ConfigurationError(
                    f"{self.home_id}: backup stage count must match the capacity list length"
                )
        if not 0 <= self.heating_deadband_offset <= 1:
            raise ConfigurationError(
                f"{self.home_id}: heating_deadband_offset must be between zero and one"
            )
        if not 0 <= self.cooling_deadband_offset <= 1:
            raise ConfigurationError(
                f"{self.home_id}: cooling_deadband_offset must be between zero and one"
            )
        nonnegative = {
            "compressor_min_on_time_min": self.compressor_min_on_time_min,
            "compressor_min_off_time_min": self.compressor_min_off_time_min,
        }
        for label, value in nonnegative.items():
            if value < 0:
                raise ConfigurationError(f"{self.home_id}: {label} cannot be negative")
        if self.cooling_setpoint_c < self.heating_setpoint_c:
            raise ConfigurationError(
                f"{self.home_id}: cooling setpoint cannot be below heating setpoint"
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
    heating_season: bool = False
    weather_metadata: dict[str, Any] | None = None
    ground_mains_epw_file: Path | None = None
    weather_interpolation: str = "step"
    weather_forecast_file: Path | None = None

    def __post_init__(self) -> None:
        # `dataclasses.replace` is used for CLI start-time overrides and HEMS
        # warmup windows, so normalize here as well as on initial JSON load.
        if not isinstance(self.weather_metadata, dict) or not isinstance(
            self.weather_metadata.get("timezone"), str
        ):
            return
        try:
            zone = ZoneInfo(self.weather_metadata["timezone"])
            start = pd.Timestamp(self.start_time)
            start = (
                start.tz_localize(zone, ambiguous="raise", nonexistent="raise")
                if start.tzinfo is None else start.tz_convert(zone)
            )
        except (InvalidTimeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ConfigurationError("start_time or weather_metadata.timezone is invalid or ambiguous") from exc
        object.__setattr__(self, "start_time", start)

    def validate(self) -> None:
        """Validate neighborhood-wide settings and home identifiers."""
        if self.weather_interpolation not in {"step", "linear"}:
            raise ConfigurationError("weather_interpolation must be step or linear")
        if self.weather_metadata is not None:
            if self.weather_file.suffix.lower() != ".csv":
                raise ConfigurationError("weather_metadata requires a realized weather CSV")
            if not isinstance(self.weather_metadata, dict):
                raise ConfigurationError("weather_metadata must be a JSON object")
            missing = {"latitude", "longitude", "timezone"} - self.weather_metadata.keys()
            if missing:
                raise ConfigurationError(f"weather_metadata is missing {sorted(missing)}")
            try:
                latitude = float(self.weather_metadata["latitude"])
                longitude = float(self.weather_metadata["longitude"])
                ZoneInfo(str(self.weather_metadata["timezone"]))
            except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
                raise ConfigurationError("weather_metadata needs numeric latitude/longitude and an IANA timezone") from exc
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ConfigurationError("weather_metadata latitude/longitude are out of range")
        if self.ground_mains_epw_file is not None and self.ground_mains_epw_file.suffix.lower() != ".epw":
            raise ConfigurationError("ground_mains_epw_file must be an EPW file")
        if not isinstance(self.heating_season, bool):
            raise ConfigurationError("heating_season must be a boolean")
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
        for home in self.homes:
            validate_home_evs(home.evs)
        if any(ev.hems is not None for home in self.homes for ev in home.evs):
            if self.time_step != dt.timedelta(minutes=1):
                raise ConfigurationError("HEMS EV simulation requires a one-minute time step")
            if self.start_time.second or self.start_time.microsecond:
                raise ConfigurationError("HEMS EV simulation must start on a minute boundary")
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

    return configuration_from_documents(data, home_data, config_directory, repo_root)


def configuration_from_documents(data, home_data, config_directory, repo_root):
    """Build the existing runtime configuration from in-memory documents."""
    config_directory = Path(config_directory).resolve()
    repo_root = Path(repo_root).resolve()
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

    weather_file = _resolve_file(repo_root, data["weather_file"], "Weather file")
    weather_metadata = data.get("weather_metadata")
    start_time = dt.datetime.fromisoformat(data["start_time"])
    ground_mains_epw_file = data.get("ground_mains_epw_file")
    weather_forecast_file = data.get("weather_forecast_file")
    config = NeighborhoodConfig(
        repo_root=repo_root,
        config_directory=config_directory,
        start_time=start_time,
        duration=dt.timedelta(hours=float(data["duration_hours"])),
        time_step=dt.timedelta(minutes=float(data["time_step_minutes"])),
        initialization_time=(
            dt.timedelta(hours=initialization_hours) if initialization_hours > 0 else None
        ),
        weather_file=weather_file,
        weather_metadata=weather_metadata,
        ground_mains_epw_file=(
            _resolve_file(repo_root, ground_mains_epw_file, "Ground and mains EPW file")
            if ground_mains_epw_file is not None else None
        ),
        weather_interpolation=str(data.get(
            "weather_interpolation", "linear" if weather_metadata is not None else "step"
        )),
        weather_forecast_file=(
            _resolve_file(repo_root, weather_forecast_file, "Weather forecast file")
            if weather_forecast_file is not None else None
        ),
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
        heating_season=data.get("heating_season", False),
    )
    config.validate()
    return config

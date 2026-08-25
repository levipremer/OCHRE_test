"""Profile caching and manifest generation for the parametric study."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from ochre import __version__ as ochre_version

from .config import ParametricStudyConfig, ResolvedSeason, resolve_all_seasons
from .equipment import create_scenario_dwelling
from .outputs import standardize_profile, validate_profile
from .scenarios import (
    FIXED_HOME_STATE,
    EquipmentState,
    generate_equipment_states,
    generate_scenario_manifest,
    profile_id,
)


@dataclass(frozen=True)
class ProfileRequest:
    season: ResolvedSeason
    home_id: str
    state: EquipmentState
    fixed: bool = False

    @property
    def profile_id(self) -> str:
        return profile_id(self.season.season_id, self.home_id, self.state, self.fixed)


@dataclass(frozen=True)
class StudyRunSummary:
    output_directory: Path
    profile_catalog_file: Path
    scenario_manifest_file: Path | None
    profile_count: int
    scenario_count: int
    cached_profile_count: int
    smoke_test: bool


class ParametricStudyRunner:
    """Run reusable OCHRE profiles and write a transformer-agnostic manifest."""

    def __init__(self, config: ParametricStudyConfig):
        self.config = config
        self.profiles_directory = config.output_directory / "profiles"
        self.provenance_directory = config.output_directory / "provenance"
        self.scenarios_directory = config.output_directory / "scenarios"
        self.validation_directory = config.output_directory / "validation"
        self.input_signature = self._calculate_input_signature()

    def _calculate_input_signature(self) -> str:
        """Fingerprint code, configs, and home inputs that define cached profiles."""
        digest = hashlib.sha256(f"OCHRE:{ochre_version}".encode())
        source_files = [
            self.config.source_file,
            self.config.base_config_directory / "neighborhood.json",
            self.config.base_config_directory / "homes.json",
            Path(__file__),
            Path(__file__).with_name("equipment.py"),
            Path(__file__).with_name("cooking.py"),
            Path(__file__).with_name("outputs.py"),
        ]
        for home in self.config.base.homes:
            source_files.extend((home.hpxml_file, home.schedule_file))
        for path in source_files:
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _prepare_directories(self) -> None:
        for directory in (
            self.profiles_directory,
            self.provenance_directory,
            self.scenarios_directory,
            self.validation_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _profile_requests(
        self, seasons: Iterable[ResolvedSeason]
    ) -> tuple[ProfileRequest, ...]:
        requests: list[ProfileRequest] = []
        states = generate_equipment_states()
        for season in seasons:
            requests.append(ProfileRequest(season, "home_01", FIXED_HOME_STATE, fixed=True))
            for home_id in self.config.variable_home_ids:
                requests.extend(ProfileRequest(season, home_id, state) for state in states)
        return tuple(requests)

    def _profile_path(self, request: ProfileRequest) -> Path:
        return self.profiles_directory / f"{request.profile_id}.parquet"

    @staticmethod
    def _profile_metadata_path(profile_path: Path) -> Path:
        return profile_path.with_suffix(".metadata.json")

    def _run_profile(self, request: ProfileRequest, force: bool) -> tuple[dict, bool]:
        path = self._profile_path(request)
        metadata_path = self._profile_metadata_path(path)
        cache_matches = False
        if path.is_file() and metadata_path.is_file() and not force:
            with metadata_path.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            cache_matches = metadata.get("input_signature") == self.input_signature
        if cache_matches:
            data = pd.read_parquet(path)
            validate_profile(
                data,
                request.season,
                self.config.base.time_step,
                self.config.base.maximum_home_power_kw,
            )
            return self._catalog_row(request, data, path), True

        home = self.config.home(request.home_id)
        print(f"Running {request.profile_id}")
        dwelling = create_scenario_dwelling(
            self.config, home, request.season, request.state
        )
        try:
            raw, _, _ = dwelling.simulate()
        except Exception as exc:
            raise RuntimeError(f"OCHRE profile failed: {request.profile_id}") from exc
        if raw is None:
            raise RuntimeError(f"OCHRE returned no profile: {request.profile_id}")
        data = standardize_profile(raw, request.season)
        validate_profile(
            data,
            request.season,
            self.config.base.time_step,
            self.config.base.maximum_home_power_kw,
        )
        data.to_parquet(
            path,
            compression=self.config.profile_storage.compression,
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "profile_id": request.profile_id,
                    "input_signature": self.input_signature,
                    "ochre_version": ochre_version,
                },
                stream,
                indent=2,
            )
        return self._catalog_row(request, data, path), False

    def _catalog_row(
        self, request: ProfileRequest, data: pd.DataFrame, path: Path
    ) -> dict[str, object]:
        reporting = data.loc[data["is_reporting_period"]]
        return {
            "profile_id": request.profile_id,
            "season_id": request.season.season_id,
            "home_id": request.home_id,
            "fixed_home": request.fixed,
            "heating_state": request.state.heating,
            "water_heating_state": request.state.water_heating,
            "cooking_state": request.state.cooking,
            "run_start": request.season.run_start.isoformat(),
            "report_start": request.season.report_start.isoformat(),
            "report_end": request.season.report_end.isoformat(),
            "run_rows": len(data),
            "reporting_rows": len(reporting),
            "peak_reporting_electric_power_kw": float(
                reporting["total_electric_power_kw"].max()
            ),
            "reporting_electric_energy_kwh": float(
                reporting["total_electric_power_kw"].sum()
                * self.config.base.time_step.total_seconds()
                / 3600
            ),
            "reporting_gas_energy_therms": float(
                reporting["total_gas_power_therms_per_hour"].sum()
                * self.config.base.time_step.total_seconds()
                / 3600
            ),
            "profile_file": str(path.relative_to(self.config.output_directory)),
        }

    def _write_provenance(self, seasons: tuple[ResolvedSeason, ...]) -> None:
        with self.config.source_file.open(encoding="utf-8") as stream:
            study_source = json.load(stream)
        base_files = {}
        for name in ("neighborhood.json", "homes.json"):
            path = self.config.base_config_directory / name
            with path.open(encoding="utf-8") as stream:
                base_files[name] = json.load(stream)
        with (self.provenance_directory / "base_config_snapshot.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(base_files, stream, indent=2)
        resolved = {
            "study_source_file": str(self.config.source_file),
            "base_config_directory": str(self.config.base_config_directory),
            "output_directory": str(self.config.output_directory),
            "input_signature": self.input_signature,
            "ochre_version": ochre_version,
            "study": study_source,
            "resolved_seasons": [season.as_json_dict() for season in seasons],
        }
        with (self.provenance_directory / "resolved_study_config.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(resolved, stream, indent=2)

    def run(self, *, force: bool = False) -> StudyRunSummary:
        """Run or reuse all 98 profiles and write the 24,576-row manifest."""
        self._prepare_directories()
        seasons = resolve_all_seasons(self.config)
        self._write_provenance(seasons)
        rows = []
        cached = 0
        for request in self._profile_requests(seasons):
            row, was_cached = self._run_profile(request, force)
            rows.append(row)
            cached += int(was_cached)
        catalog = pd.DataFrame(rows).sort_values("profile_id")
        catalog_file = self.validation_directory / "profile_summary.csv"
        catalog.to_csv(catalog_file, index=False)

        manifest = generate_scenario_manifest(
            tuple(season.season_id for season in seasons),
            self.config.variable_home_ids,
            self.config.maximum_erwh_homes,
        )
        manifest_csv = self.scenarios_directory / "scenario_manifest.csv"
        manifest_parquet = self.scenarios_directory / "scenario_manifest.parquet"
        manifest.to_csv(manifest_csv, index=False)
        manifest.to_parquet(
            manifest_parquet, compression=self.config.profile_storage.compression
        )
        self._write_run_manifest(catalog, manifest, smoke_test=False)
        return StudyRunSummary(
            output_directory=self.config.output_directory,
            profile_catalog_file=catalog_file,
            scenario_manifest_file=manifest_parquet,
            profile_count=len(catalog),
            scenario_count=len(manifest),
            cached_profile_count=cached,
            smoke_test=False,
        )

    def run_smoke_test(
        self, *, force: bool = True, reporting_hours: int = 6
    ) -> StudyRunSummary:
        """Run three short representative profiles without creating a full manifest."""
        self._prepare_directories()
        seasons = resolve_all_seasons(self.config)
        winter = next(
            (season for season in seasons if season.season_id == "winter"), seasons[0]
        )
        smoke_season = replace(
            winter,
            season_id="smoke",
            run_start=winter.report_start,
            report_end=winter.report_start + dt.timedelta(hours=reporting_hours),
            run_end=winter.report_start + dt.timedelta(hours=reporting_hours),
        )
        requests = (
            ProfileRequest(smoke_season, "home_01", FIXED_HOME_STATE, fixed=True),
            ProfileRequest(
                smoke_season, self.config.variable_home_ids[0], EquipmentState("gas", "gas", "gas")
            ),
            ProfileRequest(
                smoke_season,
                self.config.variable_home_ids[1],
                EquipmentState("electric", "erwh", "electric"),
            ),
        )
        self._write_provenance((smoke_season,))
        rows = []
        cached = 0
        for request in requests:
            row, was_cached = self._run_profile(request, force)
            rows.append(row)
            cached += int(was_cached)
        catalog = pd.DataFrame(rows).sort_values("profile_id")
        catalog_file = self.validation_directory / "smoke_profile_summary.csv"
        catalog.to_csv(catalog_file, index=False)
        self._write_run_manifest(catalog, None, smoke_test=True)
        return StudyRunSummary(
            output_directory=self.config.output_directory,
            profile_catalog_file=catalog_file,
            scenario_manifest_file=None,
            profile_count=len(catalog),
            scenario_count=0,
            cached_profile_count=cached,
            smoke_test=True,
        )

    def _write_run_manifest(
        self,
        catalog: pd.DataFrame,
        scenarios: pd.DataFrame | None,
        *,
        smoke_test: bool,
    ) -> None:
        manifest = {
            "smoke_test": smoke_test,
            "profile_count": len(catalog),
            "scenario_count": 0 if scenarios is None else len(scenarios),
            "profile_ids": catalog["profile_id"].tolist(),
        }
        with (self.provenance_directory / "run_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)

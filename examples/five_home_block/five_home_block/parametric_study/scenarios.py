"""Equipment-state enumeration and reconstructable neighborhood manifests."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, order=True)
class EquipmentState:
    """Appliance and EV choices for one variable home."""

    heating: str
    water_heating: str
    cooking: str
    ev: str

    def __post_init__(self) -> None:
        if self.heating not in {"gas", "electric"}:
            raise ValueError(f"Unknown heating state: {self.heating}")
        if self.water_heating not in {"gas", "hpwh", "erwh"}:
            raise ValueError(f"Unknown water-heating state: {self.water_heating}")
        if self.cooking not in {"gas", "electric"}:
            raise ValueError(f"Unknown cooking state: {self.cooking}")
        if self.ev not in {"none", "present"}:
            raise ValueError(f"Unknown EV state: {self.ev}")

    @property
    def state_id(self) -> str:
        return (
            f"heat-{self.heating}__water-{self.water_heating}__"
            f"cook-{self.cooking}__ev-{self.ev}"
        )

    @property
    def electrified_end_uses(self) -> int:
        return int(self.heating == "electric") + int(
            self.water_heating in {"hpwh", "erwh"}
        ) + int(self.cooking == "electric")

    @property
    def has_erwh(self) -> bool:
        return self.water_heating == "erwh"

    @property
    def has_ev(self) -> bool:
        return self.ev == "present"


FIXED_HOME_STATE = EquipmentState("electric", "hpwh", "electric", "present")


def generate_equipment_states() -> tuple[EquipmentState, ...]:
    """Return the 24 appliance/EV states for one variable home."""
    return tuple(
        EquipmentState(heating, water_heating, cooking, ev)
        for heating, water_heating, cooking, ev in itertools.product(
            ("gas", "electric"),
            ("gas", "hpwh", "erwh"),
            ("gas", "electric"),
            ("none", "present"),
        )
    )


def profile_id(season_id: str, home_id: str, state: EquipmentState, fixed: bool = False) -> str:
    state_id = "fixed-electric" if fixed else state.state_id
    return f"{season_id}__{home_id}__{state_id}"


def generate_scenario_manifest(
    season_ids: tuple[str, ...],
    variable_home_ids: tuple[str, ...],
    maximum_erwh_homes: int,
) -> pd.DataFrame:
    """Enumerate valid block scenarios and map each to cached profile IDs."""
    states = generate_equipment_states()
    rows: list[dict[str, object]] = []
    for season_id in season_ids:
        scenario_number = 0
        for selected in itertools.product(states, repeat=len(variable_home_ids)):
            erwh_count = sum(state.has_erwh for state in selected)
            if erwh_count > maximum_erwh_homes:
                continue
            scenario_number += 1
            electric_heating = sum(state.heating == "electric" for state in selected)
            electric_water = sum(
                state.water_heating in {"hpwh", "erwh"} for state in selected
            )
            electric_cooking = sum(state.cooking == "electric" for state in selected)
            ev_count = sum(state.has_ev for state in selected)
            conversions = sum(state.electrified_end_uses for state in selected)
            row: dict[str, object] = {
                "scenario_id": f"{season_id}__scenario-{scenario_number:05d}",
                "season_id": season_id,
                "home_01_profile_id": profile_id(
                    season_id, "home_01", FIXED_HOME_STATE, fixed=True
                ),
                "electric_heating_homes": electric_heating,
                "electric_water_heating_homes": electric_water,
                "hpwh_homes": sum(state.water_heating == "hpwh" for state in selected),
                "erwh_homes": erwh_count,
                "electric_cooking_homes": electric_cooking,
                "ev_homes": ev_count,
                "whole_block_ev_homes": 1 + ev_count,
                "variable_home_electrified_end_uses": conversions,
                "variable_home_electrification_fraction": conversions / 12,
                "whole_block_electrification_fraction": (3 + conversions) / 15,
                "variable_home_electrification_with_ev_fraction": (
                    conversions + ev_count
                ) / 16,
                "whole_block_electrification_with_ev_fraction": (
                    4 + conversions + ev_count
                ) / 20,
            }
            for home_id, state in zip(variable_home_ids, selected):
                row[f"{home_id}_state_id"] = state.state_id
                row[f"{home_id}_profile_id"] = profile_id(season_id, home_id, state)
            rows.append(row)
    return pd.DataFrame(rows)

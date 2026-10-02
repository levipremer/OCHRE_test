"""Virtual water-heater sensor layout used for model training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaterHeaterSensorLayout:
    """Map virtual temperature sensors to equal-height stratified tank nodes.

    OCHRE numbers nodes from top to bottom. Sensor height is reported from the
    bottom of the tank at the center of the selected node.
    """

    model_nodes: int = 12
    sensor_nodes: tuple[int, ...] = (3, 10)

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_nodes, bool)
            or not isinstance(self.model_nodes, int)
            or self.model_nodes < 1
        ):
            raise ValueError("Water-heater model_nodes must be a positive integer")
        if not self.sensor_nodes:
            raise ValueError("At least one water-heater sensor node is required")
        if len(set(self.sensor_nodes)) != len(self.sensor_nodes):
            raise ValueError("Water-heater sensor nodes must be unique")
        invalid = [
            node
            for node in self.sensor_nodes
            if isinstance(node, bool)
            or not isinstance(node, int)
            or not 1 <= node <= self.model_nodes
        ]
        if invalid:
            raise ValueError(
                f"Sensor nodes {invalid} are outside the 1-{self.model_nodes} tank"
            )

    @staticmethod
    def default_sensor_nodes(model_nodes: int) -> tuple[int, ...]:
        """Return distinct upper and lower sensor nodes near 80% and 20% height."""
        upper_node = min(model_nodes, model_nodes // 5 + 1)
        lower_node = model_nodes - upper_node + 1
        candidates = (upper_node, lower_node)
        return tuple(dict.fromkeys(candidates))

    def height_fraction_from_bottom(self, node: int) -> float:
        """Return the selected node center as a fraction of total tank height."""
        return (self.model_nodes - node + 0.5) / self.model_nodes

    def column_name(self, node: int) -> str:
        """Return the stable training-data column name for a sensor."""
        return f"water_heater_sensor_node_{node:02d}_temperature_c"

    @property
    def sensor_columns(self) -> tuple[str, ...]:
        """Return the sensor columns in configured top-to-bottom order."""
        return tuple(self.column_name(node) for node in self.sensor_nodes)

    def measure(self, model_states: np.ndarray) -> dict[str, float]:
        """Sample the configured node temperatures from an OCHRE tank state."""
        if len(model_states) != self.model_nodes:
            raise ValueError(
                f"Expected {self.model_nodes} water-heater states, "
                f"got {len(model_states)}"
            )
        return {
            self.column_name(node): float(model_states[node - 1])
            for node in self.sensor_nodes
        }

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serializable description of the virtual sensors."""
        return {
            "model_nodes": self.model_nodes,
            "node_order": "top_to_bottom",
            "height_reference": "fraction_from_tank_bottom_to_node_center",
            "sensors": [
                {
                    "node": node,
                    "height_fraction_from_bottom": self.height_fraction_from_bottom(
                        node
                    ),
                    "column": self.column_name(node),
                }
                for node in self.sensor_nodes
            ],
        }


def water_heater_training_record(
    water_heater: Any,
    sensor_layout: WaterHeaterSensorLayout,
    initial_states: np.ndarray,
) -> dict[str, float | str]:
    """Capture one unambiguous state transition over a simulation interval.

    Delivered heat is the identified plant input, the initial sensor
    temperatures provide state history, and the next sensor temperatures are
    outputs. Setpoint, electric power, and mode are retained for diagnostics
    and for the separate command-to-delivered-heat wrapper.
    """
    tank = water_heater.model
    tank_ambient = tank.current_schedule.get("Zone Temperature (C)")
    if tank_ambient is None:
        raise ValueError("Water-heater tank ambient temperature is unavailable")

    initial_temperatures = sensor_layout.measure(initial_states)
    next_temperatures = {
        f"next_{column}": temperature
        for column, temperature in sensor_layout.measure(tank.next_states).items()
    }

    return {
        **initial_temperatures,
        "water_heater_setpoint_temperature_c": float(water_heater.setpoint_temp),
        "water_heater_draw_l_per_min": float(tank.draw_total),
        "water_heater_mains_temperature_c": float(tank.mains_temp),
        "water_heater_tank_ambient_temperature_c": float(tank_ambient),
        **next_temperatures,
        "water_heater_electric_power_kw": float(water_heater.electric_kw),
        "water_heater_delivered_heat_kw": float(water_heater.delivered_heat) / 1000,
        "water_heater_mode": str(water_heater.mode),
        "water_heater_tank_volume_l": float(tank.volume),
        "water_heater_resistance_capacity_w": float(water_heater.capacity_rated),
        "water_heater_nominal_heat_pump_capacity_w": float(
            water_heater.hp_capacity_nominal
        ),
        "water_heater_nominal_heat_pump_cop": float(water_heater.cop_nominal),
    }

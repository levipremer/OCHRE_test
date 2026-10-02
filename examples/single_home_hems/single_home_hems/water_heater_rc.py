"""Two-node water-heater physics shared by fitting and predictive control.

Power and heat are in kW, time in hours, capacitance in kWh/K, and thermal
resistance in K/kW. The thermostat is used only for open-loop identification;
the MPC retains a continuous heat decision to keep its problem convex.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.linalg import expm


WATER_HEAT_CAPACITY_KWH_PER_L_K = 4.183 / 3600
WATER_CONDUCTIVITY_KW_PER_M_K = 0.6406 / 1000


@dataclass(frozen=True)
class WaterHeaterRCModel:
    volume_l: float
    height_m: float
    ua_w_per_k: float
    upper_fraction: float
    upper_heat_fraction: float
    cop: float
    on_power_kw: float
    off_power_kw: float
    deadband_c: float
    stratification_thickness_m: float | None = None

    def __post_init__(self) -> None:
        if not all(value > 0 for value in (self.volume_l, self.height_m, self.ua_w_per_k, self.cop, self.on_power_kw, self.deadband_c)):
            raise ValueError("RC tank dimensions, UA, COP, on power, and deadband must be positive")
        if not 0 < self.upper_fraction < 1:
            raise ValueError("Upper-node fraction must lie strictly between 0 and 1")
        if not 0 <= self.upper_heat_fraction <= 1:
            raise ValueError("Upper heat fraction must lie between 0 and 1")
        if not 0 <= self.off_power_kw < self.on_power_kw:
            raise ValueError("Off power must be nonnegative and below on power")
        if self.stratification_thickness_m is not None and self.stratification_thickness_m <= 0:
            raise ValueError("Stratification thickness must be positive")

    @property
    def heat_capacity_kwh_per_k(self) -> float:
        return self.volume_l * WATER_HEAT_CAPACITY_KWH_PER_L_K

    @property
    def ambient_resistance_k_per_kw(self) -> float:
        return 1000 / self.ua_w_per_k

    @property
    def inter_node_resistance_k_per_kw(self) -> float:
        area_m2 = self.volume_l / 1000 / self.height_m
        thickness_m = (
            self.height_m / 2
            if self.stratification_thickness_m is None
            else self.stratification_thickness_m
        )
        return thickness_m / (WATER_CONDUCTIVITY_KW_PER_M_K * area_m2)

    @property
    def maximum_heat_kw(self) -> float:
        return self.cop * (self.on_power_kw - self.off_power_kw)

    def metadata(self) -> dict[str, float | None]:
        return asdict(self)


def load_rc_model(metadata_path: Path) -> WaterHeaterRCModel:
    """Load the fitted physical parameters without thermostat assumptions."""
    with Path(metadata_path).open(encoding="utf-8") as stream:
        return WaterHeaterRCModel(**json.load(stream)["parameters"])


def rc_transition(
    model: WaterHeaterRCModel,
    draw_l_per_min: float,
    timestep_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold state transition and forcing integral."""
    z = model.upper_fraction
    capacity = model.heat_capacity_kwh_per_k
    conductance = 1 / model.inter_node_resistance_k_per_kw
    ambient_ua = model.ua_w_per_k / 1000
    flow_cp = max(draw_l_per_min, 0.0) * 60 * WATER_HEAT_CAPACITY_KWH_PER_L_K
    continuous = np.array([
        [-(conductance + z * ambient_ua + flow_cp) / (z * capacity),
         (conductance + flow_cp) / (z * capacity)],
        [conductance / ((1 - z) * capacity),
         -(conductance + (1 - z) * ambient_ua + flow_cp) / ((1 - z) * capacity)],
    ])
    transition = expm(continuous * timestep_hours)
    integral = np.linalg.solve(continuous, transition - np.eye(2))
    return transition, integral


def rc_step_matrices(
    model: WaterHeaterRCModel,
    draw_l_per_min: float,
    mains_temperature_c: float,
    ambient_temperature_c: float,
    timestep_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ZOH dynamics in matrix form for the convex controller."""
    transition, integral = rc_transition(model, draw_l_per_min, timestep_hours)
    z = model.upper_fraction
    capacity = model.heat_capacity_kwh_per_k
    ambient_ua = model.ua_w_per_k / 1000
    flow_cp = max(draw_l_per_min, 0.0) * 60 * WATER_HEAT_CAPACITY_KWH_PER_L_K
    heat = np.array([
        model.upper_heat_fraction / (z * capacity),
        (1 - model.upper_heat_fraction) / ((1 - z) * capacity),
    ])
    forcing = np.array([
        ambient_ua * ambient_temperature_c / capacity,
        (ambient_ua * ambient_temperature_c + flow_cp * mains_temperature_c / (1 - z)) / capacity,
    ])
    return transition, integral @ heat, integral @ forcing


def reconstruct_heat_input(frame: pd.DataFrame, model: WaterHeaterRCModel) -> pd.DataFrame:
    """Infer interval heat from observed sensor changes and the RC balance.

    This inverse uses temperatures, draw, mains, and ambient conditions; it
    does not use electric power, COP, or the emulated thermostat. Negative
    estimates are retained as model residuals rather than hidden by clipping.
    """
    upper = frame["water_heater_sensor_node_03_temperature_c"].to_numpy(float)
    lower = frame["water_heater_sensor_node_10_temperature_c"].to_numpy(float)
    next_upper = frame["next_water_heater_sensor_node_03_temperature_c"].to_numpy(float)
    next_lower = frame["next_water_heater_sensor_node_10_temperature_c"].to_numpy(float)
    draw = frame["water_heater_draw_l_per_min"].to_numpy(float)
    mains = frame["water_heater_mains_temperature_c"].to_numpy(float)
    ambient = frame["water_heater_tank_ambient_temperature_c"].to_numpy(float)
    step_hours = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    observed = np.column_stack((upper, lower))
    following = np.column_stack((next_upper, next_lower))
    estimates = np.empty(len(frame))
    for rate in np.unique(draw):
        positions = np.flatnonzero(draw == rate)
        transition, integral = rc_transition(model, float(rate), step_hours)
        z = model.upper_fraction
        capacity = model.heat_capacity_kwh_per_k
        ua_kw = model.ua_w_per_k / 1000
        flow_cp = max(float(rate), 0.0) * 60 * WATER_HEAT_CAPACITY_KWH_PER_L_K
        heat_gain = integral @ np.array([
            model.upper_heat_fraction / (z * capacity),
            (1 - model.upper_heat_fraction) / ((1 - z) * capacity),
        ])
        forcing = np.column_stack((
            ua_kw * ambient[positions] / capacity,
            (ua_kw * ambient[positions] + flow_cp * mains[positions] / (1 - z)) / capacity,
        ))
        unheated = observed[positions] @ transition.T + forcing @ integral.T
        estimates[positions] = (following[positions] - unheated) @ heat_gain / (heat_gain @ heat_gain)
    return pd.DataFrame({
        "reference_kw": frame["water_heater_delivered_heat_kw"].to_numpy(float),
        "reconstructed_kw": estimates,
    }, index=frame.index)


def thermostat_next_mode(
    upper_c: float,
    lower_c: float,
    setpoint_c: float,
    deadband_c: float,
    was_on: bool,
) -> bool:
    """OCHRE HP-only hysteresis, without its minimum on/off timers."""
    control_c = 0.75 * upper_c + 0.25 * lower_c
    if control_c < setpoint_c - deadband_c:
        return True
    if control_c >= setpoint_c or upper_c >= setpoint_c + 1:
        return False
    return was_on

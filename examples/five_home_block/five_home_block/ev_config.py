"""Opt-in, per-vehicle HEMS settings. Legacy EVs do not use these settings."""
from dataclasses import dataclass, field
from typing import Any
import math


@dataclass(frozen=True)
class EVOptions:
    """Validated EV physics/mobility and internal controls; not the unified file schema."""
    consumption_kwh_per_km: float
    charge_efficiency: float
    discharge_efficiency: float
    max_discharge_kw: float
    # Legacy aliases are accepted by old Python callers, not unified JSON.
    health_min_soc: float | None = None
    health_max_soc: float | None = None
    target_soc_min: float | None = None
    target_soc_max: float | None = None
    health_soc_min: float | None = None
    health_soc_max: float | None = None
    comfort_soc_min: float | None = None
    health_soc_weight: float | None = None
    health_weight: float = 1
    comfort_weight: float = 0
    enable_degradation: bool = False
    degradation_method: str = "rainflow"
    rainflow_relative_tolerance: float = 0.001
    rainflow_absolute_tolerance: float = 1e-6
    rainflow_max_iterations: int = 100
    battery_replacement_cost_per_kwh: float = 119
    battery_residual_value_per_kwh: float = 0
    initial_soh: float = 1.0  # Python callers; unified documents require explicit value.
    calibration_end_of_life_soh: float = 0.8
    end_of_life_soh: float = 0.7
    reference_cycle_depth: float = 0.8
    reference_mean_soc: float = 0.6
    reference_cycle_count: float = 3000
    rainflow_exponent: float = 2.03
    soc_stress_coefficient: float = 1.04
    controller: str = "baseline"
    v2h: bool = False
    formulation: str = "convex"
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=dict)
    horizon_hours: float = 12
    control_minutes: int = 15
    cycling: str = "none"
    # None migrates the legacy enum/caps; unified documents set explicit booleans.
    enforce_cycle_limit: bool | None = None
    enforce_dwell_constraints: bool | None = None
    electricity_weight: float = 1
    trip_weight: float = 100
    health_low_weight: float = 1
    health_high_weight: float = 1
    target_soc_low_weight: float | None = None
    target_soc_high_weight: float | None = None
    terminal_soc_weight: float = 0  # Retired compatibility field; never multiplies terminal value.
    enforce_terminal_soc: bool = False  # Enables daily-tariff terminal ENERGY value, not an SOC floor.
    enforce_discharge_budget: bool = False
    discharge_budget_soc: float = 1
    mip_relative_gap: float = 0.05
    persistence_weight: float = 0
    dwell_on_minutes: int = 30
    dwell_off_minutes: int = 30
    max_charge_starts_24h: int | None = None
    max_discharge_starts_24h: int | None = None
    projection: str = "nearest"
    projection_tracking_weight: float = 1
    projection_switch_weight: float = 0
    projection_threshold: float = 0.5
    source: str = "nhts"
    source_file: str | None = None
    source_vehicle: str | None = None
    seed: int = 0
    history_days: float = 0
    forecast_margin_hours: float = 12

    def __post_init__(self):
        for new, old in (("target_soc_min", "health_min_soc"), ("target_soc_max", "health_max_soc"),
                         ("target_soc_low_weight", "health_low_weight"), ("target_soc_high_weight", "health_high_weight")):
            if getattr(self, new) is None:
                object.__setattr__(self, new, getattr(self, old))
        for new, old in (("health_soc_min", "target_soc_min"), ("health_soc_max", "target_soc_max"),
                         ("comfort_soc_min", "target_soc_min")):
            if getattr(self, new) is None:
                object.__setattr__(self, new, getattr(self, old))
        if self.health_soc_min is None or self.health_soc_max is None or self.comfort_soc_min is None:
            raise ValueError("EV health_soc_min, health_soc_max and comfort_soc_min must be explicit")
        if self.health_soc_weight is None:
            if self.target_soc_low_weight != self.target_soc_high_weight:
                raise ValueError("Replace unequal legacy SOC weights with one explicit health_soc_weight")
            object.__setattr__(self, "health_soc_weight", self.target_soc_low_weight)
        if self.enforce_cycle_limit is None:
            object.__setattr__(self, "enforce_cycle_limit", self.max_charge_starts_24h is not None or self.max_discharge_starts_24h is not None)
        if self.enforce_dwell_constraints is None:
            object.__setattr__(self, "enforce_dwell_constraints", self.cycling == "dwell")
        for name in ("enforce_cycle_limit", "enforce_dwell_constraints", "enforce_terminal_soc", "enforce_discharge_budget", "enable_degradation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"EV {name} must be a boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("EV seed must be a nonnegative integer")
        if not isinstance(self.solver_options, dict):
            raise ValueError("EV solver_options must be a dictionary")
        for name in ("consumption_kwh_per_km", "horizon_hours", "control_minutes",
                     "projection_tracking_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"EV {name} must be finite and positive")
        for name in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"EV {name} must be in (0, 1]")
        if not 0 <= self.health_soc_min <= self.health_soc_max <= 1 or not 0 <= self.comfort_soc_min <= 1:
            raise ValueError("EV health SOC bounds require 0 <= min <= max <= 1; comfort SOC must be in [0,1]")
        if not 0 < self.end_of_life_soh < 1 or not 0 < self.reference_cycle_depth <= 1:
            raise ValueError("EV EOL SOH must be in (0,1), reference depth in (0,1]")
        if not 0 < self.calibration_end_of_life_soh < 1:
            raise ValueError("EV calibration_end_of_life_soh must be in (0,1)")
        if not 0 < self.initial_soh <= 1:
            raise ValueError("EV initial_soh must be in (0,1]")
        if not self.reference_cycle_depth/2 <= self.reference_mean_soc <= 1-self.reference_cycle_depth/2:
            raise ValueError("Reference cycle must fit within physical SOC [0,1]")
        if not math.isfinite(self.rainflow_exponent) or self.rainflow_exponent < 1:
            raise ValueError("Convex rainflow requires rainflow_exponent >= 1")
        # Sufficient interval-stress monotonicity/submodularity condition on
        # depths in [0,1]: mixed endpoint derivative <= 0. Avoid a declining
        # prefix-damage estimate (unphysical SOH recovery) as cycles develop.
        if self.soc_stress_coefficient > 2*math.sqrt(self.rainflow_exponent*(self.rainflow_exponent-1)):
            raise ValueError("Dynamic rainflow requires soc_stress_coefficient <= 2*sqrt(beta*(beta-1))")
        if not math.isfinite(self.reference_cycle_count) or self.reference_cycle_count <= 0:
            raise ValueError("reference_cycle_count must be finite and positive")
        if not 0 <= self.battery_residual_value_per_kwh <= self.battery_replacement_cost_per_kwh:
            raise ValueError("Battery residual value must be between zero and replacement cost")
        if type(self.rainflow_max_iterations) is not int or self.rainflow_max_iterations <= 0:
            raise ValueError("rainflow_max_iterations must be a positive integer")
        if not math.isfinite(self.mip_relative_gap) or not 0 <= self.mip_relative_gap <= 1:
            raise ValueError("EV mip_relative_gap must be in [0,1]")
        for name in ("max_discharge_kw", "electricity_weight", "trip_weight",
                     "health_low_weight", "health_high_weight", "persistence_weight",
                     "projection_switch_weight", "history_days", "forecast_margin_hours",
                     "dwell_on_minutes", "dwell_off_minutes", "target_soc_low_weight",
                     "target_soc_high_weight", "terminal_soc_weight", "discharge_budget_soc",
                     "health_soc_weight", "health_weight", "comfort_weight", "battery_replacement_cost_per_kwh",
                     "battery_residual_value_per_kwh", "soc_stress_coefficient",
                     "rainflow_relative_tolerance", "rainflow_absolute_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"EV {name} must be finite and nonnegative")
        choices = {"controller": {"baseline", "mpc"}, "formulation": {"convex", "milp"},
                   "solver": {"SCIPY", "CLARABEL", "GUROBI"},
                   "cycling": {"none", "persistence", "dwell"},
                   "projection": {"nearest", "threshold"}, "source": {"nhts", "audi"},
                   "degradation_method": {"none", "rainflow", "rainflow_multicut", "linear_throughput"}}
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError(f"EV {name} must be one of {sorted(values)}")
        if self.formulation == "milp" and self.solver == "CLARABEL":
            raise ValueError("CLARABEL cannot solve MILP; select SCIPY or GUROBI")
        if self.v2h and self.max_discharge_kw <= 0:
            raise ValueError("V2H requires positive max_discharge_kw")
        if not isinstance(self.v2h, bool):
            raise ValueError("EV v2h must be a boolean")
        if not 0 <= self.projection_threshold <= 1:
            raise ValueError("EV projection_threshold must be in [0, 1]")
        for name in ("max_charge_starts_24h", "max_discharge_starts_24h"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"EV {name} must be a nonnegative integer or null")
        if self.horizon_hours * 60 < self.control_minutes:
            raise ValueError("EV horizon must cover at least one control interval")


def validate_home_evs(evs):
    """Reject multiple V2H vehicles, including vehicles configured as baseline."""
    if sum(bool(ev.hems and ev.hems.v2h) for ev in evs) > 1:
        raise ValueError("Only one EV per home may be configured with V2H capability")
    for ev in evs:
        if ev.hems and (not all(math.isfinite(v) for v in
                        (ev.initial_soc_fraction, ev.capacity_kwh, ev.charger_power_kw)) or
                        not 0 <= ev.initial_soc_fraction <= 1 or
                        ev.capacity_kwh <= 0 or ev.charger_power_kw <= 0):
            raise ValueError(f"Invalid battery/charger/initial SOC for EV {ev.vehicle_id}")

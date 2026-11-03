"""Power-loss NMC EV plant used by the opt-in OCHRE HEMS vehicle.

Positive AC power charges the battery; negative power exports from it.  The
tabulated Kokam profile is a literature archetype, not a fit to the selected EV.
The equations and profile match the reviewed comparison implementation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from bisect import bisect_right


PROFILE_SHA256 = "5a8e4aefa2a175d6e10176fa106a2d1c2b2f0c0a102d8f1b4e00c3cb46568c15"
DEFAULT_PROFILE = Path(__file__).parent / "reference" / "kokam75_25C_profile_v1.csv"
Z_LOW = 0.1
Z_HIGH = 0.9
SERIES_CELLS = 108
CELL_AH = 75.0
CELL_V_MIN = 3.0
CELL_V_MAX = 4.2


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class NMCProfile:
    """Exact piecewise-linear cell profile and its integrated OCV lookup."""

    z: tuple[float, ...]
    ocv_v: tuple[float, ...]
    r0_ohm: tuple[float, ...]
    r1_ohm: tuple[float, ...]
    c1_f: tuple[float, ...]
    ocv_integral_v: tuple[float, ...]
    sha256: str

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_PROFILE) -> NMCProfile:
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != PROFILE_SHA256:
            raise ValueError(f"Unexpected NMC profile SHA256: {digest}")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        expected = {"z", "ocv_cell_v", "r0_cell_ohm", "r1_cell_ohm", "c1_cell_f", "tau_s"}
        if len(rows) != 801 or not rows or set(rows[0]) != expected:
            raise ValueError("NMC profile must have the expected columns and 801 rows")
        cols = [[float(r[key]) for r in rows] for key in (
            "z", "ocv_cell_v", "r0_cell_ohm", "r1_cell_ohm", "c1_cell_f", "tau_s"
        )]
        z, u, r0, r1, c1, tau = cols
        if any(not all(math.isfinite(v) for v in col) for col in cols):
            raise ValueError("Nonfinite NMC profile value")
        if abs(z[0] - Z_LOW) > 1e-12 or abs(z[-1] - Z_HIGH) > 1e-12:
            raise ValueError("Unexpected NMC charge-SOC domain")
        if any(abs(value - (Z_LOW + i * 0.001)) > 1e-12 for i, value in enumerate(z)):
            raise ValueError("Unexpected NMC profile grid")
        if any(u[i + 1] <= u[i] for i in range(800)):
            raise ValueError("NMC OCV must rise on the selected domain")
        if any(min(u[i], r0[i], r1[i], c1[i]) <= 0 for i in range(801)):
            raise ValueError("NMC profile parameters must be positive")
        if any(abs(tau[i] - r1[i] * c1[i]) > 1e-8 for i in range(801)):
            raise ValueError("NMC tau audit column is inconsistent")
        integral = [0.0]
        for i in range(800):
            integral.append(integral[-1] + 0.5 * (u[i] + u[i + 1]) * (z[i + 1] - z[i]))
        if abs(integral[-1] - 2.973605298005099) > 1e-10:
            raise ValueError("Unexpected NMC OCV lookup integral")
        return cls(*(tuple(col) for col in (z, u, r0, r1, c1, integral)), digest)

    @property
    def full_integral_v(self) -> float:
        return self.ocv_integral_v[-1]

    def _segment(self, z: float, *, bounded: bool = False) -> tuple[int, float]:
        z = _finite(z, "charge SOC")
        if bounded:
            z = min(Z_HIGH, max(Z_LOW, z))
        elif z < Z_LOW - 1e-13 or z > Z_HIGH + 1e-13:
            raise ValueError("Charge SOC outside NMC profile domain")
        z = min(Z_HIGH, max(Z_LOW, z))
        i = min(799, max(0, int((z - Z_LOW) / 0.001)))
        # Floating-point representation can put a grid knot in the prior bin.
        if z < self.z[i]:
            i -= 1
        elif z > self.z[i + 1] and i < 799:
            i += 1
        return i, z - self.z[i]

    def values(self, z: float, *, bounded: bool = False) -> tuple[float, float, float, float]:
        i, dz = self._segment(z, bounded=bounded)
        w = dz / (self.z[i + 1] - self.z[i])
        return tuple(col[i] + w * (col[i + 1] - col[i]) for col in (
            self.ocv_v, self.r0_ohm, self.r1_ohm, self.c1_f
        ))

    def integral_to(self, z: float, *, bounded: bool = False) -> float:
        i, dz = self._segment(z, bounded=bounded)
        slope = (self.ocv_v[i + 1] - self.ocv_v[i]) / (self.z[i + 1] - self.z[i])
        return self.ocv_integral_v[i] + self.ocv_v[i] * dz + 0.5 * slope * dz * dz

    def z_from_integral(self, integral_v: float, *, bounded: bool = False) -> float:
        integral_v = _finite(integral_v, "OCV integral")
        if bounded:
            integral_v = min(self.full_integral_v, max(0.0, integral_v))
        elif integral_v < -1e-12 or integral_v > self.full_integral_v + 1e-12:
            raise ValueError("Stored energy outside NMC usable window")
        integral_v = min(self.full_integral_v, max(0.0, integral_v))
        i = min(799, max(0, bisect_right(self.ocv_integral_v, integral_v) - 1))
        delta = integral_v - self.ocv_integral_v[i]
        slope = (self.ocv_v[i + 1] - self.ocv_v[i]) / (self.z[i + 1] - self.z[i])
        # Rationalized quadratic root avoids cancellation for tiny delta.
        dz = 2 * delta / (self.ocv_v[i] + math.sqrt(self.ocv_v[i] ** 2 + 2 * slope * delta))
        return min(Z_HIGH, max(Z_LOW, self.z[i] + dz))


@dataclass(frozen=True)
class NMCPack:
    profile: NMCProfile
    usable_kwh: float
    series_cells: int = SERIES_CELLS

    def __post_init__(self) -> None:
        if _finite(self.usable_kwh, "usable_kwh") <= 0:
            raise ValueError("usable_kwh must be positive")

    @property
    def charge_capacity_ah(self) -> float:
        return 1000 * self.usable_kwh / (self.series_cells * self.profile.full_integral_v)

    @property
    def parallel_effective(self) -> float:
        return self.charge_capacity_ah / CELL_AH

    @property
    def nominal_voltage_v(self) -> float:
        return self.series_cells * self.profile.full_integral_v / (Z_HIGH - Z_LOW)

    @property
    def charge_current_limit_a(self) -> float:
        return 0.5 * self.charge_capacity_ah

    @property
    def discharge_current_limit_a(self) -> float:
        return self.charge_capacity_ah

    @property
    def min_voltage_v(self) -> float:
        return CELL_V_MIN * self.series_cells

    @property
    def max_voltage_v(self) -> float:
        return CELL_V_MAX * self.series_cells

    def values(self, z: float, *, bounded: bool = False) -> tuple[float, float, float, float]:
        u, r0, r1, c1 = self.profile.values(z, bounded=bounded)
        return (
            self.series_cells * u,
            self.series_cells / self.parallel_effective * r0,
            self.series_cells / self.parallel_effective * r1,
            self.parallel_effective / self.series_cells * c1,
        )

    def energy_kwh(self, z: float, *, bounded: bool = False) -> float:
        return self.charge_capacity_ah * self.series_cells / 1000 * self.profile.integral_to(z, bounded=bounded)

    def charge_soc(self, energy_kwh: float, *, bounded: bool = False) -> float:
        energy_kwh = _finite(energy_kwh, "stored energy")
        integral = 1000 * energy_kwh / (self.charge_capacity_ah * self.series_cells)
        return self.profile.z_from_integral(integral, bounded=bounded)

    def static_terminal_limits_kw(self, z: float) -> tuple[float, float]:
        """Return (maximum charge kW, positive discharge magnitude kW)."""
        u, r0, r1, _ = self.values(z, bounded=True)
        req = r0 + r1
        ic = min(self.charge_current_limit_a, max(0.0, (self.max_voltage_v - u) / req))
        ide = min(self.discharge_current_limit_a, max(0.0, (u - self.min_voltage_v) / req), u / (2 * req))
        return ic * (u + req * ic) / 1000, ide * (u - req * ide) / 1000


@dataclass(frozen=True)
class TripEvent:
    requested_depletion_kwh: float
    applied_depletion_kwh: float
    unapplied_depletion_kwh: float
    stored_energy_kwh: float
    energy_soc: float


@dataclass(frozen=True)
class BatteryStep:
    model_type: str
    duration_s: float
    requested_ac_kw: float
    actual_ac_average_kw: float
    ac_import_kwh: float
    ac_export_kwh: float
    terminal_energy_kwh: float
    stored_energy_kwh: float
    energy_soc: float
    charge_soc: float
    loss_energy_kwh: float
    ohmic_energy_kwh: float
    polarization_transfer_kwh: float
    curtailed_ac_kwh: float
    limiting_reason: str
    current_a: float | None
    terminal_voltage_v: float | None
    polarization_v: float | None
    energy_residual_kwh: float

    @property
    def loss_kwh(self) -> float:
        return self.loss_energy_kwh


def _validate_eta(eta: float, name: str) -> float:
    eta = _finite(eta, name)
    if not 0 < eta <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return eta


def _validate_rating(power: float, name: str) -> float:
    power = _finite(power, name)
    if power < 0:
        raise ValueError(f"{name} must be nonnegative")
    return power


class _BatteryBase:
    model_type = "base"

    def __init__(
        self,
        profile: NMCProfile,
        usable_kwh: float,
        initial_energy_soc: float,
        *,
        eta_charge: float = 0.93,
        eta_discharge: float = 0.93,
        charger_charge_kw: float = 7.2,
        charger_discharge_kw: float = 7.2,
    ) -> None:
        self.pack = NMCPack(profile, usable_kwh)
        initial_energy_soc = _finite(initial_energy_soc, "initial_energy_soc")
        if not 0 <= initial_energy_soc <= 1:
            raise ValueError("initial_energy_soc must be in [0, 1]")
        self.eta_charge = _validate_eta(eta_charge, "eta_charge")
        self.eta_discharge = _validate_eta(eta_discharge, "eta_discharge")
        self.charger_charge_kw = _validate_rating(charger_charge_kw, "charger_charge_kw")
        self.charger_discharge_kw = _validate_rating(charger_discharge_kw, "charger_discharge_kw")
        self._init_state(initial_energy_soc)

    @property
    def stored_energy_kwh(self) -> float:
        raise NotImplementedError

    @property
    def energy_soc(self) -> float:
        return self.stored_energy_kwh / self.pack.usable_kwh

    @property
    def E_kwh(self) -> float:
        return self.stored_energy_kwh

    @property
    def charge_soc(self) -> float:
        return self.pack.charge_soc(self.stored_energy_kwh)

    def _init_state(self, initial_energy_soc: float) -> None:
        raise NotImplementedError

    def _set_stored_energy(self, energy_kwh: float) -> None:
        raise NotImplementedError

    def apply_trip_energy_kwh(self, requested_depletion_kwh: float) -> TripEvent:
        """Apply supplied away depletion once; negative values represent away energy added."""
        demand = _finite(requested_depletion_kwh, "requested_depletion_kwh")
        before = self.stored_energy_kwh
        after = min(self.pack.usable_kwh, max(0.0, before - demand))
        self._set_stored_energy(after)
        actual = before - after
        return TripEvent(demand, actual, demand - actual, after, self.energy_soc)

    def rest(self, dt_s: float) -> None:
        """Advance an open-circuit interval; useful while away between trip events."""
        dt_s = _finite(dt_s, "dt_s")
        if dt_s < 0:
            raise ValueError("dt_s must be nonnegative")

    def _terminal_to_ac(self, terminal_kw: float) -> float:
        return terminal_kw / self.eta_charge if terminal_kw >= 0 else terminal_kw * self.eta_discharge

    def _command_to_terminal(self, ac_kw: float) -> float:
        return ac_kw * self.eta_charge if ac_kw >= 0 else ac_kw / self.eta_discharge

    def _clip_terminal(self, command_ac_kw: float, z: float, dynamic: tuple[float, float] | None = None) -> tuple[float, float, int]:
        """Clip AC hardware, static battery, then optional dynamic power limits.

        Bits in the returned mask: 1 charger, 2 shared static envelope,
        4 one-RC instantaneous current/voltage envelope.
        """
        mask = 0
        ac = max(-self.charger_discharge_kw, min(self.charger_charge_kw, command_ac_kw))
        if abs(ac - command_ac_kw) > 1e-12:
            mask |= 1
        dc = self._command_to_terminal(ac)
        charge_max, discharge_max = getattr(self, "parameter_pack", self.pack).static_terminal_limits_kw(z)
        static_dc = max(-discharge_max, min(charge_max, dc))
        if abs(static_dc - dc) > 1e-12:
            mask |= 2
        dc = static_dc
        if dynamic is not None:
            dyn_charge, dyn_discharge = dynamic
            dynamic_dc = max(-dyn_discharge, min(dyn_charge, dc))
            if abs(dynamic_dc - dc) > 1e-12:
                mask |= 4
            dc = dynamic_dc
        return self._terminal_to_ac(dc), dc, mask

    @staticmethod
    def _reason(mask: int) -> str:
        reasons = []
        if mask & 1:
            reasons.append("charger")
        if mask & 2:
            reasons.append("static_power")
        if mask & 4:
            reasons.append("current_or_voltage")
        if mask & 8:
            reasons.append("energy_boundary")
        if mask & 16:
            reasons.append("unplugged")
        return ";".join(reasons)

    def _zero_result(self, requested_ac_kw: float, dt_s: float, reason_mask: int = 0) -> BatteryStep:
        self.rest(dt_s)
        return BatteryStep(
            self.model_type, dt_s, requested_ac_kw, 0.0, 0.0, 0.0, 0.0,
            self.stored_energy_kwh, self.energy_soc, self.charge_soc,
            0.0, 0.0, 0.0, abs(requested_ac_kw) * dt_s / 3600,
            self._reason(reason_mask), 0.0 if self.model_type == "one_rc" else None,
            self._rest_voltage(), self._polarization(), 0.0,
        )

    def _rest_voltage(self) -> float | None:
        return None

    def _polarization(self) -> float | None:
        return None


class PowerLossPlant(_BatteryBase):
    """One stored-energy state with source-derived quadratic battery loss.

    Usable capacity may follow realized SOH while electrical parameters retain
    their initial effective-pack scaling.  The default is the exact comparison
    model, with electrical and usable capacities equal.
    """

    model_type = "power_loss"

    def __init__(self, *args, max_internal_step_s: float = 60.0,
                 parameter_usable_kwh: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parameter_pack = (self.pack if parameter_usable_kwh is None else
                               NMCPack(self.pack.profile, parameter_usable_kwh))
        self.max_internal_step_s = _finite(max_internal_step_s, "max_internal_step_s")
        if self.max_internal_step_s <= 0:
            raise ValueError("max_internal_step_s must be positive")

    def _init_state(self, initial_energy_soc: float) -> None:
        self._energy_kwh = self.pack.usable_kwh * initial_energy_soc

    @property
    def stored_energy_kwh(self) -> float:
        return self._energy_kwh

    def _set_stored_energy(self, energy_kwh: float) -> None:
        self._energy_kwh = energy_kwh

    def _rate(self, energy_kwh: float, command_ac_kw: float) -> tuple[tuple[float, ...], int]:
        z = self.pack.charge_soc(energy_kwh, bounded=True)
        ac, terminal, mask = self._clip_terminal(command_ac_kw, z)
        _, r0, r1, _ = self.parameter_pack.values(z, bounded=True)
        kappa = 1000 * (r0 + r1) / self.parameter_pack.nominal_voltage_v**2
        loss = kappa * terminal**2
        # E, positive AC import, positive AC export, terminal DC, battery loss.
        return ((terminal - loss) / 3600, max(ac, 0) / 3600,
                max(-ac, 0) / 3600, terminal / 3600, loss / 3600), mask

    def _rk4(self, energy_kwh: float, command_ac_kw: float, h_s: float) -> tuple[float, tuple[float, ...], int]:
        k1, m1 = self._rate(energy_kwh, command_ac_kw)
        k2, m2 = self._rate(energy_kwh + 0.5 * h_s * k1[0], command_ac_kw)
        k3, m3 = self._rate(energy_kwh + 0.5 * h_s * k2[0], command_ac_kw)
        k4, m4 = self._rate(energy_kwh + h_s * k3[0], command_ac_kw)
        delta = tuple(h_s * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(5))
        return energy_kwh + delta[0], delta[1:], m1 | m2 | m3 | m4

    def step(self, ac_command_kw: float, dt_s: float, *, plugged_in: bool = True) -> BatteryStep:
        command = _finite(ac_command_kw, "ac_command_kw")
        dt_s = _finite(dt_s, "dt_s")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if not plugged_in:
            return self._zero_result(command, dt_s, 16 if command else 0)
        if command == 0:
            return self._zero_result(command, dt_s)
        initial = self.stored_energy_kwh
        totals = [0.0, 0.0, 0.0, 0.0]
        mask = 0
        remaining = dt_s
        while remaining > 1e-12:
            h = min(self.max_internal_step_s, remaining)
            outward = (command > 0 and self._energy_kwh >= self.pack.usable_kwh - 1e-11) or (
                command < 0 and self._energy_kwh <= 1e-11)
            if outward:
                mask |= 8
                break
            new_energy, increments, clip_mask = self._rk4(self._energy_kwh, command, h)
            mask |= clip_mask
            boundary = self.pack.usable_kwh if new_energy > self.pack.usable_kwh else (0.0 if new_energy < 0 else None)
            if boundary is not None:
                lo, hi = 0.0, h
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    trial, _, _ = self._rk4(self._energy_kwh, command, mid)
                    if (trial - boundary) * (new_energy - boundary) >= 0:
                        hi = mid
                    else:
                        lo = mid
                _, increments, event_mask = self._rk4(self._energy_kwh, command, 0.5 * (lo + hi))
                mask |= event_mask | 8
                self._energy_kwh = boundary
                for i in range(4):
                    totals[i] += increments[i]
                break
            self._energy_kwh = new_energy
            for i in range(4):
                totals[i] += increments[i]
            remaining -= h
        import_kwh, export_kwh, terminal_kwh, loss_kwh = totals
        actual_net = import_kwh - export_kwh
        residual = self._energy_kwh - initial - (terminal_kwh - loss_kwh)
        return BatteryStep(
            self.model_type, dt_s, command, actual_net * 3600 / dt_s,
            import_kwh, export_kwh, terminal_kwh, self.stored_energy_kwh,
            self.energy_soc, self.charge_soc, loss_kwh, 0.0, 0.0,
            max(0.0, abs(command) * dt_s / 3600 - import_kwh - export_kwh),
            self._reason(mask), None, None, None, residual,
        )

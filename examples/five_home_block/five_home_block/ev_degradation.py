"""Shared EV battery calibration and numerical rainflow evaluation.

OCP model: Shi et al., A Convex Cycle-based Degradation Model (2017),
depth-only power stress. Reference: Mathews et al. (2020), NMC power-law
depth stress and signed mean-SOC stress, with additive fade retained by request.
This is NOT the paper's nonlinear early-life SEI accumulation model.
No pay-for-performance control model is used. Full cycles count 1; residue
half cycles count 1/2. Costs are dollars, independent of optimizer weights.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np


def turning_points(values):
    """Original indices of endpoints/extrema; retain the last point on a plateau."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Rainflow SOC must be finite")
    points = []
    for i in range(len(x)):
        if points and x[i] == x[points[-1]]:
            points[-1] = i
        else:
            points.append(i)
        while len(points) >= 3:
            a, b, c = points[-3:]
            if (x[b]-x[a])*(x[c]-x[b]) < 0:
                break
            points.pop(-2)
    return points


def rainflow_cycles(values):
    """Return (endpoint index, endpoint index, count), including open residues.

Four-point extraction preserves closed cycles and all boundary half cycles.
Plateaus are reduced before counting; no artificial periodic closure is imposed.
"""
    x = np.asarray(values, dtype=float).reshape(-1)
    stack, cycles = [], []
    for i in turning_points(x):
        stack.append(i)
        while len(stack) >= 4:
            a, b, c, d = stack[-4:]
            middle = abs(x[b]-x[c])
            if middle > abs(x[a]-x[b]) or middle > abs(x[c]-x[d]):
                break
            cycles.append((b, c, 1.0))
            del stack[-3:-1]
    cycles.extend((a, b, .5) for a, b in zip(stack, stack[1:]))
    return cycles


@dataclass(frozen=True)
class BatteryDegradation:
    """Shared NMC calibration and net depreciation for all three cost models."""
    capacity_kwh: float
    options: object
    current_capacity_kwh: float | None = None

    @property
    def calibration_fade(self):
        return 1-self.options.calibration_end_of_life_soh

    @property
    def allowed_damage(self):
        """Calibration-normalized lifetime budget (default 1.5, not 0.3)."""
        return (1-self.options.end_of_life_soh)/self.calibration_fade

    @property
    def damage_price(self):
        """Dollars per calibration-normalized damage; fixed original pack value."""
        return self.depreciable_usd/self.allowed_damage

    def cycle_damage(self, depth, mean_soc, *, reference=True):
        o = self.options
        damage = depth**o.rainflow_exponent/(o.reference_cycle_count*o.reference_cycle_depth**o.rainflow_exponent)
        return damage*np.exp(o.soc_stress_coefficient*(mean_soc-o.reference_mean_soc)) if reference else damage

    @property
    def replacement_usd(self):
        return self.capacity_kwh*self.options.battery_replacement_cost_per_kwh

    @property
    def residual_usd(self):
        return self.capacity_kwh*self.options.battery_residual_value_per_kwh

    @property
    def depreciable_usd(self):
        return self.replacement_usd-self.residual_usd

    def reference_loss(self, depth, mean_soc):
        """Additive full-cycle fade: S_depth(d) * exp(k_soc*(mean-reference)).

Mathews et al., doi:10.1016/j.apenergy.2020.115127, degradation appendix.
Default S_depth = 0.2/(3000*0.8**2.03) * d**2.03. No absolute SOC deviation,
calendar/temperature aging, or early-life exponential accumulation is applied.
"""
        if depth <= 0:
            return 0.0
        o = self.options
        calibration_scale = (1-o.calibration_end_of_life_soh) / (
            o.reference_cycle_count * o.reference_cycle_depth**o.rainflow_exponent)
        return (calibration_scale*depth**o.rainflow_exponent
                * np.exp(o.soc_stress_coefficient*(mean_soc-o.reference_mean_soc)))


    @property
    def reference_cycle_life(self):
        """Effective cycles to economic EOL, extrapolating the calibrated fade slope.

        Default: 3000 cycles to 80% SOH -> 4500 cycles to 70% SOH.
        This assumes unchanged additive cycle fade, not a validated aging knee.
        """
        o = self.options
        return o.reference_cycle_count*(1-o.end_of_life_soh)/(1-o.calibration_end_of_life_soh)

    @property
    def power_stress_coefficient(self):
        """Life fraction per full cycle = coefficient * depth**exponent."""
        o = self.options
        return 1/(self.reference_cycle_life*o.reference_cycle_depth**o.rainflow_exponent)

    @property
    def throughput_usd_per_kwh(self):
        capacity = self.capacity_kwh if self.current_capacity_kwh is None else self.current_capacity_kwh
        return self.depreciable_usd/(2*capacity*self.options.reference_cycle_depth
                                     * self.reference_cycle_life)

    def rainflow_cost_gradient(self, soc):
        """Exact depth-only dollars and a supporting SOC subgradient.

Cycle endpoint derivatives are accumulated, including 1/2 residue weights.
No finite differences or fixed cycle pairings are used between OCP iterations.
"""
        x = np.asarray(soc, dtype=float).reshape(-1)
        gradient = np.zeros_like(x)
        cost = 0.0
        scale = self.depreciable_usd*self.power_stress_coefficient
        exponent = self.options.rainflow_exponent
        for a, b, count in rainflow_cycles(x):
            depth = abs(x[a]-x[b])
            cost += count*scale*depth**exponent
            slope = count*scale*exponent*depth**(exponent-1)*np.sign(x[a]-x[b])
            gradient[a] += slope
            gradient[b] -= slope
        return float(cost), gradient

    def costs(self, soc):
        """All unweighted models on exactly the same realized SOC path."""
        x = np.asarray(soc, dtype=float).reshape(-1)
        rainflow = self.rainflow_cost_gradient(x)[0]
        capacity = self.capacity_kwh if self.current_capacity_kwh is None else self.current_capacity_kwh
        throughput = capacity*np.abs(np.diff(x)).sum()
        linear = self.throughput_usd_per_kwh*throughput
        reference = self.depreciable_usd/(1-self.options.end_of_life_soh)*sum(
            count*self.reference_loss(abs(x[a]-x[b]), (x[a]+x[b])/2)
            for a, b, count in rainflow_cycles(x))
        return dict(rainflow_degradation_cost_usd=rainflow,
                    linear_degradation_cost_usd=float(linear),
                    reference_degradation_cost_usd=float(reference),
                    battery_throughput_kwh=float(throughput))


class RainflowLedger:
    """Streaming four-point damage, including open residue half cycles.

    Only the residue and closed-cycle totals are retained. Damage is normalized
    to the calibration endpoint. Initial pre-existing damage is NOT priced here.
    preview() is transactional: repeated OCHRE model evaluations never age twice.
    """
    def __init__(self, battery, initial_soc):
        self.battery = battery
        self.residue = [float(initial_soc)]
        self.closed = np.zeros(2)  # reference, depth-only
        self.totals = np.zeros(3)  # reference, depth-only, throughput
        self.throughput_kwh = 0.0

    def _cycle(self, a, b):
        d, m = abs(a-b), (a+b)/2
        return np.array([self.battery.cycle_damage(d, m),
                         self.battery.cycle_damage(d, m, reference=False)])

    def preview(self, soc, capacity_kwh):
        if not np.isfinite(soc) or not 0 <= soc <= 1 or not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
            raise ValueError("Rainflow ledger needs bounded SOC and positive current capacity")
        new = deepcopy(self)
        excursion = abs(float(soc)-new.residue[-1])
        new.throughput_kwh += capacity_kwh*excursion
        if excursion:
            new.residue.append(float(soc))
            if len(new.residue) >= 3:
                a, b, c = new.residue[-3:]
                if (b-a)*(c-b) >= 0:
                    new.residue.pop(-2)
            while len(new.residue) >= 4:
                a, b, c, d = new.residue[-4:]
                if abs(b-c) > abs(a-b) or abs(b-c) > abs(c-d):
                    break
                new.closed += new._cycle(b, c)
                del new.residue[-3:-1]
        new.totals[:2] = new.closed + sum((.5*new._cycle(a, b)
            for a, b in zip(new.residue, new.residue[1:])), np.zeros(2))
        o = self.battery.options
        new.totals[2] += excursion/(2*o.reference_cycle_depth*o.reference_cycle_count)
        delta = new.totals-self.totals
        if np.any(delta < -1e-12):
            raise RuntimeError("Reference rainflow damage decreased; unsupported calibration or ledger inconsistency")
        # Roundoff only: do not silently discard a material negative increment.
        new.totals = np.maximum(new.totals, self.totals)
        return new

    def snapshot(self):
        return dict(residue=list(self.residue), closed=self.closed.tolist(),
                    totals=self.totals.tolist(), throughput_kwh=self.throughput_kwh)

    @classmethod
    def restore(cls, battery, state):
        if not state.get("residue"):
            raise ValueError("Invalid EV damage ledger checkpoint: empty residue")
        new = cls(battery, state["residue"][0])
        new.residue = list(state["residue"])
        new.closed = np.asarray(state["closed"], dtype=float)
        new.totals = np.asarray(state["totals"], dtype=float)
        new.throughput_kwh = float(state["throughput_kwh"])
        if (not new.residue or not np.isfinite(new.residue).all() or
                np.any(np.asarray(new.residue) < 0) or np.any(np.asarray(new.residue) > 1) or
                new.closed.shape != (2,) or new.totals.shape != (3,) or
                not np.isfinite(new.closed).all() or not np.isfinite(new.totals).all() or
                np.any(new.closed < 0) or np.any(new.totals < 0) or
                not np.isfinite(new.throughput_kwh) or new.throughput_kwh < 0):
            raise ValueError("Invalid EV damage ledger checkpoint")
        expected = new.closed + sum((.5*new._cycle(a, b)
            for a, b in zip(new.residue, new.residue[1:])), np.zeros(2))
        if not np.allclose(expected, new.totals[:2], rtol=1e-10, atol=1e-12):
            raise ValueError("EV checkpoint totals disagree with rainflow residue")
        return new

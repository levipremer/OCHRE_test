# EV health, comfort and degradation

Current implementation. Use the unified runner and the two documented JSON templates.
Single-path OCP equations and EV controllers are in `single_home_hems/controllers/ev.py`;
`controllers/ev_scenario.py` assembles scenario problems and their joint cut loop.
`five_home_block/ev_degradation.py` is the shared numerical cycle counter/calibration,
not a second controller formulation. Import EV controllers directly from `controllers.ev`.

## Configuration and objective

Per-EV control preferences live in the **control** document:

- `health_soc_min`, `health_soc_max`: soft health limits, 0 <= min <= max <= 1.
- `health_soc_weight`: the SAME nonnegative weight for lower and upper violations.
- `health_weight`: nonnegative multiplier for the entire health/degradation term.
- `comfort_soc_min`, `comfort_weight`: independent soft comfort floor and weight.
  Baseline/fallback charges immediately to this floor, then stops; never discharges.
- `enable_degradation`: enable the optional OCP monetary aging term; default false.
- `degradation_method`: `rainflow` or `linear_throughput`.
- `rainflow_relative_tolerance`, `rainflow_absolute_tolerance`,
  `rainflow_max_iterations`: convex gap tolerance, absolute objective tolerance,
  and maximum master solves. MILP uses `mip_relative_gap` for whole-objective
  optimality; BOTH formulations also use the rainflow tolerances for independent
  aging-envelope dollar accuracy. Defaults: 0.001, 1e-6, 100, and 0.05 respectively.

Let N count control intervals (not mobility-split integration blocks), x_j be SOC,
and dt_j be hours. The new contribution to the unchanged electricity/persistence/
terminal objective is:

```
H_low     = sum_j dt_j * max(health_soc_min - x_j, 0) / N
H_high    = sum_j dt_j * max(x_j - health_soc_max, 0) / N
H_comfort = sum_j dt_j * max(comfort_soc_min - x_j, 0) / N
J_added   = health_weight * [health_soc_weight*(H_low + H_high) + C_age/N]
            + comfort_weight*H_comfort
```

Comfort and health apply throughout the horizon, including while away. With
departure energy impulses, this soft reserve can encourage charging above the
comfort floor before a trip to retain that floor afterward. Departure-comfort
metrics remain separate, based on pre-debit SOC. Hard [0,1] OCP
bounds and full distance-derived driving demand remain. Bounded unmet-energy
slack permits unreachable forecast trips; its capacity-normalized cost is averaged
over new trips and maximized across scenarios. Only the plant saturates SOC.
The plant and OCP debit driving energy at departure. Departure-comfort metrics use
the state before that impulse; rainflow includes both sides to retain charging peaks.

Persistence retains dt_j/N scaling. Convex uses absolute changes in chemical
power; MILP uses the L1 change of the one-hot (charge, discharge, idle) vector,
including the observed mode before the horizon. Each mode switch contributes
two indicator changes. Backed-load changes within discharge mode cost nothing.
The existing weight is not automatically retuned; no cap/dwell is added.

Optional EV terminal valuation now uses `J_terminal = v*(e0-eN)/N` instead of a
moving SOC-floor penalty. Both energies use measured faded capacity, frozen
within the horizon. `v` is the minimum known tariff over the solve calendar day, assigned
directly in USD per battery-side kWh, with no efficiency correction or extra
weight. `enforce_terminal_soc` remains the on/off switch; `terminal_soc_weight`
is retired. See [pipeline guide](pipeline.md) for tariff sourcing
and saved plan diagnostics. The current default tariff gives `v=0.15`. Earlier mean-price tests remain
explicitly documented in the [validation index](validation/README.md).

For an aging experiment, set `enable_degradation=true`, select the method, and
choose independent comfort/health weights. The supplied files retain existing
tuning per document; no weights are automatically increased by this revision. These settings
do not claim to be optimal study weights. All inactive fields remain explicit.

## One battery calibration

The **home** document's `evs.ID.physical` stores all calibration fields. No legacy
config lookup and no independent cycle-life constant per method are used.
The NMC calibration uses Mathews et al. (2020), with the existing **additive**
fade accumulation retained by user direction. The nonlinear early-life SEI
transformation is NOT applied, either in the OCP or in reference metrics.

| Field | Study default | Meaning |
| --- | ---: | --- |
| initial_soh | 0.85 | Required initial usable-capacity fraction; initial SOC is relative to this faded capacity |
| battery_replacement_cost_per_kwh | 119 | Gross new battery dollars per usable kWh |
| battery_residual_value_per_kwh | 0 | Residual dollars per original usable kWh at economic EOL |
| calibration_end_of_life_soh | 0.8 | Calibration endpoint reached after reference_cycle_count cycles |
| end_of_life_soh | 0.7 | Economic endpoint; sets the monetary capacity-loss budget |
| reference_cycle_depth | 0.8 | Full reference cycle's SOC range |
| reference_mean_soc | 0.6 | Reference cycle's mean SOC |
| reference_cycle_count | 3000 | Full cycles to calibration endpoint at reference depth/mean |
| rainflow_exponent | 2.03 | Convex NMC depth-stress exponent, >= 1 |
| soc_stress_coefficient | 1.04 | Retained modeling assumption for signed mean-SOC stress |

The new/residual values are user-selected study assumptions. The requested IEA
attribution for $119/kWh still needs an exact source; do not cite it as verified.
Pack price need not include installed replacement labor or thermal hardware.
Residual must be between zero and new cost. Their difference, $119/kWh,
scales all aging costs. Reference depth/mean must fit within [0,1].

For cycle depth d and mean SOC m, the NMC reference is:

```
N_cal = reference_cycle_count
k_depth = (1-calibration_end_of_life_soh) / (N_cal * reference_cycle_depth**rainflow_exponent)
N_ref = N_cal * (1-end_of_life_soh)/(1-calibration_end_of_life_soh)
phi_NMC(d,m) = k_depth * d**rainflow_exponent
               * exp(soc_stress_coefficient*(m-reference_mean_soc))
R = capacity_kwh * (battery_replacement_cost_per_kwh-battery_residual_value_per_kwh)
```

Default k_depth is approximately 0.00010486633: unchanged 20% fade in 3000 cycles.
N_ref is the effective economic life: 4500 full cycles to 30% fade. This extrapolates
the same additive fade slope below 80% SOH; it does not model an aging knee or
claim experimentally validated life to 70% SOH.
All three costs equal R/N_ref for one reference cycle (SOC 0.2 -> 1.0 -> 0.2).
Below-reference mean SOC reduces reference stress; there is no symmetric minimum
at 0.5 or 0.6. The coefficient 1.04 is retained as an assumption, not asserted to
be a newly verified cell-specific NMC fit. The OCP remains average-SOC-unaware.

## OCP methods

At each scheduled solve, measured plant SOH sets `C_k = capacity_kwh * SOH_k`.
It is frozen only within that prediction horizon and refreshed next solve.
All power/trip-to-SOC conversions and physical actuator limits use current capacity.
Original nominal capacity still determines pack value R. Historical discharge
budgets retain the capacity at each past action, not today's capacity.

**Rainflow:** follows the second supplied paper's convex, average-SOC-unaware
cycle-depth model. Four-point counting extracts closed full cycles and retains
open residual half cycles. With counts n_i in {1, 1/2}:

```
a = 1 / (N_ref * reference_cycle_depth**rainflow_exponent)
F(x) = R * sum_i n_i * a * depth_i**rainflow_exponent
C_age = F(observed_history + predicted_future) - F(observed_history)
```

The observed prefix is fixed and retained across OCP calls, including its open
cycles; already-accounted degradation is subtracted. History is not reset every
horizon or 24 hours. The implementation keeps observed turning points, including
closed cycles; storage grows with turning-point count. No artificial periodic
closure is imposed. Future-only segmentation would incorrectly split spanning cycles.

**Linear throughput:** a battery-side adaptation of paper 3's linear baseline:

```
lambda = R / (2 * C_k * reference_cycle_depth * N_ref)
C_age = lambda * [sum_j dt_j*abs(p_chemical,j) + sum_j trip_energy_j]
```

It counts battery-side charging, V2H discharge and driving, not AC throughput.
MILP modes are exclusive; convex uses one signed chemical power. Both costs are
already cumulative dollars, so neither receives a second dt multiplier.

## Rainflow solve certificate

Exact rainflow is a numerical evaluator with an SOC subgradient, not a CVXPY atom.
We adapt the second paper's convex model to supporting-cut outer approximation:

1. Solve a master with a nonnegative incremental-cost epigraph theta.
2. Evaluate exact rainflow on the feasible SOC incumbent to obtain an upper bound U.
3. Read the master dual/lower bound L in original objective units, including offsets.
4. Add a supporting plane and repeat until the full-problem gap is certified.

Accept only if `U-L <= absolute_tolerance + relative_tolerance*max(abs(U), absolute_tolerance)`.
Additionally, for BOTH MILP and convex, require the retained incumbent's rainflow
envelope underestimation in unweighted dollars to be at most
`rainflow_absolute_tolerance + rainflow_relative_tolerance*abs(incremental_aging_cost)`.
Defaults are $0.000001 + 0.1% of incremental aging. MILP still uses its independent
5% whole-objective gap, but that cannot excuse an inaccurate rainflow envelope.
No arbitrary minimum cut count is imposed: a genuinely zero/exact aging term can
pass immediately. Large comfort constants no longer hide omitted aging dollars.
Missing lower-bound evidence, infeasibility, exhausted iterations/time, or excessive
gap triggers baseline fallback until the next scheduled solve. Native MILP master
optimality or native gap alone is insufficient. Feasibility/integrality checks
remain active. Convex discrete actuation may differ from the certified continuous plan.
Lower bounds include a small numerical guard; tolerances below backend precision
can exhaust the iteration limit and correctly fall back rather than claim convergence.

Internal solves share one 300-second elapsed OCP budget; each receives only the
remaining backend time. No new master starts after expiration. Backend termination
and CVXPY canonicalization can overrun slightly; this is not a killable process
watchdog. Quiet output remains enforced; a first-use Gurobi license banner can appear.
HPWH's existing solver/formulation behavior is unchanged.

Detailed solve history records iterations/cuts, exact upper and master lower
bounds, relative gap, final supporting-envelope error at the best incumbent,
historical cost, total internal solve time and fallback reason. The consolidated
report includes total/max master solves, uncertified attempts, and mean/max timings.

## Realized evaluation and economics

All selected cases use the same battery parameters and actual, bounded SOC path,
including pre/post charging and departure states so adjacent charging/trips cannot
hide extrema. Unserved driving energy is not invented as battery throughput.
Metrics are computed even when the OCP degradation term is disabled:

- `estimated_degradation_cost_usd`: selected OCP method evaluated on the realized path.
- `reference_degradation_cost_usd`: NMC average-SOC-aware rainflow cost,
  `R/(1-end_of_life_soh) * sum_i n_i*phi_NMC(depth_i, mean_i)`.
- Both depth-rainflow and linear diagnostics, battery throughput, replacement cost
  residual/depreciable values and reference cycle life are also retained.
- Departure count, shortfall events, and mean/max pre-departure SOC shortfall
  are reported separately. Existing SOC-fraction-hour comfort columns remain
  whole-window diagnostics; the OCP separately normalizes by N. Warm-up/cooldown
  departures are excluded. Calibration cycles/SOH and economic SOH are explicit
  alongside the effective economic reference cycle life.
- `operating_cost_savings_usd = all_baseline electricity cost - case electricity cost`.
- `degradation_cost_savings_usd = all_baseline reference cost - case reference cost`.
- `net_savings_usd = operating_cost_savings_usd + degradation_cost_savings_usd`.

These are **only comparisons against all_baseline**, never twin-relative savings.
Baseline degradation is included, so all-baseline net savings is exactly zero.
The old `savings_usd`/`net_profit_usd` columns are removed from new outputs;
previously saved experiments are not rewritten.
Whole-home degradation sums its EVs; no thermal-equipment aging is invented.
`net_savings_usd` excludes inventory credits. The separate
`energy_adjusted_net_savings_usd` accounts for differences from baseline in actual
starting/ending stored energy, valued at common boundary daily-minimum tariffs.
This reporting adjustment is independent of the OCP terminal option; see
[output accounting](pipeline.md#outputs). Inspect trip shortfalls alongside savings.

`rainflow_multicut` aliases the existing `rainflow` outer-loop formulation.
`degradation_method="none"` explicitly removes the OCP degradation term even if
the legacy `enable_degradation` switch is true; false also disables every method.
For `none`, selected-method estimated degradation is unavailable (NaN), not zero;
all actual depth-only, throughput and NMC evaluation diagnostics are still computed.
Smoothed rainflow and hh are deferred by user request, not implemented OCP methods.
See the [study guide](studies.md) for the Audi-only pilot and driver validation.

All results are in `data/performance.csv`, `data/comparison.csv`, per-case EV CSVs,
and the single `reports/run_report.md`. NMC aging is a reference model, not measured true
aging. Calendar, temperature and C-rate aging are excluded by design.

For publication, constant cell temperature of approximately 25 C is an explicit
idealized thermal-management assumption, not a demonstrated property of these
winter EV traces. Xu et al., *Factoring the Cycle Aging Cost of Batteries
Participating in Electricity Markets*, section V-A, uses maintained 25 C:
https://arxiv.org/html/1707.04567. Thermal-management energy and temperature-driven
aging differences are excluded here; conclusions are conditional on that scope.

Warm-up and cooldown evolve physical SOH and continuous rainflow history normally.
Evaluation costs sum only scored ledger increments, preserving spanning-cycle context.
They are NOT obtained by recounting an isolated scored SOC path. Warm-up/cooldown
increments are excluded from baseline and controlled metrics. Previously scored
increments are never retroactively rewritten when a residual cycle closes.

## Dynamic physical SOH and accounting

```
alpha = 1 - calibration_end_of_life_soh
D_0 = (1 - initial_soh)/alpha
A = (1 - end_of_life_soh)/alpha
c_D = R/A
delta_D = delta sum_i n_i * phi_NMC(d_i,m_i)/alpha
D_next = D + delta_D
SOH_next = 1 - alpha*D_next
C_next = capacity_kwh * SOH_next
incremental_dollars = c_D * delta_D
```

D is damage normalized to the calibration endpoint, not fractional capacity loss;
alpha*D is fractional capacity loss. At initial SOH .85, D_0=.75 and A=1.5.
The initial damage is never billed again. Economic EOL is a reporting flag, not
forced retirement; physical exhaustion (SOH <= 0) raises an explicit error.

The plant always uses SOC-aware reference rainflow, including baseline, fallback,
and `enable_degradation=false`. Objective choice never changes physical aging.
Each action first updates SOC using start-of-action capacity, then applies fade
while preserving that SOC. Lost accessible energy is separately recorded as
`SOC_end * (C_before - C_after)`; it is not another degradation dollar charge.
Trip-departure energy is applied before exposing fresh SOC/SOH to the controller;
pre-departure SOC is exposed separately for departure-comfort metrics. An ongoing trip is already
paid in measured initial SOC and is never debited again at arrival.
Repeated OCHRE model evaluations preview aging; only the accepted plant step commits it.

The streaming ledger keeps closed-cycle damage and open residue half cycles.
It does not count an independent half cycle at every step. The supported stress
parameters satisfy `k_soc <= 2*sqrt(beta*(beta-1))`, a conservative monotone-damage
condition on SOC in [0,1]; the approved beta=2.03, k_soc=1.04 satisfy it.
Unsupported parameters raise errors, never silently heal the battery or clip
material negative damage. Tiny roundoff is guarded separately.

Outputs add start/end SOH and capacity, stored energy, damage/cost increments,
throughput, fade-energy loss and the economic-EOL flag. The conventional EFC
metric divides physical throughput by twice nominal capacity; a separate
current-capacity EFC integrates actual SOC excursions. Health figures accompany
each EV trajectory/comparison. Saved plans and solve histories include measured
SOH and frozen capacity. `data/ev_battery_checkpoints.json` saves final battery
state and residue (including unscored phases/twin tail); it is NOT a complete
dwelling/controller restart. Calibration must match when restoring battery state.

## Sources and migration

Mathews et al., *Applied Energy* 269 (2020), 115127,
https://doi.org/10.1016/j.apenergy.2020.115127, degradation appendix: NMC depth
exponent 2.03, 3000 cycles at depth 0.8, mean-SOC reference 0.6, and signed
exponential SOC stress. We use its stress form, not its nonlinear fade accumulation.
The supplied `convex_rainflow_battery_degradation.pdf` still informs convex
depth-only rainflow; the third paper informs the linear-throughput approximation.
The supporting-cut solver is our adaptation. No pay-for-performance content is used.

Remove `b2_cycle_scale`, `b2_depth_coefficient`, `b2_depth_offset`, and `b2_soc_stress`
from home documents. Add `reference_cycle_count`, `soc_stress_coefficient`, and
`battery_residual_value_per_kwh`, and update depth/mean/exponent. Old B2 keys are
rejected rather than silently applied to the NMC model. Existing saved results,
archives, and staged study manifests are not rewritten; stage a new study directory
for this calibration instead of resuming an old-calibration study.

New unified documents must explicitly include `calibration_end_of_life_soh` and `initial_soh`.
For the earlier calibration, set it to 0.8. Economic EOL can then be 0.7 without
altering the physical fade law. Missing fields raise a descriptive error; archived
study inputs are not silently migrated by the normal runner.

Replace home `target_soc_min/max` with control `health_soc_min/max` and explicitly
set `comfort_soc_min`. Replace the two target weights with one `health_soc_weight`.
The bundled documents/templates are migrated. Python compatibility aliases remain,
but unequal old weights require an explicit new common weight; unified JSON rejects
old keys and requires the new bounds/calibration. No archive or native OCHRE files
were rewritten for this revision.

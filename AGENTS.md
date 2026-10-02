# Project instructions

## Confirm formulation changes with the user

The user requires explicit confirmation before changing a formulation in code or
math they have started. Treat their starter code, equations, and stated modeling
choices as authoritative design input.

- Before changing a formulation, explain the current formulation, the proposed
  change, and the reason. Wait for the user's explicit confirmation of that
  specific change before implementing it.
- This includes objectives, mean/max aggregation, scenario probabilities,
  weights, normalization, constraints, model dynamics, forecast assumptions,
  and which decisions are shared across scenarios.
- A general request to finish, fix, simplify, or refactor code does not authorize
  a formulation change. If the starter code, equations, and prose disagree,
  describe the discrepancy and ask which formulation to follow.
- Implementation work that preserves the agreed formulation can proceed.
  After an approved change, verify that the implementation and tests match the
  agreed math; do not change expected results merely to accommodate a different
  formulation.

## Agreed water-heater scenario objective

For both convex and MILP scenario MPC, electricity, legionella, and persistence
costs are arithmetic means across equally probable scenarios. Comfort is the
maximum of the per-scenario total horizon comfort costs, not the mean and not
the sum of separate per-step scenario maxima. Preserve the agreed weights and
time normalization unless the user explicitly confirms a change.

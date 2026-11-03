"""Named HPWH options shared by the joint CLI and experiment adapters.

Selecting a variant changes only these HPWH options. Forecast selection,
scenario count, horizon, HVAC, EV and tariff settings belong to their callers.
"""
from __future__ import annotations

import math


HPWH_VARIANTS = ("persistence", "rolling-starts")
_PRESETS = {
    "persistence": dict(comfort_weight=.1, legionella_weight=.001, persistence_weight=1.,
                        rolling_start_limit_24h=None, rolling_start_slack_weight=0.,
                        terminal_cycle_reserve_weight=0.),
    "rolling-starts": dict(comfort_weight=100., legionella_weight=.001, persistence_weight=.1,
                           rolling_start_limit_24h=4, rolling_start_slack_weight=100.,
                           terminal_cycle_reserve_weight=.1),
}


def resolve_hpwh_variant(
    name: str = "persistence", *, comfort_weight: float | None = None,
    legionella_weight: float | None = None, persistence_weight: float | None = None,
) -> dict:
    """Return fresh constructor options; explicit weights override named defaults."""
    if name not in _PRESETS:
        raise ValueError(f"Unknown HPWH variant {name!r}; choose from {HPWH_VARIANTS}")
    options = dict(_PRESETS[name])
    for key, value in (("comfort_weight", comfort_weight), ("legionella_weight", legionella_weight),
                       ("persistence_weight", persistence_weight)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"HPWH {key} must be finite and nonnegative")
            options[key] = float(value)
    options.update(terminal_weight=0., enforce_cycle_limit=False, enforce_dwell_constraints=False)
    return options


def hpwh_options_from_arguments(args) -> dict:
    """Support older programmatic namespaces while resolving only explicit knobs."""
    return resolve_hpwh_variant(
        getattr(args, "hpwh_variant", "persistence"),
        comfort_weight=getattr(args, "hpwh_comfort_weight", None),
        legionella_weight=getattr(args, "hpwh_legionella_weight", None),
        persistence_weight=getattr(args, "hpwh_persistence_weight", None),
    )


def add_hpwh_variant_arguments(parser) -> None:
    parser.add_argument("--hpwh-variant", choices=HPWH_VARIANTS, default="persistence",
                        help="Named HPWH formulation (default: existing persistence variant)")
    for name in ("comfort", "legionella", "persistence"):
        parser.add_argument(f"--hpwh-{name}-weight", type=float, default=None,
                            help=f"Explicit {name} weight override; otherwise use --hpwh-variant")


def effective_hpwh_variant(name: str, controller) -> dict:
    """Record resolved values rather than only the selected name or raw overrides."""
    fields = (*_PRESETS["persistence"], "terminal_weight", "enforce_cycle_limit", "enforce_dwell_constraints")
    return {"name": name, **{field: getattr(controller, field) for field in fields}}

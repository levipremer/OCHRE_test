"""Static distribution-transformer capacity model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DistributionTransformer:
    """Represent fixed transformer nameplate and overload capacities.

    This first iteration intentionally has no thermal or electrical dynamics.
    The capacity values are repeated at every aggregate timestamp so they can
    be saved and plotted alongside the five-home demand.
    """

    power_rating_kva: float
    overload_capacity_percent: float = 150.0

    def __post_init__(self) -> None:
        if self.power_rating_kva <= 0:
            raise ValueError("Transformer power_rating_kva must be positive")
        if self.overload_capacity_percent < 100:
            raise ValueError(
                "Transformer overload_capacity_percent must be at least 100"
            )

    @property
    def overload_capacity_kva(self) -> float:
        """Return the fixed overload capacity in kVA."""
        return self.power_rating_kva * self.overload_capacity_percent / 100

    def add_capacity_columns(self, aggregate_power: pd.DataFrame) -> pd.DataFrame:
        """Return aggregate data with constant transformer-capacity columns."""
        data = aggregate_power.copy()
        data["transformer_power_rating_kva"] = self.power_rating_kva
        data["transformer_overload_capacity_kva"] = self.overload_capacity_kva
        return data

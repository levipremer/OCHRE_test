"""Fuel-aware cooking equipment used only by the parametric study."""

from __future__ import annotations

import pandas as pd

from ochre.Equipment import Equipment
from ochre.Equipment.EventBasedLoad import EventBasedLoad


class FuelAwareCookingRange(EventBasedLoad):
    """Carry electric and gas cooking schedules through OCHRE event extraction.

    The OCHRE HPXML parser already creates both schedules for a gas range, but
    its general EventBasedLoad currently retains only electric event power.
    Keeping this class local prevents the study from changing baseline runs.
    """

    name = "Cooking Range"

    def __init__(self, **kwargs):
        self.gas_setpoint = 0.0
        super().__init__(**kwargs)
        electric = self.all_events.get("power", pd.Series(dtype=float)).fillna(0)
        gas = self.all_events.get("power_gas", pd.Series(dtype=float)).fillna(0)
        self.is_electric = bool(electric.abs().max() > 1e-12) if len(electric) else False
        self.is_gas = bool(gas.abs().max() > 1e-12) if len(gas) else False

    def extract_events(
        self, eq_powers: pd.DataFrame, random_offset=None, **kwargs
    ) -> pd.DataFrame:
        """Extract constant electric and gas power for each contiguous event."""
        on = eq_powers.fillna(0).abs().sum(axis=1) > 0
        start_flags = on & (~on).shift(fill_value=True)
        event_counts = start_flags.astype(int).cumsum()
        start_times = on.index[start_flags]
        end_flags = ~on & on.shift(fill_value=False)
        end_times = on.index[end_flags]
        if len(end_times) < len(start_times):
            end_times = end_times.append(
                pd.DatetimeIndex([self.start_time + self.duration])
            )
        if len(start_times) != len(end_times):
            raise ValueError(f"Cannot parse electric/gas events for {self.name}")

        if random_offset is not None and len(start_times):
            # Match the base implementation while applying one common offset to
            # electric and gas portions of an event.
            import numpy as np

            offsets = np.random.random(len(start_times)) * random_offset
            start_times += offsets
            end_times += offsets

        powers = eq_powers.loc[on].groupby(event_counts.loc[on]).mean()
        events = pd.DataFrame(
            {"start_time": start_times, "end_time": end_times}
        ).reset_index(drop=True)
        events["power"] = (
            powers.get("Power (kW)", pd.Series(0.0, index=powers.index))
            .reset_index(drop=True)
        )
        events["power_gas"] = (
            powers.get("Gas (therms/hour)", pd.Series(0.0, index=powers.index))
            .reset_index(drop=True)
        )
        return events

    def start_event(self) -> None:
        super().start_event()
        self.gas_setpoint = float(
            self.all_events.loc[self.event_index].get("power_gas", 0.0)
        )

    def end_event(self) -> None:
        super().end_event()
        self.gas_setpoint = 0.0

    def calculate_power_and_heat(self):
        if self.mode == "On":
            self.electric_kw = self.p_setpoint
            self.gas_therms_per_hour = self.gas_setpoint
        else:
            self.electric_kw = 0.0
            self.gas_therms_per_hour = 0.0
        Equipment.calculate_power_and_heat(self)

"""Replay measured rest-of-home electricity while retaining native heat gains."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ochre import Dwelling
from ochre.Equipment import Equipment


MEASURED_REST_OF_HOME_NAME = "DC House Rest of Home"
MEASURED_HUB_OTHER_NAME = "DC House Hub Other"
MEASURED_OUTSIDE_HUB_NAME = "DC House Outside Hub"
SEPARATE_END_USES = frozenset(
    {"HVAC Heating", "HVAC Cooling", "Water Heating", "EV", "PV", "Battery", "Generator"}
)


class MeasuredRestOfHomeLoad(Equipment):
    """A fixed electric meter replay with no additional envelope heat gains.

    Native appliances still supply their existing sensible and latent heat gains.
    The measured series supplies only its replacement electric meter reading.
    """

    end_use = "Other"
    zone_name = None

    def __init__(self, measured_rest_of_home_power_kw: pd.Series | pd.DataFrame, **kwargs):
        # DataFrames are supported so OCHRE's argument JSON can serialize the input.
        if isinstance(measured_rest_of_home_power_kw, pd.DataFrame):
            if measured_rest_of_home_power_kw.shape[1] != 1:
                raise ValueError("Measured rest-of-home replay must contain one power column")
            measured_rest_of_home_power_kw = measured_rest_of_home_power_kw.iloc[:, 0]
        power = measured_rest_of_home_power_kw.copy()
        if not isinstance(power.index, pd.DatetimeIndex):
            raise ValueError("Measured rest-of-home replay requires a DatetimeIndex")
        if power.empty or power.index.has_duplicates or not power.index.is_monotonic_increasing:
            raise ValueError("Measured rest-of-home replay timestamps must be nonempty, unique, and increasing")
        power = pd.to_numeric(power, errors="raise")
        if not np.isfinite(power.to_numpy()).all() or power.lt(0).any():
            raise ValueError("Measured rest-of-home replay power must be finite and nonnegative")
        self.measured_rest_of_home_power_kw = power
        kwargs["envelope_model"] = None
        super().__init__(**kwargs)

    def update_internal_control(self):
        power = self.measured_rest_of_home_power_kw.get(self.current_time, 0.0)
        return "On" if power > 0 else "Off"

    def update_external_control(self, control_signal):
        # This is an exogenous measured load, not a controllable appliance.
        return self.update_internal_control()

    def calculate_power_and_heat(self):
        self.electric_kw = (
            float(self.measured_rest_of_home_power_kw.get(self.current_time, 0.0))
            if self.mode == "On" else 0.0
        )
        self.reactive_kvar = 0.0

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 2:
            results[f"{self.results_name} Electric Power (kW)"] = self.electric_kw
            results[f"{self.results_name} Replay Active (-)"] = int(
                self.current_time in self.measured_rest_of_home_power_kw.index
            )
        return results


class RestOfHomeReplayDwelling(Dwelling):
    """Substitute native rest electricity after its heat has reached the envelope.

    OCHRE calls ``finish_sub_update`` after an equipment model adds its gains and
    before its power is included in dwelling totals. This preserves the existing
    thermal dynamics and gives batteries, raw results, and standardized results
    the same replacement electric load.
    """

    def __init__(self, measured_rest_of_home_power_kw: pd.Series, **kwargs):
        self._rest_of_home_replay_index = measured_rest_of_home_power_kw.index
        self._rest_of_home_replay_enabled = True
        super().__init__(**kwargs)

    def initialize(self, *args, **kwargs):
        enabled = self._rest_of_home_replay_enabled
        self._rest_of_home_replay_enabled = False
        try:
            return super().initialize(*args, **kwargs)
        finally:
            self._rest_of_home_replay_enabled = enabled

    def finish_sub_update(self, sub):
        replay_active = (
            self._rest_of_home_replay_enabled
            and self.current_time in self._rest_of_home_replay_index
        )
        if isinstance(sub, MeasuredRestOfHomeLoad):
            if not replay_active:
                sub.electric_kw = 0.0
                sub.reactive_kvar = 0.0
        elif (
            replay_active
            and isinstance(sub, Equipment)
            and sub.is_electric
            and sub.end_use not in SEPARATE_END_USES
        ):
            # Sensible and latent gains have already been added to their zones.
            sub.electric_kw = 0.0
            sub.reactive_kvar = 0.0
        super().finish_sub_update(sub)

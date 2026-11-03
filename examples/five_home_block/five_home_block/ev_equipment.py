"""Opt-in continuous-SOC EV equipment; positive AC power is consumption."""
from functools import lru_cache
import numpy as np
import pandas as pd
from ochre.Equipment.Equipment import Equipment
from .ev_degradation import BatteryDegradation, RainflowLedger
from .ev_power_loss import NMCProfile, PowerLossPlant


@lru_cache(maxsize=1)
def _power_loss_profile():
    return NMCProfile.from_csv()


class BidirectionalEV(Equipment):
    """OCHRE EV plant with arrival impulses, safety caps and continuous SOH accounting."""
    name = "EV"
    end_use = "EV"
    zone_name = None  # EV losses do not perturb the perfectly forecast thermal load
    modes = ["Charging", "Discharging", "Away", "Off"]

    def __init__(self, ev_config, mobility, **kwargs):
        if pd.Timedelta(kwargs["time_res"]) != pd.Timedelta(minutes=1):
            raise ValueError("HEMS EV simulation requires a one-minute time step")
        start = pd.Timestamp(kwargs["start_time"])
        if start != start.floor("min"):
            raise ValueError("HEMS EV simulation must start on a minute boundary")
        if isinstance(ev_config, dict):
            from types import SimpleNamespace
            from .ev_config import EVOptions
            ev_config = SimpleNamespace(**{**ev_config, "hems": EVOptions(**ev_config["hems"])})
        if isinstance(mobility, dict):
            from .ev_schedule import EVSchedule
            mobility = EVSchedule(**mobility)
        self.ev_config = ev_config
        self.options = ev_config.hems
        self.mobility = mobility
        # Existing thermal initialization can precede the EV experiment window.
        # The simulation sets this boundary before executing that prelude.
        self.activation_start = None
        self.trips = mobility.demand(self.options.consumption_kwh_per_km,
            soc_depletion_capacity_kwh=self.options.soc_depletion_capacity_kwh(
                ev_config.capacity_kwh*self.options.initial_soh))
        self._nmc_profile = _power_loss_profile() if self.options.battery_model == "power_loss" else None
        self.battery = BatteryDegradation(ev_config.capacity_kwh, self.options)
        self.initial_damage = (1-self.options.initial_soh)/self.battery.calibration_fade
        self.ledger = self.next_ledger = RainflowLedger(self.battery, ev_config.initial_soc_fraction)
        self.max_power = ev_config.charger_power_kw
        self.soc = self.next_soc = ev_config.initial_soc_fraction
        self._boundary = None
        self.in_event = False
        self.departure_shortfall = self.trip_energy = self.trip_unmet = 0.0
        self.departures = self.safety_override = 0
        self.requested_kw = 0.0
        self._backed_load = 0.0
        self.actual_backed_load = None
        self.actual_backed_kw = 0.0
        self.soc_saturation = 0
        self.soc_clipped_kwh = 0.0
        self._boundary_saturation = 0
        self._boundary_clipped_kwh = 0.0
        self._begin_totals = np.zeros(3)
        self._begin_throughput = 0.0
        self._begin_soh = self.options.initial_soh
        self._begin_energy = self.soc*self.capacity
        self._arrival_fade_energy = self._power_fade_energy = 0.0
        self._battery_loss_kwh = self._vehicle_standby_loss_kwh = 0.0
        self._terminal_energy_kwh = self._curtailed_ac_kwh = 0.0
        self._battery_limiting_reason = ""
        super().__init__(**kwargs)

    @property
    def damage(self):
        return self.initial_damage+self.ledger.totals[0]

    @property
    def soh(self):
        return self._soh(self.ledger)

    def _soh(self, ledger):
        value = 1-self.battery.calibration_fade*(self.initial_damage+ledger.totals[0])
        if value <= 0 or not np.isfinite(value):
            raise RuntimeError("EV physical capacity exhausted; economic EOL alone does not stop operation")
        return float(value)

    @property
    def capacity(self):
        """Current usable kWh, never used to reprice original pack value."""
        return self.ev_config.capacity_kwh*self.soh

    def prepare_boundary(self, timestamp):
        """Commit return impulses before exposing the current state to a controller."""
        timestamp = pd.Timestamp(timestamp)
        if self._boundary == timestamp:
            return
        self._begin_totals = self.ledger.totals.copy()
        self._begin_throughput = self.ledger.throughput_kwh
        self._begin_soh = self.soh
        self._begin_energy = self.soc*self.capacity
        self._arrival_fade_energy = self._power_fade_energy = 0.0
        self._battery_loss_kwh = self._vehicle_standby_loss_kwh = 0.0
        self._terminal_energy_kwh = self._curtailed_ac_kwh = 0.0
        self._battery_limiting_reason = ""
        self.departure_shortfall = self.trip_energy = self.trip_unmet = 0.0
        self.departures = 0
        self.soc_saturation = 0
        self.soc_clipped_kwh = 0.0
        self.soc = self._saturate(self.soc)
        if self.activation_start is not None and timestamp < pd.Timestamp(self.activation_start):
            self.next_soc, self.next_ledger = self.soc, self.ledger
            self.in_event = False
            self._boundary_saturation = self.soc_saturation
            self._boundary_clipped_kwh = self.soc_clipped_kwh
            self._boundary = timestamp
            return
        for trip in self.trips.loc[self.trips.arrival.eq(timestamp)].itertuples():
            old_capacity = self.capacity
            demand = (trip.arrival_net_soc_depletion_pp *
                      self.options.soc_depletion_capacity_kwh(old_capacity) / 100
                      if self.options.battery_model == "power_loss" else trip.energy_kwh)
            before = self.soc*old_capacity
            self.soc = self._saturate(self.soc - demand/old_capacity)
            served = before-self.soc*old_capacity
            self.ledger = self.ledger.preview(self.soc, old_capacity)
            self._arrival_fade_energy += self.soc*(old_capacity-self.capacity)
            self.trip_energy += served
            self.trip_unmet += demand - served
        for trip in self.trips.loc[self.trips.departure.eq(timestamp)].itertuples():
            self.departures += 1
            demand = (trip.arrival_net_soc_depletion_pp *
                      self.options.soc_depletion_capacity_kwh(self.capacity) / 100
                      if self.options.battery_model == "power_loss" else trip.energy_kwh)
            self.departure_shortfall += max(demand - self.soc*self.capacity, 0)
        self.next_soc = self.soc
        self.next_ledger = self.ledger
        self.in_event = bool(self.mobility.available([timestamp])[0])
        self._boundary_saturation = self.soc_saturation
        self._boundary_clipped_kwh = self.soc_clipped_kwh
        self._boundary = timestamp

    def _mode_for_request(self):
        if not self.in_event:
            return "Away"
        if self.requested_kw > 0:
            return "Charging"
        if self.requested_kw < 0:
            return "Discharging"
        return "Off"

    def _saturate(self, value):
        """Clip finite physical SOC and retain energy-equivalent diagnostics."""
        if not np.isfinite(value):
            raise ValueError("EV SOC must be finite")
        clipped = float(np.clip(value, 0, 1))
        if abs(clipped-value) > 1e-10:
            self.soc_saturation = 1
            self.soc_clipped_kwh += abs(clipped-value)*self.capacity
        return clipped

    def update_internal_control(self):
        """Immediate charging to comfort_soc_min, never discharging."""
        self.prepare_boundary(self.current_time)
        self.requested_kw = self.baseline_power() if self.in_event else 0.0
        self._backed_load = 0.0
        return self._mode_for_request()

    def baseline_power(self):
        hours = self.time_res.total_seconds()/3600
        return min(self.max_power, max(0, self.options.comfort_soc_min-self.soc)*
                   self.capacity/(hours*self.options.charge_efficiency))

    def update_external_control(self, control_signal):
        self.prepare_boundary(self.current_time)
        self.requested_kw = float(control_signal.get("P Setpoint", 0))
        self._backed_load = float(control_signal.get("Backed Load (kW)", 0))
        if not np.isfinite([self.requested_kw, self._backed_load]).all():
            raise ValueError("EV commands must be finite")
        return self._mode_for_request()

    def calculate_power_and_heat(self):
        o = self.options
        hours = self.time_res.total_seconds()/3600
        if o.battery_model == "power_loss":
            self._calculate_power_loss(hours)
            return
        charge_limit = min(self.max_power, (1-self.soc)*self.capacity/(hours*o.charge_efficiency))
        self.actual_backed_kw = (float(self.actual_backed_load()) if self.actual_backed_load
                                 else max(self._backed_load, 0))
        discharge_limit = min(o.max_discharge_kw, self.actual_backed_kw,
                              self.soc*self.capacity*o.discharge_efficiency/hours) if o.v2h else 0
        connected = self.in_event and self.current_schedule.get("Voltage (-)", 1) != 0
        power = float(np.clip(self.requested_kw, -discharge_limit, charge_limit)) if connected else 0.0
        self.safety_override = int(abs(power-self.requested_kw) > 1e-7)
        self.electric_kw = power
        dc = power*o.charge_efficiency if power >= 0 else power/o.discharge_efficiency
        requested_dc = self.requested_kw*o.charge_efficiency if self.requested_kw >= 0 else self.requested_kw/o.discharge_efficiency
        if connected:
            # Record physical SOC saturation even when power clipping prevents it.
            self._saturate(self.soc + requested_dc*hours/self.capacity)
        self.next_soc = self._saturate(self.soc + dc*hours/self.capacity)
        # Preview only. update_results commits the final model evaluation once.
        self.next_ledger = self.ledger.preview(self.next_soc, self.capacity)
        next_capacity = self.ev_config.capacity_kwh*self._soh(self.next_ledger)
        self._power_fade_energy = self.next_soc*(self.capacity-next_capacity)
        self.sensible_gain = self.latent_gain = 0.0
        self.mode = ("Charging" if power > 1e-8 else "Discharging" if power < -1e-8
                     else "Off" if self.in_event else "Away")

    def _calculate_power_loss(self, hours):
        """Preview a fresh one-minute plant state; commit only in update_results."""
        o = self.options
        self.soc_saturation = self._boundary_saturation
        self.soc_clipped_kwh = self._boundary_clipped_kwh
        self.actual_backed_kw = (float(self.actual_backed_load()) if self.actual_backed_load
                                 else max(self._backed_load, 0))
        if not self.in_event:
            # Away energy is updated only by the supplied arrival SOC impulse.
            self.electric_kw = 0.0
            self.next_soc, self.next_ledger = self.soc, self.ledger
            self._battery_loss_kwh = self._vehicle_standby_loss_kwh = 0.0
            self._terminal_energy_kwh = 0.0
            self._curtailed_ac_kwh = abs(self.requested_kw)*hours
            self._battery_limiting_reason = "away"
            self._power_fade_energy = 0.0
            self.safety_override = int(abs(self.requested_kw) > 1e-7)
            self.sensible_gain = self.latent_gain = 0.0
            self.mode = "Away"
            return
        discharge_limit = min(o.max_discharge_kw, self.actual_backed_kw) if o.v2h else 0.0
        connected = self.in_event and self.current_schedule.get("Voltage (-)", 1) != 0
        request = float(np.clip(self.requested_kw, -discharge_limit, self.max_power)) if connected else self.requested_kw
        if connected:
            requested_dc = (self.requested_kw*o.charge_efficiency if self.requested_kw >= 0
                            else self.requested_kw/o.discharge_efficiency)
            self._saturate(self.soc+requested_dc*hours/self.capacity)
        model = PowerLossPlant(
            self._nmc_profile, self.capacity, self.soc,
            eta_charge=o.charge_efficiency, eta_discharge=o.discharge_efficiency,
            charger_charge_kw=self.max_power, charger_discharge_kw=o.max_discharge_kw,
            parameter_usable_kwh=self.ev_config.capacity_kwh*o.initial_soh,
            max_internal_step_s=60.0,
        )
        result = model.step(request, self.time_res.total_seconds(), plugged_in=connected)
        self.electric_kw = result.actual_ac_average_kw
        # At home idle, P_DC is zero, so integrating the standby term is an
        # exact energy subtraction. Charge and discharge minutes have no standby.
        self._vehicle_standby_loss_kwh = (min(result.stored_energy_kwh, o.standby_vehicle_kw*hours)
                                          if abs(self.electric_kw) <= 1e-8 else 0.0)
        self.next_soc = self._saturate(
            (result.stored_energy_kwh-self._vehicle_standby_loss_kwh)/self.capacity)
        self._battery_loss_kwh = result.loss_energy_kwh
        self._terminal_energy_kwh = result.terminal_energy_kwh
        self._curtailed_ac_kwh = max(0.0, abs(self.requested_kw)*hours
                                     - result.ac_import_kwh-result.ac_export_kwh)
        self._battery_limiting_reason = result.limiting_reason
        self.safety_override = int(abs(self.electric_kw-self.requested_kw) > 1e-7)
        self.next_ledger = self.ledger.preview(self.next_soc, self.capacity)
        next_capacity = self.ev_config.capacity_kwh*self._soh(self.next_ledger)
        self._power_fade_energy = self.next_soc*(self.capacity-next_capacity)
        self.sensible_gain = self.latent_gain = 0.0
        self.mode = ("Charging" if self.electric_kw > 1e-8 else
                     "Discharging" if self.electric_kw < -1e-8 else
                     "Off" if self.in_event else "Away")

    def generate_results(self):
        result = super().generate_results()
        end_soh = self._soh(self.next_ledger)
        end_capacity = self.ev_config.capacity_kwh*end_soh
        increments = self.next_ledger.totals-self._begin_totals
        price = self.battery.damage_price
        result.update({f"{self.results_name} {key}": value for key, value in {
            "SOC (-)": self.soc, "End SOC (-)": self.next_soc, "Parked": self.in_event,
            "Start SOH (-)": self._begin_soh, "SOH (-)": self.soh, "End SOH (-)": end_soh,
            "Capacity (kWh)": self.capacity, "End Capacity (kWh)": end_capacity,
            "Start Stored Energy (kWh)": self._begin_energy,
            "Stored Energy (kWh)": self.soc*self.capacity,
            "End Stored Energy (kWh)": self.next_soc*end_capacity,
            "End Damage (-)": self.initial_damage+self.next_ledger.totals[0],
            "Damage Increment (-)": increments[0],
            "Rainflow Damage Increment (-)": increments[1],
            "Throughput Damage Increment (-)": increments[2],
            "Reference Degradation Cost ($)": price*increments[0],
            "Rainflow Degradation Cost ($)": price*increments[1],
            "Throughput Degradation Cost ($)": price*increments[2],
            "Cumulative Reference Degradation Cost ($)": price*self.next_ledger.totals[0],
            "Battery Throughput (kWh)": self.next_ledger.throughput_kwh-self._begin_throughput,
            "Fade Energy Loss (kWh)": self._arrival_fade_energy+self._power_fade_energy,
            "Battery Loss (kWh)": self._battery_loss_kwh,
            "Vehicle Standby Loss (kWh)": self._vehicle_standby_loss_kwh,
            "Terminal Energy (kWh)": self._terminal_energy_kwh,
            "Curtailed AC Energy (kWh)": self._curtailed_ac_kwh,
            "Battery Limit Reason": self._battery_limiting_reason,
            "Economic EOL Reached": int(end_soh <= self.options.end_of_life_soh),
            "Unmet Load (kWh)": self.departure_shortfall,
            "Departures": self.departures, "Trip Energy (kWh)": self.trip_energy,
            "Trip Unserved Energy (kWh)": self.trip_unmet,
            "Requested Power (kW)": self.requested_kw, "Safety Override": self.safety_override,
            "Actual Backed Load (kW)": self.actual_backed_kw,
            "SOC Saturation": self.soc_saturation, "SOC Clipped Energy (kWh)": self.soc_clipped_kwh,
        }.items()})
        return result

    def update_results(self):
        result = super().update_results()
        self.ledger = self.next_ledger
        self.soc = self.next_soc
        return result

    def battery_checkpoint(self):
        """Committed battery state; restore only at an unprocessed step boundary."""
        return dict(soc=self.soc, initial_damage=self.initial_damage, ledger=self.ledger.snapshot(),
                    nominal_capacity_kwh=self.ev_config.capacity_kwh,
                    battery_model=self.options.battery_model,
                    standby_vehicle_kw=self.options.standby_vehicle_kw,
                    charge_efficiency=self.options.charge_efficiency,
                    discharge_efficiency=self.options.discharge_efficiency,
                    soc_depletion_reference_capacity_kwh=self.options.soc_depletion_reference_capacity_kwh,
                    calibration=self._checkpoint_calibration())

    def _checkpoint_calibration(self):
        return {key: getattr(self.options, key) for key in (
            "calibration_end_of_life_soh", "end_of_life_soh", "reference_cycle_depth",
            "reference_mean_soc", "reference_cycle_count", "rainflow_exponent",
            "soc_stress_coefficient", "battery_replacement_cost_per_kwh", "battery_residual_value_per_kwh")}

    def restore_battery_checkpoint(self, state):
        if state["nominal_capacity_kwh"] != self.ev_config.capacity_kwh or not np.isclose(
                state["initial_damage"], self.initial_damage) or state.get("calibration") != self._checkpoint_calibration() or \
                state.get("battery_model", "fixed_efficiency") != self.options.battery_model or \
                state.get("standby_vehicle_kw", 0.0) != self.options.standby_vehicle_kw or \
                state.get("charge_efficiency", self.options.charge_efficiency) != self.options.charge_efficiency or \
                state.get("discharge_efficiency", self.options.discharge_efficiency) != self.options.discharge_efficiency or \
                state.get("soc_depletion_reference_capacity_kwh") != self.options.soc_depletion_reference_capacity_kwh:
            raise ValueError("EV checkpoint battery initialization does not match configuration")
        soc = float(state["soc"])
        ledger = RainflowLedger.restore(self.battery, state["ledger"])
        if not np.isfinite(soc) or not 0 <= soc <= 1 or not np.isclose(soc, ledger.residue[-1]):
            raise ValueError("EV checkpoint SOC does not match damage residue")
        self._soh(ledger)
        self.soc = self.next_soc = soc
        self.ledger = self.next_ledger = ledger
        self._boundary = None
        self._boundary_saturation = 0
        self._boundary_clipped_kwh = 0.0

    def reset_time(self, *args, **kwargs):
        super().reset_time(*args, **kwargs)
        self.soc = self.next_soc = self.ev_config.initial_soc_fraction
        self.ledger = self.next_ledger = RainflowLedger(self.battery, self.soc)
        self._begin_totals = np.zeros(3)
        self._begin_throughput = 0.0
        self._begin_soh = self.soh
        self._begin_energy = self.soc*self.capacity
        self._arrival_fade_energy = self._power_fade_energy = 0.0
        self._battery_loss_kwh = self._vehicle_standby_loss_kwh = 0.0
        self._terminal_energy_kwh = self._curtailed_ac_kwh = 0.0
        self._battery_limiting_reason = ""
        self._boundary = None
        self._boundary_saturation = 0
        self._boundary_clipped_kwh = 0.0

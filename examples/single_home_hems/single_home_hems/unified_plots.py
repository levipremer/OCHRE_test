"""Consistent case-local and overlaid plots for the three-case pipeline."""
from pathlib import Path
import matplotlib.pyplot as plt
from .plotting import _format_time_axis
from .ev_metrics import ev_operating_state


def _finish(fig, axes, title, path, dpi):
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(alpha=0.2)
        _format_time_axis(axis)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_cases(cases, home, root, dpi):
    """Overlay evaluation-only baseline, twin and advanced realized trajectories."""
    directory = Path(root)/"figures"
    directory.mkdir(exist_ok=True)
    for case, data in cases.items():
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        mapped = data["mapped"]
        axes[0].plot(mapped.index, mapped.total_electric_power_kw, label="Whole home")
        for col in ("hvac_power_kw", "water_heating_power_kw", "ev_power_kw"):
            axes[0].plot(mapped.index, mapped[col], label=col)
        misc = mapped.total_electric_power_kw-mapped.hvac_power_kw-mapped.water_heating_power_kw-mapped.ev_power_kw
        axes[0].plot(mapped.index, misc, label="Miscellaneous")
        axes[0].set_ylabel("Power (kW)")
        axes[1].plot(mapped.index, data["commands"].electricity_price_usd_per_kwh, label="Price")
        axes[1].set_ylabel("USD/kWh")
        _finish(fig, axes, f"{home.home_id}: {case}", Path(root)/case/"figures/whole_home.png", dpi)
    for ev in home.evs:
        name = f"EV_{ev.vehicle_id}"
        groups = [(cases, directory/f"{name}_comparison.png")]
        groups += [({case: data}, Path(root)/case/"figures"/f"{name}.png") for case, data in cases.items()]
        for group, path in groups:
            fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
            for label, data in group.items():
                raw = data["raw"]
                parked = raw.get(f"{name} Parked", raw.get(f"{name} Parked (-)"))
                if parked is None:
                    # Native OCHRE exposes parked through its mode.
                    mode = raw.get(f"{name} Mode")
                    if mode is None:
                        raise ValueError(f"Missing home/away output for {name}")
                    parked = mode.ne("Away")
                axes[0].step(raw.index, parked.astype(int), where="post", label=label)
                axes[1].plot(raw.index, raw.get(f"{name} End SOC (-)", raw[f"{name} SOC (-)"]), label=label)
                axes[2].plot(raw.index, data["commands"].electricity_price_usd_per_kwh, label=label)
                axes[3].plot(raw.index, raw[f"{name} Electric Power (kW)"], label=label)
                modes = ev_operating_state(raw, name).map({"away": -2, "discharging": -1, "idle": 0, "charging": 1})
                axes[4].step(raw.index, modes, where="post", label=label)
            for ax, ylabel in zip(axes, ("Home / away", "SOC", "Price (USD/kWh)", "AC power (kW)", "Actual mode")):
                ax.set_ylabel(ylabel)
            axes[0].set_yticks([0, 1], ["Away", "Home"])
            axes[4].set_yticks([-2, -1, 0, 1], ["Away", "Discharge", "Idle", "Charge"])
            _finish(fig, axes, f"{home.home_id}: {name}", path, dpi)
            if all(f"{name} End SOH (-)" in data["raw"] for data in group.values()):
                fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
                for label, data in group.items():
                    raw = data["raw"]
                    axes[0].plot(raw.index, raw[f"{name} End SOH (-)"], label=label)
                    axes[1].plot(raw.index, raw[f"{name} End Capacity (kWh)"], label=label)
                    axes[2].plot(raw.index, raw[f"{name} Reference Degradation Cost ($)"].cumsum(), label=label)
                for axis, ylabel in zip(axes, ("SOH", "Usable capacity (kWh)", "Scored reference aging ($)")):
                    axis.set_ylabel(ylabel)
                    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
                _finish(fig, axes, f"{home.home_id}: {name} battery health", path.with_stem(path.stem+"_health"), dpi)
    for device in ("hvac", "hpwh", "whole_home"):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for case, data in cases.items():
            mapped, schedule = data["mapped"], data["schedule"]
            if device == "hvac":
                axes[0].plot(mapped.index, schedule["Ambient Dry Bulb (C)"], label=f"{case}: outdoor")
                axes[1].plot(mapped.index, mapped.indoor_temperature_c, label=f"{case}: indoor")
                for col in ("heating_setpoint_c", "cooling_setpoint_c"):
                    if col in mapped:
                        axes[1].plot(mapped.index, mapped[col], "--", alpha=0.4)
                power = mapped.hvac_power_kw
            elif device == "hpwh":
                axes[0].plot(mapped.index, data["water_training"].water_heater_draw_l_per_min, label=case)
                for col in data["sensors"]:
                    axes[1].plot(mapped.index, data["sensors"][col], label=f"{case}: {col}")
                power = mapped.water_heating_power_kw
            else:
                axes[0].plot(mapped.index, data["commands"].electricity_price_usd_per_kwh, label=case)
                axes[1].plot(mapped.index, mapped.ev_power_kw, label=f"{case}: EVs")
                power = mapped.total_electric_power_kw
            axes[2].plot(mapped.index, power, label=case)
        axes[0].set_ylabel({"hvac": "Outdoor (C)", "hpwh": "Water (L/min)", "whole_home": "Price (USD/kWh)"}[device])
        axes[1].set_ylabel("Temperature (C)" if device != "whole_home" else "EV power (kW)")
        axes[2].set_ylabel("Power (kW)")
        if device != "whole_home":
            price_axis = axes[2].twinx()
            first = next(iter(cases.values()))
            price_axis.plot(first["commands"].index, first["commands"].electricity_price_usd_per_kwh, ":", color="grey")
            price_axis.set_ylabel("Price (USD/kWh)")
        _finish(fig, axes, f"{home.home_id}: {device}", directory/f"{device}_comparison.png", dpi)

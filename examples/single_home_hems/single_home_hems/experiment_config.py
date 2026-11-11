"""Named whole-home experiments layered over the existing household schema."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .control_config import COMMON, EV_CONTROL, keys, load_unified_documents, merge
from .experiment_controllers import jsonable, resolve_thermal_control, validate_thermal_controllers

CONTROL_SECTIONS = {"shared", "hvac", "water_heater", "ev", "joint"}
HOME_SECTIONS = {"hvac", "water_heater", "joint", "ev_defaults", "evs"}
SHARED_SETTINGS = {"horizon_hours", "control_minutes", "solver", "solver_options"}


@dataclass
class ExperimentVariant:
    name: str
    unified: object
    thermal_controls: dict
    options: dict

    def resolved(self):
        """Every effective controller and household input, ready for audit JSON."""
        return jsonable({"name": self.name, "options": self.options,
                         "thermal_controls": self.thermal_controls,
                         "ev_controls": self.unified.controls,
                         "home_settings": self.unified.home_settings,
                         "neighborhood": self.unified.neighborhood,
                         "warmup_hours": self.unified.warmup_hours,
                         "cooldown_hours": self.unified.cooldown_hours})


@dataclass
class ExperimentConfiguration:
    path: Path
    requested: dict
    variants: tuple[ExperimentVariant, ...]

    def resolved(self):
        return {"experiment_file": str(self.path),
                "precedence": ["existing controller defaults", "experiment defaults",
                               "variant controls", "home device/vehicle overrides"],
                "variants": {variant.name: variant.resolved() for variant in self.variants}}


def _control_sections(value, label):
    keys(value, CONTROL_SECTIONS, label)
    keys(value.get("shared", {}), SHARED_SETTINGS, f"{label}.shared")
    for section in CONTROL_SECTIONS - {"shared"}:
        if not isinstance(value.get(section, {}), dict):
            raise ValueError(f"{label}.{section} must be an object")
    keys(value.get("ev", {}), COMMON | EV_CONTROL, f"{label}.ev")


def _home_sections(value, homes_doc, label):
    keys(value, homes_doc["homes"], label)
    for home_id, home in value.items():
        keys(home, HOME_SECTIONS, f"{label}.{home_id}")
        for section in HOME_SECTIONS - {"evs"}:
            if not isinstance(home.get(section, {}), dict):
                raise ValueError(f"{label}.{home_id}.{section} must be an object")
        keys(home.get("evs", {}), homes_doc["homes"][home_id]["evs"], f"{label}.{home_id}.evs")
        for vehicle_id, settings in home.get("evs", {}).items():
            if not isinstance(settings, dict):
                raise ValueError(f"{label}.{home_id}.evs.{vehicle_id} must be an object")
            keys(settings, COMMON | EV_CONTROL, f"{label}.{home_id}.evs.{vehicle_id}")
        keys(home.get("ev_defaults", {}), COMMON | EV_CONTROL, f"{label}.{home_id}.ev_defaults")


def _device_request(defaults, variant, home, device):
    specific = "ev_defaults" if device == "ev" else device
    plain = merge(merge(defaults.get(device, {}), variant.get(device, {})), home.get(specific, {}))
    # Shared settings describe active optimizers, not native baseline policies.
    # EV vehicles may activate MPC individually, so retain shared EV defaults.
    active = device == "ev" or plain.get("controller", "baseline") == "mpc"
    base = merge(defaults.get("shared", {}) if active else {}, defaults.get(device, {}))
    override = merge(variant.get("shared", {}) if active else {}, variant.get(device, {}))
    return merge(merge(base, override), home.get(specific, {}))


def _ev_defaults():
    # EVOptions deliberately requires explicit SOC thresholds. Reuse the
    # existing shipped EV comparison defaults instead of inventing thresholds
    # for households whose EVs remain on native charging in a thermal study.
    source = Path(__file__).resolve().parents[1] / "config" / "unified_control.json"
    return deepcopy(json.loads(source.read_text(encoding="utf-8"))["defaults"]["ev"])


def load_experiment(path, repo_root, *, variant_names=None, home_ids=None, output_directory=None):
    """Resolve named variants without writing files or invoking an optimizer.

    Experiment paths are relative to this JSON file. Physical paths retain the
    standalone household loader's existing repository/config-directory rules.
    """
    path, root = Path(path).expanduser().resolve(), Path(repo_root).resolve()
    requested = json.loads(path.read_text(encoding="utf-8"))
    keys(requested, {"home_config", "defaults", "homes", "variants", "simulation", "output_directory", "_comments"}, "experiment")
    if not isinstance(requested.get("home_config"), str) or not requested["home_config"]:
        raise ValueError("experiment.home_config must name a household JSON file")
    home_file = Path(requested["home_config"]).expanduser()
    if not home_file.is_absolute():
        home_file = path.parent / home_file
    homes_doc = json.loads(home_file.read_text(encoding="utf-8"))
    keys(homes_doc, {"simulation", "homes", "_comments"}, "home document")
    if not isinstance(homes_doc.get("homes"), dict) or not homes_doc["homes"]:
        raise ValueError("Household inputs must contain at least one home")
    for home_id, home in homes_doc["homes"].items():
        if not isinstance(home.get("evs"), dict):
            raise ValueError(f"homes.{home_id}.evs must be an object (empty is supported)")
    simulation = requested.get("simulation", {})
    keys(simulation, {"start_time", "duration_hours", "warmup_hours", "cooldown_hours"}, "experiment.simulation")
    homes_doc["simulation"] = merge(homes_doc.get("simulation", {}), simulation)
    destination = output_directory if output_directory is not None else requested.get("output_directory")
    if destination is not None:
        destination = Path(destination).expanduser()
        if not destination.is_absolute():
            destination = path.parent / destination
        homes_doc["simulation"]["output_directory"] = str(destination.resolve())
    defaults, common_homes = requested.get("defaults", {}), requested.get("homes", {})
    _control_sections(defaults, "defaults")
    _home_sections(common_homes, homes_doc, "homes")
    variants = requested.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("experiment.variants must contain named variant objects")
    chosen = list(variants) if variant_names is None else list(variant_names)
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(variants):
        raise ValueError("Select existing, unique experiment variant names")
    selected_homes = list(homes_doc["homes"]) if home_ids is None else list(home_ids)
    if not selected_homes or len(selected_homes) != len(set(selected_homes)) or set(selected_homes) - set(homes_doc["homes"]):
        raise ValueError("Select existing, unique home IDs")
    for name, spec in variants.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Variant name must be safe as one directory name: {name!r}")
        keys(spec, {"controls", "homes", "options", "description"}, f"variants.{name}")
        _control_sections(spec.get("controls", {}), f"variants.{name}.controls")
        _home_sections(spec.get("homes", {}), homes_doc, f"variants.{name}.homes")
        options = spec.get("options", {})
        keys(options, {"ev_only_comparison"}, f"variants.{name}.options")
        if not isinstance(options.get("ev_only_comparison", False), bool):
            raise ValueError("ev_only_comparison must be boolean")
    loaded = []
    for name in chosen:
        spec = variants[name]
        controls = spec.get("controls", {})
        overrides = merge(common_homes, spec.get("homes", {}))
        physical = deepcopy(homes_doc)
        physical["homes"] = {home_id: physical["homes"][home_id] for home_id in selected_homes}
        # Resolve water replay paths once. The EV runner expects explicit paths.
        for home in physical["homes"].values():
            if home.get("water_draw_file"):
                water = Path(home["water_draw_file"]).expanduser()
                if not water.is_absolute():
                    local = home_file.resolve().parent / water
                    water = local if local.exists() else root / water
                home["water_draw_file"] = str(water.resolve())
        ev_document = {"defaults": {"ev": _ev_defaults()}, "homes": {}}
        thermal = {}
        for home_id, home in physical["homes"].items():
            home_overrides = overrides.get(home_id, {})
            ev_default = _device_request(defaults, controls, home_overrides, "ev")
            evs = deepcopy(home_overrides.get("evs", {}))
            for vehicle_id in home["evs"]:
                request = merge(ev_default, evs.get(vehicle_id, {}))
                forecast = request.get("forecast", {})
                if forecast.get("file"):
                    forecast_file = Path(forecast["file"]).expanduser()
                    if not forecast_file.is_absolute():
                        forecast_file = path.parent / forecast_file
                    forecast["file"] = str(forecast_file.resolve())
                    if not forecast_file.is_file():
                        raise FileNotFoundError(f"EV forecast input not found: {forecast_file}")
                evs[vehicle_id] = request
            ev_document["homes"][home_id] = {"evs": evs}
            thermal[home_id] = {device: resolve_thermal_control(
                device, _device_request(defaults, controls, home_overrides, device),
                base_directory=path.parent, repo_root=root)
                for device in ("hvac", "water_heater", "joint")}
            selections = thermal[home_id]
            if selections["joint"]["controller"] == "mpc" and any(selections[d]["controller"] == "mpc" for d in ("hvac", "water_heater")):
                raise ValueError(f"{name}/{home_id}: joint thermal control conflicts with independent HVAC/water-heater MPC")
        unified = load_unified_documents(physical, ev_document, home_file, root)
        variant = ExperimentVariant(name, unified, thermal, {"ev_only_comparison": spec.get("options", {}).get("ev_only_comparison", False)})
        for home in unified.neighborhood.homes:
            validate_thermal_controllers(variant, home.home_id, home)
        loaded.append(variant)
    return ExperimentConfiguration(path, requested, tuple(loaded))


def resolved_experiment(experiment):
    """Convenient functional form for callers saving the resolved audit file."""
    return experiment.resolved()

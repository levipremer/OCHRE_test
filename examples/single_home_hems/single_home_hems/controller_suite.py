"""Config-only front door for existing single-home controller runners.

This dispatcher builds no optimization problem and advances no plant state.
Each selected case uses its existing runner in an isolated subprocess.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .experiment_controllers import jsonable


MODE_MODULES = {
    "hvac_only": "examples.single_home_hems.run_simulation",
    "water_heater_only": "examples.single_home_hems.run_simulation",
    "joint_thermal": "examples.single_home_hems.run_joint_hems",
    "whole_home_joint": "examples.single_home_hems.run_whole_home_joint",
    "bidirectional_charger_only": "examples.single_home_hems.run_ev_home_hub",
}
FORCED_ARGUMENTS = {
    "hvac_only": {"controller": "hvac"},
    "water_heater_only": {"controller": "water-heater"},
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
RESERVED_ARGUMENTS = frozenset({"controller", "output_directory", "validate_only", "help"})


@dataclass(frozen=True)
class ControllerCase:
    name: str
    controller: str
    argv: tuple[str, ...]
    effective_settings: dict[str, Any]


@dataclass(frozen=True)
class ControllerSuite:
    path: Path
    requested: dict[str, Any]
    output_directory: Path
    cases: tuple[ControllerCase, ...]

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_file": str(self.path),
            "output_directory": str(self.output_directory),
            "selected_cases": [case.name for case in self.cases],
            "cases": {
                case.name: {
                    "controller": case.controller,
                    "command": [sys.executable, *case.argv],
                    "effective_arguments": case.effective_settings,
                    "output_directory": str(self.output_directory / case.name),
                }
                for case in self.cases
            },
        }


def _keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} key(s): {', '.join(sorted(unknown))}")


def _config_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _runner_arguments(
    module: Any, requested: dict[str, Any], base: Path, output: Path,
    forced: dict[str, str],
):
    parser = module.build_argument_parser()
    actions = {action.dest: action for action in parser._actions if action.option_strings}
    _keys(requested, set(actions) - RESERVED_ARGUMENTS, "runner arguments")
    argv: list[str] = []
    for name, value in requested.items():
        action = actions[name]
        option = next((flag for flag in action.option_strings if flag.startswith("--")), None)
        if option is None:
            raise ValueError(f"{name} has no long command-line option")
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
            if value:
                argv.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
            if not value:
                argv.append(option)
            continue
        if value is None:
            raise ValueError(f"{name} cannot be null; omit it to use the runner default")
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list) and not values:
            raise ValueError(f"{name} cannot be an empty list")
        if any(isinstance(item, (dict, list)) or type(item) is bool for item in values):
            raise ValueError(f"{name} must contain command-line scalar values")
        if action.type is Path:
            values = [str(_config_path(str(item), base)) for item in values]
            for path in values:
                if not Path(path).exists():
                    raise FileNotFoundError(f"{name} input does not exist: {path}")
        argv.extend([option, *(str(item) for item in values)])
    argv.extend(["--output-directory", str(output)])
    if forced:
        argv.extend(["--controller", forced["controller"]])
    validator = getattr(module, "_validated_args", None)
    parsed = validator(argv) if validator is not None else parser.parse_args(argv)
    return argv, jsonable(vars(parsed))


def load_controller_suite(
    path: str | Path,
    repo_root: str | Path,
    *,
    case_names: list[str] | None = None,
    home_ids: list[str] | None = None,
    output_directory: str | Path | None = None,
) -> ControllerSuite:
    """Validate and resolve selected cases without creating any output files."""
    path = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()
    if not (root / "examples/single_home_hems").is_dir():
        raise ValueError(f"Repository root lacks the single-home runners: {root}")
    requested = json.loads(path.read_text(encoding="utf-8"))
    _keys(requested, {"schema_version", "output_directory", "selected", "variants", "_comments"}, "controller suite")
    if type(requested.get("schema_version")) is not int or requested["schema_version"] != 1:
        raise ValueError("controller suite schema_version must be 1")
    variants = requested.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("controller suite variants must be a nonempty object")
    for name, spec in variants.items():
        if not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None:
            raise ValueError(f"Unsafe controller case name: {name!r}")
        _keys(spec, {"controller", "arguments", "description"}, f"variants.{name}")
        if spec.get("controller") not in MODE_MODULES:
            raise ValueError(f"variants.{name}.controller must be one of {', '.join(sorted(MODE_MODULES))}")
    selected = case_names if case_names is not None else requested.get("selected")
    if not isinstance(selected, list) or not selected or any(not isinstance(name, str) for name in selected):
        raise ValueError("Select one or more case names with selected or --variants")
    if len(selected) != len(set(selected)) or set(selected) - set(variants):
        raise ValueError("Selected controller cases must exist and be unique")
    destination = output_directory if output_directory is not None else requested.get("output_directory")
    if not isinstance(destination, (str, Path)) or not str(destination):
        raise ValueError("controller suite output_directory is required")
    output = (
        Path(destination).expanduser().resolve()
        if output_directory is not None else _config_path(str(destination), path.parent)
    )
    if home_ids is not None and (not home_ids or len(home_ids) != 1):
        raise ValueError("Controller suite --homes currently accepts exactly one home")

    cases = []
    for name in selected:
        spec = variants[name]
        mode = spec["controller"]
        arguments = spec.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"variants.{name}.arguments must be an object")
        if home_ids is not None:
            arguments = {**arguments, "home_id": home_ids[0]}
        module = importlib.import_module(MODE_MODULES[mode])
        forced = FORCED_ARGUMENTS.get(mode, {})
        flags, effective = _runner_arguments(module, arguments, path.parent, output / name, forced)
        if mode in {"hvac_only", "water_heater_only"}:
            if effective.get("duration_hours") is None and effective.get("end_time") is None:
                raise ValueError(f"{name}: specify duration_hours or end_time")
            if mode == "hvac_only" and effective.get("duration_hours") is None:
                raise ValueError(f"{name}: HVAC requires duration_hours")
        cases.append(ControllerCase(name, mode, (str(Path(module.__file__).resolve()), *flags), effective))
    return ControllerSuite(path, requested, output, tuple(cases))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_controller_suite(suite: ControllerSuite, *, validate_only: bool = False) -> None:
    """Run existing implementations sequentially, preserving each runner's outputs."""
    if validate_only:
        return
    output = suite.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Controller suite output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "requested_controller_config.json", suite.requested)
    _write_json(output / "resolved_controller_config.json", suite.resolved())
    status: list[dict[str, str]] = []
    for case in suite.cases:
        entry = {"case": case.name, "controller": case.controller, "state": "running"}
        status.append(entry)
        _write_json(output / "run_status.json", status)
        log_path = output / "runner_logs" / f"{case.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running {case.name} ({case.controller}); log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *case.argv], cwd=Path(__file__).resolve().parents[3],
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            entry.update(state="failed", return_code=str(result.returncode), log=str(log_path))
            _write_json(output / "run_status.json", status)
            raise RuntimeError(f"{case.name} failed with exit code {result.returncode}; see {log_path}")
        entry["state"] = "complete"
        _write_json(output / "run_status.json", status)
        print(f"Completed {case.name}; results: {output / case.name}", flush=True)

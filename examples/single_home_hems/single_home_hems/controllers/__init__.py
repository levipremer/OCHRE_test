"""Individually testable predictive controller templates."""

from .base import DeviceController
from .composite import CompositeController
from .hvac import HVACPredictiveController


def __getattr__(name: str):
    """Keep optional controller families importable without loading them for HVAC-only runs."""
    modules = {
        "EVPredictiveController": "ev",
        "EVBaselineController": "ev",
        "EVMPCController": "ev",
        "EVScenarioController": "ev",
        "JointHEMSPredictiveController": "joint_hems",
        "WaterHeaterPredictiveController": "water_heater",
        "WholeHomeJointPredictiveController": "whole_home_joint",
    }
    if name not in modules:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "CompositeController",
    "DeviceController",
    "EVPredictiveController",
    "EVBaselineController",
    "EVMPCController",
    "EVScenarioController",
    "HVACPredictiveController",
    "JointHEMSPredictiveController",
    "WholeHomeJointPredictiveController",
    "WaterHeaterPredictiveController",
]

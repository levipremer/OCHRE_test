"""Modular five-home OCHRE neighborhood simulation."""

import os
import tempfile
from pathlib import Path

# OCHRE imports Matplotlib through its public package initializer. Point both
# Matplotlib and Fontconfig at a writable cache before importing OCHRE-backed
# modules so the example also runs cleanly in restricted/headless environments.
_cache_directory = Path(tempfile.gettempdir()) / "ochre_five_home_cache"
_cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_directory / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_directory))

from .config import HomeConfig, NeighborhoodConfig, load_configuration
from .simulation import NeighborhoodSimulation, SimulationSummary
from .transformer import DistributionTransformer

__all__ = [
    "HomeConfig",
    "DistributionTransformer",
    "NeighborhoodConfig",
    "NeighborhoodSimulation",
    "SimulationSummary",
    "load_configuration",
]

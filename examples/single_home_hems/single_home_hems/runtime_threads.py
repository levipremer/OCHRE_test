"""Optional runtime thread budgets without changing fitted forecast models."""
from __future__ import annotations

import os


def rf_threads(default):
    value = os.environ.get("HEMS_RF_THREADS")
    if value is None:
        return default
    try:
        threads = int(value)
    except ValueError:
        raise ValueError("HEMS_RF_THREADS must be a positive integer") from None
    if threads < 1:
        raise ValueError("HEMS_RF_THREADS must be a positive integer")
    return threads


def apply_rf_threads(estimator):
    threads = rf_threads(None)
    if threads is not None:
        estimator.set_params(n_jobs=threads)
    return estimator

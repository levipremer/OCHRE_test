"""Candidate estimators for the simulated Outside-Hub forecast comparison.

Every estimator is a direct, lead-conditioned point forecaster. The caller
supplies the same causal feature matrix and chronological folds to each one;
this module does not construct features, choose splits, or alter the established
conservative-RF/clock-p90 reference.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42


def candidate_specs() -> dict[str, list[dict[str, Any]]]:
    """Small validation grids, matched across 2/3/4-week training windows."""
    return {
        "ridge_calendar": [{"alpha": alpha} for alpha in (1.0, 100.0)],
        "ridge_history": [{"alpha": alpha} for alpha in (1.0, 100.0)],
        "rf_history": [{"min_samples_leaf": leaf} for leaf in (5, 15)],
        "hgb_history": [{"max_leaf_nodes": leaves} for leaves in (15, 31)],
        "lightgbm_history": [
            {"num_leaves": 15, "min_child_samples": 40},
            {"num_leaves": 31, "min_child_samples": 20},
        ],
        "xgboost_history": [
            {"max_depth": 4, "min_child_weight": 8},
            {"max_depth": 6, "min_child_weight": 4},
        ],
        "catboost_history": [
            {"depth": 4, "l2_leaf_reg": 5.0},
            {"depth": 6, "l2_leaf_reg": 3.0},
        ],
    }


def make_candidate(method: str, params: dict[str, Any]) -> Any:
    """Build an unfitted estimator with deterministic settings."""
    if method in ("ridge_calendar", "ridge_history"):
        return make_pipeline(StandardScaler(), Ridge(**params))
    if method == "rf_history":
        return RandomForestRegressor(
            n_estimators=80, max_features=0.8, max_samples=0.8,
            bootstrap=True, n_jobs=4, random_state=SEED, **params,
        )
    if method == "hgb_history":
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.06, l2_regularization=1.0,
            early_stopping=False, random_state=SEED, **params,
        )
    if method == "lightgbm_history":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=120, learning_rate=0.06, objective="regression",
            verbosity=-1, n_jobs=4, random_state=SEED, **params,
        )
    if method == "xgboost_history":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=120, learning_rate=0.06,
            objective="reg:squarederror", tree_method="hist",
            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
            random_state=SEED, **params,
        )
    if method == "catboost_history":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=120, learning_rate=0.06, loss_function="RMSE",
            verbose=False, thread_count=4, random_seed=SEED,
            allow_writing_files=False, **params,
        )
    raise ValueError(f"Unknown Outside-Hub candidate: {method}")

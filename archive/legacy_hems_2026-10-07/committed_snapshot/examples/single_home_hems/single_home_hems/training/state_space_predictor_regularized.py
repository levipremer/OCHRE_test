import numpy as np

from .state_space_predictor import (
    SEGMENT_COLUMN,
    _build_horizon_matrices,
    _right_matrix_divide,
    build_hankel,
)

AUTO_LAMBDA_MIN_RATIO = 1e-4
GROUP_LASSO_SCALING_METHODS = {
    "none",
    "standard_deviation",
    "min_max_except_rh",
}
LAMBDA_SELECTION_METRICS = {
    "validation_one_step_rmse",
    "validation_rolling_horizon_rmse",
    "validation_recursive_horizon_rmse",
}
CVXPY_SOLVERS = {"auto", "gurobi", "clarabel", "scs"}


def train_state_space_predictor_group_lasso(
    df_train,
    nu,
    ny,
    nw,
    horizon_steps,
    df_val=None,
    m_min=1,
    m_max=24,
    lambda_grid=None,
    group_tol=1e-6,
    relative_aicc_pct_thresh=0.05,
    max_group_lasso_iter=1000,
    group_lasso_tol=1e-6,
    group_lasso_solver="proximal_gradient",
    cvxpy_solver="auto",
    scale_group_lasso_features=True,
    auto_lambda_grid_points=30,
    group_lasso_scaling=None,
    min_max_passthrough_disturbance_indices=None,
    lambda_selection_metric="validation_one_step_rmse",
    memory_selection_method="validation",
    allow_rank_deficient_refit=False,
):
    """
    Train the state-space predictor with group-LASSO disturbance selection.

    Group LASSO is used only to select disturbances.  The returned model is the
    unregularized refit using the selected memory, lambda, and disturbance set.
    Set ``allow_rank_deficient_refit`` to use a minimum-norm least-squares
    refit and effective-rank information criteria when selected columns are
    exactly dependent.
    """
    _validate_settings(nu, ny, nw, horizon_steps, m_min, m_max)
    auto_lambda_grid = _is_auto_lambda_grid(lambda_grid)
    if auto_lambda_grid:
        lambda_grid = None
    else:
        lambda_grid = np.asarray(lambda_grid, dtype=float).reshape(-1)
        if lambda_grid.size == 0 or not np.all(np.isfinite(lambda_grid)):
            raise ValueError("lambda_grid must contain at least one finite value.")
        if np.any(lambda_grid < 0):
            raise ValueError("lambda_grid values must be nonnegative.")
    if not isinstance(auto_lambda_grid_points, int) or auto_lambda_grid_points <= 0:
        raise ValueError("auto_lambda_grid_points must be a positive integer.")
    if group_tol < 0:
        raise ValueError("group_tol must be nonnegative.")
    if not 0 <= relative_aicc_pct_thresh <= 1:
        raise ValueError("relative_aicc_pct_thresh must be between 0 and 1.")
    if group_lasso_solver not in {"proximal_gradient", "cvxpy"}:
        raise ValueError("group_lasso_solver must be 'proximal_gradient' or 'cvxpy'.")
    if cvxpy_solver not in CVXPY_SOLVERS:
        allowed = ", ".join(sorted(CVXPY_SOLVERS))
        raise ValueError(f"cvxpy_solver must be one of: {allowed}.")
    group_lasso_scaling = _resolve_group_lasso_scaling(
        group_lasso_scaling,
        scale_group_lasso_features,
    )
    min_max_passthrough_disturbance_indices = (
        _validate_passthrough_disturbance_indices(
            min_max_passthrough_disturbance_indices,
            nw,
        )
    )
    if lambda_selection_metric not in LAMBDA_SELECTION_METRICS:
        allowed = ", ".join(sorted(LAMBDA_SELECTION_METRICS))
        raise ValueError(f"lambda_selection_metric must be one of: {allowed}.")
    if memory_selection_method not in {"validation", "relative_aic"}:
        raise ValueError(
            "memory_selection_method must be 'validation' or 'relative_aic'."
        )

    segments = _extract_standard_segments(df_train, nu, ny, nw)
    validation_segments = (
        _extract_standard_segments(df_val, nu, ny, nw)
        if df_val is not None
        else None
    )
    fits = []
    memory_candidates = []
    aicc_values = []
    aic_values = []
    diagnostics_by_memory = {}
    validation_rmse_values = []

    for m in range(m_min, m_max + 1):
        try:
            fit = _fit_fixed_memory_group_lasso(
                segments=segments,
                m=m,
                h=horizon_steps,
                burn_in=m_max,
                lambda_grid=lambda_grid,
                group_tol=group_tol,
                max_iter=max_group_lasso_iter,
                tol=group_lasso_tol,
                solver=group_lasso_solver,
                cvxpy_solver=cvxpy_solver,
                scaling_method=group_lasso_scaling,
                min_max_passthrough_disturbance_indices=(
                    min_max_passthrough_disturbance_indices
                ),
                lambda_selection_metric=lambda_selection_metric,
                validation_segments=validation_segments,
                auto_lambda_grid_points=auto_lambda_grid_points,
                allow_rank_deficient_refit=allow_rank_deficient_refit,
            )
        except ValueError as exc:
            print(f"Skipping m={m}: {exc}")
            continue

        fits.append(fit)
        memory_candidates.append(m)
        aicc_values.append(fit["AICc"])
        aic_values.append(fit["AIC"])
        validation_rmse_values.append(fit["validation_rmse"])
        diagnostics_by_memory[m] = fit["lambda_diagnostics"]

    if not fits:
        raise ValueError(
            "No feasible regularized fit was found. Check data length, signal "
            "rank, lambda_grid, and missing values."
        )

    aicc_values = np.array(aicc_values, dtype=float)
    aic_values = np.array(aic_values, dtype=float)
    validation_rmse_values = np.array(validation_rmse_values, dtype=float)
    memory_candidates = np.array(memory_candidates, dtype=int)
    standard_best_idx = int(np.argmin(aicc_values))
    standard_aicc_m = int(memory_candidates[standard_best_idx])
    relative_aicc_m = relative_ic(
        aicc_values,
        memory_candidates,
        pct_thresh=relative_aicc_pct_thresh,
    )

    finite_validation = np.isfinite(validation_rmse_values)
    if memory_selection_method == "relative_aic":
        selected_idx = int(np.flatnonzero(memory_candidates == relative_aicc_m)[0])
        memory_selection_metric = "relative_AICc"
    elif np.any(finite_validation):
        selected_idx = int(np.nanargmin(validation_rmse_values))
        memory_selection_metric = lambda_selection_metric
    else:
        selected_idx = int(np.flatnonzero(memory_candidates == relative_aicc_m)[0])
        memory_selection_metric = "AICc"
    selected_memory_m = int(memory_candidates[selected_idx])
    best_fit = fits[selected_idx]

    return {
        "P": best_fit["P"],
        "F": best_fit["F"],
        "W": best_fit["W"],
        "eSig": best_fit["eSig"],
        "AIC": best_fit["AIC"],
        "AICc": best_fit["AICc"],
        "BIC": best_fit["BIC"],
        "A": best_fit["A"],
        "B": best_fit["B"],
        "E": best_fit["E"],
        "P1": best_fit["P1"],
        "F11": best_fit["F11"],
        "W11": best_fit["W11"],
        "m": selected_memory_m,
        "standard_aic_memory": standard_aicc_m,
        "relative_aic_memory": relative_aicc_m,
        "standard_aicc_memory": standard_aicc_m,
        "relative_aicc_memory": relative_aicc_m,
        "relative_aic_pct_thresh": relative_aicc_pct_thresh,
        "relative_aicc_pct_thresh": relative_aicc_pct_thresh,
        "nu": nu,
        "ny": ny,
        "nw": nw,
        "horizon_steps": horizon_steps,
        "training_burn_in": m_max,
        "aic_values": aic_values,
        "aicc_values": aicc_values,
        "validation_rmse_values": validation_rmse_values,
        "memory_candidates": memory_candidates,
        "lambda_grid": best_fit["lambda_grid"],
        "lambda_grid_mode": "auto" if auto_lambda_grid else "manual",
        "group_lasso_solver": group_lasso_solver,
        "cvxpy_solver": cvxpy_solver if group_lasso_solver == "cvxpy" else None,
        "scale_group_lasso_features": group_lasso_scaling != "none",
        "group_lasso_scaling": group_lasso_scaling,
        "min_max_passthrough_disturbance_indices": (
            min_max_passthrough_disturbance_indices
        ),
        "lambda_selection_metric": best_fit["lambda_selection_metric"],
        "memory_selection_metric": memory_selection_metric,
        "memory_selection_method": memory_selection_method,
        "allow_rank_deficient_refit": allow_rank_deficient_refit,
        "selected_memory_validation_rmse": best_fit["validation_rmse"],
        "selected_lambda_validation_rmse": best_fit["validation_rmse"],
        "selected_lambda": best_fit["lambda"],
        "selected_disturbance_indices": best_fit["active_disturbance_indices"],
        "selected_disturbance_names": best_fit["active_disturbance_names"],
        "feature_info": best_fit["feature_info"],
        "regularization_diagnostics": diagnostics_by_memory,
    }


def train_state_space_predictor_all_features(
    df_train,
    nu,
    ny,
    nw,
    horizon_steps,
    m_min=1,
    m_max=24,
    relative_aicc_pct_thresh=0.05,
    allow_rank_deficient_refit=False,
):
    """Fit every disturbance without scaling and select only the memory.

    This is the unregularized counterpart to group-LASSO selection. Every
    input, output history, and disturbance is retained in each candidate fit;
    relative AICc is the sole model-selection criterion.
    """
    _validate_settings(nu, ny, nw, horizon_steps, m_min, m_max)
    if not 0 <= relative_aicc_pct_thresh <= 1:
        raise ValueError("relative_aicc_pct_thresh must be between 0 and 1.")

    segments = _extract_standard_segments(df_train, nu, ny, nw)
    fits = []
    memory_candidates = []
    aic_values = []
    aicc_values = []

    for memory in range(m_min, m_max + 1):
        try:
            blocks = _build_training_blocks(
                segments,
                memory,
                horizon_steps,
                burn_in=m_max,
            )
            regression_matrix = np.vstack(
                [blocks["z1"], blocks["u1"], blocks["w1"]]
            )
            selected_rows = np.arange(regression_matrix.shape[0], dtype=int)
            theta = _right_matrix_divide(
                blocks["y1"],
                regression_matrix,
                require_full_rank=not allow_rank_deficient_refit,
            )
            effective_parameter_count = int(
                ny * np.linalg.matrix_rank(regression_matrix)
            )
            fit = _build_refit_model(
                theta_refit_selected=theta,
                selected_rows=selected_rows,
                blocks=blocks,
                m=memory,
                h=horizon_steps,
                nu=nu,
                ny=ny,
                nw=nw,
                effective_parameter_count=effective_parameter_count,
            )
        except ValueError as exc:
            print(f"Skipping m={memory}: {exc}")
            continue

        fits.append(fit)
        memory_candidates.append(memory)
        aic_values.append(fit["AIC"])
        aicc_values.append(fit["AICc"])

    if not fits:
        raise ValueError(
            "No feasible all-feature fit was found. Check data length, signal "
            "rank, and missing values."
        )

    memory_candidates = np.asarray(memory_candidates, dtype=int)
    aic_values = np.asarray(aic_values, dtype=float)
    aicc_values = np.asarray(aicc_values, dtype=float)
    standard_index = int(np.argmin(aicc_values))
    standard_memory = int(memory_candidates[standard_index])
    selected_memory = relative_ic(
        aicc_values,
        memory_candidates,
        pct_thresh=relative_aicc_pct_thresh,
    )
    selected_index = int(
        np.flatnonzero(memory_candidates == selected_memory)[0]
    )
    best_fit = fits[selected_index]
    selected_disturbances = list(range(nw))

    return {
        **best_fit,
        "m": selected_memory,
        "standard_aic_memory": standard_memory,
        "relative_aic_memory": selected_memory,
        "standard_aicc_memory": standard_memory,
        "relative_aicc_memory": selected_memory,
        "relative_aic_pct_thresh": relative_aicc_pct_thresh,
        "relative_aicc_pct_thresh": relative_aicc_pct_thresh,
        "nu": nu,
        "ny": ny,
        "nw": nw,
        "horizon_steps": horizon_steps,
        "training_burn_in": m_max,
        "aic_values": aic_values,
        "aicc_values": aicc_values,
        "memory_candidates": memory_candidates,
        "memory_selection_metric": "relative_AICc",
        "feature_selection_method": "none",
        "group_lasso_solver": None,
        "cvxpy_solver": None,
        "group_lasso_scaling": "none",
        "selected_lambda": None,
        "lambda_grid": np.array([], dtype=float),
        "selected_lambda_validation_rmse": None,
        "selected_disturbance_indices": selected_disturbances,
        "selected_disturbance_names": [
            f"Disturbance_{index + 1}" for index in selected_disturbances
        ],
        "regularization_diagnostics": {},
        "allow_rank_deficient_refit": allow_rank_deficient_refit,
    }


def state_space_predictor_regularized(*args, **kwargs):
    return train_state_space_predictor_group_lasso(*args, **kwargs)


def _is_auto_lambda_grid(lambda_grid):
    if lambda_grid is None:
        return True
    if isinstance(lambda_grid, str):
        if lambda_grid.lower() == "auto":
            return True
        raise ValueError("lambda_grid must be an array of values, None, or 'auto'.")
    return False


def _resolve_group_lasso_scaling(group_lasso_scaling, scale_group_lasso_features):
    if group_lasso_scaling is None:
        return "standard_deviation" if scale_group_lasso_features else "none"
    if group_lasso_scaling not in GROUP_LASSO_SCALING_METHODS:
        allowed = ", ".join(sorted(GROUP_LASSO_SCALING_METHODS))
        raise ValueError(f"group_lasso_scaling must be one of: {allowed}.")
    return group_lasso_scaling


def _validate_passthrough_disturbance_indices(indices, nw):
    if indices is None:
        return []

    validated = []
    for index in indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < nw:
            raise ValueError(
                "min_max_passthrough_disturbance_indices must contain zero-based "
                "disturbance indices between 0 and nw - 1."
            )
        validated.append(int(index))
    return sorted(set(validated))


def relative_ic(ic_values, memory_candidates, pct_thresh=0.01):
    """
    Select memory using a relative downward-drop rule.
    """
    ic_values = np.asarray(ic_values, dtype=float).reshape(-1)
    memory_candidates = np.asarray(memory_candidates, dtype=int).reshape(-1)

    if ic_values.size == 0 or ic_values.size != memory_candidates.size:
        raise ValueError("IC values and memory candidates must have the same nonzero length.")
    if not np.all(np.isfinite(ic_values)):
        raise ValueError("IC values must be finite.")
    if not 0 <= pct_thresh <= 1:
        raise ValueError("pct_thresh must be between 0 and 1.")

    standard_idx = int(np.argmin(ic_values))
    standard_m = int(memory_candidates[standard_idx])
    ic_differences = np.diff(ic_values)
    downward = ic_differences < 0

    if not np.any(downward):
        return standard_m

    max_drop = float(np.max(-ic_differences[downward]))
    relative_drop = np.zeros_like(ic_differences)
    relative_drop[downward] = -ic_differences[downward] / (
        max_drop + np.finfo(float).eps
    )

    flattening_indices = np.flatnonzero(downward & (relative_drop < pct_thresh))
    if flattening_indices.size == 0:
        return standard_m

    flattening_m = int(memory_candidates[flattening_indices[0] + 1])
    return min(standard_m, flattening_m)


def _validate_settings(nu, ny, nw, horizon_steps, m_min, m_max):
    for name, value in {
        "nu": nu,
        "ny": ny,
        "nw": nw,
        "horizon_steps": horizon_steps,
        "m_min": m_min,
        "m_max": m_max,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if m_min > m_max:
        raise ValueError("m_min must be less than or equal to m_max.")


def _extract_standard_segments(df, nu, ny, nw):
    input_cols = [f"Input_{i + 1}" for i in range(nu)]
    output_cols = [f"Output_{i + 1}" for i in range(ny)]
    disturbance_cols = [f"Disturbance_{i + 1}" for i in range(nw)]
    required_cols = input_cols + output_cols + disturbance_cols

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("Training dataframe is missing column(s): " + ", ".join(missing))

    values = df[required_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Training dataframe contains NaN or Inf values.")

    if SEGMENT_COLUMN in df.columns:
        segment_ids = df[SEGMENT_COLUMN].to_numpy()
    else:
        segment_ids = np.zeros(len(df), dtype=int)

    segments = []
    for start, stop in _segment_slices(segment_ids):
        segment = df.iloc[start:stop]
        segments.append(
            {
                "u": segment[input_cols].to_numpy(dtype=float).T,
                "y": segment[output_cols].to_numpy(dtype=float).T,
                "w": segment[disturbance_cols].to_numpy(dtype=float).T,
            }
        )

    return segments


def _segment_slices(segment_ids):
    if len(segment_ids) == 0:
        return

    start = 0
    for index in range(1, len(segment_ids)):
        if segment_ids[index] != segment_ids[index - 1]:
            yield start, index
            start = index
    yield start, len(segment_ids)


def _fit_fixed_memory_group_lasso(
    segments,
    m,
    h,
    burn_in,
    lambda_grid,
    group_tol,
    max_iter,
    tol,
    solver,
    cvxpy_solver,
    scaling_method,
    min_max_passthrough_disturbance_indices,
    lambda_selection_metric,
    validation_segments,
    auto_lambda_grid_points,
    allow_rank_deficient_refit,
):
    blocks = _build_training_blocks(segments, m, h, burn_in)
    nu = blocks["nu"]
    nw = blocks["nw"]
    ny = blocks["ny"]

    regression_matrix = np.vstack([blocks["z1"], blocks["u1"], blocks["w1"]])
    feature_info = _feature_info(m, nu, ny, nw)
    feature_offset, feature_scale = _feature_transform_for_group_lasso(
        segments,
        m=m,
        nu=nu,
        ny=ny,
        nw=nw,
        scaling_method=scaling_method,
        min_max_passthrough_disturbance_indices=(
            min_max_passthrough_disturbance_indices
        ),
    )
    if lambda_grid is None:
        lambda_grid = _automatic_lambda_grid(
            y=blocks["y1"],
            x=regression_matrix,
            feature_info=feature_info,
            feature_scale=feature_scale,
            feature_offset=feature_offset,
            n_points=auto_lambda_grid_points,
        )
    best_fit = None
    diagnostics = []
    cvxpy_path = None
    if solver == "cvxpy":
        cvxpy_path = _solve_group_lasso_cvxpy_path(
            y=blocks["y1"],
            x=regression_matrix,
            feature_info=feature_info,
            lambda_grid=lambda_grid,
            feature_scale=feature_scale,
            feature_offset=feature_offset,
            cvxpy_solver=cvxpy_solver,
        )

    for lambda_index, lambda_value in enumerate(lambda_grid):
        if cvxpy_path is None:
            theta_group = _solve_group_lasso(
                y=blocks["y1"],
                x=regression_matrix,
                feature_info=feature_info,
                lambda_value=float(lambda_value),
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                cvxpy_solver=cvxpy_solver,
                feature_scale=feature_scale,
                feature_offset=feature_offset,
            )
        else:
            theta_group = cvxpy_path[lambda_index]
        active_disturbances = _active_disturbances(
            theta_group,
            feature_info["disturbance_groups"],
            group_tol,
        )
        selected_rows = _selected_feature_rows(feature_info, active_disturbances)

        try:
            theta_refit_selected = _right_matrix_divide(
                blocks["y1"],
                regression_matrix[selected_rows, :],
                require_full_rank=not allow_rank_deficient_refit,
            )
            effective_parameter_count = int(
                ny * np.linalg.matrix_rank(regression_matrix[selected_rows, :])
            )
            fit = _build_refit_model(
                theta_refit_selected=theta_refit_selected,
                selected_rows=selected_rows,
                blocks=blocks,
                m=m,
                h=h,
                nu=nu,
                ny=ny,
                nw=nw,
                effective_parameter_count=effective_parameter_count,
            )
            validation_rmse = _validation_rmse_for_lambda_selection(
                validation_segments,
                fit,
                m,
                h,
                lambda_selection_metric,
            )
        except ValueError as exc:
            diagnostics.append(
                {
                    "lambda": float(lambda_value),
                    "active_disturbances": active_disturbances,
                    "status": f"skipped: {exc}",
                }
            )
            continue

        fit["lambda"] = float(lambda_value)
        fit["validation_rmse"] = validation_rmse
        fit["lambda_selection_metric"] = (
            lambda_selection_metric
            if np.isfinite(validation_rmse)
            else "AICc"
        )
        fit["active_disturbance_indices"] = active_disturbances
        fit["active_disturbance_names"] = [
            f"Disturbance_{index + 1}" for index in active_disturbances
        ]
        fit["feature_info"] = feature_info
        diagnostics.append(
            {
                "lambda": float(lambda_value),
                "AICc": fit["AICc"],
                "BIC": fit["BIC"],
                "validation_rmse": validation_rmse,
                "active_disturbances": active_disturbances,
                "active_disturbance_names": fit["active_disturbance_names"],
                "fit": fit,
                "status": "ok",
            }
        )

        if _is_better_lambda_fit(fit, best_fit):
            best_fit = fit

    if best_fit is None:
        raise ValueError("all lambda values produced infeasible refits")

    best_fit["lambda_diagnostics"] = diagnostics
    best_fit["lambda_grid"] = lambda_grid
    return best_fit


def _is_better_lambda_fit(candidate_fit, best_fit):
    if best_fit is None:
        return True

    candidate_validation = candidate_fit["validation_rmse"]
    best_validation = best_fit["validation_rmse"]
    if np.isfinite(candidate_validation) and np.isfinite(best_validation):
        return candidate_validation < best_validation
    if np.isfinite(candidate_validation) != np.isfinite(best_validation):
        return np.isfinite(candidate_validation)

    return candidate_fit["AICc"] < best_fit["AICc"]


def _validation_rmse_for_lambda_selection(
    validation_segments,
    fit,
    m,
    h,
    lambda_selection_metric,
):
    if lambda_selection_metric == "validation_one_step_rmse":
        return _validation_one_step_rmse(validation_segments, fit, m)
    if lambda_selection_metric in {
        "validation_rolling_horizon_rmse",
        "validation_recursive_horizon_rmse",
    }:
        return _validation_rolling_horizon_rmse(validation_segments, fit, m, h)
    allowed = ", ".join(sorted(LAMBDA_SELECTION_METRICS))
    raise ValueError(f"lambda_selection_metric must be one of: {allowed}.")


def _validation_one_step_rmse(validation_segments, fit, m):
    if validation_segments is None:
        return np.nan

    squared_error_sum = 0.0
    n_values = 0
    for segment in validation_segments:
        u = segment["u"]
        w = segment["w"]
        y = segment["y"]
        n_segment = u.shape[1] - m
        if n_segment <= 0:
            continue

        z = np.vstack([u, w, y])
        z_hist = build_hankel(z, 1, m, n_segment)
        u_current = build_hankel(u, m + 1, m + 1, n_segment)
        w_current = build_hankel(w, m + 1, m + 1, n_segment)
        y_current = build_hankel(y, m + 1, m + 1, n_segment)
        error = (
            y_current
            - fit["P1"] @ z_hist
            - fit["F11"] @ u_current
            - fit["W11"] @ w_current
        )
        finite_error = error[np.isfinite(error)]
        squared_error_sum += float(np.sum(finite_error**2))
        n_values += finite_error.size

    if n_values == 0:
        return np.nan
    return float(np.sqrt(squared_error_sum / n_values))


def _validation_rolling_horizon_rmse(validation_segments, fit, m, h):
    if validation_segments is None:
        return np.nan

    squared_error_sum = 0.0
    n_values = 0
    for segment in validation_segments:
        u = segment["u"]
        w = segment["w"]
        y = segment["y"]
        d = u.shape[1]
        if d < m + h:
            continue

        z = np.vstack([u, w, y])
        for origin in range(m, d - h + 1):
            state = z[:, origin - m:origin].reshape(-1, order="F")
            for t in range(origin, origin + h):
                prediction = (
                    fit["P1"] @ state
                    + fit["F11"] @ u[:, t]
                    + fit["W11"] @ w[:, t]
                )
                error = y[:, t] - prediction
                finite_error = error[np.isfinite(error)]
                squared_error_sum += float(np.sum(finite_error**2))
                n_values += finite_error.size
                state = (
                    fit["A"] @ state
                    + fit["B"] @ u[:, t]
                    + fit["E"] @ w[:, t]
                )

    if n_values == 0:
        return np.nan
    return float(np.sqrt(squared_error_sum / n_values))


def _validation_recursive_horizon_rmse(validation_segments, fit, m, h):
    """Backward-compatible name for the rolling recursive horizon metric."""
    return _validation_rolling_horizon_rmse(validation_segments, fit, m, h)


def _build_training_blocks(segments, m, h, burn_in):
    if not segments:
        raise ValueError("no training segments are available")

    nu = segments[0]["u"].shape[0]
    nw = segments[0]["w"].shape[0]
    ny = segments[0]["y"].shape[0]
    one_step_blocks = []
    horizon_blocks = []

    for segment in segments:
        u = segment["u"]
        y = segment["y"]
        w = segment["w"]
        d = u.shape[1]

        if not (u.shape[1] == w.shape[1] == y.shape[1]):
            raise ValueError("u, w, and y must have the same number of time steps")
        if not np.all(np.isfinite(np.concatenate([u.ravel(), w.ravel(), y.ravel()]))):
            raise ValueError("inputs contain NaN or Inf")

        n1_segment = d - burn_in
        if n1_segment > 0:
            z = np.vstack([u, w, y])
            one_step_blocks.append(
                (
                    build_hankel(z, burn_in - m + 1, burn_in, n1_segment),
                    build_hankel(u, burn_in + 1, burn_in + 1, n1_segment),
                    build_hankel(w, burn_in + 1, burn_in + 1, n1_segment),
                    build_hankel(y, burn_in + 1, burn_in + 1, n1_segment),
                )
            )

        n_segment = d - burn_in - h + 1
        if n_segment > 0:
            z = np.vstack([u, w, y])
            horizon_blocks.append(
                (
                    build_hankel(z, burn_in - m + 1, burn_in, n_segment),
                    build_hankel(u, burn_in + 1, burn_in + h, n_segment),
                    build_hankel(w, burn_in + 1, burn_in + h, n_segment),
                    build_hankel(y, burn_in + 1, burn_in + h, n_segment),
                )
            )

    if not one_step_blocks:
        raise ValueError("no segment is long enough for one-step regression")
    if sum(block[0].shape[1] for block in horizon_blocks) <= 1:
        raise ValueError("not enough within-segment samples for full-horizon covariance")

    return {
        "z1": np.hstack([block[0] for block in one_step_blocks]),
        "u1": np.hstack([block[1] for block in one_step_blocks]),
        "w1": np.hstack([block[2] for block in one_step_blocks]),
        "y1": np.hstack([block[3] for block in one_step_blocks]),
        "z_hist": np.hstack([block[0] for block in horizon_blocks]),
        "u_future": np.hstack([block[1] for block in horizon_blocks]),
        "w_future": np.hstack([block[2] for block in horizon_blocks]),
        "y_future": np.hstack([block[3] for block in horizon_blocks]),
        "nu": nu,
        "nw": nw,
        "ny": ny,
    }


def _feature_info(m, nu, ny, nw):
    nz = nu + nw + ny
    total_rows = m * nz + nu + nw
    disturbance_groups = {}
    grouped_rows = []

    for disturbance_index in range(nw):
        rows = []
        for lag in range(m):
            rows.append(lag * nz + nu + disturbance_index)
        rows.append(m * nz + nu + disturbance_index)
        disturbance_groups[f"Disturbance_{disturbance_index + 1}"] = rows
        grouped_rows.extend(rows)

    grouped_rows = set(grouped_rows)
    always_keep_idx = [row for row in range(total_rows) if row not in grouped_rows]
    return {
        "always_keep_idx": always_keep_idx,
        "disturbance_groups": disturbance_groups,
    }


def _solve_group_lasso(
    y,
    x,
    feature_info,
    lambda_value,
    max_iter,
    tol,
    solver,
    cvxpy_solver,
    feature_scale,
    feature_offset,
):
    if solver == "cvxpy":
        return _solve_group_lasso_cvxpy(
            y,
            x,
            feature_info,
            lambda_value,
            feature_scale,
            feature_offset,
            cvxpy_solver,
        )
    if solver != "proximal_gradient":
        raise ValueError("unknown group-LASSO solver")

    x_scaled, scale = _scale_features_for_group_lasso(
        x,
        feature_scale,
        feature_offset,
    )
    theta = np.zeros((y.shape[0], x_scaled.shape[0]))
    n_samples = x_scaled.shape[1]
    lipschitz = 2.0 * np.linalg.norm(x_scaled @ x_scaled.T, ord=2) / n_samples
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("group-LASSO design matrix has invalid Lipschitz constant")
    step = 1.0 / lipschitz

    for _ in range(max_iter):
        previous = theta.copy()
        residual = theta @ x_scaled - y
        gradient = 2.0 * residual @ x_scaled.T / n_samples
        theta -= step * gradient
        for group_rows in feature_info["disturbance_groups"].values():
            group = theta[:, group_rows]
            group_norm = np.linalg.norm(group, ord="fro")
            shrink = max(0.0, 1.0 - step * lambda_value / (group_norm + np.finfo(float).eps))
            theta[:, group_rows] = shrink * group

        delta = np.linalg.norm(theta - previous, ord="fro")
        baseline = max(1.0, np.linalg.norm(previous, ord="fro"))
        if delta / baseline < tol:
            break

    return theta / scale.reshape(1, -1)


def _automatic_lambda_grid(
    y,
    x,
    feature_info,
    feature_scale,
    feature_offset,
    n_points,
):
    x_scaled, _ = _scale_features_for_group_lasso(
        x,
        feature_scale,
        feature_offset,
    )
    n_samples = x_scaled.shape[1]
    always_keep_idx = feature_info["always_keep_idx"]

    if always_keep_idx:
        x_keep = x_scaled[always_keep_idx, :]
        theta_keep_t, *_ = np.linalg.lstsq(x_keep.T, y.T, rcond=None)
        residual = theta_keep_t.T @ x_keep - y
    else:
        residual = -y

    lambda_max = 0.0
    for group_rows in feature_info["disturbance_groups"].values():
        group_gradient = 2.0 * residual @ x_scaled[group_rows, :].T / n_samples
        lambda_max = max(lambda_max, float(np.linalg.norm(group_gradient, ord="fro")))

    if not np.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("automatic lambda calculation produced an invalid value")
    if lambda_max <= np.finfo(float).eps:
        return np.zeros(n_points, dtype=float)
    return np.geomspace(
        lambda_max,
        lambda_max * AUTO_LAMBDA_MIN_RATIO,
        n_points,
    )


def _solve_group_lasso_cvxpy(
    y,
    x,
    feature_info,
    lambda_value,
    feature_scale,
    feature_offset,
    cvxpy_solver,
):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ValueError(
            "group_lasso_solver='cvxpy' requires cvxpy to be installed in this "
            "Python environment"
        ) from exc

    x_scaled, scale = _scale_features_for_group_lasso(
        x,
        feature_scale,
        feature_offset,
    )
    n_samples = x_scaled.shape[1]
    theta = cp.Variable((y.shape[0], x_scaled.shape[0]))
    residual = theta @ x_scaled - y
    group_penalty = 0
    for group_rows in feature_info["disturbance_groups"].values():
        group_penalty += cp.norm(theta[:, group_rows], "fro")

    objective = cp.Minimize(
        cp.sum_squares(residual) / n_samples
        + float(lambda_value) * group_penalty
    )
    problem = cp.Problem(objective)
    solver = _resolve_cvxpy_solver(cp, cvxpy_solver)
    if solver == "GUROBI":
        problem.solve(solver=cp.GUROBI, OutputFlag=0)
    elif solver == "CLARABEL":
        problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=1000,
        )
    elif solver == "SCS":
        problem.solve(solver=cp.SCS)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"CVXPY group-LASSO solve failed with status {problem.status}")
    if theta.value is None or not np.all(np.isfinite(theta.value)):
        raise ValueError("CVXPY group-LASSO solution is not finite")

    return theta.value / scale.reshape(1, -1)


def _solve_group_lasso_cvxpy_path(
    y,
    x,
    feature_info,
    lambda_grid,
    feature_scale,
    feature_offset,
    cvxpy_solver,
):
    """Solve one lambda path from a single compiled CVXPY problem.

    The objective, scaling, tolerances, and solver match
    ``_solve_group_lasso_cvxpy``.  Only lambda is parameterized, allowing CVXPY
    to reuse canonicalization and warm-start consecutive points.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ValueError(
            "group_lasso_solver='cvxpy' requires cvxpy to be installed in this "
            "Python environment"
        ) from exc

    x_scaled, scale = _scale_features_for_group_lasso(
        x,
        feature_scale,
        feature_offset,
    )
    n_samples = x_scaled.shape[1]
    theta = cp.Variable((y.shape[0], x_scaled.shape[0]))
    lambda_parameter = cp.Parameter(nonneg=True)
    residual = theta @ x_scaled - y
    group_penalty = 0
    for group_rows in feature_info["disturbance_groups"].values():
        group_penalty += cp.norm(theta[:, group_rows], "fro")
    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(residual) / n_samples
            + lambda_parameter * group_penalty
        )
    )
    solver = _resolve_cvxpy_solver(cp, cvxpy_solver)
    solutions = []
    for lambda_value in np.asarray(lambda_grid, dtype=float):
        lambda_parameter.value = float(lambda_value)
        if solver == "GUROBI":
            problem.solve(
                solver=cp.GUROBI,
                OutputFlag=0,
                warm_start=True,
            )
        elif solver == "CLARABEL":
            problem.solve(
                solver=cp.CLARABEL,
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
                tol_feas=1e-10,
                max_iter=1000,
                warm_start=True,
            )
        elif solver == "SCS":
            problem.solve(solver=cp.SCS, warm_start=True)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(
                f"CVXPY group-LASSO solve failed with status {problem.status}"
            )
        if theta.value is None or not np.all(np.isfinite(theta.value)):
            raise ValueError("CVXPY group-LASSO solution is not finite")
        solutions.append(theta.value.copy() / scale.reshape(1, -1))
    return solutions


def _resolve_cvxpy_solver(cp, requested_solver):
    """Resolve an explicit or automatic CVXPY solver selection."""
    installed = set(cp.installed_solvers())
    if requested_solver == "auto":
        for candidate in ("GUROBI", "CLARABEL", "SCS"):
            if candidate in installed:
                return candidate
        raise ValueError("CVXPY requires GUROBI, CLARABEL, or SCS")

    solver = requested_solver.upper()
    if solver not in installed:
        raise ValueError(
            f"Requested CVXPY solver {solver} is not installed; "
            f"available solvers: {sorted(installed)}"
        )
    return solver


def _feature_transform_for_group_lasso(
    segments,
    m,
    nu,
    ny,
    nw,
    scaling_method,
    min_max_passthrough_disturbance_indices,
):
    n_features = m * (nu + nw + ny) + nu + nw
    if scaling_method == "none":
        return np.zeros(n_features, dtype=float), np.ones(n_features, dtype=float)

    if scaling_method == "standard_deviation":
        input_offset = np.zeros(nu, dtype=float)
        disturbance_offset = np.zeros(nw, dtype=float)
        output_offset = np.zeros(ny, dtype=float)
        input_scale = _series_scale_from_segments(segments, "u")
        disturbance_scale = _series_scale_from_segments(segments, "w")
        output_scale = _series_scale_from_segments(segments, "y")
    elif scaling_method == "min_max_except_rh":
        input_offset, input_scale = _series_min_range_from_segments(segments, "u")
        disturbance_offset, disturbance_scale = _series_min_range_from_segments(
            segments,
            "w",
        )
        output_offset, output_scale = _series_min_range_from_segments(segments, "y")
        for index in min_max_passthrough_disturbance_indices:
            disturbance_offset[index] = 0.0
            disturbance_scale[index] = 1.0
    else:
        allowed = ", ".join(sorted(GROUP_LASSO_SCALING_METHODS))
        raise ValueError(f"group_lasso_scaling must be one of: {allowed}.")

    if input_scale.size != nu or disturbance_scale.size != nw or output_scale.size != ny:
        raise ValueError("training segment dimensions do not match predictor settings")

    z_offset = np.concatenate([input_offset, disturbance_offset, output_offset])
    z_scale = np.concatenate([input_scale, disturbance_scale, output_scale])
    feature_offset = np.concatenate(
        [np.tile(z_offset, m), input_offset, disturbance_offset]
    )
    feature_scale = np.concatenate(
        [np.tile(z_scale, m), input_scale, disturbance_scale]
    )
    return feature_offset, feature_scale


def _series_scale_from_segments(segments, key):
    series = np.hstack([segment[key] for segment in segments])
    scale = np.std(series, axis=1)
    scale[scale <= np.finfo(float).eps] = 1.0
    return scale


def _series_min_range_from_segments(segments, key):
    series = np.hstack([segment[key] for segment in segments])
    offset = np.min(series, axis=1)
    scale = np.max(series, axis=1) - offset
    constant = scale <= np.finfo(float).eps
    offset[constant] = 0.0
    scale[constant] = 1.0
    return offset, scale


def _scale_features_for_group_lasso(x, feature_scale, feature_offset=None):
    scale = np.asarray(feature_scale, dtype=float).reshape(-1)
    if feature_offset is None:
        offset = np.zeros_like(scale)
    else:
        offset = np.asarray(feature_offset, dtype=float).reshape(-1)
    if scale.size != x.shape[0]:
        raise ValueError("feature scale length does not match regression matrix rows")
    if offset.size != x.shape[0]:
        raise ValueError("feature offset length does not match regression matrix rows")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("feature scale values must be finite and positive")
    if not np.all(np.isfinite(offset)):
        raise ValueError("feature offset values must be finite")

    return (x - offset.reshape(-1, 1)) / scale.reshape(-1, 1), scale


def _active_disturbances(theta, disturbance_groups, group_tol):
    active = []
    for disturbance_name, group_rows in disturbance_groups.items():
        group_norm = np.linalg.norm(theta[:, group_rows], ord="fro")
        if group_norm > group_tol:
            active.append(int(disturbance_name.split("_")[-1]) - 1)
    return active


def _selected_feature_rows(feature_info, active_disturbances):
    selected = list(feature_info["always_keep_idx"])
    for disturbance_index in active_disturbances:
        selected.extend(feature_info["disturbance_groups"][f"Disturbance_{disturbance_index + 1}"])
    return np.array(sorted(selected), dtype=int)


def _build_refit_model(
    theta_refit_selected,
    selected_rows,
    blocks,
    m,
    h,
    nu,
    ny,
    nw,
    effective_parameter_count=None,
):
    nz = nu + nw + ny
    n_features = m * nz + nu + nw
    theta_refit = np.zeros((ny, n_features))
    theta_refit[:, selected_rows] = theta_refit_selected

    p1 = theta_refit[:, : m * nz]
    f11 = theta_refit[:, m * nz : m * nz + nu]
    w11 = theta_refit[:, m * nz + nu : m * nz + nu + nw]

    a_mat = np.zeros((m * nz, m * nz))
    if m > 1:
        a_mat[: (m - 1) * nz, nz : m * nz] = np.eye((m - 1) * nz)
    y_rows = slice((m - 1) * nz + nu + nw, m * nz)
    a_mat[y_rows, :] = p1

    b_mat = np.zeros((m * nz, nu))
    b_mat[(m - 1) * nz : (m - 1) * nz + nu, :] = np.eye(nu)
    b_mat[y_rows, :] = f11

    e_mat = np.zeros((m * nz, nw))
    e_mat[(m - 1) * nz + nu : (m - 1) * nz + nu + nw, :] = np.eye(nw)
    e_mat[y_rows, :] = w11

    p_mat, f_mat, w_mat = _build_horizon_matrices(
        p1,
        f11,
        w11,
        a_mat,
        b_mat,
        e_mat,
        h,
    )

    error = (
        blocks["y_future"]
        - p_mat @ blocks["z_hist"]
        - f_mat @ blocks["u_future"]
        - w_mat @ blocks["w_future"]
    )
    error1 = blocks["y1"] - p1 @ blocks["z1"] - f11 @ blocks["u1"] - w11 @ blocks["w1"]
    if not np.all(np.isfinite(np.concatenate([error.ravel(), error1.ravel()]))):
        raise ValueError("prediction errors contain NaN or Inf")

    e_sig = np.cov(error)
    if np.ndim(e_sig) == 0:
        e_sig = np.array([[float(e_sig)]])

    rss = float(np.sum(error1**2))
    if rss <= 0:
        rss = np.finfo(float).tiny
    n_scalar = int(blocks["y1"].size)
    k = (
        int(theta_refit_selected.size)
        if effective_parameter_count is None
        else int(effective_parameter_count)
    )
    if n_scalar - k - 1 <= 0:
        raise ValueError("AICc is undefined because the refit has too many parameters")

    aic = n_scalar * np.log(rss / n_scalar) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n_scalar - k - 1)
    bic = n_scalar * np.log(rss / n_scalar) + k * np.log(n_scalar)
    if not np.all(np.isfinite([aic, aicc, bic])) or not np.all(np.isfinite(e_sig)):
        raise ValueError("AICc/BIC or covariance is not finite")

    return {
        "P": p_mat,
        "F": f_mat,
        "W": w_mat,
        "eSig": e_sig,
        "AIC": float(aic),
        "AICc": float(aicc),
        "BIC": float(bic),
        "A": a_mat,
        "B": b_mat,
        "E": e_mat,
        "P1": p1,
        "F11": f11,
        "W11": w11,
    }

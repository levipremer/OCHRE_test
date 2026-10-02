import numpy as np


SEGMENT_COLUMN = "_segment_id"


def state_space_predictor(
    df_train,
    nu,
    ny,
    nw,
    horizon_steps,
    m_min=1,
    m_max=24,
    relative_aic_pct_thresh=0.05,
    allow_rank_deficient=False,
):
    """
    Train the exogenous state-space predictor and select memory by relative AIC.

    When ``allow_rank_deficient`` is true, least squares returns the minimum-
    norm solution and information criteria use the effective design rank.

    """
    _validate_settings(nu, ny, nw, horizon_steps, m_min, m_max)
    if not 0 <= relative_aic_pct_thresh <= 1:
        raise ValueError("relative_aic_pct_thresh must be between 0 and 1.")
    segments = _extract_standard_segments(df_train, nu, ny, nw)

    fits = []
    memory_candidates = []
    aic_values = []

    for m in range(m_min, m_max + 1):
        try:
            fit = _fit_fixed_memory(
                segments,
                m,
                horizon_steps,
                burn_in=m_max,
                allow_rank_deficient=allow_rank_deficient,
            )
        except ValueError as exc:
            print(f"Skipping m={m}: {exc}")
            continue

        fits.append(fit)
        memory_candidates.append(m)
        aic_values.append(fit["AIC"])

    if not fits:
        raise ValueError(
            "No feasible memory value was found. Check data length, signal rank, and missing values."
        )

    aic_values = np.array(aic_values, dtype=float)
    memory_candidates = np.array(memory_candidates, dtype=int)

    standard_best_idx = int(np.nanargmin(aic_values))
    standard_aic_m = int(memory_candidates[standard_best_idx])
    relative_aic_m = relative_aic(
        aic_values,
        memory_candidates,
        pct_thresh=relative_aic_pct_thresh,
    )

    selected_idx = int(np.flatnonzero(memory_candidates == relative_aic_m)[0])
    best_fit = fits[selected_idx]

    model = {
        "P": best_fit["P"],
        "F": best_fit["F"],
        "W": best_fit["W"],
        "eSig": best_fit["eSig"],
        "AIC": best_fit["AIC"],
        "BIC": best_fit["BIC"],
        "A": best_fit["A"],
        "B": best_fit["B"],
        "E": best_fit["E"],
        "P1": best_fit["P1"],
        "F11": best_fit["F11"],
        "W11": best_fit["W11"],
        "m": relative_aic_m,
        "standard_aic_memory": standard_aic_m,
        "relative_aic_memory": relative_aic_m,
        "relative_aic_pct_thresh": relative_aic_pct_thresh,
        "nu": nu,
        "ny": ny,
        "nw": nw,
        "horizon_steps": horizon_steps,
        "training_burn_in": m_max,
        "allow_rank_deficient": allow_rank_deficient,
        "aic_values": aic_values,
        "memory_candidates": memory_candidates,
    }
    return model


def relative_aic(aic_values, memory_candidates, pct_thresh=0.01):
    """
    Select memory using the relative downward-drop rule from relativeAIC.m.
    """
    aic_values = np.asarray(aic_values, dtype=float).reshape(-1)
    memory_candidates = np.asarray(memory_candidates, dtype=int).reshape(-1)

    if aic_values.size == 0 or aic_values.size != memory_candidates.size:
        raise ValueError("AIC values and memory candidates must have the same nonzero length.")
    if not np.all(np.isfinite(aic_values)):
        raise ValueError("AIC values must be finite.")
    if not 0 <= pct_thresh <= 1:
        raise ValueError("pct_thresh must be between 0 and 1.")

    standard_idx = int(np.argmin(aic_values))
    standard_m = int(memory_candidates[standard_idx])
    aic_differences = np.diff(aic_values)
    downward = aic_differences < 0

    if not np.any(downward):
        return standard_m

    max_drop = float(np.max(-aic_differences[downward]))
    relative_drop = np.zeros_like(aic_differences)
    relative_drop[downward] = -aic_differences[downward] / (
        max_drop + np.finfo(float).eps
    )

    flattening_indices = np.flatnonzero(downward & (relative_drop < pct_thresh))
    if flattening_indices.size == 0:
        return standard_m

    flattening_m = int(memory_candidates[flattening_indices[0]])
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


def _fit_fixed_memory(segments, m, h, burn_in, allow_rank_deficient=False):
    if not segments:
        raise ValueError("no training segments are available")

    nu = segments[0]["u"].shape[0]
    nw = segments[0]["w"].shape[0]
    ny = segments[0]["y"].shape[0]
    nz = nu + nw + ny
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

    z1 = np.hstack([block[0] for block in one_step_blocks])
    u1 = np.hstack([block[1] for block in one_step_blocks])
    w1 = np.hstack([block[2] for block in one_step_blocks])
    y1 = np.hstack([block[3] for block in one_step_blocks])
    n1 = z1.shape[1]

    regression_matrix = np.vstack([z1, u1, w1])
    if not allow_rank_deficient:
        _check_rank(regression_matrix, "one-step regression matrix")

    theta = _right_matrix_divide(
        y1,
        regression_matrix,
        require_full_rank=not allow_rank_deficient,
    )
    if not np.all(np.isfinite(theta)):
        raise ValueError("one-step regression produced NaN or Inf")

    p1 = theta[:, : m * nz]
    f11 = theta[:, m * nz : m * nz + nu]
    w11 = theta[:, m * nz + nu : m * nz + nu + nw]

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

    p_mat, f_mat, w_mat = _build_horizon_matrices(p1, f11, w11, a_mat, b_mat,
                                                  e_mat, h)

    z_hist = np.hstack([block[0] for block in horizon_blocks])
    u_future = np.hstack([block[1] for block in horizon_blocks])
    w_future = np.hstack([block[2] for block in horizon_blocks])
    y_future = np.hstack([block[3] for block in horizon_blocks])

    error = y_future - p_mat @ z_hist - f_mat @ u_future - w_mat @ w_future
    error1 = y1 - p1 @ z1 - f11 @ u1 - w11 @ w1
    if not np.all(np.isfinite(np.concatenate([error.ravel(), error1.ravel()]))):
        raise ValueError("prediction errors contain NaN or Inf")

    e_sig = np.cov(error)
    if np.ndim(e_sig) == 0:
        e_sig = np.array([[float(e_sig)]])

    sse1 = np.sum(error1**2, axis=1)
    if np.any(sse1 <= 0):
        raise ValueError("one-step prediction error is zero or negative for AIC calculation")

    if allow_rank_deficient:
        k = ny * np.linalg.matrix_rank(regression_matrix)
    else:
        k = ny * (m * nz + nu + nw)
    aic = 2 * k + float(np.mean(n1 * np.log(sse1 / n1)))
    bic = k * np.log(n1) + float(np.mean(n1 * np.log(sse1 / n1)))
    if not np.all(np.isfinite([aic, bic])) or not np.all(np.isfinite(e_sig)):
        raise ValueError("AIC/BIC or covariance is not finite")

    return {
        "P": p_mat,
        "F": f_mat,
        "W": w_mat,
        "eSig": e_sig,
        "AIC": aic,
        "BIC": bic,
        "A": a_mat,
        "B": b_mat,
        "E": e_mat,
        "P1": p1,
        "F11": f11,
        "W11": w11,
    }


def build_hankel(x, start, stop, width):
    """
    MATLAB-style block Hankel matrix for samples in [start, stop], 1-based.
    """
    if start < 1 or stop < start or width < 1:
        raise ValueError("Invalid Hankel start, stop, or width.")

    n_signals, n_samples = x.shape
    block_count = stop - start + 1
    last_index = stop + width - 2
    if last_index >= n_samples:
        raise ValueError("Hankel request exceeds available samples.")

    hankel = np.zeros((block_count * n_signals, width))
    for col in range(width):
        window = x[:, start - 1 + col : stop + col]
        hankel[:, col] = window.reshape(-1, order="F")
    return hankel


def history_vector(u, w, y, start_index, m):
    z = np.vstack(
        [
            u[:, start_index : start_index + m],
            w[:, start_index : start_index + m],
            y[:, start_index : start_index + m],
        ]
    )
    return z.reshape(-1, order="F")


def _right_matrix_divide(left, right, require_full_rank=True):
    solution_t, residuals, rank, singular_values = np.linalg.lstsq(right.T, left.T, rcond=None)
    if require_full_rank and rank < min(right.T.shape):
        raise ValueError("least-squares system is rank deficient")
    if (
        require_full_rank
        and singular_values.size
        and singular_values[-1] <= np.finfo(float).eps * singular_values[0]
    ):
        raise ValueError("least-squares system is numerically singular")
    return solution_t.T


def _check_rank(matrix, name):
    rank = np.linalg.matrix_rank(matrix)
    target_rank = min(matrix.shape)
    if rank < target_rank:
        raise ValueError(f"{name} is rank deficient ({rank} < {target_rank})")


def _build_horizon_matrices(p1, f11, w11, a_mat, b_mat, e_mat, h):
    ny = p1.shape[0]
    nu = f11.shape[1]
    nw = w11.shape[1]

    p_mat = np.zeros((h * ny, a_mat.shape[1]))
    f_mat = np.zeros((h * ny, h * nu))
    w_mat = np.zeros((h * ny, h * nw))

    powers = [np.eye(a_mat.shape[0])]
    for _ in range(1, h):
        powers.append(powers[-1] @ a_mat)

    for i in range(1, h + 1):
        y_rows = slice((i - 1) * ny, i * ny)
        p_mat[y_rows, :] = p1 @ powers[i - 1]
        f_mat[y_rows, (i - 1) * nu : i * nu] = f11
        w_mat[y_rows, (i - 1) * nw : i * nw] = w11

        for j in range(1, i):
            f_cols = slice((j - 1) * nu, j * nu)
            w_cols = slice((j - 1) * nw, j * nw)
            a_power = powers[i - j - 1]
            f_mat[y_rows, f_cols] = p1 @ a_power @ b_mat
            w_mat[y_rows, w_cols] = p1 @ a_power @ e_mat

    if not np.all(np.isfinite(np.concatenate([p_mat.ravel(), f_mat.ravel(),
                                              w_mat.ravel()]))):
        raise ValueError("predictor matrices contain NaN or Inf")

    return p_mat, f_mat, w_mat

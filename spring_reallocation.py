import warnings

import numpy as np

from geometry import build_hicks_henne_basis
from settings import SETTINGS

try:
    from scipy.optimize import lsq_linear as _scipy_lsq_linear
except Exception:
    _scipy_lsq_linear = None


def _as_sorted_1d(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.all(np.diff(arr) >= 0.0):
        arr = np.sort(arr)
    return arr


def _finite_or_zero(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _min_spacing(centers):
    centers = _as_sorted_1d(centers)
    if centers.size <= 1:
        return float("inf")
    return float(np.min(np.diff(centers)))


def normalize_importance(weights, eps=1e-12):
    weights = np.abs(_finite_or_zero(weights))
    if weights.size == 0:
        return weights

    w_min = float(np.min(weights))
    w_max = float(np.max(weights))
    span = w_max - w_min

    if not np.isfinite(span) or span < float(eps):
        return np.zeros_like(weights, dtype=float)

    out = (weights - w_min) / span
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, 1.0)


def spring_transfer_function(f, mode="sin", power=1.0):
    f_arr = np.asarray(f, dtype=float)
    f_clamped = np.clip(np.where(np.isfinite(f_arr), f_arr, 0.0), 0.0, 1.0)

    mode = str(mode).strip().lower()
    if mode == "linear":
        out = f_clamped
    elif mode == "power":
        out = f_clamped ** float(power)
    elif mode == "sin":
        out = np.minimum(1.0, 2.0 * np.sin(f_clamped))
    else:
        raise ValueError(
            f"Unknown spring transfer_mode={mode!r}. Allowed values: linear, power, sin."
        )

    out = np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, 1.0)
    if np.isscalar(f):
        return float(out)
    return out


def solve_spring_1d(
    centers,
    importance,
    A=10.0,
    fix_ends=True,
    transfer_mode="sin",
    transfer_power=1.0,
):
    centers = _as_sorted_1d(centers)
    importance = _finite_or_zero(importance)

    if centers.size != importance.size:
        raise ValueError("centers and importance must have the same size.")
    if centers.size <= 2:
        normalized = normalize_importance(importance)
        return centers.copy(), {
            "normalized_importance": normalized,
            "stiffness": np.zeros(max(0, centers.size - 1), dtype=float),
            "spring_centers": centers.copy(),
            "min_spacing_before": _min_spacing(centers),
            "min_spacing_spring": _min_spacing(centers),
        }

    if float(A) <= 0.0:
        raise ValueError("A must be strictly positive.")

    if not bool(fix_ends):
        warnings.warn(
            "solve_spring_1d currently supports only fix_ends=True. "
            "Proceeding with fixed endpoints.",
            RuntimeWarning,
        )
        fix_ends = True

    normalized = normalize_importance(importance)
    face_importance = 0.5 * (normalized[:-1] + normalized[1:])
    transfer = spring_transfer_function(
        face_importance,
        mode=transfer_mode,
        power=transfer_power,
    )
    stiffness = 1.0 + (float(A) - 1.0) * transfer

    n = centers.size
    n_internal = n - 2
    mat = np.zeros((n_internal, n_internal), dtype=float)
    rhs = np.zeros(n_internal, dtype=float)

    for row, i in enumerate(range(1, n - 1)):
        k_left = float(stiffness[i - 1])
        k_right = float(stiffness[i])

        if row > 0:
            mat[row, row - 1] = -k_left
        else:
            rhs[row] += k_left * float(centers[0])

        mat[row, row] = k_left + k_right

        if row < n_internal - 1:
            mat[row, row + 1] = -k_right
        else:
            rhs[row] += k_right * float(centers[-1])

    try:
        internal = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Spring system is singular; unable to solve 1D spring equilibrium.") from exc

    spring_centers = centers.copy()
    spring_centers[1:-1] = internal
    spring_centers[0] = centers[0]
    spring_centers[-1] = centers[-1]

    return spring_centers, {
        "normalized_importance": normalized,
        "stiffness": stiffness,
        "spring_centers": spring_centers.copy(),
        "min_spacing_before": _min_spacing(centers),
        "min_spacing_spring": _min_spacing(spring_centers),
    }


def enforce_min_spacing(
    centers,
    min_spacing,
    fixed_first=True,
    fixed_last=True,
):
    centers = _as_sorted_1d(centers)
    min_spacing = float(min_spacing)

    if centers.size <= 1 or min_spacing <= 0.0:
        return centers.copy()

    x = centers.copy()
    first = float(centers[0])
    last = float(centers[-1])
    n = centers.size

    if fixed_first and fixed_last:
        if (last - first) < min_spacing * (n - 1) - 1.0e-14:
            raise ValueError(
                "Infeasible min_spacing: required spacing exceeds the available span "
                "between fixed endpoints."
            )

    if fixed_first:
        x[0] = first
        i_stop = n if not fixed_last else n - 1
        for i in range(1, i_stop):
            x[i] = max(x[i], x[i - 1] + min_spacing)
    else:
        for i in range(1, n):
            x[i] = max(x[i], x[i - 1] + min_spacing)

    if fixed_last:
        x[-1] = last
        j_start = 0 if not fixed_first else 1
        for i in range(n - 2, j_start - 1, -1):
            x[i] = min(x[i], x[i + 1] - min_spacing)
    elif fixed_first:
        for i in range(n - 2, -1, -1):
            x[i] = min(x[i], x[i + 1] - min_spacing)

    if fixed_first:
        x[0] = first
    if fixed_last:
        x[-1] = last

    if np.any(np.diff(x) < min_spacing - 1.0e-12):
        raise ValueError("Unable to enforce the requested min_spacing while preserving center ordering.")

    return x


def spring_reallocate_1d(
    centers,
    importance,
    A=10.0,
    omega=0.25,
    max_dx=0.05,
    min_spacing=0.02,
    fix_ends=True,
    transfer_mode="sin",
    transfer_power=1.0,
):
    centers_old = _as_sorted_1d(centers)
    importance = _finite_or_zero(importance)

    if centers_old.size != importance.size:
        raise ValueError("centers and importance must have the same size.")

    spring_centers, spring_diag = solve_spring_1d(
        centers=centers_old,
        importance=importance,
        A=A,
        fix_ends=fix_ends,
        transfer_mode=transfer_mode,
        transfer_power=transfer_power,
    )

    blended = (1.0 - float(omega)) * centers_old + float(omega) * spring_centers
    dx = blended - centers_old
    dx = np.clip(dx, -float(max_dx), float(max_dx))

    new_centers = centers_old + dx
    if fix_ends and new_centers.size > 0:
        new_centers[0] = centers_old[0]
        new_centers[-1] = centers_old[-1]

    new_centers = enforce_min_spacing(
        new_centers,
        min_spacing=min_spacing,
        fixed_first=fix_ends,
        fixed_last=fix_ends,
    )

    if fix_ends and new_centers.size > 0:
        new_centers[0] = centers_old[0]
        new_centers[-1] = centers_old[-1]

    displacement = new_centers - centers_old
    diagnostics = {
        "old_centers": centers_old.copy(),
        "spring_centers": spring_centers.copy(),
        "new_centers": new_centers.copy(),
        "importance": np.abs(importance).copy(),
        "normalized_importance": np.asarray(spring_diag["normalized_importance"], dtype=float).copy(),
        "stiffness": np.asarray(spring_diag["stiffness"], dtype=float).copy(),
        "displacement": displacement.copy(),
        "max_abs_dx": float(np.max(np.abs(displacement))) if displacement.size > 0 else 0.0,
        "min_spacing_before": _min_spacing(centers_old),
        "min_spacing_after": _min_spacing(new_centers),
    }
    return new_centers, diagnostics


def build_hh_basis_matrix(x, centers):
    x = np.asarray(x, dtype=float).reshape(-1)
    centers = _as_sorted_1d(centers)

    if centers.size == 0:
        return np.zeros((x.size, 0), dtype=float)

    hh_power = SETTINGS["optimization"]["hh_power"]
    basis_rows = build_hicks_henne_basis(x, centers, power=hh_power)
    return np.asarray(basis_rows, dtype=float).T


def _parse_bounds(bounds, n_coeffs):
    if bounds is None:
        return None, None

    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]) and np.isscalar(bounds[1]):
        lb = np.full(n_coeffs, float(bounds[0]), dtype=float)
        ub = np.full(n_coeffs, float(bounds[1]), dtype=float)
        return lb, ub

    if len(bounds) != n_coeffs:
        raise ValueError("bounds must be a single (lb, ub) pair or have one pair per coefficient.")

    lb = np.array([float(pair[0]) for pair in bounds], dtype=float)
    ub = np.array([float(pair[1]) for pair in bounds], dtype=float)
    return lb, ub


def reproject_coefficients(
    x,
    old_centers,
    new_centers,
    old_a,
    ridge=1e-10,
    weights=None,
    bounds=None,
):
    x = np.asarray(x, dtype=float).reshape(-1)
    old_centers = _as_sorted_1d(old_centers)
    new_centers = _as_sorted_1d(new_centers)
    old_a = np.asarray(old_a, dtype=float).reshape(-1)

    if old_centers.size != old_a.size:
        raise ValueError("old_centers and old_a must have the same size.")

    n_coeffs = new_centers.size
    if n_coeffs == 0:
        return np.zeros(0, dtype=float), {
            "projection_error_l2": 0.0,
            "projection_error_linf": 0.0,
            "relative_projection_error_l2": 0.0,
            "old_deformation_norm_l2": 0.0,
            "condition_number": 1.0,
            "max_abs_a_new": 0.0,
            "n_coeffs_at_bound": 0,
            "solution_mode": "empty",
        }

    phi_old = build_hh_basis_matrix(x, old_centers)
    phi_new = build_hh_basis_matrix(x, new_centers)
    delta_old = phi_old @ old_a

    if weights is None:
        weighted_phi = phi_new
        weighted_rhs = delta_old
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != x.size:
            raise ValueError("weights must have the same length as x.")
        sqrt_w = np.sqrt(np.clip(np.where(np.isfinite(weights), weights, 0.0), 0.0, None))
        weighted_phi = sqrt_w[:, None] * phi_new
        weighted_rhs = sqrt_w * delta_old

    ridge = float(ridge)
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")

    if ridge > 0.0:
        aug_matrix = np.vstack([weighted_phi, np.sqrt(ridge) * np.eye(n_coeffs)])
        aug_rhs = np.concatenate([weighted_rhs, np.zeros(n_coeffs, dtype=float)])
    else:
        aug_matrix = weighted_phi
        aug_rhs = weighted_rhs

    lb, ub = _parse_bounds(bounds, n_coeffs)
    solution_mode = "lstsq"

    if lb is not None and ub is not None:
        if _scipy_lsq_linear is not None:
            res = _scipy_lsq_linear(aug_matrix, aug_rhs, bounds=(lb, ub), lsmr_tol="auto")
            if not res.success:
                raise RuntimeError(f"Bounded reprojection failed: {res.message}")
            a_new = np.asarray(res.x, dtype=float)
            solution_mode = "bounded_lsq_linear"
        else:
            warnings.warn(
                "scipy.optimize.lsq_linear is not available; falling back to unbounded reprojection "
                "followed by coefficient clipping.",
                RuntimeWarning,
            )
            a_new, *_ = np.linalg.lstsq(aug_matrix, aug_rhs, rcond=None)
            a_new = np.clip(np.asarray(a_new, dtype=float), lb, ub)
            solution_mode = "lstsq_plus_clip"
    else:
        a_new, *_ = np.linalg.lstsq(aug_matrix, aug_rhs, rcond=None)
        a_new = np.asarray(a_new, dtype=float)

    delta_new = phi_new @ a_new
    diff = delta_new - delta_old
    old_norm_l2 = float(np.linalg.norm(delta_old))
    err_l2 = float(np.linalg.norm(diff))
    err_linf = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    cond_number = float(np.linalg.cond(aug_matrix)) if aug_matrix.size > 0 else 1.0

    n_at_bound = 0
    if lb is not None and ub is not None:
        hit_lower = np.isclose(a_new, lb, atol=1.0e-10, rtol=0.0)
        hit_upper = np.isclose(a_new, ub, atol=1.0e-10, rtol=0.0)
        n_at_bound = int(np.count_nonzero(hit_lower | hit_upper))

    diagnostics = {
        "projection_error_l2": err_l2,
        "projection_error_linf": err_linf,
        "relative_projection_error_l2": err_l2 / max(old_norm_l2, 1.0e-16),
        "old_deformation_norm_l2": old_norm_l2,
        "condition_number": cond_number,
        "max_abs_a_new": float(np.max(np.abs(a_new))) if a_new.size > 0 else 0.0,
        "n_coeffs_at_bound": n_at_bound,
        "solution_mode": solution_mode,
    }
    return a_new, diagnostics


def split_a_by_sides(a, n_upper, n_lower):
    a = np.asarray(a, dtype=float).reshape(-1)
    n_upper = int(n_upper)
    n_lower = int(n_lower)

    if a.size != n_upper + n_lower:
        raise ValueError("Expected len(a) = n_upper + n_lower.")

    a_upper = np.asarray(a[:n_upper], dtype=float)
    a_lower = np.asarray(a[n_upper:n_upper + n_lower], dtype=float)
    return a_upper, a_lower


def compute_zero_out_loss_weights(*args, **kwargs):
    raise NotImplementedError(
        "weight_mode='zero_out_loss' is not implemented yet. "
        "Use weight_mode='abs_a' for the standalone spring reallocation experiment."
    )


def reallocate_airfoil_centers(
    x,
    upper_centers,
    lower_centers,
    a_opt,
    weight_mode="abs_a",
    A=10.0,
    omega=0.25,
    max_dx=0.05,
    min_spacing=0.02,
    ridge=1e-10,
    fix_ends=True,
    transfer_mode="sin",
    transfer_power=1.0,
    coeff_bounds=None,
):
    upper_centers = _as_sorted_1d(upper_centers)
    lower_centers = _as_sorted_1d(lower_centers)
    a_opt = np.asarray(a_opt, dtype=float).reshape(-1)

    a_upper, a_lower = split_a_by_sides(a_opt, upper_centers.size, lower_centers.size)

    weight_mode = str(weight_mode).strip().lower()
    if weight_mode == "abs_a":
        importance_upper = np.abs(a_upper)
        importance_lower = np.abs(a_lower)
    elif weight_mode == "zero_out_loss":
        raise NotImplementedError(
            "weight_mode='zero_out_loss' is planned but not implemented yet. "
            "Use weight_mode='abs_a' for now."
        )
    else:
        raise ValueError(
            f"Unknown spring weight_mode={weight_mode!r}. Allowed values: abs_a, zero_out_loss."
        )

    new_upper_centers, upper_spring = spring_reallocate_1d(
        centers=upper_centers,
        importance=importance_upper,
        A=A,
        omega=omega,
        max_dx=max_dx,
        min_spacing=min_spacing,
        fix_ends=fix_ends,
        transfer_mode=transfer_mode,
        transfer_power=transfer_power,
    )
    new_lower_centers, lower_spring = spring_reallocate_1d(
        centers=lower_centers,
        importance=importance_lower,
        A=A,
        omega=omega,
        max_dx=max_dx,
        min_spacing=min_spacing,
        fix_ends=fix_ends,
        transfer_mode=transfer_mode,
        transfer_power=transfer_power,
    )

    a_upper_new, upper_projection = reproject_coefficients(
        x=x,
        old_centers=upper_centers,
        new_centers=new_upper_centers,
        old_a=a_upper,
        ridge=ridge,
        weights=None,
        bounds=coeff_bounds,
    )
    a_lower_new, lower_projection = reproject_coefficients(
        x=x,
        old_centers=lower_centers,
        new_centers=new_lower_centers,
        old_a=a_lower,
        ridge=ridge,
        weights=None,
        bounds=coeff_bounds,
    )

    a0_new = np.concatenate([a_upper_new, a_lower_new])
    diagnostics = {
        "upper_spring": upper_spring,
        "lower_spring": lower_spring,
        "upper_projection": upper_projection,
        "lower_projection": lower_projection,
        "importance_upper": np.asarray(importance_upper, dtype=float),
        "importance_lower": np.asarray(importance_lower, dtype=float),
    }

    return {
        "new_upper_centers": new_upper_centers,
        "new_lower_centers": new_lower_centers,
        "a0_new": a0_new,
        "diagnostics": diagnostics,
    }

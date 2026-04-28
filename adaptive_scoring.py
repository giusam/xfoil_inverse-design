from pathlib import Path

import numpy as np

from geometry import apply_hicks_henne_deformation, build_normal_peak_fd_steps
from settings import SETTINGS
from objective import make_objective
from adaptive_candidate import (
    _build_extended_space,
    _evaluate_objective_state,
    _make_candidate_objective,
    _rebuild_geometry_from_a,
)
from adaptive_fd import (
    _compute_full_aero_gradients,
    _compute_full_objective_gradient,
)
from adaptive_constraints import (
    _build_active_ikkt_system,
    _compute_geometric_gradients_analytic,
    _solve_bounded_least_squares,
)
from adaptive_pred import _score_candidate_pred


def _format_candidate_context(candidate):
    parts = []

    interval_label = candidate.get("interval_label")
    if interval_label is None and "interval_id" in candidate:
        interval_label = candidate["interval_id"]
    if interval_label is not None:
        parts.append(f"interval={interval_label}")

    if "local_fraction" in candidate:
        parts.append(f"fraction={float(candidate['local_fraction']):.2f}")

    if len(parts) == 0:
        return ""

    return "  " + "  ".join(parts)


def trapezoid_weights(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("trapezoid_weights expects a 1D x array")
    if len(x) == 0:
        return np.zeros(0, dtype=float)
    if len(x) == 1:
        return np.ones(1, dtype=float)

    dx = np.diff(x)
    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * abs(dx[0])
    w[-1] = 0.5 * abs(dx[-1])
    if len(x) > 2:
        w[1:-1] = 0.5 * (np.abs(dx[:-1]) + np.abs(dx[1:]))
    return np.maximum(w, 0.0)


def _single_hicks_henne_bump_from_deformation(x, side, center, hh_power):
    zero = np.zeros_like(np.asarray(x, dtype=float))
    side_u = str(side).upper()

    if side_u == "UPPER":
        yu, _ = apply_hicks_henne_deformation(
            x=x,
            yu_base=zero,
            yl_base=zero,
            a_upper=[1.0],
            a_lower=[],
            upper_centers=[center],
            lower_centers=[],
            power=hh_power,
        )
        return np.asarray(yu, dtype=float)

    if side_u == "LOWER":
        _, yl = apply_hicks_henne_deformation(
            x=x,
            yu_base=zero,
            yl_base=zero,
            a_upper=[],
            a_lower=[1.0],
            upper_centers=[],
            lower_centers=[center],
            power=hh_power,
        )
        return np.asarray(yl, dtype=float)

    raise ValueError(f"Unknown candidate side for Hicks-Henne bump: {side}")


def compute_geometric_novelty(
    x,
    side,
    candidate_center,
    active_centers_same_side,
    hh_power,
    rcond=1e-10,
    eps_norm=1e-30,
):
    x = np.asarray(x, dtype=float)
    active_centers_same_side = list(active_centers_same_side)
    opt_ad = SETTINGS["optimization"]["adaptive"]
    novelty_eps = float(opt_ad.get("grad_novelty_eps", 0.10))
    novelty_power = float(opt_ad.get("grad_novelty_power", 2.0))

    phi_c = _single_hicks_henne_bump_from_deformation(
        x=x,
        side=side,
        center=candidate_center,
        hh_power=hh_power,
    )
    w = trapezoid_weights(x)
    sqrt_w = np.sqrt(w)
    phi_w = sqrt_w * phi_c
    norm_phi = float(np.linalg.norm(phi_w))

    if norm_phi < eps_norm:
        novelty_distance = 0.0
        return {
            "novelty_distance": novelty_distance,
            "novelty_factor": float(novelty_eps + (1.0 - novelty_eps) * novelty_distance**novelty_power),
            "max_corr_geo": np.nan,
            "norm_phi": norm_phi,
            "projection_residual_norm": 0.0,
            "projection_rank": 0,
        }

    if len(active_centers_same_side) == 0:
        novelty_distance = 1.0
        return {
            "novelty_distance": novelty_distance,
            "novelty_factor": float(novelty_eps + (1.0 - novelty_eps) * novelty_distance**novelty_power),
            "max_corr_geo": 0.0,
            "norm_phi": norm_phi,
            "projection_residual_norm": norm_phi,
            "projection_rank": 0,
        }

    phi_active = [
        _single_hicks_henne_bump_from_deformation(
            x=x,
            side=side,
            center=center,
            hh_power=hh_power,
        )
        for center in active_centers_same_side
    ]
    Phi_A = np.column_stack(phi_active)
    Phi_w = sqrt_w[:, None] * Phi_A

    coeffs, _residuals, rank, _s = np.linalg.lstsq(Phi_w, phi_w, rcond=rcond)
    residual = phi_w - Phi_w @ coeffs
    residual_norm = float(np.linalg.norm(residual))
    novelty_distance = float(np.clip(residual_norm / norm_phi, 0.0, 1.0))

    active_norms = np.linalg.norm(Phi_w, axis=0)
    valid = active_norms >= eps_norm
    if np.any(valid):
        corrs = np.abs(Phi_w[:, valid].T @ phi_w) / (active_norms[valid] * norm_phi)
        max_corr_geo = float(np.max(corrs))
    else:
        max_corr_geo = np.nan

    return {
        "novelty_distance": novelty_distance,
        "novelty_factor": float(novelty_eps + (1.0 - novelty_eps) * novelty_distance**novelty_power),
        "max_corr_geo": max_corr_geo,
        "norm_phi": norm_phi,
        "projection_residual_norm": residual_norm,
        "projection_rank": int(rank),
    }


def _get_grad_score_mode():
    mode = SETTINGS["optimization"]["adaptive"].get("grad_score_mode", "grad_norm")
    mode = str(mode).strip().lower()
    allowed = {"grad_norm", "grad_new", "grad_orth", "gn_schur"}
    if mode not in allowed:
        raise ValueError(f"Unknown ADAPT_GRAD_SCORE_MODE={mode!r}; expected one of {sorted(allowed)}")
    return mode


def _interp_candidate_cp_on_target(x_target, x_candidate, cp_candidate):
    x_target = np.asarray(x_target, dtype=float)
    x_candidate = np.asarray(x_candidate, dtype=float)
    cp_candidate = np.asarray(cp_candidate)

    if len(x_candidate) > 1 and x_candidate[0] > x_candidate[-1]:
        x_candidate = x_candidate[::-1]
        cp_candidate = cp_candidate[::-1]

    return np.interp(x_target, x_candidate, cp_candidate)


def cp_residual_vector(cp_target, cp_candidate):
    pieces = []

    for side in ("upper", "lower"):
        x_t = np.asarray(cp_target[side]["x"], dtype=float)
        cp_t = np.asarray(cp_target[side]["cp"])
        x_c = np.asarray(cp_candidate[side]["x"], dtype=float)
        cp_c = np.asarray(cp_candidate[side]["cp"])

        if len(x_t) > 1 and x_t[0] > x_t[-1]:
            x_eval = x_t[::-1]
            cp_target_eval = cp_t[::-1]
        else:
            x_eval = x_t
            cp_target_eval = cp_t

        cp_interp = _interp_candidate_cp_on_target(x_eval, x_c, cp_c)
        residual = cp_interp - cp_target_eval

        span = float(x_eval[-1] - x_eval[0]) if len(x_eval) > 1 else 0.0
        if span > 0.0:
            w_side = trapezoid_weights(x_eval) / span
        else:
            w_side = np.ones_like(x_eval, dtype=float) / max(len(x_eval), 1)

        pieces.append(np.sqrt(0.5 * w_side) * residual)

    return np.concatenate(pieces)


def _cp_response_column_fd(
    objective,
    a_base,
    idx,
    bounds,
    h,
    cp_target,
    base_y=None,
):
    a_base = np.asarray(a_base, dtype=float)
    idx = int(idx)
    h = float(abs(h))
    bmin, bmax = bounds[idx]

    if not np.isfinite(h) or h <= 0.0:
        return {
            "column": np.zeros_like(base_y, dtype=float) if base_y is not None else np.zeros(0, dtype=float),
            "mode": "NO_ROOM",
            "h": h,
            "n_fail": 1,
        }

    can_minus = a_base[idx] - h >= bmin
    can_plus = a_base[idx] + h <= bmax

    if not can_minus and not can_plus:
        return {
            "column": np.zeros_like(base_y, dtype=float) if base_y is not None else np.zeros(0, dtype=float),
            "mode": "NO_ROOM",
            "h": h,
            "n_fail": 1,
        }

    def _state_y(a_vec):
        state = objective.evaluate_cp_state(a_vec)
        if state.get("status") != "OK":
            return None
        return cp_residual_vector(cp_target, state["cp_candidate"])

    if can_minus and can_plus:
        a_plus = a_base.copy()
        a_minus = a_base.copy()
        a_plus[idx] += h
        a_minus[idx] -= h
        y_plus = _state_y(a_plus)
        y_minus = _state_y(a_minus)
        if y_plus is None or y_minus is None:
            col_shape = base_y if base_y is not None else (y_plus if y_plus is not None else y_minus)
            return {
                "column": np.zeros_like(col_shape, dtype=float) if col_shape is not None else np.zeros(0, dtype=float),
                "mode": "FAIL",
                "h": h,
                "n_fail": 1,
            }
        return {
            "column": (y_plus - y_minus) / (2.0 * h),
            "mode": "CENTRAL",
            "h": h,
            "n_fail": 0,
        }

    if base_y is None:
        base_y = _state_y(a_base)
        if base_y is None:
            return {"column": np.zeros(0, dtype=float), "mode": "FAIL", "h": h, "n_fail": 1}

    if can_plus:
        a_plus = a_base.copy()
        a_plus[idx] += h
        y_plus = _state_y(a_plus)
        if y_plus is None:
            return {"column": np.zeros_like(base_y, dtype=float), "mode": "FAIL", "h": h, "n_fail": 1}
        return {"column": (y_plus - base_y) / h, "mode": "FORWARD", "h": h, "n_fail": 0}

    a_minus = a_base.copy()
    a_minus[idx] -= h
    y_minus = _state_y(a_minus)
    if y_minus is None:
        return {"column": np.zeros_like(base_y, dtype=float), "mode": "FAIL", "h": h, "n_fail": 1}
    return {"column": (base_y - y_minus) / h, "mode": "BACKWARD", "h": h, "n_fail": 0}


def _gn_schur_target_peak_normal():
    opt_ad = SETTINGS["optimization"]["adaptive"]
    target = opt_ad.get("gn_schur_fd_target_peak_normal", None)
    if target is None:
        target = SETTINGS["optimization"]["score_fd_target_peak_normal"]
    return float(target)


def _warn_if_constraints_active_for_gn_schur():
    active = []
    constraints = SETTINGS.get("constraints", {})
    for name, spec in constraints.items():
        if isinstance(spec, dict) and bool(spec.get("enabled", False)):
            active.append(name)
    if active:
        print(
            "[warning] ADAPT_GRAD_SCORE_MODE='gn_schur' uses Cp-only micro-model; "
            f"active constraints ignored in candidate scoring: {active}"
        )


def prepare_gn_schur_level_context(
    x,
    yu_init,
    yl_init,
    cp_target,
    active_upper,
    active_lower,
    active_a,
    current_best_error,
    workdir,
):
    _warn_if_constraints_active_for_gn_schur()

    opt_ad = SETTINGS["optimization"]["adaptive"]
    active_upper = list(active_upper)
    active_lower = list(active_lower)
    active_a = np.asarray(active_a, dtype=float)

    objective = make_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=active_upper,
        lower_centers=active_lower,
        hh_power=SETTINGS["optimization"]["hh_power"],
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / "gn_schur_level_active",
        current_best_error=current_best_error,
    )
    objective.set_eval_phase("score")

    base_state = objective.evaluate_cp_state(active_a)
    n_scoring = int(objective.aero_call_counter_by_phase.get("score", 0))
    if base_state.get("status") != "OK":
        return {
            "y": np.zeros(0, dtype=float),
            "J_A": np.zeros((0, len(active_a)), dtype=float),
            "Q_A": np.zeros((0, 0), dtype=float),
            "rank_A": 0,
            "singular_values_A": np.zeros(0, dtype=float),
            "base_status": base_state.get("status", "FAIL"),
            "n_scoring_aero_calls": n_scoring,
            "fd_diag_active": {"n_fail_dirs": len(active_a), "modes": [], "h": []},
        }

    y = cp_residual_vector(cp_target, base_state["cp_candidate"])
    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * len(active_a)
    step_vector = build_normal_peak_fd_steps(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=active_upper,
        lower_centers=active_lower,
        a=active_a,
        target_peak_normal=_gn_schur_target_peak_normal(),
        power=SETTINGS["optimization"]["hh_power"],
    )

    cols = []
    modes = []
    hs = []
    n_fail_dirs = 0
    for idx in range(len(active_a)):
        fd = _cp_response_column_fd(
            objective=objective,
            a_base=active_a,
            idx=idx,
            bounds=bounds,
            h=step_vector[idx],
            cp_target=cp_target,
            base_y=y,
        )
        col = np.asarray(fd["column"], dtype=float)
        if col.size != y.size:
            col = np.zeros_like(y, dtype=float)
        cols.append(col)
        modes.append(fd["mode"])
        hs.append(float(fd["h"]))
        n_fail_dirs += int(fd["n_fail"])

    if len(cols) == 0:
        J_A = np.zeros((len(y), 0), dtype=float)
    else:
        J_A = np.column_stack(cols)

    if J_A.shape[1] == 0:
        singular_values = np.zeros(0, dtype=float)
        Q_A = np.zeros((len(y), 0), dtype=float)
        rank_A = 0
    else:
        U, singular_values, _Vt = np.linalg.svd(J_A, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= 0.0:
            keep = np.zeros_like(singular_values, dtype=bool)
        else:
            rcond = float(opt_ad.get("gn_schur_rcond", 1.0e-10))
            keep = singular_values > rcond * singular_values[0]
        Q_A = U[:, keep]
        rank_A = int(np.count_nonzero(keep))

    return {
        "y": y,
        "J_A": J_A,
        "Q_A": Q_A,
        "rank_A": rank_A,
        "singular_values_A": singular_values,
        "base_status": base_state["status"],
        "n_scoring_aero_calls": int(objective.aero_call_counter_by_phase.get("score", 0)),
        "fd_diag_active": {
            "n_fail_dirs": int(n_fail_dirs),
            "modes": modes,
            "h": hs,
            "ndv": len(active_a),
        },
    }


def _score_candidate_gn_schur(
    objective,
    a_base,
    new_idx,
    bounds,
    cp_target,
    gn_context,
    step_vector,
):
    if gn_context is None or gn_context.get("base_status") != "OK":
        return {
            "score": -np.inf,
            "component": np.nan,
            "raw_grad": np.nan,
            "mode": "GN_SCHUR_FAIL_CONTEXT",
            "score_mode": "gn_schur",
            "g_norm": np.nan,
            "g_new": np.nan,
            "abs_g_new": np.nan,
            "gn_g_perp": np.nan,
            "gn_q_schur": np.nan,
            "gn_alpha_unclipped": np.nan,
            "gn_alpha_clipped": np.nan,
            "gn_norm_jc": np.nan,
            "gn_norm_z": np.nan,
            "gn_rank_A": int(gn_context.get("rank_A", 0)) if gn_context is not None else 0,
            "gn_fd_mode": "FAIL_CONTEXT",
            "gn_fd_h": np.nan,
            "gn_n_fail_candidate": 1,
        }

    y = np.asarray(gn_context["y"], dtype=float)
    Q_A = np.asarray(gn_context["Q_A"], dtype=float)

    fd = _cp_response_column_fd(
        objective=objective,
        a_base=a_base,
        idx=new_idx,
        bounds=bounds,
        h=step_vector[new_idx],
        cp_target=cp_target,
        base_y=y,
    )
    j_c = np.asarray(fd["column"], dtype=float)
    if j_c.size != y.size:
        j_c = np.zeros_like(y, dtype=float)

    if Q_A.size == 0 or Q_A.shape[1] == 0:
        z_c = j_c
    else:
        z_c = j_c - Q_A @ (Q_A.T @ j_c)

    reg = float(SETTINGS["optimization"]["adaptive"].get("gn_schur_reg", 1.0e-10))
    g_perp = float(z_c @ y)
    q_schur = float(z_c @ z_c + reg)
    norm_jc = float(np.linalg.norm(j_c))
    norm_z = float(np.linalg.norm(z_c))

    if q_schur <= 0.0 or not np.isfinite(q_schur):
        print(f"[warning] invalid GN-Schur q_schur={q_schur}; forcing score=-inf")
        alpha_unclipped = np.nan
        alpha_clipped = np.nan
        score = -np.inf
    else:
        alpha_unclipped = float(-g_perp / q_schur)
        bmin, bmax = bounds[new_idx]
        alpha_clipped = float(np.clip(alpha_unclipped, bmin, bmax))
        score = float(max(0.0, -g_perp * alpha_clipped - 0.5 * q_schur * alpha_clipped**2))
        if not np.isfinite(score):
            print(f"[warning] non-finite GN-Schur score={score}; forcing score=-inf")
            score = -np.inf

    return {
        "score": score,
        "component": g_perp,
        "raw_grad": g_perp,
        "mode": "GN_SCHUR",
        "score_mode": "gn_schur",
        "g_norm": np.nan,
        "g_new": g_perp,
        "abs_g_new": abs(g_perp),
        "gn_g_perp": g_perp,
        "gn_q_schur": q_schur,
        "gn_alpha_unclipped": alpha_unclipped,
        "gn_alpha_clipped": alpha_clipped,
        "gn_norm_jc": norm_jc,
        "gn_norm_z": norm_z,
        "gn_rank_A": int(gn_context.get("rank_A", 0)),
        "gn_fd_mode": fd["mode"],
        "gn_fd_h": float(fd["h"]),
        "gn_n_fail_candidate": int(fd["n_fail"]),
    }


def _score_candidate_grad(
    x,
    side,
    xc,
    active_centers_same_side,
    objective,
    a_base,
    new_idx,
    bounds,
    rel_step,
    abs_step_floor,
    step_vector,
    base_item,
):
    deriv_mode = SETTINGS.get("aero", {}).get("derivatives", "fd").strip().lower()

    if deriv_mode == "cs" and hasattr(objective, "compute_gradient_cs"):
        grad_j = objective.compute_gradient_cs(np.asarray(a_base, dtype=float))
        fd_diag = {
            "n_fail_dirs": 0,
            "fail_indices": [],
            "ndv": len(grad_j),
            "mode": "CS",
        }
    else:
        grad_j, fd_diag = _compute_full_objective_gradient(
            objective=objective,
            a_base=a_base,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
            step_vector=step_vector,
            base_item=base_item,
        )

    opt_ad = SETTINGS["optimization"]["adaptive"]
    grad_score_mode = _get_grad_score_mode()
    g_norm = float(np.linalg.norm(grad_j))
    g_new = float(np.real(grad_j[new_idx]))
    abs_g_new = abs(g_new)

    novelty = compute_geometric_novelty(
        x=x,
        side=side,
        candidate_center=xc,
        active_centers_same_side=active_centers_same_side,
        hh_power=SETTINGS["optimization"]["hh_power"],
        rcond=float(opt_ad.get("grad_novelty_rcond", 1.0e-10)),
    )
    novelty_factor = float(novelty["novelty_factor"])

    if grad_score_mode == "grad_norm":
        score = g_norm
    elif grad_score_mode == "grad_new":
        score = abs_g_new
    elif grad_score_mode == "grad_orth":
        score = abs_g_new * novelty_factor
    else:
        raise ValueError(f"Unknown ADAPT_GRAD_SCORE_MODE={grad_score_mode!r}")

    if not np.isfinite(score):
        print(
            f"[warning] non-finite ADAPT_GRAD score for side={side} center={float(xc):.6f}: "
            f"score={score}; forcing score=-inf"
        )
        score = -np.inf

    return {
        "score": score,
        "component": g_new,
        "raw_grad": g_new,
        "grad_j": grad_j,
        "fd_diag": fd_diag,
        "mode": "GRAD_CS" if deriv_mode == "cs" else "GRAD_FD",
        "score_mode": grad_score_mode,
        "g_norm": g_norm,
        "g_new": g_new,
        "abs_g_new": abs_g_new,
        **novelty,
    }


def _score_candidate_ikkt(
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    objective,
    a_base,
    new_idx,
    bounds,
    rel_step,
    abs_step_floor,
    step_vector,
    base_item,
):
    if base_item.get("status") != "OK":
        z = np.zeros_like(a_base)
        return {
            "score": 0.0,
            "component": 0.0,
            "active_entries": [],
            "lam": np.zeros(0),
            "r": z,
            "grad_j": z,
            "mode": "FAIL_BASE",
            "fd_diag": {"n_fail_dirs": len(a_base), "fail_indices": list(range(len(a_base))), "ndv": len(a_base)},
        }

    enabled_aero = []
    for name in ("CL", "CD", "CM"):
        spec = SETTINGS.get("constraints", {}).get(name, {})
        if spec.get("enabled", False):
            enabled_aero.append(name)

    deriv_mode = SETTINGS.get("aero", {}).get("derivatives", "fd").strip().lower()

    if deriv_mode == "cs" and hasattr(objective, "compute_aero_gradients_cs"):
        grad_j, grad_metrics_aero, fd_diag = objective.compute_aero_gradients_cs(
            np.asarray(a_base, dtype=float),
            enabled_metric_names=enabled_aero,
        )
        ikkt_mode = "IKKT_CS"
    else:
        grad_j, grad_metrics_aero, fd_diag = _compute_full_aero_gradients(
            objective=objective,
            a_base=a_base,
            enabled_metric_names=enabled_aero,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
            step_vector=step_vector,
            base_item=base_item,
        )
        ikkt_mode = "IKKT_FD"

    yu, yl = _rebuild_geometry_from_a(x, yu_init, yl_init, upper_centers, lower_centers, a_base)
    grad_geom_base = _compute_geometric_gradients_analytic(x, upper_centers, lower_centers)

    active_entries, G = _build_active_ikkt_system(
        x=x,
        yu=yu,
        yl=yl,
        metrics=base_item["metrics"],
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        grad_metrics_aero=grad_metrics_aero,
        grad_geom_base=grad_geom_base,
    )

    if G is None or len(active_entries) == 0:
        gnew = float(grad_j[new_idx])
        score = float(np.linalg.norm(grad_j))
        return {
            "score": score,
            "component": gnew,
            "active_entries": [],
            "lam": np.zeros(0),
            "r": grad_j.copy(),
            "grad_j": grad_j.copy(),
            "mode": f"NO_ACTIVE_{'CS' if deriv_mode == 'cs' else 'FD'}",
            "fd_diag": fd_diag,
        }

    lam = _solve_bounded_least_squares(grad_j, G, active_entries)
    r = grad_j - G @ lam
    score = float(np.linalg.norm(r))
    return {
        "score": score,
        "component": float(r[new_idx]),
        "active_entries": active_entries,
        "lam": lam,
        "r": r,
        "grad_j": grad_j,
        "mode": ikkt_mode,
        "fd_diag": fd_diag,
    }


def score_candidate(
    x,
    yu_init,
    yl_init,
    cp_target,
    active_upper,
    active_lower,
    active_a,
    candidate,
    current_best_error,
    workdir,
    indicator,
    pred_context=None,
    gn_context=None,
):
    side, xc, new_upper, new_lower, a_base, new_idx, old_to_new = _build_extended_space(
        active_upper=active_upper,
        active_lower=active_lower,
        active_a=active_a,
        candidate=candidate,
    )

    objective = _make_candidate_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        new_upper=new_upper,
        new_lower=new_lower,
        current_best_error=current_best_error,
        workdir=workdir,
        side=side,
        xc=xc,
    )
    objective.set_eval_phase("score")

    indicator = str(indicator).upper()
    candidate_context = _format_candidate_context(candidate)

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * len(a_base)
    rel_step = 0.0
    abs_step_floor = 0.0

    grad_score_mode = _get_grad_score_mode() if indicator == "GRAD" else None
    base_item = None
    if not (indicator == "GRAD" and grad_score_mode == "gn_schur"):
        base_item = _evaluate_objective_state(objective, a_base)
    step_vector = None

    if indicator in ("GRAD", "IKKT"):
        if indicator == "GRAD" and grad_score_mode == "gn_schur":
            target_peak_normal = _gn_schur_target_peak_normal()
        else:
            target_peak_normal = SETTINGS["optimization"]["score_fd_target_peak_normal"]
        step_vector = build_normal_peak_fd_steps(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a=a_base,
            target_peak_normal=target_peak_normal,
            power=SETTINGS["optimization"]["hh_power"],
        )

    if indicator == "IKKT":
        out = _score_candidate_ikkt(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            upper_centers=new_upper,
            lower_centers=new_lower,
            objective=objective,
            a_base=a_base,
            new_idx=new_idx,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
            step_vector=step_vector,
            base_item=base_item,
        )
        active_names = [
            f"{e['name']}@{e['x']:.3f}" if e["name"] == "thickness_stations" else e["name"]
            for e in out["active_entries"]
        ]
        n_scoring_aero_calls = int(objective.aero_call_counter_by_phase.get("score", 0))

        msg = (
            f"SCORE[{indicator}] side={side} candidate={xc:.6f}{candidate_context}  "
            f"mode={out['mode']}  "
            f"r_norm={out['score']:.6e}  "
            f"r_new={out['component']:.6e}  "
            f"gnew={out['grad_j'][new_idx]:.6e}  "
            f"n_active={len(out['active_entries'])}  "
            f"score_aero_calls={n_scoring_aero_calls}  "
            f"debug_evals={objective.eval_counter['k']}"
        )
        print(msg)

        if len(out["active_entries"]) > 0:
            print(f"  active = {active_names}")
            print(f"  lambda = {np.array2string(out['lam'], precision=4, suppress_small=False)}")

        return {
            "side": side,
            "x": xc,
            "score": out["score"],
            "component": out["component"],
            "raw_grad": float(out["grad_j"][new_idx]),
            "active_names": active_names,
            "lambda": out["lam"].copy(),
            "n_scoring_aero_calls": n_scoring_aero_calls,
            "indicator": indicator,
            "mode": out["mode"],
        }
    if indicator == "PRED":
        out = _score_candidate_pred(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            upper_centers=new_upper,
            lower_centers=new_lower,
            objective=objective,
            a_base=a_base,
            new_idx=new_idx,
            old_to_new=old_to_new,
            bounds=bounds,
            base_item=base_item,
            pred_context=pred_context,
        )

        active_names = [
            f"{e['name']}@{e['x']:.3f}" if e["name"] == "thickness_stations" else e["name"]
            for e in out["active_entries"]
        ]
        n_scoring_aero_calls = int(objective.aero_call_counter_by_phase.get("score", 0))

        print(
            f"SCORE[{indicator}] side={side} candidate={xc:.6f}{candidate_context}  "
            f"delta_pred={out['score']:.6e}  "
            f"d_new={out['component']:.6e}  "
            f"gnew={out['raw_grad']:.6e}  "
            f"tau={out.get('tau', 1.0):.6e}  "
            f"n_active={len(out['active_entries'])}  "
            f"score_aero_calls={n_scoring_aero_calls}  "
            f"debug_evals={objective.eval_counter['k']}  "
            f"mode={out['mode']}"
        )

        return {
            "side": side,
            "x": xc,
            "score": out["score"],
            "component": out["component"],
            "raw_grad": out["raw_grad"],
            "active_names": active_names,
            "lambda": np.zeros(0),
            "n_scoring_aero_calls": n_scoring_aero_calls,
            "indicator": indicator,
            "mode": out["mode"],
            "tau": out.get("tau", 1.0),
        }

    if grad_score_mode == "gn_schur":
        out = _score_candidate_gn_schur(
            objective=objective,
            a_base=a_base,
            new_idx=new_idx,
            bounds=bounds,
            cp_target=cp_target,
            gn_context=gn_context,
            step_vector=step_vector,
        )
        n_scoring_aero_calls = int(objective.aero_call_counter_by_phase.get("score", 0))

        print(
            f"SCORE[{indicator}] side={side} candidate={xc:.6f}{candidate_context}  "
            f"score_mode=gn_schur  "
            f"score={out['score']:.6e}  "
            f"g_perp={out['gn_g_perp']:.6e}  "
            f"q_schur={out['gn_q_schur']:.6e}  "
            f"alpha={out['gn_alpha_clipped']:.6e}  "
            f"norm_jc={out['gn_norm_jc']:.6e}  "
            f"norm_z={out['gn_norm_z']:.6e}  "
            f"rank_A={out['gn_rank_A']}  "
            f"fd_mode={out['gn_fd_mode']}  "
            f"score_aero_calls={n_scoring_aero_calls}  "
            f"debug_evals={objective.eval_counter['k']}"
        )

        return {
            "side": side,
            "x": xc,
            "score": out["score"],
            "component": out["component"],
            "raw_grad": out["raw_grad"],
            "active_names": [],
            "lambda": np.zeros(0),
            "n_scoring_aero_calls": n_scoring_aero_calls,
            "indicator": indicator,
            "mode": out["mode"],
            "score_mode": out["score_mode"],
            "g_norm": out["g_norm"],
            "g_new": out["g_new"],
            "abs_g_new": out["abs_g_new"],
            "novelty_distance": np.nan,
            "novelty_factor": np.nan,
            "max_corr_geo": np.nan,
            "norm_phi": np.nan,
            "projection_residual_norm": np.nan,
            "projection_rank": 0,
            "gn_g_perp": out["gn_g_perp"],
            "gn_q_schur": out["gn_q_schur"],
            "gn_alpha_unclipped": out["gn_alpha_unclipped"],
            "gn_alpha_clipped": out["gn_alpha_clipped"],
            "gn_norm_jc": out["gn_norm_jc"],
            "gn_norm_z": out["gn_norm_z"],
            "gn_rank_A": out["gn_rank_A"],
            "gn_fd_mode": out["gn_fd_mode"],
            "gn_fd_h": out["gn_fd_h"],
            "gn_n_fail_candidate": out["gn_n_fail_candidate"],
        }

    out = _score_candidate_grad(
        x=x,
        side=side,
        xc=xc,
        active_centers_same_side=active_upper if str(side).upper() == "UPPER" else active_lower,
        objective=objective,
        a_base=a_base,
        new_idx=new_idx,
        bounds=bounds,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
        step_vector=step_vector,
        base_item=base_item,
    )

    print(
        f"SCORE[{indicator}] side={side} candidate={xc:.6f}{candidate_context}  "
        f"score_mode={out['score_mode']}  "
        f"score={out['score']:.6e}  "
        f"g_norm={out['g_norm']:.6e}  "
        f"g_new={out['component']:.6e}  "
        f"novelty_d={out['novelty_distance']:.6e}  "
        f"novelty_factor={out['novelty_factor']:.6e}  "
        f"max_corr_geo={out['max_corr_geo']:.6e}  "
        f"mode={out['mode']}  "
        f"n_fail_dirs={out['fd_diag']['n_fail_dirs']}  "
        f"score_aero_calls={int(objective.aero_call_counter_by_phase.get('score', 0))}  "
        f"debug_evals={objective.eval_counter['k']}"
    )

    return {
        "side": side,
        "x": xc,
        "score": out["score"],
        "component": out["component"],
        "raw_grad": out["raw_grad"],
        "active_names": [],
        "lambda": np.zeros(0),
        "n_scoring_aero_calls": int(objective.aero_call_counter_by_phase.get("score", 0)),
        "indicator": indicator,
        "mode": out["mode"],
        "score_mode": out["score_mode"],
        "g_norm": out["g_norm"],
        "g_new": out["g_new"],
        "abs_g_new": out["abs_g_new"],
        "novelty_distance": out["novelty_distance"],
        "novelty_factor": out["novelty_factor"],
        "max_corr_geo": out["max_corr_geo"],
        "norm_phi": out["norm_phi"],
        "projection_residual_norm": out["projection_residual_norm"],
        "projection_rank": out["projection_rank"],
        "gn_g_perp": np.nan,
        "gn_q_schur": np.nan,
        "gn_alpha_unclipped": np.nan,
        "gn_alpha_clipped": np.nan,
        "gn_norm_jc": np.nan,
        "gn_norm_z": np.nan,
        "gn_rank_A": np.nan,
        "gn_fd_mode": "",
        "gn_fd_h": np.nan,
        "gn_n_fail_candidate": np.nan,
    }

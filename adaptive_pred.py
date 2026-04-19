from pathlib import Path

import numpy as np

from settings import SETTINGS
from objective import make_objective
from adaptive_candidate import _evaluate_objective_state, _rebuild_geometry_from_a
from adaptive_fd import _adaptive_fd_step, _compute_full_aero_gradients
from adaptive_constraints import (
    _build_active_ikkt_system,
    _compute_geometric_gradients_analytic,
)


def _extract_objective_base(item):
    return float(item.get("objective_base", item["objective"]))


def _null_space_from_constraint_columns(G, ndv, rtol=1.0e-10):
    if G is None or G.size == 0 or G.shape[1] == 0:
        return np.eye(ndv)

    A = np.asarray(G, dtype=float).T
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    if len(s) == 0:
        return np.eye(ndv)

    rank = int(np.sum(s > rtol * max(1.0, s[0])))
    Z = Vt[rank:].T

    if Z.size == 0:
        return np.zeros((ndv, 0))

    return Z


def _compute_restoration_step(active_entries, G, ndv):
    if G is None or len(active_entries) == 0:
        return np.zeros(ndv), np.zeros(0), np.zeros((0, ndv))

    A = np.asarray(G, dtype=float).T
    c = np.array([float(e["c_value"]) for e in active_entries], dtype=float)

    Y = A.T
    M = A @ Y
    p_y = -np.linalg.pinv(M) @ c
    d_y = Y @ p_y

    return d_y, c, A


def _compute_full_objective_hessian(
    objective,
    a_base,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)
    H = np.zeros((ndv, ndv), dtype=float)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    if base_item.get("status") != "OK":
        return H

    f0 = _extract_objective_base(base_item)

    for i in range(ndv):
        ai = float(a_base[i])
        lo_i, hi_i = bounds[i]
        hi_step = _adaptive_fd_step(ai, rel_step, abs_step_floor)

        if (ai + hi_step) > hi_i or (ai - hi_step) < lo_i:
            continue

        a_p = a_base.copy()
        a_m = a_base.copy()
        a_p[i] += hi_step
        a_m[i] -= hi_step

        item_p = _evaluate_objective_state(objective, a_p)
        item_m = _evaluate_objective_state(objective, a_m)

        if item_p.get("status") == "OK" and item_m.get("status") == "OK":
            f_p = _extract_objective_base(item_p)
            f_m = _extract_objective_base(item_m)
            H[i, i] = (f_p - 2.0 * f0 + f_m) / (hi_step ** 2)

        for j in range(i + 1, ndv):
            aj = float(a_base[j])
            lo_j, hi_j = bounds[j]
            hj_step = _adaptive_fd_step(aj, rel_step, abs_step_floor)

            if (
                (ai + hi_step) > hi_i or (ai - hi_step) < lo_i
                or (aj + hj_step) > hi_j or (aj - hj_step) < lo_j
            ):
                continue

            a_pp = a_base.copy()
            a_pm = a_base.copy()
            a_mp = a_base.copy()
            a_mm = a_base.copy()

            a_pp[i] += hi_step
            a_pp[j] += hj_step

            a_pm[i] += hi_step
            a_pm[j] -= hj_step

            a_mp[i] -= hi_step
            a_mp[j] += hj_step

            a_mm[i] -= hi_step
            a_mm[j] -= hj_step

            item_pp = _evaluate_objective_state(objective, a_pp)
            item_pm = _evaluate_objective_state(objective, a_pm)
            item_mp = _evaluate_objective_state(objective, a_mp)
            item_mm = _evaluate_objective_state(objective, a_mm)

            if (
                item_pp.get("status") == "OK"
                and item_pm.get("status") == "OK"
                and item_mp.get("status") == "OK"
                and item_mm.get("status") == "OK"
            ):
                f_pp = _extract_objective_base(item_pp)
                f_pm = _extract_objective_base(item_pm)
                f_mp = _extract_objective_base(item_mp)
                f_mm = _extract_objective_base(item_mm)

                hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi_step * hj_step)
                H[i, j] = hij
                H[j, i] = hij

    return H


def _compute_candidate_hessian_border(
    objective,
    a_base,
    new_idx,
    old_to_new,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    h_col = np.zeros(ndv, dtype=float)

    if base_item.get("status") != "OK":
        return 0.0, h_col

    f0 = _extract_objective_base(base_item)

    an = float(a_base[new_idx])
    lo_n, hi_n = bounds[new_idx]
    hn = _adaptive_fd_step(an, rel_step, abs_step_floor)

    if (an + hn) <= hi_n and (an - hn) >= lo_n:
        a_p = a_base.copy()
        a_m = a_base.copy()
        a_p[new_idx] += hn
        a_m[new_idx] -= hn

        item_p = _evaluate_objective_state(objective, a_p)
        item_m = _evaluate_objective_state(objective, a_m)

        if item_p.get("status") == "OK" and item_m.get("status") == "OK":
            f_p = _extract_objective_base(item_p)
            f_m = _extract_objective_base(item_m)
            h_nn = (f_p - 2.0 * f0 + f_m) / (hn ** 2)
        else:
            h_nn = 0.0
    else:
        h_nn = 0.0

    for idx_old_new in old_to_new:
        ai = float(a_base[idx_old_new])
        lo_i, hi_i = bounds[idx_old_new]
        hi_step = _adaptive_fd_step(ai, rel_step, abs_step_floor)

        if (
            (ai + hi_step) > hi_i or (ai - hi_step) < lo_i
            or (an + hn) > hi_n or (an - hn) < lo_n
        ):
            continue

        a_pp = a_base.copy()
        a_pm = a_base.copy()
        a_mp = a_base.copy()
        a_mm = a_base.copy()

        a_pp[idx_old_new] += hi_step
        a_pp[new_idx] += hn

        a_pm[idx_old_new] += hi_step
        a_pm[new_idx] -= hn

        a_mp[idx_old_new] -= hi_step
        a_mp[new_idx] += hn

        a_mm[idx_old_new] -= hi_step
        a_mm[new_idx] -= hn

        item_pp = _evaluate_objective_state(objective, a_pp)
        item_pm = _evaluate_objective_state(objective, a_pm)
        item_mp = _evaluate_objective_state(objective, a_mp)
        item_mm = _evaluate_objective_state(objective, a_mm)

        if (
            item_pp.get("status") == "OK"
            and item_pm.get("status") == "OK"
            and item_mp.get("status") == "OK"
            and item_mm.get("status") == "OK"
        ):
            f_pp = _extract_objective_base(item_pp)
            f_pm = _extract_objective_base(item_pm)
            f_mp = _extract_objective_base(item_mp)
            f_mm = _extract_objective_base(item_mm)

            hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi_step * hn)
            h_col[idx_old_new] = hij

    return h_nn, h_col


def _prepare_pred_level_context(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a_opt,
    current_best_error,
    workdir,
):
    pred_rel_step = SETTINGS["optimization"].get(
        "pred_fd_rel_step",
        SETTINGS["optimization"]["fd_rel_step"],
    )
    pred_abs_step_floor = SETTINGS["optimization"].get(
        "pred_fd_abs_step_floor",
        SETTINGS["optimization"]["fd_abs_step_floor"],
    )

    objective_old = make_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=SETTINGS["optimization"]["hh_power"],
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / "pred_level_base",
        current_best_error=current_best_error,
    )

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds_old = [(bmin, bmax)] * len(a_opt)

    base_item_old = _evaluate_objective_state(objective_old, a_opt)

    enabled_aero = []
    for name in ("CL", "CD", "CM"):
        spec = SETTINGS.get("constraints", {}).get(name, {})
        if spec.get("enabled", False):
            enabled_aero.append(name)

    g_old, grad_metrics_old, _ = _compute_full_aero_gradients(
        objective=objective_old,
        a_base=a_opt,
        enabled_metric_names=enabled_aero,
        bounds=bounds_old,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item_old,
    )

    yu_old, yl_old = _rebuild_geometry_from_a(
        x, yu_init, yl_init, upper_centers, lower_centers, a_opt
    )
    grad_geom_old = _compute_geometric_gradients_analytic(x, upper_centers, lower_centers)

    active_entries_old, G_old = _build_active_ikkt_system(
        x=x,
        yu=yu_old,
        yl=yl_old,
        metrics=base_item_old["metrics"],
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        grad_metrics_aero=grad_metrics_old,
        grad_geom_base=grad_geom_old,
    )

    Z_old = _null_space_from_constraint_columns(G_old, len(a_opt))

    H_old = _compute_full_objective_hessian(
        objective=objective_old,
        a_base=a_opt,
        bounds=bounds_old,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item_old,
    )

    return {
        "a_k": np.asarray(a_opt, dtype=float).copy(),
        "g_k": g_old.copy(),
        "active_entries_k": list(active_entries_old),
        "G_k": None if G_old is None else G_old.copy(),
        "Z_k": Z_old.copy(),
        "H_k": H_old.copy(),
        "pred_rel_step": pred_rel_step,
        "pred_abs_step_floor": pred_abs_step_floor,
    }


def _compute_max_feasible_step(a_base, d_full, bounds, tol=1.0e-12):
    """
    Restituisce il massimo tau in [0, 1] tale che:
        a_base + tau * d_full
    resti dentro tutti i bounds.
    """
    tau = 1.0

    for j, (lo, hi) in enumerate(bounds):
        aj = float(a_base[j])
        dj = float(d_full[j])

        if abs(dj) <= tol:
            continue

        if dj > 0.0:
            tau_j = (hi - aj) / dj
        else:
            tau_j = (lo - aj) / dj

        tau = min(tau, tau_j)

    tau = max(0.0, min(1.0, float(tau)))
    return tau


def _quadratic_predicted_reduction(grad_j, H_full, d_trial):
    """
    Riduzione predetta del modello quadratico lungo il passo d_trial:
        m(d) = g^T d + 0.5 d^T H d
        Delta_pred = -m(d)
    """
    d_trial = np.asarray(d_trial, dtype=float)
    val = float(np.dot(grad_j, d_trial) + 0.5 * d_trial @ H_full @ d_trial)
    return -val


def _score_candidate_pred(
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    objective,
    a_base,
    new_idx,
    old_to_new,
    bounds,
    base_item,
    pred_context,
):
    if base_item.get("status") != "OK":
        z = np.zeros_like(a_base)
        return {
            "score": -np.inf,
            "component": 0.0,
            "raw_grad": 0.0,
            "grad_j": z,
            "gamma": np.zeros(0),
            "d_full": z,
            "H_red": np.zeros((0, 0)),
            "mode": "FAIL_BASE",
            "active_entries": [],
        }

    pred_rel_step = pred_context["pred_rel_step"]
    pred_abs_step_floor = pred_context["pred_abs_step_floor"]

    enabled_aero = []
    for name in ("CL", "CD", "CM"):
        spec = SETTINGS.get("constraints", {}).get(name, {})
        if spec.get("enabled", False):
            enabled_aero.append(name)

    grad_j, grad_metrics_aero, _ = _compute_full_aero_gradients(
        objective=objective,
        a_base=a_base,
        enabled_metric_names=enabled_aero,
        bounds=bounds,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item,
    )

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

    Z = _null_space_from_constraint_columns(G, len(a_base))
    d_y, c_vec, A = _compute_restoration_step(active_entries, G, len(a_base))

    H_full = np.zeros((len(a_base), len(a_base)), dtype=float)
    H_old = pred_context["H_k"]
    H_full[np.ix_(old_to_new, old_to_new)] = H_old

    h_nn, h_col = _compute_candidate_hessian_border(
        objective=objective,
        a_base=a_base,
        new_idx=new_idx,
        old_to_new=old_to_new,
        bounds=bounds,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item,
    )

    H_full[new_idx, new_idx] = h_nn
    for idx_old_new in old_to_new:
        H_full[idx_old_new, new_idx] = h_col[idx_old_new]
        H_full[new_idx, idx_old_new] = h_col[idx_old_new]

    rhs_full = grad_j + H_full @ d_y
    gamma = Z.T @ rhs_full
    H_red = Z.T @ H_full @ Z

    if H_red.size == 0:
        d_full = d_y.copy()
        delta_pred = 0.0
        mode = "NO_FREE_DIRS"
    else:
        reg = float(SETTINGS["optimization"].get("pred_hessian_reg", 1.0e-8))
        H_red_reg = H_red + reg * np.eye(H_red.shape[0])

        Hred_inv = np.linalg.pinv(H_red_reg)
        p_z = -Hred_inv @ gamma
        d_full = d_y + Z @ p_z
        delta_pred = 0.5 * float(gamma.T @ Hred_inv @ gamma)
        mode = "PRED"

    tau = _compute_max_feasible_step(a_base, d_full, bounds)
    d_trial = tau * d_full

    if tau < 1.0 - 1.0e-12:
        delta_pred = _quadratic_predicted_reduction(grad_j, H_full, d_trial)
        mode = "BOUNDS_SCALED"
    else:
        delta_pred = _quadratic_predicted_reduction(grad_j, H_full, d_trial)

    return {
        "score": float(delta_pred),
        "component": float(d_trial[new_idx]),
        "raw_grad": float(grad_j[new_idx]),
        "grad_j": grad_j,
        "gamma": gamma,
        "d_full": d_trial,
        "H_red": H_red,
        "mode": mode,
        "active_entries": active_entries,
        "tau": float(tau),
    }

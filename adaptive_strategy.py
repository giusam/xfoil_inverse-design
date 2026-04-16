from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from settings import SETTINGS
from objective import make_objective
from optimization import optimize_for_centers
from geometry import apply_hicks_henne_deformation, build_hicks_henne_basis
from constraints import build_all_signed_constraints
from adaptive_utils import (
    build_side_specific_initial_centers,
    get_midpoint_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)


IKKT_EXCLUDED_NAMES = {"tmax"}


def _build_extended_space(active_upper, active_lower, active_a, candidate):
    side = str(candidate["side"]).upper()
    xc = float(candidate["x"])

    if side == "UPPER":
        new_upper = sorted(list(active_upper) + [xc])
        new_lower = sorted(list(active_lower))
    elif side == "LOWER":
        new_upper = sorted(list(active_upper))
        new_lower = sorted(list(active_lower) + [xc])
    else:
        raise ValueError(f"Unknown candidate side: {side}")

    a_base = lift_a_to_new_side_centers(
        old_upper=active_upper,
        old_lower=active_lower,
        old_a=active_a,
        new_upper=new_upper,
        new_lower=new_lower,
    )

    if side == "UPPER":
        new_idx = new_upper.index(xc)
    else:
        new_idx = len(new_upper) + new_lower.index(xc)

    return side, xc, new_upper, new_lower, a_base, new_idx


def _make_candidate_objective(
    x,
    yu_init,
    yl_init,
    cp_target,
    new_upper,
    new_lower,
    current_best_error,
    workdir,
    side,
    xc,
):
    return make_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=new_upper,
        lower_centers=new_lower,
        hh_power=SETTINGS["optimization"]["hh_power"],
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / f"score_{side}_{xc:.6f}",
        current_best_error=current_best_error,
    )


def _evaluate_objective_state(objective, a):
    objective(np.asarray(a, dtype=float))
    return objective.eval_history[-1]


def _adaptive_fd_step(a_value, rel_step, abs_step_floor):
    return max(float(rel_step) * abs(float(a_value)), float(abs_step_floor))


def _score_fd_component(
    objective,
    a_base,
    idx,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    aj = float(a_base[idx])
    lo, hi = bounds[idx]
    h = _adaptive_fd_step(aj, rel_step, abs_step_floor)

    room_plus = max(0.0, hi - aj)
    room_minus = max(0.0, aj - lo)

    if room_plus >= h and room_minus >= h:
        a_p = a_base.copy()
        a_m = a_base.copy()
        a_p[idx] += h
        a_m[idx] -= h

        item_p = _evaluate_objective_state(objective, a_p)
        item_m = _evaluate_objective_state(objective, a_m)

        if item_p.get("status") != "OK" or item_m.get("status") != "OK":
            return {"grad": 0.0, "mode": "CENTRAL_FAIL", "h": h}

        j_p = item_p.get("objective_base", item_p["objective"])
        j_m = item_m.get("objective_base", item_m["objective"])
        g = (j_p - j_m) / (2.0 * h)
        return {"grad": float(g), "mode": "CENTRAL", "h": h}

    if room_plus > 0.0:
        h_fwd = min(h, room_plus)
        a_p = a_base.copy()
        a_p[idx] += h_fwd

        item_p = _evaluate_objective_state(objective, a_p)

        if base_item.get("status") != "OK" or item_p.get("status") != "OK" or h_fwd <= 0.0:
            return {"grad": 0.0, "mode": "FWD_FAIL", "h": h_fwd}

        j_0 = base_item.get("objective_base", base_item["objective"])
        j_p = item_p.get("objective_base", item_p["objective"])
        g = (j_p - j_0) / h_fwd
        return {"grad": float(g), "mode": "FORWARD", "h": h_fwd}

    if room_minus > 0.0:
        h_bwd = min(h, room_minus)
        a_m = a_base.copy()
        a_m[idx] -= h_bwd

        item_m = _evaluate_objective_state(objective, a_m)

        if base_item.get("status") != "OK" or item_m.get("status") != "OK" or h_bwd <= 0.0:
            return {"grad": 0.0, "mode": "BWD_FAIL", "h": h_bwd}

        j_0 = base_item.get("objective_base", base_item["objective"])
        j_m = item_m.get("objective_base", item_m["objective"])
        g = (j_0 - j_m) / h_bwd
        return {"grad": float(g), "mode": "BACKWARD", "h": h_bwd}

    return {"grad": 0.0, "mode": "NO_ROOM", "h": 0.0}


def _compute_full_objective_gradient(
    objective,
    a_base,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    grad_j = np.zeros(ndv)
    n_fail_dirs = 0
    fail_indices = []

    for j in range(ndv):
        out = _score_fd_component(
            objective=objective,
            a_base=a_base,
            idx=j,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
            base_item=base_item,
        )
        grad_j[j] = float(out["grad"])
        if out["mode"] in ("CENTRAL_FAIL", "FWD_FAIL", "BWD_FAIL", "NO_ROOM"):
            n_fail_dirs += 1
            fail_indices.append(j)

    diag = {"n_fail_dirs": n_fail_dirs, "fail_indices": fail_indices, "ndv": ndv}
    return grad_j, diag


def _rebuild_geometry_from_a(x, yu_init, yl_init, upper_centers, lower_centers, a):
    nu = len(upper_centers)
    nl = len(lower_centers)
    a_upper = np.asarray(a[:nu], dtype=float)
    a_lower = np.asarray(a[nu:nu + nl], dtype=float)

    yu, yl = apply_hicks_henne_deformation(
        x=x,
        yu_base=yu_init,
        yl_base=yl_init,
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        power=SETTINGS["optimization"]["hh_power"],
    )
    return yu, yl


def _compute_full_aero_gradients(
    objective,
    a_base,
    enabled_metric_names,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    grad_j = np.zeros(ndv)
    grad_metrics = {name: np.zeros(ndv) for name in enabled_metric_names}
    n_fail_dirs = 0
    fail_indices = []

    for j in range(ndv):
        aj = float(a_base[j])
        lo, hi = bounds[j]
        h = _adaptive_fd_step(aj, rel_step, abs_step_floor)

        room_plus = max(0.0, hi - aj)
        room_minus = max(0.0, aj - lo)

        if room_plus >= h and room_minus >= h:
            a_p = a_base.copy()
            a_m = a_base.copy()
            a_p[j] += h
            a_m[j] -= h

            item_p = _evaluate_objective_state(objective, a_p)
            item_m = _evaluate_objective_state(objective, a_m)

            if item_p.get("status") != "OK" or item_m.get("status") != "OK":
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_p = item_p.get("objective_base", item_p["objective"])
            j_m = item_m.get("objective_base", item_m["objective"])
            grad_j[j] = (j_p - j_m) / (2.0 * h)

            metrics_p = item_p.get("metrics", {})
            metrics_m = item_m.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_p[name]) - float(metrics_m[name])) / (2.0 * h)
            continue

        if room_plus > 0.0:
            h_fwd = min(h, room_plus)
            a_p = a_base.copy()
            a_p[j] += h_fwd

            item_0 = base_item
            item_p = _evaluate_objective_state(objective, a_p)

            if item_0.get("status") != "OK" or item_p.get("status") != "OK" or h_fwd <= 0.0:
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_0 = item_0.get("objective_base", item_0["objective"])
            j_p = item_p.get("objective_base", item_p["objective"])
            grad_j[j] = (j_p - j_0) / h_fwd

            metrics_0 = item_0.get("metrics", {})
            metrics_p = item_p.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_p[name]) - float(metrics_0[name])) / h_fwd
            continue

        if room_minus > 0.0:
            h_bwd = min(h, room_minus)
            a_m = a_base.copy()
            a_m[j] -= h_bwd

            item_0 = base_item
            item_m = _evaluate_objective_state(objective, a_m)

            if item_0.get("status") != "OK" or item_m.get("status") != "OK" or h_bwd <= 0.0:
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_0 = item_0.get("objective_base", item_0["objective"])
            j_m = item_m.get("objective_base", item_m["objective"])
            grad_j[j] = (j_0 - j_m) / h_bwd

            metrics_0 = item_0.get("metrics", {})
            metrics_m = item_m.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_0[name]) - float(metrics_m[name])) / h_bwd
            continue

        n_fail_dirs += 1
        fail_indices.append(j)
        grad_j[j] = 0.0
        for name in enabled_metric_names:
            grad_metrics[name][j] = 0.0

    diag = {"n_fail_dirs": n_fail_dirs, "fail_indices": fail_indices, "ndv": ndv}
    return grad_j, grad_metrics, diag


def _compute_geometric_gradients_analytic(x, upper_centers, lower_centers):
    grads = {}

    if len(upper_centers) > 0:
        basis_upper = build_hicks_henne_basis(
            x, upper_centers, power=SETTINGS["optimization"]["hh_power"]
        )
        area_upper = np.array([np.trapezoid(phi, x) for phi in basis_upper], dtype=float)
    else:
        area_upper = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        basis_lower = build_hicks_henne_basis(
            x, lower_centers, power=SETTINGS["optimization"]["hh_power"]
        )
        area_lower = -np.array([np.trapezoid(phi, x) for phi in basis_lower], dtype=float)
    else:
        area_lower = np.zeros(0, dtype=float)

    grads["area"] = np.concatenate([area_upper, area_lower])
    return grads


def _compute_thickness_station_gradient(x, upper_centers, lower_centers, x_station):
    x_station = float(x_station)

    if len(upper_centers) > 0:
        phi_u = build_hicks_henne_basis(
            np.array([x_station], dtype=float),
            upper_centers,
            power=SETTINGS["optimization"]["hh_power"],
        )[:, 0]
    else:
        phi_u = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        phi_l = -build_hicks_henne_basis(
            np.array([x_station], dtype=float),
            lower_centers,
            power=SETTINGS["optimization"]["hh_power"],
        )[:, 0]
    else:
        phi_l = np.zeros(0, dtype=float)

    return np.concatenate([phi_u, phi_l])


def _constraint_gradient_from_metric_grad(metric_grad, kind):
    if kind in ("eq", "le"):
        return np.asarray(metric_grad, dtype=float)
    if kind == "ge":
        return -np.asarray(metric_grad, dtype=float)
    raise ValueError(f"Unknown constraint kind: {kind}")


def _build_active_ikkt_system(
    x,
    yu,
    yl,
    metrics,
    upper_centers,
    lower_centers,
    grad_metrics_aero,
    grad_geom_base,
):
    tol_active = float(SETTINGS["optimization"]["adaptive"].get("tol_active", 0.0))
    all_entries = build_all_signed_constraints(
        SETTINGS.get("constraints", {}),
        metrics,
        x,
        yu,
        yl,
        tol_active=tol_active,
    )

    active_entries = []
    grad_columns = []

    for entry in all_entries:
        if entry["name"] in IKKT_EXCLUDED_NAMES:
            continue
        if not entry.get("active", False):
            continue

        name = entry["name"]
        kind = entry["kind"]

        if name in ("CL", "CD", "CM"):
            metric_grad = grad_metrics_aero.get(name)
            if metric_grad is None:
                continue
            c_grad = _constraint_gradient_from_metric_grad(metric_grad, kind)
        elif name == "area":
            c_grad = _constraint_gradient_from_metric_grad(grad_geom_base["area"], kind)
        elif name == "thickness_stations":
            xs = float(entry["x"])
            t_grad = _compute_thickness_station_gradient(x, upper_centers, lower_centers, xs)
            c_grad = _constraint_gradient_from_metric_grad(t_grad, kind)
        else:
            continue

        active_entries.append(entry)
        grad_columns.append(np.asarray(c_grad, dtype=float))

    if len(grad_columns) == 0:
        return active_entries, None

    return active_entries, np.column_stack(grad_columns)


def _solve_bounded_least_squares(grad_j, G, active_entries):
    m = G.shape[1]
    bounds = []

    for entry in active_entries:
        if entry["kind"] == "eq":
            bounds.append((None, None))
        else:
            bounds.append((0.0, None))

    def obj(lam):
        r = grad_j - G @ lam
        return 0.5 * float(np.dot(r, r))

    x0 = np.zeros(m, dtype=float)
    result = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

    if not result.success:
        return np.zeros(m, dtype=float)
    return np.asarray(result.x, dtype=float)


def _score_candidate_grad(
    objective,
    a_base,
    new_idx,
    bounds,
    rel_step,
    abs_step_floor,
    base_item,
):
    grad_j, fd_diag = _compute_full_objective_gradient(
        objective=objective,
        a_base=a_base,
        bounds=bounds,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
        base_item=base_item,
    )

    score = float(np.linalg.norm(grad_j))
    gnew = float(grad_j[new_idx])
    return {
        "score": score,
        "component": gnew,
        "raw_grad": gnew,
        "grad_j": grad_j,
        "fd_diag": fd_diag,
        "mode": "GRAD",
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

    grad_j, grad_metrics_aero, fd_diag = _compute_full_aero_gradients(
        objective=objective,
        a_base=a_base,
        enabled_metric_names=enabled_aero,
        bounds=bounds,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
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
            "mode": "NO_ACTIVE",
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
        "mode": "IKKT",
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
):
    side, xc, new_upper, new_lower, a_base, new_idx = _build_extended_space(
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

    indicator = str(indicator).upper()

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * len(a_base)
    rel_step = SETTINGS["optimization"]["fd_rel_step"]
    abs_step_floor = SETTINGS["optimization"]["fd_abs_step_floor"]

    base_item = _evaluate_objective_state(objective, a_base)

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
            base_item=base_item,
        )
        active_names = [
            f"{e['name']}@{e['x']:.3f}" if e["name"] == "thickness_stations" else e["name"]
            for e in out["active_entries"]
        ]

        msg = (
            f"SCORE[{indicator}] side={side} candidate={xc:.6f}  "
            f"mode={out['mode']}  "
            f"r_norm={out['score']:.6e}  "
            f"r_new={out['component']:.6e}  "
            f"gnew={out['grad_j'][new_idx]:.6e}  "
            f"n_active={len(out['active_entries'])}  "
            f"evals={objective.eval_counter['k']}"
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
            "n_evals": objective.eval_counter["k"],
            "indicator": indicator,
            "mode": out["mode"],
        }

    out = _score_candidate_grad(
        objective=objective,
        a_base=a_base,
        new_idx=new_idx,
        bounds=bounds,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
        base_item=base_item,
    )

    print(
        f"SCORE[{indicator}] side={side} candidate={xc:.6f}  "
        f"g_norm={out['score']:.6e}  "
        f"g_new={out['component']:.6e}  "
        f"n_fail_dirs={out['fd_diag']['n_fail_dirs']}  "
        f"evals={objective.eval_counter['k']}"
    )

    return {
        "side": side,
        "x": xc,
        "score": out["score"],
        "component": out["component"],
        "raw_grad": out["raw_grad"],
        "active_names": [],
        "lambda": np.zeros(0),
        "n_evals": objective.eval_counter["k"],
        "indicator": indicator,
        "mode": "GRAD",
    }


def _print_candidate_diagnostics(scored, topk=12):
    if len(scored) == 0:
        return

    print("===== CANDIDATE RANKING (TOP) =====")
    for i, item in enumerate(scored[:topk], start=1):
        lam_str = np.array2string(item["lambda"], precision=3, suppress_small=False) if len(item["lambda"]) > 0 else "[]"
        print(
            f"{i:02d} | side={item['side']:<5s} "
            f"x={item['x']:.6f}  "
            f"score={item['score']:.6e}  "
            f"component={item['component']:.6e}  "
            f"raw={item['raw_grad']:.6e}  "
            f"mode={item['mode']}  "
            f"active={item['active_names']}  "
            f"lambda={lam_str}"
        )

    print("===== SAME-X PAIRS =====")

    grouped = {}
    for item in scored:
        key = round(float(item["x"]), 6)
        grouped.setdefault(key, []).append(item)

    found_pair = False
    for key in sorted(grouped):
        items = grouped[key]
        if len(items) < 2:
            continue

        found_pair = True
        items = sorted(items, key=lambda z: z["side"])

        line = f"x={key:.6f}"
        print(line)

        for item in items:
            line = (
                f"  side={item['side']:<5s} "
                f"score={item['score']:.6e}  "
                f"component={item['component']:.6e}  "
                f"raw={item['raw_grad']:.6e}"
            )
            print(line)

    if not found_pair:
        print("No same-x upper/lower pairs in current candidate set.")


def run_adaptive_strategy(
    x,
    yu_init,
    yl_init,
    cp_target,
    initial_error,
    workdir,
    indicator,
):
    opt_ad = SETTINGS["optimization"]["adaptive"]

    n0 = opt_ad["n0"]
    n_final = opt_ad["n_final"]

    history_evals = []
    history_full = []
    upper_centers, lower_centers = build_side_specific_initial_centers(n0)

    a0 = np.zeros(len(upper_centers) + len(lower_centers))
    history = []
    center_history = []
    n_scoring_evals_total = 0
    n_optimization_evals_total = 0
    n_xfoil_calls_total = 0
    current_best_error = float(initial_error)

    adaptive_out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a0=a0,
        label="adaptive_level_0",
        current_best_error=current_best_error,
        workdir=Path(workdir) / "adaptive",
    )

    current_best_error = adaptive_out["err_opt"]

    n_optimization_evals_total += adaptive_out["n_objective_evals"]
    n_xfoil_calls_total += adaptive_out["n_xfoil_calls_total"]

    history.append((adaptive_out["ndv_total"], adaptive_out["err_opt"]))
    history_evals.append(n_xfoil_calls_total)

    history_full.append(
        {
            "ndv_total": adaptive_out["ndv_total"],
            "eval_offset_end": n_xfoil_calls_total,
            "objective_history": [dict(item) for item in adaptive_out["objective_history"]],
        }
    )

    center_history.append(
        {
            "ndv_total": adaptive_out["ndv_total"],
            "upper": list(upper_centers),
            "lower": list(lower_centers),
        }
    )

    best_adaptive = {
        "err_opt": adaptive_out["err_opt"],
        "ndv_total": adaptive_out["ndv_total"],
        "a_opt": np.array(adaptive_out["a_opt"], copy=True),
        "yu_opt": np.array(adaptive_out["yu_opt"], copy=True),
        "yl_opt": np.array(adaptive_out["yl_opt"], copy=True),
        "cp_opt": {
            "upper": {
                "x": np.array(adaptive_out["cp_opt"]["upper"]["x"], copy=True),
                "cp": np.array(adaptive_out["cp_opt"]["upper"]["cp"], copy=True),
            },
            "lower": {
                "x": np.array(adaptive_out["cp_opt"]["lower"]["x"], copy=True),
                "cp": np.array(adaptive_out["cp_opt"]["lower"]["cp"], copy=True),
            },
        },
        "opt_res": {
            "polar": dict(adaptive_out["opt_res"]["polar"]),
            "cp_data": {
                "x": np.array(adaptive_out["opt_res"]["cp_data"]["x"], copy=True),
                "cp": np.array(adaptive_out["opt_res"]["cp_data"]["cp"], copy=True),
            },
        },
        "upper_centers": np.array(adaptive_out["upper_centers"], copy=True),
        "lower_centers": np.array(adaptive_out["lower_centers"], copy=True),
        "label": "adaptive_level_0",
    }

    while (len(upper_centers) + len(lower_centers)) < n_final:
        cand_upper_raw = get_midpoint_candidates(upper_centers)
        cand_lower_raw = get_midpoint_candidates(lower_centers)

        candidates = []
        for c in cand_upper_raw:
            candidates.append({"side": "UPPER", "x": float(c["x"]), "interval_id": c["interval_id"]})
        for c in cand_lower_raw:
            candidates.append({"side": "LOWER", "x": float(c["x"]), "interval_id": c["interval_id"]})

        if len(candidates) == 0:
            break

        scored = []
        for cand in candidates:
            info = score_candidate(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                active_upper=upper_centers,
                active_lower=lower_centers,
                active_a=adaptive_out["a_opt"],
                candidate=cand,
                current_best_error=current_best_error,
                workdir=Path(workdir) / "adaptive_candidate_scoring",
                indicator=indicator,
            )
            n_scoring_evals_total += info["n_evals"]
            n_xfoil_calls_total += info["n_evals"]

            scored.append(
                {
                    "side": info["side"],
                    "x": info["x"],
                    "interval_id": cand["interval_id"],
                    "score": info["score"],
                    "component": info["component"],
                    "raw_grad": info["raw_grad"],
                    "active_names": list(info["active_names"]),
                    "lambda": np.array(info["lambda"], copy=True),
                    "mode": info["mode"],
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["x"]))
        _print_candidate_diagnostics(scored, topk=12)

        current_ndv = len(upper_centers) + len(lower_centers)
        n_remaining = n_final - current_ndv
        n_add = _compute_adaptive_nadd(current_ndv, len(scored))
        n_add = min(n_add, n_remaining)

        if n_add <= 0:
            break

        chosen = scored[:n_add]

        print("===== ADAPTIVE REFINE =====")
        print(f"current ndv       = {current_ndv}")
        print(f"growth ratio      = {SETTINGS['optimization']['adaptive']['growth_ratio']:.6f}")
        print(f"current best err  = {current_best_error:.6e}")
        print(f"chosen candidates = {chosen}")

        new_upper = sorted(list(upper_centers))
        new_lower = sorted(list(lower_centers))

        for cand in chosen:
            if cand["side"] == "UPPER":
                new_upper.append(float(cand["x"]))
            else:
                new_lower.append(float(cand["x"]))

        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))

        new_a0 = lift_a_to_new_side_centers(
            old_upper=upper_centers,
            old_lower=lower_centers,
            old_a=adaptive_out["a_opt"],
            new_upper=new_upper,
            new_lower=new_lower,
        )

        level_label = f"adaptive_level_{len(new_upper) + len(new_lower)}"

        adaptive_out = optimize_for_centers(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a0=new_a0,
            label=level_label,
            current_best_error=current_best_error,
            workdir=Path(workdir) / "adaptive",
        )

        current_best_error = adaptive_out["err_opt"]

        n_optimization_evals_total += adaptive_out["n_objective_evals"]
        n_xfoil_calls_total += adaptive_out["n_xfoil_calls_total"]

        upper_centers = new_upper
        lower_centers = new_lower

        history.append((adaptive_out["ndv_total"], adaptive_out["err_opt"]))
        history_evals.append(n_xfoil_calls_total)

        history_full.append(
            {
                "ndv_total": adaptive_out["ndv_total"],
                "eval_offset_end": n_xfoil_calls_total,
                "objective_history": [dict(item) for item in adaptive_out["objective_history"]],
            }
        )

        center_history.append(
            {
                "ndv_total": adaptive_out["ndv_total"],
                "upper": list(upper_centers),
                "lower": list(lower_centers),
            }
        )

        if adaptive_out["err_opt"] < best_adaptive["err_opt"]:
            best_adaptive = {
                "err_opt": adaptive_out["err_opt"],
                "ndv_total": adaptive_out["ndv_total"],
                "a_opt": np.array(adaptive_out["a_opt"], copy=True),
                "yu_opt": np.array(adaptive_out["yu_opt"], copy=True),
                "yl_opt": np.array(adaptive_out["yl_opt"], copy=True),
                "cp_opt": {
                    "upper": {
                        "x": np.array(adaptive_out["cp_opt"]["upper"]["x"], copy=True),
                        "cp": np.array(adaptive_out["cp_opt"]["upper"]["cp"], copy=True),
                    },
                    "lower": {
                        "x": np.array(adaptive_out["cp_opt"]["lower"]["x"], copy=True),
                        "cp": np.array(adaptive_out["cp_opt"]["lower"]["cp"], copy=True),
                    },
                },
                "opt_res": {
                    "polar": dict(adaptive_out["opt_res"]["polar"]),
                    "cp_data": {
                        "x": np.array(adaptive_out["opt_res"]["cp_data"]["x"], copy=True),
                        "cp": np.array(adaptive_out["opt_res"]["cp_data"]["cp"], copy=True),
                    },
                },
                "upper_centers": np.array(adaptive_out["upper_centers"], copy=True),
                "lower_centers": np.array(adaptive_out["lower_centers"], copy=True),
                "label": level_label,
            }

    adaptive_out["history"] = history
    adaptive_out["history_evals"] = history_evals
    adaptive_out["history_full"] = history_full
    adaptive_out["center_history"] = center_history
    adaptive_out["n_scoring_evals_total"] = n_scoring_evals_total
    adaptive_out["n_optimization_evals_total"] = n_optimization_evals_total
    adaptive_out["n_total_evals"] = n_xfoil_calls_total
    adaptive_out["best_err_opt"] = best_adaptive["err_opt"]
    adaptive_out["best_ndv_total"] = best_adaptive["ndv_total"]
    adaptive_out["best_a_opt"] = best_adaptive["a_opt"]
    adaptive_out["best_yu_opt"] = best_adaptive["yu_opt"]
    adaptive_out["best_yl_opt"] = best_adaptive["yl_opt"]
    adaptive_out["best_cp_opt"] = best_adaptive["cp_opt"]
    adaptive_out["best_opt_res"] = best_adaptive["opt_res"]
    adaptive_out["best_upper_centers"] = best_adaptive["upper_centers"]
    adaptive_out["best_lower_centers"] = best_adaptive["lower_centers"]
    adaptive_out["best_label"] = best_adaptive["label"]

    return adaptive_out
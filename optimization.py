from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from xfoil_wrapper import run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from objective import make_objective, total_cp_error
from constraints import build_slsqp_geometric_constraints


def _status_of_last_eval(objective):
    if not hasattr(objective, "eval_history") or len(objective.eval_history) == 0:
        return None
    return objective.eval_history[-1].get("status")


def _evaluate_with_status(objective, a_vec):
    value = objective(np.asarray(a_vec, dtype=float))
    status = _status_of_last_eval(objective)
    return float(value), status


def _compute_explicit_jac(
    objective,
    a,
    bounds,
    rel_step=1.0e-2,
    abs_step_floor=2.0e-5,
):
    """
    Explicit finite-difference Jacobian for SLSQP.

    Step logic:
    - nominal step h_j = max(rel_step * abs(a_j), abs_step_floor)
    - clipped to bounds
    - central FD when possible
    - one-sided FD near bounds
    - if XFOIL/geometry fails on required evals, gradient component -> 0
    """
    a = np.asarray(a, dtype=float)
    g = np.zeros_like(a, dtype=float)

    for j in range(len(a)):
        aj = float(a[j])
        lo, hi = bounds[j]

        h = max(rel_step * abs(aj), abs_step_floor)

        room_plus = max(0.0, hi - aj)
        room_minus = max(0.0, aj - lo)

        # Prefer central difference if both sides available
        if room_plus >= h and room_minus >= h:
            a_p = a.copy()
            a_m = a.copy()
            a_p[j] += h
            a_m[j] -= h

            Jp, sp = _evaluate_with_status(objective, a_p)
            Jm, sm = _evaluate_with_status(objective, a_m)

            if sp == "OK" and sm == "OK":
                g[j] = (Jp - Jm) / (2.0 * h)
            else:
                g[j] = 0.0
            continue

        # Forward one-sided if only plus side available
        if room_plus > 0.0:
            h_fwd = min(h, room_plus)
            a_p = a.copy()
            a_p[j] += h_fwd

            J0, s0 = _evaluate_with_status(objective, a)
            Jp, sp = _evaluate_with_status(objective, a_p)

            if s0 == "OK" and sp == "OK" and h_fwd > 0.0:
                g[j] = (Jp - J0) / h_fwd
            else:
                g[j] = 0.0
            continue

        # Backward one-sided if only minus side available
        if room_minus > 0.0:
            h_bwd = min(h, room_minus)
            a_m = a.copy()
            a_m[j] -= h_bwd

            J0, s0 = _evaluate_with_status(objective, a)
            Jm, sm = _evaluate_with_status(objective, a_m)

            if s0 == "OK" and sm == "OK" and h_bwd > 0.0:
                g[j] = (J0 - Jm) / h_bwd
            else:
                g[j] = 0.0
            continue

        # No room at all (degenerate bound case)
        g[j] = 0.0

    return g


def optimize_for_centers(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a0,
    label,
    current_best_error,
    workdir,
):
    hh_power = SETTINGS["optimization"]["hh_power"]

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    objective = make_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / label,
        current_best_error=current_best_error,
    )

    geometric_constraints = build_slsqp_geometric_constraints(
        settings_constraints=SETTINGS.get("constraints", {}),
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * (len(upper_centers) + len(lower_centers))
    a0 = np.asarray(a0, dtype=float)

    # Finite-difference settings for explicit Jacobian
    jac_rel_step = SETTINGS["optimization"]["fd_rel_step"]
    jac_abs_step_floor = SETTINGS["optimization"]["fd_abs_step_floor"]

    def jac_explicit(a_vec):
        return _compute_explicit_jac(
            objective=objective,
            a=a_vec,
            bounds=bounds,
            rel_step=jac_rel_step,
            abs_step_floor=jac_abs_step_floor,
        )

    # Gradient snapshots for debug/scaling analysis
    def _save_gradient_snapshot(tag, a_vec):
        if not hasattr(objective, "compute_gradient_snapshot"):
            return
        if not hasattr(objective, "append_grad_history"):
            return

        print(f"\n>>> Computing gradient snapshot: {tag}")
        g_vec = objective.compute_gradient_snapshot(np.asarray(a_vec, dtype=float))
        objective.append_grad_history(objective.eval_counter["k"], g_vec)

        grad_norm = np.linalg.norm(g_vec)
        grad_max = np.max(np.abs(g_vec)) if len(g_vec) > 0 else 0.0
        print(f">>> Gradient snapshot saved: {tag}")
        print(f">>> ||g||_2 = {grad_norm:.6e}")
        print(f">>> max|g|  = {grad_max:.6e}")

    print(f"\n===== OBJECTIVE CHECK AT a0 ({label}) =====")
    j0 = objective(a0)
    print(f"Initial objective value at a0 = {j0:.6e}")
    print(f"Dynamic penalty value         = {objective.get_penalty():.6e}")
    print(f"Geometric SLSQP constraints   = {len(geometric_constraints)}")
    print("Jacobian mode                 = EXPLICIT_FD")
    print(f"jac_rel_step                  = {jac_rel_step:.6e}")
    print(f"jac_abs_step_floor            = {jac_abs_step_floor:.6e}")
    _save_gradient_snapshot(f"{label}_initial", a0)

    result = minimize(
        objective,
        a0,
        method="SLSQP",
        jac=jac_explicit,
        bounds=bounds,
        constraints=geometric_constraints,
        options={
            "maxiter": SETTINGS["optimization"]["maxiter"],
            "ftol": SETTINGS["optimization"]["ftol"],
            "disp": True,
        },
    )

    print(f"\n===== OPTIMIZATION RESULT ({label}) =====")
    print("success :", result.success)
    print("status  :", result.status)
    print("message :", result.message)
    print("fun     :", result.fun)
    print("nit     :", result.nit)

    a_opt = np.asarray(result.x, dtype=float)
    _save_gradient_snapshot(f"{label}_final", a_opt)

    nu = len(upper_centers)
    nl = len(lower_centers)
    a_upper = np.asarray(a_opt[:nu], dtype=float)
    a_lower = np.asarray(a_opt[nu:nu + nl], dtype=float)

    yu_opt, yl_opt = apply_hicks_henne_deformation(
        x=x,
        yu_base=yu_init,
        yl_base=yl_init,
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        power=hh_power,
    )

    opt_dat = Path(workdir) / f"{label}_optimized_airfoil.dat"
    write_dat(opt_dat, x, yu_opt, yl_opt, name=f"{label.upper()}_OPTIMIZED")

    opt_res = run_xfoil(
        airfoil_dat=opt_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / f"{label}_optimized_run",
    )

    if not opt_res["success"]:
        print(opt_res["stdout"])
        print(opt_res["stderr"])
        raise RuntimeError(f"Optimized XFOIL run failed for {label}.")

    cp_opt = split_upper_lower_cp_from_x(
        opt_res["cp_data"]["x"],
        opt_res["cp_data"]["cp"],
    )

    err_opt = total_cp_error(cp_target, cp_opt)

    n_objective_evals = objective.eval_counter["k"]
    n_xfoil_calls_total = n_objective_evals + 1

    return {
        "upper_centers": np.asarray(upper_centers, dtype=float),
        "lower_centers": np.asarray(lower_centers, dtype=float),
        "a_opt": a_opt,
        "yu_opt": yu_opt,
        "yl_opt": yl_opt,
        "cp_opt": cp_opt,
        "opt_res": opt_res,
        "err_opt": err_opt,
        "result": result,
        "n_objective_evals": n_objective_evals,
        "n_xfoil_calls_total": n_xfoil_calls_total,
        "objective_history": list(objective.eval_history),
        "ndv_total": len(upper_centers) + len(lower_centers),
        "penalty_value": objective.get_penalty(),
        "n_geometric_constraints": len(geometric_constraints),
    }
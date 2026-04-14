from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from xfoil_wrapper import run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from objective import make_objective, total_cp_error
from constraints import build_slsqp_geometric_constraints


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

    print(f"\n===== OBJECTIVE CHECK AT a0 ({label}) =====")
    j0 = objective(a0)
    print(f"Initial objective value at a0 = {j0:.6e}")
    print(f"Dynamic penalty value         = {objective.get_penalty():.6e}")
    print(f"Geometric SLSQP constraints   = {len(geometric_constraints)}")

    result = minimize(
        objective,
        a0,
        method="SLSQP",
        bounds=bounds,
        constraints=geometric_constraints,
        options={
            "maxiter": SETTINGS["optimization"]["maxiter"],
            "ftol": SETTINGS["optimization"]["ftol"],
            "disp": True,
            "eps": SETTINGS["optimization"]["eps"],
        },
    )
    a_opt = result.x
    abs_a = np.abs(a_opt)

    print("\n===== DV SCALE CHECK =====")
    print(f"max |a_opt|  = {np.max(abs_a):.6e}")
    print(f"mean |a_opt|  = {np.mean(abs_a):.6e}")
    print(f"min |a_opt|  = {np.min(abs_a):.6e}")
    print("=============================\n")

    print(f"\n===== OPTIMIZATION RESULT ({label}) =====")
    print("success :", result.success)
    print("status  :", result.status)
    print("message :", result.message)
    print("fun     :", result.fun)
    print("nit     :", result.nit)

    a_opt = result.x

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

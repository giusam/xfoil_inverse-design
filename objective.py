from pathlib import Path

import numpy as np

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from xfoil_wrapper import run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from constraints import compute_metrics, evaluate_constraints, split_constraints_by_domain


def cp_error_interp(x_ref, cp_ref, x_cmp, cp_cmp):
    x_ref = np.asarray(x_ref, dtype=float)
    cp_ref = np.asarray(cp_ref, dtype=float)
    x_cmp = np.asarray(x_cmp, dtype=float)
    cp_cmp = np.asarray(cp_cmp, dtype=float)

    if len(x_ref) > 1 and x_ref[0] > x_ref[-1]:
        x_ref = x_ref[::-1]
        cp_ref = cp_ref[::-1]

    if len(x_cmp) > 1 and x_cmp[0] > x_cmp[-1]:
        x_cmp = x_cmp[::-1]
        cp_cmp = cp_cmp[::-1]

    cp_cmp_interp = np.interp(x_ref, x_cmp, cp_cmp)
    err_sq = (cp_ref - cp_cmp_interp) ** 2

    x_span = x_ref[-1] - x_ref[0]
    if x_span <= 0.0:
        return float(np.mean(err_sq))

    return float(np.trapezoid(err_sq, x_ref) / x_span)


def total_cp_error(cp_target, cp_candidate):
    err_upper = cp_error_interp(
        cp_target["upper"]["x"],
        cp_target["upper"]["cp"],
        cp_candidate["upper"]["x"],
        cp_candidate["upper"]["cp"],
    )

    err_lower = cp_error_interp(
        cp_target["lower"]["x"],
        cp_target["lower"]["cp"],
        cp_candidate["lower"]["x"],
        cp_candidate["lower"]["cp"],
    )

    return 0.5 * (err_upper + err_lower)


def make_objective(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    hh_power,
    alpha_deg,
    reynolds,
    xfoil_iter,
    timeout,
    working_dir,
    current_best_error,
):
    eval_counter = {"k": 0}
    eval_history = []
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    penalty_factor = SETTINGS["optimization"]["penalty_factor"]
    penalty_floor = 1.0e-6
    aero_constraints, _ = split_constraints_by_domain(SETTINGS.get("constraints", {}))

    def get_penalty():
        scale = max(float(current_best_error), penalty_floor)
        return penalty_factor * scale

    def objective(a):
        eval_counter["k"] += 1
        k = eval_counter["k"]

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
            power=hh_power,
        )

        thickness = yu - yl
        min_thickness = np.min(thickness)
        thickness_internal = thickness[1:-1]
        thickness_tol = -1.0e-10

        if np.any(thickness_internal < thickness_tol):
            penalty_value = get_penalty()
            print(
                f"eval={k:04d}  FAIL thickness  "
                f"min_thickness={min_thickness:.6e}  penalty={penalty_value:.6e}"
            )
            eval_history.append(
                {
                    "eval": k,
                    "objective": penalty_value,
                    "status": "FAIL_THICKNESS",
                }
            )
            return penalty_value

        airfoil_dat = working_dir / f"candidate_{k:05d}.dat"
        write_dat(airfoil_dat, x, yu, yl, name=f"CAND_{k:05d}")

        run_dir = working_dir / f"candidate_run_{k:05d}"

        res = run_xfoil(
            airfoil_dat=airfoil_dat,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
            xfoil_iter=xfoil_iter,
            timeout=timeout,
            working_dir=run_dir,
        )
        if not res["success"]:
            penalty_value = get_penalty()
            print(f"eval={k:04d}  FAIL xfoil  penalty={penalty_value:.6e}")

            eval_history.append(
                {
                    "eval": k,
                    "objective": penalty_value,
                    "status": "FAIL_XFOIL",
                }
            )
            return penalty_value

        cp_candidate = split_upper_lower_cp_from_x(
            res["cp_data"]["x"],
            res["cp_data"]["cp"],
        )

        err = total_cp_error(cp_target, cp_candidate)

        metrics = compute_metrics(x, yu, yl, res["polar"])
        constraint_penalty, constraint_details = evaluate_constraints(
            aero_constraints,
            metrics,
            x,
            yu,
            yl,
            current_best_error,
        )

        objective_total = err + constraint_penalty

        print(
            f"eval={k:04d}  OK   "
            f"Jcp={err:.6e}  "
            f"Pcon={constraint_penalty:.6e}  "
            f"Jtot={objective_total:.6e}  "
            f"min_thickness={min_thickness:.6e}"
        )

        eval_history.append(
            {
                "eval": k,
                "objective": objective_total,
                "objective_base": err,
                "constraint_penalty": constraint_penalty,
                "metrics": metrics,
                "constraints": constraint_details,
                "status": "OK",
            }
        )

        return objective_total

    objective.eval_counter = eval_counter
    objective.eval_history = eval_history
    objective.get_penalty = get_penalty
    return objective

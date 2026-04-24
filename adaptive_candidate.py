
from pathlib import Path

import numpy as np

from settings import SETTINGS
from objective import make_objective
from geometry import apply_hicks_henne_deformation
from adaptive_utils import lift_a_to_new_side_centers


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
    nu_old = len(active_upper)
    nl_old = len(active_lower)

    old_to_new = np.zeros(nu_old + nl_old, dtype=int)

    for i, c in enumerate(active_upper):
        old_to_new[i] = new_upper.index(c)

    for i, c in enumerate(active_lower):
        old_to_new[nu_old + i] = len(new_upper) + new_lower.index(c)
    return side, xc, new_upper, new_lower, a_base, new_idx, old_to_new

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
    backend = str(SETTINGS.get("aero", {}).get("backend", "xfoil")).strip().lower()

    if backend == "cmplxfoil":
        # Reuse one shared CMPLXFOIL scoring session instead of creating
        # one solver/session per candidate.
        score_workdir = Path(workdir) / "score_shared"
    else:
        # Keep separate folders for XFOIL/debug output.
        score_workdir = Path(workdir) / f"score_{side}_{xc:.6f}"

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
        working_dir=score_workdir,
        current_best_error=current_best_error,
    )


def _evaluate_objective_state(objective, a):
    objective(np.asarray(a, dtype=float))
    return objective.eval_history[-1]


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



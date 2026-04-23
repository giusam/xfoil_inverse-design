import numpy as np

from geometry import build_normal_peak_fd_steps
from settings import SETTINGS
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

def _score_candidate_grad(
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

    score = float(np.linalg.norm(grad_j))
    gnew = float(grad_j[new_idx])
    return {
        "score": score,
        "component": gnew,
        "raw_grad": gnew,
        "grad_j": grad_j,
        "fd_diag": fd_diag,
        "mode": "GRAD_CS" if deriv_mode == "cs" else "GRAD_FD",
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

    indicator = str(indicator).upper()

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * len(a_base)
    rel_step = 0.0
    abs_step_floor = 0.0

    base_item = _evaluate_objective_state(objective, a_base)
    step_vector = None

    if indicator in ("GRAD", "IKKT"):
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

        print(
            f"SCORE[{indicator}] side={side} candidate={xc:.6f}  "
            f"delta_pred={out['score']:.6e}  "
            f"d_new={out['component']:.6e}  "
            f"gnew={out['raw_grad']:.6e}  "
            f"tau={out.get('tau', 1.0):.6e}  "
            f"n_active={len(out['active_entries'])}  "
            f"evals={objective.eval_counter['k']}  "
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
            "n_evals": objective.eval_counter["k"],
            "indicator": indicator,
            "mode": out["mode"],
            "tau": out.get("tau", 1.0),
        }

    out = _score_candidate_grad(
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
        f"SCORE[{indicator}] side={side} candidate={xc:.6f}  "
        f"g_norm={out['score']:.6e}  "
        f"g_new={out['component']:.6e}  "
        f"mode={out['mode']}  "
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
        "mode": out["mode"],
    }

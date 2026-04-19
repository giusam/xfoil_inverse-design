from pathlib import Path

import numpy as np

from settings import SETTINGS
from optimization import optimize_for_centers
from adaptive_utils import (
    build_side_specific_initial_centers,
    get_midpoint_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)
from adaptive_scoring import score_candidate
from adaptive_pred import _prepare_pred_level_context

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

        print(f"x={key:.6f}")

        for item in items:
            print(
                f"  side={item['side']:<5s} "
                f"score={item['score']:.6e}  "
                f"component={item['component']:.6e}  "
                f"raw={item['raw_grad']:.6e}"
            )

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

        pred_context = None
        if str(indicator).upper() == "PRED":
            pred_context = _prepare_pred_level_context(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                upper_centers=upper_centers,
                lower_centers=lower_centers,
                a_opt=np.asarray(adaptive_out["a_opt"], dtype=float),
                current_best_error=current_best_error,
                workdir=Path(workdir) / "adaptive_candidate_scoring",
            )

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
                pred_context=pred_context,
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
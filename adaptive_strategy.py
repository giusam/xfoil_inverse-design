from pathlib import Path
import shutil
import numpy as np

from settings import SETTINGS
from optimization import optimize_for_centers
from adaptive_utils import (
    build_side_specific_initial_centers,
    build_interval_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)
from adaptive_scoring import score_candidate
from adaptive_pred import _prepare_pred_level_context


def _save_level_snapshot(
    x,
    yu_init,
    yl_init,
    cp_target,
    adaptive_out,
    label,
    workdir,
):
    snapshots_cfg = SETTINGS.get("snapshots", {})
    if not bool(snapshots_cfg.get("enabled", False)):
        return None

    snapshot_dir = Path(workdir) / str(snapshots_cfg.get("dir_name", "snapshots"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ndv_total = int(adaptive_out["ndv_total"])
    snapshot_path = snapshot_dir / f"level_{ndv_total:03d}.npz"

    payload = {
        "ndv_total": np.asarray(ndv_total),
        "x": np.asarray(x, dtype=float),
        "yu_init": np.asarray(yu_init, dtype=float),
        "yl_init": np.asarray(yl_init, dtype=float),
        "upper_centers": np.asarray(adaptive_out["upper_centers"], dtype=float),
        "lower_centers": np.asarray(adaptive_out["lower_centers"], dtype=float),
        "a_opt": np.asarray(adaptive_out["a_opt"], dtype=float),
        "err_opt": np.asarray(float(adaptive_out["err_opt"])),
        "yu_opt": np.asarray(adaptive_out["yu_opt"], dtype=float),
        "yl_opt": np.asarray(adaptive_out["yl_opt"], dtype=float),
        "cp_target_upper_x": np.asarray(cp_target["upper"]["x"], dtype=float),
        "cp_target_upper_cp": np.asarray(cp_target["upper"]["cp"], dtype=float),
        "cp_target_lower_x": np.asarray(cp_target["lower"]["x"], dtype=float),
        "cp_target_lower_cp": np.asarray(cp_target["lower"]["cp"], dtype=float),
        "cp_opt_upper_x": np.asarray(adaptive_out["cp_opt"]["upper"]["x"], dtype=float),
        "cp_opt_upper_cp": np.asarray(adaptive_out["cp_opt"]["upper"]["cp"], dtype=float),
        "cp_opt_lower_x": np.asarray(adaptive_out["cp_opt"]["lower"]["x"], dtype=float),
        "cp_opt_lower_cp": np.asarray(adaptive_out["cp_opt"]["lower"]["cp"], dtype=float),
        "label": np.asarray(str(label)),
    }

    for metric_name in ("CL", "CD", "CM"):
        target_value = SETTINGS.get("constraints", {}).get(metric_name, {}).get("target")
        if target_value is None:
            payload[f"constraint_{metric_name}_target"] = np.asarray(np.nan)
        else:
            payload[f"constraint_{metric_name}_target"] = np.asarray(float(target_value))

    np.savez(snapshot_path, **payload)
    return snapshot_path


def _print_candidate_diagnostics(scored, topk=12):
    if len(scored) == 0:
        return

    print("===== CANDIDATE RANKING (TOP) =====")
    for i, item in enumerate(scored[:topk], start=1):
        lam_str = np.array2string(item["lambda"], precision=3, suppress_small=False) if len(item["lambda"]) > 0 else "[]"
        interval_label = str(item.get("interval_label", item.get("interval_id", "?")))
        fraction = item.get("local_fraction")
        fraction_str = f"{float(fraction):.2f}" if fraction is not None else "-"
        print(
            f"{i:02d} | side={item['side']:<5s} "
            f"interval={interval_label:<10s} "
            f"frac={fraction_str:<4s}  "
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


def _candidate_beats_interval_best(candidate, current_best):
    if current_best is None:
        return True

    if candidate["score"] > current_best["score"]:
        return True
    if candidate["score"] < current_best["score"]:
        return False

    candidate_midpoint = abs(float(candidate.get("local_fraction", np.nan)) - 0.5) <= 1.0e-12
    current_best_midpoint = abs(float(current_best.get("local_fraction", np.nan)) - 0.5) <= 1.0e-12
    if candidate_midpoint and not current_best_midpoint:
        return True
    if current_best_midpoint and not candidate_midpoint:
        return False

    return False


def reduce_to_best_candidate_per_interval(scored_candidates):
    winners = {}
    encounter_order = []

    for item in scored_candidates:
        interval_key = (item["side"], item["interval_id"])
        if interval_key not in winners:
            winners[interval_key] = item
            encounter_order.append(interval_key)
            continue

        if _candidate_beats_interval_best(item, winners[interval_key]):
            winners[interval_key] = item

    return [winners[key] for key in encounter_order]


def _format_selected_candidate_summary(indicator, candidate):
    interval_label = candidate.get("interval_label", candidate.get("interval_id", "?"))
    return (
        f"[{str(indicator).upper()}] interval={interval_label} "
        f"winner_fraction={float(candidate.get('local_fraction', 0.5)):.2f} "
        f"center={float(candidate['x']):.6f} "
        f"score={float(candidate['score']):.6e}"
    )


def _score_candidate_oracle(
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
    level_ndv,
):
    side = str(candidate["side"]).upper()
    xc = float(candidate["x"])

    new_upper = sorted(list(active_upper))
    new_lower = sorted(list(active_lower))

    if side == "UPPER":
        new_upper.append(xc)
    elif side == "LOWER":
        new_lower.append(xc)
    else:
        raise ValueError(f"Unknown candidate side: {side}")

    new_upper = sorted(set(new_upper))
    new_lower = sorted(set(new_lower))

    new_a0 = lift_a_to_new_side_centers(
        old_upper=active_upper,
        old_lower=active_lower,
        old_a=active_a,
        new_upper=new_upper,
        new_lower=new_lower,
    )

    label = f"oracle_L{int(level_ndv):03d}_{side}_{xc:.6f}".replace(".", "p")
    oracle_workdir = Path(workdir) / "oracle_candidate_scoring" / f"level_{int(level_ndv):03d}" / label


    out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=new_upper,
        lower_centers=new_lower,
        a0=new_a0,
        label=label,
        current_best_error=current_best_error,
        workdir=oracle_workdir,
    )

    delta_real = float(current_best_error - out["err_opt"])
    interval_label = candidate.get("interval_label", candidate.get("interval_id", "?"))
    candidate_context = ""
    if interval_label is not None:
        candidate_context += f"  interval={interval_label}"
    if "local_fraction" in candidate:
        candidate_context += f"  fraction={float(candidate['local_fraction']):.2f}"

    print(
        f"SCORE[ORACLE] side={side} candidate={xc:.6f}{candidate_context}  "
        f"delta_real={delta_real:.6e}  "
        f"err_after={float(out['err_opt']):.6e}  "
        f"score_aero_calls={int(out['n_aero_calls_total_all_phases'])}"
    )

    return {
        "side": side,
        "x": xc,
        "score": delta_real,
        "component": delta_real,
        "raw_grad": np.nan,
        "active_names": [],
        "lambda": np.zeros(0),
        "n_scoring_aero_calls": int(out["n_aero_calls_total_all_phases"]),
        "indicator": "ORACLE",
        "mode": "ORACLE_REAL",
        "err_after": float(out["err_opt"]),
    }

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

    snapshots_cfg = SETTINGS.get("snapshots", {})
    if bool(snapshots_cfg.get("enabled", False)):
        snapshot_dir = Path(workdir) / str(snapshots_cfg.get("dir_name", "snapshots"))
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    n0 = opt_ad["n0"]
    n_final = opt_ad["n_final"]

    history_evals = []
    history_full = []
    history_opt_aero_calls = []
    history_function_evals = []
    history_gradient_evals = []
    history_full_opt = []
    upper_centers, lower_centers = build_side_specific_initial_centers(n0)

    a0 = np.zeros(len(upper_centers) + len(lower_centers))
    history = []
    center_history = []
    selection_history = []
    n_scoring_aero_calls_total = 0
    n_optimization_aero_calls_total = 0
    n_function_evals_total = 0
    n_gradient_evals_total = 0
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
    _save_level_snapshot(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        adaptive_out=adaptive_out,
        label="adaptive_level_0",
        workdir=workdir,
    )

    current_best_error = adaptive_out["err_opt"]

    n_optimization_aero_calls_total += adaptive_out["n_optimization_aero_calls_total"]
    n_function_evals_total += adaptive_out["n_function_evals"]
    n_gradient_evals_total += adaptive_out["n_gradient_evals"]

    history.append((adaptive_out["ndv_total"], adaptive_out["err_opt"]))
    history_evals.append(n_optimization_aero_calls_total)
    history_opt_aero_calls.append(n_optimization_aero_calls_total)
    history_function_evals.append(n_function_evals_total)
    history_gradient_evals.append(n_gradient_evals_total)

    history_full.append(
        {
            "ndv_total": adaptive_out["ndv_total"],
            "eval_offset_end": n_optimization_aero_calls_total,
            "objective_history": [dict(item) for item in adaptive_out["objective_history"]],
        }
    )
    history_full_opt.append(
        {
            "ndv_total": adaptive_out["ndv_total"],
            "opt_aero_offset_end": n_optimization_aero_calls_total,
            "function_eval_offset_end": n_function_evals_total,
            "gradient_eval_offset_end": n_gradient_evals_total,
            "function_eval_history": [dict(item) for item in adaptive_out["function_eval_history"]],
            "gradient_eval_history": [dict(item) for item in adaptive_out["gradient_eval_history"]],
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
        candidates = list(build_interval_candidates(upper_centers, side="UPPER"))
        candidates.extend(build_interval_candidates(lower_centers, side="LOWER"))

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
            n_scoring_aero_calls_total += int(pred_context.get("n_scoring_aero_calls", 0))

        current_ndv = len(upper_centers) + len(lower_centers)
        scored_local = []
        for cand in candidates:
            if str(indicator).upper() == "ORACLE":
                info = _score_candidate_oracle(
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    active_upper=upper_centers,
                    active_lower=lower_centers,
                    active_a=adaptive_out["a_opt"],
                    candidate=cand,
                    current_best_error=current_best_error,
                    workdir=Path(workdir),
                    level_ndv=current_ndv
                )
            else:
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

            n_scoring_aero_calls_total += int(info["n_scoring_aero_calls"])

            scored_local.append(
                {
                    "side": info["side"],
                    "x": info["x"],
                    "interval_id": cand["interval_id"],
                    "interval_label": cand.get("interval_label", f"{cand['side']}_{cand['interval_id']:02d}"),
                    "x_left": float(cand["x_left"]),
                    "x_right": float(cand["x_right"]),
                    "local_fraction": float(cand.get("local_fraction", 0.5)),
                    "score": info["score"],
                    "component": info["component"],
                    "raw_grad": info["raw_grad"],
                    "active_names": list(info["active_names"]),
                    "lambda": np.array(info["lambda"], copy=True),
                    "mode": info["mode"],
                }
            )

        scored = reduce_to_best_candidate_per_interval(scored_local)
        scored.sort(key=lambda item: (-item["score"], item["x"]))
        _print_candidate_diagnostics(scored, topk=12)

        
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
        print("chosen candidates =")
        for cand in chosen:
            print(f"  {_format_selected_candidate_summary(indicator, cand)}")

        new_upper = sorted(list(upper_centers))
        new_lower = sorted(list(lower_centers))

        for cand in chosen:
            if cand["side"] == "UPPER":
                new_upper.append(float(cand["x"]))
            else:
                new_lower.append(float(cand["x"]))

        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))

        selection_history.append(
            {
                "ndv_before": current_ndv,
                "indicator": str(indicator).upper(),
                "selected_candidates": [
                    {
                        "side": cand["side"],
                        "interval_id": int(cand["interval_id"]),
                        "interval_label": str(cand.get("interval_label", cand["interval_id"])),
                        "x_left": float(cand["x_left"]),
                        "x_right": float(cand["x_right"]),
                        "local_fraction": float(cand.get("local_fraction", 0.5)),
                        "center": float(cand["x"]),
                        "score": float(cand["score"]),
                    }
                    for cand in chosen
                ],
            }
        )

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
        _save_level_snapshot(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            adaptive_out=adaptive_out,
            label=level_label,
            workdir=workdir,
        )

        current_best_error = adaptive_out["err_opt"]

        n_optimization_aero_calls_total += adaptive_out["n_optimization_aero_calls_total"]
        n_function_evals_total += adaptive_out["n_function_evals"]
        n_gradient_evals_total += adaptive_out["n_gradient_evals"]

        upper_centers = new_upper
        lower_centers = new_lower

        history.append((adaptive_out["ndv_total"], adaptive_out["err_opt"]))
        history_evals.append(n_optimization_aero_calls_total)
        history_opt_aero_calls.append(n_optimization_aero_calls_total)
        history_function_evals.append(n_function_evals_total)
        history_gradient_evals.append(n_gradient_evals_total)

        history_full.append(
            {
                "ndv_total": adaptive_out["ndv_total"],
                "eval_offset_end": n_optimization_aero_calls_total,
                "objective_history": [dict(item) for item in adaptive_out["objective_history"]],
            }
        )
        history_full_opt.append(
            {
                "ndv_total": adaptive_out["ndv_total"],
                "opt_aero_offset_end": n_optimization_aero_calls_total,
                "function_eval_offset_end": n_function_evals_total,
                "gradient_eval_offset_end": n_gradient_evals_total,
                "function_eval_history": [dict(item) for item in adaptive_out["function_eval_history"]],
                "gradient_eval_history": [dict(item) for item in adaptive_out["gradient_eval_history"]],
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
    adaptive_out["history_opt_aero_calls"] = history_opt_aero_calls
    adaptive_out["history_function_evals"] = history_function_evals
    adaptive_out["history_gradient_evals"] = history_gradient_evals
    adaptive_out["history_full_opt"] = history_full_opt
    adaptive_out["center_history"] = center_history
    adaptive_out["selection_history"] = selection_history
    adaptive_out["n_scoring_aero_calls_total"] = n_scoring_aero_calls_total
    adaptive_out["n_optimization_aero_calls_total"] = n_optimization_aero_calls_total
    adaptive_out["n_function_evals_total"] = n_function_evals_total
    adaptive_out["n_gradient_evals_total"] = n_gradient_evals_total
    adaptive_out["n_scoring_evals_total"] = n_scoring_aero_calls_total
    adaptive_out["n_optimization_evals_total"] = n_optimization_aero_calls_total
    adaptive_out["n_total_evals"] = n_scoring_aero_calls_total + n_optimization_aero_calls_total
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

from pathlib import Path
import shutil
import csv
import numpy as np

from settings import SETTINGS
from optimization import optimize_for_centers
from adaptive_utils import (
    build_side_specific_initial_centers,
    build_interval_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)
from adaptive_scoring import prepare_gn_schur_level_context, score_candidate
from adaptive_pred import _prepare_pred_level_context
import gc
from cmplxfoil_wrapper import clear_cmplxfoil_solver_cache

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
            f"score_mode={item.get('score_mode', item.get('mode', '-'))}  "
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

    cand_key = _interval_best_sort_key(candidate)
    best_key = _interval_best_sort_key(current_best)
    return cand_key < best_key


def _finite_score(value, default=-np.inf):
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _interval_best_sort_key(candidate):
    score = _finite_score(candidate.get("score", -np.inf))
    local_fraction = float(candidate.get("local_fraction", np.nan))
    if not np.isfinite(local_fraction):
        midpoint_distance = float("inf")
    else:
        midpoint_distance = abs(local_fraction - 0.5)
    center = float(candidate.get("x", np.inf))
    if not np.isfinite(center):
        center = float("inf")
    return (
        -score,
        midpoint_distance,
        center,
        str(candidate.get("side", "")),
        int(candidate.get("interval_id", 0)),
        str(candidate.get("interval_label", "")),
    )


def reduce_to_best_candidate_per_interval(scored_candidates):
    winners = {}

    for item in scored_candidates:
        interval_key = (str(item["side"]).upper(), int(item["interval_id"]))
        if interval_key not in winners:
            winners[interval_key] = item
            continue

        if _candidate_beats_interval_best(item, winners[interval_key]):
            winners[interval_key] = item

    return [winners[key] for key in sorted(winners)]


def _candidate_global_sort_key(candidate):
    score = _finite_score(candidate.get("score", -np.inf))
    center = float(candidate.get("x", np.inf))
    if not np.isfinite(center):
        center = float("inf")
    return (
        -score,
        str(candidate.get("side", "")),
        center,
        int(candidate.get("interval_id", 0)),
        str(candidate.get("interval_label", "")),
    )


def select_adaptive_candidate_batch(
    interval_best_candidates,
    n_remaining,
    max_batch_size,
    score_rel_tol,
    min_separation,
    max_per_side=None,
):
    if int(n_remaining) <= 0 or not interval_best_candidates:
        return []

    limit = min(int(max_batch_size), int(n_remaining))
    if limit <= 0:
        return []

    candidates = sorted(interval_best_candidates, key=_candidate_global_sort_key)
    selected = [candidates[0]]
    selected_per_side = {str(candidates[0].get("side", "")).upper(): 1}
    best_score = _finite_score(candidates[0].get("score", -np.inf))
    if not np.isfinite(best_score) or best_score <= 0.0:
        return selected

    threshold = float(score_rel_tol) * best_score
    min_separation = float(min_separation)
    for candidate in candidates[1:]:
        if len(selected) >= limit:
            break

        score = _finite_score(candidate.get("score", -np.inf))
        if score < threshold:
            continue

        side = str(candidate.get("side", "")).upper()
        if max_per_side is not None and selected_per_side.get(side, 0) >= int(max_per_side):
            continue

        center = float(candidate.get("x", np.nan))
        if not np.isfinite(center):
            continue
        too_close = False
        for chosen in selected:
            if str(chosen.get("side", "")).upper() != side:
                continue
            chosen_center = float(chosen.get("x", np.nan))
            if np.isfinite(chosen_center) and abs(center - chosen_center) < min_separation:
                too_close = True
                break
        if too_close:
            continue

        selected.append(candidate)
        selected_per_side[side] = selected_per_side.get(side, 0) + 1

    return selected


def _annotate_candidate_selection(scored_candidates, interval_best_candidates, selected_candidates):
    interval_best_ids = {id(item) for item in interval_best_candidates}
    selected_ids = {id(item) for item in selected_candidates}
    interval_best_score = {}
    for item in interval_best_candidates:
        key = (str(item.get("side", "")).upper(), int(item.get("interval_id", 0)))
        interval_best_score[key] = _finite_score(item.get("score", np.nan), default=np.nan)
    finite_scores = [_finite_score(item.get("score", np.nan), default=np.nan) for item in scored_candidates]
    finite_scores = [v for v in finite_scores if np.isfinite(v)]
    global_best = max(finite_scores) if finite_scores else float("nan")

    for item in scored_candidates:
        key = (str(item.get("side", "")).upper(), int(item.get("interval_id", 0)))
        score = _finite_score(item.get("score", np.nan), default=np.nan)
        best_interval = interval_best_score.get(key, float("nan"))
        item["is_interval_best"] = bool(id(item) in interval_best_ids)
        item["selected_in_batch"] = bool(id(item) in selected_ids)
        item["score_ratio_to_interval_best"] = (
            score / best_interval if np.isfinite(score) and np.isfinite(best_interval) and best_interval != 0.0 else np.nan
        )
        item["score_ratio_to_global_best"] = (
            score / global_best if np.isfinite(score) and np.isfinite(global_best) and global_best != 0.0 else np.nan
        )


def _score_ratio_to_best(candidate, best_score):
    score = _finite_score(candidate.get("score", np.nan), default=np.nan)
    if not np.isfinite(score) or not np.isfinite(best_score) or best_score == 0.0:
        return np.nan
    return score / best_score


def _selected_candidate_record(candidate, best_score):
    return {
        "side": candidate["side"],
        "interval_id": int(candidate["interval_id"]),
        "interval_label": str(candidate.get("interval_label", candidate["interval_id"])),
        "x_left": float(candidate["x_left"]),
        "x_right": float(candidate["x_right"]),
        "local_fraction": float(candidate.get("local_fraction", 0.5)),
        "center": float(candidate["x"]),
        "score": float(candidate["score"]),
        "score_ratio_to_best": float(_score_ratio_to_best(candidate, best_score)),
    }


def _print_refinement_batch_summary(
    title,
    current_ndv,
    n_final,
    batch_enabled,
    batch_size_max,
    score_rel_tol,
    min_separation,
    max_per_side,
    n_raw,
    n_interval_best,
    chosen,
    current_best_error,
):
    best_score = _finite_score(chosen[0].get("score", np.nan), default=np.nan) if chosen else np.nan
    print(title)
    print(f"current ndv before refinement = {int(current_ndv)}")
    print(f"target final ndv              = {int(n_final)}")
    print("batch enabled                 = " + ("YES" if batch_enabled else "NO"))
    print(f"max batch size                = {int(batch_size_max)}")
    print(f"score relative tolerance      = {float(score_rel_tol):.6f}")
    print(f"min separation                = {float(min_separation):.6f}")
    print(f"max per side                  = {max_per_side}")
    print(f"raw candidates                = {int(n_raw)}")
    print(f"interval-best candidates      = {int(n_interval_best)}")
    print(f"selected candidates           = {len(chosen)}")
    print(f"current best err              = {float(current_best_error):.6e}")
    print("selected candidates =")
    for cand in chosen:
        ratio = _score_ratio_to_best(cand, best_score)
        interval_label = cand.get("interval_label", cand.get("interval_id", "?"))
        print(
            f"  side={cand['side']} interval={cand['interval_id']} label={interval_label} "
            f"x_left={float(cand['x_left']):.6f} x_right={float(cand['x_right']):.6f} "
            f"fraction={float(cand.get('local_fraction', 0.5)):.2f} "
            f"center={float(cand['x']):.6f} score={float(cand['score']):.6e} "
            f"score_ratio_to_best={float(ratio):.6e}"
        )


def _select_refinement_candidates(scored_local, current_ndv, n_final, opt_ad):
    interval_best = reduce_to_best_candidate_per_interval(scored_local)
    batch_enabled = bool(opt_ad.get("refine_batch_enabled", False))
    n_remaining = int(n_final) - int(current_ndv)

    if batch_enabled:
        chosen = select_adaptive_candidate_batch(
            interval_best_candidates=interval_best,
            n_remaining=n_remaining,
            max_batch_size=int(opt_ad.get("refine_batch_size_max", 1)),
            score_rel_tol=float(opt_ad.get("refine_batch_score_rel_tol", 0.85)),
            min_separation=float(opt_ad.get("refine_batch_min_separation", 0.04)),
            max_per_side=opt_ad.get("refine_batch_max_per_side", None),
        )
    else:
        scored = list(interval_best)
        scored.sort(key=lambda item: (-item["score"], item["x"]))
        n_add = _compute_adaptive_nadd(current_ndv, len(scored))
        n_add = min(n_add, n_remaining)
        chosen = scored[:n_add] if n_add > 0 else []

    _annotate_candidate_selection(scored_local, interval_best, chosen)
    return interval_best, chosen


def _format_selected_candidate_summary(indicator, candidate):
    interval_label = candidate.get("interval_label", candidate.get("interval_id", "?"))
    return (
        f"[{str(indicator).upper()}] interval={interval_label} "
        f"winner_fraction={float(candidate.get('local_fraction', 0.5)):.2f} "
        f"center={float(candidate['x']):.6f} "
        f"score={float(candidate['score']):.6e}"
    )


def _make_adaptive_trigger_options(current_ndv, n_final, level_label):
    adapt_cfg = SETTINGS["optimization"]["adaptive"]
    is_final_level = int(current_ndv) >= int(n_final)
    enabled = bool(adapt_cfg.get("trigger_enabled", False))

    print("ADAPTIVE TRIGGER enabled = " + ("YES" if enabled else "NO"))
    print(f"mode = {str(adapt_cfg.get('trigger_mode', 'slope_efficiency')).strip().lower()}")
    print(f"current_ndv = {int(current_ndv)}")
    print("final_level = " + ("YES" if is_final_level else "NO"))

    if not enabled:
        return None

    trigger_apply_to = str(adapt_cfg.get("trigger_apply_to", "adaptive_intermediate")).strip().lower()
    if trigger_apply_to == "adaptive_intermediate" and is_final_level:
        return None
    if bool(adapt_cfg.get("trigger_disable_on_final_level", True)) and is_final_level:
        return None

    return {
        "enabled": True,
        "mode": str(adapt_cfg.get("trigger_mode", "slope_efficiency")).strip().lower(),
        "min_major_iters": int(adapt_cfg.get("trigger_min_major_iters", 5)),
        "write_csv": bool(adapt_cfg.get("trigger_write_csv", True)),
        "eps": float(adapt_cfg.get("trigger_eps", 1.0e-300)),
        "window": int(adapt_cfg.get("trigger_window", 3)),
        "slope_rel_tol": float(adapt_cfg.get("trigger_slope_rel_tol", 0.05)),
        "slope_patience": int(adapt_cfg.get("trigger_slope_patience", 1)),
        "log_objective": bool(adapt_cfg.get("trigger_log_objective", True)),
        "stag_tol": float(adapt_cfg.get("trigger_stag_tol", 1.0e-3)),
        "stag_band": float(adapt_cfg.get("trigger_stag_band", 0.02)),
        "stag_patience": int(adapt_cfg.get("trigger_stag_patience", 3)),
        "max_major_iters": adapt_cfg.get("trigger_max_major_iters", None),
        "current_ndv": int(current_ndv),
        "target_ndv": int(n_final),
        "level_label": str(level_label),
        "is_final_level": bool(is_final_level),
    }


def _write_candidate_score_csv(workdir, level, ndv_current, scored_candidates):
    if len(scored_candidates) == 0:
        return None

    out_path = Path(workdir) / f"candidate_scores_level_{int(level):03d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persistent_path = (
        Path(workdir).parent
        / "summary"
        / "candidate_scores"
        / Path(workdir).name
        / out_path.name
    )
    persistent_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "level",
        "ndv_current",
        "ndv_candidate",
        "side",
        "center",
        "interval_left",
        "interval_right",
        "fraction",
        "interval_width",
        "n_samples_for_interval",
        "sampling_fallback_midpoint",
        "is_interval_best",
        "selected_in_batch",
        "score_ratio_to_interval_best",
        "score_ratio_to_global_best",
        "score_mode",
        "score",
        "g_norm",
        "g_new",
        "abs_g_new",
        "gn_g_perp",
        "gn_q_schur",
        "gn_alpha_unclipped",
        "gn_alpha_clipped",
        "gn_norm_jc",
        "gn_norm_z",
        "gn_rank_A",
        "gn_fd_mode",
        "gn_fd_h",
        "gn_n_fail_candidate",
    ]

    rows = []
    for item in scored_candidates:
        rows.append(
            {
                "level": int(level),
                "ndv_current": int(ndv_current),
                "ndv_candidate": int(ndv_current) + 1,
                "side": item.get("side", ""),
                "center": float(item.get("x", np.nan)),
                "interval_left": float(item.get("x_left", np.nan)),
                "interval_right": float(item.get("x_right", np.nan)),
                "fraction": float(item.get("local_fraction", np.nan)),
                "interval_width": float(item.get("interval_width", np.nan)),
                "n_samples_for_interval": int(item.get("n_samples_for_interval", 0)),
                "sampling_fallback_midpoint": bool(item.get("sampling_fallback_midpoint", False)),
                "is_interval_best": bool(item.get("is_interval_best", False)),
                "selected_in_batch": bool(item.get("selected_in_batch", False)),
                "score_ratio_to_interval_best": float(item.get("score_ratio_to_interval_best", np.nan)),
                "score_ratio_to_global_best": float(item.get("score_ratio_to_global_best", np.nan)),
                "score_mode": item.get("score_mode", ""),
                "score": float(item.get("score", np.nan)),
                "g_norm": float(item.get("g_norm", np.nan)),
                "g_new": float(item.get("g_new", np.nan)),
                "abs_g_new": float(item.get("abs_g_new", np.nan)),
                "gn_g_perp": float(item.get("gn_g_perp", np.nan)),
                "gn_q_schur": float(item.get("gn_q_schur", np.nan)),
                "gn_alpha_unclipped": float(item.get("gn_alpha_unclipped", np.nan)),
                "gn_alpha_clipped": float(item.get("gn_alpha_clipped", np.nan)),
                "gn_norm_jc": float(item.get("gn_norm_jc", np.nan)),
                "gn_norm_z": float(item.get("gn_norm_z", np.nan)),
                "gn_rank_A": item.get("gn_rank_A", np.nan),
                "gn_fd_mode": item.get("gn_fd_mode", ""),
                "gn_fd_h": float(item.get("gn_fd_h", np.nan)),
                "gn_n_fail_candidate": item.get("gn_n_fail_candidate", np.nan),
            }
        )

    for path in (out_path, persistent_path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return out_path, persistent_path


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

def _run_adaptive_strategy_legacy_unused(
    x,
    yu_init,
    yl_init,
    cp_target,
    initial_error,
    workdir,
    indicator,
):
    opt_ad = SETTINGS["optimization"]["adaptive"]
    indicator_u = str(indicator).upper()
    grad_score_mode = str(opt_ad.get("grad_score_mode", "grad_norm")).strip().lower()
    if indicator_u == "GRAD":
        allowed_grad_score_modes = ["grad_norm", "gn_schur"]
        if grad_score_mode not in allowed_grad_score_modes:
            raise ValueError(
                f"Unsupported ADAPT_GRAD_SCORE_MODE={grad_score_mode!r}. "
                f"Supported modes: {allowed_grad_score_modes}."
            )
        print(f"ADAPT_GRAD score mode = {grad_score_mode}")

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
        label=f"adaptive_from_state_level_{len(upper_centers) + len(lower_centers)}",
        workdir=workdir,
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
        if indicator_u == "PRED":
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
        gn_context = None
        if indicator_u == "GRAD" and grad_score_mode == "gn_schur":
            gn_context = prepare_gn_schur_level_context(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                active_upper=upper_centers,
                active_lower=lower_centers,
                active_a=np.asarray(adaptive_out["a_opt"], dtype=float),
                current_best_error=current_best_error,
                workdir=Path(workdir) / "adaptive_candidate_scoring",
            )
            n_scoring_aero_calls_total += int(gn_context.get("n_scoring_aero_calls", 0))
            print(
                "GN-SCHUR level context: "
                f"base_status={gn_context.get('base_status')}  "
                f"rank_A={gn_context.get('rank_A')}  "
                f"score_aero_calls={int(gn_context.get('n_scoring_aero_calls', 0))}  "
                f"active_fd_fail={gn_context.get('fd_diag_active', {}).get('n_fail_dirs', 0)}"
            )

        scored_local = []

        for cand in candidates:
            if indicator_u == "ORACLE":
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
                    level_ndv=current_ndv,
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
                    gn_context=gn_context,
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
                    "interval_width": float(cand.get("interval_width", np.nan)),
                    "n_samples_for_interval": int(cand.get("n_samples_for_interval", 0)),
                    "sampling_fallback_midpoint": bool(cand.get("sampling_fallback_midpoint", False)),
                    "score": info["score"],
                    "component": info["component"],
                    "raw_grad": info["raw_grad"],
                    "active_names": list(info["active_names"]),
                    "lambda": np.array(info["lambda"], copy=True),
                    "mode": info["mode"],
                    "score_mode": info.get("score_mode", ""),
                    "g_norm": info.get("g_norm", np.nan),
                    "g_new": info.get("g_new", info.get("raw_grad", np.nan)),
                    "abs_g_new": info.get("abs_g_new", abs(float(info.get("raw_grad", np.nan))) if np.isfinite(info.get("raw_grad", np.nan)) else np.nan),
                    "gn_g_perp": info.get("gn_g_perp", np.nan),
                    "gn_q_schur": info.get("gn_q_schur", np.nan),
                    "gn_alpha_unclipped": info.get("gn_alpha_unclipped", np.nan),
                    "gn_alpha_clipped": info.get("gn_alpha_clipped", np.nan),
                    "gn_norm_jc": info.get("gn_norm_jc", np.nan),
                    "gn_norm_z": info.get("gn_norm_z", np.nan),
                    "gn_rank_A": info.get("gn_rank_A", np.nan),
                    "gn_fd_mode": info.get("gn_fd_mode", ""),
                    "gn_fd_h": info.get("gn_fd_h", np.nan),
                    "gn_n_fail_candidate": info.get("gn_n_fail_candidate", np.nan),
                }
            )

        interval_best, chosen = _select_refinement_candidates(
            scored_local=scored_local,
            current_ndv=current_ndv,
            n_final=n_final,
            opt_ad=opt_ad,
        )
        diag_scored = list(interval_best)
        if bool(opt_ad.get("refine_batch_enabled", False)):
            diag_scored.sort(key=_candidate_global_sort_key)
        else:
            diag_scored.sort(key=lambda item: (-item["score"], item["x"]))
        _print_candidate_diagnostics(diag_scored, topk=12)

        if indicator_u == "GRAD" and bool(opt_ad.get("write_candidate_score_csv", True)):
            csv_path = _write_candidate_score_csv(
                workdir=workdir,
                level=current_ndv,
                ndv_current=current_ndv,
                scored_candidates=scored_local,
            )
            if csv_path is not None:
                print(f"Candidate score CSV written: {csv_path[0]}")
                print(f"Persistent candidate score CSV written: {csv_path[1]}")

        if str(SETTINGS.get("aero", {}).get("backend", "xfoil")).strip().lower() == "cmplxfoil":
            clear_cmplxfoil_solver_cache()
            gc.collect()

        
        if len(chosen) <= 0:
            break

        new_upper = sorted(list(upper_centers))
        new_lower = sorted(list(lower_centers))

        for cand in chosen:
            if cand["side"] == "UPPER":
                new_upper.append(float(cand["x"]))
            else:
                new_lower.append(float(cand["x"]))

        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))
        ndv_after = len(new_upper) + len(new_lower)
        best_score = _finite_score(chosen[0].get("score", np.nan), default=np.nan)

        _print_refinement_batch_summary(
            title="===== ADAPTIVE REFINE =====",
            current_ndv=current_ndv,
            n_final=n_final,
            batch_enabled=bool(opt_ad.get("refine_batch_enabled", False)),
            batch_size_max=int(opt_ad.get("refine_batch_size_max", 1)),
            score_rel_tol=float(opt_ad.get("refine_batch_score_rel_tol", 0.85)),
            min_separation=float(opt_ad.get("refine_batch_min_separation", 0.04)),
            max_per_side=opt_ad.get("refine_batch_max_per_side", None),
            n_raw=len(scored_local),
            n_interval_best=len(interval_best),
            chosen=chosen,
            current_best_error=current_best_error,
        )
        print(
            f"adaptive batch summary: ndv {current_ndv} -> {ndv_after}, "
            f"n_added={ndv_after - current_ndv}, "
            f"centers={[float(c['x']) for c in chosen]}, "
            f"scores={[float(c['score']) for c in chosen]}, "
            f"ratios={[float(_score_ratio_to_best(c, best_score)) for c in chosen]}"
        )

        selection_history.append(
            {
                "ndv_before": current_ndv,
                "ndv_after": ndv_after,
                "n_added": ndv_after - current_ndv,
                "indicator": str(indicator).upper(),
                "batch_enabled": bool(opt_ad.get("refine_batch_enabled", False)),
                "batch_size_max": int(opt_ad.get("refine_batch_size_max", 1)),
                "batch_score_rel_tol": float(opt_ad.get("refine_batch_score_rel_tol", 0.85)),
                "batch_min_separation": float(opt_ad.get("refine_batch_min_separation", 0.04)),
                "batch_max_per_side": opt_ad.get("refine_batch_max_per_side", None),
                "selected_candidates": [_selected_candidate_record(cand, best_score) for cand in chosen],
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
    if indicator_u == "GRAD":
        adaptive_out["grad_score_mode"] = grad_score_mode

    return adaptive_out


def run_adaptive_strategy_from_state(
    x,
    yu_init,
    yl_init,
    cp_target,
    initial_error,
    workdir,
    indicator,
    initial_upper_centers,
    initial_lower_centers,
    initial_a0,
    n_final_override,
    max_refinements=None,
):
    opt_ad = SETTINGS["optimization"]["adaptive"]
    indicator_u = str(indicator).upper()
    grad_score_mode = str(opt_ad.get("grad_score_mode", "grad_norm")).strip().lower()
    if indicator_u == "GRAD":
        allowed_grad_score_modes = ["grad_norm", "gn_schur"]
        if grad_score_mode not in allowed_grad_score_modes:
            raise ValueError(
                f"Unsupported ADAPT_GRAD_SCORE_MODE={grad_score_mode!r}. "
                f"Supported modes: {allowed_grad_score_modes}."
            )
        print(f"ADAPT_GRAD score mode = {grad_score_mode}")

    upper_centers = sorted(float(v) for v in initial_upper_centers)
    lower_centers = sorted(float(v) for v in initial_lower_centers)
    a0 = np.asarray(initial_a0, dtype=float)
    n_final = int(n_final_override)
    current_best_error = float(initial_error)

    history = []
    center_history = []
    selection_history = []
    history_full = []
    history_full_opt = []
    history_evals = []
    history_opt_aero_calls = []
    history_function_evals = []
    history_gradient_evals = []
    n_scoring_aero_calls_total = 0
    n_optimization_aero_calls_total = 0
    n_function_evals_total = 0
    n_gradient_evals_total = 0
    n_refinements_done = 0

    level_label = f"adaptive_from_state_level_{len(upper_centers) + len(lower_centers)}"
    trigger_options = _make_adaptive_trigger_options(
        current_ndv=len(upper_centers) + len(lower_centers),
        n_final=n_final,
        level_label=level_label,
    )
    adaptive_out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a0=a0,
        label=level_label,
        current_best_error=current_best_error,
        workdir=Path(workdir) / "adaptive",
        trigger_options=trigger_options,
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

    def _record_level(out):
        history.append((out["ndv_total"], out["err_opt"]))
        history_evals.append(n_optimization_aero_calls_total)
        history_opt_aero_calls.append(n_optimization_aero_calls_total)
        history_function_evals.append(n_function_evals_total)
        history_gradient_evals.append(n_gradient_evals_total)
        history_full.append(
            {
                "ndv_total": out["ndv_total"],
                "eval_offset_end": n_optimization_aero_calls_total,
                "objective_history": [dict(item) for item in out["objective_history"]],
            }
        )
        history_full_opt.append(
            {
                "ndv_total": out["ndv_total"],
                "opt_aero_offset_end": n_optimization_aero_calls_total,
                "function_eval_offset_end": n_function_evals_total,
                "gradient_eval_offset_end": n_gradient_evals_total,
                "function_eval_history": [dict(item) for item in out["function_eval_history"]],
                "gradient_eval_history": [dict(item) for item in out["gradient_eval_history"]],
                "trigger_history": [dict(item) for item in out.get("trigger_history", [])],
                "early_stop_triggered": bool(out.get("early_stop_triggered", False)),
                "early_stop_reason": out.get("early_stop_reason", ""),
                "early_stop_major_iter": out.get("early_stop_major_iter", ""),
            }
        )
        center_history.append(
            {
                "ndv_total": out["ndv_total"],
                "upper": list(upper_centers),
                "lower": list(lower_centers),
            }
        )

    _record_level(adaptive_out)

    while (len(upper_centers) + len(lower_centers)) < n_final:
        candidates = list(build_interval_candidates(upper_centers, side="UPPER"))
        candidates.extend(build_interval_candidates(lower_centers, side="LOWER"))
        if len(candidates) == 0:
            break

        pred_context = None
        if indicator_u == "PRED":
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
        gn_context = None
        if indicator_u == "GRAD" and grad_score_mode == "gn_schur":
            gn_context = prepare_gn_schur_level_context(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                active_upper=upper_centers,
                active_lower=lower_centers,
                active_a=np.asarray(adaptive_out["a_opt"], dtype=float),
                current_best_error=current_best_error,
                workdir=Path(workdir) / "adaptive_candidate_scoring",
            )
            n_scoring_aero_calls_total += int(gn_context.get("n_scoring_aero_calls", 0))

        scored_local = []
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
                gn_context=gn_context,
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
                    "interval_width": float(cand.get("interval_width", np.nan)),
                    "n_samples_for_interval": int(cand.get("n_samples_for_interval", 0)),
                    "sampling_fallback_midpoint": bool(cand.get("sampling_fallback_midpoint", False)),
                    "score": info["score"],
                    "component": info["component"],
                    "raw_grad": info["raw_grad"],
                    "active_names": list(info["active_names"]),
                    "lambda": np.array(info["lambda"], copy=True),
                    "mode": info["mode"],
                    "score_mode": info.get("score_mode", ""),
                    "g_norm": info.get("g_norm", np.nan),
                    "g_new": info.get("g_new", info.get("raw_grad", np.nan)),
                    "abs_g_new": info.get("abs_g_new", abs(float(info.get("raw_grad", np.nan))) if np.isfinite(info.get("raw_grad", np.nan)) else np.nan),
                    "gn_g_perp": info.get("gn_g_perp", np.nan),
                    "gn_q_schur": info.get("gn_q_schur", np.nan),
                    "gn_alpha_unclipped": info.get("gn_alpha_unclipped", np.nan),
                    "gn_alpha_clipped": info.get("gn_alpha_clipped", np.nan),
                    "gn_norm_jc": info.get("gn_norm_jc", np.nan),
                    "gn_norm_z": info.get("gn_norm_z", np.nan),
                    "gn_rank_A": info.get("gn_rank_A", np.nan),
                    "gn_fd_mode": info.get("gn_fd_mode", ""),
                    "gn_fd_h": info.get("gn_fd_h", np.nan),
                    "gn_n_fail_candidate": info.get("gn_n_fail_candidate", np.nan),
                }
            )

        interval_best, chosen = _select_refinement_candidates(
            scored_local=scored_local,
            current_ndv=current_ndv,
            n_final=n_final,
            opt_ad=opt_ad,
        )
        diag_scored = list(interval_best)
        if bool(opt_ad.get("refine_batch_enabled", False)):
            diag_scored.sort(key=_candidate_global_sort_key)
        else:
            diag_scored.sort(key=lambda item: (-item["score"], item["x"]))
        _print_candidate_diagnostics(diag_scored, topk=12)

        if indicator_u == "GRAD" and bool(opt_ad.get("write_candidate_score_csv", True)):
            csv_path = _write_candidate_score_csv(
                workdir=workdir,
                level=current_ndv,
                ndv_current=current_ndv,
                scored_candidates=scored_local,
            )
            if csv_path is not None:
                print(f"Candidate score CSV written: {csv_path[0]}")
                print(f"Persistent candidate score CSV written: {csv_path[1]}")

        if str(SETTINGS.get("aero", {}).get("backend", "xfoil")).strip().lower() == "cmplxfoil":
            clear_cmplxfoil_solver_cache()
            gc.collect()

        if len(chosen) <= 0:
            break

        new_upper = sorted(list(upper_centers))
        new_lower = sorted(list(lower_centers))
        for cand in chosen:
            if cand["side"] == "UPPER":
                new_upper.append(float(cand["x"]))
            else:
                new_lower.append(float(cand["x"]))
        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))
        ndv_after = len(new_upper) + len(new_lower)
        best_score = _finite_score(chosen[0].get("score", np.nan), default=np.nan)

        _print_refinement_batch_summary(
            title="===== ADAPTIVE REFINE FROM STATE =====",
            current_ndv=current_ndv,
            n_final=n_final,
            batch_enabled=bool(opt_ad.get("refine_batch_enabled", False)),
            batch_size_max=int(opt_ad.get("refine_batch_size_max", 1)),
            score_rel_tol=float(opt_ad.get("refine_batch_score_rel_tol", 0.85)),
            min_separation=float(opt_ad.get("refine_batch_min_separation", 0.04)),
            max_per_side=opt_ad.get("refine_batch_max_per_side", None),
            n_raw=len(scored_local),
            n_interval_best=len(interval_best),
            chosen=chosen,
            current_best_error=current_best_error,
        )
        print(
            f"adaptive batch summary: ndv {current_ndv} -> {ndv_after}, "
            f"n_added={ndv_after - current_ndv}, "
            f"centers={[float(c['x']) for c in chosen]}, "
            f"scores={[float(c['score']) for c in chosen]}, "
            f"ratios={[float(_score_ratio_to_best(c, best_score)) for c in chosen]}"
        )

        selection_history.append(
            {
                "ndv_before": current_ndv,
                "ndv_after": ndv_after,
                "n_added": ndv_after - current_ndv,
                "indicator": indicator_u,
                "early_stop_triggered_before_refine": bool(adaptive_out.get("early_stop_triggered", False)),
                "early_stop_reason": adaptive_out.get("early_stop_reason", ""),
                "early_stop_major_iter": adaptive_out.get("early_stop_major_iter", ""),
                "batch_enabled": bool(opt_ad.get("refine_batch_enabled", False)),
                "batch_size_max": int(opt_ad.get("refine_batch_size_max", 1)),
                "batch_score_rel_tol": float(opt_ad.get("refine_batch_score_rel_tol", 0.85)),
                "batch_min_separation": float(opt_ad.get("refine_batch_min_separation", 0.04)),
                "batch_max_per_side": opt_ad.get("refine_batch_max_per_side", None),
                "selected_candidates": [_selected_candidate_record(cand, best_score) for cand in chosen],
            }
        )

        new_a0 = lift_a_to_new_side_centers(
            old_upper=upper_centers,
            old_lower=lower_centers,
            old_a=adaptive_out["a_opt"],
            new_upper=new_upper,
            new_lower=new_lower,
        )
        upper_centers = new_upper
        lower_centers = new_lower
        level_label = f"adaptive_from_state_level_{len(new_upper) + len(new_lower)}"
        trigger_options = _make_adaptive_trigger_options(
            current_ndv=len(new_upper) + len(new_lower),
            n_final=n_final,
            level_label=level_label,
        )
        adaptive_out = optimize_for_centers(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            upper_centers=upper_centers,
            lower_centers=lower_centers,
            a0=new_a0,
            label=level_label,
            current_best_error=current_best_error,
            workdir=Path(workdir) / "adaptive",
            trigger_options=trigger_options,
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
        _record_level(adaptive_out)
        n_refinements_done += 1
        if max_refinements is not None and n_refinements_done >= int(max_refinements):
            break

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
    adaptive_out["n_early_stopped_levels"] = int(
        sum(1 for level in history_full_opt if bool(level.get("early_stop_triggered", False)))
    )
    if indicator_u == "GRAD":
        adaptive_out["grad_score_mode"] = grad_score_mode
    return adaptive_out


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

    upper_centers, lower_centers = build_side_specific_initial_centers(int(opt_ad["n0"]))
    a0 = np.zeros(len(upper_centers) + len(lower_centers), dtype=float)
    out = run_adaptive_strategy_from_state(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        initial_error=initial_error,
        workdir=workdir,
        indicator=indicator,
        initial_upper_centers=upper_centers,
        initial_lower_centers=lower_centers,
        initial_a0=a0,
        n_final_override=int(opt_ad["n_final"]),
    )
    return out

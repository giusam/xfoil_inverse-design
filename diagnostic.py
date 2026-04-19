from pathlib import Path
import csv
import shutil

import numpy as np

from settings import SETTINGS
from geometry import build_naca0012_surfaces, build_random_initial_geometry, write_dat
from xfoil_wrapper import clean_workdir, run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from adaptive_utils import (
    build_side_specific_initial_centers,
    get_midpoint_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)
from adaptive_candidate import (
    _build_extended_space,
    _make_candidate_objective,
    _evaluate_objective_state,
    _rebuild_geometry_from_a,
)
from adaptive_fd import _compute_full_aero_gradients
from adaptive_constraints import (
    _build_active_ikkt_system,
    _compute_geometric_gradients_analytic,
)
from adaptive_pred import (
    _prepare_pred_level_context,
    _null_space_from_constraint_columns,
    _compute_restoration_step,
    _compute_candidate_hessian_border,
)


SEED = 0
STATE_SELECTOR = "GRAD"
VALIDATE_AT_NDV = 6

CANDIDATES_TO_TEST = [
    ("LOWER", 0.275000),
    ("LOWER", 0.725000),
]

H_ABS_LIST = [5.0e-4, 1e-3]
H_REL_STEP = 5.0e-3
REG_LIST = [1.0e-8]

DO_REAL_REOPT = True
WRITE_SUMMARY_CSV = True
WRITE_MATRIX_CSV = True


def _enabled_aero_names():
    names = []
    for name in ("CL", "CD", "CM"):
        spec = SETTINGS.get("constraints", {}).get(name, {})
        if spec.get("enabled", False):
            names.append(name)
    return names


def _safe_tag_float(x):
    return f"{float(x):.1e}".replace("+", "").replace(".", "p").replace("-", "m")


def _write_section(writer, title, arr):
    arr = np.asarray(arr)
    writer.writerow([f"# {title}"])

    if arr.ndim == 0:
        writer.writerow([float(arr)])
    elif arr.ndim == 1:
        writer.writerow(["idx", "value"])
        for i, val in enumerate(arr):
            writer.writerow([i, float(val)])
    elif arr.ndim == 2:
        header = ["row"] + [f"c{j}" for j in range(arr.shape[1])]
        writer.writerow(header)
        for i in range(arr.shape[0]):
            writer.writerow([i] + [float(v) for v in arr[i]])
    else:
        flat = arr.reshape(arr.shape[0], -1)
        header = ["row"] + [f"c{j}" for j in range(flat.shape[1])]
        writer.writerow(header)
        for i in range(flat.shape[0]):
            writer.writerow([i] + [float(v) for v in flat[i]])

    writer.writerow([])


def _save_candidate_matrix_csv(out, save_dir, h_abs, reg):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    side = str(out["side"]).upper()
    x_val = float(out["x"])

    fname = (
        f"diagnostic_{side}_{x_val:.6f}"
        f"_h_{_safe_tag_float(h_abs)}"
        f"_reg_{_safe_tag_float(reg)}.csv"
    )
    fpath = save_dir / fname

    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["# diagnostic candidate matrix dump"])
        writer.writerow(["side", side])
        writer.writerow(["x", x_val])
        writer.writerow(["h_abs", float(h_abs)])
        writer.writerow(["rel_step", float(out["rel_step"])])
        writer.writerow(["reg", float(reg)])
        writer.writerow(["delta_pred_full", float(out["delta_pred_full"])])
        writer.writerow(["delta_pred_no_rest", float(out["delta_pred_no_rest"])])
        writer.writerow(["delta_real", float(out["delta_real"])])
        writer.writerow(["raw_grad_new", float(out["raw_grad"])])
        writer.writerow(["grad_norm", float(out["grad_norm"])])
        writer.writerow(["dy_new", float(out["dy_new"])])
        writer.writerow(["dy_norm", float(out["dy_norm"])])
        writer.writerow(["gamma_norm", float(out["gamma_norm"])])
        writer.writerow(["gamma_no_rest_norm", float(out["gamma_no_rest_norm"])])
        writer.writerow(["h_nn", float(out["h_nn"])])
        writer.writerow(["h_col_maxabs", float(out["h_col_maxabs"])])
        writer.writerow(["eig_min", float(out["eig_min"])])
        writer.writerow(["eig_max", float(out["eig_max"])])
        writer.writerow(["cond_raw", float(out["cond_raw"])])
        writer.writerow(["cond_reg", float(out["cond_reg"])])
        writer.writerow(["n_active", int(out["n_active"])])
        writer.writerow(["n_free", int(out["n_free"])])
        writer.writerow(["active_names", "|".join(out["active_names"])])
        writer.writerow([])

        _write_section(writer, "g", out["g"])
        _write_section(writer, "Z", out["Z"])
        _write_section(writer, "A", out["A"])
        _write_section(writer, "H_full", out["H_full"])
        _write_section(writer, "H_red", out["H_red"])
        _write_section(writer, "eig_H_full", out["eig_H_full"])
        _write_section(writer, "eig_H_red", out["eig_H_red"])

    return fpath


def _build_problem(seed, workdir):
    x, yu_target, yl_target = build_naca0012_surfaces(
        n_points=SETTINGS["geom"]["n_points"],
        thickness=SETTINGS["geom"]["thickness"],
    )

    target_dat = Path(workdir) / "target_airfoil.dat"
    write_dat(target_dat, x, yu_target, yl_target, name="TARGET_NACA0012")

    target_res = run_xfoil(
        airfoil_dat=target_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / "target_run",
    )
    if not target_res["success"]:
        print(target_res["stdout"])
        print(target_res["stderr"])
        raise RuntimeError("Target XFOIL run failed.")

    cp_target = split_upper_lower_cp_from_x(
        target_res["cp_data"]["x"],
        target_res["cp_data"]["cp"],
    )

    init_geom = build_random_initial_geometry(
        x=x,
        yu_target=yu_target,
        yl_target=yl_target,
        seed=seed,
        amp=SETTINGS["initial_shape"]["random_amp"],
        order=SETTINGS["initial_shape"]["bernstein_order"],
    )

    yu_init = init_geom["yu_init"]
    yl_init = init_geom["yl_init"]

    init_dat = Path(workdir) / "initial_airfoil.dat"
    write_dat(init_dat, x, yu_init, yl_init, name="INITIAL_RANDOM")

    init_res = run_xfoil(
        airfoil_dat=init_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / "initial_run",
    )
    if not init_res["success"]:
        print(init_res["stdout"])
        print(init_res["stderr"])
        raise RuntimeError("Initial XFOIL run failed.")

    cp_init = split_upper_lower_cp_from_x(
        init_res["cp_data"]["x"],
        init_res["cp_data"]["cp"],
    )
    err_init = total_cp_error(cp_target, cp_init)

    if SETTINGS.get("constraints", {}).get("CL", {}).get("enabled", False):
        SETTINGS["constraints"]["CL"]["target"] = float(target_res["polar"]["CL"])

    return {
        "x": x,
        "yu_init": yu_init,
        "yl_init": yl_init,
        "cp_target": cp_target,
        "err_init": err_init,
    }


def _build_candidates(upper_centers, lower_centers):
    cand_upper_raw = get_midpoint_candidates(upper_centers)
    cand_lower_raw = get_midpoint_candidates(lower_centers)

    candidates = []
    for c in cand_upper_raw:
        candidates.append({"side": "UPPER", "x": float(c["x"]), "interval_id": c["interval_id"]})
    for c in cand_lower_raw:
        candidates.append({"side": "LOWER", "x": float(c["x"]), "interval_id": c["interval_id"]})
    return candidates


def _prepare_state_at_ndv(problem, target_ndv, selector_indicator, workdir):
    x = problem["x"]
    yu_init = problem["yu_init"]
    yl_init = problem["yl_init"]
    cp_target = problem["cp_target"]
    current_best_error = float(problem["err_init"])

    upper_centers, lower_centers = build_side_specific_initial_centers(
        SETTINGS["optimization"]["adaptive"]["n0"]
    )
    a0 = np.zeros(len(upper_centers) + len(lower_centers))

    adaptive_out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a0=a0,
        label="diag_level_0",
        current_best_error=current_best_error,
        workdir=Path(workdir) / "levels",
    )
    current_best_error = adaptive_out["err_opt"]

    if target_ndv < adaptive_out["ndv_total"]:
        raise ValueError(
            f"Requested VALIDATE_AT_NDV={target_ndv} but initial adaptive level has ndv={adaptive_out['ndv_total']}."
        )

    while adaptive_out["ndv_total"] < target_ndv:
        candidates = _build_candidates(upper_centers, lower_centers)
        if len(candidates) == 0:
            break

        pred_context = None
        if str(selector_indicator).upper() == "PRED":
            pred_context = _prepare_pred_level_context(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                upper_centers=upper_centers,
                lower_centers=lower_centers,
                a_opt=np.asarray(adaptive_out["a_opt"], dtype=float),
                current_best_error=current_best_error,
                workdir=Path(workdir) / "selector_pred_context",
            )

        from adaptive_scoring import score_candidate

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
                workdir=Path(workdir) / "selector_scoring",
                indicator=selector_indicator,
                pred_context=pred_context,
            )
            scored.append(
                {
                    "side": info["side"],
                    "x": info["x"],
                    "interval_id": cand["interval_id"],
                    "score": info["score"],
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["x"]))

        current_ndv = len(upper_centers) + len(lower_centers)
        n_remaining = target_ndv - current_ndv
        n_add = _compute_adaptive_nadd(current_ndv, len(scored))
        n_add = min(n_add, n_remaining)

        if n_add <= 0:
            break

        chosen = scored[:n_add]

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

        level_label = f"diag_level_{len(new_upper) + len(new_lower)}"

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
            workdir=Path(workdir) / "levels",
        )

        current_best_error = adaptive_out["err_opt"]
        upper_centers = new_upper
        lower_centers = new_lower

    return {
        "x": x,
        "yu_init": yu_init,
        "yl_init": yl_init,
        "cp_target": cp_target,
        "upper_centers": list(upper_centers),
        "lower_centers": list(lower_centers),
        "a_opt": np.asarray(adaptive_out["a_opt"], dtype=float).copy(),
        "err_opt": float(adaptive_out["err_opt"]),
    }


def _find_candidate(candidates, side, x_target, tol=1.0e-12):
    side = str(side).upper()
    for cand in candidates:
        if str(cand["side"]).upper() == side and abs(float(cand["x"]) - float(x_target)) <= tol:
            return cand
    raise ValueError(f"Candidate not found: side={side}, x={x_target}")


def _compute_pred_breakdown(state, candidate, workdir):
    x = state["x"]
    yu_init = state["yu_init"]
    yl_init = state["yl_init"]
    cp_target = state["cp_target"]
    upper_centers = state["upper_centers"]
    lower_centers = state["lower_centers"]
    a_opt = state["a_opt"]
    current_best_error = state["err_opt"]

    side, xc, new_upper, new_lower, a_base, new_idx, old_to_new = _build_extended_space(
        active_upper=upper_centers,
        active_lower=lower_centers,
        active_a=a_opt,
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
        workdir=Path(workdir),
        side=side,
        xc=xc,
    )

    base_item = _evaluate_objective_state(objective, a_base)

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * len(a_base)
    pred_rel_step = SETTINGS["optimization"]["pred_fd_rel_step"]
    pred_abs_step_floor = SETTINGS["optimization"]["pred_fd_abs_step_floor"]
    reg = SETTINGS["optimization"]["pred_hessian_reg"]

    pred_context = _prepare_pred_level_context(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a_opt=a_opt,
        current_best_error=current_best_error,
        workdir=Path(workdir) / "pred_context",
    )

    enabled_aero = _enabled_aero_names()

    grad_j, grad_metrics_aero, _ = _compute_full_aero_gradients(
        objective=objective,
        a_base=a_base,
        enabled_metric_names=enabled_aero,
        bounds=bounds,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item,
    )

    yu, yl = _rebuild_geometry_from_a(
        x, yu_init, yl_init, new_upper, new_lower, a_base
    )
    grad_geom_base = _compute_geometric_gradients_analytic(x, new_upper, new_lower)

    active_entries, G = _build_active_ikkt_system(
        x=x,
        yu=yu,
        yl=yl,
        metrics=base_item["metrics"],
        upper_centers=new_upper,
        lower_centers=new_lower,
        grad_metrics_aero=grad_metrics_aero,
        grad_geom_base=grad_geom_base,
    )

    Z = _null_space_from_constraint_columns(G, len(a_base))
    d_y, _, A = _compute_restoration_step(active_entries, G, len(a_base))

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

    H_red = Z.T @ H_full @ Z
    gamma_no_rest = Z.T @ grad_j
    gamma_full = Z.T @ (grad_j + H_full @ d_y)

    eig_H_full = np.linalg.eigvalsh(H_full) if H_full.size > 0 else np.array([], dtype=float)

    if H_red.size == 0:
        eig_H_red = np.array([], dtype=float)
        cond_raw = np.nan
        cond_reg = np.nan
        delta_no_rest = 0.0
        delta_full = 0.0
        eig_min = np.nan
        eig_max = np.nan
    else:
        eig_H_red = np.linalg.eigvalsh(H_red)
        eig_min = float(np.min(eig_H_red))
        eig_max = float(np.max(eig_H_red))
        try:
            cond_raw = float(np.linalg.cond(H_red))
        except np.linalg.LinAlgError:
            cond_raw = np.inf

        H_red_reg = H_red + float(reg) * np.eye(H_red.shape[0])
        try:
            cond_reg = float(np.linalg.cond(H_red_reg))
        except np.linalg.LinAlgError:
            cond_reg = np.inf

        Hinv = np.linalg.pinv(H_red_reg)
        delta_no_rest = 0.5 * float(gamma_no_rest.T @ Hinv @ gamma_no_rest)
        delta_full = 0.5 * float(gamma_full.T @ Hinv @ gamma_full)

    return {
        "side": side,
        "x": xc,
        "rel_step": pred_rel_step,
        "g": grad_j.copy(),
        "Z": Z.copy(),
        "A": np.asarray(A, dtype=float).copy(),
        "H_full": H_full.copy(),
        "H_red": H_red.copy(),
        "eig_H_full": eig_H_full.copy(),
        "eig_H_red": eig_H_red.copy(),
        "raw_grad": float(grad_j[new_idx]),
        "grad_norm": float(np.linalg.norm(grad_j)),
        "dy_new": float(d_y[new_idx]),
        "dy_norm": float(np.linalg.norm(d_y)),
        "gamma_norm": float(np.linalg.norm(gamma_full)),
        "gamma_no_rest_norm": float(np.linalg.norm(gamma_no_rest)),
        "delta_pred_full": float(delta_full),
        "delta_pred_no_rest": float(delta_no_rest),
        "h_nn": float(h_nn),
        "h_col_maxabs": float(np.max(np.abs(h_col))) if len(h_col) > 0 else 0.0,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "cond_raw": cond_raw,
        "cond_reg": cond_reg,
        "n_active": len(active_entries),
        "n_free": int(H_red.shape[0]) if H_red.ndim == 2 else 0,
        "active_names": [
            f"{e['name']}@{e['x']:.3f}" if e["name"] == "thickness_stations" else e["name"]
            for e in active_entries
        ],
    }


def _compute_real_delta(state, candidate, workdir):
    x = state["x"]
    yu_init = state["yu_init"]
    yl_init = state["yl_init"]
    cp_target = state["cp_target"]
    upper_centers = state["upper_centers"]
    lower_centers = state["lower_centers"]
    a_opt = state["a_opt"]
    err_current = float(state["err_opt"])

    side, xc, new_upper, new_lower, _, _, _ = _build_extended_space(
        active_upper=upper_centers,
        active_lower=lower_centers,
        active_a=a_opt,
        candidate=candidate,
    )

    new_a0 = lift_a_to_new_side_centers(
        old_upper=upper_centers,
        old_lower=lower_centers,
        old_a=a_opt,
        new_upper=new_upper,
        new_lower=new_lower,
    )

    label = f"real_reopt_{side}_{xc:.6f}".replace(".", "p")
    out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=new_upper,
        lower_centers=new_lower,
        a0=new_a0,
        label=label,
        current_best_error=err_current,
        workdir=Path(workdir),
    )

    return float(err_current - out["err_opt"])


def main():
    run_dir = Path("run_diagnostic_pred")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    SETTINGS["initial_shape"]["random_seed"] = SEED
    SETTINGS["run"]["seed"] = SEED
    SETTINGS["xfoil"]["working_dir"] = run_dir / f"seed_{SEED}"
    clean_workdir(SETTINGS["xfoil"]["working_dir"])

    problem = _build_problem(SEED, SETTINGS["xfoil"]["working_dir"])
    state = _prepare_state_at_ndv(
        problem=problem,
        target_ndv=VALIDATE_AT_NDV,
        selector_indicator=STATE_SELECTOR,
        workdir=run_dir / "state_build",
    )

    print("===== DIAGNOSTIC PRED =====")
    print(f"seed           = {SEED}")
    print(f"validate_ndv   = {VALIDATE_AT_NDV}")
    print(f"state_selector = {STATE_SELECTOR}")
    print(f"current_err    = {state['err_opt']:.6e}")
    print(f"upper_centers  = {state['upper_centers']}")
    print(f"lower_centers  = {state['lower_centers']}")

    candidates = _build_candidates(state["upper_centers"], state["lower_centers"])
    test_candidates = [
        _find_candidate(candidates, side, x_val) for side, x_val in CANDIDATES_TO_TEST
    ]

    real_delta = {}
    if DO_REAL_REOPT:
        print("\n===== REAL MINI-REOPT =====")
        for cand in test_candidates:
            key = (cand["side"], round(float(cand["x"]), 6))
            real_delta[key] = _compute_real_delta(
                state=state,
                candidate=cand,
                workdir=run_dir / "real_reopt",
            )
            print(
                f"REAL [{cand['side']:<5s} x={cand['x']:.6f}]  "
                f"delta_real={real_delta[key]:.6e}"
            )

    old_rel = SETTINGS["optimization"].get("pred_fd_rel_step", SETTINGS["optimization"]["fd_rel_step"])
    old_abs = SETTINGS["optimization"].get("pred_fd_abs_step_floor", SETTINGS["optimization"]["fd_abs_step_floor"])
    old_reg = SETTINGS["optimization"].get("pred_hessian_reg", 1.0e-8)

    rows = []

    try:
        for reg in REG_LIST:
            for h_abs in H_ABS_LIST:
                SETTINGS["optimization"]["pred_fd_rel_step"] = H_REL_STEP
                SETTINGS["optimization"]["pred_fd_abs_step_floor"] = h_abs
                SETTINGS["optimization"]["pred_hessian_reg"] = reg

                print("\n============================================================")
                print(f"SWEEP CASE: h_abs={h_abs:.6e}  rel_step={H_REL_STEP:.6e}  reg={reg:.6e}")
                print("============================================================")

                case_rows = []
                for cand in test_candidates:
                    out = _compute_pred_breakdown(
                        state=state,
                        candidate=cand,
                        workdir=run_dir / f"diag_h_{h_abs:.0e}_reg_{reg:.0e}",
                    )

                    key = (out["side"], round(float(out["x"]), 6))
                    delta_real = real_delta.get(key, np.nan)

                    row = {
                        "side": out["side"],
                        "x": float(out["x"]),
                        "h_abs": float(h_abs),
                        "rel_step": float(H_REL_STEP),
                        "reg": float(reg),
                        "delta_pred_full": out["delta_pred_full"],
                        "delta_pred_no_rest": out["delta_pred_no_rest"],
                        "delta_real": float(delta_real),
                        "raw_grad": out["raw_grad"],
                        "grad_norm": out["grad_norm"],
                        "dy_new": out["dy_new"],
                        "dy_norm": out["dy_norm"],
                        "gamma_norm": out["gamma_norm"],
                        "gamma_no_rest_norm": out["gamma_no_rest_norm"],
                        "h_nn": out["h_nn"],
                        "h_col_maxabs": out["h_col_maxabs"],
                        "eig_min": out["eig_min"],
                        "eig_max": out["eig_max"],
                        "cond_raw": out["cond_raw"],
                        "cond_reg": out["cond_reg"],
                        "n_active": out["n_active"],
                        "n_free": out["n_free"],
                        "active_names": "|".join(out["active_names"]),
                        "rho_full": (delta_real / out["delta_pred_full"]) if abs(out["delta_pred_full"]) > 1.0e-16 else np.nan,
                        "rho_no_rest": (delta_real / out["delta_pred_no_rest"]) if abs(out["delta_pred_no_rest"]) > 1.0e-16 else np.nan,
                    }
                    case_rows.append(row)
                    rows.append(row)

                    if WRITE_MATRIX_CSV:
                        out_for_save = dict(out)
                        out_for_save["delta_real"] = float(delta_real)
                        csv_path = _save_candidate_matrix_csv(
                            out_for_save,
                            save_dir=run_dir / "candidate_csv",
                            h_abs=h_abs,
                            reg=reg,
                        )
                        print(f"  saved CSV -> {csv_path.name}")

                case_rows.sort(key=lambda z: (-z["delta_pred_full"], z["x"]))

                print("rank | side  | x        | d_pred_full    | d_pred_no_rest | d_real         | rho_full | h_nn         | h_col_max     | eig_min       | cond_reg")
                for i, row in enumerate(case_rows, start=1):
                    print(
                        f"{i:4d} | "
                        f"{row['side']:<5s} | "
                        f"{row['x']:.6f} | "
                        f"{row['delta_pred_full']:.6e} | "
                        f"{row['delta_pred_no_rest']:.6e} | "
                        f"{row['delta_real']:.6e} | "
                        f"{row['rho_full']:.6e} | "
                        f"{row['h_nn']:.6e} | "
                        f"{row['h_col_maxabs']:.6e} | "
                        f"{row['eig_min']:.6e} | "
                        f"{row['cond_reg']:.6e}"
                    )

    finally:
        SETTINGS["optimization"]["pred_fd_rel_step"] = old_rel
        SETTINGS["optimization"]["pred_fd_abs_step_floor"] = old_abs
        SETTINGS["optimization"]["pred_hessian_reg"] = old_reg

    if WRITE_SUMMARY_CSV:
        csv_path = run_dir / f"diagnostic_pred_summary_ndv_{VALIDATE_AT_NDV}.csv"
        fieldnames = [
            "side",
            "x",
            "h_abs",
            "rel_step",
            "reg",
            "delta_pred_full",
            "delta_pred_no_rest",
            "delta_real",
            "raw_grad",
            "grad_norm",
            "dy_new",
            "dy_norm",
            "gamma_norm",
            "gamma_no_rest_norm",
            "h_nn",
            "h_col_maxabs",
            "eig_min",
            "eig_max",
            "cond_raw",
            "cond_reg",
            "n_active",
            "n_free",
            "active_names",
            "rho_full",
            "rho_no_rest",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nSummary CSV written to: {csv_path}")


if __name__ == "__main__":
    main()

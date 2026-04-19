from pathlib import Path
import csv
import shutil

import numpy as np

from settings import SETTINGS
from adaptive_utils import get_midpoint_candidates
from adaptive_candidate import (
    _build_extended_space,
    _evaluate_objective_state,
    _make_candidate_objective,
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


BASE_DIR = Path(__file__).resolve().parent
SNAP_ROOT = BASE_DIR / SETTINGS.get("snapshots", {}).get("dir_name", "snapshots")

SNAP_SEED = 0
SNAP_METHOD = "adapt_grad"   # "adapt_grad" | "adapt_ikkt" | "adapt_pred"
SNAP_LEVEL = 1

SNAPSHOT_PATH = SNAP_ROOT / f"seed_{SNAP_SEED}" / SNAP_METHOD / f"level_{SNAP_LEVEL:03d}.npz"

CANDIDATES_TO_TEST = None

H_ABS_LIST = None   # es: [5.0e-4, 1.0e-3] per override manuale
H_REL_STEP = None   # es: 5.0e-3 per override manuale
REG_LIST = None     # es: [1.0e-8, 1.0e-6] per override manuale

WRITE_SUMMARY_CSV = True
WRITE_MATRIX_CSV = True

def _enabled_aero_names():
    names = []
    for name in ("CL", "CD", "CM"):
        spec = SETTINGS.get("constraints", {}).get(name, {})
        if spec.get("enabled", False):
            names.append(name)
    return names


def _restore_snapshot_constraint_targets(data):
    for metric_name in ("CL", "CD", "CM"):
        key = f"constraint_{metric_name}_target"
        if key not in data.files:
            continue

        value = float(np.asarray(data[key]).item())
        if np.isnan(value):
            continue

        if metric_name in SETTINGS.get("constraints", {}):
            SETTINGS["constraints"][metric_name]["target"] = value


def _load_level_snapshot(path):
    path = Path(path)
    required_fields = [
        "ndv_total",
        "x",
        "yu_init",
        "yl_init",
        "upper_centers",
        "lower_centers",
        "a_opt",
        "err_opt",
        "yu_opt",
        "yl_opt",
        "cp_target_upper_x",
        "cp_target_upper_cp",
        "cp_target_lower_x",
        "cp_target_lower_cp",
        "cp_opt_upper_x",
        "cp_opt_upper_cp",
        "cp_opt_lower_x",
        "cp_opt_lower_cp",
        "label",
    ]

    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in required_fields if name not in data.files]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Snapshot {path} is missing required fields: {missing_str}")

        _restore_snapshot_constraint_targets(data)

        return {
            "snapshot_path": path,
            "x": np.array(data["x"], dtype=float, copy=True),
            "yu_init": np.array(data["yu_init"], dtype=float, copy=True),
            "yl_init": np.array(data["yl_init"], dtype=float, copy=True),
            "cp_target": {
                "upper": {
                    "x": np.array(data["cp_target_upper_x"], dtype=float, copy=True),
                    "cp": np.array(data["cp_target_upper_cp"], dtype=float, copy=True),
                },
                "lower": {
                    "x": np.array(data["cp_target_lower_x"], dtype=float, copy=True),
                    "cp": np.array(data["cp_target_lower_cp"], dtype=float, copy=True),
                },
            },
            "upper_centers": np.array(data["upper_centers"], dtype=float, copy=True).tolist(),
            "lower_centers": np.array(data["lower_centers"], dtype=float, copy=True).tolist(),
            "a_opt": np.array(data["a_opt"], dtype=float, copy=True),
            "err_opt": float(np.asarray(data["err_opt"]).item()),
            "ndv_total": int(np.asarray(data["ndv_total"]).item()),
            "label": str(np.asarray(data["label"]).item()),
        }


def _build_candidates(upper_centers, lower_centers):
    cand_upper_raw = get_midpoint_candidates(upper_centers)
    cand_lower_raw = get_midpoint_candidates(lower_centers)

    candidates = []
    for cand in cand_upper_raw:
        candidates.append(
            {
                "side": "UPPER",
                "x": float(cand["x"]),
                "interval_id": cand["interval_id"],
            }
        )
    for cand in cand_lower_raw:
        candidates.append(
            {
                "side": "LOWER",
                "x": float(cand["x"]),
                "interval_id": cand["interval_id"],
            }
        )
    return candidates


def _find_candidate(candidates, side, x_target, tol=1.0e-12):
    side = str(side).upper()
    for cand in candidates:
        if str(cand["side"]).upper() != side:
            continue
        if abs(float(cand["x"]) - float(x_target)) <= tol:
            return cand
    raise ValueError(f"Candidate not found: side={side}, x={x_target}")


def _select_candidates(candidates, selected):
    if selected is None:
        return list(candidates)
    return [_find_candidate(candidates, side, x_val) for side, x_val in selected]


def _safe_tag_float(x):
    return f"{float(x):.1e}".replace("+", "").replace(".", "p").replace("-", "m")


def _resolve_diagnostic_sweep_params():
    opt_cfg = SETTINGS.get("optimization", {})

    default_h_abs = opt_cfg.get(
        "pred_fd_abs_step_floor",
        opt_cfg.get("fd_abs_step_floor", 1.0e-6),
    )
    default_h_rel = opt_cfg.get(
        "pred_fd_rel_step",
        opt_cfg.get("fd_rel_step", 1.0e-3),
    )
    default_reg = opt_cfg.get("pred_hessian_reg", 1.0e-8)

    h_abs_list = (
        [float(v) for v in H_ABS_LIST]
        if H_ABS_LIST is not None
        else [float(default_h_abs)]
    )
    h_rel_step = float(H_REL_STEP) if H_REL_STEP is not None else float(default_h_rel)
    reg_list = (
        [float(v) for v in REG_LIST]
        if REG_LIST is not None
        else [float(default_reg)]
    )

    return h_abs_list, h_rel_step, reg_list


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


def _compute_pred_breakdown(state, candidate, pred_context, workdir):
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

    grad_j, grad_metrics_aero, _ = _compute_full_aero_gradients(
        objective=objective,
        a_base=a_base,
        enabled_metric_names=_enabled_aero_names(),
        bounds=bounds,
        rel_step=pred_rel_step,
        abs_step_floor=pred_abs_step_floor,
        base_item=base_item,
    )

    yu, yl = _rebuild_geometry_from_a(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=new_upper,
        lower_centers=new_lower,
        a=a_base,
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
    H_full[np.ix_(old_to_new, old_to_new)] = pred_context["H_k"]

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
            f"{entry['name']}@{entry['x']:.3f}" if entry["name"] == "thickness_stations" else entry["name"]
            for entry in active_entries
        ],
    }


def main():
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_PATH}")

    snapshot = _load_level_snapshot(SNAPSHOT_PATH)
    h_abs_list, h_rel_step, reg_list = _resolve_diagnostic_sweep_params()
    run_dir = BASE_DIR / "run_diagnostic_pred" / snapshot["snapshot_path"].stem

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("===== DIAGNOSTIC PRED =====")
    print(f"snapshot       = {snapshot['snapshot_path']}")
    print(f"label          = {snapshot['label']}")
    print(f"ndv_total      = {snapshot['ndv_total']}")
    print(f"current_err    = {snapshot['err_opt']:.6e}")
    print(f"upper_centers  = {snapshot['upper_centers']}")
    print(f"lower_centers  = {snapshot['lower_centers']}")

    candidates = _build_candidates(snapshot["upper_centers"], snapshot["lower_centers"])
    test_candidates = _select_candidates(candidates, CANDIDATES_TO_TEST)

    if len(test_candidates) == 0:
        raise RuntimeError("No candidates available for the selected snapshot.")

    old_rel = SETTINGS["optimization"].get(
        "pred_fd_rel_step",
        SETTINGS["optimization"]["fd_rel_step"],
    )
    old_abs = SETTINGS["optimization"].get(
        "pred_fd_abs_step_floor",
        SETTINGS["optimization"]["fd_abs_step_floor"],
    )
    old_reg = SETTINGS["optimization"].get("pred_hessian_reg", 1.0e-8)

    rows = []

    try:
        for reg in reg_list:
            for h_abs in h_abs_list:
                SETTINGS["optimization"]["pred_fd_rel_step"] = h_rel_step
                SETTINGS["optimization"]["pred_fd_abs_step_floor"] = h_abs
                SETTINGS["optimization"]["pred_hessian_reg"] = reg

                pred_context = _prepare_pred_level_context(
                    x=snapshot["x"],
                    yu_init=snapshot["yu_init"],
                    yl_init=snapshot["yl_init"],
                    cp_target=snapshot["cp_target"],
                    upper_centers=snapshot["upper_centers"],
                    lower_centers=snapshot["lower_centers"],
                    a_opt=snapshot["a_opt"],
                    current_best_error=snapshot["err_opt"],
                    workdir=run_dir / f"pred_context_h_{_safe_tag_float(h_abs)}_reg_{_safe_tag_float(reg)}",
                )

                print("\n============================================================")
                print(f"SWEEP CASE: h_abs={h_abs:.6e}  rel_step={h_rel_step:.6e}  reg={reg:.6e}")
                print("============================================================")

                case_rows = []
                for cand in test_candidates:
                    out = _compute_pred_breakdown(
                        state=snapshot,
                        candidate=cand,
                        pred_context=pred_context,
                        workdir=run_dir / f"diag_h_{_safe_tag_float(h_abs)}_reg_{_safe_tag_float(reg)}",
                    )

                    row = {
                        "side": out["side"],
                        "x": float(out["x"]),
                        "h_abs": float(h_abs),
                        "rel_step": float(h_rel_step),
                        "reg": float(reg),
                        "delta_pred_full": out["delta_pred_full"],
                        "delta_pred_no_rest": out["delta_pred_no_rest"],
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
                    }
                    case_rows.append(row)
                    rows.append(row)

                    if WRITE_MATRIX_CSV:
                        csv_path = _save_candidate_matrix_csv(
                            out=out,
                            save_dir=run_dir / "candidate_csv",
                            h_abs=h_abs,
                            reg=reg,
                        )
                        print(f"  saved CSV -> {csv_path.name}")

                case_rows.sort(key=lambda item: (-item["delta_pred_full"], item["x"]))

                print("rank | side  | x        | d_pred_full    | d_pred_no_rest | h_nn         | h_col_max     | eig_min       | cond_reg")
                for idx, row in enumerate(case_rows, start=1):
                    print(
                        f"{idx:4d} | "
                        f"{row['side']:<5s} | "
                        f"{row['x']:.6f} | "
                        f"{row['delta_pred_full']:.6e} | "
                        f"{row['delta_pred_no_rest']:.6e} | "
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
        csv_path = run_dir / f"diagnostic_pred_summary_{snapshot['snapshot_path'].stem}.csv"
        fieldnames = [
            "side",
            "x",
            "h_abs",
            "rel_step",
            "reg",
            "delta_pred_full",
            "delta_pred_no_rest",
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
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"\nSummary CSV written to: {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from settings import SETTINGS, apply_cfg_overrides, validate_settings
from geometry import (
    apply_hicks_henne_deformation,
    build_naca0012_surfaces,
    build_normal_peak_fd_steps,
    build_random_initial_geometry,
    write_dat,
)
from adaptive_utils import build_interval_candidates, build_side_specific_initial_centers


EPS = 1.0e-300


def load_runtime_imports():
    global run_aero
    global reduce_to_best_candidate_per_interval
    global prepare_gn_schur_level_context, score_candidate
    global split_upper_lower_cp_from_x
    global make_objective, total_cp_error
    global _compute_explicit_jac
    global run_target_aero
    global clear_cmplxfoil_solver_cache
    global run_cmplxfoil_coords

    from aero_wrapper import run_aero
    from adaptive_strategy import reduce_to_best_candidate_per_interval
    from adaptive_scoring import prepare_gn_schur_level_context, score_candidate
    from cp_utils import split_upper_lower_cp_from_x
    from objective import make_objective, total_cp_error
    from optimization import _compute_explicit_jac
    from experiment_utils import run_target_aero
    from cmplxfoil_wrapper import clear_cmplxfoil_solver_cache, run_cmplxfoil_coords


@contextmanager
def temporary_derivative_mode(mode):
    old = SETTINGS.setdefault("aero", {}).get("derivatives", "fd")
    SETTINGS["aero"]["derivatives"] = str(mode).lower()
    try:
        yield
    finally:
        SETTINGS["aero"]["derivatives"] = old


@contextmanager
def pushd(path):
    old = Path.cwd()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_vector_csv(path, name, values):
    rows = [{"index": i, name: float(v)} for i, v in enumerate(np.asarray(values, dtype=float))]
    write_csv(path, ["index", name], rows)


def safe_float(value):
    try:
        return float(np.real(value))
    except Exception:
        return float("nan")


def load_snapshot(path):
    z = np.load(path, allow_pickle=False)
    return {
        "source": str(path),
        "mode": "snapshot",
        "ndv_total": int(np.asarray(z["ndv_total"]).item()),
        "x": np.asarray(z["x"], dtype=float),
        "yu_init": np.asarray(z["yu_init"], dtype=float),
        "yl_init": np.asarray(z["yl_init"], dtype=float),
        "upper_centers": np.asarray(z["upper_centers"], dtype=float),
        "lower_centers": np.asarray(z["lower_centers"], dtype=float),
        "a_vec": np.asarray(z["a_opt"], dtype=float),
        "current_best_error": float(np.asarray(z["err_opt"]).item()),
        "cp_target": {
            "upper": {
                "x": np.asarray(z["cp_target_upper_x"], dtype=float),
                "cp": np.asarray(z["cp_target_upper_cp"]),
            },
            "lower": {
                "x": np.asarray(z["cp_target_lower_x"], dtype=float),
                "cp": np.asarray(z["cp_target_lower_cp"]),
            },
        },
    }


def snapshot_sort_key(path):
    try:
        z = np.load(path, allow_pickle=False)
        ndv = int(np.asarray(z["ndv_total"]).item())
    except Exception:
        ndv = 10**9
    m = re.search(r"level_(\d+)\.npz$", path.name)
    idx = int(m.group(1)) if m else 10**9
    return ndv, idx, str(path)


def find_snapshot(project_root, seed, level=None, explicit=None):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            matches = []
            for cand in project_root.rglob(Path(explicit).name):
                if cand.suffix != ".npz":
                    continue
                parts = set(cand.parts)
                if f"seed_{int(seed)}" in parts:
                    matches.append(cand)
            if matches:
                return sorted(matches, key=snapshot_sort_key)[0]
            raise FileNotFoundError(f"Snapshot not found: {path}")
        return path

    candidates = []
    preferred = project_root / "snapshots" / f"seed_{int(seed)}" / "adapt_grad"
    if preferred.exists():
        candidates.extend(sorted(preferred.glob("*.npz")))

    if not candidates:
        for path in project_root.rglob("*.npz"):
            parts = set(path.parts)
            if f"seed_{int(seed)}" in parts and "adapt_grad" in parts:
                candidates.append(path)

    candidates = sorted(set(candidates), key=snapshot_sort_key)
    if not candidates:
        return None

    if level is None:
        return candidates[0]

    level = int(level)
    by_ndv = []
    by_name = []
    for path in candidates:
        try:
            z = np.load(path, allow_pickle=False)
            ndv = int(np.asarray(z["ndv_total"]).item())
        except Exception:
            ndv = None
        if ndv == level:
            by_ndv.append(path)
        if path.name == f"level_{level:03d}.npz":
            by_name.append(path)

    if by_ndv:
        return sorted(by_ndv, key=snapshot_sort_key)[0]
    if by_name:
        return sorted(by_name, key=snapshot_sort_key)[0]
    return None


def build_minimal_state(seed, out_dir):
    workdir = out_dir / "minimal_state"
    workdir.mkdir(parents=True, exist_ok=True)

    x, yu_target, yl_target = build_naca0012_surfaces(
        n_points=SETTINGS["geom"]["n_points"],
        thickness=SETTINGS["geom"]["thickness"],
    )
    target_dat = workdir / "target_airfoil.dat"
    write_dat(target_dat, x, yu_target, yl_target, name="TARGET_NACA0012")
    target_res = run_target_aero(
        airfoil_dat=target_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=workdir / "target_run",
    )
    if not target_res["success"]:
        raise RuntimeError("Target aero run failed while building minimal diagnostic state.")
    cp_target = split_upper_lower_cp_from_x(target_res["cp_data"]["x"], target_res["cp_data"]["cp"])

    init_geom = build_random_initial_geometry(
        x=x,
        yu_target=yu_target,
        yl_target=yl_target,
        seed=int(seed),
        amp=SETTINGS["initial_shape"]["random_amp"],
        order=SETTINGS["initial_shape"]["bernstein_order"],
    )
    yu_init = init_geom["yu_init"]
    yl_init = init_geom["yl_init"]

    init_dat = workdir / "initial_airfoil.dat"
    write_dat(init_dat, x, yu_init, yl_init, name="INITIAL_RANDOM")
    init_res = run_aero(
        airfoil_dat=init_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=workdir / "initial_run",
    )
    if not init_res["success"]:
        raise RuntimeError("Initial aero run failed while building minimal diagnostic state.")

    cp_init = split_upper_lower_cp_from_x(init_res["cp_data"]["x"], init_res["cp_data"]["cp"])
    err_init = total_cp_error(cp_target, cp_init)
    upper, lower = build_side_specific_initial_centers(int(SETTINGS["optimization"]["adaptive"]["n0"]))
    a_vec = np.zeros(len(upper) + len(lower), dtype=float)
    return {
        "source": "minimal_state",
        "mode": "minimal",
        "ndv_total": len(a_vec),
        "x": x,
        "yu_init": yu_init,
        "yl_init": yl_init,
        "upper_centers": np.asarray(upper, dtype=float),
        "lower_centers": np.asarray(lower, dtype=float),
        "a_vec": a_vec,
        "current_best_error": float(err_init),
        "cp_target": cp_target,
    }


def make_diag_objective(state, workdir, label, current_best_error=None):
    return make_objective(
        x=state["x"],
        yu_init=state["yu_init"],
        yl_init=state["yl_init"],
        cp_target=state["cp_target"],
        upper_centers=state["upper_centers"],
        lower_centers=state["lower_centers"],
        hh_power=SETTINGS["optimization"]["hh_power"],
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / label,
        current_best_error=(
            state["current_best_error"] if current_best_error is None else current_best_error
        ),
    )


def compute_optimizer_gradient(state, mode, workdir):
    a_vec = np.asarray(state["a_vec"], dtype=float)
    objective = make_diag_objective(state, workdir, f"gradient_{mode}")
    objective.set_eval_phase("gradient")

    with temporary_derivative_mode(mode):
        if mode == "cs":
            grad = objective.compute_gradient_cs(a_vec)
            diag = {"mode": "CS", "n_fail_dirs": 0, "fail_indices": []}
        else:
            bmin, bmax = SETTINGS["optimization"]["bounds"]
            bounds = [(bmin, bmax)] * len(a_vec)
            grad = _compute_explicit_jac(
                objective=objective,
                a=a_vec,
                bounds=bounds,
                x=state["x"],
                yu_init=state["yu_init"],
                yl_init=state["yl_init"],
                upper_centers=state["upper_centers"],
                lower_centers=state["lower_centers"],
                target_peak_normal=SETTINGS["optimization"]["opt_fd_target_peak_normal"],
            )
            diag = {"mode": "FD", "n_fail_dirs": None, "fail_indices": []}
    return np.asarray(grad, dtype=float), diag


def gradient_stats(grad_fd, grad_cs):
    grad_fd = np.asarray(grad_fd, dtype=float)
    grad_cs = np.asarray(grad_cs, dtype=float)
    diff = grad_cs - grad_fd
    norm_fd = float(np.linalg.norm(grad_fd))
    norm_cs = float(np.linalg.norm(grad_cs))
    denom = max(norm_fd, EPS)
    denom_cos = max(norm_fd * norm_cs, EPS)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    idx_max = int(np.argmax(np.abs(diff))) if diff.size else -1
    sign_same = np.signbit(grad_fd) == np.signbit(grad_cs)
    both_zero = (grad_fd == 0.0) & (grad_cs == 0.0)
    sign_same = sign_same | both_zero
    n_mismatch = int(np.count_nonzero(~sign_same))
    frac_mismatch = float(n_mismatch / max(len(grad_fd), 1))
    return {
        "norm_fd": norm_fd,
        "norm_cs": norm_cs,
        "norm_ratio_cs_over_fd": norm_cs / max(norm_fd, EPS),
        "relative_l2_error": float(np.linalg.norm(diff) / denom),
        "cosine_similarity": float(np.dot(grad_fd, grad_cs) / denom_cos),
        "max_abs_diff": max_abs,
        "index_max_abs_diff": idx_max,
        "n_sign_mismatch": n_mismatch,
        "fraction_sign_mismatch": frac_mismatch,
    }


def write_gradient_outputs(out_dir, grad_fd, grad_cs):
    write_vector_csv(out_dir / "grad_fd.csv", "grad_fd", grad_fd)
    write_vector_csv(out_dir / "grad_cs.csv", "grad_cs", grad_cs)
    rows = []
    for i, (gfd, gcs) in enumerate(zip(grad_fd, grad_cs)):
        abs_diff = abs(float(gcs) - float(gfd))
        rel_diff = abs_diff / max(abs(float(gfd)), EPS)
        sign_same = (math.copysign(1.0, float(gfd)) == math.copysign(1.0, float(gcs))) or (
            float(gfd) == 0.0 and float(gcs) == 0.0
        )
        rows.append(
            {
                "index": i,
                "grad_fd": float(gfd),
                "grad_cs": float(gcs),
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "sign_same": bool(sign_same),
            }
        )
    write_csv(
        out_dir / "grad_compare.csv",
        ["index", "grad_fd", "grad_cs", "abs_diff", "rel_diff", "sign_same"],
        rows,
    )


def objective_real_value(state, a_vec, workdir, label):
    obj = make_diag_objective(state, workdir, label)
    obj.set_eval_phase("function")
    value = obj(np.asarray(a_vec, dtype=float))
    item = obj.eval_history[-1] if obj.eval_history else {}
    return safe_float(item.get("objective_base", value)), item.get("status", "")


def vector_compare_metrics(v_fd, v_cs):
    v_fd = np.asarray(v_fd, dtype=complex).ravel()
    v_cs = np.asarray(v_cs, dtype=complex).ravel()

    n_fd = float(np.linalg.norm(v_fd))
    n_cs = float(np.linalg.norm(v_cs))

    # Caso importante: entrambe le derivate sono nulle.
    # Questo è un match, non un mismatch.
    zero_tol = 1.0e-14
    if n_fd < zero_tol and n_cs < zero_tol:
        return n_fd, n_cs, 0.0, 1.0

    diff = v_cs - v_fd
    rel_err = float(np.linalg.norm(diff) / max(n_fd, EPS))

    if n_fd < zero_tol or n_cs < zero_tol:
        cosine = 0.0
    else:
        dot = np.vdot(v_fd, v_cs)
        cosine = float(np.real(dot) / max(n_fd * n_cs, EPS))

    return n_fd, n_cs, rel_err, cosine


def split_design_by_side(state, a_vec):
    nu = len(state["upper_centers"])
    a_vec = np.asarray(a_vec)
    return a_vec[:nu], a_vec[nu:]


def build_geometry_from_state(state, a_vec):
    a_upper, a_lower = split_design_by_side(state, a_vec)
    return apply_hicks_henne_deformation(
        x=state["x"],
        yu_base=state["yu_init"],
        yl_base=state["yl_init"],
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=state["upper_centers"],
        lower_centers=state["lower_centers"],
        power=SETTINGS["optimization"]["hh_power"],
    )


def selected_indices(ndv, nmax=5):
    return list(range(min(int(nmax), int(ndv))))


def objective_path_compare(state, out_dir):
    eps = 1.0e-6
    a = np.asarray(state["a_vec"], dtype=float)
    rows = []
    workdir = out_dir / "objective_path_compare_runs"
    objective = make_diag_objective(state, workdir, "objective_path_compare")

    cases = [("base", -1, a)]
    for idx in selected_indices(len(a)):
        a_plus = a.copy()
        a_plus[idx] += eps
        cases.append((f"plus_{idx}", idx, a_plus))

    with temporary_derivative_mode("cs"):
        for case_name, idx, avec in cases:
            try:
                j_real = objective(np.asarray(avec, dtype=float))
                item = objective.eval_history[-1] if objective.eval_history else {}
                j_real_base = item.get("objective_base", j_real)
                status_real = item.get("status", "UNKNOWN")
            except Exception as exc:
                j_real_base = float("nan")
                status_real = f"ERROR: {exc}"

            try:
                j_cs = objective.evaluate_cs(np.asarray(avec, dtype=complex))
                j_cs_real = float(np.real(j_cs))
                status_cs = "OK"
            except Exception as exc:
                j_cs_real = float("nan")
                status_cs = f"ERROR: {exc}"

            j_real_float = safe_float(j_real_base)
            abs_diff = abs(j_real_float - j_cs_real) if np.isfinite(j_real_float) and np.isfinite(j_cs_real) else float("nan")
            rel_diff = abs_diff / max(abs(j_real_float), EPS) if np.isfinite(abs_diff) else float("nan")
            rows.append(
                {
                    "case": case_name,
                    "index": idx,
                    "J_real": j_real_float,
                    "J_cs_realpart": j_cs_real,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "status_real": status_real,
                    "status_cs": status_cs,
                }
            )

    write_csv(
        out_dir / "objective_path_compare.csv",
        ["case", "index", "J_real", "J_cs_realpart", "abs_diff", "rel_diff", "status_real", "status_cs"],
        rows,
    )
    ok_rows = [r for r in rows if r["status_real"] == "OK" and r["status_cs"] == "OK"]
    if not ok_rows:
        match = False
    else:
        match = all(
            np.isfinite(r["abs_diff"])
            and r["abs_diff"] <= 1.0e-8 * max(1.0, abs(r["J_real"]))
            for r in ok_rows
        )
    return {"objective_real_vs_cs_match": bool(match), "rows": rows}


def geometry_derivative_check(state, out_dir):
    h_cs = 1.0e-30
    h_fd = 1.0e-6
    a = np.asarray(state["a_vec"], dtype=float)
    rows = []

    for idx in selected_indices(len(a)):
        a_complex = a.astype(complex)
        a_complex[idx] += 1j * h_cs
        yu_complex, yl_complex = build_geometry_from_state(state, a_complex)
        dyu_cs = np.imag(yu_complex) / h_cs
        dyl_cs = np.imag(yl_complex) / h_cs

        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[idx] += h_fd
        a_minus[idx] -= h_fd
        yu_plus, yl_plus = build_geometry_from_state(state, a_plus)
        yu_minus, yl_minus = build_geometry_from_state(state, a_minus)
        dyu_fd = (yu_plus - yu_minus) / (2.0 * h_fd)
        dyl_fd = (yl_plus - yl_minus) / (2.0 * h_fd)

        n_dyu_fd, n_dyu_cs, rel_dyu, cos_dyu = vector_compare_metrics(dyu_fd, dyu_cs)
        n_dyl_fd, n_dyl_cs, rel_dyl, cos_dyl = vector_compare_metrics(dyl_fd, dyl_cs)
        rows.append(
            {
                "index": idx,
                "h_cs": h_cs,
                "h_fd": h_fd,
                "norm_dyu_fd": n_dyu_fd,
                "norm_dyu_cs": n_dyu_cs,
                "rel_err_dyu": rel_dyu,
                "cosine_dyu": cos_dyu,
                "norm_dyl_fd": n_dyl_fd,
                "norm_dyl_cs": n_dyl_cs,
                "rel_err_dyl": rel_dyl,
                "cosine_dyl": cos_dyl,
            }
        )

    write_csv(
        out_dir / "geometry_derivative_check.csv",
        [
            "index",
            "h_cs",
            "h_fd",
            "norm_dyu_fd",
            "norm_dyu_cs",
            "rel_err_dyu",
            "cosine_dyu",
            "norm_dyl_fd",
            "norm_dyl_cs",
            "rel_err_dyl",
            "cosine_dyl",
        ],
        rows,
    )
    if not rows:
        return {"geometry_cs_match": "UNAVAILABLE", "rows": rows}
    max_rel = max(max(r["rel_err_dyu"], r["rel_err_dyl"]) for r in rows)
    min_cos = min(min(r["cosine_dyu"], r["cosine_dyl"]) for r in rows)
    match = max_rel <= 1.0e-6 and min_cos >= 1.0 - 1.0e-8
    return {"geometry_cs_match": "YES" if match else "NO", "rows": rows}


def cp_complex_response_check(state, out_dir):
    if str(SETTINGS.get("aero", {}).get("backend", "xfoil")).strip().lower() != "cmplxfoil":
        note = "Cp complex response unavailable from current API."
        (out_dir / "cp_complex_response_check.csv").write_text(
            "index,max_abs_imag_cp_upper,max_abs_imag_cp_lower,norm_imag_cp_upper_over_h,norm_imag_cp_lower_over_h,n_nan,status\n",
            encoding="utf-8",
        )
        return {"cp_complex_response": "UNAVAILABLE", "rows": [], "note": note}

    h_cs = 1.0e-30
    a = np.asarray(state["a_vec"], dtype=float)
    rows = []
    workdir = out_dir / "cp_complex_response_runs"
    workdir.mkdir(parents=True, exist_ok=True)
    template_dat = workdir / "_cmplxfoil_template_airfoil.dat"
    write_dat(template_dat, state["x"], np.real(state["yu_init"]), np.real(state["yl_init"]), name="CMPLXFOIL_TEMPLATE")

    with temporary_derivative_mode("cs"):
        for idx in selected_indices(len(a)):
            a_complex = a.astype(complex)
            a_complex[idx] += 1j * h_cs
            yu_complex, yl_complex = build_geometry_from_state(state, a_complex)
            try:
                res = run_cmplxfoil_coords(
                    airfoil_dat=template_dat,
                    x=state["x"],
                    yu=yu_complex,
                    yl=yl_complex,
                    alpha_deg=SETTINGS["xfoil"]["alpha"],
                    reynolds=SETTINGS["xfoil"]["Re"],
                    xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
                    timeout=SETTINGS["xfoil"]["timeout"],
                    working_dir=workdir / f"idx_{idx:03d}",
                    session_key=f"{workdir.resolve()}::idx_{idx:03d}",
                )
                if not res.get("success") or res.get("cp_data") is None:
                    raise RuntimeError("CMPLXFOIL complex Cp run failed")
                cp_candidate = split_upper_lower_cp_from_x(res["cp_data"]["x"], res["cp_data"]["cp"])
                cp_u = np.asarray(cp_candidate["upper"]["cp"])
                cp_l = np.asarray(cp_candidate["lower"]["cp"])
                imag_u = np.imag(cp_u)
                imag_l = np.imag(cp_l)
                n_nan = int(np.count_nonzero(~np.isfinite(cp_u)) + np.count_nonzero(~np.isfinite(cp_l)))
                row = {
                    "index": idx,
                    "max_abs_imag_cp_upper": float(np.max(np.abs(imag_u))) if imag_u.size else 0.0,
                    "max_abs_imag_cp_lower": float(np.max(np.abs(imag_l))) if imag_l.size else 0.0,
                    "norm_imag_cp_upper_over_h": float(np.linalg.norm(imag_u) / h_cs),
                    "norm_imag_cp_lower_over_h": float(np.linalg.norm(imag_l) / h_cs),
                    "n_nan": n_nan,
                    "status": "OK",
                }
            except Exception as exc:
                row = {
                    "index": idx,
                    "max_abs_imag_cp_upper": float("nan"),
                    "max_abs_imag_cp_lower": float("nan"),
                    "norm_imag_cp_upper_over_h": float("nan"),
                    "norm_imag_cp_lower_over_h": float("nan"),
                    "n_nan": -1,
                    "status": f"ERROR: {exc}",
                }
            rows.append(row)

    write_csv(
        out_dir / "cp_complex_response_check.csv",
        [
            "index",
            "max_abs_imag_cp_upper",
            "max_abs_imag_cp_lower",
            "norm_imag_cp_upper_over_h",
            "norm_imag_cp_lower_over_h",
            "n_nan",
            "status",
        ],
        rows,
    )
    ok_rows = [r for r in rows if r["status"] == "OK"]
    if not ok_rows:
        status = "ERROR" if rows else "UNAVAILABLE"
    elif all(
        r["max_abs_imag_cp_upper"] == 0.0 and r["max_abs_imag_cp_lower"] == 0.0
        for r in ok_rows
    ):
        status = "ZERO"
    else:
        status = "OK"
    return {"cp_complex_response": status, "rows": rows, "note": ""}


def make_cp_case(cp_target, cp_current_upper, cp_current_lower):
    return {
        "upper": {
            "x": np.asarray(cp_target["upper"]["x"], dtype=float),
            "cp": np.asarray(cp_current_upper),
        },
        "lower": {
            "x": np.asarray(cp_target["lower"]["x"], dtype=float),
            "cp": np.asarray(cp_current_lower),
        },
    }


def total_cp_error_complex_check(state, out_dir):
    h_cs = 1.0e-30
    h_fd = 1.0e-6
    cp_target = state["cp_target"]

    cp_u_base = np.asarray(cp_target["upper"]["cp"], dtype=float) + 0.01 * np.sin(
        np.linspace(0.0, np.pi, len(cp_target["upper"]["cp"]))
    )
    cp_l_base = np.asarray(cp_target["lower"]["cp"], dtype=float) - 0.01 * np.cos(
        np.linspace(0.0, np.pi, len(cp_target["lower"]["cp"]))
    )

    rows = []
    tests = []

    if len(cp_u_base) > 0:
        tests.append(("upper", len(cp_u_base) // 2))
    if len(cp_l_base) > 0:
        tests.append(("lower", len(cp_l_base) // 2))

    def _row_complex_safe(row, tol_abs=1.0e-10, tol_rel=1.0e-6):
        if row.get("status") != "OK":
            return False

        d_cs = row.get("imag_over_h", np.nan)
        d_fd = row.get("dJ_fd", np.nan)

        if not (np.isfinite(d_cs) and np.isfinite(d_fd)):
            return False

        abs_diff = abs(float(d_cs) - float(d_fd))
        scale = max(1.0, abs(float(d_cs)), abs(float(d_fd)))

        return abs_diff <= tol_abs or (abs_diff / scale) <= tol_rel

    for side, idx in tests:
        cp_u_complex = cp_u_base.astype(complex)
        cp_l_complex = cp_l_base.astype(complex)

        if side == "upper":
            cp_u_complex[idx] += 1j * h_cs
        else:
            cp_l_complex[idx] += 1j * h_cs

        j_complex = total_cp_error(
            cp_target,
            make_cp_case(cp_target, cp_u_complex, cp_l_complex),
        )
        dJ_cs = float(np.imag(j_complex) / h_cs)

        cp_u_plus = cp_u_base.copy()
        cp_l_plus = cp_l_base.copy()
        cp_u_minus = cp_u_base.copy()
        cp_l_minus = cp_l_base.copy()

        if side == "upper":
            cp_u_plus[idx] += h_fd
            cp_u_minus[idx] -= h_fd
        else:
            cp_l_plus[idx] += h_fd
            cp_l_minus[idx] -= h_fd

        j_plus = total_cp_error(
            cp_target,
            make_cp_case(cp_target, cp_u_plus, cp_l_plus),
        )
        j_minus = total_cp_error(
            cp_target,
            make_cp_case(cp_target, cp_u_minus, cp_l_minus),
        )

        dJ_fd = float(np.real(j_plus - j_minus) / (2.0 * h_fd))
        abs_diff = abs(dJ_cs - dJ_fd)
        scale = max(1.0, abs(dJ_cs), abs(dJ_fd))
        rel_diff = abs_diff / scale

        row = {
            "side": side,
            "index": idx,
            "h_cs": h_cs,
            "h_fd": h_fd,
            "J_complex_type": type(j_complex).__name__,
            "imag_over_h": dJ_cs,
            "dJ_fd": dJ_fd,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "status": "OK",
        }
        row["safe"] = _row_complex_safe(row)
        rows.append(row)

    write_csv(
        out_dir / "total_cp_error_complex_check.csv",
        [
            "side",
            "index",
            "h_cs",
            "h_fd",
            "J_complex_type",
            "imag_over_h",
            "dJ_fd",
            "abs_diff",
            "rel_diff",
            "status",
            "safe",
        ],
        rows,
    )

    safe = bool(rows) and all(bool(r["safe"]) for r in rows)

    return {"total_cp_error_complex_safe": bool(safe), "rows": rows}

def directional_derivative_check(state, grad_fd, grad_cs, out_dir):
    a = np.asarray(state["a_vec"], dtype=float)
    ndv = len(a)
    rows = []
    rng = np.random.default_rng(123456)
    directions = []
    for i in range(min(5, ndv)):
        v = np.zeros(ndv, dtype=float)
        v[i] = 1.0
        directions.append((f"unit_{i}", "unit", v))
    for i in range(5):
        v = rng.normal(size=ndv)
        n = np.linalg.norm(v)
        if n > 0.0:
            v = v / n
        directions.append((f"random_{i}", "random", v))

    workdir = out_dir / "directional_runs"
    for direction_id, direction_type, v in directions:
        grad_fd_dot = float(np.dot(grad_fd, v))
        grad_cs_dot = float(np.dot(grad_cs, v))
        for h in (1.0e-4, 1.0e-5, 1.0e-6):
            jp, sp = objective_real_value(state, a + h * v, workdir, f"{direction_id}_h{h:.0e}_plus")
            jm, sm = objective_real_value(state, a - h * v, workdir, f"{direction_id}_h{h:.0e}_minus")
            if sp == "OK" and sm == "OK":
                fd_ref = (jp - jm) / (2.0 * h)
            else:
                fd_ref = float("nan")
            abs_err_fd = abs(grad_fd_dot - fd_ref) if np.isfinite(fd_ref) else float("nan")
            abs_err_cs = abs(grad_cs_dot - fd_ref) if np.isfinite(fd_ref) else float("nan")
            denom = max(abs(fd_ref), EPS) if np.isfinite(fd_ref) else float("nan")
            rows.append(
                {
                    "direction_id": direction_id,
                    "direction_type": direction_type,
                    "h": h,
                    "fd_reference": fd_ref,
                    "grad_fd_dot_v": grad_fd_dot,
                    "grad_cs_dot_v": grad_cs_dot,
                    "abs_err_grad_fd": abs_err_fd,
                    "abs_err_grad_cs": abs_err_cs,
                    "rel_err_grad_fd": abs_err_fd / denom if np.isfinite(denom) else float("nan"),
                    "rel_err_grad_cs": abs_err_cs / denom if np.isfinite(denom) else float("nan"),
                }
            )
    fields = [
        "direction_id",
        "direction_type",
        "h",
        "fd_reference",
        "grad_fd_dot_v",
        "grad_cs_dot_v",
        "abs_err_grad_fd",
        "abs_err_grad_cs",
        "rel_err_grad_fd",
        "rel_err_grad_cs",
    ]
    write_csv(out_dir / "directional_derivative_check.csv", fields, rows)
    fd_better = 0
    cs_better = 0
    for row in rows:
        efd = row["abs_err_grad_fd"]
        ecs = row["abs_err_grad_cs"]
        if np.isfinite(efd) and np.isfinite(ecs):
            if efd < ecs:
                fd_better += 1
            elif ecs < efd:
                cs_better += 1
    return {"directional_fd_better_count": fd_better, "directional_cs_better_count": cs_better}


def score_candidates_for_mode(state, mode, out_dir):
    candidates = list(build_interval_candidates(state["upper_centers"], side="UPPER"))
    candidates.extend(build_interval_candidates(state["lower_centers"], side="LOWER"))
    grad_score_mode = str(SETTINGS["optimization"]["adaptive"].get("grad_score_mode", "grad_norm")).strip().lower()
    gn_context = None

    workdir = out_dir / f"candidate_scoring_{mode}"
    with temporary_derivative_mode(mode):
        if grad_score_mode == "gn_schur":
            gn_context = prepare_gn_schur_level_context(
                x=state["x"],
                yu_init=state["yu_init"],
                yl_init=state["yl_init"],
                cp_target=state["cp_target"],
                active_upper=state["upper_centers"],
                active_lower=state["lower_centers"],
                active_a=state["a_vec"],
                current_best_error=state["current_best_error"],
                workdir=workdir,
            )

        rows = []
        for idx, cand in enumerate(candidates):
            info = score_candidate(
                x=state["x"],
                yu_init=state["yu_init"],
                yl_init=state["yl_init"],
                cp_target=state["cp_target"],
                active_upper=state["upper_centers"],
                active_lower=state["lower_centers"],
                active_a=state["a_vec"],
                candidate=cand,
                current_best_error=state["current_best_error"],
                workdir=workdir,
                indicator="GRAD",
                pred_context=None,
                gn_context=gn_context,
            )
            rows.append(
                {
                    "candidate_id": f"{cand['side']}_{int(cand['interval_id']):03d}_{float(cand['x']):.12f}",
                    "candidate_order": idx,
                    "side": info["side"],
                    "x": float(info["x"]),
                    "interval_id": int(cand["interval_id"]),
                    "interval_label": cand.get("interval_label", ""),
                    "x_left": float(cand["x_left"]),
                    "x_right": float(cand["x_right"]),
                    "local_fraction": float(cand.get("local_fraction", 0.5)),
                    "score": float(info["score"]),
                    "component": safe_float(info.get("component")),
                    "raw_grad": safe_float(info.get("raw_grad")),
                    "mode": info.get("mode", ""),
                    "score_mode": info.get("score_mode", ""),
                    "n_scoring_aero_calls": int(info.get("n_scoring_aero_calls", 0)),
                }
            )
    ranked = sorted(rows, key=lambda r: (-safe_float(r["score"]), safe_float(r["x"]), r["side"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    ranks = {row["candidate_id"]: row["rank"] for row in ranked}
    for row in rows:
        row["rank"] = ranks[row["candidate_id"]]
    return rows


def write_candidate_outputs(out_dir, rows_fd, rows_cs):
    fields = [
        "candidate_id",
        "candidate_order",
        "side",
        "x",
        "interval_id",
        "interval_label",
        "x_left",
        "x_right",
        "local_fraction",
        "score",
        "rank",
        "component",
        "raw_grad",
        "mode",
        "score_mode",
        "n_scoring_aero_calls",
    ]
    write_csv(out_dir / "candidate_score_fd.csv", fields, rows_fd)
    write_csv(out_dir / "candidate_score_cs.csv", fields, rows_cs)

    by_fd = {r["candidate_id"]: r for r in rows_fd}
    by_cs = {r["candidate_id"]: r for r in rows_cs}
    common_ids = sorted(set(by_fd) & set(by_cs), key=lambda cid: by_fd[cid]["candidate_order"])

    n_add = int(SETTINGS["optimization"]["adaptive"].get("n_add_per_level", 1) or 1)
    k = max(1, n_add)
    reduced_fd = reduce_to_best_candidate_per_interval([dict(r, x=float(r["x"])) for r in rows_fd])
    reduced_cs = reduce_to_best_candidate_per_interval([dict(r, x=float(r["x"])) for r in rows_cs])
    reduced_fd = sorted(reduced_fd, key=lambda r: (-safe_float(r["score"]), safe_float(r["x"]), r["side"]))
    reduced_cs = sorted(reduced_cs, key=lambda r: (-safe_float(r["score"]), safe_float(r["x"]), r["side"]))
    top1_fd = reduced_fd[0]["candidate_id"] if reduced_fd else ""
    top1_cs = reduced_cs[0]["candidate_id"] if reduced_cs else ""
    topk_fd = {r["candidate_id"] for r in reduced_fd[:k]}
    topk_cs = {r["candidate_id"] for r in reduced_cs[:k]}

    compare_rows = []
    for cid in common_ids:
        fd = by_fd[cid]
        cs = by_cs[cid]
        compare_rows.append(
            {
                "candidate_id": cid,
                "side": fd["side"],
                "x": float(fd["x"]),
                "interval_id": int(fd["interval_id"]),
                "score_fd": safe_float(fd["score"]),
                "score_cs": safe_float(cs["score"]),
                "rank_fd": int(fd["rank"]),
                "rank_cs": int(cs["rank"]),
                "rank_shift": int(cs["rank"]) - int(fd["rank"]),
                "selected_fd_top1": cid == top1_fd,
                "selected_cs_top1": cid == top1_cs,
                "selected_fd_topk": cid in topk_fd,
                "selected_cs_topk": cid in topk_cs,
            }
        )
    write_csv(
        out_dir / "candidate_score_compare.csv",
        [
            "candidate_id",
            "side",
            "x",
            "interval_id",
            "score_fd",
            "score_cs",
            "rank_fd",
            "rank_cs",
            "rank_shift",
            "selected_fd_top1",
            "selected_cs_top1",
            "selected_fd_topk",
            "selected_cs_topk",
        ],
        compare_rows,
    )

    scores_fd = np.asarray([r["score_fd"] for r in compare_rows], dtype=float)
    scores_cs = np.asarray([r["score_cs"] for r in compare_rows], dtype=float)
    finite = np.isfinite(scores_fd) & np.isfinite(scores_cs)
    if np.count_nonzero(finite) >= 2:
        pearson = float(np.corrcoef(scores_fd[finite], scores_cs[finite])[0, 1])
    else:
        pearson = float("nan")

    spearman = None
    try:
        from scipy.stats import spearmanr

        if np.count_nonzero(finite) >= 2:
            spearman = float(spearmanr(scores_fd[finite], scores_cs[finite]).correlation)
    except Exception:
        spearman = None

    return {
        "top1_fd": top1_fd,
        "top1_cs": top1_cs,
        "topk_fd": ",".join(sorted(topk_fd)),
        "topk_cs": ",".join(sorted(topk_cs)),
        "topk_overlap_count": int(len(topk_fd & topk_cs)),
        "topk_overlap_fraction": float(len(topk_fd & topk_cs) / max(len(topk_fd), 1)),
        "score_correlation_pearson": pearson,
        "score_correlation_spearman": spearman,
    }


def complex_safety_check(state, out_dir):
    h = 1.0e-30
    a = np.asarray(state["a_vec"], dtype=float)
    ntest = min(5, len(a))
    results = []
    with temporary_derivative_mode("cs"):
        objective = make_diag_objective(state, out_dir / "complex_safety_runs", "complex_safety")
        for i in range(ntest):
            a_complex = a.astype(complex)
            a_complex[i] += 1j * h
            try:
                j_complex = objective.evaluate_cs(a_complex)
                is_complex = np.iscomplexobj(j_complex)
                imag = float(np.imag(j_complex))
                status = "OK"
                typ = type(j_complex).__name__
            except Exception as exc:
                is_complex = False
                imag = float("nan")
                status = f"ERROR: {exc}"
                typ = "error"
            results.append(
                {
                    "index": i,
                    "type": typ,
                    "is_complex": bool(is_complex),
                    "imag": imag,
                    "imag_over_h": imag / h if np.isfinite(imag) else float("nan"),
                    "status": status,
                }
            )
    all_zero = all((not np.isfinite(r["imag"])) or r["imag"] == 0.0 for r in results)
    any_not_complex = any(not r["is_complex"] for r in results)
    possible_loss = bool(any_not_complex or all_zero)
    write_csv(out_dir / "complex_safety_check.csv", ["index", "type", "is_complex", "imag", "imag_over_h", "status"], results)
    return {
        "complex_safety_rows": results,
        "possible_imaginary_part_loss": possible_loss,
        "complex_safety_message": (
            "Possible imaginary part loss: complex-step path may not be complex-safe."
            if possible_loss
            else "Complex-step objective returned nonzero imaginary response in at least one tested direction."
        ),
    }


def complex_safety_grep(project_root, out_dir):
    patterns = [
        "dtype=float",
        "float(",
        "np.real",
        ".real",
        "abs(",
        "np.abs",
        "np.linalg.norm",
        "math.",
        "min(",
        "max(",
        "np.clip",
    ]
    py_files = [
        p
        for p in sorted(project_root.glob("*.py"))
        if p.name != Path(__file__).name and not p.name.startswith("diagnostic_derivatives")
    ]
    lines = []
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            hits = [pat for pat in patterns if pat in line]
            if hits:
                rel = path.relative_to(project_root)
                lines.append(f"{rel}:{lineno}: {','.join(hits)}: {line.strip()}")
    out_path = out_dir / "complex_safety_grep.txt"
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def complex_path_suspects(project_root, out_dir):
    target_files = [
        "objective.py",
        "geometry.py",
        "hicks_henne.py",
        "aero_wrapper.py",
        "cmplxfoil_wrapper.py",
        "cp_utils.py",
        "optimization.py",
    ]
    patterns = [
        "dtype=float",
        "np.asarray",
        "float(",
        "np.real",
        ".real",
        "abs(",
        "np.abs",
        "np.linalg.norm",
        "math.",
        "min(",
        "max(",
        "np.clip",
    ]
    grouped = []
    total = 0
    for name in target_files:
        path = project_root / name
        if not path.exists():
            continue
        entries = []
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            hits = []
            for pat in patterns:
                if pat == "np.asarray":
                    if "np.asarray" in stripped and "dtype=float" in stripped:
                        hits.append("np.asarray(..., dtype=float)")
                elif pat in stripped:
                    hits.append(pat)
            if hits:
                entries.append(f"  {lineno}: {','.join(hits)}: {stripped}")
        if entries:
            grouped.append(name)
            grouped.extend(entries)
            grouped.append("")
            total += len(entries)

    out_path = out_dir / "complex_path_suspects.txt"
    out_path.write_text("\n".join(grouped) + ("\n" if grouped else ""), encoding="utf-8")
    return total


def root_cause_summary(objective_path_stats, geometry_stats, cp_stats, total_cp_stats):
    objective_match = "YES" if objective_path_stats.get("objective_real_vs_cs_match") else "NO"
    geometry_match = geometry_stats.get("geometry_cs_match", "UNAVAILABLE")
    cp_response = cp_stats.get("cp_complex_response", "UNAVAILABLE")
    total_safe = "YES" if total_cp_stats.get("total_cp_error_complex_safe") else "NO"

    if objective_match == "NO":
        suspect = "evaluate_cs computes a different objective than real path"
    elif geometry_match == "NO":
        suspect = "geometry/Hicks-Henne complex path"
    elif cp_response in {"ZERO", "UNAVAILABLE", "ERROR"}:
        suspect = "CMPLXFOIL/aero wrapper complex response"
    elif total_safe == "NO":
        suspect = "total_cp_error/objective formula not complex-step-safe"
    else:
        suspect = "not identified; inspect per-level candidate divergence"

    return {
        "objective_real_vs_cs_match": objective_match,
        "geometry_cs_match": geometry_match,
        "cp_complex_response": cp_response,
        "total_cp_error_complex_safe": total_safe,
        "main_suspect": suspect,
    }


def write_summary(
    out_dir,
    state,
    grad_stats_out,
    dir_stats,
    candidate_stats,
    complex_stats,
    grep_count,
    root_stats,
    objective_path_stats,
    geometry_stats,
    cp_stats,
    total_cp_stats,
    suspect_count,
    args,
):
    lines = []
    lines.append("DERIVATIVE DIAGNOSTIC FD VS CS")
    lines.append(f"cfg_path = {args.cfg_path}")
    lines.append(f"seed = {args.seed}")
    lines.append(f"requested_level = {args.level}")
    lines.append(f"state_mode = {state['mode']}")
    lines.append(f"state_source = {state['source']}")
    lines.append(f"ndv_total = {state['ndv_total']}")
    lines.append(f"n_upper_centers = {len(state['upper_centers'])}")
    lines.append(f"n_lower_centers = {len(state['lower_centers'])}")
    lines.append(f"original_aero_derivatives = {args.original_derivatives}")
    lines.append(f"aero_backend = {SETTINGS.get('aero', {}).get('backend')}")
    lines.append("")
    lines.append("CS ROOT-CAUSE CHECK")
    lines.append("-------------------")
    lines.append(f"objective_real_vs_cs_match = {root_stats['objective_real_vs_cs_match']}")
    lines.append(f"geometry_cs_match = {root_stats['geometry_cs_match']}")
    lines.append(f"cp_complex_response = {root_stats['cp_complex_response']}")
    lines.append(f"total_cp_error_complex_safe = {root_stats['total_cp_error_complex_safe']}")
    lines.append(f"main_suspect = {root_stats['main_suspect']}")
    if root_stats["objective_real_vs_cs_match"] == "NO":
        lines.append("CS path evaluates a different real objective than the standard path.")
    if cp_stats.get("note"):
        lines.append(cp_stats["note"])
    lines.append(
        "objective_path_compare_max_abs_diff = "
        f"{max((r['abs_diff'] for r in objective_path_stats.get('rows', []) if np.isfinite(r['abs_diff'])), default=float('nan'))}"
    )
    lines.append(
        "geometry_max_rel_err = "
        f"{max((max(r['rel_err_dyu'], r['rel_err_dyl']) for r in geometry_stats.get('rows', [])), default=float('nan'))}"
    )
    lines.append(
        "total_cp_error_max_rel_diff = "
        f"{max((r['rel_diff'] for r in total_cp_stats.get('rows', []) if np.isfinite(r['rel_diff'])), default=float('nan'))}"
    )
    lines.append("")
    for key in (
        "norm_fd",
        "norm_cs",
        "norm_ratio_cs_over_fd",
        "relative_l2_error",
        "cosine_similarity",
        "max_abs_diff",
        "index_max_abs_diff",
        "n_sign_mismatch",
        "fraction_sign_mismatch",
    ):
        lines.append(f"{key} = {grad_stats_out[key]}")
    lines.append("")
    lines.append(f"directional_fd_better_count = {dir_stats['directional_fd_better_count']}")
    lines.append(f"directional_cs_better_count = {dir_stats['directional_cs_better_count']}")
    lines.append("")
    if candidate_stats:
        for key in (
            "top1_fd",
            "top1_cs",
            "topk_fd",
            "topk_cs",
            "topk_overlap_count",
            "topk_overlap_fraction",
            "score_correlation_pearson",
        ):
            lines.append(f"{key} = {candidate_stats[key]}")
        if candidate_stats.get("score_correlation_spearman") is None:
            lines.append("score_correlation_spearman = skipped")
        else:
            lines.append(f"score_correlation_spearman = {candidate_stats['score_correlation_spearman']}")
    else:
        lines.append("candidate_scoring = unavailable")
    lines.append("")
    for row in complex_stats["complex_safety_rows"]:
        lines.append(
            "complex_safety "
            f"index={row['index']} type={row['type']} is_complex={row['is_complex']} "
            f"imag={row['imag']} imag_over_h={row['imag_over_h']} status={row['status']}"
        )
    lines.append(complex_stats["complex_safety_message"])
    lines.append("")
    lines.append(f"complex_safety_grep_matches = {grep_count}")
    lines.append("complex_safety_grep_file = complex_safety_grep.txt")
    lines.append(f"complex_path_suspects_matches = {suspect_count}")
    lines.append("complex_path_suspects_file = complex_path_suspects.txt")
    text = "\n".join(lines) + "\n"
    (out_dir / "derivative_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic comparison between AERO_DERIVATIVES='fd' and 'cs'."
    )
    parser.add_argument("cfg_path", help="case.cfg path")
    parser.add_argument("--seed", type=int, default=2, help="Initial geometry seed")
    parser.add_argument("--level", type=int, default=None, help="Adaptive level/ndv_total to load from snapshot")
    parser.add_argument("--snapshot", default=None, help="Explicit .npz snapshot path")
    parser.add_argument("--out", default="derivative_diagnostic", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_cfg_overrides(args.cfg_path)
    SETTINGS["initial_shape"]["random_seed"] = int(args.seed)
    SETTINGS["initial_shape"]["seed_list"] = [int(args.seed)]
    SETTINGS.setdefault("run", {})["seed"] = int(args.seed)
    SETTINGS["xfoil"]["working_dir"] = out_dir / f"seed_{int(args.seed)}"
    validate_settings()
    args.original_derivatives = SETTINGS.get("aero", {}).get("derivatives", "fd")

    try:
        load_runtime_imports()
    except ModuleNotFoundError as exc:
        message = (
            "Cannot run derivative diagnostic because a runtime dependency is missing.\n"
            f"missing_module = {exc.name}\n"
            f"error = {exc}\n"
            "For AERO_BACKEND='cmplxfoil', the CMPLXFOIL Python environment must provide "
            "cmplxfoil and baseclasses.\n"
        )
        (out_dir / "derivative_summary.txt").write_text(message, encoding="utf-8")
        print(message)
        sys.exit(2)

    snapshot_path = find_snapshot(project_root, seed=args.seed, level=args.level, explicit=args.snapshot)
    if snapshot_path is not None:
        state = load_snapshot(snapshot_path)
    else:
        state = build_minimal_state(args.seed, out_dir)

    with pushd(out_dir):
        objective_path_stats = objective_path_compare(state, out_dir)
        clear_cmplxfoil_solver_cache()
        geometry_stats = geometry_derivative_check(state, out_dir)
        cp_stats = cp_complex_response_check(state, out_dir)
        clear_cmplxfoil_solver_cache()
        total_cp_stats = total_cp_error_complex_check(state, out_dir)
        root_stats = root_cause_summary(
            objective_path_stats=objective_path_stats,
            geometry_stats=geometry_stats,
            cp_stats=cp_stats,
            total_cp_stats=total_cp_stats,
        )

        grad_fd, _ = compute_optimizer_gradient(state, "fd", out_dir / "gradient_runs")
        clear_cmplxfoil_solver_cache()
        grad_cs, _ = compute_optimizer_gradient(state, "cs", out_dir / "gradient_runs")
        clear_cmplxfoil_solver_cache()

        write_gradient_outputs(out_dir, grad_fd, grad_cs)
        grad_stats_out = gradient_stats(grad_fd, grad_cs)

        dir_stats = directional_derivative_check(state, grad_fd, grad_cs, out_dir)
        clear_cmplxfoil_solver_cache()

        candidate_stats = {}
        try:
            rows_fd = score_candidates_for_mode(state, "fd", out_dir)
            clear_cmplxfoil_solver_cache()
            rows_cs = score_candidates_for_mode(state, "cs", out_dir)
            clear_cmplxfoil_solver_cache()
            candidate_stats = write_candidate_outputs(out_dir, rows_fd, rows_cs)
        except Exception as exc:
            candidate_stats = {}
            (out_dir / "candidate_scoring_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

        complex_stats = complex_safety_check(state, out_dir)
        grep_count = complex_safety_grep(project_root, out_dir)
        suspect_count = complex_path_suspects(project_root, out_dir)

        write_summary(
            out_dir=out_dir,
            state=state,
            grad_stats_out=grad_stats_out,
            dir_stats=dir_stats,
            candidate_stats=candidate_stats,
            complex_stats=complex_stats,
            grep_count=grep_count,
            root_stats=root_stats,
            objective_path_stats=objective_path_stats,
            geometry_stats=geometry_stats,
            cp_stats=cp_stats,
            total_cp_stats=total_cp_stats,
            suspect_count=suspect_count,
            args=args,
        )


if __name__ == "__main__":
    main()

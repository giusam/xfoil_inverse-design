import csv
from pathlib import Path

import numpy as np

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from xfoil_wrapper import run_xfoil
from cmplxfoil_wrapper import run_cmplxfoil
from aero_wrapper import run_aero
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from spring_reallocation import reallocate_airfoil_centers, split_a_by_sides
from adaptive_strategy import run_adaptive_strategy, run_adaptive_strategy_from_state
from adaptive_utils import build_side_specific_initial_centers


def run_target_aero(*args, **kwargs):
    target_backend = SETTINGS.get("aero", {}).get(
        "target_backend",
        SETTINGS.get("aero", {}).get("backend", "xfoil"),
    )
    target_backend = str(target_backend).strip().lower()

    if target_backend == "cmplxfoil":
        return run_cmplxfoil(*args, **kwargs)
    if target_backend == "xfoil":
        return run_xfoil(*args, **kwargs)

    raise ValueError(f"Unknown target backend: {target_backend}")


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_array(arr):
    return np.array2string(np.asarray(arr, dtype=float), precision=6, separator=", ")


def safe_float(value):
    if value is None:
        return float("nan")
    try:
        return float(np.real(value))
    except Exception:
        return float("nan")


def _positive_finite(value):
    value = safe_float(value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _append_curve_point(curve, x, j, ndv=None, stage="adapt", label=""):
    j = _positive_finite(j)
    if j is None:
        return
    x = float(x)
    if curve and abs(float(curve[-1]["x"]) - x) < 1.0e-12 and abs(float(curve[-1]["J"]) - j) < 1.0e-15:
        return
    curve.append(
        {
            "x": x,
            "J": float(j),
            "ndv": int(ndv) if ndv is not None and np.isfinite(safe_float(ndv)) else "",
            "stage": str(stage),
            "label": str(label),
        }
    )


def build_grad_history_curve(adapt_out, initial_error=None, x_offset=0.0, include_initial=True):
    """Build a numeric Cp-error history using cumulative gradient evaluations."""
    curve = []
    events = []
    x_offset = float(x_offset)
    initial_ndv = None
    history = adapt_out.get("history") if isinstance(adapt_out, dict) else None
    if history:
        try:
            initial_ndv = int(history[0][0])
        except Exception:
            initial_ndv = None

    if include_initial and initial_error is not None:
        _append_curve_point(curve, x_offset, initial_error, ndv=initial_ndv, stage="initial", label="initial")

    history_full_opt = adapt_out.get("history_full_opt", []) if isinstance(adapt_out, dict) else []
    offset_start = 0.0
    if history_full_opt:
        for level in history_full_opt:
            ndv = level.get("ndv_total")
            for item in level.get("gradient_eval_history", []):
                if item.get("status") != "OK":
                    continue
                jval = _positive_finite(item.get("objective_base_at_grad_point"))
                if jval is None:
                    continue
                local_x = safe_float(item.get("cumulative_gradient_evals"))
                if not np.isfinite(local_x):
                    continue
                _append_curve_point(
                    curve,
                    x_offset + offset_start + local_x,
                    jval,
                    ndv=ndv,
                    stage="adapt",
                    label=f"ndv={ndv}" if ndv not in (None, "") else "adapt",
                )

            level_end = safe_float(level.get("gradient_eval_offset_end", offset_start))
            if not np.isfinite(level_end):
                level_end = offset_start
            x_event = x_offset + level_end
            events.append(
                {
                    "x": float(x_event),
                    "ndv": int(ndv) if ndv is not None and np.isfinite(safe_float(ndv)) else "",
                    "stage": "level",
                    "label": f"ndv={int(ndv)}" if ndv is not None and np.isfinite(safe_float(ndv)) else "level",
                }
            )
            offset_start = level_end
        return curve, events

    # Fallback: older compact history plus cumulative gradient counters.
    if history:
        print("[warning] detailed gradient history missing; using compact adaptive history fallback.")
        xs = adapt_out.get("history_gradient_evals", [])
        for idx, item in enumerate(history):
            ndv, err = item[0], item[1]
            xval = safe_float(xs[idx]) if idx < len(xs) else float(idx + 1)
            if not np.isfinite(xval):
                xval = float(idx + 1)
            _append_curve_point(curve, x_offset + xval, err, ndv=ndv, stage="adapt", label=f"ndv={ndv}")
            events.append({"x": float(x_offset + xval), "ndv": int(ndv), "stage": "level", "label": f"ndv={int(ndv)}"})
    return curve, events


def build_static_history_curve(static_out, initial_error=None):
    """Build a numeric Cp-error history for the static optimization."""
    curve = []
    events = []

    if not isinstance(static_out, dict):
        return curve, events

    ndv = static_out.get("ndv_total", static_out.get("ndv", ""))

    if initial_error is not None:
        _append_curve_point(
            curve,
            0.0,
            initial_error,
            ndv=ndv,
            stage="initial",
            label="initial",
        )

    gradient_history = static_out.get("gradient_eval_history", [])
    if not gradient_history:
        print("[warning] STATIC gradient history missing; using final point fallback.")
        final_j = static_out.get("err_opt")
        if _positive_finite(final_j) is not None:
            x_final = safe_float(static_out.get("n_gradient_evals", 1))
            if not np.isfinite(x_final) or x_final <= 0:
                x_final = 1.0
            _append_curve_point(
                curve,
                x_final,
                final_j,
                ndv=ndv,
                stage="static",
                label="static final",
            )
        return curve, events

    for item in gradient_history:
        if item.get("status") != "OK":
            continue

        jval = item.get(
            "objective_base_at_grad_point",
            item.get("objective_base", item.get("objective")),
        )
        xval = safe_float(item.get("cumulative_gradient_evals"))
        if not np.isfinite(xval):
            xval = float(len(curve))

        _append_curve_point(
            curve,
            xval,
            jval,
            ndv=ndv,
            stage="static",
            label="static",
        )

    return curve, events


def _curve_last_x(curve, fallback=0.0):
    if not curve:
        return float(fallback)
    return float(curve[-1].get("x", fallback))


def build_grad_spring_final_history_curve(pipeline_out, initial_error=None):
    adapt_out = pipeline_out.get("adapt_out", {}) if isinstance(pipeline_out, dict) else {}
    curve, events = build_grad_history_curve(adapt_out, initial_error=initial_error, x_offset=0.0, include_initial=True)
    x_last = _curve_last_x(curve, fallback=safe_float(adapt_out.get("n_gradient_evals_total", 0)))
    spring_out = pipeline_out.get("spring_out") if isinstance(pipeline_out, dict) else None
    spring_grad = int(spring_out.get("n_gradient_evals", 0)) if isinstance(spring_out, dict) else 0
    step = spring_grad if spring_grad > 0 else 1
    x_spring = x_last + float(step)
    j_final = pipeline_out.get("final_out", {}).get("err_opt") if isinstance(pipeline_out.get("final_out"), dict) else None
    if j_final is None:
        j_final = pipeline_out.get("J_after_spring", pipeline_out.get("J_before_spring"))
    ndv = adapt_out.get("ndv_total", "")
    accepted = bool(pipeline_out.get("accepted", False))
    label = "spring rebase" if accepted else "spring rebase rejected"
    _append_curve_point(curve, x_spring, j_final, ndv=ndv, stage="spring", label=label)
    events.append({"x": float(x_spring), "ndv": int(ndv) if ndv != "" else "", "stage": "spring", "label": label})
    return curve, events


def build_periodic_spring_history_curve(pipeline_out, initial_error=None):
    curve = []
    events = []
    x_offset = 0.0
    blocks = pipeline_out.get("blocks", []) if isinstance(pipeline_out, dict) else []
    if not blocks:
        print("[warning] periodic block details missing; using block_history fallback.")
        for item in pipeline_out.get("block_history", []):
            level = int(item.get("target_level"))
            _append_curve_point(curve, x_offset, item.get("J_before_spring"), ndv=level, stage="adapt", label=f"ndv={level}")
            events.append({"x": float(x_offset), "ndv": level, "stage": "level", "label": f"ndv={level}"})
            x_offset += 1.0
            _append_curve_point(curve, x_offset, item.get("J_after_spring"), ndv=level, stage="spring", label=f"spring {level}")
            events.append({"x": float(x_offset), "ndv": level, "stage": "spring", "label": f"spring {level}"})
        return curve, events

    for idx, block in enumerate(blocks):
        adapt_out = block.get("adapt_out", {})
        block_initial = block.get("initial_error", initial_error if idx == 0 else None)
        block_curve, block_events = build_grad_history_curve(
            adapt_out,
            initial_error=block_initial,
            x_offset=x_offset,
            include_initial=(idx == 0),
        )
        for point in block_curve:
            _append_curve_point(
                curve,
                point.get("x"),
                point.get("J"),
                ndv=point.get("ndv"),
                stage=point.get("stage", "adapt"),
                label=point.get("label", ""),
            )
        events.extend(block_events)
        local_grad = int(adapt_out.get("n_gradient_evals_total", 0))
        x_offset = max(_curve_last_x(curve, fallback=x_offset), x_offset + float(local_grad))

        target_level = int(block.get("target_level", adapt_out.get("ndv_total", 0)))

        spring_out = block.get("spring_out")
        spring_grad = int(spring_out.get("n_gradient_evals", 0)) if isinstance(spring_out, dict) else 0

        # Advance the x-axis by the spring reoptimization gradient evaluations.
        # If unavailable, add 1 synthetic gradient evaluation so the spring event
        # remains visible and the x-axis stays numeric.
        x_offset += float(spring_grad if spring_grad > 0 else 1)

        accepted = bool(block.get("spring_accepted", False))
        is_intermediate = bool(block.get("intermediate", False))

        # Plot the state that is actually carried forward by the pipeline.
        # If spring is accepted: J_state_after should be spring_out["err_opt"].
        # If spring is rejected: J_state_after should be the rebase/adapt state,
        # not the raw worse spring result.
        spring_j = block.get("J_state_after")

        if _positive_finite(spring_j) is None:
            if accepted:
                spring_j = block.get("J_after_spring")
            else:
                spring_j = block.get("J_before_spring")

        if _positive_finite(spring_j) is None:
            spring_j = block.get("J_after_spring")

        if is_intermediate:
            spring_label = (
                f"spring {target_level} accepted"
                if accepted
                else f"spring {target_level} rejected"
            )
        else:
            spring_label = (
                f"spring final {target_level} accepted"
                if accepted
                else f"spring final {target_level} rejected"
            )

        _append_curve_point(
            curve,
            x_offset,
            spring_j,
            ndv=target_level,
            stage="spring",
            label=spring_label,
        )

        events.append(
            {
                "x": float(x_offset),
                "ndv": target_level,
                "stage": "spring",
                "label": spring_label,
            }
        )
    return curve, events


def polar_metric(out, name):
    if out is None:
        return float("nan")
    opt_res = out.get("opt_res")
    if opt_res is None:
        return float("nan")
    polar = opt_res.get("polar")
    if polar is None:
        return float("nan")
    return safe_float(polar.get(name))


def write_polar_from_res(res, dst_path):
    if res is None or res.get("polar") is None:
        return

    polar = res["polar"]
    alpha = safe_float(polar.get("alpha"))
    cl = safe_float(polar.get("CL"))
    cd = safe_float(polar.get("CD"))
    cm = safe_float(polar.get("CM"))

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("alpha CL CD CM\n")
        f.write(f"{alpha:.10e} {cl:.10e} {cd:.10e} {cm:.10e}\n")


def write_cp_from_res(res, dst_path):
    if res is None or res.get("cp_data") is None:
        return

    cp_x = np.asarray(res["cp_data"]["x"], dtype=float)
    cp_v = np.real(np.asarray(res["cp_data"]["cp"]))
    data = np.column_stack([cp_x, cp_v])
    np.savetxt(dst_path, data, header="x cp", comments="")


def save_out_bundle(bundle_dir, x, out, name):
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if out is None:
        return

    if "yu_opt" in out and "yl_opt" in out:
        write_dat(
            bundle_dir / "optimized_airfoil.dat",
            x,
            out["yu_opt"],
            out["yl_opt"],
            name=name,
        )

    opt_res = out.get("opt_res")
    if opt_res is not None:
        write_polar_from_res(opt_res, bundle_dir / "polar.txt")
        write_cp_from_res(opt_res, bundle_dir / "cp.txt")

    with open(bundle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"name = {name}\n")
        f.write(f"err_opt = {safe_float(out.get('err_opt')):.10e}\n")
        f.write(f"CL = {polar_metric(out, 'CL'):.10e}\n")
        f.write(f"CD = {polar_metric(out, 'CD'):.10e}\n")
        f.write(f"CM = {polar_metric(out, 'CM'):.10e}\n")
        if "upper_centers" in out:
            f.write(f"upper_centers = {format_array(out['upper_centers'])}\n")
        if "lower_centers" in out:
            f.write(f"lower_centers = {format_array(out['lower_centers'])}\n")
        if "a_opt" in out:
            f.write(f"a_opt = {format_array(out['a_opt'])}\n")


def evaluate_geometry_state(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a_vec,
    label,
    workdir,
    include_aero_calls=False,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    upper_centers = np.asarray(upper_centers, dtype=float)
    lower_centers = np.asarray(lower_centers, dtype=float)
    a_vec = np.asarray(a_vec, dtype=float)

    a_upper, a_lower = split_a_by_sides(
        a_vec,
        len(upper_centers),
        len(lower_centers),
    )

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

    airfoil_dat = workdir / f"{label}_airfoil.dat"
    write_dat(airfoil_dat, x, yu, yl, name=label.upper())

    res = run_aero(
        airfoil_dat=airfoil_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=workdir / f"{label}_run",
    )
    if not res["success"]:
        out = {
            "success": False,
            "res": res,
            "yu": yu,
            "yl": yl,
            "cp_opt": None,
            "err": float("nan"),
        }
        if include_aero_calls:
            out["aero_calls"] = 1
        return out

    cp_candidate = split_upper_lower_cp_from_x(
        res["cp_data"]["x"],
        res["cp_data"]["cp"],
    )
    err = total_cp_error(cp_target, cp_candidate)
    out = {
        "success": True,
        "res": res,
        "yu": yu,
        "yl": yl,
        "cp_opt": cp_candidate,
        "err": float(err),
    }
    if include_aero_calls:
        out["aero_calls"] = 1
    return out


def write_centers_before_after_csv(path, diagnostics):
    rows = []
    for side_key, side_name, imp_key in (
        ("upper", "UPPER", "importance_upper"),
        ("lower", "LOWER", "importance_lower"),
    ):
        spring_diag = diagnostics[f"{side_key}_spring"]
        importance = np.asarray(diagnostics[imp_key], dtype=float)
        old_centers = np.asarray(spring_diag["old_centers"], dtype=float)
        new_centers = np.asarray(spring_diag["new_centers"], dtype=float)
        for idx, (old_c, new_c, imp) in enumerate(zip(old_centers, new_centers, importance)):
            rows.append(
                {
                    "side": side_name,
                    "index": idx,
                    "old_center": float(old_c),
                    "new_center": float(new_c),
                    "dx": float(new_c - old_c),
                    "importance": float(imp),
                }
            )
    write_csv(path, ["side", "index", "old_center", "new_center", "dx", "importance"], rows)


def write_spring_diagnostics_csv(path, diagnostics):
    rows = []
    for side_key, side_name in (("upper", "UPPER"), ("lower", "LOWER")):
        spring_diag = diagnostics[f"{side_key}_spring"]
        proj_diag = diagnostics[f"{side_key}_projection"]
        rows.append(
            {
                "side": side_name,
                "max_abs_dx": float(spring_diag["max_abs_dx"]),
                "min_spacing_before": float(spring_diag["min_spacing_before"]),
                "min_spacing_after": float(spring_diag["min_spacing_after"]),
                "projection_error_l2": float(proj_diag["projection_error_l2"]),
                "projection_error_linf": float(proj_diag["projection_error_linf"]),
                "relative_projection_error_l2": float(proj_diag["relative_projection_error_l2"]),
                "condition_number": float(proj_diag["condition_number"]),
            }
        )
    write_csv(
        path,
        [
            "side",
            "max_abs_dx",
            "min_spacing_before",
            "min_spacing_after",
            "projection_error_l2",
            "projection_error_linf",
            "relative_projection_error_l2",
            "condition_number",
        ],
        rows,
    )


def write_block_history_csv(path, block_history):
    rows = []
    for item in block_history:
        rows.append(
            {
                "target_level": int(item["target_level"]),
                "J_before_spring": safe_float(item.get("J_before_spring")),
                "J_rebase_start": safe_float(item.get("J_rebase_start")),
                "J_after_spring": safe_float(item.get("J_after_spring")),
                "spring_accepted": bool(item.get("spring_accepted", False)),
                "intermediate": bool(item.get("intermediate", False)),
                "state_used_after": str(item.get("state_used_after", "")),
                "old_upper_centers": format_array(item.get("old_upper_centers", [])),
                "new_upper_centers": format_array(item.get("new_upper_centers", [])),
                "old_lower_centers": format_array(item.get("old_lower_centers", [])),
                "new_lower_centers": format_array(item.get("new_lower_centers", [])),
            }
        )
    write_csv(
        path,
        [
            "target_level",
            "J_before_spring",
            "J_rebase_start",
            "J_after_spring",
            "spring_accepted",
            "intermediate",
            "state_used_after",
            "old_upper_centers",
            "new_upper_centers",
            "old_lower_centers",
            "new_lower_centers",
        ],
        rows,
    )


def _make_rebase_start_out(adapt_out, new_upper_centers, new_lower_centers, start_eval):
    a0 = np.zeros(len(new_upper_centers) + len(new_lower_centers), dtype=float)
    opt_res = start_eval.get("res") if start_eval is not None and start_eval.get("success", False) else None
    if opt_res is None:
        opt_res = adapt_out.get("opt_res")
    return {
        "err_opt": float(adapt_out["err_opt"]),
        "ndv_total": int(len(new_upper_centers) + len(new_lower_centers)),
        "upper_centers": np.asarray(new_upper_centers, dtype=float),
        "lower_centers": np.asarray(new_lower_centers, dtype=float),
        "a_opt": a0,
        "yu_opt": np.asarray(adapt_out["yu_opt"], dtype=float),
        "yl_opt": np.asarray(adapt_out["yl_opt"], dtype=float),
        "cp_opt": adapt_out.get("cp_opt"),
        "opt_res": opt_res,
        "is_rebase_start": True,
    }


def _spring_reopt_succeeded(spring_out, reference_err):
    if spring_out is None:
        return False
    err = safe_float(spring_out.get("err_opt"))
    return np.isfinite(err) and err < float(reference_err)


def _reallocate_with_settings(x, adapt_out):
    spring_cfg = SETTINGS["spring_reallocation"]
    return reallocate_airfoil_centers(
        x=x,
        upper_centers=adapt_out["upper_centers"],
        lower_centers=adapt_out["lower_centers"],
        a_opt=adapt_out["a_opt"],
        weight_mode=spring_cfg["weight_mode"],
        A=spring_cfg["A"],
        omega=spring_cfg["omega"],
        max_dx=spring_cfg["max_dx"],
        min_spacing=spring_cfg["min_spacing"],
        ridge=spring_cfg["ridge"],
        fix_ends=spring_cfg["fix_ends"],
        transfer_mode=spring_cfg["transfer_mode"],
        transfer_power=spring_cfg["transfer_power"],
        coeff_bounds=SETTINGS["optimization"]["bounds"],
    )


def run_grad_spring_final_pipeline(
    x,
    yu_init,
    yl_init,
    cp_target,
    initial_error,
    workdir,
    adapt_out=None,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if adapt_out is None:
        adapt_out = run_adaptive_strategy(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            initial_error=initial_error,
            workdir=workdir / "adapt_grad",
            indicator="GRAD",
        )
    save_out_bundle(workdir / "adapt_grad", x=x, out=adapt_out, name="ADAPT_GRAD_FOR_SPRING")

    realloc_out = _reallocate_with_settings(x, adapt_out)
    diagnostics = realloc_out["diagnostics"]
    new_upper = np.asarray(realloc_out["new_upper_centers"], dtype=float)
    new_lower = np.asarray(realloc_out["new_lower_centers"], dtype=float)
    write_centers_before_after_csv(workdir / "centers_before_after.csv", diagnostics)
    write_spring_diagnostics_csv(workdir / "spring_diagnostics.csv", diagnostics)

    a0_rebase = np.zeros(len(new_upper) + len(new_lower), dtype=float)
    start_eval = evaluate_geometry_state(
        x=x,
        yu_init=adapt_out["yu_opt"],
        yl_init=adapt_out["yl_opt"],
        cp_target=cp_target,
        upper_centers=new_upper,
        lower_centers=new_lower,
        a_vec=a0_rebase,
        label="spring_rebase_start",
        workdir=workdir / "spring_rebase_start",
        include_aero_calls=True,
    )
    rel = abs(safe_float(start_eval.get("err")) - float(adapt_out["err_opt"])) / max(abs(float(adapt_out["err_opt"])), 1.0e-16)
    if rel > 1.0e-3:
        print(f"[warning] final spring rebase mismatch relative_diff={rel:.6e}")

    spring_out = None
    try:
        spring_out = optimize_for_centers(
            x=x,
            yu_init=adapt_out["yu_opt"],
            yl_init=adapt_out["yl_opt"],
            cp_target=cp_target,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a0=a0_rebase,
            label="grad_spring_final_rebase",
            current_best_error=float(adapt_out["err_opt"]),
            workdir=workdir / "spring_rebase" / "reopt",
        )
    except Exception as exc:
        print("[warning] final spring reoptimization failed; keeping ADAPT_GRAD result.")
        print(str(exc))

    if spring_out is not None:
        save_out_bundle(workdir / "spring_raw", x=x, out=spring_out, name="GRAD_SPRING_FINAL_RAW")

    accepted = _spring_reopt_succeeded(spring_out, adapt_out["err_opt"])
    final_out = spring_out if accepted else adapt_out
    save_out_bundle(workdir / "final", x=x, out=final_out, name="GRAD_SPRING_FINAL")

    spring_opt_calls = int(spring_out.get("n_optimization_aero_calls_total", 0)) if spring_out is not None else 0
    spring_fun = int(spring_out.get("n_function_evals", 0)) if spring_out is not None else 0
    spring_grad = int(spring_out.get("n_gradient_evals", 0)) if spring_out is not None else 0
    start_calls = int(start_eval.get("aero_calls", 0))

    return {
        "method_name": "GRAD_SPRING_FINAL",
        "final_out": final_out,
        "adapt_out": adapt_out,
        "spring_out": spring_out,
        "accepted": bool(accepted),
        "J_before_spring": safe_float(adapt_out.get("err_opt")),
        "J_rebase_start": safe_float(start_eval.get("err")),
        "J_after_spring": safe_float(spring_out.get("err_opt") if spring_out is not None else None),
        "diagnostics": diagnostics,
        "spring_history": [
            ("ADAPT_GRAD", safe_float(adapt_out.get("err_opt"))),
            ("SPRING", safe_float(final_out.get("err_opt"))),
        ],
        "total_scoring_aero_calls": int(adapt_out.get("n_scoring_aero_calls_total", 0)),
        "total_optimization_aero_calls": int(adapt_out.get("n_optimization_aero_calls_total", 0) + spring_opt_calls),
        "total_function_evals": int(adapt_out.get("n_function_evals_total", 0) + spring_fun),
        "total_gradient_evals": int(adapt_out.get("n_gradient_evals_total", 0) + spring_grad),
        "total_aero_calls": int(adapt_out.get("n_scoring_aero_calls_total", 0) + adapt_out.get("n_optimization_aero_calls_total", 0) + start_calls + spring_opt_calls),
    }


def run_grad_spring_periodic_pipeline(
    x,
    yu_init,
    yl_init,
    cp_target,
    initial_error,
    workdir,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = SETTINGS["adaptive_spring"]
    levels = sorted(int(v) for v in cfg.get("periodic_levels", [12, 16, 20]))
    if not levels:
        raise ValueError("ADAPTIVE_SPRING_PERIODIC_LEVELS must contain at least one level.")
    if str(cfg.get("accept_mode_intermediate", "rebase_keep_new_centers")).strip().lower() != "rebase_keep_new_centers":
        raise ValueError("ADAPTIVE_SPRING_ACCEPT_MODE_INTERMEDIATE supports only 'rebase_keep_new_centers'.")
    if str(cfg.get("accept_mode_final", "accept_if_improved")).strip().lower() != "accept_if_improved":
        raise ValueError("ADAPTIVE_SPRING_ACCEPT_MODE_FINAL supports only 'accept_if_improved'.")

    n0 = int(SETTINGS["optimization"]["adaptive"]["n0"])
    current_upper, current_lower = build_side_specific_initial_centers(n0)
    current_a0 = np.zeros(len(current_upper) + len(current_lower), dtype=float)
    current_base_yu = np.asarray(yu_init, dtype=float)
    current_base_yl = np.asarray(yl_init, dtype=float)
    current_error = float(initial_error)

    total_scoring = 0
    total_optimization = 0
    total_function = 0
    total_gradient = 0
    total_rebase_start = 0
    block_history = []
    blocks = []
    final_out = None
    final_method = ""

    for target_level in levels:
        block_initial_error = float(current_error)
        current_ndv = len(current_upper) + len(current_lower)
        if target_level < current_ndv:
            raise ValueError(f"Periodic target level {target_level} is smaller than current ndv {current_ndv}.")
        print("===== PERIODIC ADAPT_GRAD BLOCK =====")
        print(f"current ndv = {current_ndv}")
        print(f"target ndv  = {target_level}")

        block_dir = workdir / f"block_{target_level:03d}_adapt"
        adapt_out = run_adaptive_strategy_from_state(
            x=x,
            yu_init=current_base_yu,
            yl_init=current_base_yl,
            cp_target=cp_target,
            initial_error=current_error,
            workdir=block_dir,
            indicator="GRAD",
            initial_upper_centers=current_upper,
            initial_lower_centers=current_lower,
            initial_a0=current_a0,
            n_final_override=target_level,
        )
        save_out_bundle(block_dir, x=x, out=adapt_out, name=f"PERIODIC_ADAPT_{target_level}")
        total_scoring += int(adapt_out.get("n_scoring_aero_calls_total", 0))
        total_optimization += int(adapt_out.get("n_optimization_aero_calls_total", 0))
        total_function += int(adapt_out.get("n_function_evals_total", 0))
        total_gradient += int(adapt_out.get("n_gradient_evals_total", 0))

        realloc_out = _reallocate_with_settings(x, adapt_out)
        diagnostics = realloc_out["diagnostics"]
        new_upper = np.asarray(realloc_out["new_upper_centers"], dtype=float)
        new_lower = np.asarray(realloc_out["new_lower_centers"], dtype=float)
        write_centers_before_after_csv(workdir / f"centers_before_after_level_{target_level:03d}.csv", diagnostics)
        write_spring_diagnostics_csv(workdir / f"spring_diagnostics_level_{target_level:03d}.csv", diagnostics)

        a0_rebase = np.zeros(len(new_upper) + len(new_lower), dtype=float)
        spring_dir = workdir / f"block_{target_level:03d}_spring"
        start_eval = evaluate_geometry_state(
            x=x,
            yu_init=adapt_out["yu_opt"],
            yl_init=adapt_out["yl_opt"],
            cp_target=cp_target,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a_vec=a0_rebase,
            label=f"periodic_rebase_start_level_{target_level}",
            workdir=spring_dir / "rebase_start",
            include_aero_calls=True,
        )
        total_rebase_start += int(start_eval.get("aero_calls", 0))
        rel = abs(safe_float(start_eval.get("err")) - float(adapt_out["err_opt"])) / max(abs(float(adapt_out["err_opt"])), 1.0e-16)
        if rel > 1.0e-3:
            print(f"[warning] periodic rebase mismatch level={target_level} relative_diff={rel:.6e}")

        spring_out = None
        try:
            spring_out = optimize_for_centers(
                x=x,
                yu_init=adapt_out["yu_opt"],
                yl_init=adapt_out["yl_opt"],
                cp_target=cp_target,
                upper_centers=new_upper,
                lower_centers=new_lower,
                a0=a0_rebase,
                label=f"periodic_spring_rebase_level_{target_level}",
                current_best_error=float(adapt_out["err_opt"]),
                workdir=spring_dir / "reopt",
            )
        except Exception as exc:
            print(f"[warning] periodic spring reoptimization failed at level {target_level}.")
            print(str(exc))
        if spring_out is not None:
            save_out_bundle(spring_dir, x=x, out=spring_out, name=f"PERIODIC_SPRING_{target_level}")
            total_optimization += int(spring_out.get("n_optimization_aero_calls_total", 0))
            total_function += int(spring_out.get("n_function_evals", 0))
            total_gradient += int(spring_out.get("n_gradient_evals", 0))

        is_final = target_level == levels[-1]
        improved = _spring_reopt_succeeded(spring_out, adapt_out["err_opt"])
        rebase_start_out = _make_rebase_start_out(adapt_out, new_upper, new_lower, start_eval)
        if is_final:
            if improved:
                final_out = spring_out
                spring_accepted = True
                state_used_after = "spring_reopt"
                final_method = f"periodic_spring_reopt_level_{target_level}"
            else:
                final_out = adapt_out
                spring_accepted = False
                state_used_after = "adapt_before_spring"
                final_method = f"periodic_adapt_level_{target_level}"
            save_out_bundle(workdir / "final", x=x, out=final_out, name="GRAD_SPRING_PERIODIC_FINAL")
        else:
            if improved:
                state_for_next = spring_out
                spring_accepted = True
                state_used_after = "spring_reopt"
                current_error = float(spring_out["err_opt"])
            else:
                state_for_next = rebase_start_out
                spring_accepted = False
                state_used_after = "rebase_start"
                current_error = float(adapt_out["err_opt"])
            current_base_yu = np.asarray(state_for_next["yu_opt"], dtype=float)
            current_base_yl = np.asarray(state_for_next["yl_opt"], dtype=float)
            current_upper = list(np.asarray(state_for_next["upper_centers"], dtype=float))
            current_lower = list(np.asarray(state_for_next["lower_centers"], dtype=float))
            current_a0 = np.zeros(len(current_upper) + len(current_lower), dtype=float)
            save_out_bundle(workdir / f"block_{target_level:03d}_state_for_next", x=x, out=state_for_next, name=f"PERIODIC_STATE_{target_level}")

        block_history.append(
            {
                "target_level": int(target_level),
                "J_before_spring": safe_float(adapt_out.get("err_opt")),
                "J_rebase_start": safe_float(start_eval.get("err")),
                "J_after_spring": safe_float(spring_out.get("err_opt") if spring_out is not None else None),
                "spring_accepted": bool(spring_accepted),
                "intermediate": not is_final,
                "state_used_after": state_used_after,
                "old_upper_centers": np.asarray(adapt_out["upper_centers"], dtype=float),
                "new_upper_centers": new_upper,
                "old_lower_centers": np.asarray(adapt_out["lower_centers"], dtype=float),
                "new_lower_centers": new_lower,
            }
        )
        blocks.append(
            {
                "target_level": int(target_level),
                "initial_error": float(block_initial_error),
                "adapt_out": adapt_out,
                "spring_out": spring_out,
                "J_before_spring": safe_float(adapt_out.get("err_opt")),
                "J_rebase_start": safe_float(start_eval.get("err")),
                "J_after_spring": safe_float(spring_out.get("err_opt") if spring_out is not None else None),
                "J_state_after": safe_float((spring_out if improved else rebase_start_out).get("err_opt") if not is_final else final_out.get("err_opt")),
                "spring_accepted": bool(spring_accepted),
                "intermediate": not is_final,
                "state_used_after": state_used_after,
            }
        )

    if final_out is None:
        raise RuntimeError("Periodic spring pipeline did not produce a final result.")
    write_block_history_csv(workdir / "block_history.csv", block_history)
    return {
        "method_name": "GRAD_SPRING_PERIODIC",
        "final_out": final_out,
        "block_history": block_history,
        "blocks": blocks,
        "levels": levels,
        "final_method": final_method,
        "total_scoring_aero_calls": int(total_scoring),
        "total_optimization_aero_calls": int(total_optimization),
        "total_function_evals": int(total_function),
        "total_gradient_evals": int(total_gradient),
        "total_aero_calls": int(total_scoring + total_optimization + total_rebase_start),
    }


def make_method_result_bundle(method_name, seed, x, yu_init, yl_init, cp_target, cp_init, out, workdir, extra=None):
    extra = {} if extra is None else dict(extra)
    opt_res = out.get("opt_res", {}) if out is not None else {}
    polar = opt_res.get("polar", {}) if isinstance(opt_res, dict) else {}
    initial_error = safe_float(extra.get("initial_error"))
    if not np.isfinite(initial_error):
        try:
            initial_error = safe_float(total_cp_error(cp_target, cp_init))
        except Exception:
            initial_error = float("nan")
    bundle = {
        "method_name": method_name,
        "seed": int(seed),
        "x": np.asarray(x, dtype=float),
        "yu_initial": np.asarray(yu_init, dtype=float),
        "yl_initial": np.asarray(yl_init, dtype=float),
        "yu_final": np.asarray(out.get("yu_opt", []), dtype=float),
        "yl_final": np.asarray(out.get("yl_opt", []), dtype=float),
        "cp_target": cp_target,
        "cp_initial": cp_init,
        "cp_final": out.get("cp_opt"),
        "polar_final": polar,
        "err_initial": initial_error,
        "err_final": safe_float(out.get("err_opt")),
        "upper_centers_final": np.asarray(out.get("upper_centers", []), dtype=float),
        "lower_centers_final": np.asarray(out.get("lower_centers", []), dtype=float),
        "a_opt_final": np.asarray(out.get("a_opt", []), dtype=float),
        "history": out.get("history"),
        "history_full_opt": out.get("history_full_opt", []),
        "function_eval_history": out.get("function_eval_history", []),
        "gradient_eval_history": out.get("gradient_eval_history", []),
        "total_scoring_aero_calls": int(extra.get("total_scoring_aero_calls", out.get("n_scoring_aero_calls_total", 0))),
        "total_optimization_aero_calls": int(extra.get("total_optimization_aero_calls", out.get("n_optimization_aero_calls_total", 0))),
        "total_function_evals": int(extra.get("total_function_evals", out.get("n_function_evals_total", out.get("n_function_evals", 0)))),
        "total_gradient_evals": int(extra.get("total_gradient_evals", out.get("n_gradient_evals_total", out.get("n_gradient_evals", 0)))),
        "total_aero_calls": int(extra.get("total_aero_calls", out.get("n_aero_calls_total_all_phases", out.get("n_total_evals", 0)))),
        "workdir": str(workdir),
    }
    bundle.update(extra)
    if "history_curve" not in bundle:
        if method_name == "STATIC":
            curve, events = build_static_history_curve(out, initial_error=initial_error)
        elif method_name == "GRAD_SPRING_FINAL":
            curve, events = build_grad_spring_final_history_curve(bundle, initial_error=initial_error)
        elif method_name == "GRAD_SPRING_PERIODIC":
            curve, events = build_periodic_spring_history_curve(bundle, initial_error=initial_error)
        elif method_name.startswith("ADAPT_"):
            curve, events = build_grad_history_curve(out, initial_error=initial_error, x_offset=0.0, include_initial=True)
        else:
            curve, events = [], []
        bundle["history_curve"] = curve
        bundle["history_events"] = events
    return bundle


def write_standard_method_plots(method_results, x, yu_target, yl_target, yu_init, yl_init, cp_target, cp_init, output_dir):
    from plotting import (
        plot_history_comparison,
        plot_method_error_comparison,
        plot_single_method_airfoil,
        plot_single_method_centers,
        plot_single_method_cp,
        plot_single_method_history,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, bundle in method_results.items():
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        plot_single_method_airfoil(x, yu_target, yl_target, yu_init, yl_init, bundle, method_dir / "airfoil.png")
        plot_single_method_cp(cp_target, bundle, method_dir / "cp.png")
        upper = np.asarray(bundle.get("upper_centers_final", []), dtype=float)
        lower = np.asarray(bundle.get("lower_centers_final", []), dtype=float)
        if upper.size > 0 or lower.size > 0:
            plot_single_method_centers(bundle, method_dir / "centers.png")
        if bundle.get("history_curve"):
            plot_single_method_history(bundle, method_dir / "history.png")

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_method_error_comparison(method_results, comparison_dir / "final_error_bar.png")
    plot_history_comparison(method_results, comparison_dir / "history_comparison.png")

    # Compatibility names: keep old paths, but with the corrected numeric history.
    plot_method_error_comparison(method_results, output_dir / "error_comparison.png")
    plot_history_comparison(method_results, output_dir / "history_comparison.png")


def build_method_summary_row(seed, err_init, method_results):
    row = {"seed": int(seed), "J_initial": safe_float(err_init)}
    suffixes = {
        "STATIC": "static",
        "ADAPT_GRAD": "adapt_grad",
        "GRAD_SPRING_FINAL": "grad_spring_final",
        "GRAD_SPRING_PERIODIC": "grad_spring_periodic",
    }
    for method, suffix in suffixes.items():
        bundle = method_results.get(method)
        row[f"J_{suffix}"] = safe_float(bundle.get("err_final")) if bundle else float("nan")
        for metric in ("CL", "CD", "CM"):
            row[f"{metric}_{suffix}"] = safe_float(bundle.get("polar_final", {}).get(metric)) if bundle else float("nan")
        row[f"total_aero_calls_{suffix}"] = int(bundle.get("total_aero_calls", 0)) if bundle else ""

    periodic = method_results.get("GRAD_SPRING_PERIODIC")
    levels = []
    if periodic:
        for item in periodic.get("block_history", []):
            level = int(item["target_level"])
            levels.append(level)
            row[f"J_after_add_{level}"] = safe_float(item.get("J_before_spring"))
            row[f"J_rebase_start_{level}"] = safe_float(item.get("J_rebase_start"))
            row[f"J_after_spring_{level}"] = safe_float(item.get("J_after_spring"))
            row[f"spring_accepted_{level}"] = bool(item.get("spring_accepted", False))
    for level in sorted(set([12, 16, 20] + levels)):
        row.setdefault(f"J_after_add_{level}", float("nan"))
        row.setdefault(f"J_rebase_start_{level}", float("nan"))
        row.setdefault(f"J_after_spring_{level}", float("nan"))
        row.setdefault(f"spring_accepted_{level}", "")
    return row


def method_summary_fields(rows):
    base = ["seed", "J_initial"]
    methods = []
    for suffix in ("static", "adapt_grad", "grad_spring_final", "grad_spring_periodic"):
        methods.extend([f"J_{suffix}", f"CL_{suffix}", f"CD_{suffix}", f"CM_{suffix}", f"total_aero_calls_{suffix}"])
    level_fields = sorted({k for row in rows for k in row if k.startswith(("J_after_add_", "J_rebase_start_", "J_after_spring_", "spring_accepted_"))})
    def _level_sort_key(name):
        return (int(name.rsplit("_", 1)[-1]), name)
    level_fields = sorted(level_fields, key=_level_sort_key)
    return base + methods + level_fields

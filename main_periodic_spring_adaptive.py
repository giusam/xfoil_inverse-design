import argparse
from pathlib import Path

import numpy as np

from settings import SETTINGS, apply_cfg_overrides
from geometry import (
    build_naca0012_surfaces,
    build_random_initial_geometry,
    write_dat,
)
from aero_wrapper import run_aero
from xfoil_wrapper import clean_workdir
from cmplxfoil_wrapper import clear_cmplxfoil_solver_cache
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from spring_reallocation import reallocate_airfoil_centers
from adaptive_utils import build_side_specific_initial_centers
from adaptive_strategy import run_adaptive_strategy_from_state
from experiment_utils import (
    evaluate_geometry_state,
    format_array,
    polar_metric,
    run_target_aero,
    safe_float,
    save_out_bundle,
    write_csv,
)
from reporting import print_initial_state, report_initial_aero_failure


BASE_SUMMARY_FIELDS = [
    "seed",
    "levels",
    "J_initial",
    "J_final",
    "final_method",
    "CL_final",
    "CD_final",
    "CM_final",
    "total_scoring_aero_calls",
    "total_optimization_aero_calls",
    "total_function_evals",
    "total_gradient_evals",
    "total_aero_calls",
    "upper_centers_final",
    "lower_centers_final",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADAPT_GRAD with periodic spring rebase blocks.",
    )
    parser.add_argument(
        "cfg_path",
        nargs="?",
        help="Optional CFG override file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If provided, run only this seed.",
    )
    parser.add_argument(
        "--out",
        default="periodic_spring_adaptive",
        help="Base output directory. Relative paths are resolved from the script directory.",
    )
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        default=True,
        help="Clean the base output directory before running.",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Do not clean the base output directory before running.",
    )
    return parser.parse_args()


def _summary_fields(levels):
    fields = list(BASE_SUMMARY_FIELDS)
    level_fields = []
    for level in sorted(set([12, 16, 20] + [int(v) for v in levels])):
        level_fields.extend(
            [
                f"J_after_add_{level}",
                f"J_rebase_start_{level}",
                f"J_after_spring_{level}",
                f"spring_accepted_{level}",
                f"state_used_after_{level}",
            ]
        )
    return fields[:3] + level_fields + fields[3:]


def _write_centers_csv(path, diagnostics):
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

    write_csv(
        path,
        ["side", "index", "old_center", "new_center", "dx", "importance"],
        rows,
    )


def _write_diagnostics_csv(path, diagnostics):
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


def _write_block_history_csv(path, block_history):
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


def _print_rebase_check(level, j_before, start_eval):
    j_start = safe_float(start_eval.get("err"))
    diff = j_start - float(j_before)
    rel = abs(diff) / max(abs(float(j_before)), 1.0e-16)

    print(f"===== PERIODIC SPRING REBASE CHECK level={level} =====")
    print(f"J_before_spring = {float(j_before):.6e}")
    print(f"J_rebase_start  = {j_start:.6e}")
    print(f"relative_diff   = {rel:.6e}")

    if rel > 1.0e-3:
        print("#############################################")
        print("WARNING: rebase start mismatch is larger than expected")
        print(f"level = {level}")
        print(f"relative_difference = {rel:.6e}")
        print("#############################################")


def _make_rebase_start_out(adapt_out, new_upper_centers, new_lower_centers, start_eval):
    a0 = np.zeros(len(new_upper_centers) + len(new_lower_centers), dtype=float)
    opt_res = None
    if start_eval is not None and start_eval.get("success", False):
        opt_res = start_eval.get("res")
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
        "opt_res": opt_res,
        "is_rebase_start": True,
    }


def _spring_reopt_succeeded(spring_out, reference_err):
    if spring_out is None:
        return False
    err = safe_float(spring_out.get("err_opt"))
    return np.isfinite(err) and err < float(reference_err)


def _run_seed(seed, base_output_dir):
    print("==============================")
    print(f"RUN FOR SEED = {seed}")
    print("==============================")

    SETTINGS["initial_shape"]["random_seed"] = int(seed)
    SETTINGS["run"]["seed"] = int(seed)

    seed_dir = base_output_dir / f"seed_{seed}"
    clean_workdir(seed_dir)
    SETTINGS["xfoil"]["working_dir"] = seed_dir

    x, yu_target, yl_target = build_naca0012_surfaces(
        n_points=SETTINGS["geom"]["n_points"],
        thickness=SETTINGS["geom"]["thickness"],
    )

    target_dat = seed_dir / "target_airfoil.dat"
    write_dat(target_dat, x, yu_target, yl_target, name="TARGET_NACA0012")

    target_res = run_target_aero(
        airfoil_dat=target_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=seed_dir / "target_run",
    )
    if not target_res["success"]:
        print(target_res["stdout"])
        print(target_res["stderr"])
        raise RuntimeError(f"Target aero run failed for seed={seed}.")

    cp_target = split_upper_lower_cp_from_x(
        target_res["cp_data"]["x"],
        target_res["cp_data"]["cp"],
    )

    init_geom = build_random_initial_geometry(
        x=x,
        yu_target=yu_target,
        yl_target=yl_target,
        seed=SETTINGS["initial_shape"]["random_seed"],
        amp=SETTINGS["initial_shape"]["random_amp"],
        order=SETTINGS["initial_shape"]["bernstein_order"],
    )
    yu_init = init_geom["yu_init"]
    yl_init = init_geom["yl_init"]

    init_dat = seed_dir / "initial_airfoil.dat"
    write_dat(init_dat, x, yu_init, yl_init, name="INITIAL_RANDOM")

    init_res = run_aero(
        airfoil_dat=init_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=seed_dir / "initial_run",
    )
    if not init_res["success"]:
        report_initial_aero_failure(seed_dir, seed, init_res)
        clear_cmplxfoil_solver_cache()
        return None

    cp_init = split_upper_lower_cp_from_x(
        init_res["cp_data"]["x"],
        init_res["cp_data"]["cp"],
    )
    err_init = total_cp_error(cp_target, cp_init)

    if SETTINGS.get("constraints", {}).get("CL", {}).get("enabled", False):
        SETTINGS["constraints"]["CL"]["target"] = float(target_res["polar"]["CL"])

    print_initial_state(
        seed=seed,
        err_init=err_init,
        init_res=init_res,
        penalty_scale=(
            SETTINGS["optimization"]["penalty_factor"] * max(err_init, 1.0e-6)
        ),
    )

    periodic_cfg = SETTINGS["periodic_spring_adaptive"]
    levels = [int(v) for v in periodic_cfg.get("levels", [12, 16, 20])]
    levels = sorted(levels)
    if len(levels) == 0:
        raise ValueError("PERIODIC_SPRING_LEVELS must contain at least one level.")
    accept_mode_intermediate = str(
        periodic_cfg.get("accept_mode_intermediate", "rebase_keep_new_centers")
    ).strip().lower()
    accept_mode_final = str(
        periodic_cfg.get("accept_mode_final", "accept_if_improved")
    ).strip().lower()
    if accept_mode_intermediate != "rebase_keep_new_centers":
        raise ValueError(
            "PERIODIC_SPRING_ACCEPT_MODE_INTERMEDIATE currently supports only "
            "'rebase_keep_new_centers'."
        )
    if accept_mode_final != "accept_if_improved":
        raise ValueError(
            "PERIODIC_SPRING_ACCEPT_MODE_FINAL currently supports only "
            "'accept_if_improved'."
        )

    n0 = int(SETTINGS["optimization"]["adaptive"]["n0"])
    initial_upper_centers, initial_lower_centers = build_side_specific_initial_centers(n0)
    current_upper_centers = list(initial_upper_centers)
    current_lower_centers = list(initial_lower_centers)
    current_a0 = np.zeros(len(current_upper_centers) + len(current_lower_centers), dtype=float)
    current_base_yu = np.asarray(yu_init, dtype=float)
    current_base_yl = np.asarray(yl_init, dtype=float)
    current_error = float(err_init)

    total_scoring_aero_calls = 0
    total_optimization_aero_calls = 0
    total_function_evals = 0
    total_gradient_evals = 0
    spring_rebase_start_aero_calls = 0
    block_history = []
    level_summary = {}
    final_out = None
    final_method = ""

    for target_level in levels:
        current_ndv = len(current_upper_centers) + len(current_lower_centers)
        if target_level < current_ndv:
            raise ValueError(
                f"Periodic target level {target_level} is smaller than current ndv {current_ndv}."
            )

        print("===== PERIODIC ADAPT_GRAD BLOCK =====")
        print(f"seed = {seed}")
        print(f"current ndv = {current_ndv}")
        print(f"target ndv  = {target_level}")

        block_root = seed_dir / "periodic" / f"block_{target_level:03d}"
        adapt_out = run_adaptive_strategy_from_state(
            x=x,
            yu_init=current_base_yu,
            yl_init=current_base_yl,
            cp_target=cp_target,
            initial_error=current_error,
            workdir=block_root / "adapt",
            indicator="GRAD",
            initial_upper_centers=current_upper_centers,
            initial_lower_centers=current_lower_centers,
            initial_a0=current_a0,
            n_final_override=target_level,
        )
        save_out_bundle(
            seed_dir / f"block_{target_level:03d}_adapt",
            x=x,
            out=adapt_out,
            name=f"PERIODIC_ADAPT_LEVEL_{target_level}",
        )

        total_scoring_aero_calls += int(adapt_out.get("n_scoring_aero_calls_total", 0))
        total_optimization_aero_calls += int(adapt_out.get("n_optimization_aero_calls_total", 0))
        total_function_evals += int(adapt_out.get("n_function_evals_total", 0))
        total_gradient_evals += int(adapt_out.get("n_gradient_evals_total", 0))

        spring_cfg = SETTINGS["spring_reallocation"]
        realloc_out = reallocate_airfoil_centers(
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
        diagnostics = realloc_out["diagnostics"]
        new_upper_centers = np.asarray(realloc_out["new_upper_centers"], dtype=float)
        new_lower_centers = np.asarray(realloc_out["new_lower_centers"], dtype=float)
        a0_rebase = np.zeros(len(new_upper_centers) + len(new_lower_centers), dtype=float)

        _write_centers_csv(
            seed_dir / f"centers_before_after_level_{target_level:03d}.csv",
            diagnostics,
        )
        _write_diagnostics_csv(
            seed_dir / f"spring_diagnostics_level_{target_level:03d}.csv",
            diagnostics,
        )

        spring_raw_dir = seed_dir / f"block_{target_level:03d}_spring_raw"
        start_eval = evaluate_geometry_state(
            x=x,
            yu_init=adapt_out["yu_opt"],
            yl_init=adapt_out["yl_opt"],
            cp_target=cp_target,
            upper_centers=new_upper_centers,
            lower_centers=new_lower_centers,
            a_vec=a0_rebase,
            label=f"periodic_rebase_start_level_{target_level}",
            workdir=spring_raw_dir / "rebase_start",
            include_aero_calls=True,
        )
        spring_rebase_start_aero_calls += int(start_eval.get("aero_calls", 0))
        _print_rebase_check(target_level, adapt_out["err_opt"], start_eval)

        if not start_eval["success"]:
            print("#############################################")
            print("WARNING: periodic spring rebase start aero evaluation failed.")
            print(f"level = {target_level}")
            print("Continuing with spring reoptimization attempt.")
            print("#############################################")

        spring_out = None
        try:
            spring_out = optimize_for_centers(
                x=x,
                yu_init=adapt_out["yu_opt"],
                yl_init=adapt_out["yl_opt"],
                cp_target=cp_target,
                upper_centers=new_upper_centers,
                lower_centers=new_lower_centers,
                a0=a0_rebase,
                label=f"periodic_spring_rebase_level_{target_level}",
                current_best_error=float(adapt_out["err_opt"]),
                workdir=spring_raw_dir / "reopt",
            )
        except Exception as exc:
            print("#############################################")
            print("WARNING: periodic spring reoptimization failed.")
            print(f"level = {target_level}")
            print(str(exc))
            print("#############################################")

        if spring_out is not None:
            save_out_bundle(
                spring_raw_dir,
                x=x,
                out=spring_out,
                name=f"PERIODIC_SPRING_RAW_LEVEL_{target_level}",
            )
            total_optimization_aero_calls += int(
                spring_out.get("n_optimization_aero_calls_total", 0)
            )
            total_function_evals += int(spring_out.get("n_function_evals", 0))
            total_gradient_evals += int(spring_out.get("n_gradient_evals", 0))

        is_final_level = target_level == levels[-1]
        improved = _spring_reopt_succeeded(spring_out, adapt_out["err_opt"])
        rebase_start_out = _make_rebase_start_out(
            adapt_out,
            new_upper_centers,
            new_lower_centers,
            start_eval,
        )

        if is_final_level:
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
            save_out_bundle(
                seed_dir / "final",
                x=x,
                out=final_out,
                name="PERIODIC_SPRING_ADAPTIVE_FINAL",
            )
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
            current_upper_centers = list(np.asarray(state_for_next["upper_centers"], dtype=float))
            current_lower_centers = list(np.asarray(state_for_next["lower_centers"], dtype=float))
            current_a0 = np.zeros(
                len(current_upper_centers) + len(current_lower_centers),
                dtype=float,
            )
            save_out_bundle(
                seed_dir / f"block_{target_level:03d}_state_for_next",
                x=x,
                out=state_for_next,
                name=f"PERIODIC_STATE_FOR_NEXT_LEVEL_{target_level}",
            )

        level_summary[target_level] = {
            f"J_after_add_{target_level}": safe_float(adapt_out.get("err_opt")),
            f"J_rebase_start_{target_level}": safe_float(start_eval.get("err")),
            f"J_after_spring_{target_level}": safe_float(
                spring_out.get("err_opt") if spring_out is not None else None
            ),
            f"spring_accepted_{target_level}": bool(spring_accepted),
            f"state_used_after_{target_level}": state_used_after,
        }

        block_history.append(
            {
                "target_level": int(target_level),
                "J_before_spring": safe_float(adapt_out.get("err_opt")),
                "J_rebase_start": safe_float(start_eval.get("err")),
                "J_after_spring": safe_float(
                    spring_out.get("err_opt") if spring_out is not None else None
                ),
                "spring_accepted": bool(spring_accepted),
                "intermediate": not is_final_level,
                "state_used_after": state_used_after,
                "old_upper_centers": np.asarray(adapt_out["upper_centers"], dtype=float),
                "new_upper_centers": new_upper_centers,
                "old_lower_centers": np.asarray(adapt_out["lower_centers"], dtype=float),
                "new_lower_centers": new_lower_centers,
            }
        )

    if final_out is None:
        raise RuntimeError("Periodic spring adaptive run did not produce a final state.")

    total_aero_calls = (
        total_scoring_aero_calls
        + total_optimization_aero_calls
        + spring_rebase_start_aero_calls
    )

    summary_row = {
        "seed": int(seed),
        "levels": format_array(levels),
        "J_initial": float(err_init),
        "J_final": safe_float(final_out.get("err_opt")),
        "final_method": final_method,
        "CL_final": polar_metric(final_out, "CL"),
        "CD_final": polar_metric(final_out, "CD"),
        "CM_final": polar_metric(final_out, "CM"),
        "total_scoring_aero_calls": int(total_scoring_aero_calls),
        "total_optimization_aero_calls": int(total_optimization_aero_calls),
        "total_function_evals": int(total_function_evals),
        "total_gradient_evals": int(total_gradient_evals),
        "total_aero_calls": int(total_aero_calls),
        "upper_centers_final": format_array(final_out["upper_centers"]),
        "lower_centers_final": format_array(final_out["lower_centers"]),
    }
    for level in sorted(set([12, 16, 20] + levels)):
        level_data = level_summary.get(level, {})
        summary_row[f"J_after_add_{level}"] = level_data.get(f"J_after_add_{level}", float("nan"))
        summary_row[f"J_rebase_start_{level}"] = level_data.get(
            f"J_rebase_start_{level}",
            float("nan"),
        )
        summary_row[f"J_after_spring_{level}"] = level_data.get(
            f"J_after_spring_{level}",
            float("nan"),
        )
        summary_row[f"spring_accepted_{level}"] = level_data.get(
            f"spring_accepted_{level}",
            "",
        )
        summary_row[f"state_used_after_{level}"] = level_data.get(
            f"state_used_after_{level}",
            "",
        )

    summary_fields = _summary_fields(levels)
    write_csv(seed_dir / "summary.csv", summary_fields, [summary_row])
    _write_block_history_csv(seed_dir / "block_history.csv", block_history)

    final_dict = {
        "seed": int(seed),
        "summary": summary_row,
        "block_history": block_history,
        "final_out": final_out,
    }

    print("===== PERIODIC SPRING ADAPTIVE RECAP =====")
    print(f"seed = {seed}")
    print(f"levels = {levels}")
    print(f"J_initial = {float(err_init):.6e}")
    for item in block_history:
        print(
            f"level {int(item['target_level'])}: "
            f"add={safe_float(item['J_before_spring']):.6e}, "
            f"rebase_start={safe_float(item['J_rebase_start']):.6e}, "
            f"spring={safe_float(item['J_after_spring']):.6e}, "
            f"accepted={bool(item['spring_accepted'])}, "
            f"state={item['state_used_after']}"
        )
    print(f"J_final = {safe_float(final_out.get('err_opt')):.6e}")
    print(f"final_method = {final_method}")
    print(f"total_scoring_aero_calls = {int(total_scoring_aero_calls)}")
    print(f"total_optimization_aero_calls = {int(total_optimization_aero_calls)}")
    print(f"total_aero_calls = {int(total_aero_calls)}")

    clear_cmplxfoil_solver_cache()
    return final_dict


def main():
    args = _parse_args()

    if args.cfg_path:
        apply_cfg_overrides(args.cfg_path)
        print(f"Loaded CFG overrides from: {args.cfg_path}")

    if args.seed is not None:
        SETTINGS["initial_shape"]["seed_list"] = [int(args.seed)]

    periodic_cfg = SETTINGS.get("periodic_spring_adaptive", {})
    if not bool(periodic_cfg.get("enabled", False)):
        print("#############################################")
        print("WARNING: PERIODIC_SPRING_ENABLED is false; running dedicated pipeline anyway.")
        print("#############################################")

    old_grad_score_mode = SETTINGS["optimization"]["adaptive"].get("grad_score_mode", "grad_norm")
    force_mode = str(periodic_cfg.get("force_grad_score_mode", "grad_norm")).strip()
    if force_mode:
        if force_mode != "grad_norm":
            print("#############################################")
            print("WARNING: periodic pipeline currently expects force_grad_score_mode='grad_norm'.")
            print(f"Configured value = {force_mode}")
            print("#############################################")
        SETTINGS["optimization"]["adaptive"]["grad_score_mode"] = force_mode

    base_dir = Path(__file__).resolve().parent
    out_path = Path(args.out)
    if out_path.is_absolute():
        base_output_dir = out_path
    else:
        base_output_dir = base_dir / out_path

    if args.clean_output:
        clean_workdir(base_output_dir)
    else:
        base_output_dir.mkdir(parents=True, exist_ok=True)

    seed_list = SETTINGS["initial_shape"].get("seed_list")
    if seed_list is None:
        n_seeds = int(SETTINGS["initial_shape"]["n_seeds"])
        seeds_to_run = list(range(n_seeds))
    else:
        seeds_to_run = [int(seed) for seed in seed_list]

    summary_rows = []
    try:
        for seed in seeds_to_run:
            out = _run_seed(seed, base_output_dir)
            if out is None:
                continue
            summary_rows.append(out["summary"])
    finally:
        SETTINGS["optimization"]["adaptive"]["grad_score_mode"] = old_grad_score_mode

    if len(summary_rows) == 0:
        print("Nessun seed completato con successo.")
        return

    levels = [int(v) for v in SETTINGS["periodic_spring_adaptive"].get("levels", [12, 16, 20])]
    summary_fields = _summary_fields(levels)
    if len(summary_rows) > 1:
        write_csv(base_output_dir / "summary_all.csv", summary_fields, summary_rows)
    else:
        write_csv(base_output_dir / "summary_all.csv", summary_fields, summary_rows)
    print(f"Periodic spring adaptive summaries written in: {base_output_dir}")


if __name__ == "__main__":
    main()

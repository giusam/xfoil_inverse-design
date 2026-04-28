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
from adaptive_strategy import run_adaptive_strategy
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


SUMMARY_FIELDS = [
    "seed",
    "n0",
    "n_final",
    "n_dv_final",
    "spring_weight_mode",
    "spring_A",
    "spring_omega",
    "spring_max_dx",
    "spring_min_spacing",
    "J_initial",
    "J_grad",
    "J_grad_spring_start",
    "J_grad_spring_raw",
    "J_grad_spring_final",
    "spring_accepted",
    "relative_improvement_raw_vs_grad",
    "relative_improvement_final_vs_grad",
    "CL_grad",
    "CD_grad",
    "CM_grad",
    "CL_grad_spring_raw",
    "CD_grad_spring_raw",
    "CM_grad_spring_raw",
    "CL_final",
    "CD_final",
    "CM_final",
    "grad_scoring_aero_calls",
    "grad_optimization_aero_calls",
    "grad_function_evals",
    "grad_gradient_evals",
    "spring_optimization_aero_calls",
    "spring_function_evals",
    "spring_gradient_evals",
    "total_aero_calls_grad_only",
    "total_aero_calls_grad_spring_raw",
    "total_aero_calls_final",
    "projection_rel_l2_upper",
    "projection_rel_l2_lower",
    "old_upper_centers",
    "new_upper_centers",
    "old_lower_centers",
    "new_lower_centers",
]


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


def _build_summary_row(
    seed,
    grad_out,
    grad_spring_start_err,
    grad_spring_raw_out,
    final_out,
    accepted,
    diagnostics,
    err_init,
    total_grad_only,
    total_grad_spring_raw,
):
    grad_err = float(grad_out["err_opt"])
    raw_err = (
        float(grad_spring_raw_out["err_opt"])
        if grad_spring_raw_out is not None
        else float("nan")
    )
    final_err = float(final_out["err_opt"])

    denom = max(abs(grad_err), 1.0e-16)
    rel_raw = (grad_err - raw_err) / denom if np.isfinite(raw_err) else float("nan")
    rel_final = (grad_err - final_err) / denom

    total_final = total_grad_spring_raw if accepted else total_grad_only

    return {
        "seed": int(seed),
        "n0": int(SETTINGS["optimization"]["adaptive"]["n0"]),
        "n_final": int(SETTINGS["optimization"]["adaptive"]["n_final"]),
        "n_dv_final": int(final_out["ndv_total"]),
        "spring_weight_mode": str(SETTINGS["spring_reallocation"]["weight_mode"]),
        "spring_A": float(SETTINGS["spring_reallocation"]["A"]),
        "spring_omega": float(SETTINGS["spring_reallocation"]["omega"]),
        "spring_max_dx": float(SETTINGS["spring_reallocation"]["max_dx"]),
        "spring_min_spacing": float(SETTINGS["spring_reallocation"]["min_spacing"]),
        "J_initial": float(err_init),
        "J_grad": grad_err,
        "J_grad_spring_start": safe_float(grad_spring_start_err),
        "J_grad_spring_raw": raw_err,
        "J_grad_spring_final": final_err,
        "spring_accepted": bool(accepted),
        "relative_improvement_raw_vs_grad": rel_raw,
        "relative_improvement_final_vs_grad": rel_final,
        "CL_grad": polar_metric(grad_out, "CL"),
        "CD_grad": polar_metric(grad_out, "CD"),
        "CM_grad": polar_metric(grad_out, "CM"),
        "CL_grad_spring_raw": polar_metric(grad_spring_raw_out, "CL"),
        "CD_grad_spring_raw": polar_metric(grad_spring_raw_out, "CD"),
        "CM_grad_spring_raw": polar_metric(grad_spring_raw_out, "CM"),
        "CL_final": polar_metric(final_out, "CL"),
        "CD_final": polar_metric(final_out, "CD"),
        "CM_final": polar_metric(final_out, "CM"),
        "grad_scoring_aero_calls": int(grad_out["n_scoring_aero_calls_total"]),
        "grad_optimization_aero_calls": int(grad_out["n_optimization_aero_calls_total"]),
        "grad_function_evals": int(grad_out["n_function_evals_total"]),
        "grad_gradient_evals": int(grad_out["n_gradient_evals_total"]),
        "spring_optimization_aero_calls": (
            int(grad_spring_raw_out["n_optimization_aero_calls_total"])
            if grad_spring_raw_out is not None
            else 0
        ),
        "spring_function_evals": (
            int(grad_spring_raw_out["n_function_evals"])
            if grad_spring_raw_out is not None
            else 0
        ),
        "spring_gradient_evals": (
            int(grad_spring_raw_out["n_gradient_evals"])
            if grad_spring_raw_out is not None
            else 0
        ),
        "total_aero_calls_grad_only": int(total_grad_only),
        "total_aero_calls_grad_spring_raw": int(total_grad_spring_raw),
        "total_aero_calls_final": int(total_final),
        "projection_rel_l2_upper": float(
            diagnostics["upper_projection"]["relative_projection_error_l2"]
        ),
        "projection_rel_l2_lower": float(
            diagnostics["lower_projection"]["relative_projection_error_l2"]
        ),
        "old_upper_centers": format_array(grad_out["upper_centers"]),
        "new_upper_centers": format_array(diagnostics["upper_spring"]["new_centers"]),
        "old_lower_centers": format_array(grad_out["lower_centers"]),
        "new_lower_centers": format_array(diagnostics["lower_spring"]["new_centers"]),
    }


def _print_rebase_check(j_grad, start_eval):
    print("Rebase start should match ADAPT_GRAD final geometry.")
    print(f"J_grad = {float(j_grad):.6e}")
    print(f"J_start = {safe_float(start_eval.get('err')):.6e}")
    diff = safe_float(start_eval.get("err")) - float(j_grad)
    print(f"Difference = {diff:.6e}")

    rel = abs(diff) / max(abs(float(j_grad)), 1.0e-16)
    if rel > 1.0e-3:
        print("#############################################")
        print("WARNING: rebase start mismatch is larger than expected")
        print(f"relative_difference = {rel:.6e}")
        print("#############################################")


def _print_final_recap(
    seed,
    err_init,
    grad_out,
    start_eval,
    grad_spring_raw_out,
    final_out,
    accepted,
    diagnostics,
    total_grad_only,
    total_grad_spring_raw,
):
    raw_err = (
        float(grad_spring_raw_out["err_opt"])
        if grad_spring_raw_out is not None
        else float("nan")
    )

    print("===== FINAL GRAD + SPRING REBASE RECAP =====")
    print(f"seed = {seed}")
    print(f"Initial Cp error = {float(err_init):.6e}")
    print(f"ADAPT_GRAD Cp error = {float(grad_out['err_opt']):.6e}")
    print(f"GRAD+spring rebase start Cp error = {safe_float(start_eval.get('err')):.6e}")
    print(f"GRAD+spring raw Cp error = {raw_err:.6e}")
    print(f"GRAD+spring final Cp error = {float(final_out['err_opt']):.6e}")
    print(f"Spring accepted = {'YES' if accepted else 'NO'}")
    print()
    print(
        "Projection relative L2 upper/lower = "
        f"{float(diagnostics['upper_projection']['relative_projection_error_l2']):.6e} / "
        f"{float(diagnostics['lower_projection']['relative_projection_error_l2']):.6e}"
    )
    print(f"Old/new centers upper = {format_array(grad_out['upper_centers'])} -> {format_array(diagnostics['upper_spring']['new_centers'])}")
    print(f"Old/new centers lower = {format_array(grad_out['lower_centers'])} -> {format_array(diagnostics['lower_spring']['new_centers'])}")
    print()
    print(
        "GRAD CL/CD/CM = "
        f"{polar_metric(grad_out, 'CL'):.6e} / "
        f"{polar_metric(grad_out, 'CD'):.6e} / "
        f"{polar_metric(grad_out, 'CM'):.6e}"
    )
    print(
        "GRAD+spring raw CL/CD/CM = "
        f"{polar_metric(grad_spring_raw_out, 'CL'):.6e} / "
        f"{polar_metric(grad_spring_raw_out, 'CD'):.6e} / "
        f"{polar_metric(grad_spring_raw_out, 'CM'):.6e}"
    )
    print(
        "FINAL CL/CD/CM = "
        f"{polar_metric(final_out, 'CL'):.6e} / "
        f"{polar_metric(final_out, 'CD'):.6e} / "
        f"{polar_metric(final_out, 'CM'):.6e}"
    )
    print()
    print(f"GRAD scoring aero calls = {int(grad_out['n_scoring_aero_calls_total'])}")
    print(f"GRAD optimization aero calls = {int(grad_out['n_optimization_aero_calls_total'])}")
    spring_calls = (
        int(grad_spring_raw_out["n_optimization_aero_calls_total"])
        if grad_spring_raw_out is not None
        else 0
    )
    print(f"Spring optimization aero calls = {spring_calls}")
    print(f"Total aero calls grad only = {int(total_grad_only)}")
    print(f"Total aero calls grad+spring = {int(total_grad_spring_raw)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADAPT_GRAD + final spring rebase experiment.",
    )
    parser.add_argument(
        "cfg_path",
        nargs="?",
        help="Optional CFG override file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="If provided, run only this seed.",
    )
    parser.add_argument(
        "--out",
        default="grad_spring_rebase",
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


def main():
    args = _parse_args()

    if args.cfg_path:
        apply_cfg_overrides(args.cfg_path)
        print(f"Loaded CFG overrides from: {args.cfg_path}")

    if args.seed is not None:
        SETTINGS["initial_shape"]["seed_list"] = [int(args.seed)]

    restart_mode_cfg = str(
        SETTINGS.get("spring_reallocation", {}).get("restart_mode", "rebase")
    ).strip().lower()
    if restart_mode_cfg != "rebase":
        print("#############################################")
        print(
            "WARNING: SPRING_REALLOC_RESTART_MODE is not 'rebase'. "
            "This standalone V1 forces restart_mode='rebase'."
        )
        print(f"Configured value = {restart_mode_cfg}")
        print("#############################################")

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

    for seed in seeds_to_run:
        print("==============================")
        print(f"RUN FOR SEED = {seed}")
        print("==============================")

        SETTINGS["initial_shape"]["random_seed"] = seed
        SETTINGS["run"]["seed"] = seed

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
            continue

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

        grad_out = run_adaptive_strategy(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            initial_error=err_init,
            workdir=seed_dir / "adapt_grad",
            indicator="GRAD",
        )
        save_out_bundle(
            seed_dir / "adapt_grad_output",
            x=x,
            out=grad_out,
            name="ADAPT_GRAD_FINAL",
        )

        spring_cfg = SETTINGS["spring_reallocation"]
        realloc_out = reallocate_airfoil_centers(
            x=x,
            upper_centers=grad_out["upper_centers"],
            lower_centers=grad_out["lower_centers"],
            a_opt=grad_out["a_opt"],
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

        _write_centers_csv(seed_dir / "centers_before_after.csv", diagnostics)
        _write_diagnostics_csv(seed_dir / "spring_diagnostics.csv", diagnostics)

        new_upper_centers = np.asarray(realloc_out["new_upper_centers"], dtype=float)
        new_lower_centers = np.asarray(realloc_out["new_lower_centers"], dtype=float)
        a0_rebase = np.zeros(len(new_upper_centers) + len(new_lower_centers), dtype=float)

        start_eval = evaluate_geometry_state(
            x=x,
            yu_init=grad_out["yu_opt"],
            yl_init=grad_out["yl_opt"],
            cp_target=cp_target,
            upper_centers=new_upper_centers,
            lower_centers=new_lower_centers,
            a_vec=a0_rebase,
            label="spring_rebase_start",
            workdir=seed_dir / "spring_rebase",
            include_aero_calls=True,
        )
        _print_rebase_check(grad_out["err_opt"], start_eval)

        if not start_eval["success"]:
            print("#############################################")
            print("WARNING: rebase start aero evaluation failed.")
            print("Continuing with spring reoptimization attempt.")
            print("#############################################")

        grad_spring_raw_out = None
        try:
            grad_spring_raw_out = optimize_for_centers(
                x=x,
                yu_init=grad_out["yu_opt"],
                yl_init=grad_out["yl_opt"],
                cp_target=cp_target,
                upper_centers=new_upper_centers,
                lower_centers=new_lower_centers,
                a0=a0_rebase,
                label="spring_rebase_final",
                current_best_error=float(grad_out["err_opt"]),
                workdir=seed_dir / "spring_rebase" / "reopt",
            )
        except Exception as exc:
            print("#############################################")
            print("WARNING: spring reoptimization failed.")
            print(str(exc))
            print("Spring result will be rejected and GRAD will be kept.")
            print("#############################################")

        if grad_spring_raw_out is not None:
            save_out_bundle(
                seed_dir / "spring_rebase_raw",
                x=x,
                out=grad_spring_raw_out,
                name="GRAD_SPRING_REBASE_RAW",
            )

        accepted = (
            grad_spring_raw_out is not None
            and float(grad_spring_raw_out["err_opt"]) < float(grad_out["err_opt"])
        )
        if accepted:
            final_out = grad_spring_raw_out
        else:
            final_out = grad_out

        save_out_bundle(
            seed_dir / "spring_rebase_final",
            x=x,
            out=final_out,
            name="GRAD_SPRING_REBASE_FINAL",
        )

        total_grad_only = int(
            grad_out["n_scoring_aero_calls_total"] + grad_out["n_optimization_aero_calls_total"]
        )
        spring_start_calls = int(start_eval.get("aero_calls", 1))
        spring_opt_calls = (
            int(grad_spring_raw_out["n_optimization_aero_calls_total"])
            if grad_spring_raw_out is not None
            else 0
        )
        total_grad_spring_raw = total_grad_only + spring_start_calls + spring_opt_calls

        summary_row = _build_summary_row(
            seed=seed,
            grad_out=grad_out,
            grad_spring_start_err=start_eval.get("err"),
            grad_spring_raw_out=grad_spring_raw_out,
            final_out=final_out,
            accepted=accepted,
            diagnostics=diagnostics,
            err_init=err_init,
            total_grad_only=total_grad_only,
            total_grad_spring_raw=total_grad_spring_raw,
        )
        write_csv(seed_dir / "summary.csv", SUMMARY_FIELDS, [summary_row])
        summary_rows.append(summary_row)

        _print_final_recap(
            seed=seed,
            err_init=err_init,
            grad_out=grad_out,
            start_eval=start_eval,
            grad_spring_raw_out=grad_spring_raw_out,
            final_out=final_out,
            accepted=accepted,
            diagnostics=diagnostics,
            total_grad_only=total_grad_only,
            total_grad_spring_raw=total_grad_spring_raw,
        )

        clear_cmplxfoil_solver_cache()

    if len(summary_rows) > 0:
        write_csv(base_output_dir / "summary_all.csv", SUMMARY_FIELDS, summary_rows)


if __name__ == "__main__":
    main()

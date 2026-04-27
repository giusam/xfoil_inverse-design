import csv
import sys
from pathlib import Path

import numpy as np

from settings import SETTINGS, apply_cfg_overrides
from geometry import (
    apply_hicks_henne_deformation,
    build_naca0012_surfaces,
    build_random_initial_geometry,
    write_dat,
)
from xfoil_wrapper import clean_workdir, run_xfoil
from cmplxfoil_wrapper import run_cmplxfoil, clear_cmplxfoil_solver_cache
from aero_wrapper import run_aero
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from adaptive_utils import build_hh_centers, split_total_across_sides
from reporting import print_initial_state, report_initial_aero_failure
from spring_reallocation import (
    reallocate_airfoil_centers,
    split_a_by_sides,
)


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


def _format_array(arr):
    return np.array2string(np.asarray(arr, dtype=float), precision=6, separator=", ")


def _start_metric_label(restart_mode):
    if str(restart_mode).strip().lower() == "rebase":
        return "Spring rebase start Cp error"
    return "Spring projected Cp error"


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_geometry_state(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a_vec,
    label,
    workdir,
):
    upper_centers = np.asarray(upper_centers, dtype=float)
    lower_centers = np.asarray(lower_centers, dtype=float)
    a_vec = np.asarray(a_vec, dtype=float)

    a_upper, a_lower = split_a_by_sides(a_vec, len(upper_centers), len(lower_centers))
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

    airfoil_dat = Path(workdir) / f"{label}_airfoil.dat"
    write_dat(airfoil_dat, x, yu, yl, name=label.upper())

    res = run_aero(
        airfoil_dat=airfoil_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / f"{label}_run",
    )
    if not res["success"]:
        return {
            "success": False,
            "res": res,
            "yu": yu,
            "yl": yl,
            "cp_opt": None,
            "err": float("nan"),
        }

    cp_candidate = split_upper_lower_cp_from_x(
        res["cp_data"]["x"],
        res["cp_data"]["cp"],
    )
    err = total_cp_error(cp_target, cp_candidate)
    return {
        "success": True,
        "res": res,
        "yu": yu,
        "yl": yl,
        "cp_opt": cp_candidate,
        "err": float(err),
    }


def _maybe_make_plots(cycle_dir, centers_rows, objective_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    cycle_dir = Path(cycle_dir)

    for side in ("UPPER", "LOWER"):
        rows = [row for row in centers_rows if row["side"] == side]
        if len(rows) == 0:
            continue

        idx = np.array([int(row["index"]) for row in rows], dtype=int)
        old = np.array([float(row["old_center"]) for row in rows], dtype=float)
        new = np.array([float(row["new_center"]) for row in rows], dtype=float)
        dx = np.array([float(row["dx"]) for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(idx, old, "o-", label=f"{side} old")
        ax.plot(idx, new, "s-", label=f"{side} new")
        ax.set_xlabel("Index")
        ax.set_ylabel("Center")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(cycle_dir / f"centers_before_after_{side.lower()}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(idx, dx)
        ax.set_xlabel("Index")
        ax.set_ylabel("dx")
        ax.set_title(f"{side} center displacement")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(cycle_dir / f"dx_barplot_{side.lower()}.png", dpi=160)
        plt.close(fig)

    labels = [row["label"] for row in objective_rows]
    values = [float(row["objective"]) for row in objective_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Cp error")
    ax.set_title("Objective comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cycle_dir / "objective_comparison.png", dpi=160)
    plt.close(fig)


def _write_cycle_outputs(cycle_dir, diagnostics, projected_eval, reopt_out, static_out, restart_mode):
    cycle_dir = Path(cycle_dir)
    cycle_dir.mkdir(parents=True, exist_ok=True)

    centers_rows = []
    for side_key, side_name in (("upper", "UPPER"), ("lower", "LOWER")):
        spring_diag = diagnostics[f"{side_key}_spring"]
        importance = np.asarray(spring_diag["importance"], dtype=float)
        old_centers = np.asarray(spring_diag["old_centers"], dtype=float)
        new_centers = np.asarray(spring_diag["new_centers"], dtype=float)

        for idx, (old_c, new_c, imp) in enumerate(zip(old_centers, new_centers, importance)):
            centers_rows.append(
                {
                    "side": side_name,
                    "index": idx,
                    "old_center": float(old_c),
                    "new_center": float(new_c),
                    "dx": float(new_c - old_c),
                    "importance": float(imp),
                }
            )

    _write_csv(
        cycle_dir / "centers_before_after.csv",
        ["side", "index", "old_center", "new_center", "dx", "importance"],
        centers_rows,
    )

    diag_rows = []
    for side_key, side_name in (("upper", "UPPER"), ("lower", "LOWER")):
        spring_diag = diagnostics[f"{side_key}_spring"]
        proj_diag = diagnostics[f"{side_key}_projection"]
        diag_rows.append(
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

    _write_csv(
        cycle_dir / "spring_diagnostics.csv",
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
        diag_rows,
    )

    objective_rows = []
    objective_rows.append({"label": "static", "objective": float(static_out["err_opt"])})
    if projected_eval is not None and projected_eval.get("success", False):
        objective_rows.append(
            {
                "label": "rebase_start" if str(restart_mode).lower() == "rebase" else "projected_a0",
                "objective": float(projected_eval["err"]),
            }
        )
    objective_rows.append({"label": "reoptimized", "objective": float(reopt_out["err_opt"])})
    _maybe_make_plots(cycle_dir, centers_rows, objective_rows)


def _projection_warning(side_name, proj_diag):
    rel_err = float(proj_diag["relative_projection_error_l2"])
    if rel_err > 0.1:
        print("#############################################")
        print(f"WARNING: large projection error on {side_name}")
        print(f"relative_projection_error_l2 = {rel_err:.6e}")
        print("#############################################")


def _print_cycle_recap(cycle_idx, diagnostics, projected_eval, restart_mode, current_j):
    print(f"\n===== SPRING REALLOCATION CYCLE {cycle_idx} =====")
    print(f"Restart mode = {restart_mode}")

    for side_key, side_name in (("upper", "UPPER"), ("lower", "LOWER")):
        spring_diag = diagnostics[f"{side_key}_spring"]
        proj_diag = diagnostics[f"{side_key}_projection"]

        print(f"{side_name} old centers = {_format_array(spring_diag['old_centers'])}")
        print(f"{side_name} new centers = {_format_array(spring_diag['new_centers'])}")
        print(f"{side_name} dx          = {_format_array(spring_diag['displacement'])}")
        print(
            f"{side_name} projection L2/Linf/rel = "
            f"{proj_diag['projection_error_l2']:.6e} / "
            f"{proj_diag['projection_error_linf']:.6e} / "
            f"{proj_diag['relative_projection_error_l2']:.6e}"
        )
        _projection_warning(side_name, proj_diag)

    if projected_eval is None:
        return

    start_label = _start_metric_label(restart_mode)
    if projected_eval.get("success", False):
        print(f"{start_label} = {projected_eval['err']:.6e}")
        print(f"Start polar = {projected_eval['res']['polar']}")
        if str(restart_mode).strip().lower() == "rebase":
            diff = float(projected_eval["err"]) - float(current_j)
            rel = diff / max(abs(float(current_j)), 1.0e-16)
            print("Rebase start should match current optimized geometry.")
            print(f"Current J = {float(current_j):.6e}")
            print(f"Rebase start J = {float(projected_eval['err']):.6e}")
            print(f"Difference = {diff:.6e}")
            if abs(rel) > 1.0e-3:
                print("#############################################")
                print("WARNING: rebase start differs from current optimized geometry")
                print(f"relative_difference = {rel:.6e}")
                print("#############################################")
    else:
        print(f"{start_label} = FAILED_AERO")
        print(projected_eval["res"].get("stdout", ""))
        print(projected_eval["res"].get("stderr", ""))


def _seed_summary_row(seed, cfg, static_out, projected_eval, final_out):
    j_static = float(static_out["err_opt"])
    j_spring = float(final_out["err_opt"])
    delta_j = j_spring - j_static
    rel_delta_j = delta_j / max(abs(j_static), 1.0e-16)
    j_start_second_opt = float("nan")
    if projected_eval is not None and projected_eval.get("success", False):
        j_start_second_opt = float(projected_eval["err"])

    return {
        "seed": int(seed),
        "n_dv": int(cfg["n_dv"]),
        "weight_mode": str(cfg["weight_mode"]),
        "restart_mode": str(cfg.get("restart_mode", "project")),
        "A": float(cfg["A"]),
        "omega": float(cfg["omega"]),
        "max_dx": float(cfg["max_dx"]),
        "min_spacing": float(cfg["min_spacing"]),
        "J_static": j_static,
        "J_start_second_opt": j_start_second_opt,
        "J_spring": j_spring,
        "delta_J": delta_j,
        "relative_delta_J": rel_delta_j,
        "CL_static": float(static_out["opt_res"]["polar"]["CL"]),
        "CD_static": float(static_out["opt_res"]["polar"]["CD"]),
        "CM_static": float(static_out["opt_res"]["polar"]["CM"]),
        "CL_spring": float(final_out["opt_res"]["polar"]["CL"]),
        "CD_spring": float(final_out["opt_res"]["polar"]["CD"]),
        "CM_spring": float(final_out["opt_res"]["polar"]["CM"]),
        "fev_static": int(static_out["n_function_evals"]),
        "gev_static": int(static_out["n_gradient_evals"]),
        "aero_calls_static": int(static_out["n_optimization_aero_calls_total"]),
        "fev_spring": int(final_out["n_function_evals"]),
        "gev_spring": int(final_out["n_gradient_evals"]),
        "aero_calls_spring": int(final_out["n_optimization_aero_calls_total"]),
    }


def _print_final_recap(
    err_init,
    static_out,
    projected_eval,
    final_out,
    diagnostics,
    restart_mode,
):
    print("\n===== FINAL SPRING REALLOCATION RECAP =====")
    print(f"Restart mode                      = {restart_mode}")
    print(f"Initial Cp error                    = {err_init:.6e}")
    print(f"Static Cp error before reallocation = {static_out['err_opt']:.6e}")

    start_label = _start_metric_label(restart_mode)
    if projected_eval is not None and projected_eval.get("success", False):
        print(f"{start_label:<34} = {projected_eval['err']:.6e}")
    else:
        print(f"{start_label:<34} = FAILED_AERO_OR_SKIPPED")

    print(f"Spring Cp error after reoptimization = {final_out['err_opt']:.6e}")
    print("")

    for side_key, side_name in (("upper", "upper"), ("lower", "lower")):
        proj_diag = diagnostics[f"{side_key}_projection"]
        print(
            f"Projection relative L2 error {side_name} = "
            f"{proj_diag['relative_projection_error_l2']:.6e}"
        )
        print(
            f"Projection Linf error {side_name}        = "
            f"{proj_diag['projection_error_linf']:.6e}"
        )

    print("")
    print(
        f"Static CL/CD/CM before  = "
        f"{static_out['opt_res']['polar']['CL']:.6e}, "
        f"{static_out['opt_res']['polar']['CD']:.6e}, "
        f"{static_out['opt_res']['polar']['CM']:.6e}"
    )
    print(
        f"Spring CL/CD/CM after   = "
        f"{final_out['opt_res']['polar']['CL']:.6e}, "
        f"{final_out['opt_res']['polar']['CD']:.6e}, "
        f"{final_out['opt_res']['polar']['CM']:.6e}"
    )
    print("")
    print(
        f"Function evaluations before/after = "
        f"{static_out['n_function_evals']} / {final_out['n_function_evals']}"
    )
    print(
        f"Gradient evaluations before/after = "
        f"{static_out['n_gradient_evals']} / {final_out['n_gradient_evals']}"
    )
    print(
        f"Optimization aero calls before/after = "
        f"{static_out['n_optimization_aero_calls_total']} / "
        f"{final_out['n_optimization_aero_calls_total']}"
    )
    print("")
    print(f"Old centers upper = {_format_array(diagnostics['upper_spring']['old_centers'])}")
    print(f"New centers upper = {_format_array(diagnostics['upper_spring']['new_centers'])}")
    print(f"Old centers lower = {_format_array(diagnostics['lower_spring']['old_centers'])}")
    print(f"New centers lower = {_format_array(diagnostics['lower_spring']['new_centers'])}")


def _get_seeds_to_run():
    seed_list = SETTINGS["initial_shape"].get("seed_list", None)
    if seed_list is None:
        n_seeds = int(SETTINGS["initial_shape"]["n_seeds"])
        return list(range(n_seeds))
    return [int(s) for s in seed_list]


def main():
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python main_spring_reallocation.py [case.cfg]")

    if len(sys.argv) == 2:
        apply_cfg_overrides(sys.argv[1])
        print(f"Loaded CFG overrides from: {sys.argv[1]}")

    spring_cfg = SETTINGS["spring_reallocation"]
    restart_mode = str(spring_cfg.get("restart_mode", "project")).strip().lower()
    if restart_mode not in {"project", "rebase"}:
        raise ValueError(
            f"Invalid spring_reallocation.restart_mode={restart_mode!r}. "
            "Allowed values: project, rebase."
        )
    n_dv = int(spring_cfg["n_dv"])
    n_cycles = int(spring_cfg["n_cycles"])
    if n_dv <= 0:
        raise ValueError("spring_reallocation.n_dv must be positive.")
    if n_cycles <= 0:
        raise ValueError("spring_reallocation.n_cycles must be positive.")

    base_dir = Path(__file__).resolve().parent
    base_output_dir = base_dir / "spring_reallocation"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    seeds_to_run = _get_seeds_to_run()

    for seed in seeds_to_run:
        print("\n==============================")
        print(f"SPRING REALLOCATION RUN FOR SEED = {seed}")
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
            seed=seed,
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

        n_upper, n_lower = split_total_across_sides(n_dv)
        static_upper = list(build_hh_centers(n_upper))
        static_lower = list(build_hh_centers(n_lower))
        static_a0 = np.zeros(n_dv, dtype=float)

        static_out = optimize_for_centers(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            upper_centers=static_upper,
            lower_centers=static_lower,
            a0=static_a0,
            label="static_uniform",
            current_best_error=err_init,
            workdir=seed_dir / "static",
        )

        current_out = static_out
        last_projected_eval = None
        last_diagnostics = None

        for cycle_idx in range(1, n_cycles + 1):
            cycle_dir = seed_dir / f"cycle_{cycle_idx:02d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            realloc_out = reallocate_airfoil_centers(
                x=x,
                upper_centers=current_out["upper_centers"],
                lower_centers=current_out["lower_centers"],
                a_opt=current_out["a_opt"],
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

            last_diagnostics = realloc_out["diagnostics"]
            if restart_mode == "project":
                second_yu_init = yu_init
                second_yl_init = yl_init
                second_a0 = np.asarray(realloc_out["a0_new"], dtype=float)
                start_label = f"spring_projected_cycle_{cycle_idx:02d}"
                reopt_label = f"spring_reopt_cycle_{cycle_idx:02d}"
                second_current_best_error = min(float(current_out["err_opt"]), float(err_init))
            else:
                n_rebase = len(realloc_out["new_upper_centers"]) + len(realloc_out["new_lower_centers"])
                second_a0 = np.zeros(n_rebase, dtype=float)
                if second_a0.size != n_rebase:
                    raise ValueError("Invalid rebase restart state: zero restart vector has wrong size.")
                second_yu_init = np.asarray(current_out["yu_opt"], dtype=float)
                second_yl_init = np.asarray(current_out["yl_opt"], dtype=float)
                start_label = f"spring_rebased_cycle_{cycle_idx:02d}"
                reopt_label = f"spring_rebase_reopt_cycle_{cycle_idx:02d}"
                second_current_best_error = float(current_out["err_opt"])

            last_projected_eval = _evaluate_geometry_state(
                x=x,
                yu_init=second_yu_init,
                yl_init=second_yl_init,
                cp_target=cp_target,
                upper_centers=realloc_out["new_upper_centers"],
                lower_centers=realloc_out["new_lower_centers"],
                a_vec=second_a0,
                label=start_label,
                workdir=cycle_dir,
            )

            _print_cycle_recap(
                cycle_idx,
                last_diagnostics,
                last_projected_eval,
                restart_mode=restart_mode,
                current_j=float(current_out["err_opt"]),
            )

            reopt_out = optimize_for_centers(
                x=x,
                yu_init=second_yu_init,
                yl_init=second_yl_init,
                cp_target=cp_target,
                upper_centers=realloc_out["new_upper_centers"],
                lower_centers=realloc_out["new_lower_centers"],
                a0=second_a0,
                label=reopt_label,
                current_best_error=second_current_best_error,
                workdir=cycle_dir / "reopt",
            )

            _write_cycle_outputs(
                cycle_dir=cycle_dir,
                diagnostics=last_diagnostics,
                projected_eval=last_projected_eval,
                reopt_out=reopt_out,
                static_out=static_out,
                restart_mode=restart_mode,
            )

            current_out = reopt_out

        if last_diagnostics is None:
            raise RuntimeError("No spring reallocation cycle was executed.")

        _print_final_recap(
            err_init=err_init,
            static_out=static_out,
            projected_eval=last_projected_eval,
            final_out=current_out,
            diagnostics=last_diagnostics,
            restart_mode=restart_mode,
        )

        summary_row = _seed_summary_row(
            seed=seed,
            cfg=spring_cfg,
            static_out=static_out,
            projected_eval=last_projected_eval,
            final_out=current_out,
        )
        summary_rows.append(summary_row)

        _write_csv(
            seed_dir / "summary.csv",
            [
                "seed",
                "n_dv",
                "weight_mode",
                "restart_mode",
                "A",
                "omega",
                "max_dx",
                "min_spacing",
                "J_static",
                "J_start_second_opt",
                "J_spring",
                "delta_J",
                "relative_delta_J",
                "CL_static",
                "CD_static",
                "CM_static",
                "CL_spring",
                "CD_spring",
                "CM_spring",
                "fev_static",
                "gev_static",
                "aero_calls_static",
                "fev_spring",
                "gev_spring",
                "aero_calls_spring",
            ],
            [summary_row],
        )

        clear_cmplxfoil_solver_cache()

    if len(summary_rows) > 0:
        _write_csv(
            base_output_dir / "summary_all.csv",
            [
                "seed",
                "n_dv",
                "weight_mode",
                "restart_mode",
                "A",
                "omega",
                "max_dx",
                "min_spacing",
                "J_static",
                "J_start_second_opt",
                "J_spring",
                "delta_J",
                "relative_delta_J",
                "CL_static",
                "CD_static",
                "CM_static",
                "CL_spring",
                "CD_spring",
                "CM_spring",
                "fev_static",
                "gev_static",
                "aero_calls_static",
                "fev_spring",
                "gev_spring",
                "aero_calls_spring",
            ],
            summary_rows,
        )


if __name__ == "__main__":
    main()

from pathlib import Path
import shutil
import argparse


def _load_runtime_imports():
    global np
    global SETTINGS, apply_cfg_overrides, validate_settings
    global build_naca0012_surfaces, build_random_initial_geometry, write_dat
    global clean_workdir, run_xfoil
    global run_cmplxfoil, clear_cmplxfoil_solver_cache
    global run_aero
    global split_upper_lower_cp_from_x
    global total_cp_error
    global optimize_for_centers
    global cleanup_debug_files
    global build_hh_centers, split_total_across_sides
    global run_adaptive_strategy
    global print_initial_state, report_initial_aero_failure
    global make_method_result_bundle, run_grad_spring_final_pipeline
    global run_grad_spring_periodic_pipeline, save_out_bundle
    global build_method_summary_row, method_summary_fields, write_csv
    global write_standard_method_plots

    import numpy as np

    from settings import SETTINGS, apply_cfg_overrides, validate_settings
    from geometry import build_naca0012_surfaces, build_random_initial_geometry, write_dat
    from xfoil_wrapper import clean_workdir, run_xfoil
    from cmplxfoil_wrapper import run_cmplxfoil, clear_cmplxfoil_solver_cache
    from aero_wrapper import run_aero
    from cp_utils import split_upper_lower_cp_from_x
    from objective import total_cp_error
    from optimization import optimize_for_centers
    from cleanup import cleanup_debug_files
    from adaptive_utils import build_hh_centers, split_total_across_sides
    from adaptive_strategy import run_adaptive_strategy
    from reporting import print_initial_state, report_initial_aero_failure
    from experiment_utils import (
        build_method_summary_row,
        make_method_result_bundle,
        method_summary_fields,
        run_grad_spring_final_pipeline,
        run_grad_spring_periodic_pipeline,
        save_out_bundle,
        write_csv,
        write_standard_method_plots,
    )


def _static_placeholder(err_init, init_res):
    cp_data = {"x": np.array([], dtype=float), "cp": np.array([], dtype=float)}
    return {
        "upper_centers": np.array([], dtype=float),
        "lower_centers": np.array([], dtype=float),
        "a_opt": np.array([], dtype=float),
        "yu_opt": np.array([], dtype=float),
        "yl_opt": np.array([], dtype=float),
        "cp_opt": {
            "upper": {"x": np.array([], dtype=float), "cp": np.array([], dtype=float)},
            "lower": {"x": np.array([], dtype=float), "cp": np.array([], dtype=float)},
        },
        "opt_res": {
            "polar": {
                "alpha": float(init_res["polar"]["alpha"]),
                "CL": float("nan"),
                "CD": float("nan"),
                "CM": float("nan"),
            },
            "cp_data": cp_data,
        },
        "err_opt": float("nan"),
        "result": None,
        "n_objective_evals": 0,
        "n_xfoil_calls_total": 0,
        "n_aero_calls_total_all_phases": 0,
        "n_setup_aero_calls": 0,
        "n_postprocess_aero_calls": 0,
        "n_optimization_aero_calls_total": 0,
        "n_optimization_function_aero_calls": 0,
        "n_optimization_gradient_aero_calls": 0,
        "n_function_evals": 0,
        "n_gradient_evals": 0,
        "objective_history": [],
        "function_eval_history": [],
        "gradient_eval_history": [],
        "aero_call_counter_by_phase": {},
        "ndv_total": 0,
        "penalty_value": 0.0,
        "n_geometric_constraints": 0,
        "is_skipped": True,
        "err_init_reference": float(err_init),
    }


def _get_active_adaptive_modes():
    modes = []
    run_cfg = SETTINGS.get("run", {})
    if bool(run_cfg.get("do_adaptive_grad", False)):
        modes.append("GRAD")
    if bool(run_cfg.get("do_adaptive_ikkt", False)):
        modes.append("IKKT")
    if bool(run_cfg.get("do_adaptive_pred", False)):
        modes.append("PRED")
    if bool(run_cfg.get("do_adaptive_oracle", False)):
        modes.append("ORACLE")
    return modes

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


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Official main for static, adaptive, and adaptive spring inverse airfoil optimization. "
            "Typical use: python3 main_inverse_static.py case.cfg --seed 2 --out run_name"
        ),
    )
    parser.add_argument(
        "cfg_path",
        nargs="?",
        help="Optional case.cfg path with settings overrides.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single initial-shape seed.",
    )
    parser.add_argument(
        "--out",
        default="run_debug",
        help="Base output directory. Relative paths are resolved from the script directory.",
    )
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action="store_true",
        default=True,
        help="Remove the base output directory before running. This is the default.",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Create the base output directory if needed without deleting existing contents.",
    )
    return parser.parse_args()


def main(forced_run_settings=None, forced_adaptive_spring_settings=None):
    args = _parse_args()
    _load_runtime_imports()

    if args.cfg_path is not None:
        apply_cfg_overrides(args.cfg_path)
        print(f"Loaded CFG overrides from: {args.cfg_path}")

    if args.seed is not None:
        SETTINGS["initial_shape"]["seed_list"] = [int(args.seed)]

    if forced_run_settings:
        SETTINGS["run"].update(forced_run_settings)
    if forced_adaptive_spring_settings:
        SETTINGS["adaptive_spring"].update(forced_adaptive_spring_settings)

    validate_settings()

    base_dir = Path(__file__).resolve().parent

    out_path = Path(args.out)
    if out_path.is_absolute():
        base_workdir = out_path
    else:
        base_workdir = base_dir / out_path

    if args.clean_output:
        if base_workdir.exists():
            shutil.rmtree(base_workdir)
        base_workdir.mkdir(parents=True, exist_ok=True)
    else:
        base_workdir.mkdir(parents=True, exist_ok=True)

    for pattern in ("a_history_seed_*.dat", "grad_history_seed_*.dat"):
        for p in base_workdir.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass

    results = []

    snapshots_enabled = bool(SETTINGS.get("snapshots", {}).get("enabled", False))
    snap_root = base_workdir / SETTINGS.get("snapshots", {}).get("dir_name", "snapshots")

    if snapshots_enabled:
        snap_root.mkdir(parents=True, exist_ok=True)

    seed_list = SETTINGS["initial_shape"].get("seed_list", None)
    if seed_list is None:
        n_seeds = int(SETTINGS["initial_shape"]["n_seeds"])
        seeds_to_run = list(range(n_seeds))
    else:
        seeds_to_run = [int(s) for s in seed_list]

    run_cfg = SETTINGS.get("run", {})
    do_static = bool(run_cfg.get("do_static", True))
    do_adaptive_grad = bool(run_cfg.get("do_adaptive_grad", False))
    do_adaptive_ikkt = bool(run_cfg.get("do_adaptive_ikkt", False))
    do_adaptive_pred = bool(run_cfg.get("do_adaptive_pred", False))
    do_adaptive_oracle = bool(run_cfg.get("do_adaptive_oracle", False))
    spring_cfg = SETTINGS.get("adaptive_spring", {})
    do_spring = bool(spring_cfg.get("enabled", False))
    spring_mode = str(spring_cfg.get("mode", "final")).strip().lower()
    do_grad_method = do_adaptive_grad
    adaptive_modes = []
    if do_adaptive_ikkt:
        adaptive_modes.append("IKKT")
    if do_adaptive_pred:
        adaptive_modes.append("PRED")
    if do_adaptive_oracle:
        adaptive_modes.append("ORACLE")

    if not do_static and not do_grad_method and not do_spring and len(adaptive_modes) == 0:
        raise RuntimeError(
            "Activate at least one among RUN_DO_STATIC, RUN_DO_ADAPTIVE_GRAD, ADAPTIVE_SPRING_ENABLED, "
            "RUN_DO_ADAPTIVE_IKKT, RUN_DO_ADAPTIVE_PRED, RUN_DO_ADAPTIVE_ORACLE."
        )
    import gc

    for seed in seeds_to_run:
        print("==============================")
        print(f"RUN FOR SEED = {seed}")
        print("==============================")

        SETTINGS["initial_shape"]["random_seed"] = seed
        SETTINGS["run"]["seed"] = seed
        SETTINGS["xfoil"]["working_dir"] = base_workdir / f"seed_{seed}"

        workdir = SETTINGS["xfoil"]["working_dir"]
        clean_workdir(workdir)

        summary_dir = Path(workdir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        x, yu_target, yl_target = build_naca0012_surfaces(
            n_points=SETTINGS["geom"]["n_points"],
            thickness=SETTINGS["geom"]["thickness"],
        )

        target_dat = Path(workdir) / "target_airfoil.dat"
        write_dat(target_dat, x, yu_target, yl_target, name="TARGET_NACA0012")

        target_res = run_target_aero(
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
            raise RuntimeError(f"Target XFOIL run failed for seed={seed}.")

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

        init_dat = Path(workdir) / "initial_airfoil.dat"
        write_dat(init_dat, x, yu_init, yl_init, name="INITIAL_RANDOM")

        init_res = run_aero(
            airfoil_dat=init_dat,
            alpha_deg=SETTINGS["xfoil"]["alpha"],
            reynolds=SETTINGS["xfoil"]["Re"],
            xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
            timeout=SETTINGS["xfoil"]["timeout"],
            working_dir=Path(workdir) / "initial_run",
        )

        if not init_res["success"]:
            report_initial_aero_failure(summary_dir, seed, init_res)
            cleanup_debug_files(workdir)
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

        method_results = {}
        adaptive_aux = {}

        if do_static:
            n_static = SETTINGS["optimization"]["n_hh_static"]
            n_upper, n_lower = split_total_across_sides(n_static)

            static_upper = list(build_hh_centers(n_upper))
            static_lower = list(build_hh_centers(n_lower))

            static_a0 = np.zeros(len(static_upper) + len(static_lower))

            static_out = optimize_for_centers(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                upper_centers=static_upper,
                lower_centers=static_lower,
                a0=static_a0,
                label="static_final",
                current_best_error=err_init,
                workdir=Path(workdir) / "static",
            )
            save_out_bundle(Path(workdir) / "static", x=x, out=static_out, name="STATIC")
            method_results["STATIC"] = make_method_result_bundle(
                method_name="STATIC",
                seed=seed,
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                cp_init=cp_init,
                out=static_out,
                workdir=Path(workdir) / "static",
                extra={
                    "total_scoring_aero_calls": 0,
                    "total_optimization_aero_calls": int(static_out.get("n_optimization_aero_calls_total", 0)),
                    "total_function_evals": int(static_out.get("n_function_evals", 0)),
                    "total_gradient_evals": int(static_out.get("n_gradient_evals", 0)),
                    "total_aero_calls": int(static_out.get("n_optimization_aero_calls_total", 0)),
                },
            )
        else:
            print("===== STATIC OPTIMIZATION =====")
            print("Skipped because SETTINGS['run']['do_static'] = False")
            static_out = _static_placeholder(err_init, init_res)

        grad_needed_for_spring_final = do_spring and spring_mode == "final"
        grad_out = None
        if do_grad_method or grad_needed_for_spring_final:
            grad_out = run_adaptive_strategy(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                initial_error=err_init,
                workdir=Path(workdir) / "adapt_grad",
                indicator="GRAD",
            )
            save_out_bundle(Path(workdir) / "adapt_grad", x=x, out=grad_out, name="ADAPT_GRAD")
            adaptive_aux["ADAPT_GRAD"] = grad_out
            if do_grad_method:
                method_results["ADAPT_GRAD"] = make_method_result_bundle(
                    method_name="ADAPT_GRAD",
                    seed=seed,
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    cp_init=cp_init,
                    out=grad_out,
                    workdir=Path(workdir) / "adapt_grad",
                )

        old_grad_score_mode = SETTINGS["optimization"]["adaptive"].get("grad_score_mode", "grad_norm")
        force_mode = str(SETTINGS.get("adaptive_spring", {}).get("force_grad_score_mode", "")).strip().lower()
        if do_spring and force_mode:
            if force_mode != "grad_norm":
                print(f"[warning] adaptive spring requested force_grad_score_mode={force_mode!r}; current stabilized mode is 'grad_norm'.")
            SETTINGS["optimization"]["adaptive"]["grad_score_mode"] = force_mode

        try:
            if do_spring and spring_mode == "final":
                spring_adapt_out = grad_out
                if force_mode and str(old_grad_score_mode).strip().lower() != force_mode:
                    spring_adapt_out = None
                final_pipeline = run_grad_spring_final_pipeline(
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    initial_error=err_init,
                    workdir=Path(workdir) / "grad_spring_final",
                    adapt_out=spring_adapt_out,
                )
                final_out = final_pipeline["final_out"]
                method_results["GRAD_SPRING_FINAL"] = make_method_result_bundle(
                    method_name="GRAD_SPRING_FINAL",
                    seed=seed,
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    cp_init=cp_init,
                    out=final_out,
                    workdir=Path(workdir) / "grad_spring_final",
                    extra=final_pipeline,
                )

            if do_spring and spring_mode in {"every_refine", "levels"}:
                periodic_pipeline = run_grad_spring_periodic_pipeline(
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    initial_error=err_init,
                    workdir=Path(workdir) / "grad_spring_periodic",
                )
                periodic_out = periodic_pipeline["final_out"]
                method_results["GRAD_SPRING_PERIODIC"] = make_method_result_bundle(
                    method_name="GRAD_SPRING_PERIODIC",
                    seed=seed,
                    x=x,
                    yu_init=yu_init,
                    yl_init=yl_init,
                    cp_target=cp_target,
                    cp_init=cp_init,
                    out=periodic_out,
                    workdir=Path(workdir) / "grad_spring_periodic",
                    extra=periodic_pipeline,
                )
        finally:
            SETTINGS["optimization"]["adaptive"]["grad_score_mode"] = old_grad_score_mode

        for mode in adaptive_modes:
            mode_key = f"ADAPT_{mode}"
            out = run_adaptive_strategy(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                initial_error=err_init,
                workdir=Path(workdir) / mode_key.lower(),
                indicator=mode,
            )
            save_out_bundle(Path(workdir) / mode_key.lower(), x=x, out=out, name=mode_key)
            method_results[mode_key] = make_method_result_bundle(
                method_name=mode_key,
                seed=seed,
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                cp_init=cp_init,
                out=out,
                workdir=Path(workdir) / mode_key.lower(),
            )

        print("===== FINAL METHOD RECAP =====")
        print(f"seed = {seed}")
        print(f"Initial Cp error = {float(err_init):.6e}")
        for method_name, bundle in method_results.items():
            print(f"{method_name:<22s} Cp error = {float(bundle['err_final']):.6e}")

        write_standard_method_plots(
            method_results=method_results,
            x=x,
            yu_target=yu_target,
            yl_target=yl_target,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            cp_init=cp_init,
            output_dir=Path(workdir) / "plots",
        )

        summary_row = build_method_summary_row(seed=seed, err_init=err_init, method_results=method_results)
        summary_fields = method_summary_fields([summary_row])
        write_csv(Path(workdir) / "summary.csv", summary_fields, [summary_row])
        write_csv(summary_dir / "global_summary.csv", summary_fields, [summary_row])
        results.append(summary_row)

        if snapshots_enabled:
            for mode_key in ["adapt_grad", "grad_spring_final", "grad_spring_periodic"] + [f"adapt_{m.lower()}" for m in adaptive_modes]:
                local_snapshot_dir = Path(workdir) / mode_key / SETTINGS["snapshots"]["dir_name"]
                if not local_snapshot_dir.exists():
                    continue

                persistent_snapshot_dir = snap_root / f"seed_{seed}" / mode_key
                persistent_snapshot_dir.mkdir(parents=True, exist_ok=True)

                for old_file in persistent_snapshot_dir.glob("*.npz"):
                    try:
                        old_file.unlink()
                    except OSError:
                        pass

                npz_files = sorted(local_snapshot_dir.rglob("*.npz"))

                if len(npz_files) == 0:
                    print(f"[warning] no npz snapshots found in: {local_snapshot_dir}")
                    continue

                for idx, src_npz in enumerate(npz_files):
                    dst_npz = persistent_snapshot_dir / f"level_{idx:03d}.npz"
                    shutil.copy2(src_npz, dst_npz)

                print(f"Snapshots written in: {persistent_snapshot_dir}")

        cleanup_debug_files(workdir)

        clear_cmplxfoil_solver_cache()

        for name in [
            "static_out", "method_results",
            "target_res", "init_res",
            "cp_target", "cp_init",
            "x", "yu_target", "yl_target", "yu_init", "yl_init",
        ]:
            if name in locals():
                del locals()[name]

        gc.collect() 

    if len(results) == 0:
        print("Nessun seed completato con successo.")
        return

    global_fields = method_summary_fields(results)
    write_csv(base_workdir / "global_summary.csv", global_fields, results)
    print("\n===== AGGREGATE SUMMARY =====")
    for row in results:
        print(f"seed={row['seed']}  J_initial={float(row['J_initial']):.6e}")


if __name__ == "__main__":
    main()

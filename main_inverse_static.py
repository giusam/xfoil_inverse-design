from pathlib import Path
import shutil

import numpy as np

from settings import SETTINGS
from geometry import build_naca0012_surfaces, build_random_initial_geometry, write_dat
from xfoil_wrapper import clean_workdir, run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from cleanup import cleanup_debug_files
from adaptive_utils import build_hh_centers, split_total_across_sides
from adaptive_strategy import run_adaptive_strategy
from reporting import (
    make_seed_result,
    print_aggregate_summary,
    print_initial_state,
    print_seed_recap,
    report_initial_xfoil_failure,
    save_seed_outputs,
    write_global_summary,
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
        "objective_history": [],
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
    return modes


def main():
    base_dir = Path(__file__).resolve().parent

    for pattern in ("a_history_seed_*.dat", "grad_history_seed_*.dat"):
        for p in base_dir.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass

    results = []
    base_workdir = base_dir / "run_debug"

    snapshots_enabled = bool(SETTINGS.get("snapshots", {}).get("enabled", False))
    snap_root = base_dir / SETTINGS.get("snapshots", {}).get("dir_name", "snap")

    if snapshots_enabled:
        snap_root.mkdir(parents=True, exist_ok=True)

    if base_workdir.exists():
        shutil.rmtree(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)

    seed_list = SETTINGS["initial_shape"].get("seed_list", None)
    if seed_list is None:
        n_seeds = int(SETTINGS["initial_shape"]["n_seeds"])
        seeds_to_run = list(range(n_seeds))
    else:
        seeds_to_run = [int(s) for s in seed_list]

    do_static = bool(SETTINGS.get("run", {}).get("do_static", True))
    adaptive_modes = _get_active_adaptive_modes()

    if not do_static and len(adaptive_modes) == 0:
        raise RuntimeError(
            "Activate at least one among run.do_static, run.do_adaptive_grad, run.do_adaptive_ikkt."
        )

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

        init_res = run_xfoil(
            airfoil_dat=init_dat,
            alpha_deg=SETTINGS["xfoil"]["alpha"],
            reynolds=SETTINGS["xfoil"]["Re"],
            xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
            timeout=SETTINGS["xfoil"]["timeout"],
            working_dir=Path(workdir) / "initial_run",
        )

        if not init_res["success"]:
            report_initial_xfoil_failure(summary_dir, seed, init_res)
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
        else:
            print("===== STATIC OPTIMIZATION =====")
            print("Skipped because SETTINGS['run']['do_static'] = False")
            static_out = _static_placeholder(err_init, init_res)

        adaptive_runs = {}
        snapshot_dirs_to_move = []

        for mode in adaptive_modes:
            mode_key = f"adapt_{mode.lower()}"

            adaptive_runs[mode_key] = run_adaptive_strategy(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                initial_error=err_init,
                workdir=Path(workdir) / mode_key,
                indicator=mode,
            )

            if snapshots_enabled:
                local_snapshot_dir = Path(workdir) / mode_key / SETTINGS["snapshots"]["dir_name"]
                snapshot_dirs_to_move.append((mode_key, local_snapshot_dir))

        print_seed_recap(
            seed=seed,
            err_init=err_init,
            init_res=init_res,
            static_out=static_out,
            adaptive_runs=adaptive_runs,
            target_res=target_res,
        )

        save_seed_outputs(
            summary_dir=summary_dir,
            workdir=workdir,
            seed=seed,
            x=x,
            yu_target=yu_target,
            yl_target=yl_target,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            cp_init=cp_init,
            err_init=err_init,
            init_res=init_res,
            static_out=static_out,
            adaptive_runs=adaptive_runs,
            target_res=target_res,
        )

        results.append(
            make_seed_result(
                seed=seed,
                err_init=err_init,
                static_out=static_out,
                adaptive_runs=adaptive_runs,
            )
        )

        if snapshots_enabled:
            for mode_key, local_snapshot_dir in snapshot_dirs_to_move:
                if not local_snapshot_dir.exists():
                    print(f"[warning] snapshot dir not found: {local_snapshot_dir}")
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

    if len(results) == 0:
        print("Nessun seed completato con successo.")
        return

    write_global_summary(base_workdir, results)
    print_aggregate_summary(results)


if __name__ == "__main__":
    main()
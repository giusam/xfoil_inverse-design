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


def main():
    results = []
    base_workdir = Path("run_debug")
    if base_workdir.exists():
        shutil.rmtree(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)

    for fname in ["ikkt_pairs.log", "ikkt_score.log"]:
        p = Path(fname)
        if p.exists():
            p.unlink()

    n_seeds = int(SETTINGS["initial_shape"]["n_seeds"])
    do_static = bool(SETTINGS.get("run", {}).get("do_static", True))
    do_adaptive = bool(SETTINGS.get("run", {}).get("do_adaptive", True))

    if not do_static and not do_adaptive:
        raise RuntimeError("At least one of run.do_static or run.do_adaptive must be True.")

    for seed in range(n_seeds):
        print("\n\n==============================")
        print(f"RUN FOR SEED = {seed}")
        print("==============================")

        SETTINGS["initial_shape"]["random_seed"] = seed
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
            print("\n===== STATIC OPTIMIZATION =====")
            print("Skipped because SETTINGS['run']['do_static'] = False")
            static_out = _static_placeholder(err_init, init_res)

        if do_adaptive:
            adaptive_out = run_adaptive_strategy(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                initial_error=err_init,
                workdir=Path(workdir),
            )
        else:
            raise RuntimeError("This branch expects run.do_adaptive=True.")

        print_seed_recap(seed, err_init, init_res, static_out, adaptive_out, target_res=target_res)

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
            adaptive_out=adaptive_out,
            target_res=target_res,
        )

        if do_static:
            results.append(make_seed_result(seed, err_init, static_out, adaptive_out))

        cleanup_debug_files(workdir)

    if len(results) == 0:
        print("\nNessun seed completato con successo.")
        return

    write_global_summary(base_workdir, results)
    print_aggregate_summary(results)


if __name__ == "__main__":
    main()

from pathlib import Path
import shutil

import numpy as np

from geometry import write_dat
from plotting import (
    plot_3way_geometry_comparison,
    plot_3way_cp_comparison,
    plot_error_vs_evals_with_refine,
    plot_error_vs_evals_full,
)


def print_initial_state(seed, err_init, init_res, penalty_scale):
    print("\n===== INITIAL STATE =====")
    print(f"seed               = {seed}")
    print(f"Initial Cp error   = {err_init:.6e}")
    print(f"Initial polar      = {init_res['polar']}")
    print(f"Initial CL         = {init_res['polar']['CL']:.6e}")
    print(f"Initial CD         = {init_res['polar']['CD']:.6e}")
    print(f"Initial CM         = {init_res['polar']['CM']:.6e}")
    print(f"Initial dynamic penalty scale = {penalty_scale:.6e}")


def report_initial_xfoil_failure(summary_dir, seed, init_res):
    print(f"\n[WARNING] Initial XFOIL run failed for seed={seed}. Seed skipped.")
    print("stdout:")
    print(init_res["stdout"])
    print("stderr:")
    print(init_res["stderr"])

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("===== FINAL RECAP =====\n")
        f.write(f"seed = {seed}\n")
        f.write("status = FAILED_AT_INITIAL_XFOIL\n")
        f.write("stdout:\n")
        f.write(str(init_res["stdout"]) + "\n")
        f.write("stderr:\n")
        f.write(str(init_res["stderr"]) + "\n")


def _extract_last_ok_objective_parts(out_dict):
    hist = out_dict.get("objective_history", [])
    for item in reversed(hist):
        if item.get("status") == "OK":
            return (
                item.get("objective_base", float("nan")),
                item.get("constraint_penalty", 0.0),
                item.get("objective", float("nan")),
            )
    return float("nan"), float("nan"), float("nan")


def _fmt_or_skipped(value, skipped):
    return "SKIPPED" if skipped else f"{value:.6e}"


def _build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_out, target_res=None):
    static_skipped = bool(static_out.get("is_skipped", False))
    static_jcp, static_pcon, static_jtot = _extract_last_ok_objective_parts(static_out)
    adaptive_jcp, adaptive_pcon, adaptive_jtot = _extract_last_ok_objective_parts(adaptive_out)

    lines = [
        "===== FINAL RECAP =====",
        f"seed                 = {seed}",
        f"Initial Cp error     = {err_init:.6e}",
        f"Static Cp error      = {_fmt_or_skipped(static_out['err_opt'], static_skipped)}",
        f"Adaptive Cp error    = {adaptive_out['err_opt']:.6e}",
        f"Adaptive BEST error  = {adaptive_out['best_err_opt']:.6e}",
        f"Adaptive BEST ndv    = {adaptive_out['best_ndv_total']}",
        "",
        f"Static last Jcp      = {_fmt_or_skipped(static_jcp, static_skipped)}",
        f"Static last Pcon     = {_fmt_or_skipped(static_pcon, static_skipped)}",
        f"Static last Jtot     = {_fmt_or_skipped(static_jtot, static_skipped)}",
        f"Adaptive last Jcp    = {adaptive_jcp:.6e}",
        f"Adaptive last Pcon   = {adaptive_pcon:.6e}",
        f"Adaptive last Jtot   = {adaptive_jtot:.6e}",
        "",
    ]

    if target_res is not None:
        lines.extend(
            [
                f"Target CL            = {target_res['polar']['CL']:.6e}",
                f"Target CD            = {target_res['polar']['CD']:.6e}",
                f"Target CM            = {target_res['polar']['CM']:.6e}",
                "",
            ]
        )

    lines.extend(
        [
            f"Initial CL           = {init_res['polar']['CL']:.6e}",
            f"Initial CD           = {init_res['polar']['CD']:.6e}",
            f"Initial CM           = {init_res['polar']['CM']:.6e}",
            "",
            f"Static CL            = {'SKIPPED' if static_skipped else format(static_out['opt_res']['polar']['CL'], '.6e')}",
            f"Static CD            = {'SKIPPED' if static_skipped else format(static_out['opt_res']['polar']['CD'], '.6e')}",
            f"Static CM            = {'SKIPPED' if static_skipped else format(static_out['opt_res']['polar']['CM'], '.6e')}",
            "",
            f"Adaptive CL          = {adaptive_out['opt_res']['polar']['CL']:.6e}",
            f"Adaptive CD          = {adaptive_out['opt_res']['polar']['CD']:.6e}",
            f"Adaptive CM          = {adaptive_out['opt_res']['polar']['CM']:.6e}",
            "",
            f"Adaptive BEST CL     = {adaptive_out['best_opt_res']['polar']['CL']:.6e}",
            f"Adaptive BEST CD     = {adaptive_out['best_opt_res']['polar']['CD']:.6e}",
            f"Adaptive BEST CM     = {adaptive_out['best_opt_res']['polar']['CM']:.6e}",
            "",
        ]
    )

    if target_res is not None:
        if static_skipped:
            lines.extend(
                [
                    "ΔCL Static           = SKIPPED",
                    f"ΔCL Adaptive         = {adaptive_out['opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}",
                    f"ΔCL Adaptive BEST    = {adaptive_out['best_opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"ΔCL Static           = {static_out['opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}",
                    f"ΔCL Adaptive         = {adaptive_out['opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}",
                    f"ΔCL Adaptive BEST    = {adaptive_out['best_opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}",
                    "",
                ]
            )

    lines.extend(
        [
            f"Static objective evals      = {'SKIPPED' if static_skipped else static_out['n_objective_evals']}",
            f"Static XFOIL calls          = {'SKIPPED' if static_skipped else static_out['n_xfoil_calls_total']}",
            f"Adaptive evals opt          = {adaptive_out['n_optimization_evals_total']}",
            f"Adaptive evals score        = {adaptive_out['n_scoring_evals_total']}",
            f"Adaptive XFOIL calls        = {adaptive_out['n_total_evals']}",
            "",
            "Adaptive history:",
        ]
    )

    for m, err in adaptive_out["history"]:
        lines.append(f"  ndv = {m:2d} | Cp error = {err:.6e}")

    lines.append("")
    lines.append("Adaptive center history:")

    for entry in adaptive_out["center_history"]:
        upper_str = ", ".join(f"{c:.6f}" for c in entry["upper"])
        lower_str = ", ".join(f"{c:.6f}" for c in entry["lower"])
        lines.append(f"  ndv   = {entry['ndv_total']:2d}")
        lines.append(f"    upper = [{upper_str}]")
        lines.append(f"    lower = [{lower_str}]")

    return "\n".join(lines) + "\n"


def print_seed_recap(seed, err_init, init_res, static_out, adaptive_out, target_res=None):
    print()
    print(_build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_out, target_res=target_res), end="")


def save_seed_outputs(
    summary_dir,
    workdir,
    seed,
    x,
    yu_target,
    yl_target,
    yu_init,
    yl_init,
    cp_target,
    cp_init,
    err_init,
    init_res,
    static_out,
    adaptive_out,
    target_res=None,
):
    summary_dir = Path(summary_dir)
    workdir = Path(workdir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    static_skipped = bool(static_out.get("is_skipped", False))

    if not static_skipped:
        plot_3way_geometry_comparison(
            x,
            yu_target,
            yl_target,
            yu_init,
            yl_init,
            static_out["yu_opt"],
            static_out["yl_opt"],
            title="Geometry: Target vs Initial vs Static",
            savepath=summary_dir / "geometry_target_initial_static.png",
        )

    plot_3way_geometry_comparison(
        x,
        yu_target,
        yl_target,
        yu_init,
        yl_init,
        adaptive_out["yu_opt"],
        adaptive_out["yl_opt"],
        title="Geometry: Target vs Initial vs Adaptive",
        savepath=summary_dir / "geometry_target_initial_adaptive.png",
    )

    plot_3way_geometry_comparison(
        x,
        yu_target,
        yl_target,
        yu_init,
        yl_init,
        adaptive_out["best_yu_opt"],
        adaptive_out["best_yl_opt"],
        title="Geometry: Target vs Initial vs Adaptive BEST",
        savepath=summary_dir / "geometry_target_initial_adaptive_best.png",
    )

    if not static_skipped:
        plot_3way_cp_comparison(
            cp_target,
            cp_init,
            static_out["cp_opt"],
            title="Cp: Target vs Initial vs Static",
            savepath=summary_dir / "cp_target_initial_static.png",
        )

    plot_3way_cp_comparison(
        cp_target,
        cp_init,
        adaptive_out["cp_opt"],
        title="Cp: Target vs Initial vs Adaptive",
        savepath=summary_dir / "cp_target_initial_adaptive.png",
    )

    plot_3way_cp_comparison(
        cp_target,
        cp_init,
        adaptive_out["best_cp_opt"],
        title="Cp: Target vs Initial vs Adaptive BEST",
        savepath=summary_dir / "cp_target_initial_adaptive_best.png",
    )

    if not static_skipped:
        plot_error_vs_evals_with_refine(
            initial_err=err_init,
            static_calls=static_out["n_xfoil_calls_total"],
            static_err=static_out["err_opt"],
            adaptive_history=adaptive_out["history"],
            adaptive_evals=adaptive_out["history_evals"],
            savepath=summary_dir / "error_vs_evals.png",
        )

        plot_error_vs_evals_full(
            initial_err=err_init,
            static_history=static_out["objective_history"],
            adaptive_history_full=adaptive_out["history_full"],
            savepath=summary_dir / "error_vs_evals_full.png",
        )

    write_dat(summary_dir / "target_airfoil.dat", x, yu_target, yl_target, name="TARGET_NACA0012")
    write_dat(summary_dir / "initial_airfoil.dat", x, yu_init, yl_init, name="INITIAL_RANDOM")
    if not static_skipped:
        write_dat(summary_dir / "static_airfoil.dat", x, static_out["yu_opt"], static_out["yl_opt"], name="STATIC_OPT")
    write_dat(summary_dir / "adaptive_airfoil.dat", x, adaptive_out["yu_opt"], adaptive_out["yl_opt"], name="ADAPTIVE_OPT")
    write_dat(summary_dir / "adaptive_best_airfoil.dat", x, adaptive_out["best_yu_opt"], adaptive_out["best_yl_opt"], name="ADAPTIVE_BEST_OPT")

    shutil.copy(workdir / "target_run" / "cp.txt", summary_dir / "target_cp.txt")
    shutil.copy(workdir / "target_run" / "polar.txt", summary_dir / "target_polar.txt")
    shutil.copy(workdir / "initial_run" / "cp.txt", summary_dir / "initial_cp.txt")
    shutil.copy(workdir / "initial_run" / "polar.txt", summary_dir / "initial_polar.txt")

    if not static_skipped:
        shutil.copy(
            workdir / "static" / "static_final_optimized_run" / "cp.txt",
            summary_dir / "static_cp.txt",
        )
        shutil.copy(
            workdir / "static" / "static_final_optimized_run" / "polar.txt",
            summary_dir / "static_polar.txt",
        )

    shutil.copy(
        workdir / "adaptive" / f"adaptive_level_{adaptive_out['ndv_total']}_optimized_run" / "cp.txt",
        summary_dir / "adaptive_cp.txt",
    )
    shutil.copy(
        workdir / "adaptive" / f"adaptive_level_{adaptive_out['ndv_total']}_optimized_run" / "polar.txt",
        summary_dir / "adaptive_polar.txt",
    )

    with open(summary_dir / "adaptive_best_cp.txt", "w", encoding="utf-8") as f:
        for xi, cpi in zip(
            adaptive_out["best_opt_res"]["cp_data"]["x"],
            adaptive_out["best_opt_res"]["cp_data"]["cp"],
        ):
            f.write(f"{xi:.10f} {cpi:.10f}\n")

    with open(summary_dir / "adaptive_best_polar.txt", "w", encoding="utf-8") as f:
        p = adaptive_out["best_opt_res"]["polar"]
        f.write(f"alpha {p['alpha']:.10f}\n")
        f.write(f"CL {p['CL']:.10f}\n")
        f.write(f"CD {p['CD']:.10f}\n")
        f.write(f"CM {p['CM']:.10f}\n")

    with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(_build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_out, target_res=target_res))


def make_seed_result(seed, err_init, static_out, adaptive_out):
    return {
        "seed": seed,
        "err_init": err_init,
        "static_err": static_out["err_opt"],
        "adaptive_err": adaptive_out["err_opt"],
        "adaptive_best_err": adaptive_out["best_err_opt"],
        "adaptive_best_ndv": adaptive_out["best_ndv_total"],
        "static_xfoil_calls": static_out["n_xfoil_calls_total"],
        "adaptive_xfoil_calls": adaptive_out["n_total_evals"],
    }


def _compute_aggregate_stats(results):
    init_errs = np.array([r["err_init"] for r in results])
    static_errs = np.array([r["static_err"] for r in results])
    adaptive_errs = np.array([r["adaptive_err"] for r in results])
    adaptive_best_errs = np.array([r["adaptive_best_err"] for r in results])

    return {
        "init_errs": init_errs,
        "static_errs": static_errs,
        "adaptive_errs": adaptive_errs,
        "adaptive_best_errs": adaptive_best_errs,
        "init_mean": np.mean(init_errs),
        "init_std": np.std(init_errs),
        "static_mean": np.mean(static_errs),
        "static_std": np.std(static_errs),
        "adaptive_mean": np.mean(adaptive_errs),
        "adaptive_std": np.std(adaptive_errs),
        "adaptive_best_mean": np.mean(adaptive_best_errs),
        "adaptive_best_std": np.std(adaptive_best_errs),
    }


def write_global_summary(base_workdir, results):
    stats = _compute_aggregate_stats(results)
    global_summary_path = Path(base_workdir) / "global_summary.csv"

    with open(global_summary_path, "w", encoding="utf-8") as f:
        f.write("seed,err_init,static_err,adaptive_err,adaptive_best_err,adaptive_best_ndv,static_xfoil_calls,adaptive_xfoil_calls\n")
        for r in results:
            f.write(
                f"{r['seed']},"
                f"{r['err_init']:.6e},"
                f"{r['static_err']:.6e},"
                f"{r['adaptive_err']:.6e},"
                f"{r['adaptive_best_err']:.6e},"
                f"{r['adaptive_best_ndv']},"
                f"{r['static_xfoil_calls']},"
                f"{r['adaptive_xfoil_calls']}\n"
            )
        f.write("\n")
        f.write("GLOBAL_STATS\n")
        f.write(f"init_mean,{stats['init_mean']:.6e}\n")
        f.write(f"init_std,{stats['init_std']:.6e}\n")
        f.write(f"static_mean,{stats['static_mean']:.6e}\n")
        f.write(f"static_std,{stats['static_std']:.6e}\n")
        f.write(f"adaptive_mean,{stats['adaptive_mean']:.6e}\n")
        f.write(f"adaptive_std,{stats['adaptive_std']:.6e}\n")
        f.write(f"adaptive_best_mean,{stats['adaptive_best_mean']:.6e}\n")
        f.write(f"adaptive_best_std,{stats['adaptive_best_std']:.6e}\n")


def print_aggregate_summary(results):
    stats = _compute_aggregate_stats(results)

    print("\n===== AGGREGATE SUMMARY =====")
    print(f"Initial  mean = {stats['init_mean']:.6e}  std = {stats['init_std']:.6e}")
    print(f"Static   mean = {stats['static_mean']:.6e}  std = {stats['static_std']:.6e}")
    print(f"Adaptive mean = {stats['adaptive_mean']:.6e}  std = {stats['adaptive_std']:.6e}")
    print(
        f"Adaptive BEST mean = {stats['adaptive_best_mean']:.6e}  "
        f"std = {stats['adaptive_best_std']:.6e}"
    )


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


def report_initial_aero_failure(summary_dir, seed, init_res):
    print(f"\n[WARNING] Initial aero run failed for seed={seed}. Seed skipped.")
    print("stdout:")
    print(init_res["stdout"])
    print("stderr:")
    print(init_res["stderr"])

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("===== FINAL RECAP =====\n")
        f.write(f"seed = {seed}\n")
        f.write("status = FAILED_AT_INITIAL_AERO\n")
        f.write("stdout:\n")
        f.write(str(init_res["stdout"]) + "\n")
        f.write("stderr:\n")
        f.write(str(init_res["stderr"]) + "\n")


def _extract_last_ok_j(out_dict):
    hist = out_dict.get("objective_history", [])
    for item in reversed(hist):
        if item.get("status") == "OK":
            return item.get("objective_base", float("nan"))
    return float("nan")


def _fmt_or_skipped(value, skipped):
    return "SKIPPED" if skipped else f"{value:.6e}"


def _mode_title(mode_key):
    return mode_key.upper()

def _write_cp_from_res(res, dst_path):
    if res is None or res.get("cp_data") is None:
        return

    cp_x = np.asarray(res["cp_data"]["x"], dtype=float)
    cp_v = np.real(np.asarray(res["cp_data"]["cp"]))

    data = np.column_stack([cp_x, cp_v])
    np.savetxt(dst_path, data, header="x cp", comments="")


def _write_polar_from_res(res, dst_path):
    if res is None or res.get("polar") is None:
        return

    polar = res["polar"]
    alpha = float(np.real(polar["alpha"]))
    cl = float(np.real(polar["CL"]))
    cd = float(np.real(polar["CD"]))
    cm = float(np.real(polar["CM"]))

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("alpha CL CD CM\n")
        f.write(f"{alpha:.10e} {cl:.10e} {cd:.10e} {cm:.10e}\n")


def _copy_or_dump_aero_outputs(run_dir, summary_dir, stem, res):
    run_dir = Path(run_dir)
    summary_dir = Path(summary_dir)

    cp_src = run_dir / "cp.txt"
    polar_src = run_dir / "polar.txt"

    cp_dst = summary_dir / f"{stem}_cp.txt"
    polar_dst = summary_dir / f"{stem}_polar.txt"

    if cp_src.exists():
        shutil.copy(cp_src, cp_dst)
    else:
        _write_cp_from_res(res, cp_dst)

    if polar_src.exists():
        shutil.copy(polar_src, polar_dst)
    else:
        _write_polar_from_res(res, polar_dst)

def _build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_runs, target_res=None):
    static_skipped = bool(static_out.get("is_skipped", False))
    static_j = _extract_last_ok_j(static_out)

    lines = [
        "===== FINAL RECAP =====",
        f"seed                 = {seed}",
        f"Initial Cp error     = {err_init:.6e}",
        f"Static Cp error      = {_fmt_or_skipped(static_out['err_opt'], static_skipped)}",
    ]

    for mode_key, out in adaptive_runs.items():
        lines.append(f"{_mode_title(mode_key)} Cp error   = {out['err_opt']:.6e}")

    lines.extend([
        "",
        f"Static last J        = {_fmt_or_skipped(static_j, static_skipped)}",
    ])

    for mode_key, out in adaptive_runs.items():
        j_last = _extract_last_ok_j(out)
        tag = _mode_title(mode_key)
        lines.append(f"{tag} last J      = {j_last:.6e}")

    lines.append("")

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
        ]
    )

    for mode_key, out in adaptive_runs.items():
        tag = _mode_title(mode_key)
        lines.extend(
            [
                "",
                f"{tag} CL        = {out['opt_res']['polar']['CL']:.6e}",
                f"{tag} CD        = {out['opt_res']['polar']['CD']:.6e}",
                f"{tag} CM        = {out['opt_res']['polar']['CM']:.6e}",
            ]
        )

    lines.append("")

    if target_res is not None:
        if static_skipped:
            lines.append("ΔCL Static           = SKIPPED")
        else:
            lines.append(f"ΔCL Static           = {static_out['opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}")

        for mode_key, out in adaptive_runs.items():
            tag = _mode_title(mode_key)
            lines.append(f"ΔCL {tag:<14s}= {out['opt_res']['polar']['CL'] - target_res['polar']['CL']:.6e}")

        lines.append("")

    lines.extend(
        [
            f"Static objective evals      = {'SKIPPED' if static_skipped else static_out['n_objective_evals']}",
            f"Static aero calls          = {'SKIPPED' if static_skipped else static_out['n_xfoil_calls_total']}",
        ]
    )

    for mode_key, out in adaptive_runs.items():
        tag = _mode_title(mode_key)
        lines.extend(
            [
                f"{tag} evals opt        = {out['n_optimization_evals_total']}",
                f"{tag} evals score      = {out['n_scoring_evals_total']}",
                f"{tag} aero calls       = {out['n_total_evals']}",
            ]
        )

    for mode_key, out in adaptive_runs.items():
        tag = _mode_title(mode_key)
        lines.extend(["", f"{tag} history:"])
        for m, err in out["history"]:
            lines.append(f"  ndv = {m:2d} | Cp error = {err:.6e}")

        lines.append("")
        lines.append(f"{tag} center history:")
        for entry in out["center_history"]:
            upper_str = ", ".join(f"{c:.6f}" for c in entry["upper"])
            lower_str = ", ".join(f"{c:.6f}" for c in entry["lower"])
            lines.append(f"  ndv   = {entry['ndv_total']:2d}")
            lines.append(f"    upper = [{upper_str}]")
            lines.append(f"    lower = [{lower_str}]")

    return "\n".join(lines) + "\n"


def print_seed_recap(seed, err_init, init_res, static_out, adaptive_runs, target_res=None):
    print()
    print(_build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_runs, target_res=target_res), end="")


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
    adaptive_runs,
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

        plot_3way_cp_comparison(
            cp_target,
            cp_init,
            static_out["cp_opt"],
            title="Cp: Target vs Initial vs Static",
            savepath=summary_dir / "cp_target_initial_static.png",
        )

    for mode_key, out in adaptive_runs.items():
        plot_3way_geometry_comparison(
            x,
            yu_target,
            yl_target,
            yu_init,
            yl_init,
            out["yu_opt"],
            out["yl_opt"],
            title=f"Geometry: Target vs Initial vs {_mode_title(mode_key)}",
            savepath=summary_dir / f"geometry_target_initial_{mode_key}.png",
        )

        plot_3way_cp_comparison(
            cp_target,
            cp_init,
            out["cp_opt"],
            title=f"Cp: Target vs Initial vs {_mode_title(mode_key)}",
            savepath=summary_dir / f"cp_target_initial_{mode_key}.png",
        )

        plot_error_vs_evals_with_refine(
            initial_err=err_init,
            static_calls=None if static_skipped else static_out["n_xfoil_calls_total"],
            static_err=None if static_skipped else static_out["err_opt"],
            adaptive_history=out["history"],
            adaptive_evals=out["history_evals"],
            savepath=summary_dir / f"error_vs_evals_{mode_key}.png",
        )

        plot_error_vs_evals_full(
            initial_err=err_init,
            static_history=[] if static_skipped else static_out["objective_history"],
            adaptive_history_full=out["history_full"],
            savepath=summary_dir / f"error_vs_evals_full_{mode_key}.png",
        )

    write_dat(summary_dir / "target_airfoil.dat", x, yu_target, yl_target, name="TARGET_NACA0012")
    write_dat(summary_dir / "initial_airfoil.dat", x, yu_init, yl_init, name="INITIAL_RANDOM")

    if not static_skipped:
        write_dat(summary_dir / "static_airfoil.dat", x, static_out["yu_opt"], static_out["yl_opt"], name="STATIC_OPT")

    for mode_key, out in adaptive_runs.items():
        write_dat(summary_dir / f"{mode_key}_airfoil.dat", x, out["yu_opt"], out["yl_opt"], name=mode_key.upper())

    _copy_or_dump_aero_outputs(
        workdir / "target_run",
        summary_dir,
        "target",
        target_res,
    )
    _copy_or_dump_aero_outputs(
        workdir / "initial_run",
        summary_dir,
        "initial",
        init_res,
    )

    if not static_skipped:
        _copy_or_dump_aero_outputs(
            workdir / "static" / "static_final_optimized_run",
            summary_dir,
            "static",
            static_out["opt_res"],
        )

    for mode_key, out in adaptive_runs.items():
        _copy_or_dump_aero_outputs(
            workdir / mode_key / "adaptive" / f"adaptive_level_{out['ndv_total']}_optimized_run",
            summary_dir,
            mode_key,
            out["opt_res"],
        )

    with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(_build_seed_summary_text(seed, err_init, init_res, static_out, adaptive_runs, target_res=target_res))


def make_seed_result(seed, err_init, static_out, adaptive_runs):
    row = {
        "seed": seed,
        "err_init": err_init,
        "static_err": static_out["err_opt"],
        "static_xfoil_calls": static_out["n_xfoil_calls_total"],
    }
    for mode_key, out in adaptive_runs.items():
        row[f"{mode_key}_err"] = out["err_opt"]
        row[f"{mode_key}_xfoil_calls"] = out["n_total_evals"]
    return row


def _compute_mean_std(values):
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _compute_aggregate_stats(results):
    stats = {
        "init_mean": float(np.mean([r["err_init"] for r in results])),
        "init_std": float(np.std([r["err_init"] for r in results])),
    }

    static_mean, static_std = _compute_mean_std([r["static_err"] for r in results])
    stats["static_mean"] = static_mean
    stats["static_std"] = static_std

    mode_keys = sorted({k[:-4] for r in results for k in r.keys() if k.endswith("_err") and k not in {"static_err"}})
    stats["mode_keys"] = mode_keys

    for mode_key in mode_keys:
        mean_v, std_v = _compute_mean_std([r.get(f"{mode_key}_err", float("nan")) for r in results])
        stats[f"{mode_key}_mean"] = mean_v
        stats[f"{mode_key}_std"] = std_v

    return stats


def write_global_summary(base_workdir, results):
    stats = _compute_aggregate_stats(results)
    global_summary_path = Path(base_workdir) / "global_summary.csv"
    mode_keys = stats["mode_keys"]

    header = ["seed", "err_init", "static_err"]
    for mode_key in mode_keys:
        header.extend([f"{mode_key}_err", f"{mode_key}_xfoil_calls"])
    header.append("static_xfoil_calls")

    with open(global_summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            row = [
                str(r["seed"]),
                f"{r['err_init']:.6e}",
                f"{r['static_err']:.6e}",
            ]
            for mode_key in mode_keys:
                row.append(f"{r.get(f'{mode_key}_err', float('nan')):.6e}")
                row.append(str(r.get(f"{mode_key}_xfoil_calls", "")))
            row.append(str(r["static_xfoil_calls"]))
            f.write(",".join(row) + "\n")

        f.write("\nGLOBAL_STATS\n")
        f.write(f"init_mean,{stats['init_mean']:.6e}\n")
        f.write(f"init_std,{stats['init_std']:.6e}\n")
        f.write(f"static_mean,{stats['static_mean']:.6e}\n")
        f.write(f"static_std,{stats['static_std']:.6e}\n")
        for mode_key in mode_keys:
            f.write(f"{mode_key}_mean,{stats[f'{mode_key}_mean']:.6e}\n")
            f.write(f"{mode_key}_std,{stats[f'{mode_key}_std']:.6e}\n")


def print_aggregate_summary(results):
    stats = _compute_aggregate_stats(results)

    print("\n===== AGGREGATE SUMMARY =====")
    print(f"Initial      mean = {stats['init_mean']:.6e}  std = {stats['init_std']:.6e}")
    print(f"Static       mean = {stats['static_mean']:.6e}  std = {stats['static_std']:.6e}")
    for mode_key in stats["mode_keys"]:
        print(
            f"{mode_key.upper():<12s}mean = {stats[f'{mode_key}_mean']:.6e}  "
            f"std = {stats[f'{mode_key}_std']:.6e}"
        )
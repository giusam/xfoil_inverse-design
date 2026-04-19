from pathlib import Path
import csv
import shutil

import numpy as np

from settings import SETTINGS
from geometry import build_naca0012_surfaces, build_random_initial_geometry, write_dat
from xfoil_wrapper import clean_workdir, run_xfoil
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from optimization import optimize_for_centers
from adaptive_utils import (
    build_side_specific_initial_centers,
    get_midpoint_candidates,
    lift_a_to_new_side_centers,
    _compute_adaptive_nadd,
)
from adaptive_scoring import score_candidate
from adaptive_pred import _prepare_pred_level_context
from adaptive_candidate import _build_extended_space


# =========================
# USER SETTINGS FOR TEST
# =========================
SEED = 0
VALIDATE_AT_NDV = 7          # example: 6 validates the first candidate expansion
STATE_SELECTOR = "GRAD"      # indicator used to reconstruct the state if VALIDATE_AT_NDV > n0
WRITE_CSV = True


def _build_problem(seed, workdir):
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
        raise RuntimeError("Target XFOIL run failed.")

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
        print(init_res["stdout"])
        print(init_res["stderr"])
        raise RuntimeError("Initial XFOIL run failed.")

    cp_init = split_upper_lower_cp_from_x(
        init_res["cp_data"]["x"],
        init_res["cp_data"]["cp"],
    )
    err_init = total_cp_error(cp_target, cp_init)

    if SETTINGS.get("constraints", {}).get("CL", {}).get("enabled", False):
        SETTINGS["constraints"]["CL"]["target"] = float(target_res["polar"]["CL"])

    return {
        "x": x,
        "yu_init": yu_init,
        "yl_init": yl_init,
        "cp_target": cp_target,
        "err_init": err_init,
        "target_res": target_res,
        "init_res": init_res,
    }


def _build_candidates(upper_centers, lower_centers):
    cand_upper_raw = get_midpoint_candidates(upper_centers)
    cand_lower_raw = get_midpoint_candidates(lower_centers)

    candidates = []
    for c in cand_upper_raw:
        candidates.append({"side": "UPPER", "x": float(c["x"]), "interval_id": c["interval_id"]})
    for c in cand_lower_raw:
        candidates.append({"side": "LOWER", "x": float(c["x"]), "interval_id": c["interval_id"]})
    return candidates


def _prepare_state_at_ndv(problem, target_ndv, selector_indicator, workdir):
    x = problem["x"]
    yu_init = problem["yu_init"]
    yl_init = problem["yl_init"]
    cp_target = problem["cp_target"]
    current_best_error = float(problem["err_init"])

    upper_centers, lower_centers = build_side_specific_initial_centers(
        SETTINGS["optimization"]["adaptive"]["n0"]
    )
    a0 = np.zeros(len(upper_centers) + len(lower_centers))

    adaptive_out = optimize_for_centers(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a0=a0,
        label="validate_level_0",
        current_best_error=current_best_error,
        workdir=Path(workdir) / "levels",
    )
    current_best_error = adaptive_out["err_opt"]

    if target_ndv < adaptive_out["ndv_total"]:
        raise ValueError(
            f"Requested VALIDATE_AT_NDV={target_ndv} but initial adaptive level has ndv={adaptive_out['ndv_total']}."
        )

    while adaptive_out["ndv_total"] < target_ndv:
        candidates = _build_candidates(upper_centers, lower_centers)
        if len(candidates) == 0:
            break

        pred_context = None
        if str(selector_indicator).upper() == "PRED":
            pred_context = _prepare_pred_level_context(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                upper_centers=upper_centers,
                lower_centers=lower_centers,
                a_opt=np.asarray(adaptive_out["a_opt"], dtype=float),
                current_best_error=current_best_error,
                workdir=Path(workdir) / "selector_pred_context",
            )

        scored = []
        for cand in candidates:
            info = score_candidate(
                x=x,
                yu_init=yu_init,
                yl_init=yl_init,
                cp_target=cp_target,
                active_upper=upper_centers,
                active_lower=lower_centers,
                active_a=adaptive_out["a_opt"],
                candidate=cand,
                current_best_error=current_best_error,
                workdir=Path(workdir) / "selector_scoring",
                indicator=selector_indicator,
                pred_context=pred_context,
            )
            scored.append(
                {
                    "side": info["side"],
                    "x": info["x"],
                    "interval_id": cand["interval_id"],
                    "score": info["score"],
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["x"]))

        current_ndv = len(upper_centers) + len(lower_centers)
        n_remaining = target_ndv - current_ndv
        n_add = _compute_adaptive_nadd(current_ndv, len(scored))
        n_add = min(n_add, n_remaining)

        if n_add <= 0:
            break

        chosen = scored[:n_add]

        new_upper = sorted(list(upper_centers))
        new_lower = sorted(list(lower_centers))

        for cand in chosen:
            if cand["side"] == "UPPER":
                new_upper.append(float(cand["x"]))
            else:
                new_lower.append(float(cand["x"]))

        new_upper = sorted(set(new_upper))
        new_lower = sorted(set(new_lower))

        new_a0 = lift_a_to_new_side_centers(
            old_upper=upper_centers,
            old_lower=lower_centers,
            old_a=adaptive_out["a_opt"],
            new_upper=new_upper,
            new_lower=new_lower,
        )

        level_label = f"validate_level_{len(new_upper) + len(new_lower)}"

        adaptive_out = optimize_for_centers(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a0=new_a0,
            label=level_label,
            current_best_error=current_best_error,
            workdir=Path(workdir) / "levels",
        )

        current_best_error = adaptive_out["err_opt"]
        upper_centers = new_upper
        lower_centers = new_lower

    return {
        "x": x,
        "yu_init": yu_init,
        "yl_init": yl_init,
        "cp_target": cp_target,
        "upper_centers": list(upper_centers),
        "lower_centers": list(lower_centers),
        "a_opt": np.asarray(adaptive_out["a_opt"], dtype=float).copy(),
        "err_opt": float(adaptive_out["err_opt"]),
        "adaptive_out": adaptive_out,
    }


def _validate_pred_for_state(state, workdir):
    x = state["x"]
    yu_init = state["yu_init"]
    yl_init = state["yl_init"]
    cp_target = state["cp_target"]
    upper_centers = state["upper_centers"]
    lower_centers = state["lower_centers"]
    a_opt = state["a_opt"]
    err_current = float(state["err_opt"])

    candidates = _build_candidates(upper_centers, lower_centers)
    if len(candidates) == 0:
        raise RuntimeError("No candidates available at this level.")

    pred_context = _prepare_pred_level_context(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a_opt=a_opt,
        current_best_error=err_current,
        workdir=Path(workdir) / "pred_context",
    )

    rows = []

    print("\n===== PRED VALIDATION: PREDICTED SCORES =====")
    for cand in candidates:
        info = score_candidate(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            active_upper=upper_centers,
            active_lower=lower_centers,
            active_a=a_opt,
            candidate=cand,
            current_best_error=err_current,
            workdir=Path(workdir) / "pred_scoring",
            indicator="PRED",
            pred_context=pred_context,
        )

        rows.append(
            {
                "side": info["side"],
                "x": float(info["x"]),
                "interval_id": cand["interval_id"],
                "delta_pred": float(info["score"]),
                "d_new": float(info["component"]),
                "raw_grad": float(info["raw_grad"]),
                "mode_pred": info["mode"],
            }
        )

    print("\n===== PRED VALIDATION: REAL RE-OPT FOR EACH CANDIDATE =====")
    for row in rows:
        candidate = {
            "side": row["side"],
            "x": row["x"],
            "interval_id": row["interval_id"],
        }

        side, xc, new_upper, new_lower, _, _, _ = _build_extended_space(
            active_upper=upper_centers,
            active_lower=lower_centers,
            active_a=a_opt,
            candidate=candidate,
        )

        new_a0 = lift_a_to_new_side_centers(
            old_upper=upper_centers,
            old_lower=lower_centers,
            old_a=a_opt,
            new_upper=new_upper,
            new_lower=new_lower,
        )

        label = f"real_reopt_{side}_{xc:.6f}".replace(".", "p")
        out = optimize_for_centers(
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            cp_target=cp_target,
            upper_centers=new_upper,
            lower_centers=new_lower,
            a0=new_a0,
            label=label,
            current_best_error=err_current,
            workdir=Path(workdir) / "real_reopt",
        )

        row["err_before"] = err_current
        row["err_after"] = float(out["err_opt"])
        row["delta_real"] = float(err_current - out["err_opt"])
        row["ndv_after"] = int(out["ndv_total"])

        print(
            f"REAL [{row['side']:<5s} x={row['x']:.6f}]  "
            f"err_before={row['err_before']:.6e}  "
            f"err_after={row['err_after']:.6e}  "
            f"delta_real={row['delta_real']:.6e}"
        )

    pred_sorted = sorted(rows, key=lambda z: (-z["delta_pred"], z["x"]))
    real_sorted = sorted(rows, key=lambda z: (-z["delta_real"], z["x"]))

    pred_rank = {(r["side"], round(r["x"], 6)): i + 1 for i, r in enumerate(pred_sorted)}
    real_rank = {(r["side"], round(r["x"], 6)): i + 1 for i, r in enumerate(real_sorted)}

    for row in rows:
        key = (row["side"], round(row["x"], 6))
        row["rank_pred"] = pred_rank[key]
        row["rank_real"] = real_rank[key]
        if abs(row["delta_pred"]) > 1.0e-16:
            row["rho"] = row["delta_real"] / row["delta_pred"]
        else:
            row["rho"] = np.nan

    rows.sort(key=lambda z: z["rank_pred"])

    print("\n===== PRED VALIDATION SUMMARY =====")
    print("rank_pred | rank_real | side  | x        | delta_pred      | delta_real      | rho")
    for row in rows:
        print(
            f"{row['rank_pred']:9d} | "
            f"{row['rank_real']:9d} | "
            f"{row['side']:<5s} | "
            f"{row['x']:.6f} | "
            f"{row['delta_pred']:.6e} | "
            f"{row['delta_real']:.6e} | "
            f"{row['rho']:.6e}"
        )

    return rows


def main():
    run_dir = Path("run_validate_pred")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    SETTINGS["initial_shape"]["random_seed"] = SEED
    SETTINGS["run"]["seed"] = SEED
    SETTINGS["xfoil"]["working_dir"] = run_dir / f"seed_{SEED}"
    clean_workdir(SETTINGS["xfoil"]["working_dir"])

    problem = _build_problem(SEED, SETTINGS["xfoil"]["working_dir"])

    state = _prepare_state_at_ndv(
        problem=problem,
        target_ndv=VALIDATE_AT_NDV,
        selector_indicator=STATE_SELECTOR,
        workdir=run_dir / "state_build",
    )

    print("\n===== STATE TO VALIDATE =====")
    print(f"seed           = {SEED}")
    print(f"validate_ndv   = {VALIDATE_AT_NDV}")
    print(f"state_selector = {STATE_SELECTOR}")
    print(f"current_err    = {state['err_opt']:.6e}")
    print(f"upper_centers  = {state['upper_centers']}")
    print(f"lower_centers  = {state['lower_centers']}")

    rows = _validate_pred_for_state(state, run_dir / "validation")

    if WRITE_CSV:
        csv_path = run_dir / f"pred_validation_ndv_{VALIDATE_AT_NDV}.csv"
        fieldnames = [
            "side",
            "x",
            "interval_id",
            "delta_pred",
            "d_new",
            "raw_grad",
            "mode_pred",
            "err_before",
            "err_after",
            "delta_real",
            "ndv_after",
            "rank_pred",
            "rank_real",
            "rho",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()

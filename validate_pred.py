from pathlib import Path
import csv
import shutil

import numpy as np

from settings import SETTINGS
from optimization import optimize_for_centers
from adaptive_utils import get_midpoint_candidates, lift_a_to_new_side_centers
from adaptive_scoring import score_candidate
from adaptive_pred import _prepare_pred_level_context
from adaptive_candidate import _build_extended_space


BASE_DIR = Path(__file__).resolve().parent
SNAP_ROOT = BASE_DIR / SETTINGS.get("snapshots", {}).get("dir_name", "snapshots")

SNAP_SEED = 0
SNAP_METHOD = "adapt_grad"   # "adapt_grad" | "adapt_ikkt" | "adapt_pred"
SNAP_LEVEL = 1

SNAPSHOT_PATH = SNAP_ROOT / f"seed_{SNAP_SEED}" / SNAP_METHOD / f"level_{SNAP_LEVEL:03d}.npz"

CANDIDATES_TO_TEST = None
WRITE_CSV = True

def _restore_snapshot_constraint_targets(data):
    for metric_name in ("CL", "CD", "CM"):
        key = f"constraint_{metric_name}_target"
        if key not in data.files:
            continue

        value = float(np.asarray(data[key]).item())
        if np.isnan(value):
            continue

        if metric_name in SETTINGS.get("constraints", {}):
            SETTINGS["constraints"][metric_name]["target"] = value


def _load_level_snapshot(path):
    path = Path(path)
    required_fields = [
        "ndv_total",
        "x",
        "yu_init",
        "yl_init",
        "upper_centers",
        "lower_centers",
        "a_opt",
        "err_opt",
        "yu_opt",
        "yl_opt",
        "cp_target_upper_x",
        "cp_target_upper_cp",
        "cp_target_lower_x",
        "cp_target_lower_cp",
        "cp_opt_upper_x",
        "cp_opt_upper_cp",
        "cp_opt_lower_x",
        "cp_opt_lower_cp",
        "label",
    ]

    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in required_fields if name not in data.files]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Snapshot {path} is missing required fields: {missing_str}")

        _restore_snapshot_constraint_targets(data)

        return {
            "snapshot_path": path,
            "x": np.array(data["x"], dtype=float, copy=True),
            "yu_init": np.array(data["yu_init"], dtype=float, copy=True),
            "yl_init": np.array(data["yl_init"], dtype=float, copy=True),
            "cp_target": {
                "upper": {
                    "x": np.array(data["cp_target_upper_x"], dtype=float, copy=True),
                    "cp": np.array(data["cp_target_upper_cp"], dtype=float, copy=True),
                },
                "lower": {
                    "x": np.array(data["cp_target_lower_x"], dtype=float, copy=True),
                    "cp": np.array(data["cp_target_lower_cp"], dtype=float, copy=True),
                },
            },
            "upper_centers": np.array(data["upper_centers"], dtype=float, copy=True).tolist(),
            "lower_centers": np.array(data["lower_centers"], dtype=float, copy=True).tolist(),
            "a_opt": np.array(data["a_opt"], dtype=float, copy=True),
            "err_opt": float(np.asarray(data["err_opt"]).item()),
            "ndv_total": int(np.asarray(data["ndv_total"]).item()),
            "label": str(np.asarray(data["label"]).item()),
        }


def _build_candidates(upper_centers, lower_centers):
    cand_upper_raw = get_midpoint_candidates(upper_centers)
    cand_lower_raw = get_midpoint_candidates(lower_centers)

    candidates = []
    for cand in cand_upper_raw:
        candidates.append(
            {
                "side": "UPPER",
                "x": float(cand["x"]),
                "interval_id": cand["interval_id"],
            }
        )
    for cand in cand_lower_raw:
        candidates.append(
            {
                "side": "LOWER",
                "x": float(cand["x"]),
                "interval_id": cand["interval_id"],
            }
        )
    return candidates


def _find_candidate(candidates, side, x_target, tol=1.0e-12):
    side = str(side).upper()
    for cand in candidates:
        if str(cand["side"]).upper() != side:
            continue
        if abs(float(cand["x"]) - float(x_target)) <= tol:
            return cand
    raise ValueError(f"Candidate not found: side={side}, x={x_target}")


def _select_candidates(candidates, selected):
    if selected is None:
        return list(candidates)
    return [_find_candidate(candidates, side, x_val) for side, x_val in selected]


def _validate_pred_for_state(state, candidates, workdir):
    x = state["x"]
    yu_init = state["yu_init"]
    yl_init = state["yl_init"]
    cp_target = state["cp_target"]
    upper_centers = state["upper_centers"]
    lower_centers = state["lower_centers"]
    a_opt = state["a_opt"]
    err_current = float(state["err_opt"])

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

    pred_sorted = sorted(rows, key=lambda item: (-item["delta_pred"], item["x"]))
    real_sorted = sorted(rows, key=lambda item: (-item["delta_real"], item["x"]))

    pred_rank = {(row["side"], round(row["x"], 6)): idx + 1 for idx, row in enumerate(pred_sorted)}
    real_rank = {(row["side"], round(row["x"], 6)): idx + 1 for idx, row in enumerate(real_sorted)}

    for row in rows:
        key = (row["side"], round(row["x"], 6))
        row["rank_pred"] = pred_rank[key]
        row["rank_real"] = real_rank[key]
        if abs(row["delta_pred"]) > 1.0e-16:
            row["rho"] = row["delta_real"] / row["delta_pred"]
        else:
            row["rho"] = np.nan

    rows.sort(key=lambda item: item["rank_pred"])

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
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot file not found: {SNAPSHOT_PATH}")

    snapshot = _load_level_snapshot(SNAPSHOT_PATH)
    run_dir = BASE_DIR / "run_validate_pred" / snapshot["snapshot_path"].stem

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== STATE TO VALIDATE =====")
    print(f"snapshot       = {snapshot['snapshot_path']}")
    print(f"label          = {snapshot['label']}")
    print(f"validate_ndv   = {snapshot['ndv_total']}")
    print(f"current_err    = {snapshot['err_opt']:.6e}")
    print(f"upper_centers  = {snapshot['upper_centers']}")
    print(f"lower_centers  = {snapshot['lower_centers']}")

    candidates = _build_candidates(snapshot["upper_centers"], snapshot["lower_centers"])
    test_candidates = _select_candidates(candidates, CANDIDATES_TO_TEST)

    if len(test_candidates) == 0:
        raise RuntimeError("No candidates available for the selected snapshot.")

    rows = _validate_pred_for_state(snapshot, test_candidates, run_dir / "validation")

    if WRITE_CSV:
        csv_path = run_dir / f"pred_validation_{snapshot['snapshot_path'].stem}.csv"
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

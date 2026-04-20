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

SNAP_METHOD = "adapt_grad"   # "adapt_grad" | "adapt_ikkt" | "adapt_pred"
SNAP_SEEDS = [0]
SNAP_LEVELS = [0]   # es: [0, 3, 4, 5, 9] ; None -> tutti i level_*.npz trovati

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


def _candidate_key(side, x):
    return (str(side).upper(), round(float(x), 12))


def _side_order(side):
    return 0 if str(side).upper() == "UPPER" else 1


def _score_sort_key(row, score_key):
    score = float(row.get(score_key, np.nan))
    if np.isfinite(score):
        score_part = -score
    else:
        score_part = np.inf
    return (score_part, float(row["x"]), _side_order(row["side"]))


def _real_sort_key(row):
    delta = float(row.get("delta_real", np.nan))
    if np.isfinite(delta):
        delta_part = -delta
    else:
        delta_part = np.inf
    return (delta_part, float(row["x"]), _side_order(row["side"]))


def _extract_level_from_snapshot_path(path):
    stem = Path(path).stem
    if not stem.startswith("level_"):
        raise ValueError(f"Unexpected snapshot stem: {stem}")
    return int(stem.split("_")[1])


def _discover_snapshot_paths_for_seed(seed):
    seed_dir = SNAP_ROOT / f"seed_{seed}" / SNAP_METHOD
    if not seed_dir.exists():
        print(f"[warning] snapshot directory not found: {seed_dir}")
        return []

    if SNAP_LEVELS is None:
        return sorted(seed_dir.glob("level_*.npz"))

    return [seed_dir / f"level_{int(level):03d}.npz" for level in SNAP_LEVELS]


def _build_snapshot_tasks():
    tasks = []
    for seed in SNAP_SEEDS:
        for snapshot_path in _discover_snapshot_paths_for_seed(seed):
            tasks.append(
                {
                    "seed": int(seed),
                    "snapshot_path": Path(snapshot_path),
                }
            )

    tasks.sort(key=lambda item: (item["seed"], _extract_level_from_snapshot_path(item["snapshot_path"])))
    return tasks


def _compute_indicator_scores(state, candidates, indicator, workdir, pred_context=None):
    x = state["x"]
    yu_init = state["yu_init"]
    yl_init = state["yl_init"]
    cp_target = state["cp_target"]
    upper_centers = state["upper_centers"]
    lower_centers = state["lower_centers"]
    a_opt = state["a_opt"]
    err_current = float(state["err_opt"])

    score_map = {}

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
            workdir=Path(workdir),
            indicator=indicator,
            pred_context=pred_context,
        )

        key = _candidate_key(info["side"], info["x"])
        score_map[key] = float(info["score"])

    return score_map


def _assign_rank(rows, score_key, rank_key):
    sorted_rows = sorted(rows, key=lambda item: _score_sort_key(item, score_key))
    rank_map = {
        _candidate_key(row["side"], row["x"]): idx + 1
        for idx, row in enumerate(sorted_rows)
    }

    for row in rows:
        row[rank_key] = rank_map[_candidate_key(row["side"], row["x"])]


def _assign_real_rank(rows):
    sorted_rows = sorted(rows, key=_real_sort_key)
    rank_map = {
        _candidate_key(row["side"], row["x"]): idx + 1
        for idx, row in enumerate(sorted_rows)
    }

    for row in rows:
        row["rank_real"] = rank_map[_candidate_key(row["side"], row["x"])]


def _validate_state(state, candidates, workdir):
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
    for cand in candidates:
        rows.append(
            {
                "side": str(cand["side"]).upper(),
                "x": float(cand["x"]),
                "interval_id": cand["interval_id"],
            }
        )

    grad_scores = _compute_indicator_scores(
        state=state,
        candidates=candidates,
        indicator="GRAD",
        workdir=Path(workdir) / "score_grad",
        pred_context=None,
    )
    ikkt_scores = _compute_indicator_scores(
        state=state,
        candidates=candidates,
        indicator="IKKT",
        workdir=Path(workdir) / "score_ikkt",
        pred_context=None,
    )
    pred_scores = _compute_indicator_scores(
        state=state,
        candidates=candidates,
        indicator="PRED",
        workdir=Path(workdir) / "score_pred",
        pred_context=pred_context,
    )

    for row in rows:
        key = _candidate_key(row["side"], row["x"])
        row["score_grad"] = float(grad_scores[key])
        row["score_ikkt"] = float(ikkt_scores[key])
        row["score_pred"] = float(pred_scores[key])

    _assign_rank(rows, "score_grad", "rank_grad")
    _assign_rank(rows, "score_ikkt", "rank_ikkt")
    _assign_rank(rows, "score_pred", "rank_pred")

    print("\n===== REAL RE-OPT FOR EACH CANDIDATE =====")
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

        row["err_after"] = float(out["err_opt"])
        row["delta_real"] = float(err_current - out["err_opt"])

        print(
            f"REAL [{row['side']:<5s} x={row['x']:.6f}]  "
            f"delta_real={row['delta_real']:.6e}"
        )

    _assign_real_rank(rows)

    rows.sort(key=lambda item: (float(item["x"]), _side_order(item["side"])))
    return rows


def _write_recap_csv(csv_path, snapshot_name, seed, ndv_before, err_before, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "side",
        "x",
        "score_grad",
        "rank_grad",
        "score_ikkt",
        "rank_ikkt",
        "score_pred",
        "rank_pred",
        "delta_real",
        "rank_real",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# snapshot = {snapshot_name}\n")
        f.write(f"# seed = {seed}\n")
        f.write(f"# ndv_before = {ndv_before}\n")
        f.write(f"# err_before = {err_before:.6e}\n")
        f.write("\n")

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "side": row["side"],
                    "x": f"{float(row['x']):.6f}",
                    "score_grad": f"{float(row['score_grad']):.6e}",
                    "rank_grad": int(row["rank_grad"]),
                    "score_ikkt": f"{float(row['score_ikkt']):.6e}",
                    "rank_ikkt": int(row["rank_ikkt"]),
                    "score_pred": f"{float(row['score_pred']):.6e}",
                    "rank_pred": int(row["rank_pred"]),
                    "delta_real": f"{float(row['delta_real']):.6e}",
                    "rank_real": int(row["rank_real"]),
                }
            )


def main():
    tasks = _build_snapshot_tasks()
    if len(tasks) == 0:
        print("No snapshot tasks found.")
        return

    for task in tasks:
        seed = int(task["seed"])
        snapshot_path = Path(task["snapshot_path"])

        if not snapshot_path.exists():
            print(f"[warning] snapshot file not found: {snapshot_path}")
            continue

        snapshot = _load_level_snapshot(snapshot_path)
        snapshot_name = snapshot["snapshot_path"].stem
        level_dir = BASE_DIR / "run_validate_pred" / f"seed_{seed}" / snapshot_name

        if level_dir.exists():
            shutil.rmtree(level_dir)
        level_dir.mkdir(parents=True, exist_ok=True)

        print("\n============================================================")
        print(f"VALIDATING snapshot = {snapshot_name}")
        print(f"seed               = {seed}")
        print(f"ndv_before         = {snapshot['ndv_total']}")
        print(f"err_before         = {snapshot['err_opt']:.6e}")
        print("============================================================")

        candidates = _build_candidates(snapshot["upper_centers"], snapshot["lower_centers"])
        test_candidates = _select_candidates(candidates, CANDIDATES_TO_TEST)

        if len(test_candidates) == 0:
            print(f"[warning] no candidates available for snapshot: {snapshot_path}")
            continue

        temp_workdir = level_dir / "_tmp"
        rows = _validate_state(snapshot, test_candidates, temp_workdir)

        if WRITE_CSV:
            csv_path = level_dir / "recap.csv"
            _write_recap_csv(
                csv_path=csv_path,
                snapshot_name=snapshot_name,
                seed=seed,
                ndv_before=int(snapshot["ndv_total"]),
                err_before=float(snapshot["err_opt"]),
                rows=rows,
            )
            print(f"\nRecap CSV written to: {csv_path}")

        if temp_workdir.exists():
            shutil.rmtree(temp_workdir)


if __name__ == "__main__":
    main()
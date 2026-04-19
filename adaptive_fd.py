import numpy as np

from adaptive_candidate import _evaluate_objective_state


def _adaptive_fd_step(a_value, rel_step, abs_step_floor):
    return max(float(rel_step) * abs(float(a_value)), float(abs_step_floor))


def _score_fd_component(
    objective,
    a_base,
    idx,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    aj = float(a_base[idx])
    lo, hi = bounds[idx]
    h = _adaptive_fd_step(aj, rel_step, abs_step_floor)

    room_plus = max(0.0, hi - aj)
    room_minus = max(0.0, aj - lo)

    if room_plus >= h and room_minus >= h:
        a_p = a_base.copy()
        a_m = a_base.copy()
        a_p[idx] += h
        a_m[idx] -= h

        item_p = _evaluate_objective_state(objective, a_p)
        item_m = _evaluate_objective_state(objective, a_m)

        if item_p.get("status") != "OK" or item_m.get("status") != "OK":
            return {"grad": 0.0, "mode": "CENTRAL_FAIL", "h": h}

        j_p = item_p.get("objective_base", item_p["objective"])
        j_m = item_m.get("objective_base", item_m["objective"])
        g = (j_p - j_m) / (2.0 * h)
        return {"grad": float(g), "mode": "CENTRAL", "h": h}

    if room_plus > 0.0:
        h_fwd = min(h, room_plus)
        a_p = a_base.copy()
        a_p[idx] += h_fwd

        item_p = _evaluate_objective_state(objective, a_p)

        if base_item.get("status") != "OK" or item_p.get("status") != "OK" or h_fwd <= 0.0:
            return {"grad": 0.0, "mode": "FWD_FAIL", "h": h_fwd}

        j_0 = base_item.get("objective_base", base_item["objective"])
        j_p = item_p.get("objective_base", item_p["objective"])
        g = (j_p - j_0) / h_fwd
        return {"grad": float(g), "mode": "FORWARD", "h": h_fwd}

    if room_minus > 0.0:
        h_bwd = min(h, room_minus)
        a_m = a_base.copy()
        a_m[idx] -= h_bwd

        item_m = _evaluate_objective_state(objective, a_m)

        if base_item.get("status") != "OK" or item_m.get("status") != "OK" or h_bwd <= 0.0:
            return {"grad": 0.0, "mode": "BWD_FAIL", "h": h_bwd}

        j_0 = base_item.get("objective_base", base_item["objective"])
        j_m = item_m.get("objective_base", item_m["objective"])
        g = (j_0 - j_m) / h_bwd
        return {"grad": float(g), "mode": "BACKWARD", "h": h_bwd}

    return {"grad": 0.0, "mode": "NO_ROOM", "h": 0.0}


def _compute_full_objective_gradient(
    objective,
    a_base,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    grad_j = np.zeros(ndv)
    n_fail_dirs = 0
    fail_indices = []

    for j in range(ndv):
        out = _score_fd_component(
            objective=objective,
            a_base=a_base,
            idx=j,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
            base_item=base_item,
        )
        grad_j[j] = float(out["grad"])
        if out["mode"] in ("CENTRAL_FAIL", "FWD_FAIL", "BWD_FAIL", "NO_ROOM"):
            n_fail_dirs += 1
            fail_indices.append(j)

    diag = {"n_fail_dirs": n_fail_dirs, "fail_indices": fail_indices, "ndv": ndv}
    return grad_j, diag


def _compute_full_aero_gradients(
    objective,
    a_base,
    enabled_metric_names,
    bounds,
    rel_step,
    abs_step_floor,
    base_item=None,
):
    a_base = np.asarray(a_base, dtype=float)
    ndv = len(a_base)

    if base_item is None:
        base_item = _evaluate_objective_state(objective, a_base)

    grad_j = np.zeros(ndv)
    grad_metrics = {name: np.zeros(ndv) for name in enabled_metric_names}
    n_fail_dirs = 0
    fail_indices = []

    for j in range(ndv):
        aj = float(a_base[j])
        lo, hi = bounds[j]
        h = _adaptive_fd_step(aj, rel_step, abs_step_floor)

        room_plus = max(0.0, hi - aj)
        room_minus = max(0.0, aj - lo)

        if room_plus >= h and room_minus >= h:
            a_p = a_base.copy()
            a_m = a_base.copy()
            a_p[j] += h
            a_m[j] -= h

            item_p = _evaluate_objective_state(objective, a_p)
            item_m = _evaluate_objective_state(objective, a_m)

            if item_p.get("status") != "OK" or item_m.get("status") != "OK":
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_p = item_p.get("objective_base", item_p["objective"])
            j_m = item_m.get("objective_base", item_m["objective"])
            grad_j[j] = (j_p - j_m) / (2.0 * h)

            metrics_p = item_p.get("metrics", {})
            metrics_m = item_m.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_p[name]) - float(metrics_m[name])) / (2.0 * h)
            continue

        if room_plus > 0.0:
            h_fwd = min(h, room_plus)
            a_p = a_base.copy()
            a_p[j] += h_fwd

            item_0 = base_item
            item_p = _evaluate_objective_state(objective, a_p)

            if item_0.get("status") != "OK" or item_p.get("status") != "OK" or h_fwd <= 0.0:
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_0 = item_0.get("objective_base", item_0["objective"])
            j_p = item_p.get("objective_base", item_p["objective"])
            grad_j[j] = (j_p - j_0) / h_fwd

            metrics_0 = item_0.get("metrics", {})
            metrics_p = item_p.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_p[name]) - float(metrics_0[name])) / h_fwd
            continue

        if room_minus > 0.0:
            h_bwd = min(h, room_minus)
            a_m = a_base.copy()
            a_m[j] -= h_bwd

            item_0 = base_item
            item_m = _evaluate_objective_state(objective, a_m)

            if item_0.get("status") != "OK" or item_m.get("status") != "OK" or h_bwd <= 0.0:
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            j_0 = item_0.get("objective_base", item_0["objective"])
            j_m = item_m.get("objective_base", item_m["objective"])
            grad_j[j] = (j_0 - j_m) / h_bwd

            metrics_0 = item_0.get("metrics", {})
            metrics_m = item_m.get("metrics", {})
            for name in enabled_metric_names:
                grad_metrics[name][j] = (float(metrics_0[name]) - float(metrics_m[name])) / h_bwd
            continue

        n_fail_dirs += 1
        fail_indices.append(j)
        grad_j[j] = 0.0
        for name in enabled_metric_names:
            grad_metrics[name][j] = 0.0

    diag = {"n_fail_dirs": n_fail_dirs, "fail_indices": fail_indices, "ndv": ndv}
    return grad_j, grad_metrics, diag
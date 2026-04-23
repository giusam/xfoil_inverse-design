from pathlib import Path

import numpy as np

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from aero_wrapper import run_aero
from cmplxfoil_wrapper import run_cmplxfoil_coords
from cp_utils import split_upper_lower_cp_from_x
from constraints import compute_metrics


def cp_error_interp(x_ref, cp_ref, x_cmp, cp_cmp):
    x_ref = np.asarray(x_ref, dtype=float)
    cp_ref = np.asarray(cp_ref, dtype=float)
    x_cmp = np.asarray(x_cmp, dtype=float)
    cp_cmp = np.asarray(cp_cmp)

    if len(x_ref) > 1 and x_ref[0] > x_ref[-1]:
        x_ref = x_ref[::-1]
        cp_ref = cp_ref[::-1]

    if len(x_cmp) > 1 and x_cmp[0] > x_cmp[-1]:
        x_cmp = x_cmp[::-1]
        cp_cmp = cp_cmp[::-1]

    cp_cmp_interp = np.interp(x_ref, x_cmp, cp_cmp)
    err_sq = (cp_ref - cp_cmp_interp) ** 2

    x_span = x_ref[-1] - x_ref[0]
    if x_span <= 0.0:
        return np.mean(err_sq)

    return np.trapezoid(err_sq, x_ref) / x_span


def total_cp_error(cp_target, cp_candidate):
    err_upper = cp_error_interp(
        cp_target["upper"]["x"],
        cp_target["upper"]["cp"],
        cp_candidate["upper"]["x"],
        cp_candidate["upper"]["cp"],
    )

    err_lower = cp_error_interp(
        cp_target["lower"]["x"],
        cp_target["lower"]["cp"],
        cp_candidate["lower"]["x"],
        cp_candidate["lower"]["cp"],
    )

    return 0.5 * (err_upper + err_lower)


def make_objective(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    hh_power,
    alpha_deg,
    reynolds,
    xfoil_iter,
    timeout,
    working_dir,
    current_best_error,
):
    eval_counter = {"k": 0}
    eval_history = []
    function_eval_history = []
    gradient_eval_history = []
    current_phase = {"name": "function"}
    real_objective_cache = {}
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    aero_backend = SETTINGS.get("aero", {}).get("backend", "xfoil").strip().lower()

    cmplxfoil_template_dat = working_dir / "_cmplxfoil_template_airfoil.dat"
    cmplxfoil_real_session_key = f"{working_dir.resolve()}::cmplxfoil_real"
    cmplxfoil_cs_session_key = f"{working_dir.resolve()}::cmplxfoil_cs"

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    penalty_factor = SETTINGS["optimization"]["penalty_factor"]
    penalty_floor = 1.0e-6

    # Global logs
    seed_value = SETTINGS.get("run", {}).get("seed", "unknown")
    a_history_path = Path(f"a_history_seed_{seed_value}.dat")
    grad_history_path = Path(f"grad_history_seed_{seed_value}.dat")

    if not a_history_path.exists():
        with open(a_history_path, "w", encoding="utf-8") as f:
            f.write("# workdir eval status objective_total a[0] a[1] ...\n")

    if not grad_history_path.exists():
        with open(grad_history_path, "w", encoding="utf-8") as f:
            f.write("# eval grad[0] grad[1] ...\n")

    def _cache_key(a_vec):
        return tuple(np.round(np.real(np.asarray(a_vec)), 12))

    def _optimization_aero_calls():
        by_phase = objective.aero_call_counter_by_phase
        return int(by_phase.get("function", 0) + by_phase.get("gradient", 0))

    def set_eval_phase(phase):
        current_phase["name"] = str(phase)
        objective.current_phase = current_phase["name"]

    def _record_aero_call():
        phase = current_phase["name"]
        objective.aero_call_counter_total += 1
        objective.aero_call_counter_by_phase[phase] = (
            objective.aero_call_counter_by_phase.get(phase, 0) + 1
        )

    def _store_real_objective_cache(a_vec, objective_value, status, objective_base=None):
        if objective_base is None:
            objective_base = objective_value

        real_objective_cache[_cache_key(a_vec)] = {
            "objective": float(np.real(objective_value)),
            "objective_base": float(np.real(objective_base)),
            "status": status,
        }

    def get_cached_real_objective(a_vec):
        item = real_objective_cache.get(_cache_key(a_vec))
        if item is None:
            return None
        return dict(item)

    def record_function_eval(a_vec):
        cached = get_cached_real_objective(a_vec)
        if cached is None:
            objective_value = float("nan")
            objective_base = float("nan")
            status = "NO_REAL_CACHE"
        else:
            objective_value = cached["objective"]
            objective_base = cached["objective_base"]
            status = cached["status"]

        function_eval_history.append(
            {
                "function_eval_id": int(objective.n_function_evals),
                "objective": objective_value,
                "objective_base": objective_base,
                "status": status,
                "cumulative_optimization_aero_calls": _optimization_aero_calls(),
                "cumulative_function_evals": int(objective.n_function_evals),
                "cumulative_gradient_evals": int(objective.n_gradient_evals),
            }
        )

    def record_gradient_eval(a_vec):
        cached = get_cached_real_objective(a_vec)
        if cached is None:
            objective_value = float("nan")
            objective_base = float("nan")
            status = "NO_REAL_CACHE"
        else:
            objective_value = cached["objective"]
            objective_base = cached["objective_base"]
            status = cached["status"]

        gradient_eval_history.append(
            {
                "gradient_eval_id": int(objective.n_gradient_evals),
                "objective": objective_value,
                "objective_base_at_grad_point": objective_base,
                "status": status,
                "cumulative_optimization_aero_calls": _optimization_aero_calls(),
                "cumulative_function_evals": int(objective.n_function_evals),
                "cumulative_gradient_evals": int(objective.n_gradient_evals),
            }
        )

    def get_penalty():
        scale = max(float(current_best_error), penalty_floor)
        return penalty_factor * scale

    def _append_a_history(k, status, objective_value, a_vec):
        with open(a_history_path, "a", encoding="utf-8") as f:
            coeffs = " ".join(f"{float(ai):.10e}" for ai in np.asarray(a_vec, dtype=float))
            f.write(
                f"{working_dir.name} "
                f"{k:05d} "
                f"{status} "
                f"{float(objective_value):.10e} "
                f"{coeffs}\n"
            )

    def _append_grad_history(k, g_vec):
        with open(grad_history_path, "a", encoding="utf-8") as f:
            grads = " ".join(f"{float(gi):.10e}" for gi in np.asarray(g_vec, dtype=float))
            f.write(f"{k:05d} {grads}\n")

    def _split_a(a):
        nu = len(upper_centers)
        nl = len(lower_centers)

        a_upper = np.asarray(a[:nu])
        a_lower = np.asarray(a[nu:nu + nl])

        if len(a_lower) != nl:
            raise ValueError("Invalid design vector length in objective evaluation.")

        return a_upper, a_lower

    def _build_geometry(a):
        a_upper, a_lower = _split_a(a)

        yu, yl = apply_hicks_henne_deformation(
            x=x,
            yu_base=yu_init,
            yl_base=yl_init,
            a_upper=a_upper,
            a_lower=a_lower,
            upper_centers=upper_centers,
            lower_centers=lower_centers,
            power=hh_power,
        )
        return yu, yl

    def _thickness_check(yu, yl):
        thickness = yu - yl
        thickness_real = np.real(thickness)
        min_thickness = np.min(thickness_real)
        thickness_internal = thickness_real[1:-1]
        thickness_tol = -1.0e-10
        fail = np.any(thickness_internal < thickness_tol)
        return fail, min_thickness

    def _evaluate_objective_real(a):
        eval_counter["k"] += 1
        k = eval_counter["k"]

        yu, yl = _build_geometry(a)
        fail_thickness, min_thickness = _thickness_check(yu, yl)

        if fail_thickness:
            penalty_value = get_penalty()
            print(
                f"eval={k:04d}  FAIL thickness  "
                f"min_thickness={min_thickness:.6e}  penalty={penalty_value:.6e}"
            )

            _append_a_history(k, "FAIL_THICKNESS", penalty_value, a)

            eval_history.append(
                {
                    "eval": k,
                    "objective": penalty_value,
                    "status": "FAIL_THICKNESS",
                }
            )
            _store_real_objective_cache(a, penalty_value, "FAIL_THICKNESS")
            return penalty_value

        if aero_backend == "cmplxfoil":
            if not cmplxfoil_template_dat.exists():
                write_dat(
                    cmplxfoil_template_dat,
                    x,
                    np.real(yu_init),
                    np.real(yl_init),
                    name="CMPLXFOIL_TEMPLATE",
                )

            run_dir = working_dir / "cmplxfoil_objective_run"

            _record_aero_call()
            res = run_cmplxfoil_coords(
                airfoil_dat=cmplxfoil_template_dat,
                x=x,
                yu=yu,
                yl=yl,
                alpha_deg=alpha_deg,
                reynolds=reynolds,
                xfoil_iter=xfoil_iter,
                timeout=timeout,
                working_dir=run_dir,
                session_key=cmplxfoil_real_session_key,
            )
        else:
            airfoil_dat = working_dir / f"candidate_{k:05d}.dat"
            write_dat(airfoil_dat, x, yu, yl, name=f"CAND_{k:05d}")

            run_dir = working_dir / f"candidate_run_{k:05d}"

            _record_aero_call()
            res = run_aero(
                airfoil_dat=airfoil_dat,
                alpha_deg=alpha_deg,
                reynolds=reynolds,
                xfoil_iter=xfoil_iter,
                timeout=timeout,
                working_dir=run_dir,
            )
        if not res["success"]:
            penalty_value = get_penalty()
            print(f"eval={k:04d}  FAIL aero  penalty={penalty_value:.6e}")

            _append_a_history(k, "FAIL_AERO", penalty_value, a)

            eval_history.append(
                {
                    "eval": k,
                    "objective": penalty_value,
                    "status": "FAIL_AERO",
                }
            )
            _store_real_objective_cache(a, penalty_value, "FAIL_AERO")
            return penalty_value

        cp_candidate = split_upper_lower_cp_from_x(
            res["cp_data"]["x"],
            res["cp_data"]["cp"],
        )

        err = total_cp_error(cp_target, cp_candidate)
        metrics = compute_metrics(x, yu, yl, res["polar"])
        objective_total = err

        print(
            f"eval={k:04d}  OK   "
            f"J={err:.6e}  "
            f"min_thickness={min_thickness:.6e}"
        )

        _append_a_history(k, "OK", objective_total, a)

        eval_history.append(
            {
                "eval": k,
                "objective": objective_total,
                "objective_base": err,
                "metrics": metrics,
                "status": "OK",
            }
        )

        _store_real_objective_cache(a, objective_total, "OK", objective_base=err)
        return objective_total

    def _evaluate_aero_state_cs(a, enabled_metric_names=None):
        if aero_backend != "cmplxfoil":
            raise RuntimeError("Complex-step aero evaluation requires aero.backend='cmplxfoil'.")

        if enabled_metric_names is None:
            enabled_metric_names = []

        yu, yl = _build_geometry(a)
        fail_thickness, _ = _thickness_check(yu, yl)

        if fail_thickness:
            return None

        if not cmplxfoil_template_dat.exists():
            write_dat(
                cmplxfoil_template_dat,
                x,
                np.real(yu_init),
                np.real(yl_init),
                name="CMPLXFOIL_TEMPLATE",
            )

        cs_run_dir = working_dir / "cmplxfoil_cs_run"

        _record_aero_call()
        res = run_cmplxfoil_coords(
            airfoil_dat=cmplxfoil_template_dat,
            x=x,
            yu=yu,
            yl=yl,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
            xfoil_iter=xfoil_iter,
            timeout=timeout,
            working_dir=cs_run_dir,
            session_key=cmplxfoil_cs_session_key,
        )

        if not res["success"] or res.get("polar") is None or res.get("cp_data") is None:
            return None

        cp_candidate = split_upper_lower_cp_from_x(
            res["cp_data"]["x"],
            res["cp_data"]["cp"],
        )

        err = total_cp_error(cp_target, cp_candidate)

        metrics = {}
        polar = res["polar"]

        for name in enabled_metric_names:
            if name == "CL":
                metrics[name] = polar["CL"]
            elif name == "CD":
                metrics[name] = polar["CD"]
            elif name == "CM":
                metrics[name] = polar["CM"]
            else:
                raise ValueError(f"Unsupported aero metric for CS gradients: {name}")

        return {
            "objective_base": err,
            "metrics": metrics,
            "polar": polar,
        }

    def _evaluate_objective_cs(a):
        state_cs = _evaluate_aero_state_cs(a, enabled_metric_names=[])
        if state_cs is None:
            return complex(get_penalty(), 0.0)
        return state_cs["objective_base"]

    def objective(a):
        return _evaluate_objective_real(a)

    def compute_gradient_snapshot(a, h=2.0e-5):
        g = np.zeros_like(a, dtype=float)

        for j in range(len(a)):
            a_p = np.asarray(a, dtype=float).copy()
            a_m = np.asarray(a, dtype=float).copy()

            a_p[j] += h
            a_m[j] -= h

            Jp = objective(a_p)
            Jm = objective(a_m)

            status_p = eval_history[-2]["status"]
            status_m = eval_history[-1]["status"]

            if status_p != "OK" or status_m != "OK":
                g[j] = 0.0
            else:
                g[j] = (Jp - Jm) / (2.0 * h)

        return g

    def compute_gradient_cs(a, h=1.0e-200):
        a0 = np.asarray(a, dtype=float)
        g = np.zeros_like(a0, dtype=float)

        for j in range(len(a0)):
            a_cs = np.asarray(a0, dtype=complex).copy()
            a_cs[j] += 1j * h

            J_cs = _evaluate_objective_cs(a_cs)
            g[j] = np.imag(J_cs) / h

        return g

    def compute_aero_gradients_cs(a, enabled_metric_names, h=1.0e-200):
        if aero_backend != "cmplxfoil":
            raise RuntimeError("Complex-step aero gradients require aero.backend='cmplxfoil'.")

        a0 = np.asarray(a, dtype=float)
        grad_j = np.zeros_like(a0, dtype=float)
        grad_metrics = {name: np.zeros_like(a0, dtype=float) for name in enabled_metric_names}

        n_fail_dirs = 0
        fail_indices = []

        for j in range(len(a0)):
            a_cs = np.asarray(a0, dtype=complex).copy()
            a_cs[j] += 1j * h

            state_cs = _evaluate_aero_state_cs(
                a_cs,
                enabled_metric_names=enabled_metric_names,
            )

            if state_cs is None:
                n_fail_dirs += 1
                fail_indices.append(j)
                grad_j[j] = 0.0
                for name in enabled_metric_names:
                    grad_metrics[name][j] = 0.0
                continue

            grad_j[j] = np.imag(state_cs["objective_base"]) / h

            for name in enabled_metric_names:
                grad_metrics[name][j] = np.imag(state_cs["metrics"][name]) / h

        diag = {
            "n_fail_dirs": n_fail_dirs,
            "fail_indices": fail_indices,
            "ndv": len(a0),
            "mode": "CS",
        }

        return grad_j, grad_metrics, diag

    objective.eval_counter = eval_counter
    objective.eval_history = eval_history
    objective.aero_call_counter_total = 0
    objective.aero_call_counter_by_phase = {}
    objective.n_function_evals = 0
    objective.n_gradient_evals = 0
    objective.function_eval_history = function_eval_history
    objective.gradient_eval_history = gradient_eval_history
    objective.current_phase = current_phase["name"]
    objective.set_eval_phase = set_eval_phase
    objective.record_aero_call = _record_aero_call
    objective.record_function_eval = record_function_eval
    objective.record_gradient_eval = record_gradient_eval
    objective.get_cached_real_objective = get_cached_real_objective
    objective.get_penalty = get_penalty
    objective.a_history_path = a_history_path
    objective.grad_history_path = grad_history_path
    objective.compute_gradient_snapshot = compute_gradient_snapshot
    objective.compute_gradient_cs = compute_gradient_cs
    objective.compute_aero_gradients_cs = compute_aero_gradients_cs
    objective.evaluate_cs = _evaluate_objective_cs
    objective.append_grad_history = _append_grad_history
    return objective

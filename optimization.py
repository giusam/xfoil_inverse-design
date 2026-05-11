import csv
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, build_normal_peak_fd_steps, write_dat
from aero_wrapper import run_aero
from cp_utils import split_upper_lower_cp_from_x
from objective import make_objective, total_cp_error
from constraints import build_slsqp_all_constraints
from cmplxfoil_wrapper import clear_cmplxfoil_solver_cache


class OptimizationEarlyStop(Exception):
    pass


class _AdaptiveTriggerMonitor:
    def __init__(self, options):
        self.options = dict(options or {})
        self.mode = str(self.options.get("mode", "slope_efficiency")).strip().lower()
        self.min_major_iters = int(self.options.get("min_major_iters", 5))
        self.eps = float(self.options.get("eps", 1.0e-300))
        self.window = int(self.options.get("window", 3))
        self.slope_rel_tol = float(self.options.get("slope_rel_tol", 0.05))
        self.slope_patience = int(self.options.get("slope_patience", 1))
        self.log_objective = bool(self.options.get("log_objective", True))
        self.stag_tol = float(self.options.get("stag_tol", 1.0e-3))
        self.stag_band = float(self.options.get("stag_band", 0.02))
        self.stag_patience = int(self.options.get("stag_patience", 3))
        self.max_major_iters = self.options.get("max_major_iters", None)

        self.history = []
        self.slope_best_J = None
        self.last_Y = None
        self.improvements = []
        self.max_slope_seen = 0.0
        self.slope_bad_count = 0
        self.stag_best_J = None
        self.stag_counter = 0

    def _valid_for_trigger(self, J):
        J = float(J)
        if not np.isfinite(J):
            return None
        if self.log_objective and J <= 0.0:
            return None
        return J

    def update(self, major_iter, J):
        J = self._valid_for_trigger(J)
        if J is None:
            return None

        if self.slope_best_J is None or J < self.slope_best_J:
            self.slope_best_J = J

        if self.log_objective:
            Y = float(np.log(max(float(self.slope_best_J), self.eps)))
            log_best_J = Y
        else:
            Y = float(self.slope_best_J)
            log_best_J = float(np.log(max(float(self.slope_best_J), self.eps)))

        improvement = float("nan")
        if self.last_Y is not None:
            improvement = float(self.last_Y - Y)
            self.improvements.append(improvement)
        self.last_Y = Y

        recent_slope = float("nan")
        slope_ratio = float("nan")
        slope_triggered = False
        if int(major_iter) >= self.min_major_iters and len(self.improvements) >= self.window:
            recent_slope = float(np.mean(self.improvements[-self.window:]))
            if recent_slope > 0.0:
                self.max_slope_seen = max(self.max_slope_seen, recent_slope)
            slope_ratio = recent_slope / max(self.max_slope_seen, self.eps)
            if slope_ratio < self.slope_rel_tol:
                self.slope_bad_count += 1
            else:
                self.slope_bad_count = 0
            slope_triggered = self.slope_bad_count >= self.slope_patience

        stag_rel_improvement = float("nan")
        stag_gap_to_best = float("nan")
        if self.stag_best_J is None:
            self.stag_best_J = J
        elif J < self.stag_best_J:
            denom = max(abs(float(self.stag_best_J)), self.eps)
            stag_rel_improvement = float((self.stag_best_J - J) / denom)
            self.stag_best_J = J
            if stag_rel_improvement > self.stag_tol:
                self.stag_counter = 0
            else:
                self.stag_counter += 1
        else:
            denom = max(abs(float(self.stag_best_J)), self.eps)
            stag_gap_to_best = float((J - self.stag_best_J) / denom)
            if stag_gap_to_best <= self.stag_band:
                self.stag_counter += 1
            else:
                self.stag_counter = 0

        stagnation_triggered = (
            int(major_iter) >= self.min_major_iters
            and self.stag_counter >= self.stag_patience
        )

        triggered = False
        trigger_reason = ""
        if self.max_major_iters is not None and int(major_iter) >= int(self.max_major_iters):
            triggered = True
            trigger_reason = "max_major_iters"
        else:
            reasons = []
            if self.mode in {"slope_efficiency", "hybrid"} and slope_triggered:
                reasons.append("slope_efficiency")
            if self.mode in {"stagnation", "hybrid"} and stagnation_triggered:
                reasons.append("stagnation")
            if reasons:
                triggered = True
                if self.mode == "hybrid":
                    trigger_reason = "hybrid:" + "+".join(reasons)
                else:
                    trigger_reason = reasons[0]

        row = {
            "major_iter": int(major_iter),
            "J": float(J),
            "best_J": float(self.slope_best_J),
            "log_best_J": float(log_best_J),
            "improvement": improvement,
            "recent_slope": recent_slope,
            "max_slope_seen": float(self.max_slope_seen),
            "slope_ratio": slope_ratio,
            "slope_bad_count": int(self.slope_bad_count),
            "stag_rel_improvement": stag_rel_improvement,
            "stag_gap_to_best": stag_gap_to_best,
            "stag_counter": int(self.stag_counter),
            "slope_triggered": bool(slope_triggered),
            "stagnation_triggered": bool(stagnation_triggered),
            "triggered": bool(triggered),
            "trigger_reason": trigger_reason,
        }
        self.history.append(row)
        return row


def _write_trigger_history_csv(path, rows):
    rows = list(rows or [])
    if not rows:
        return None
    fieldnames = [
        "major_iter",
        "J",
        "best_J",
        "log_best_J",
        "improvement",
        "recent_slope",
        "max_slope_seen",
        "slope_ratio",
        "slope_bad_count",
        "stag_rel_improvement",
        "stag_gap_to_best",
        "stag_counter",
        "slope_triggered",
        "stagnation_triggered",
        "triggered",
        "trigger_reason",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def _status_of_last_eval(objective):
    if not hasattr(objective, "eval_history") or len(objective.eval_history) == 0:
        return None
    return objective.eval_history[-1].get("status")


def _evaluate_with_status(objective, a_vec):
    value = objective(np.asarray(a_vec, dtype=float))
    status = _status_of_last_eval(objective)
    return float(value), status


def _compute_explicit_jac(
    objective,
    a,
    bounds,
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    target_peak_normal,
    record_valid_eval=None,
):
    """
    Explicit finite-difference Jacobian for SLSQP.

    Step logic:
    - geometry-based step from target peak normal deformation
    - clipped to bounds
    - central FD when possible
    - one-sided FD near bounds
    - if XFOIL/geometry fails on required evals, gradient component -> 0
    """
    a = np.asarray(a, dtype=float)
    g = np.zeros_like(a, dtype=float)
    step_vector = build_normal_peak_fd_steps(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        a=a,
        target_peak_normal=target_peak_normal,
        power=SETTINGS["optimization"]["hh_power"],
    )

    for j in range(len(a)):
        aj = float(a[j])
        lo, hi = bounds[j]

        h = float(step_vector[j])

        room_plus = max(0.0, hi - aj)
        room_minus = max(0.0, aj - lo)

        # Prefer central difference if both sides available
        if room_plus >= h and room_minus >= h:
            a_p = a.copy()
            a_m = a.copy()
            a_p[j] += h
            a_m[j] -= h

            Jp, sp = _evaluate_with_status(objective, a_p)
            Jm, sm = _evaluate_with_status(objective, a_m)
            if record_valid_eval is not None:
                record_valid_eval(a_p, Jp, sp)
                record_valid_eval(a_m, Jm, sm)

            if sp == "OK" and sm == "OK":
                g[j] = (Jp - Jm) / (2.0 * h)
            else:
                g[j] = 0.0
            continue

        # Forward one-sided if only plus side available
        if room_plus > 0.0:
            h_fwd = min(h, room_plus)
            a_p = a.copy()
            a_p[j] += h_fwd

            J0, s0 = _evaluate_with_status(objective, a)
            Jp, sp = _evaluate_with_status(objective, a_p)
            if record_valid_eval is not None:
                record_valid_eval(a, J0, s0)
                record_valid_eval(a_p, Jp, sp)

            if s0 == "OK" and sp == "OK" and h_fwd > 0.0:
                g[j] = (Jp - J0) / h_fwd
            else:
                g[j] = 0.0
            continue

        # Backward one-sided if only minus side available
        if room_minus > 0.0:
            h_bwd = min(h, room_minus)
            a_m = a.copy()
            a_m[j] -= h_bwd

            J0, s0 = _evaluate_with_status(objective, a)
            Jm, sm = _evaluate_with_status(objective, a_m)
            if record_valid_eval is not None:
                record_valid_eval(a, J0, s0)
                record_valid_eval(a_m, Jm, sm)

            if s0 == "OK" and sm == "OK" and h_bwd > 0.0:
                g[j] = (J0 - Jm) / h_bwd
            else:
                g[j] = 0.0
            continue

        # No room at all (degenerate bound case)
        g[j] = 0.0

    return g


def optimize_for_centers(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a0,
    label,
    current_best_error,
    workdir,
    trigger_options=None,
):
    hh_power = SETTINGS["optimization"]["hh_power"]
    opt_workdir = Path(workdir) / label
    trigger_enabled = bool(trigger_options and trigger_options.get("enabled", False))
    trigger_monitor = _AdaptiveTriggerMonitor(trigger_options) if trigger_enabled else None
    trigger_major_iter = {"k": 0}
    early_stop_triggered = False
    early_stop_reason = ""
    early_stop_major_iter = ""

    if SETTINGS.get("aero", {}).get("backend", "xfoil").strip().lower() == "cmplxfoil":
        clear_cmplxfoil_solver_cache()

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    objective = make_objective(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        cp_target=cp_target,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / label,
        current_best_error=current_best_error,
    )

    all_constraints = build_slsqp_all_constraints(
        settings_constraints=SETTINGS.get("constraints", {}),
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / f"{label}_constraints",
        record_aero_call=objective.record_aero_call,
    )

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * (len(upper_centers) + len(lower_centers))
    a0 = np.asarray(a0, dtype=float)
    best_valid = {
        "a": None,
        "err": float("inf"),
    }

    def _record_valid_eval(a_vec, objective_value=None, status=None):
        if status is None:
            status = _status_of_last_eval(objective)
        if status != "OK":
            return
        cached = objective.get_cached_real_objective(a_vec)
        if cached is not None:
            value = cached.get("objective_base", cached.get("objective", objective_value))
        else:
            value = objective_value
        try:
            value = float(np.real(value))
        except Exception:
            return
        if not np.isfinite(value):
            return
        if value < float(best_valid["err"]):
            best_valid["err"] = value
            best_valid["a"] = np.asarray(a_vec, dtype=float).copy()

    def _latest_valid_trigger_objective(a_vec):
        cached = objective.get_cached_real_objective(a_vec)
        if cached is not None and cached.get("status") == "OK":
            value = cached.get("objective_base", cached.get("objective"))
            try:
                value = float(np.real(value))
            except Exception:
                value = float("nan")
            if np.isfinite(value) and (not trigger_monitor.log_objective or value > 0.0):
                return value
        return None

    jac_target_peak_normal = SETTINGS["optimization"]["opt_fd_target_peak_normal"]
    deriv_mode = SETTINGS.get("aero", {}).get("derivatives", "fd").strip().lower()
    if deriv_mode not in {"fd", "cs"}:
        raise ValueError(f"Unknown derivative mode: {deriv_mode}")

    def jac_explicit(a_vec):
        return _compute_explicit_jac(
            objective=objective,
            a=a_vec,
            bounds=bounds,
            x=x,
            yu_init=yu_init,
            yl_init=yl_init,
            upper_centers=upper_centers,
            lower_centers=lower_centers,
            target_peak_normal=jac_target_peak_normal,
            record_valid_eval=_record_valid_eval,
        )

    # Gradient snapshots for debug/scaling analysis
    def _save_gradient_snapshot(tag, a_vec):
        if not hasattr(objective, "append_grad_history"):
            return

        print(f"\n>>> Computing gradient snapshot: {tag}")

        if deriv_mode == "cs" and hasattr(objective, "compute_gradient_cs"):
            g_vec = objective.compute_gradient_cs(np.asarray(a_vec, dtype=float))
        elif hasattr(objective, "compute_gradient_snapshot"):
            g_vec = objective.compute_gradient_snapshot(np.asarray(a_vec, dtype=float))
        else:
            return

        objective.append_grad_history(objective.eval_counter["k"], g_vec)

        grad_norm = np.linalg.norm(g_vec)
        grad_max = np.max(np.abs(g_vec)) if len(g_vec) > 0 else 0.0
        print(f">>> Gradient snapshot saved: {tag}")
        print(f">>> ||g||_2 = {grad_norm:.6e}")
        print(f">>> max|g|  = {grad_max:.6e}")

    print(f"\n===== OBJECTIVE CHECK AT a0 ({label}) =====")
    prev_phase = objective.current_phase
    objective.set_eval_phase("setup")
    try:
        j0 = objective(a0)
        _record_valid_eval(a0, j0)
    finally:
        objective.set_eval_phase(prev_phase)
    print(f"Initial objective value at a0 = {j0:.6e}")
    print(f"Failure penalty value         = {objective.get_penalty():.6e}")
    print(f"SLSQP constraints             = {len(all_constraints)}")
    if deriv_mode == "cs":
        print("Jacobian mode                 = COMPLEX_STEP")
    else:
        print("Jacobian mode                 = EXPLICIT_FD")
        print(f"jac_target_peak_normal        = {jac_target_peak_normal:.6e}")
    #_save_gradient_snapshot(f"{label}_initial", a0)

    if deriv_mode == "cs":
        if not hasattr(objective, "compute_gradient_cs"):
            raise RuntimeError("Objective does not expose compute_gradient_cs().")
        jac_fun_raw = lambda a_vec: objective.compute_gradient_cs(np.asarray(a_vec, dtype=float))
    else:
        jac_fun_raw = jac_explicit

    def _with_eval_phase(phase, func):
        def wrapped(a_vec):
            prev_phase = objective.current_phase
            objective.set_eval_phase(phase)
            try:
                return func(a_vec)
            finally:
                objective.set_eval_phase(prev_phase)

        return wrapped

    all_constraints_wrapped = []
    for constraint in all_constraints:
        item = dict(constraint)
        item["fun"] = _with_eval_phase("function", item["fun"])
        if "jac" in item:
            item["jac"] = _with_eval_phase("gradient", item["jac"])
        all_constraints_wrapped.append(item)

    def objective_for_optimizer(a_vec):
        prev_phase = objective.current_phase
        objective.set_eval_phase("function")
        objective.n_function_evals += 1
        try:
            value = objective(np.asarray(a_vec, dtype=float))
            _record_valid_eval(a_vec, value)
            objective.record_function_eval(a_vec)
            return value
        finally:
            objective.set_eval_phase(prev_phase)

    def jacobian_for_optimizer(a_vec):
        prev_phase = objective.current_phase
        objective.set_eval_phase("gradient")
        objective.n_gradient_evals += 1
        try:
            g_vec = jac_fun_raw(np.asarray(a_vec, dtype=float))
            objective.record_gradient_eval(a_vec)
            return g_vec
        finally:
            objective.set_eval_phase(prev_phase)

    def trigger_callback(xk):
        if not trigger_enabled:
            return
        J = _latest_valid_trigger_objective(np.asarray(xk, dtype=float))
        if J is None:
            return
        trigger_major_iter["k"] += 1
        row = trigger_monitor.update(trigger_major_iter["k"], J)
        if row is None or not row.get("triggered", False):
            return
        raise OptimizationEarlyStop(str(row.get("trigger_reason", "trigger")))

    try:
        result = minimize(
            objective_for_optimizer,
            a0,
            method="SLSQP",
            jac=jacobian_for_optimizer,
            bounds=bounds,
            constraints=all_constraints_wrapped,
            callback=trigger_callback if trigger_enabled else None,
            options={
                "maxiter": SETTINGS["optimization"]["maxiter"],
                "ftol": SETTINGS["optimization"]["ftol"],
                "disp": True,
            },
        )
    except OptimizationEarlyStop as exc:
        early_stop_triggered = True
        early_stop_reason = str(exc)
        early_stop_major_iter = int(trigger_major_iter["k"])
        if best_valid["a"] is None:
            best_valid["a"] = a0.copy()
            best_valid["err"] = float(j0)
        result = OptimizeResult(
            x=np.asarray(best_valid["a"], dtype=float).copy(),
            fun=float(best_valid["err"]),
            success=True,
            status=0,
            message=f"Stopped by adaptive trigger: {early_stop_reason}",
            nit=int(early_stop_major_iter),
        )

    print(f"\n===== OPTIMIZATION RESULT ({label}) =====")
    print("success :", result.success)
    print("status  :", result.status)
    print("message :", result.message)
    print("fun     :", result.fun)
    print("nit     :", result.nit)
    if early_stop_triggered:
        print("\n===== ADAPTIVE LEVEL EARLY STOP =====")
        print(f"label = {trigger_options.get('level_label', label) if trigger_options else label}")
        print(f"ndv = {len(upper_centers) + len(lower_centers)}")
        print(f"reason = {early_stop_reason}")
        print(f"major_iter = {early_stop_major_iter}")
        print(f"best_err = {float(best_valid['err']):.6e}")

    a_opt = np.asarray(result.x, dtype=float)
    #_save_gradient_snapshot(f"{label}_final", a_opt)

    nu = len(upper_centers)
    nl = len(lower_centers)
    a_upper = np.asarray(a_opt[:nu], dtype=float)
    a_lower = np.asarray(a_opt[nu:nu + nl], dtype=float)

    yu_opt, yl_opt = apply_hicks_henne_deformation(
        x=x,
        yu_base=yu_init,
        yl_base=yl_init,
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        power=hh_power,
    )

    opt_dat = Path(workdir) / f"{label}_optimized_airfoil.dat"
    write_dat(opt_dat, x, yu_opt, yl_opt, name=f"{label.upper()}_OPTIMIZED")

    n_postprocess_aero_calls = 1
    opt_res = run_aero(
        airfoil_dat=opt_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=Path(workdir) / f"{label}_optimized_run",
    )

    if not opt_res["success"]:
        if early_stop_triggered:
            print("[warning] optimized aero postprocess failed after trigger; trying cached-state evaluator.")
            state = objective.evaluate_cp_state(a_opt)
            if state.get("status") != "OK":
                print(opt_res["stdout"])
                print(opt_res["stderr"])
                raise RuntimeError(f"Early-stopped best-state aero run failed for {label}.")
            opt_res = {
                "success": True,
                "polar": state["polar"],
                "cp_data": {
                    "x": np.concatenate([
                        np.asarray(state["cp_candidate"]["upper"]["x"], dtype=float),
                        np.asarray(state["cp_candidate"]["lower"]["x"], dtype=float),
                    ]),
                    "cp": np.concatenate([
                        np.asarray(state["cp_candidate"]["upper"]["cp"], dtype=float),
                        np.asarray(state["cp_candidate"]["lower"]["cp"], dtype=float),
                    ]),
                },
                "stdout": "",
                "stderr": "",
            }
            cp_opt = state["cp_candidate"]
        else:
            print(opt_res["stdout"])
            print(opt_res["stderr"])
            raise RuntimeError(f"Optimized XFOIL run failed for {label}.")
    else:
        cp_opt = split_upper_lower_cp_from_x(
            opt_res["cp_data"]["x"],
            opt_res["cp_data"]["cp"],
        )

    err_opt = total_cp_error(cp_target, cp_opt)

    n_objective_evals = objective.eval_counter["k"]
    n_optimization_function_aero_calls = int(objective.aero_call_counter_by_phase.get("function", 0))
    n_optimization_gradient_aero_calls = int(objective.aero_call_counter_by_phase.get("gradient", 0))
    n_optimization_aero_calls_total = (
        n_optimization_function_aero_calls + n_optimization_gradient_aero_calls
    )
    n_setup_aero_calls = int(objective.aero_call_counter_by_phase.get("setup", 0))
    n_xfoil_calls_total = int(objective.aero_call_counter_total + n_postprocess_aero_calls)

    trigger_history = list(trigger_monitor.history) if trigger_monitor is not None else []
    if trigger_enabled and bool(trigger_options.get("write_csv", True)) and trigger_history:
        try:
            csv_path = _write_trigger_history_csv(
                opt_workdir / "trigger_history.csv",
                trigger_history,
            )
            print(f"Adaptive trigger history CSV written: {csv_path}")

            workdir_path = Path(workdir)

            persistent_path = (
                workdir_path.parent.parent
                / "summary"
                / "trigger_history"
                / workdir_path.parent.name
                / workdir_path.name
                / str(label)
                / "trigger_history.csv"
            )

            persistent_csv = _write_trigger_history_csv(
                persistent_path,
                trigger_history,
            )
            print(f"Persistent adaptive trigger history CSV written: {persistent_csv}")

        except Exception as exc:
            print(f"[warning] could not write trigger_history.csv for {label}: {exc}")
    if SETTINGS.get("aero", {}).get("backend", "xfoil").strip().lower() == "cmplxfoil":
        clear_cmplxfoil_solver_cache()

    return {
        "upper_centers": np.asarray(upper_centers, dtype=float),
        "lower_centers": np.asarray(lower_centers, dtype=float),
        "a_opt": a_opt,
        "yu_opt": yu_opt,
        "yl_opt": yl_opt,
        "cp_opt": cp_opt,
        "opt_res": opt_res,
        "err_opt": err_opt,
        "result": result,
        "n_objective_evals": n_objective_evals,
        "n_xfoil_calls_total": n_xfoil_calls_total,
        "n_aero_calls_total_all_phases": n_xfoil_calls_total,
        "n_setup_aero_calls": n_setup_aero_calls,
        "n_postprocess_aero_calls": n_postprocess_aero_calls,
        "n_optimization_aero_calls_total": n_optimization_aero_calls_total,
        "n_optimization_function_aero_calls": n_optimization_function_aero_calls,
        "n_optimization_gradient_aero_calls": n_optimization_gradient_aero_calls,
        "n_function_evals": int(objective.n_function_evals),
        "n_gradient_evals": int(objective.n_gradient_evals),
        "objective_history": list(objective.eval_history),
        "function_eval_history": [dict(item) for item in objective.function_eval_history],
        "gradient_eval_history": [dict(item) for item in objective.gradient_eval_history],
        "aero_call_counter_by_phase": dict(objective.aero_call_counter_by_phase),
        "ndv_total": len(upper_centers) + len(lower_centers),
        "penalty_value": objective.get_penalty(),
        "n_slsqp_constraints": len(all_constraints),
        "n_geometric_constraints": len(all_constraints),
        "early_stop_triggered": bool(early_stop_triggered),
        "early_stop_reason": str(early_stop_reason),
        "early_stop_level": str(trigger_options.get("level_label", label) if trigger_options else ""),
        "early_stop_major_iter": early_stop_major_iter,
        "early_stop_best_err": float(best_valid["err"]) if early_stop_triggered else float("nan"),
        "trigger_history": trigger_history,
    }

from pathlib import Path

import numpy as np

AERODYNAMIC_CONSTRAINT_NAMES = {"CL", "CD", "CM"}
GEOMETRIC_CONSTRAINT_NAMES = {"tmax", "area", "thickness_stations"}


def compute_violation(value, kind, target, tol=0.0):
    value = float(value)
    target = float(target)
    tol = float(tol)

    if kind == "ge":
        return max(0.0, target - value)

    if kind == "le":
        return max(0.0, value - target)

    if kind == "eq":
        return max(0.0, abs(value - target) - tol)

    raise ValueError(f"Unknown constraint kind: {kind}")


def signed_constraint_value(value, kind, target):
    """
    Restituisce il vincolo in forma standard per IKKT:

        C(a) <= 0

    Convenzioni:
    - eq : C(a) = value - target
    - le : C(a) = value - target
    - ge : C(a) = target - value
    """
    value = float(value)
    target = float(target)

    if kind == "eq":
        return value - target

    if kind == "le":
        return value - target

    if kind == "ge":
        return target - value

    raise ValueError(f"Unknown constraint kind: {kind}")


def is_constraint_active(c_value, kind, tol_active=0.0):
    """
    Stabilisce se un vincolo e' attivo nel senso IKKT.

    - equality: sempre attivo
    - inequality: attivo se vicino al bordo o violato
                  cioe' se C(a) > -tol_active
    """
    c_value = float(c_value)
    tol_active = float(tol_active)

    if kind == "eq":
        return True

    if kind in ("le", "ge"):
        return c_value > -tol_active

    raise ValueError(f"Unknown constraint kind: {kind}")


def compute_max_thickness(yu, yl):
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    return float(np.max(yu - yl))


def compute_area(x, yu, yl):
    x = np.asarray(x, dtype=float)
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    thickness = yu - yl
    return float(np.trapezoid(thickness, x))


def thickness_at_x(x, yu, yl, x_station):
    x = np.asarray(x, dtype=float)
    yu = np.asarray(yu, dtype=float)
    yl = np.asarray(yl, dtype=float)
    x_station = float(x_station)

    yu_i = np.interp(x_station, x, yu)
    yl_i = np.interp(x_station, x, yl)
    return float(yu_i - yl_i)


def compute_metrics(x, yu, yl, polar):
    return {
        "CL": float(polar["CL"]),
        "CD": float(polar["CD"]),
        "CM": float(polar["CM"]),
        "tmax": compute_max_thickness(yu, yl),
        "area": compute_area(x, yu, yl),
    }


def split_constraints_by_domain(settings_constraints):
    aero = {}
    geom = {}

    for name, spec in settings_constraints.items():
        if name in AERODYNAMIC_CONSTRAINT_NAMES:
            aero[name] = spec
        elif name in GEOMETRIC_CONSTRAINT_NAMES:
            geom[name] = spec
        else:
            aero[name] = spec

    return aero, geom


def get_constraint_value(name, spec, metrics, x, yu, yl, station=None):
    """
    Restituisce il valore grezzo della metrica associata al vincolo.
    Serve come base comune sia per la penalty sia per IKKT.
    """
    if name == "thickness_stations":
        if station is None:
            raise ValueError("For 'thickness_stations' a station spec must be provided.")
        return thickness_at_x(x, yu, yl, float(station["x"]))

    if name not in metrics:
        raise KeyError(f"Constraint metric '{name}' not found in metrics.")

    return float(metrics[name])


def build_signed_constraint_entry(name, spec, metrics, x, yu, yl, tol_active=0.0):
    kind = spec["kind"]
    target = float(spec["target"])
    value = get_constraint_value(name, spec, metrics, x, yu, yl)
    c_value = signed_constraint_value(value, kind, target)
    active = is_constraint_active(c_value, kind, tol_active=tol_active)

    return {
        "name": name,
        "kind": kind,
        "target": target,
        "value": value,
        "c_value": c_value,
        "active": active,
    }


def build_all_signed_constraints(settings_constraints, metrics, x, yu, yl, tol_active=0.0):
    entries = []

    for name, spec in settings_constraints.items():
        if not spec.get("enabled", False):
            continue

        if name == "thickness_stations":
            kind = spec["kind"]
            for i, st in enumerate(spec.get("stations", [])):
                station_target = float(st["target"])
                value = thickness_at_x(x, yu, yl, float(st["x"]))
                c_value = signed_constraint_value(value, kind, station_target)
                active = is_constraint_active(c_value, kind, tol_active=tol_active)

                entries.append(
                    {
                        "name": name,
                        "station_id": i,
                        "x": float(st["x"]),
                        "kind": kind,
                        "target": station_target,
                        "value": value,
                        "c_value": c_value,
                        "active": active,
                        "alpha": float(st.get("alpha", 1.0)),
                        "scale": float(st.get("scale", 1.0e-2)),
                    }
                )
            continue

        entry = build_signed_constraint_entry(
            name=name,
            spec=spec,
            metrics=metrics,
            x=x,
            yu=yu,
            yl=yl,
            tol_active=tol_active,
        )
        entry["tol"] = float(spec.get("tol", 0.0))
        entry["alpha"] = float(spec.get("alpha", 1.0))
        entry["scale"] = float(spec.get("scale", 1.0e-2))
        entries.append(entry)

    return entries


def evaluate_constraints(settings_constraints, metrics, x, yu, yl, current_best_error):
    penalty_total = 0.0
    details = {}

    scale_obj = max(float(current_best_error), 1.0e-6)

    for name, spec in settings_constraints.items():
        if not spec.get("enabled", False):
            continue

        if name == "thickness_stations":
            kind = spec["kind"]
            station_details = []

            for st in spec.get("stations", []):
                xs = float(st["x"])
                target = float(st["target"])
                alpha = float(st.get("alpha", 1.0))
                scale = float(st.get("scale", 1.0e-2))

                value = thickness_at_x(x, yu, yl, xs)
                violation = compute_violation(value, kind, target)
                penalty = alpha * scale_obj * (violation / scale) ** 2

                penalty_total += penalty
                station_details.append(
                    {
                        "x": xs,
                        "value": value,
                        "target": target,
                        "kind": kind,
                        "alpha": alpha,
                        "scale": scale,
                        "violation": violation,
                        "penalty": penalty,
                        "c_value": signed_constraint_value(value, kind, target),
                    }
                )

            details[name] = station_details
            continue

        if name not in metrics:
            raise KeyError(f"Constraint metric '{name}' not found in metrics.")

        kind = spec["kind"]
        target = float(spec["target"])
        alpha = float(spec.get("alpha", 1.0))
        scale = float(spec.get("scale", 1.0e-2))
        tol = float(spec.get("tol", 0.0))

        value = float(metrics[name])
        violation = compute_violation(value, kind, target, tol=tol)
        penalty = alpha * scale_obj * (violation / scale) ** 2

        penalty_total += penalty
        details[name] = {
            "value": value,
            "target": target,
            "kind": kind,
            "tol": tol,
            "alpha": alpha,
            "scale": scale,
            "violation": violation,
            "penalty": penalty,
            "c_value": signed_constraint_value(value, kind, target),
        }

    return penalty_total, details


def _append_slsqp_constraint(constraints, value_fun, kind, target, tol=0.0, jac_fun=None):
    target = float(target)
    tol = float(tol)

    def _ineq_ge(a, vf=value_fun, t=target):
        value = vf(a)
        if value is None:
            return -1.0e9
        return float(value) - t

    def _ineq_le(a, vf=value_fun, t=target):
        value = vf(a)
        if value is None:
            return -1.0e9
        return t - float(value)

    def _eq_fun(a, vf=value_fun, t=target):
        value = vf(a)
        if value is None:
            return 1.0e9
        return float(value) - t

    def _jac_pos(a, jf=jac_fun):
        return np.asarray(jf(a), dtype=float)

    def _jac_neg(a, jf=jac_fun):
        return -np.asarray(jf(a), dtype=float)

    if kind == "ge":
        entry = {"type": "ineq", "fun": _ineq_ge}
        if jac_fun is not None:
            entry["jac"] = _jac_pos
        constraints.append(entry)
        return

    if kind == "le":
        entry = {"type": "ineq", "fun": _ineq_le}
        if jac_fun is not None:
            entry["jac"] = _jac_neg
        constraints.append(entry)
        return

    if kind == "eq":
        if tol > 0.0:
            def _eq_lower(a, vf=value_fun, t=target, to=tol):
                value = vf(a)
                if value is None:
                    return -1.0e9
                return float(value) - (t - to)

            def _eq_upper(a, vf=value_fun, t=target, to=tol):
                value = vf(a)
                if value is None:
                    return -1.0e9
                return (t + to) - float(value)

            entry_lower = {"type": "ineq", "fun": _eq_lower}
            entry_upper = {"type": "ineq", "fun": _eq_upper}

            if jac_fun is not None:
                entry_lower["jac"] = _jac_pos
                entry_upper["jac"] = _jac_neg

            constraints.append(entry_lower)
            constraints.append(entry_upper)
        else:
            entry = {"type": "eq", "fun": _eq_fun}
            if jac_fun is not None:
                entry["jac"] = _jac_pos
            constraints.append(entry)
        return

    raise ValueError(f"Unknown constraint kind: {kind}")


def _build_geometry_from_a_factory(x, yu_init, yl_init, upper_centers, lower_centers, hh_power):
    from geometry import apply_hicks_henne_deformation

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    def _build_geometry_from_a(a):
        nu = len(upper_centers)
        nl = len(lower_centers)
        a_upper = np.asarray(a[:nu], dtype=float)
        a_lower = np.asarray(a[nu:nu + nl], dtype=float)
        return apply_hicks_henne_deformation(
            x=x,
            yu_base=yu_init,
            yl_base=yl_init,
            a_upper=a_upper,
            a_lower=a_lower,
            upper_centers=upper_centers,
            lower_centers=lower_centers,
            power=hh_power,
        )

    return _build_geometry_from_a


def _adaptive_fd_step(a_value, rel_step, abs_step_floor):
    return max(float(rel_step) * abs(float(a_value)), float(abs_step_floor))


def _fd_scalar_gradient(value_fun, a, bounds, rel_step, abs_step_floor):
    a = np.asarray(a, dtype=float)
    g = np.zeros_like(a, dtype=float)

    f0 = value_fun(a)
    if f0 is None:
        return g

    f0 = float(f0)

    for j in range(len(a)):
        aj = float(a[j])
        lo, hi = bounds[j]

        h = _adaptive_fd_step(aj, rel_step, abs_step_floor)

        room_plus = max(0.0, hi - aj)
        room_minus = max(0.0, aj - lo)

        if room_plus >= h and room_minus >= h:
            a_p = a.copy()
            a_m = a.copy()
            a_p[j] += h
            a_m[j] -= h

            fp = value_fun(a_p)
            fm = value_fun(a_m)

            if fp is not None and fm is not None:
                g[j] = (float(fp) - float(fm)) / (2.0 * h)
            else:
                g[j] = 0.0
            continue

        if room_plus > 0.0:
            h_fwd = min(h, room_plus)
            a_p = a.copy()
            a_p[j] += h_fwd

            fp = value_fun(a_p)

            if fp is not None and h_fwd > 0.0:
                g[j] = (float(fp) - f0) / h_fwd
            else:
                g[j] = 0.0
            continue

        if room_minus > 0.0:
            h_bwd = min(h, room_minus)
            a_m = a.copy()
            a_m[j] -= h_bwd

            fm = value_fun(a_m)

            if fm is not None and h_bwd > 0.0:
                g[j] = (f0 - float(fm)) / h_bwd
            else:
                g[j] = 0.0
            continue

        g[j] = 0.0

    return g


def _build_area_gradient(x, upper_centers, lower_centers, hh_power):
    from geometry import build_hicks_henne_basis

    if len(upper_centers) > 0:
        basis_upper = build_hicks_henne_basis(x, upper_centers, power=hh_power)
        grad_upper = np.array([np.trapezoid(phi, x) for phi in basis_upper], dtype=float)
    else:
        grad_upper = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        basis_lower = build_hicks_henne_basis(x, lower_centers, power=hh_power)
        grad_lower = -np.array([np.trapezoid(phi, x) for phi in basis_lower], dtype=float)
    else:
        grad_lower = np.zeros(0, dtype=float)

    return np.concatenate([grad_upper, grad_lower])


def _build_thickness_station_gradient(x_station, upper_centers, lower_centers, hh_power):
    from geometry import build_hicks_henne_basis

    x_eval = np.array([float(x_station)], dtype=float)

    if len(upper_centers) > 0:
        phi_u = build_hicks_henne_basis(x_eval, upper_centers, power=hh_power)[:, 0]
    else:
        phi_u = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        phi_l = -build_hicks_henne_basis(x_eval, lower_centers, power=hh_power)[:, 0]
    else:
        phi_l = np.zeros(0, dtype=float)

    return np.concatenate([phi_u, phi_l])


def _build_geometric_constraint_functions(
    settings_constraints,
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    hh_power,
):
    constraints = []

    _build_geometry_from_a = _build_geometry_from_a_factory(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

    from settings import SETTINGS

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * (len(upper_centers) + len(lower_centers))
    rel_step = SETTINGS["optimization"]["fd_rel_step"]
    abs_step_floor = SETTINGS["optimization"]["fd_abs_step_floor"]

    area_grad = _build_area_gradient(
        x=x,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

    for name, spec in settings_constraints.items():
        if not spec.get("enabled", False):
            continue

        if name not in GEOMETRIC_CONSTRAINT_NAMES:
            continue

        kind = spec["kind"]
        tol = float(spec.get("tol", 0.0))

        if name == "thickness_stations":
            for st in spec.get("stations", []):
                target = float(st["target"])
                xs = float(st["x"])
                st_tol = float(st.get("tol", tol))

                station_grad = _build_thickness_station_gradient(
                    x_station=xs,
                    upper_centers=upper_centers,
                    lower_centers=lower_centers,
                    hh_power=hh_power,
                )

                def value_fun(a, xs=xs):
                    yu, yl = _build_geometry_from_a(a)
                    return thickness_at_x(x, yu, yl, xs)

                def jac_fun(a, g=station_grad):
                    return np.asarray(g, dtype=float)

                _append_slsqp_constraint(
                    constraints=constraints,
                    value_fun=value_fun,
                    kind=kind,
                    target=target,
                    tol=st_tol,
                    jac_fun=jac_fun,
                )
            continue

        target = float(spec["target"])

        def value_fun(a, name=name):
            yu, yl = _build_geometry_from_a(a)

            if name == "area":
                return compute_area(x, yu, yl)
            if name == "tmax":
                return compute_max_thickness(yu, yl)

            raise ValueError(f"Unknown geometric constraint name: {name}")

        if name == "area":
            def jac_fun(a, g=area_grad):
                return np.asarray(g, dtype=float)
        else:
            def jac_fun(a, vf=value_fun, bnds=bounds, rs=rel_step, af=abs_step_floor):
                return _fd_scalar_gradient(vf, a, bnds, rs, af)

        _append_slsqp_constraint(
            constraints=constraints,
            value_fun=value_fun,
            kind=kind,
            target=target,
            tol=tol,
            jac_fun=jac_fun,
        )

    return constraints


def build_slsqp_geometric_constraints(
    settings_constraints,
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    hh_power,
):
    return _build_geometric_constraint_functions(
        settings_constraints=settings_constraints,
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )


def build_slsqp_all_constraints(
    settings_constraints,
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    hh_power,
    alpha_deg,
    reynolds,
    xfoil_iter,
    timeout,
    working_dir,
):
    from geometry import write_dat
    from xfoil_wrapper import run_xfoil
    from settings import SETTINGS

    constraints = _build_geometric_constraint_functions(
        settings_constraints=settings_constraints,
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

    _build_geometry_from_a = _build_geometry_from_a_factory(
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    bmin, bmax = SETTINGS["optimization"]["bounds"]
    bounds = [(bmin, bmax)] * (len(upper_centers) + len(lower_centers))
    rel_step = SETTINGS["optimization"]["fd_rel_step"]
    abs_step_floor = SETTINGS["optimization"]["fd_abs_step_floor"]

    eval_counter = {"k": 0}
    metrics_cache = {}
    grad_cache = {}

    def _evaluate_aero_metrics(a):
        key = tuple(np.round(np.asarray(a, dtype=float), 12))

        if key in metrics_cache:
            return metrics_cache[key]

        eval_counter["k"] += 1
        k = eval_counter["k"]

        yu, yl = _build_geometry_from_a(a)
        airfoil_dat = working_dir / f"constraint_airfoil_{k:05d}.dat"
        run_dir = working_dir / f"constraint_run_{k:05d}"

        write_dat(airfoil_dat, x, yu, yl, name=f"CONSTRAINT_{k:05d}")

        res = run_xfoil(
            airfoil_dat=airfoil_dat,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
            xfoil_iter=xfoil_iter,
            timeout=timeout,
            working_dir=run_dir,
        )

        if not res["success"] or res["polar"] is None:
            metrics_cache[key] = None
            return None

        metrics = compute_metrics(x, yu, yl, res["polar"])
        metrics_cache[key] = metrics
        return metrics

    def _evaluate_aero_metric_gradient(metric_name, a):
        key = (metric_name, tuple(np.round(np.asarray(a, dtype=float), 12)))

        if key in grad_cache:
            return grad_cache[key]

        def scalar_value(z):
            metrics = _evaluate_aero_metrics(z)
            if metrics is None:
                return None
            return float(metrics[metric_name])

        grad = _fd_scalar_gradient(
            value_fun=scalar_value,
            a=a,
            bounds=bounds,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
        )

        grad_cache[key] = grad
        return grad

    for name, spec in settings_constraints.items():
        if not spec.get("enabled", False):
            continue

        if name not in AERODYNAMIC_CONSTRAINT_NAMES:
            continue

        kind = spec["kind"]
        target = float(spec["target"])
        tol = float(spec.get("tol", 0.0))

        def value_fun(a, metric_name=name):
            metrics = _evaluate_aero_metrics(a)
            if metrics is None:
                return None
            return float(metrics[metric_name])

        def jac_fun(a, metric_name=name):
            return _evaluate_aero_metric_gradient(metric_name, a)

        _append_slsqp_constraint(
            constraints=constraints,
            value_fun=value_fun,
            kind=kind,
            target=target,
            tol=tol,
            jac_fun=jac_fun,
        )

    return constraints

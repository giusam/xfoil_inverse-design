import numpy as np
from scipy.optimize import minimize

from settings import SETTINGS
from geometry import build_hicks_henne_basis
from constraints import build_all_signed_constraints


IKKT_EXCLUDED_NAMES = {"tmax"}


def _compute_geometric_gradients_analytic(x, upper_centers, lower_centers):
    grads = {}

    if len(upper_centers) > 0:
        basis_upper = build_hicks_henne_basis(
            x, upper_centers, power=SETTINGS["optimization"]["hh_power"]
        )
        area_upper = np.array([np.trapezoid(phi, x) for phi in basis_upper], dtype=float)
    else:
        area_upper = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        basis_lower = build_hicks_henne_basis(
            x, lower_centers, power=SETTINGS["optimization"]["hh_power"]
        )
        area_lower = -np.array([np.trapezoid(phi, x) for phi in basis_lower], dtype=float)
    else:
        area_lower = np.zeros(0, dtype=float)

    grads["area"] = np.concatenate([area_upper, area_lower])
    return grads


def _compute_thickness_station_gradient(x, upper_centers, lower_centers, x_station):
    x_station = float(x_station)

    if len(upper_centers) > 0:
        phi_u = build_hicks_henne_basis(
            np.array([x_station], dtype=float),
            upper_centers,
            power=SETTINGS["optimization"]["hh_power"],
        )[:, 0]
    else:
        phi_u = np.zeros(0, dtype=float)

    if len(lower_centers) > 0:
        phi_l = -build_hicks_henne_basis(
            np.array([x_station], dtype=float),
            lower_centers,
            power=SETTINGS["optimization"]["hh_power"],
        )[:, 0]
    else:
        phi_l = np.zeros(0, dtype=float)

    return np.concatenate([phi_u, phi_l])


def _constraint_gradient_from_metric_grad(metric_grad, kind):
    if kind in ("eq", "le"):
        return np.asarray(metric_grad, dtype=float)
    if kind == "ge":
        return -np.asarray(metric_grad, dtype=float)
    raise ValueError(f"Unknown constraint kind: {kind}")


def _build_active_ikkt_system(
    x,
    yu,
    yl,
    metrics,
    upper_centers,
    lower_centers,
    grad_metrics_aero,
    grad_geom_base,
):
    tol_active = float(SETTINGS["optimization"]["adaptive"].get("tol_active", 0.0))
    all_entries = build_all_signed_constraints(
        SETTINGS.get("constraints", {}),
        metrics,
        x,
        yu,
        yl,
        tol_active=tol_active,
    )

    active_entries = []
    grad_columns = []

    for entry in all_entries:
        if entry["name"] in IKKT_EXCLUDED_NAMES:
            continue
        if not entry.get("active", False):
            continue

        name = entry["name"]
        kind = entry["kind"]

        if name in ("CL", "CD", "CM"):
            metric_grad = grad_metrics_aero.get(name)
            if metric_grad is None:
                continue
            c_grad = _constraint_gradient_from_metric_grad(metric_grad, kind)
        elif name == "area":
            c_grad = _constraint_gradient_from_metric_grad(grad_geom_base["area"], kind)
        elif name == "thickness_stations":
            xs = float(entry["x"])
            t_grad = _compute_thickness_station_gradient(x, upper_centers, lower_centers, xs)
            c_grad = _constraint_gradient_from_metric_grad(t_grad, kind)
        else:
            continue

        active_entries.append(entry)
        grad_columns.append(np.asarray(c_grad, dtype=float))

    if len(grad_columns) == 0:
        return active_entries, None

    return active_entries, np.column_stack(grad_columns)


def _solve_bounded_least_squares(grad_j, G, active_entries):
    m = G.shape[1]
    bounds = []

    for entry in active_entries:
        if entry["kind"] == "eq":
            bounds.append((None, None))
        else:
            bounds.append((0.0, None))

    def obj(lam):
        r = grad_j - G @ lam
        return 0.5 * float(np.dot(r, r))

    x0 = np.zeros(m, dtype=float)
    result = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

    if not result.success:
        return np.zeros(m, dtype=float)
    return np.asarray(result.x, dtype=float)
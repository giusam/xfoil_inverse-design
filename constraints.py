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
            # fallback conservativo: tutto cio' che non e' noto resta nella penalty
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
    """
    Costruisce una rappresentazione standardizzata di un singolo vincolo
    da usare nella logica IKKT.

    Ritorna un dizionario con:
    - name
    - kind
    - target
    - value
    - c_value   (signed, in forma C(a) <= 0)
    - active
    """
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
    """
    Costruisce tutti i vincoli abilitati in forma standardizzata per IKKT.

    Restituisce una lista di dizionari.
    Per thickness_stations crea una entry separata per ogni stazione.
    """
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
    """
    Valuta i vincoli attivi e costruisce la penalty soft totale.

    La penalty di ciascun vincolo e':
        alpha * scale_obj * (violation / scale)^2

    dove:
        scale_obj = max(current_best_error, 1e-6)
    """
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


def _build_geometric_constraint_functions(settings_constraints, x, yu_init, yl_init, upper_centers, lower_centers, hh_power):
    from geometry import apply_hicks_henne_deformation

    constraints = []

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

    for name, spec in settings_constraints.items():
        if not spec.get("enabled", False):
            continue

        if name not in GEOMETRIC_CONSTRAINT_NAMES:
            continue

        kind = spec["kind"]

        if name == "thickness_stations":
            for st in spec.get("stations", []):
                target = float(st["target"])
                xs = float(st["x"])

                def fun(a, xs=xs, target=target, kind=kind):
                    yu, yl = _build_geometry_from_a(a)
                    value = thickness_at_x(x, yu, yl, xs)
                    if kind == "eq":
                        return value - target
                    if kind == "ge":
                        return value - target
                    if kind == "le":
                        return target - value
                    raise ValueError(f"Unknown constraint kind: {kind}")

                ctype = "eq" if kind == "eq" else "ineq"
                constraints.append({"type": ctype, "fun": fun})
            continue

        target = float(spec["target"])

        def fun(a, name=name, target=target, kind=kind):
            yu, yl = _build_geometry_from_a(a)

            if name == "area":
                value = compute_area(x, yu, yl)
            elif name == "tmax":
                value = compute_max_thickness(yu, yl)
            else:
                raise ValueError(f"Unknown geometric constraint name: {name}")

            if kind == "eq":
                return value - target
            if kind == "ge":
                return value - target
            if kind == "le":
                return target - value
            raise ValueError(f"Unknown constraint kind: {kind}")

        ctype = "eq" if kind == "eq" else "ineq"
        constraints.append({"type": ctype, "fun": fun})

    return constraints


def build_slsqp_geometric_constraints(settings_constraints, x, yu_init, yl_init, upper_centers, lower_centers, hh_power):
    """
    Costruisce la lista di vincoli geometrici nel formato richiesto da SciPy SLSQP.

    Convenzione SciPy:
    - 'eq'  : fun(a) = 0
    - 'ineq': fun(a) >= 0

    Per questo motivo:
    - ge  -> value - target >= 0
    - le  -> target - value >= 0
    - eq  -> value - target = 0
    """
    return _build_geometric_constraint_functions(
        settings_constraints=settings_constraints,
        x=x,
        yu_init=yu_init,
        yl_init=yl_init,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        hh_power=hh_power,
    )

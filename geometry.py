import math
import numpy as np
from pathlib import Path


def cosine_spacing(n):
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def naca0012_thickness(x, t=0.12):
    x = np.asarray(x, dtype=float)
    return 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 1e-12, 1.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def build_naca0012_surfaces(n_points, thickness):
    x = cosine_spacing(n_points)
    yt = naca0012_thickness(x, thickness)

    yu = yt
    yl = -yt

    return x, yu, yl


def write_dat(path, x, yu, yl, name="AIRFOIL"):
    path = Path(path)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")

        for xi, yi in zip(x[::-1], yu[::-1]):
            f.write(f"{xi:.10f} {yi:.10f}\n")

        for xi, yi in zip(x[1:], yl[1:]):
            f.write(f"{xi:.10f} {yi:.10f}\n")


def kulfan_class_function(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.clip(x, 0.0, 1.0)) * (1.0 - x)


def bernstein_basis(n, k, x):
    x = np.asarray(x, dtype=float)
    coeff = math.comb(n, k)
    return coeff * (x ** k) * ((1.0 - x) ** (n - k))


def bernstein_deformation(x, weights):
    x = np.asarray(x, dtype=float)
    n = len(weights) - 1

    s = np.zeros_like(x)
    for k, wk in enumerate(weights):
        s += wk * bernstein_basis(n, k, x)

    return kulfan_class_function(x) * s


def build_random_initial_geometry(x, yu_target, yl_target, seed=0, amp=0.03, order=15):
    rng = np.random.default_rng(seed)

    weights_u = rng.uniform(-amp, amp, order + 1)
    weights_l = rng.uniform(-amp, amp, order + 1)

    # Preserve leading-edge radius:
    # the coefficient of sqrt(x) must not change
    weights_u[0] = 0.0
    weights_l[0] = 0.0

    dy_u = bernstein_deformation(x, weights_u)
    dy_l = bernstein_deformation(x, weights_l)

    yu_init = yu_target + dy_u
    yl_init = yl_target + dy_l

    return {
        "yu_init": yu_init,
        "yl_init": yl_init,
        "weights_u": weights_u,
        "weights_l": weights_l,
        "dy_u": dy_u,
        "dy_l": dy_l,
    }


def hicks_henne_bump(x, x0, power=4):
    x = np.asarray(x, dtype=float)

    eps = 1.0e-12
    x = np.clip(x, eps, 1.0 - eps)
    x0 = float(np.clip(x0, eps, 1.0 - eps))

    exponent = math.log(0.5) / math.log(x0)
    return np.sin(np.pi * x**exponent) ** power


def build_hicks_henne_basis(x, centers, power=4):
    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)

    basis = []
    for c in centers:
        basis.append(hicks_henne_bump(x, c, power=power))

    return np.array(basis)


def apply_hicks_henne_deformation(
    x,
    yu_base,
    yl_base,
    a_upper,
    a_lower,
    upper_centers,
    lower_centers,
    power=4,
):
    x = np.asarray(x, dtype=float)
    yu_base = np.asarray(yu_base, dtype=float)
    yl_base = np.asarray(yl_base, dtype=float)

    a_upper = np.asarray(a_upper)
    a_lower = np.asarray(a_lower)

    upper_centers = np.asarray(upper_centers, dtype=float)
    lower_centers = np.asarray(lower_centers, dtype=float)

    if len(a_upper) != len(upper_centers):
        raise ValueError("Expected len(a_upper) = len(upper_centers)")

    if len(a_lower) != len(lower_centers):
        raise ValueError("Expected len(a_lower) = len(lower_centers)")

    if len(upper_centers) > 0:
        basis_upper = build_hicks_henne_basis(x, upper_centers, power=power)
        dy_upper = np.sum(a_upper[:, None] * basis_upper, axis=0)
    else:
        dy_upper = np.zeros_like(x)

    if len(lower_centers) > 0:
        basis_lower = build_hicks_henne_basis(x, lower_centers, power=power)
        dy_lower = np.sum(a_lower[:, None] * basis_lower, axis=0)
    else:
        dy_lower = np.zeros_like(x)

    yu = yu_base + dy_upper
    yl = yl_base + dy_lower

    return yu, yl


def build_normal_peak_fd_steps(
    x,
    yu_init,
    yl_init,
    upper_centers,
    lower_centers,
    a,
    target_peak_normal,
    power=4,
    eps=1.0e-14,
):
    x = np.asarray(x, dtype=float)
    yu_init = np.asarray(yu_init, dtype=float)
    yl_init = np.asarray(yl_init, dtype=float)
    a = np.asarray(a, dtype=float)

    upper_centers = list(upper_centers)
    lower_centers = list(lower_centers)

    nu = len(upper_centers)
    nl = len(lower_centers)

    if len(a) != nu + nl:
        raise ValueError("Expected len(a) = len(upper_centers) + len(lower_centers)")

    a_upper = a[:nu]
    a_lower = a[nu:nu + nl]

    yu, yl = apply_hicks_henne_deformation(
        x=x,
        yu_base=yu_init,
        yl_base=yl_init,
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        power=power,
    )

    dyu_dx = np.gradient(yu, x)
    dyl_dx = np.gradient(yl, x)

    nfac_u = 1.0 / np.sqrt(1.0 + dyu_dx**2)
    nfac_l = 1.0 / np.sqrt(1.0 + dyl_dx**2)

    if nu > 0:
        basis_u = build_hicks_henne_basis(x, upper_centers, power=power)
        denom_u = np.max(np.abs(basis_u * nfac_u[None, :]), axis=1)
        h_u = float(target_peak_normal) / np.maximum(denom_u, eps)
    else:
        h_u = np.zeros(0, dtype=float)

    if nl > 0:
        basis_l = build_hicks_henne_basis(x, lower_centers, power=power)
        denom_l = np.max(np.abs(basis_l * nfac_l[None, :]), axis=1)
        h_l = float(target_peak_normal) / np.maximum(denom_l, eps)
    else:
        h_l = np.zeros(0, dtype=float)

    return np.concatenate([h_u, h_l])

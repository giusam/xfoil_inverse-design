import numpy as np

from settings import SETTINGS


def build_hh_centers(n_hh):
    xmin = SETTINGS["optimization"]["adaptive"]["xmin"]
    xmax = SETTINGS["optimization"]["adaptive"]["xmax"]
    return np.linspace(xmin, xmax, n_hh)


def get_midpoint_candidates(centers):
    xmin = SETTINGS["optimization"]["adaptive"]["xmin"]
    xmax = SETTINGS["optimization"]["adaptive"]["xmax"]

    centers = np.asarray(sorted(centers), dtype=float)

    if len(centers) == 0:
        return []

    extended = np.concatenate(([xmin], centers, [xmax]))
    mids = 0.5 * (extended[:-1] + extended[1:])

    candidates = []
    for i, c in enumerate(mids):
        if np.min(np.abs(centers - c)) > 1.0e-12:
            candidates.append(
                {
                    "x": float(c),
                    "interval_id": i,
                }
            )

    return candidates


def split_total_across_sides(n_total):
    n_upper = (n_total + 1) // 2
    n_lower = n_total // 2
    return n_upper, n_lower


def build_side_specific_initial_centers(n0_total):
    n_upper, n_lower = split_total_across_sides(n0_total)

    upper = list(build_hh_centers(n_upper))
    lower = list(build_hh_centers(n_lower))

    return upper, lower


def lift_a_to_new_side_centers(old_upper, old_lower, old_a, new_upper, new_lower):
    old_upper = list(old_upper)
    old_lower = list(old_lower)
    new_upper = list(new_upper)
    new_lower = list(new_lower)

    nu_old = len(old_upper)
    nl_old = len(old_lower)
    nu_new = len(new_upper)
    nl_new = len(new_lower)

    a_new = np.zeros(nu_new + nl_new)

    if old_a is None:
        return a_new

    old_upper_index = {round(c, 12): i for i, c in enumerate(old_upper)}
    old_lower_index = {round(c, 12): i for i, c in enumerate(old_lower)}

    for j, c in enumerate(new_upper):
        key = round(c, 12)
        if key in old_upper_index:
            i_old = old_upper_index[key]
            a_new[j] = old_a[i_old]

    for j, c in enumerate(new_lower):
        key = round(c, 12)
        if key in old_lower_index:
            i_old = old_lower_index[key]
            a_new[nu_new + j] = old_a[nu_old + i_old]

    return a_new


def _compute_adaptive_nadd(current_ndv, ncandidates):
    if ncandidates <= 0:
        return 0

    opt_ad = SETTINGS["optimization"]["adaptive"]
    growth_ratio = float(opt_ad.get("growth_ratio", 2.0))

    if growth_ratio <= 1.0:
        target_ndv = current_ndv + 1
    else:
        target_ndv = int(np.ceil(growth_ratio * current_ndv))

    nadd = max(1, target_ndv - current_ndv)
    nadd = min(nadd, ncandidates)

    max_user = opt_ad.get("n_add_per_level")
    if max_user is not None:
        nadd = min(nadd, max_user)

    return nadd

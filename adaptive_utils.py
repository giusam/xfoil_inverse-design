import numpy as np

from settings import SETTINGS


_VALID_INTERVAL_SAMPLING_MODES = {"midpoint", "best_of_3"}


def build_hh_centers(n_hh):
    xmin = SETTINGS["optimization"]["adaptive"]["xmin"]
    xmax = SETTINGS["optimization"]["adaptive"]["xmax"]
    return np.linspace(xmin, xmax, n_hh)


def get_interval_sampling_spec():
    opt_ad = SETTINGS["optimization"]["adaptive"]
    mode = str(opt_ad.get("interval_sampling_mode", "midpoint")).strip().lower()

    if mode not in _VALID_INTERVAL_SAMPLING_MODES:
        allowed = ", ".join(sorted(_VALID_INTERVAL_SAMPLING_MODES))
        raise ValueError(
            f"Invalid optimization.adaptive.interval_sampling_mode={mode!r}. "
            f"Allowed values: {allowed}."
        )

    if mode == "midpoint":
        return mode, (0.5,)

    raw_fractions = opt_ad.get("interval_sampling_fractions", [0.25, 0.5, 0.75])
    if not isinstance(raw_fractions, (list, tuple, np.ndarray)):
        raise ValueError(
            "optimization.adaptive.interval_sampling_fractions must be a list-like "
            "with exactly 3 values in (0, 1)."
        )

    fractions = tuple(float(frac) for frac in raw_fractions)
    if len(fractions) != 3:
        raise ValueError(
            "optimization.adaptive.interval_sampling_fractions must contain exactly 3 values "
            "when interval_sampling_mode='best_of_3'."
        )

    for frac in fractions:
        if not 0.0 < frac < 1.0:
            raise ValueError(
                "optimization.adaptive.interval_sampling_fractions values must lie strictly in (0, 1)."
            )

    return mode, fractions


def build_candidate_intervals(centers, side=None):
    xmin = SETTINGS["optimization"]["adaptive"]["xmin"]
    xmax = SETTINGS["optimization"]["adaptive"]["xmax"]

    centers = np.asarray(sorted(centers), dtype=float)

    if len(centers) == 0:
        return []

    extended = np.concatenate(([xmin], centers, [xmax]))
    side_name = None if side is None else str(side).upper()
    intervals = []
    for i, (x_left, x_right) in enumerate(zip(extended[:-1], extended[1:])):
        interval = {
            "interval_id": i,
            "x_left": float(x_left),
            "x_right": float(x_right),
        }
        if side_name is not None:
            interval["side"] = side_name
            interval["interval_label"] = f"{side_name}_{i:02d}"
        else:
            interval["interval_label"] = f"INTERVAL_{i:02d}"
        intervals.append(interval)

    return intervals


def build_interval_candidates(centers, side=None, sampling_mode=None, sampling_fractions=None):
    centers = np.asarray(sorted(centers), dtype=float)
    intervals = build_candidate_intervals(centers, side=side)

    if sampling_mode is None or sampling_fractions is None:
        sampling_mode, sampling_fractions = get_interval_sampling_spec()
    sampling_mode = str(sampling_mode).strip().lower()

    candidates = []
    for interval in intervals:
        x_left = float(interval["x_left"])
        x_right = float(interval["x_right"])
        dx = x_right - x_left
        sampling_fallback_midpoint = False

        if sampling_mode == "midpoint":
            fractions_for_interval = (0.5,)
        elif sampling_mode == "best_of_3":
            threshold = SETTINGS["optimization"]["adaptive"].get(
                "interval_multi_sample_min_width",
                None,
            )
            if threshold is not None and float(threshold) > 0.0 and dx < float(threshold):
                fractions_for_interval = (0.5,)
                sampling_fallback_midpoint = True
            else:
                fractions_for_interval = sampling_fractions
        else:
            allowed = ", ".join(sorted(_VALID_INTERVAL_SAMPLING_MODES))
            raise ValueError(
                f"Invalid optimization.adaptive.interval_sampling_mode={sampling_mode!r}. "
                f"Allowed values: {allowed}."
            )

        for frac in fractions_for_interval:
            xc = x_left + float(frac) * dx
            if len(centers) > 0 and np.min(np.abs(centers - xc)) <= 1.0e-12:
                continue

            candidate = dict(interval)
            candidate.update(
                {
                    "x": float(xc),
                    "local_fraction": float(frac),
                    "sampling_mode": str(sampling_mode),
                    "interval_width": float(dx),
                    "n_samples_for_interval": int(len(fractions_for_interval)),
                    "sampling_fallback_midpoint": bool(sampling_fallback_midpoint),
                }
            )
            candidates.append(candidate)

    return candidates


def get_midpoint_candidates(centers):
    centers = np.asarray(sorted(centers), dtype=float)

    return build_interval_candidates(
        centers,
        sampling_mode="midpoint",
        sampling_fractions=(0.5,),
    )


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

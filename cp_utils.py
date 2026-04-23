import numpy as np


def split_upper_lower_cp_from_x(x_cp, cp):
    x_cp = np.asarray(x_cp, dtype=float)
    cp = np.asarray(cp)

    i_le = np.argmin(x_cp)

    upper = {
        "x": x_cp[:i_le + 1],
        "cp": cp[:i_le + 1],
    }
    lower = {
        "x": x_cp[i_le:],
        "cp": cp[i_le:],
    }

    return {
        "upper": upper,
        "lower": lower,
    }
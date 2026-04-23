
from pathlib import Path

SETTINGS = {

    "run": {
        "do_static": False,
        "do_adaptive_grad": True,
        "do_adaptive_ikkt": False,
        "do_adaptive_pred": False,
        "do_adaptive_oracle": False,
    },

    "xfoil": {
        "alpha": 2.0,
        "mach": 0.001,
        "Re": 1e6,
        "xfoil_iter": 300,
        "timeout": 60,
        "working_dir": Path("run_debug"),
        "xtr_upper": 0.03,
        "xtr_lower": 0.03,
    },

    "aero": {
        "backend": "cmplxfoil",      # "xfoil" oppure "cmplxfoil"
        "target_backend": "cmplxfoil",       # "xfoil" oppure "cmplxfoil"
        "derivatives": "fd",       # "fd" oppure "cs"
        "quiet": True,             # Se True, sopprime output di XFOIL/CMPLXFOIL
    },
    
    "snapshots": {
        "enabled": True,
        "dir_name": "snapshots",
    },

    "geom": {
        "n_points": 201,
        "thickness": 0.12,
    },

    "constraints": {
        "CL": {
            "enabled": False,
            "kind": "eq",
            "target": None,
            "alpha": 0.5,
            "scale": 0.01,
            "tol": 0,
        },
        "CD": {
            "enabled": False,
            "kind": "le",
            "target": None,
            "alpha": 0.5,
            "scale": 0.001,
            "tol": 0.0,
        },
        "CM": {
            "enabled": False,
            "kind": "eq",
            "target": 0.0,
            "alpha": 0.5,
            "scale": 0.01,
            "tol": 0.001,
        },
        "tmax": {
            "enabled": False,
            "kind": "ge",
            "target": 0.11,
            "alpha": 0.5,
            "scale": 0.01,
            "tol": 0.0,
        },
        "area": {
            "enabled": False,
            "kind": "ge",
            "target": 0.08,
            "alpha": 0.5,
            "scale": 0.01,
            "tol": 0.0,
        },
        "thickness_stations": {
            "enabled": False,
            "kind": "ge",
            "stations": [
                {"x": 0.30, "target": 0.10, "alpha": 0.5, "scale": 0.01},
                {"x": 0.60, "target": 0.08, "alpha": 0.5, "scale": 0.01},
            ],
        },
    },

    "initial_shape": {
        "bernstein_order": 15,
        "random_seed": 2,
        "random_amp": 0.1,
        "n_seeds": 3,
        "seed_list": [9],
    },

    "optimization": {
        "hh_power": 4,
        "n_hh_static": 20,
        "bounds": (-0.01, 0.01),
        "maxiter": 200,
        "ftol": 1.0e-8,
        "opt_fd_target_peak_normal": 1e-4,
        "score_fd_target_peak_normal": 1e-4,
        "pred_fd_target_peak_normal": 5e-4,
        "pred_hessian_reg": 1.0e-8,
        "penalty_factor": 20.0,
        "adaptive": {
            "tol_active": 1.0e-3,
            "n0": 8,
            "n_final": 20,
            "n_add_per_level": 1,
            "growth_ratio": 1.25,
            "xmin": 0.05,
            "xmax": 0.95,
        },
    },
}

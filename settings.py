import ast
from pathlib import Path

SETTINGS = {

    "run": {
        "do_static": False,
        "do_adaptive_grad": True,
        "do_adaptive_ikkt": False,
        "do_adaptive_pred": False,
        "do_adaptive_oracle": False,
        "do_adaptive_spring": False,
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
            "interval_sampling_mode": "midpoint",
            "interval_sampling_fractions": [0.25, 0.5, 0.75],
            "grad_score_mode": "grad_norm",
            "write_candidate_score_csv": True,
            "gn_schur_reg": 1.0e-10,
            "gn_schur_rcond": 1.0e-10,
            "gn_schur_fd_target_peak_normal": None,
            "force_candidates_enabled": False,
            "force_candidates_file": None,
            "force_candidates_strict": True,
            "force_candidates_tol": 1.0e-10,
        },
    },

    "spring_reallocation": {
        "enabled": False,
        "n_dv": 20,
        "n_cycles": 1,
        "weight_mode": "abs_a",
        "A": 10.0,
        "omega": 0.25,
        "max_dx": 0.05,
        "min_spacing": 0.02,
        "ridge": 1.0e-10,
        "fix_ends": True,
        "transfer_mode": "sin",
        "transfer_power": 1.0,
    },

    "periodic_spring_adaptive": {
        "enabled": False,
        "levels": [12, 16, 20],
        "accept_mode_intermediate": "rebase_keep_new_centers",
        "accept_mode_final": "accept_if_improved",
        "force_grad_score_mode": "grad_norm",
    },

    "adaptive_spring": {
        "enabled": False,
        "mode": "final",
        "levels": [12, 16, 20],
        "accept_mode_intermediate": "rebase_keep_new_centers",
        "accept_mode_final": "accept_if_improved",
        "force_grad_score_mode": "grad_norm",
    },

    "_legacy": {
        "adaptive_spring_periodic_policy": None,
        "adaptive_spring_periodic_levels": None,
        "spring_realloc_restart_mode": None,
    },
}
def _parse_cfg_value(raw: str):
    s = raw.strip()
    up = s.upper()

    if up in {"YES", "TRUE", "ON"}:
        return True
    if up in {"NO", "FALSE", "OFF"}:
        return False
    if up == "NONE":
        return None

    try:
        return ast.literal_eval(s)
    except Exception:
        return s


_CFG_KEY_MAP = {
    # ---------------------------
    # aero
    # ---------------------------
    "AERO_BACKEND": ("aero", "backend"),
    "AERO_TARGET_BACKEND": ("aero", "target_backend"),
    "AERO_DERIVATIVES": ("aero", "derivatives"),
    "AERO_QUIET": ("aero", "quiet"),

    # ---------------------------
    # run
    # ---------------------------
    "RUN_DO_STATIC": ("run", "do_static"),
    "RUN_DO_ADAPTIVE_GRAD": ("run", "do_adaptive_grad"),
    "RUN_DO_ADAPTIVE_IKKT": ("run", "do_adaptive_ikkt"),
    "RUN_DO_ADAPTIVE_PRED": ("run", "do_adaptive_pred"),
    "RUN_DO_ADAPTIVE_ORACLE": ("run", "do_adaptive_oracle"),
    "RUN_DO_ADAPTIVE_SPRING": ("run", "do_adaptive_spring"),

    # ---------------------------
    # xfoil / aero conditions
    # ---------------------------
    "XFOIL_ALPHA": ("xfoil", "alpha"),
    "XFOIL_MACH": ("xfoil", "mach"),
    "XFOIL_RE": ("xfoil", "Re"),
    "XFOIL_ITER": ("xfoil", "xfoil_iter"),
    "XFOIL_TIMEOUT": ("xfoil", "timeout"),
    "XFOIL_WORKING_DIR": ("xfoil", "working_dir"),
    "XFOIL_XTR_UPPER": ("xfoil", "xtr_upper"),
    "XFOIL_XTR_LOWER": ("xfoil", "xtr_lower"),

    # ---------------------------
    # snapshots
    # ---------------------------
    "SNAPSHOTS_ENABLED": ("snapshots", "enabled"),
    "SNAPSHOTS_DIR_NAME": ("snapshots", "dir_name"),

    # ---------------------------
    # geometry
    # ---------------------------
    "GEOM_N_POINTS": ("geom", "n_points"),
    "GEOM_THICKNESS": ("geom", "thickness"),

    # ---------------------------
    # initial shape
    # ---------------------------
    "INITIAL_SHAPE_BERNSTEIN_ORDER": ("initial_shape", "bernstein_order"),
    "INITIAL_SHAPE_RANDOM_SEED": ("initial_shape", "random_seed"),
    "INITIAL_SHAPE_RANDOM_AMP": ("initial_shape", "random_amp"),
    "INITIAL_SHAPE_N_SEEDS": ("initial_shape", "n_seeds"),
    "INITIAL_SHAPE_SEED_LIST": ("initial_shape", "seed_list"),

    # ---------------------------
    # optimization
    # ---------------------------
    "OPT_HH_POWER": ("optimization", "hh_power"),
    "OPT_N_HH_STATIC": ("optimization", "n_hh_static"),
    "OPT_BOUNDS": ("optimization", "bounds"),
    "OPT_MAXITER": ("optimization", "maxiter"),
    "OPT_FTOL": ("optimization", "ftol"),
    "OPT_FD_TARGET_PEAK_NORMAL": ("optimization", "opt_fd_target_peak_normal"),
    "OPT_SCORE_FD_TARGET_PEAK_NORMAL": ("optimization", "score_fd_target_peak_normal"),
    "OPT_PRED_FD_TARGET_PEAK_NORMAL": ("optimization", "pred_fd_target_peak_normal"),
    "OPT_PRED_HESSIAN_REG": ("optimization", "pred_hessian_reg"),
    "OPT_PENALTY_FACTOR": ("optimization", "penalty_factor"),

    # ---------------------------
    # optimization.adaptive
    # ---------------------------
    "ADAPT_TOL_ACTIVE": ("optimization", "adaptive", "tol_active"),
    "ADAPT_N0": ("optimization", "adaptive", "n0"),
    "ADAPT_N_FINAL": ("optimization", "adaptive", "n_final"),
    "ADAPT_N_ADD_PER_LEVEL": ("optimization", "adaptive", "n_add_per_level"),
    "ADAPT_GROWTH_RATIO": ("optimization", "adaptive", "growth_ratio"),
    "ADAPT_XMIN": ("optimization", "adaptive", "xmin"),
    "ADAPT_XMAX": ("optimization", "adaptive", "xmax"),
    "ADAPT_INTERVAL_SAMPLING_MODE": ("optimization", "adaptive", "interval_sampling_mode"),
    "ADAPT_INTERVAL_SAMPLING_FRACTIONS": ("optimization", "adaptive", "interval_sampling_fractions"),
    "ADAPT_GRAD_SCORE_MODE": ("optimization", "adaptive", "grad_score_mode"),
    "ADAPT_WRITE_CANDIDATE_SCORE_CSV": ("optimization", "adaptive", "write_candidate_score_csv"),
    "ADAPT_GN_SCHUR_REG": ("optimization", "adaptive", "gn_schur_reg"),
    "ADAPT_GN_SCHUR_RCOND": ("optimization", "adaptive", "gn_schur_rcond"),
    "ADAPT_GN_SCHUR_FD_TARGET_PEAK_NORMAL": ("optimization", "adaptive", "gn_schur_fd_target_peak_normal"),
    "ADAPT_FORCE_CANDIDATES_ENABLED": ("optimization", "adaptive", "force_candidates_enabled"),
    "ADAPT_FORCE_CANDIDATES_FILE": ("optimization", "adaptive", "force_candidates_file"),
    "ADAPT_FORCE_CANDIDATES_STRICT": ("optimization", "adaptive", "force_candidates_strict"),
    "ADAPT_FORCE_CANDIDATES_TOL": ("optimization", "adaptive", "force_candidates_tol"),

    # ---------------------------
    # adaptive_spring
    # ---------------------------
    "ADAPTIVE_SPRING_MODE": ("adaptive_spring", "mode"),
    "ADAPTIVE_SPRING_ENABLED": ("adaptive_spring", "enabled"),
    "ADAPTIVE_SPRING_LEVELS": ("adaptive_spring", "levels"),
    "ADAPTIVE_SPRING_FORCE_GRAD_SCORE_MODE": ("adaptive_spring", "force_grad_score_mode"),
    "ADAPTIVE_SPRING_ACCEPT_MODE_INTERMEDIATE": ("adaptive_spring", "accept_mode_intermediate"),
    "ADAPTIVE_SPRING_ACCEPT_MODE_FINAL": ("adaptive_spring", "accept_mode_final"),
    "ADAPTIVE_SPRING_PERIODIC_POLICY": ("_legacy", "adaptive_spring_periodic_policy"),
    "ADAPTIVE_SPRING_PERIODIC_LEVELS": ("_legacy", "adaptive_spring_periodic_levels"),

    # ---------------------------
    # spring_reallocation
    # ---------------------------
    "SPRING_REALLOC_ENABLED": ("spring_reallocation", "enabled"),
    "SPRING_REALLOC_N_DV": ("spring_reallocation", "n_dv"),
    "SPRING_REALLOC_N_CYCLES": ("spring_reallocation", "n_cycles"),
    "SPRING_REALLOC_WEIGHT_MODE": ("spring_reallocation", "weight_mode"),
    # Legacy: accepted for old cfg files, ignored. Spring restart is always rebase.
    "SPRING_REALLOC_RESTART_MODE": ("_legacy", "spring_realloc_restart_mode"),
    "SPRING_REALLOC_A": ("spring_reallocation", "A"),
    "SPRING_REALLOC_OMEGA": ("spring_reallocation", "omega"),
    "SPRING_REALLOC_MAX_DX": ("spring_reallocation", "max_dx"),
    "SPRING_REALLOC_MIN_SPACING": ("spring_reallocation", "min_spacing"),
    "SPRING_REALLOC_RIDGE": ("spring_reallocation", "ridge"),
    "SPRING_REALLOC_FIX_ENDS": ("spring_reallocation", "fix_ends"),
    "SPRING_REALLOC_TRANSFER_MODE": ("spring_reallocation", "transfer_mode"),
    "SPRING_REALLOC_TRANSFER_POWER": ("spring_reallocation", "transfer_power"),

    # ---------------------------
    # periodic_spring_adaptive
    # ---------------------------
    "PERIODIC_SPRING_ENABLED": ("periodic_spring_adaptive", "enabled"),
    "PERIODIC_SPRING_LEVELS": ("periodic_spring_adaptive", "levels"),
    "PERIODIC_SPRING_ACCEPT_MODE_INTERMEDIATE": ("periodic_spring_adaptive", "accept_mode_intermediate"),
    "PERIODIC_SPRING_ACCEPT_MODE_FINAL": ("periodic_spring_adaptive", "accept_mode_final"),
    "PERIODIC_SPRING_FORCE_GRAD_SCORE_MODE": ("periodic_spring_adaptive", "force_grad_score_mode"),

    # ---------------------------
    # constraints: CL
    # ---------------------------
    "CONSTRAINT_CL_ENABLED": ("constraints", "CL", "enabled"),
    "CONSTRAINT_CL_KIND": ("constraints", "CL", "kind"),
    "CONSTRAINT_CL_TARGET": ("constraints", "CL", "target"),
    "CONSTRAINT_CL_ALPHA": ("constraints", "CL", "alpha"),
    "CONSTRAINT_CL_SCALE": ("constraints", "CL", "scale"),
    "CONSTRAINT_CL_TOL": ("constraints", "CL", "tol"),

    # ---------------------------
    # constraints: CD
    # ---------------------------
    "CONSTRAINT_CD_ENABLED": ("constraints", "CD", "enabled"),
    "CONSTRAINT_CD_KIND": ("constraints", "CD", "kind"),
    "CONSTRAINT_CD_TARGET": ("constraints", "CD", "target"),
    "CONSTRAINT_CD_ALPHA": ("constraints", "CD", "alpha"),
    "CONSTRAINT_CD_SCALE": ("constraints", "CD", "scale"),
    "CONSTRAINT_CD_TOL": ("constraints", "CD", "tol"),

    # ---------------------------
    # constraints: CM
    # ---------------------------
    "CONSTRAINT_CM_ENABLED": ("constraints", "CM", "enabled"),
    "CONSTRAINT_CM_KIND": ("constraints", "CM", "kind"),
    "CONSTRAINT_CM_TARGET": ("constraints", "CM", "target"),
    "CONSTRAINT_CM_ALPHA": ("constraints", "CM", "alpha"),
    "CONSTRAINT_CM_SCALE": ("constraints", "CM", "scale"),
    "CONSTRAINT_CM_TOL": ("constraints", "CM", "tol"),

    # ---------------------------
    # constraints: tmax
    # ---------------------------
    "CONSTRAINT_TMAX_ENABLED": ("constraints", "tmax", "enabled"),
    "CONSTRAINT_TMAX_KIND": ("constraints", "tmax", "kind"),
    "CONSTRAINT_TMAX_TARGET": ("constraints", "tmax", "target"),
    "CONSTRAINT_TMAX_ALPHA": ("constraints", "tmax", "alpha"),
    "CONSTRAINT_TMAX_SCALE": ("constraints", "tmax", "scale"),
    "CONSTRAINT_TMAX_TOL": ("constraints", "tmax", "tol"),

    # ---------------------------
    # constraints: area
    # ---------------------------
    "CONSTRAINT_AREA_ENABLED": ("constraints", "area", "enabled"),
    "CONSTRAINT_AREA_KIND": ("constraints", "area", "kind"),
    "CONSTRAINT_AREA_TARGET": ("constraints", "area", "target"),
    "CONSTRAINT_AREA_ALPHA": ("constraints", "area", "alpha"),
    "CONSTRAINT_AREA_SCALE": ("constraints", "area", "scale"),
    "CONSTRAINT_AREA_TOL": ("constraints", "area", "tol"),

    # ---------------------------
    # constraints: thickness stations
    # ---------------------------
    "CONSTRAINT_THICKNESS_STATIONS_ENABLED": ("constraints", "thickness_stations", "enabled"),
    "CONSTRAINT_THICKNESS_STATIONS_KIND": ("constraints", "thickness_stations", "kind"),
    "CONSTRAINT_THICKNESS_STATIONS_STATIONS": ("constraints", "thickness_stations", "stations"),
}

def _set_nested_value(root, path, value):
    node = root
    for key in path[:-1]:
        node = node[key]

    if tuple(path) == ("xfoil", "working_dir"):
        value = Path(str(value))

    node[path[-1]] = value


def validate_settings():
    adapt_cfg = SETTINGS["optimization"]["adaptive"]
    n0 = int(adapt_cfg.get("n0", 0))
    n_final = int(adapt_cfg.get("n_final", 0))
    n_add = int(adapt_cfg.get("n_add_per_level", 0))
    if n0 >= n_final:
        raise ValueError("ADAPT_N0 must be smaller than ADAPT_N_FINAL.")
    if n_add <= 0:
        raise ValueError("ADAPT_N_ADD_PER_LEVEL must be positive.")

    bounds = SETTINGS["optimization"].get("bounds", None)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("OPT_BOUNDS must be a pair (min, max).")
    bmin = float(bounds[0])
    bmax = float(bounds[1])
    if not bmin < bmax:
        raise ValueError("OPT_BOUNDS must satisfy min < max.")
    SETTINGS["optimization"]["bounds"] = (bmin, bmax)

    score_mode = str(
        SETTINGS["optimization"]["adaptive"].get("grad_score_mode", "grad_norm")
    ).strip().lower()
    supported_score_modes = ["grad_norm", "gn_schur"]
    if score_mode not in supported_score_modes:
        raise ValueError(
            f"Unsupported ADAPT_GRAD_SCORE_MODE={score_mode!r}. "
            f"Supported modes: {supported_score_modes}."
        )
    SETTINGS["optimization"]["adaptive"]["grad_score_mode"] = score_mode

    force_enabled = adapt_cfg.get("force_candidates_enabled", False)
    if not isinstance(force_enabled, bool):
        raise ValueError("ADAPT_FORCE_CANDIDATES_ENABLED must be boolean.")
    adapt_cfg["force_candidates_enabled"] = force_enabled

    force_file = adapt_cfg.get("force_candidates_file", None)
    if force_file is not None and not isinstance(force_file, str):
        raise ValueError("ADAPT_FORCE_CANDIDATES_FILE must be None or a string path.")
    adapt_cfg["force_candidates_file"] = force_file

    force_strict = adapt_cfg.get("force_candidates_strict", True)
    if not isinstance(force_strict, bool):
        raise ValueError("ADAPT_FORCE_CANDIDATES_STRICT must be boolean.")
    adapt_cfg["force_candidates_strict"] = force_strict

    force_tol = float(adapt_cfg.get("force_candidates_tol", 1.0e-10))
    if force_tol <= 0.0:
        raise ValueError("ADAPT_FORCE_CANDIDATES_TOL must be > 0.")
    adapt_cfg["force_candidates_tol"] = force_tol

    adaptive_spring = SETTINGS.setdefault("adaptive_spring", {})
    mode = str(adaptive_spring.get("mode", "final")).strip().lower()
    supported_spring_modes = ["final", "every_refine", "levels"]
    if mode not in supported_spring_modes:
        raise ValueError(
            f"Unsupported ADAPTIVE_SPRING_MODE={mode!r}. "
            f"Supported modes: {supported_spring_modes}."
        )
    adaptive_spring["mode"] = mode
    enabled = bool(adaptive_spring.get("enabled", False))
    adaptive_spring["enabled"] = enabled
    SETTINGS["run"]["do_adaptive_spring"] = enabled

    if mode == "levels" and enabled:
        try:
            levels = [int(v) for v in adaptive_spring.get("levels", [])]
        except Exception as exc:
            raise ValueError("ADAPTIVE_SPRING_LEVELS must contain integer levels.") from exc
        if not levels:
            raise ValueError("ADAPTIVE_SPRING_LEVELS must contain at least one integer level.")
        bad = [v for v in levels if not (n0 < v <= n_final)]
        if bad:
            raise ValueError(
                "ADAPTIVE_SPRING_LEVELS must satisfy ADAPT_N0 < level <= ADAPT_N_FINAL. "
                f"Invalid levels: {bad}."
            )
        adaptive_spring["levels"] = levels

    accept_intermediate = str(
        adaptive_spring.get("accept_mode_intermediate", "rebase_keep_new_centers")
    ).strip().lower()
    if accept_intermediate != "rebase_keep_new_centers":
        raise ValueError("ADAPTIVE_SPRING_ACCEPT_MODE_INTERMEDIATE supports only 'rebase_keep_new_centers'.")
    adaptive_spring["accept_mode_intermediate"] = accept_intermediate

    accept_final = str(adaptive_spring.get("accept_mode_final", "accept_if_improved")).strip().lower()
    if accept_final != "accept_if_improved":
        raise ValueError("ADAPTIVE_SPRING_ACCEPT_MODE_FINAL supports only 'accept_if_improved'.")
    adaptive_spring["accept_mode_final"] = accept_final

    adaptive_spring["force_grad_score_mode"] = str(
        adaptive_spring.get("force_grad_score_mode", "grad_norm")
    ).strip().lower()
    if adaptive_spring["force_grad_score_mode"] not in supported_score_modes:
        raise ValueError(
            f"Unsupported ADAPTIVE_SPRING_FORCE_GRAD_SCORE_MODE={adaptive_spring['force_grad_score_mode']!r}. "
            f"Supported modes: {supported_score_modes}."
        )

    spring_cfg = SETTINGS["spring_reallocation"]
    omega = float(spring_cfg.get("omega", 0.0))
    if not (0.0 <= omega <= 1.0):
        raise ValueError("SPRING_REALLOC_OMEGA must be between 0 and 1.")
    spring_cfg["omega"] = omega
    if float(spring_cfg.get("min_spacing", 0.0)) <= 0.0:
        raise ValueError("SPRING_REALLOC_MIN_SPACING must be positive.")
    if float(spring_cfg.get("max_dx", 0.0)) <= 0.0:
        raise ValueError("SPRING_REALLOC_MAX_DX must be positive.")
    if float(spring_cfg.get("A", 0.0)) < 1.0:
        raise ValueError("SPRING_REALLOC_A must be >= 1.")
    spring_cfg["min_spacing"] = float(spring_cfg["min_spacing"])
    spring_cfg["max_dx"] = float(spring_cfg["max_dx"])
    spring_cfg["A"] = float(spring_cfg["A"])


def _sync_periodic_spring_aliases(seen_keys):
    periodic = SETTINGS.get("periodic_spring_adaptive", {})
    adaptive = SETTINGS.get("adaptive_spring", {})
    legacy = SETTINGS.get("_legacy", {})

    if "RUN_DO_ADAPTIVE_SPRING" in seen_keys and "ADAPTIVE_SPRING_ENABLED" not in seen_keys:
        adaptive["enabled"] = bool(SETTINGS["run"].get("do_adaptive_spring", False))
    if "PERIODIC_SPRING_ENABLED" in seen_keys and "ADAPTIVE_SPRING_ENABLED" not in seen_keys and "RUN_DO_ADAPTIVE_SPRING" not in seen_keys:
        adaptive["enabled"] = bool(periodic.get("enabled", False))
        if "ADAPTIVE_SPRING_MODE" not in seen_keys and "ADAPTIVE_SPRING_PERIODIC_POLICY" not in seen_keys:
            adaptive["mode"] = "levels"

    if "ADAPTIVE_SPRING_PERIODIC_POLICY" in seen_keys and "ADAPTIVE_SPRING_MODE" not in seen_keys:
        policy = str(legacy.get("adaptive_spring_periodic_policy", "")).strip().lower()
        if policy in {"every_refine", "levels"}:
            adaptive["mode"] = policy
        else:
            raise ValueError(
                f"Unsupported ADAPTIVE_SPRING_PERIODIC_POLICY={policy!r}. "
                "Supported legacy policies: ['levels', 'every_refine']."
            )

    if "ADAPTIVE_SPRING_PERIODIC_LEVELS" in seen_keys and "ADAPTIVE_SPRING_LEVELS" not in seen_keys:
        adaptive["levels"] = list(legacy.get("adaptive_spring_periodic_levels", [12, 16, 20]))
        if "ADAPTIVE_SPRING_MODE" not in seen_keys and "ADAPTIVE_SPRING_PERIODIC_POLICY" not in seen_keys:
            adaptive["mode"] = "levels"
    if (
        "PERIODIC_SPRING_LEVELS" in seen_keys
        and "ADAPTIVE_SPRING_LEVELS" not in seen_keys
        and "ADAPTIVE_SPRING_PERIODIC_LEVELS" not in seen_keys
    ):
        adaptive["levels"] = list(periodic.get("levels", [12, 16, 20]))
        if "ADAPTIVE_SPRING_MODE" not in seen_keys and "ADAPTIVE_SPRING_PERIODIC_POLICY" not in seen_keys:
            adaptive["mode"] = "levels"
    if (
        "PERIODIC_SPRING_ACCEPT_MODE_INTERMEDIATE" in seen_keys
        and "ADAPTIVE_SPRING_ACCEPT_MODE_INTERMEDIATE" not in seen_keys
    ):
        adaptive["accept_mode_intermediate"] = periodic.get(
            "accept_mode_intermediate",
            "rebase_keep_new_centers",
        )
    if (
        "PERIODIC_SPRING_ACCEPT_MODE_FINAL" in seen_keys
        and "ADAPTIVE_SPRING_ACCEPT_MODE_FINAL" not in seen_keys
    ):
        adaptive["accept_mode_final"] = periodic.get("accept_mode_final", "accept_if_improved")
    if (
        "PERIODIC_SPRING_FORCE_GRAD_SCORE_MODE" in seen_keys
        and "ADAPTIVE_SPRING_FORCE_GRAD_SCORE_MODE" not in seen_keys
    ):
        adaptive["force_grad_score_mode"] = periodic.get("force_grad_score_mode", "grad_norm")


def apply_cfg_overrides(cfg_path):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"CFG file not found: {cfg_path}")

    seen_keys = set()
    for lineno, raw_line in enumerate(cfg_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue
        if line.startswith("%") or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid CFG line {lineno}: {raw_line}")

        key, raw_value = line.split("=", 1)
        key = key.strip().upper()
        raw_value = raw_value.strip()

        if key not in _CFG_KEY_MAP:
            raise KeyError(f"Unknown CFG key at line {lineno}: {key}")

        value = _parse_cfg_value(raw_value)
        _set_nested_value(SETTINGS, _CFG_KEY_MAP[key], value)
        seen_keys.add(key)

    _sync_periodic_spring_aliases(seen_keys)
    validate_settings()

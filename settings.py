from pathlib import Path

SETTINGS = {

    "run": {
        "do_static": False,
        "do_adaptive": True,
    },

    # =========================
    # XFOIL SETTINGS
    # =========================
    "xfoil": {
        "alpha": 2.0,              # angolo d’attacco [deg] a cui valutare il profilo
        "Re": 1e6,                 # numero di Reynolds
        "xfoil_iter": 300,         # max iterazioni interne di XFOIL
        "timeout": 60,             # timeout per evitare blocchi (s)
        "working_dir": Path("run_debug"),  # cartella di lavoro per i file XFOIL
    },

    # =========================
    # GEOMETRIA
    # =========================
    "geom": {
        "n_points": 201,           # numero punti discretizzazione profilo
        "thickness": 0.12,         # spessore target iniziale (NACA0012)
    },

    # =========================
    # VINCOLI (penalty aero + geometrici per SLSQP)
    # =========================
    "constraints": {

        # ---- CL ----
        "CL": {
            "enabled": True,       # attiva il vincolo
            "kind": "eq",          # tipo: eq (uguaglianza)
            "target": None,        # valore target (se None → preso da target airfoil)
            "alpha": 0.1,          # peso della penalty
            "scale": 0.01,         # scala per normalizzare errore
            "tol": 0.003,          # tolleranza (zona senza penalty)
        },

        # ---- CD ----
        "CD": {
            "enabled": False,      # disattivo
            "kind": "le",          # tipo: <=
            "target": None,        # valore massimo
            "alpha": 1.0,          # peso penalty
            "scale": 0.001,        # scala errore
            "tol": 0.0,            # tolleranza (zero = rigido)
        },

        # ---- CM ----
        "CM": {
            "enabled": False,
            "kind": "eq",
            "target": 0.0,
            "alpha": 1.0,
            "scale": 0.01,
            "tol": 0.001,
        },

        # ---- spessore massimo ----
        # gestito da SLSQP, escluso per ora da IKKT
        "tmax": {
            "enabled": False,
            "kind": "ge",
            "target": 0.11,
            "alpha": 1.0,
            "scale": 0.01,
            "tol": 0.0,
        },

        # ---- area ----
        # gestito da SLSQP, incluso in IKKT con gradiente analitico
        "area": {
            "enabled": False,
            "kind": "ge",
            "target": 0.08,
            "alpha": 1.0,
            "scale": 0.01,
            "tol": 0.0,
        },

        # ---- vincoli locali di spessore ----
        # gestiti da SLSQP, inclusi in IKKT con gradiente analitico
        "thickness_stations": {
            "enabled": False,
            "kind": "ge",
            "stations": [
                {"x": 0.30, "target": 0.10, "alpha": 1.0, "scale": 0.01},
                {"x": 0.60, "target": 0.08, "alpha": 1.0, "scale": 0.01},
            ],
        },
    },

    # =========================
    # FORMA INIZIALE
    # =========================
    "initial_shape": {
        "bernstein_order": 15,
        "random_seed": 2,
        "random_amp": 0.04,
        "n_seeds": 1,
    },

    # =========================
    # OTTIMIZZAZIONE
    # =========================
    "optimization": {

        "hh_power": 4,
        "n_hh_static": 16,

        "bounds": (-0.01, 0.01),

        "maxiter": 200,
        "ftol": 1.0e-8,
        "eps": 0.2e-4,

        "penalty_factor": 20.0,

        "adaptive": {

            "indicator": "GRAD",   # "GRAD" oppure "IKKT"
            "tol_active": 1.0e-3,  # per le disuguaglianze: attivo se C(a) > -tol_active

            "n0": 6,
            "n_final": 16,

            "n_add_per_level": 2,
            "growth_ratio": 1.25,

            "xmin": 0.05,
            "xmax": 0.95,

            "fd_step": 0.2e-4,
        },
    },
}

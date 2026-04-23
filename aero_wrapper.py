from settings import SETTINGS
from xfoil_wrapper import run_xfoil
from cmplxfoil_wrapper import run_cmplxfoil


def get_aero_backend():
    backend = SETTINGS.get("aero", {}).get("backend", "xfoil")
    backend = str(backend).strip().lower()

    if backend not in {"xfoil", "cmplxfoil"}:
        raise ValueError(
            f"Unknown aero backend: {backend}. "
            "Allowed values are 'xfoil' and 'cmplxfoil'."
        )

    return backend


def run_aero(*args, **kwargs):
    backend = get_aero_backend()

    if backend == "cmplxfoil":
        return run_cmplxfoil(*args, **kwargs)

    return run_xfoil(*args, **kwargs)
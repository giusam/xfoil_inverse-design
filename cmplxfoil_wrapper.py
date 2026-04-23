import pickle
import shutil
from pathlib import Path
import os
import sys
from contextlib import contextmanager
import numpy as np

from baseclasses import AeroProblem
from cmplxfoil import CMPLXFOIL



_CMPLXFOIL_SOLVER_CACHE = {}


def clear_cmplxfoil_solver_cache():
    _CMPLXFOIL_SOLVER_CACHE.clear()


def _cmplxfoil_solver_cache_key(session_key, airfoil_dat, xtr_upper, xtr_lower, xfoil_iter):
    if session_key is None:
        session_id = str(Path(airfoil_dat).resolve())
    else:
        session_id = str(session_key)

    return (
        session_id,
        float(xtr_upper),
        float(xtr_lower),
        int(xfoil_iter),
    )


def _get_cached_cmplxfoil_solver(
    airfoil_dat,
    session_key,
    workdir,
    xtr_upper,
    xtr_lower,
    xfoil_iter,
):
    key = _cmplxfoil_solver_cache_key(
        session_key=session_key,
        airfoil_dat=airfoil_dat,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        xfoil_iter=xfoil_iter,
    )

    solver = _CMPLXFOIL_SOLVER_CACHE.get(key)
    if solver is None:
        solver = _instantiate_cmplxfoil_solver(
            airfoil_dat=airfoil_dat,
            workdir=workdir,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            xfoil_iter=xfoil_iter,
        )
        _CMPLXFOIL_SOLVER_CACHE[key] = solver

    return solver


def clean_workdir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

def _build_airfoil_coords_array(x, yu, yl):
    x = np.asarray(x, dtype=float)
    yu = np.asarray(yu)
    yl = np.asarray(yl)

    # TE upper -> LE
    xu = x[::-1]
    yu_u = yu[::-1]

    # LE -> TE lower, evitando il duplicato al LE
    xl = x[1:]
    yl_l = yl[1:]

    x_all = np.concatenate([xu, xl])
    y_all = np.concatenate([yu_u, yl_l])

    coords = np.zeros((len(x_all), 3), dtype=np.result_type(x_all, y_all))
    coords[:, 0] = x_all
    coords[:, 1] = y_all
    coords[:, 2] = 0.0

    return coords

def _instantiate_cmplxfoil_solver(
    airfoil_dat,
    workdir,
    xtr_upper,
    xtr_lower,
    xfoil_iter,
):
    local_airfoil = _prepare_airfoil_for_cmplxfoil(airfoil_dat, workdir)

    aero_options = {
        "maxIters": int(xfoil_iter),
        "printRealConvergence": False,
        "printComplexConvergence": False,
        "writeSolution": False,
        "writeSliceFile": False,
        "writeCoordinates": False,
        "plotAirfoil": False,
        "outputDirectory": str(workdir),
        "xTrip": np.array([xtr_upper, xtr_lower], dtype=float),
    }

    return CMPLXFOIL(str(local_airfoil), options=aero_options)

def _prepare_airfoil_for_cmplxfoil(src_path, outdir):
    """
    CMPLXFOIL non digerisce l'header testuale nel .dat.
    Questa funzione crea una copia 'numeric-only' del file airfoil.
    """
    src_path = Path(src_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = src_path.read_text(errors="ignore").splitlines()

    cleaned = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            cleaned.append((x, y))
        except ValueError:
            # ignora header e qualsiasi riga non numerica
            continue

    if len(cleaned) == 0:
        raise ValueError(f"No numeric airfoil coordinates found in {src_path}")

    dst_path = outdir / f"{src_path.stem}_cmplxfoil.dat"
    with open(dst_path, "w", encoding="utf-8") as f:
        for x, y in cleaned:
            f.write(f"{x:.16e} {y:.16e}\n")

    return dst_path


def _load_slice_data(slice_file, ap_name):
    slice_file = Path(slice_file)
    if not slice_file.exists():
        return None

    with open(slice_file, "rb") as f:
        data = pickle.load(f)

    if ap_name not in data:
        return None

    return data[ap_name]


def _build_combined_cp_data(slice_data):
    """
    Costruisce cp_data nel formato atteso dal resto del codice:
        {
            "x": ...,
            "cp": ...
        }

    Per imitare il tracciato classico di XFOIL:
    - upper: da TE a LE  -> x decrescente
    - lower: da LE a TE  -> x crescente
    """
    x_upper = np.real(np.asarray(slice_data["x_upper"])).astype(float)
    cp_upper = np.asarray(slice_data["cp_visc_upper"])

    x_lower = np.real(np.asarray(slice_data["x_lower"])).astype(float)
    cp_lower = np.asarray(slice_data["cp_visc_lower"])

    # upper: TE -> LE
    idx_u = np.argsort(x_upper)[::-1]
    x_upper_ord = x_upper[idx_u]
    cp_upper_ord = cp_upper[idx_u]

    # lower: LE -> TE
    idx_l = np.argsort(x_lower)
    x_lower_ord = x_lower[idx_l]
    cp_lower_ord = cp_lower[idx_l]

    x_all = np.concatenate([x_upper_ord, x_lower_ord])
    cp_all = np.concatenate([cp_upper_ord, cp_lower_ord])

    return {
        "x": x_all,
        "cp": cp_all,
    }

@contextmanager
def _suppress_native_output(enabled=True):
    """
    Sopprime stdout/stderr anche per librerie native Fortran/C.
    """
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def run_cmplxfoil_coords(
    airfoil_dat,
    x,
    yu,
    yl,
    alpha_deg,
    reynolds,
    xfoil_iter,
    timeout,
    working_dir,
    session_key=None,
    xtr_upper=None,
    xtr_lower=None,
    mach=None,
):
    from settings import SETTINGS

    xfoil_cfg = SETTINGS.get("xfoil", {})
    aero_cfg = SETTINGS.get("aero", {})
    quiet = bool(aero_cfg.get("quiet", True))

    if xtr_upper is None:
        xtr_upper = float(xfoil_cfg.get("xtr_upper", 1.0))
    else:
        xtr_upper = float(xtr_upper)

    if xtr_lower is None:
        xtr_lower = float(xfoil_cfg.get("xtr_lower", 1.0))
    else:
        xtr_lower = float(xtr_lower)

    if mach is None:
        mach = float(xfoil_cfg.get("mach", 0.001))
    else:
        mach = float(mach)

    if not (0.0 <= xtr_upper <= 1.0):
        raise ValueError(f"xtr_upper must be in [0, 1], got {xtr_upper}")
    if not (0.0 <= xtr_lower <= 1.0):
        raise ValueError(f"xtr_lower must be in [0, 1], got {xtr_lower}")

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    stdout_lines = []
    stderr_lines = []

    try:
        ap = AeroProblem(
            name="cmplxfoil_coords_run",
            alpha=float(alpha_deg),
            mach=float(mach),
            reynolds=float(reynolds),
            reynoldsLength=1.0,
            T=288.15,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd", "cm"],
        )

        coords = _build_airfoil_coords_array(x, yu, yl)
        use_complex = np.iscomplexobj(coords)

        with _suppress_native_output(enabled=quiet):
            solver = _get_cached_cmplxfoil_solver(
                airfoil_dat=airfoil_dat,
                session_key=session_key,
                workdir=workdir,
                xtr_upper=xtr_upper,
                xtr_lower=xtr_lower,
                xfoil_iter=xfoil_iter,
            )

            if use_complex:
                solver.setCoordinatesComplex(coords)
                solver(ap, useComplex=True, deriv=True)
                funcs_raw = solver.funcsComplex[ap.name]
                slice_data = solver.sliceDataComplex[ap.name]
            else:
                solver.setCoordinates(coords)
                solver(ap, useComplex=False, deriv=False)
                funcs_raw = solver.funcs[ap.name]
                slice_data = solver.sliceData[ap.name]

            fail_info = {}
            solver.checkSolutionFailure(ap, fail_info)
            fail_flag = bool(fail_info.get("fail", False))

        cp_data = None
        if slice_data is not None:
            cp_data = _build_combined_cp_data(slice_data)

        polar = None
        if not fail_flag:
            polar = {
                "alpha": float(alpha_deg),
                "CL": funcs_raw["cl"],
                "CD": funcs_raw["cd"],
                "CM": funcs_raw["cm"],
            }

        success = (not fail_flag) and (polar is not None) and (cp_data is not None)

        if fail_flag:
            stderr_lines.append("CMPLXFOIL reported solution failure.")
        if cp_data is None:
            stderr_lines.append("Slice data not found or invalid.")

        return {
            "success": success,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "returncode": 0 if success else -1,
            "polar": polar,
            "cp_data": cp_data,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]),
        }

    except Exception as e:
        stderr_lines.append(str(e))
        return {
            "success": False,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "returncode": -1,
            "polar": None,
            "cp_data": None,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]) if workdir.exists() else [],
        }
    
def run_cmplxfoil(
    airfoil_dat,
    alpha_deg,
    reynolds,
    xfoil_iter,
    timeout,
    working_dir,
    xtr_upper=None,
    xtr_lower=None,
    mach=None,
):
    """
    Wrapper CMPLXFOIL compatibile con run_xfoil(...), basato su file .dat.
    """
    from settings import SETTINGS

    xfoil_cfg = SETTINGS.get("xfoil", {})
    aero_cfg = SETTINGS.get("aero", {})
    quiet = bool(aero_cfg.get("quiet", True))

    if xtr_upper is None:
        xtr_upper = float(xfoil_cfg.get("xtr_upper", 1.0))
    else:
        xtr_upper = float(xtr_upper)

    if xtr_lower is None:
        xtr_lower = float(xfoil_cfg.get("xtr_lower", 1.0))
    else:
        xtr_lower = float(xtr_lower)

    if mach is None:
        mach = float(xfoil_cfg.get("mach", 0.001))
    else:
        mach = float(mach)

    if not (0.0 <= xtr_upper <= 1.0):
        raise ValueError(f"xtr_upper must be in [0, 1], got {xtr_upper}")
    if not (0.0 <= xtr_lower <= 1.0):
        raise ValueError(f"xtr_lower must be in [0, 1], got {xtr_lower}")

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    stdout_lines = []
    stderr_lines = []

    try:
        local_airfoil = _prepare_airfoil_for_cmplxfoil(airfoil_dat, workdir)

        aero_options = {
            "maxIters": int(xfoil_iter),
            "printRealConvergence": False,
            "printComplexConvergence": False,
            "writeSolution": False,
            "writeSliceFile": False,
            "writeCoordinates": False,
            "plotAirfoil": False,
            "outputDirectory": str(workdir),
            "xTrip": np.array([xtr_upper, xtr_lower], dtype=float),
        }

        with _suppress_native_output(enabled=quiet):
            solver = CMPLXFOIL(str(local_airfoil), options=aero_options)

            ap = AeroProblem(
                name="cmplxfoil_run",
                alpha=float(alpha_deg),
                mach=float(mach),
                reynolds=float(reynolds),
                reynoldsLength=1.0,
                T=288.15,
                areaRef=1.0,
                chordRef=1.0,
                evalFuncs=["cl", "cd", "cm"],
            )

            funcs = {}
            solver(ap)
            solver.evalFunctions(ap, funcs, evalFuncs=["cl", "cd", "cm"])
            solver.checkSolutionFailure(ap, funcs)

            fail_flag = bool(funcs.get("fail", False))

            slice_base = workdir / "slice"
            solver.writeSlice(str(slice_base))

        slice_data = _load_slice_data(slice_base.with_suffix(".pkl"), ap.name)
        cp_data = None
        if slice_data is not None:
            cp_data = _build_combined_cp_data(slice_data)

        polar = None
        if not fail_flag:
            polar = {
                "alpha": float(alpha_deg),
                "CL": float(funcs[f"{ap.name}_cl"]),
                "CD": float(funcs[f"{ap.name}_cd"]),
                "CM": float(funcs[f"{ap.name}_cm"]),
            }

        success = (not fail_flag) and (polar is not None) and (cp_data is not None)

        if fail_flag:
            stderr_lines.append("CMPLXFOIL reported solution failure.")
        if cp_data is None:
            stderr_lines.append("Slice data not found or invalid.")

        return {
            "success": success,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "returncode": 0 if success else -1,
            "polar": polar,
            "cp_data": cp_data,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]),
        }

    except Exception as e:
        stderr_lines.append(str(e))
        return {
            "success": False,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "returncode": -1,
            "polar": None,
            "cp_data": None,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]) if workdir.exists() else [],
        }
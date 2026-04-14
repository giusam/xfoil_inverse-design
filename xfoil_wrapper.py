import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def clean_workdir(path):
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def read_xfoil_polar(path):
    path = Path(path)
    if not path.exists():
        return None

    lines = path.read_text(errors="ignore").splitlines()
    data_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(parts[i]) for i in range(5)]
            data_lines.append(vals)
        except Exception:
            continue

    if not data_lines:
        return None

    alpha, cl, cd, _, cm = data_lines[-1]
    return {
        "alpha": alpha,
        "CL": cl,
        "CD": cd,
        "CM": cm,
    }


def read_xfoil_cp(path):
    path = Path(path)
    if not path.exists():
        return None

    lines = path.read_text(errors="ignore").splitlines()
    rows = []

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue

        try:
            x = float(parts[0])
            cp = float(parts[1])
            rows.append([x, cp])
        except Exception:
            continue

    if not rows:
        return None

    arr = np.array(rows, dtype=float)
    return {
        "x": arr[:, 0],
        "cp": arr[:, 1],
    }


def run_xfoil(airfoil_dat, alpha_deg, reynolds, xfoil_iter, timeout, working_dir):
    import signal

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    local_airfoil = workdir / "airfoil.dat"
    local_polar = workdir / "polar.txt"
    local_cp = workdir / "cp.txt"

    if local_polar.exists():
        local_polar.unlink()
    if local_cp.exists():
        local_cp.unlink()

    local_airfoil.write_text(Path(airfoil_dat).read_text())

    xfoil_script = (
        f"LOAD {local_airfoil.name}\n"
        "PANE\n"
        "OPER\n"
        f"VISC {reynolds}\n"
        f"ITER {xfoil_iter}\n"
        "PACC\n"
        f"{local_polar.name}\n"
        "\n"
        f"ALFA {alpha_deg}\n"
        "CPWR\n"
        f"{local_cp.name}\n"
        "\n"
        "QUIT\n"
    )

    proc = None

    try:
        proc = subprocess.Popen(
            ["xvfb-run", "-a", "xfoil"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            env=os.environ.copy(),
            start_new_session=True,
        )

        stdout, stderr = proc.communicate(input=xfoil_script, timeout=timeout)

    except subprocess.TimeoutExpired as e:
        stdout = ""
        stderr = str(e)

        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            try:
                stdout_kill, stderr_kill = proc.communicate(timeout=5)
                stdout = stdout_kill or stdout
                stderr = stderr_kill or stderr
            except Exception:
                pass

        return {
            "success": False,
            "stdout": stdout if stdout else "XFOIL TIMEOUT EXPIRED",
            "stderr": stderr,
            "returncode": -1,
            "polar": None,
            "cp_data": None,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]),
        }

    except Exception as e:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "polar": None,
            "cp_data": None,
            "workdir_files": sorted([p.name for p in workdir.iterdir()]),
        }

    parsed_polar = read_xfoil_polar(local_polar)
    parsed_cp = read_xfoil_cp(local_cp)

    return {
        "success": parsed_polar is not None and parsed_cp is not None,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "polar": parsed_polar,
        "cp_data": parsed_cp,
        "workdir_files": sorted([p.name for p in workdir.iterdir()]),
    }
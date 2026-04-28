import csv
from pathlib import Path

import numpy as np

from settings import SETTINGS
from geometry import apply_hicks_henne_deformation, write_dat
from xfoil_wrapper import run_xfoil
from cmplxfoil_wrapper import run_cmplxfoil
from aero_wrapper import run_aero
from cp_utils import split_upper_lower_cp_from_x
from objective import total_cp_error
from spring_reallocation import split_a_by_sides


def run_target_aero(*args, **kwargs):
    target_backend = SETTINGS.get("aero", {}).get(
        "target_backend",
        SETTINGS.get("aero", {}).get("backend", "xfoil"),
    )
    target_backend = str(target_backend).strip().lower()

    if target_backend == "cmplxfoil":
        return run_cmplxfoil(*args, **kwargs)
    if target_backend == "xfoil":
        return run_xfoil(*args, **kwargs)

    raise ValueError(f"Unknown target backend: {target_backend}")


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_array(arr):
    return np.array2string(np.asarray(arr, dtype=float), precision=6, separator=", ")


def safe_float(value):
    if value is None:
        return float("nan")
    try:
        return float(np.real(value))
    except Exception:
        return float("nan")


def polar_metric(out, name):
    if out is None:
        return float("nan")
    opt_res = out.get("opt_res")
    if opt_res is None:
        return float("nan")
    polar = opt_res.get("polar")
    if polar is None:
        return float("nan")
    return safe_float(polar.get(name))


def write_polar_from_res(res, dst_path):
    if res is None or res.get("polar") is None:
        return

    polar = res["polar"]
    alpha = safe_float(polar.get("alpha"))
    cl = safe_float(polar.get("CL"))
    cd = safe_float(polar.get("CD"))
    cm = safe_float(polar.get("CM"))

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("alpha CL CD CM\n")
        f.write(f"{alpha:.10e} {cl:.10e} {cd:.10e} {cm:.10e}\n")


def write_cp_from_res(res, dst_path):
    if res is None or res.get("cp_data") is None:
        return

    cp_x = np.asarray(res["cp_data"]["x"], dtype=float)
    cp_v = np.real(np.asarray(res["cp_data"]["cp"]))
    data = np.column_stack([cp_x, cp_v])
    np.savetxt(dst_path, data, header="x cp", comments="")


def save_out_bundle(bundle_dir, x, out, name):
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if out is None:
        return

    if "yu_opt" in out and "yl_opt" in out:
        write_dat(
            bundle_dir / "optimized_airfoil.dat",
            x,
            out["yu_opt"],
            out["yl_opt"],
            name=name,
        )

    opt_res = out.get("opt_res")
    if opt_res is not None:
        write_polar_from_res(opt_res, bundle_dir / "polar.txt")
        write_cp_from_res(opt_res, bundle_dir / "cp.txt")

    with open(bundle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"name = {name}\n")
        f.write(f"err_opt = {safe_float(out.get('err_opt')):.10e}\n")
        f.write(f"CL = {polar_metric(out, 'CL'):.10e}\n")
        f.write(f"CD = {polar_metric(out, 'CD'):.10e}\n")
        f.write(f"CM = {polar_metric(out, 'CM'):.10e}\n")
        if "upper_centers" in out:
            f.write(f"upper_centers = {format_array(out['upper_centers'])}\n")
        if "lower_centers" in out:
            f.write(f"lower_centers = {format_array(out['lower_centers'])}\n")
        if "a_opt" in out:
            f.write(f"a_opt = {format_array(out['a_opt'])}\n")


def evaluate_geometry_state(
    x,
    yu_init,
    yl_init,
    cp_target,
    upper_centers,
    lower_centers,
    a_vec,
    label,
    workdir,
    include_aero_calls=False,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    upper_centers = np.asarray(upper_centers, dtype=float)
    lower_centers = np.asarray(lower_centers, dtype=float)
    a_vec = np.asarray(a_vec, dtype=float)

    a_upper, a_lower = split_a_by_sides(
        a_vec,
        len(upper_centers),
        len(lower_centers),
    )

    yu, yl = apply_hicks_henne_deformation(
        x=x,
        yu_base=yu_init,
        yl_base=yl_init,
        a_upper=a_upper,
        a_lower=a_lower,
        upper_centers=upper_centers,
        lower_centers=lower_centers,
        power=SETTINGS["optimization"]["hh_power"],
    )

    airfoil_dat = workdir / f"{label}_airfoil.dat"
    write_dat(airfoil_dat, x, yu, yl, name=label.upper())

    res = run_aero(
        airfoil_dat=airfoil_dat,
        alpha_deg=SETTINGS["xfoil"]["alpha"],
        reynolds=SETTINGS["xfoil"]["Re"],
        xfoil_iter=SETTINGS["xfoil"]["xfoil_iter"],
        timeout=SETTINGS["xfoil"]["timeout"],
        working_dir=workdir / f"{label}_run",
    )
    if not res["success"]:
        out = {
            "success": False,
            "res": res,
            "yu": yu,
            "yl": yl,
            "cp_opt": None,
            "err": float("nan"),
        }
        if include_aero_calls:
            out["aero_calls"] = 1
        return out

    cp_candidate = split_upper_lower_cp_from_x(
        res["cp_data"]["x"],
        res["cp_data"]["cp"],
    )
    err = total_cp_error(cp_target, cp_candidate)
    out = {
        "success": True,
        "res": res,
        "yu": yu,
        "yl": yl,
        "cp_opt": cp_candidate,
        "err": float(err),
    }
    if include_aero_calls:
        out["aero_calls"] = 1
    return out

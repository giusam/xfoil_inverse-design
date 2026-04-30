import math
import matplotlib.pyplot as plt
import numpy as np

def _set_log_error_axis():
    plt.yscale("log")


def _apply_log_grid(ax):
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)

def plot_multiple_initial_geometries(x, yu_target, yl_target, initial_geometries, title="Initial geometries"):
    plt.figure(figsize=(10, 4))

    plt.plot(x, yu_target, "k-", lw=2, label="target upper")
    plt.plot(x, yl_target, "k-", lw=2, label="target lower")

    colors = plt.cm.tab10.colors
    for i, (yu_init, yl_init, label) in enumerate(initial_geometries):
        color = colors[i % len(colors)]
        plt.plot(x, yu_init, "-", lw=1.5, label=f"{label} upper", color=color)
        plt.plot(x, yl_init, "-", lw=1.5, label=f"{label} lower", color=color)

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_initial_geometries_subplots(
    x,
    yu_target,
    yl_target,
    initial_geometries,
    title="Initial geometries",
):
    n = len(initial_geometries)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for i, (yu_init, yl_init, label) in enumerate(initial_geometries):
        ax = axes[i]
        color = colors[i % len(colors)]

        ax.plot(x, yu_target, "k-", lw=2, label="target")
        ax.plot(x, yl_target, "k-", lw=2)

        ax.plot(x, yu_init, "-", lw=1.5, color=color, label=label)
        ax.plot(x, yl_init, "-", lw=1.5, color=color)

        ax.set_title(label)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_geometry(x, yu, yl, title="Geometry"):
    plt.figure(figsize=(10, 4))
    plt.plot(x, yu, label="upper")
    plt.plot(x, yl, label="lower")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_geometry_comparison(x, yu_target, yl_target, yu_init, yl_init, title="Geometry comparison"):
    plt.figure(figsize=(10, 4))
    plt.plot(x, yu_target, "k-", lw=2, label="target upper")
    plt.plot(x, yl_target, "k-", lw=2, label="target lower")
    plt.plot(x, yu_init, "r--", lw=2, label="init upper")
    plt.plot(x, yl_init, "r--", lw=2, label="init lower")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cp(cp_split, title="Cp"):
    plt.figure(figsize=(8, 4))
    plt.plot(cp_split["upper"]["x"], cp_split["upper"]["cp"], "o-", label="Cp upper")
    plt.plot(cp_split["lower"]["x"], cp_split["lower"]["cp"], "o-", label="Cp lower")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cp_comparison(cp_target, cp_init, title="Cp comparison"):
    plt.figure(figsize=(8, 4))
    plt.plot(cp_target["upper"]["x"], cp_target["upper"]["cp"], "k-", lw=2, label="target upper")
    plt.plot(cp_target["lower"]["x"], cp_target["lower"]["cp"], "k-", lw=2, label="target lower")
    plt.plot(cp_init["upper"]["x"], cp_init["upper"]["cp"], "r--", lw=2, label="init upper")
    plt.plot(cp_init["lower"]["x"], cp_init["lower"]["cp"], "r--", lw=2, label="init lower")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cp_subplots(cp_target, cp_list, title="Cp comparison"):
    n = len(cp_list)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for i, (cp_init, label) in enumerate(cp_list):
        ax = axes[i]
        color = colors[i % len(colors)]

        ax.plot(cp_target["upper"]["x"], cp_target["upper"]["cp"], "k-", lw=2, label="target")
        ax.plot(cp_target["lower"]["x"], cp_target["lower"]["cp"], "k-", lw=2)

        ax.plot(cp_init["upper"]["x"], cp_init["upper"]["cp"], "-", lw=1.5, color=color, label=label)
        ax.plot(cp_init["lower"]["x"], cp_init["lower"]["cp"], "-", lw=1.5, color=color)

        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title(label)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_3way_geometry_comparison(
    x,
    yu_target,
    yl_target,
    yu_init,
    yl_init,
    yu_opt,
    yl_opt,
    title="Geometry 3-Way Comparison",
    savepath=None,
):
    plt.figure(figsize=(10, 4))

    plt.plot(x, yu_target, "k-", lw=2.5, label="Target upper")
    plt.plot(x, yl_target, "k-", lw=2.5, label="Target lower")

    plt.plot(x, yu_init, "b--", lw=1.5, label="Initial upper")
    plt.plot(x, yl_init, "b--", lw=1.5, label="Initial lower")

    plt.plot(x, yu_opt, "r-", lw=2, label="Optimized upper")
    plt.plot(x, yl_opt, "r-", lw=2, label="Optimized lower")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(title)
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_3way_cp_comparison(
    cp_target,
    cp_init,
    cp_opt,
    title="Cp 3-Way Comparison",
    savepath=None,
):
    plt.figure(figsize=(8, 4))

    plt.plot(cp_target["upper"]["x"], cp_target["upper"]["cp"], "k-", lw=2.5, label="Target upper")
    plt.plot(cp_target["lower"]["x"], cp_target["lower"]["cp"], "k-", lw=2.5, label="Target lower")

    plt.plot(cp_init["upper"]["x"], cp_init["upper"]["cp"], "b--", lw=1.5, label="Initial upper")
    plt.plot(cp_init["lower"]["x"], cp_init["lower"]["cp"], "b--", lw=1.5, label="Initial lower")

    plt.plot(cp_opt["upper"]["x"], cp_opt["upper"]["cp"], "r-", lw=2, label="Optimized upper")
    plt.plot(cp_opt["lower"]["x"], cp_opt["lower"]["cp"], "r-", lw=2, label="Optimized lower")

    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title(title)
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_error_vs_seed(results, savepath=None):
    seeds = [r["seed"] for r in results]
    static_err = np.array([r["static_err"] for r in results], dtype=float)
    adaptive_err = np.array([r["adaptive_err"] for r in results], dtype=float)
    adaptive_best_err = np.array([r["adaptive_best_err"] for r in results], dtype=float)

    plt.figure(figsize=(8, 4))
    if np.any(np.isfinite(static_err)):
        plt.plot(seeds, static_err, "o-", lw=2, label="Static")
    plt.plot(seeds, adaptive_err, "s-", lw=2, label="Adaptive final")
    plt.plot(seeds, adaptive_best_err, "^-", lw=2, label="Adaptive best")

    _set_log_error_axis()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Seed")
    plt.ylabel("Cp error")
    plt.title("Cp error vs seed")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_error_mean_std(results, savepath=None):
    series = [
        ("Static", np.array([r["static_err"] for r in results], dtype=float)),
        ("Adaptive final", np.array([r["adaptive_err"] for r in results], dtype=float)),
        ("Adaptive best", np.array([r["adaptive_best_err"] for r in results], dtype=float)),
    ]

    labels = []
    means = []
    stds = []
    for label, values in series:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        labels.append(label)
        means.append(np.mean(finite))
        stds.append(np.std(finite))

    x = np.arange(len(labels))

    plt.figure(figsize=(7, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    _set_log_error_axis()
    plt.ylabel("Cp error")
    plt.title("Mean Cp error ± std across seeds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _finish_convergence_plot(xlabel, ylabel, title, savepath=None):
    ax = plt.gca()
    _set_log_error_axis()
    _apply_log_grid(ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _valid_error_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _plot_static_function_history(initial_err, function_eval_history, x_key, xlabel, title, savepath=None):
    xs = [0]
    ys = [float(initial_err)]

    for item in function_eval_history:
        if item.get("status") != "OK":
            continue
        yval = _valid_error_value(item.get("objective_base", item.get("objective")))
        if yval is None:
            continue
        xs.append(int(item[x_key]))
        ys.append(yval)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, "-", lw=2, label="STATIC")
    _finish_convergence_plot(
        xlabel=xlabel,
        ylabel="Cp error",
        title=title,
        savepath=savepath,
    )


def _plot_static_gradient_history(initial_err, gradient_eval_history, x_key, xlabel, title, savepath=None):
    xs = [0]
    ys = [float(initial_err)]

    for item in gradient_eval_history:
        if item.get("status") != "OK":
            continue
        yval = _valid_error_value(item.get("objective_base_at_grad_point"))
        if yval is None:
            continue
        xs.append(int(item[x_key]))
        ys.append(yval)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, "-", lw=2, label="STATIC")
    _finish_convergence_plot(
        xlabel=xlabel,
        ylabel="Cp error",
        title=title,
        savepath=savepath,
    )


def plot_error_vs_function_evals_static(initial_err, function_eval_history, savepath=None):
    _plot_static_function_history(
        initial_err=initial_err,
        function_eval_history=function_eval_history,
        x_key="cumulative_function_evals",
        xlabel="Function evaluations",
        title="Static Cp error vs function evaluations",
        savepath=savepath,
    )


def plot_error_vs_gradient_evals_static(initial_err, gradient_eval_history, savepath=None):
    _plot_static_gradient_history(
        initial_err=initial_err,
        gradient_eval_history=gradient_eval_history,
        x_key="cumulative_gradient_evals",
        xlabel="Gradient evaluations",
        title="Static Cp error vs gradient evaluations",
        savepath=savepath,
    )


def _plot_adaptive_history_full_opt(
    initial_err,
    history_full_opt,
    history_key,
    y_key,
    local_x_key,
    offset_end_key,
    xlabel,
    title,
    label,
    savepath=None,
):
    xs = [0]
    ys = [float(initial_err)]
    offset_start = 0
    refine_positions = []
    refine_labels = []

    for level in history_full_opt:
        for item in level.get(history_key, []):
            if item.get("status") != "OK":
                continue
            yval = _valid_error_value(item.get(y_key))
            if yval is None:
                continue
            xs.append(int(offset_start + item[local_x_key]))
            ys.append(yval)

        level_end = int(level.get(offset_end_key, offset_start))
        refine_positions.append(level_end)
        refine_labels.append(level.get("ndv_total"))
        offset_start = level_end

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, "-", lw=2, label=label)

    ax = plt.gca()
    for idx, (x_refine, ndv) in enumerate(zip(refine_positions, refine_labels)):
        plt.axvline(x=x_refine, color="gray", alpha=0.3, linestyle=":")
        label_text = f"ndv={ndv}" if idx == 0 else f"{ndv}"
        ax.text(
            x_refine,
            0.985,
            label_text,
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            rotation=0,
            ha="center",
            va="top",
        )

    _finish_convergence_plot(
        xlabel=xlabel,
        ylabel="Cp error",
        title=title,
        savepath=savepath,
    )


def plot_error_vs_function_evals_adaptive(initial_err, history_full_opt, savepath=None, label="Adaptive Cp error"):
    _plot_adaptive_history_full_opt(
        initial_err=initial_err,
        history_full_opt=history_full_opt,
        history_key="function_eval_history",
        y_key="objective_base",
        local_x_key="cumulative_function_evals",
        offset_end_key="function_eval_offset_end",
        xlabel="Function evaluations",
        title="Adaptive Cp error vs function evaluations",
        label=label,
        savepath=savepath,
    )


def plot_error_vs_gradient_evals_adaptive(initial_err, history_full_opt, savepath=None, label="Adaptive Cp error"):
    _plot_adaptive_history_full_opt(
        initial_err=initial_err,
        history_full_opt=history_full_opt,
        history_key="gradient_eval_history",
        y_key="objective_base_at_grad_point",
        local_x_key="cumulative_gradient_evals",
        offset_end_key="gradient_eval_offset_end",
        xlabel="Gradient evaluations",
        title="Adaptive Cp error vs gradient evaluations",
        label=label,
        savepath=savepath,
    )


def plot_error_vs_function_evals_compare(series_list, savepath=None):
    plt.figure(figsize=(9, 5))

    for series in series_list:
        x = list(series.get("x", []))
        y = list(series.get("y", []))
        if len(x) == 0 or len(y) == 0:
            continue
        plt.plot(x, y, "-", lw=2, label=series.get("label", ""))

    _finish_convergence_plot(
        xlabel="Function evaluations",
        ylabel="Cp error",
        title="Cp error vs function evaluations",
        savepath=savepath,
    )



def plot_error_vs_gradient_evals_compare(series_list, savepath=None):
    plt.figure(figsize=(9, 5))

    for series in series_list:
        x = list(series.get("x", []))
        y = list(series.get("y", []))
        if len(x) == 0 or len(y) == 0:
            continue
        plt.plot(x, y, "-", lw=2, label=series.get("label", ""))

    _finish_convergence_plot(
        xlabel="Gradient evaluations",
        ylabel="Cp error",
        title="Cp error vs gradient evaluations",
        savepath=savepath,
    )


def plot_method_airfoil_comparison(x, yu_target, yl_target, yu_init, yl_init, method_results, savepath):
    plt.figure(figsize=(10, 4))
    plt.plot(x, yu_target, "k-", lw=2.0, label="TARGET")
    plt.plot(x, yl_target, "k-", lw=2.0)
    plt.plot(x, yu_init, color="0.55", lw=1.3, ls="--", label="INITIAL")
    plt.plot(x, yl_init, color="0.55", lw=1.3, ls="--")
    colors = plt.cm.tab10.colors
    for i, (name, bundle) in enumerate(method_results.items()):
        yu = bundle.get("yu_final")
        yl = bundle.get("yl_final")
        if yu is None or yl is None:
            continue
        color = colors[i % len(colors)]
        plt.plot(x, yu, lw=1.5, color=color, label=name)
        plt.plot(x, yl, lw=1.5, color=color)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil comparison")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


METHOD_LABELS = {
    "STATIC": "Static",
    "ADAPT_GRAD": "ADAPT-GRAD",
    "GRAD_SPRING_FINAL": "ADAPT-GRAD + final spring",
    "GRAD_SPRING_PERIODIC": "ADAPT-GRAD + periodic spring",
}


def _method_label(name):
    return METHOD_LABELS.get(str(name), str(name).replace("_", " "))


def plot_single_method_airfoil(x, yu_target, yl_target, yu_init, yl_init, bundle, savepath):
    method_name = bundle.get("method_name", "")
    label = _method_label(method_name)
    yu = bundle.get("yu_final")
    yl = bundle.get("yl_final")
    plt.figure(figsize=(10, 4))
    plt.plot(x, yu_init, color="0.55", lw=1.3, ls="--", label="Initial")
    plt.plot(x, yl_init, color="0.55", lw=1.3, ls="--")
    plt.plot(x, yu_target, "k-", lw=2.0, label="Target")
    plt.plot(x, yl_target, "k-", lw=2.0)
    if yu is not None and yl is not None and len(yu) > 0 and len(yl) > 0:
        plt.plot(x, yu, color="tab:blue", lw=1.7, label=label)
        plt.plot(x, yl, color="tab:blue", lw=1.7)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"{label} airfoil")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_cp_side(cp, side, style, color, label=None, lw=1.5):
    if cp is None or side not in cp:
        return
    x = np.asarray(cp[side].get("x", []), dtype=float)
    y = np.real(np.asarray(cp[side].get("cp", [])))
    if x.size == 0 or y.size == 0:
        return
    plt.plot(x, y, style, color=color, lw=lw, label=label)


def plot_single_method_cp(cp_target, bundle, savepath):
    method_name = bundle.get("method_name", "")
    label = _method_label(method_name)
    cp_final = bundle.get("cp_final")
    plt.figure(figsize=(10, 5))
    _plot_cp_side(cp_target, "upper", "-", "k", "Target upper", lw=2.0)
    _plot_cp_side(cp_target, "lower", "--", "k", "Target lower", lw=2.0)
    _plot_cp_side(cp_final, "upper", "-", "tab:blue", f"{label} upper", lw=1.6)
    _plot_cp_side(cp_final, "lower", "--", "tab:blue", f"{label} lower", lw=1.6)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title(f"{label} Cp")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_method_cp_comparison(cp_target, cp_init, method_results, savepath):
    plt.figure(figsize=(10, 5))
    _plot_cp_side(cp_target, "upper", "-", "k", "TARGET upper", lw=2.0)
    _plot_cp_side(cp_target, "lower", "--", "k", "TARGET lower", lw=2.0)
    _plot_cp_side(cp_init, "upper", "-", "0.6", "INITIAL upper", lw=1.2)
    _plot_cp_side(cp_init, "lower", "--", "0.6", "INITIAL lower", lw=1.2)
    colors = plt.cm.tab10.colors
    for i, (name, bundle) in enumerate(method_results.items()):
        cp = bundle.get("cp_final")
        color = colors[i % len(colors)]
        _plot_cp_side(cp, "upper", "-", color, f"{name} upper")
        _plot_cp_side(cp, "lower", "--", color, f"{name} lower")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title("Cp comparison")
    plt.legend(loc="best", fontsize="x-small", ncol=2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_single_method_centers(bundle, savepath):
    method_name = bundle.get("method_name", "")
    label = _method_label(method_name)
    upper = np.asarray(bundle.get("upper_centers_final", []), dtype=float)
    lower = np.asarray(bundle.get("lower_centers_final", []), dtype=float)
    plt.figure(figsize=(10, 2.8))
    if upper.size > 0:
        plt.scatter(upper, np.ones_like(upper), marker="^", s=42, color="tab:blue", label="upper final")
    if lower.size > 0:
        plt.scatter(lower, np.zeros_like(lower), marker="v", s=42, color="tab:orange", label="lower final")
    plt.xlim(0.0, 1.0)
    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], ["lower", "upper"])
    plt.grid(True, axis="x", alpha=0.25)
    plt.xlabel("center x/c")
    plt.title(f"{label} final Hicks-Henne centers")
    if upper.size > 0 or lower.size > 0:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_method_centers_comparison(method_results, savepath):
    plt.figure(figsize=(10, max(3.0, 0.45 * max(len(method_results), 1) + 1.5)))
    ytick_pos = []
    ytick_lab = []
    row = 0
    for name, bundle in method_results.items():
        upper = np.asarray(bundle.get("upper_centers_final", []), dtype=float)
        lower = np.asarray(bundle.get("lower_centers_final", []), dtype=float)
        if upper.size == 0 and lower.size == 0:
            continue
        plt.scatter(upper, np.full_like(upper, row + 0.12), marker="^", s=28, label="upper" if row == 0 else None)
        plt.scatter(lower, np.full_like(lower, row - 0.12), marker="v", s=28, label="lower" if row == 0 else None)
        ytick_pos.append(row)
        ytick_lab.append(name)
        row += 1
    plt.xlim(0.0, 1.0)
    plt.yticks(ytick_pos, ytick_lab)
    plt.grid(True, axis="x", alpha=0.25)
    plt.xlabel("center x/c")
    plt.title("Final Hicks-Henne centers")
    if row > 0:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_method_error_comparison(method_results, savepath):
    names = list(method_results.keys())
    values = [float(method_results[name].get("err_final", np.nan)) for name in names]
    labels = [_method_label(name) for name in names]
    plt.figure(figsize=(max(7, 1.2 * len(names)), 4))
    plt.bar(labels, values, color=plt.cm.tab10.colors[: len(names)])
    finite = [v for v in values if np.isfinite(v) and v > 0.0]
    if finite and max(finite) / max(min(finite), 1.0e-300) > 20.0:
        plt.yscale("log")
    plt.ylabel("Final Cp error")
    plt.title("Final Cp error comparison")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def _extract_history_curve(bundle):
    curve = bundle.get("history_curve", [])
    clean = []
    for item in curve:
        try:
            x = float(item.get("x"))
            j = float(item.get("J"))
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(j) or j <= 0.0:
            continue
        clean.append(
            {
                "x": x,
                "J": j,
                "ndv": item.get("ndv", ""),
                "stage": item.get("stage", ""),
                "label": item.get("label", ""),
            }
        )
    if clean:
        return clean

    history = bundle.get("history")
    if history:
        print(f"[warning] history_curve missing for {bundle.get('method_name', '')}; using synthetic x fallback.")
        for idx, item in enumerate(history):
            try:
                ndv, err = item[0], float(item[1])
            except Exception:
                continue
            if np.isfinite(err) and err > 0.0:
                clean.append({"x": float(idx + 1), "J": err, "ndv": ndv, "stage": "adapt", "label": f"ndv={ndv}"})
    return clean


def _extract_history_events(bundle):
    events = []
    for item in bundle.get("history_events", []):
        try:
            x = float(item.get("x"))
        except Exception:
            continue
        if not np.isfinite(x):
            continue
        events.append(
            {
                "x": x,
                "ndv": item.get("ndv", ""),
                "stage": item.get("stage", ""),
                "label": item.get("label", ""),
            }
        )
    return events


def _draw_history_events(ax, events, show_labels=True):
    seen_level_labels = 0
    for event in events:
        x = float(event["x"])
        stage = str(event.get("stage", ""))
        label = str(event.get("label", ""))
        if stage == "spring":
            ax.axvline(x=x, color="tab:orange", alpha=0.55, linestyle="--", lw=1.1)
            if show_labels:
                ax.text(
                    x,
                    0.05,
                    label,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    rotation=90,
                    ha="right",
                    va="bottom",
                    color="tab:orange",
                )
        else:
            ax.axvline(x=x, color="gray", alpha=0.22, linestyle=":", lw=0.9)
            if show_labels:
                ndv = event.get("ndv", "")
                if ndv != "":
                    txt = f"ndv={ndv}" if seen_level_labels == 0 else f"{ndv}"
                    ax.text(
                        x,
                        0.985,
                        txt,
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                        rotation=0,
                        ha="center",
                        va="top",
                        color="0.35",
                    )
                    seen_level_labels += 1


def plot_single_method_history(bundle, savepath):
    method_name = bundle.get("method_name", "")
    label = _method_label(method_name)
    curve = _extract_history_curve(bundle)
    events = _extract_history_events(bundle)
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    if curve:
        xs = np.asarray([p["x"] for p in curve], dtype=float)
        ys = np.asarray([p["J"] for p in curve], dtype=float)
        ax.plot(xs, ys, "-", lw=2.0, color="tab:blue", label=label)
        spring_x = [p["x"] for p in curve if p.get("stage") == "spring"]
        spring_y = [p["J"] for p in curve if p.get("stage") == "spring"]
        if spring_x:
            ax.scatter(spring_x, spring_y, marker="D", s=42, color="tab:orange", zorder=3, label="spring")
        _draw_history_events(ax, events, show_labels=True)
    else:
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
    ax.set_yscale("log")
    _apply_log_grid(ax)
    ax.set_xlabel("Cumulative gradient evaluations")
    ax.set_ylabel("Cp error")
    ax.set_title(f"{label} Cp error vs gradient evaluations")
    if curve:
        ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_history_comparison(method_results, savepath):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    any_series = False

    preferred_order = [
        "STATIC",
        "GRAD_SPRING_FINAL",
        "GRAD_SPRING_PERIODIC",
        "ADAPT_GRAD",
    ]
    ordered_names = [name for name in preferred_order if name in method_results]
    ordered_names += [name for name in method_results if name not in ordered_names]

    styles = {
        "STATIC": {
            "color": "0.35",
            "lw": 1.6,
            "alpha": 0.75,
            "linestyle": "-",
            "zorder": 1,
        },
        "GRAD_SPRING_FINAL": {
            "color": "tab:green",
            "lw": 1.8,
            "alpha": 0.80,
            "linestyle": "-",
            "zorder": 3,
        },
        "GRAD_SPRING_PERIODIC": {
            "color": "tab:red",
            "lw": 1.8,
            "alpha": 0.80,
            "linestyle": "-",
            "zorder": 4,
        },
        "ADAPT_GRAD": {
            "color": "tab:orange",
            "lw": 3.0,
            "alpha": 1.00,
            "linestyle": "-",
            "zorder": 10,
        },
    }
    default_style = {
        "color": None,
        "lw": 1.8,
        "alpha": 0.85,
        "linestyle": "-",
        "zorder": 2,
    }

    for name in ordered_names:
        bundle = method_results[name]
        curve = _extract_history_curve(bundle)
        style = styles.get(name, default_style)

        if curve:
            xs = np.asarray([p["x"] for p in curve], dtype=float)
            ys = np.asarray([p["J"] for p in curve], dtype=float)
            ax.plot(
                xs,
                ys,
                linestyle=style["linestyle"],
                lw=style["lw"],
                color=style["color"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=_method_label(name),
            )
            spring_x = [p["x"] for p in curve if p.get("stage") == "spring"]
            spring_y = [p["J"] for p in curve if p.get("stage") == "spring"]
            if spring_x:
                ax.scatter(
                    spring_x,
                    spring_y,
                    marker="D",
                    s=36,
                    color=style["color"],
                    alpha=style["alpha"],
                    zorder=style["zorder"] + 1,
                )
            any_series = True
        elif name == "STATIC":
            err = bundle.get("err_final", np.nan)
            try:
                err = float(err)
            except Exception:
                err = np.nan
            if np.isfinite(err) and err > 0.0:
                ax.axhline(
                    err,
                    color="0.55",
                    linestyle="--",
                    lw=1.0,
                    alpha=0.7,
                    zorder=0,
                    label="Static final J",
                )
    if not any_series:
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
    ax.set_yscale("log")
    _apply_log_grid(ax)
    ax.set_xlabel("Cumulative gradient evaluations")
    ax.set_ylabel("Cp error")
    ax.set_title("Cp error vs gradient evaluations")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_method_history_comparison(method_results, savepath):
    plot_history_comparison(method_results, savepath)

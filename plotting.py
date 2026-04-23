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

import math
import matplotlib.pyplot as plt
import numpy as np

def _set_log_error_axis():
    plt.yscale("log")

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


def plot_error_vs_evals_with_refine(
    initial_err,
    static_calls,
    static_err,
    adaptive_history,
    adaptive_evals,
    savepath=None,
):
    plt.figure(figsize=(8, 4))

    if static_calls is not None and static_err is not None and np.isfinite(static_err):
        plt.plot(
            [0, static_calls],
            [initial_err, static_err],
            "o-",
            label="Static Cp error",
        )

    errs = [e for (_, e) in adaptive_history]
    ndvs = [m for (m, _) in adaptive_history]

    plt.plot(
        [0] + adaptive_evals,
        [initial_err] + errs,
        "o-",
        label="Adaptive Cp error",
    )

    for x, y, ndv in zip(adaptive_evals, errs, ndvs):
        plt.axvline(x=x, linestyle="--", alpha=0.3)
        plt.text(x, y, f"{ndv}", fontsize=8)

    _set_log_error_axis()
    plt.xlabel("XFOIL calls")
    plt.ylabel("Cp error")
    plt.title("Cp error vs evaluations (with refine)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_error_vs_evals_full(
    initial_err,
    static_history,
    adaptive_history_full,
    savepath=None,
):
    plt.figure(figsize=(9, 5))

    static_x = [0]
    static_y = [initial_err]

    for item in static_history:
        if item.get("status") != "OK":
            continue
        yval = item.get("objective_base", item["objective"])
        static_x.append(item["eval"])
        static_y.append(yval)

    if len(static_x) > 1:
        plt.plot(static_x, static_y, "-", lw=2, label="Static Cp error")

    adaptive_x = [0]
    adaptive_y = [initial_err]

    eval_offset_start = 0
    refine_positions = []
    refine_labels = []

    for level in adaptive_history_full:
        level_hist = level["objective_history"]
        ndv = level["ndv_total"]

        for item in level_hist:
            if item.get("status") != "OK":
                continue
            global_eval = eval_offset_start + item["eval"]
            yval = item.get("objective_base", item["objective"])
            adaptive_x.append(global_eval)
            adaptive_y.append(yval)

        if len(level_hist) > 0:
            level_end = eval_offset_start + level_hist[-1]["eval"]
            refine_positions.append(level_end)
            refine_labels.append(ndv)
            eval_offset_start = level["eval_offset_end"]

    plt.plot(adaptive_x, adaptive_y, "-", lw=2, label="Adaptive Cp error")

    for x, ndv in zip(refine_positions, refine_labels):
        plt.axvline(x=x, linestyle="--", alpha=0.3)
        y_text = min(adaptive_y) if len(adaptive_y) > 0 else initial_err
        plt.text(x, y_text, f"{ndv}", fontsize=8, rotation=90, va="bottom")

    ymin = min(adaptive_y + static_y)
    plt.yscale("log")
    plt.ylim(bottom=max(1e-8, ymin * 0.5))

    plt.xlabel("Objective evaluations / XFOIL calls")
    plt.ylabel("Cp error")
    plt.title("Cp error history vs evaluations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

"""Figure: time-to-solution, energy and merit factor against QAOA depth for each LABS instance.
"""

import json
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from labs_qaoa import labs_energies, energy_mf_tts
from labs_styling import style_plot, figsize

# APS-like plotting style with LaTeX text rendering.
mpl.rcParams.update(
    {
        "figure.figsize": (3.4, 2.5),  # default single-column figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    }
)

DATA = Path(__file__).resolve().parents[1] / "data" / "simulations" / "labs"
HERE = Path(__file__).resolve().parent

plot_order = ("F", "I", "trivial", "LR", "LR*", "TQA", "TQA*", "RTS")
_KEY_TO_LABEL = {
    "F": "Fourier",
    "I": "Interp.",
    "LR": "Linear Ramp",
    "LR*": "Linear Ramp*",
    "TQA": "TQA",
    "TQA*": "TQA*",
    "RTS": "Recursive TS",
}
_REF_OPTIMAL = "#2d2d2d"
_REF_UNIFORM = "tab:cyan"
_TRIVIAL_STYLE = {"color": "#777777", "linestyle": ":", "marker": None, "linewidth": 0.9}


def recursive_series(folder, n, e):
    """(depths, energies, merit factors, TTS) for a recursive run (Fourier / Interp.)."""
    data = json.load(open(glob(str(DATA / folder / f"*labs_{n}_*"))[0]))
    depths = [1] + sorted(int(k) for k in data["2"] if k.isdigit())
    rows = [energy_mf_tts(e, np.array((data["1"] if d == 1 else data["2"][str(d)])["optimized_qaoa_angles"]))
            for d in depths]
    energies, mfs, tts = zip(*rows)
    return depths, list(energies), list(mfs), list(tts)


def linear_ramp_series(n, e):
    """(depths, energies, merit factors, TTS) taking the best initial slope at each depth."""
    depths = sorted({int(f.rsplit("depth_", 1)[1].split(".")[0]) for f in glob(str(DATA / "LR" / f"*labs_{n}_*"))})
    depths = [d for d in depths if d <= 50]
    energies, mfs, tts = [], [], []
    for d in depths:
        best = min((json.load(open(f))["1"] for f in glob(str(DATA / "LR" / f"*labs_{n}_*depth_{d}.json"))),
                   key=lambda r: r["energy"])
        E, mf, t = energy_mf_tts(e, np.array(best["optimized_qaoa_angles"]))
        energies.append(E); mfs.append(mf); tts.append(t)
    return depths, energies, mfs, tts


def figure(n):
    e = labs_energies(n)
    series = {"F": recursive_series("F", n, e), "I": recursive_series("I", n, e), "LR*": linear_ramp_series(n, e)}

    opt_E = e.min()
    opt_MF = n * n / (2 * opt_E)
    tts_uniform = 2 ** n / (e == opt_E).sum()
    uniform_E = e.mean()
    uniform_MF = (n * n / (2 * e)).mean()

    fig, axes = plt.subplots(1, 3, figsize=figsize(aspect_ratio=6.8 / 2.1, two_columns=True))
    handles, labels = [], []

    ax = axes[0]
    for key in plot_order:
        if key not in series:
            continue
        depths, _, _, tts_list = series[key]
        y = np.where(np.isfinite(tts_list) & (np.asarray(tts_list) > 0), tts_list, np.nan)
        label = _KEY_TO_LABEL[key]
        (line,) = ax.semilogy(depths, y, linestyle="-", label=label, **style_plot(label, "SV"))
        handles.append(line)
        labels.append(label)
    line_ideal = ax.axhline(1, color=_REF_OPTIMAL, linestyle="--", linewidth=0.8)
    line_random_tts = ax.axhline(tts_uniform, color=_REF_UNIFORM, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Depth $p$")
    ax.set_ylabel("TTS")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for key in plot_order:
        if key not in series:
            continue
        depths, energies, _, _ = series[key]
        ax.plot(depths, energies, linestyle="-", **style_plot(_KEY_TO_LABEL[key], "SV"))
    ax.axhline(opt_E, color=_REF_OPTIMAL, linestyle="--", linewidth=0.8)
    ax.axhline(uniform_E, color=_REF_UNIFORM, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Depth $p$")
    ax.set_ylabel("LABS energy")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for key in plot_order:
        if key not in series:
            continue
        depths, _, mfs, _ = series[key]
        ax.plot(depths, mfs, linestyle="-", **style_plot(_KEY_TO_LABEL[key], "SV"))
    ax.axhline(opt_MF, color=_REF_OPTIMAL, linestyle="--", linewidth=0.8)
    ax.axhline(uniform_MF, color=_REF_UNIFORM, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Depth $p$")
    ax.set_ylabel("Merit factor")
    ax.grid(True, alpha=0.3)

    all_handles = handles + [line_ideal, line_random_tts]
    all_labels = labels + ["Optimal", "Uniform Sampling"]
    fig.tight_layout(rect=[0, 0, 1, 0.74])
    fig.legend(all_handles, all_labels, loc="upper center", ncol=len(all_handles),
               frameon=False, bbox_to_anchor=(0.5, 0.82))
    fig.savefig(HERE / "method_comparison_N_{}.pdf".format(n))
    plt.close(fig)


if __name__ == "__main__":
    for n in range(10, 22):
        figure(n)

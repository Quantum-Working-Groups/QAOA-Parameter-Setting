"""Shared plotting style helpers extracted from ``figure-styles.ipynb``."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DAGGER = "\N{DAGGER}"
MOJIBAKE_DAGGER = "â€ "

RCPARAMS = {
    "legend.fontsize": "small",
    "legend.columnspacing": 1,
    "axes.titlesize": "small",
    "axes.labelsize": "small",
    "figure.labelsize": "small",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
}

FIGURE_DPI = 300
PREVIEW_DPI = 150

# Paul Tol's bright colour scheme, as used by the notebook.
METHOD_TO_COLOUR = {
    "Fixed Angles*": "#4477AA",
    "Fixed Angles": "#4477AA",
    f"Fixed Angles{DAGGER}": "#4477AA",
    "Fourier*": "#EE6677",
    "Interp.*": "#228833",
    "Linear Ramp*": "#CCBB44",
    "Linear Ramp": "#CCBB44",
    f"Linear Ramp{DAGGER}": "#CCBB44",
    "Recursive TS*": "#66CCEE",
    "TQA*": "#AA3377",
    "TQA": "#AA3377",
    f"TQA{DAGGER}": "#AA3377",
    "Parameter Transfer": "#BBBBBB",
}

EVALUATION_MARKERS = {
    "SV": "o",
    "MPS (Quimb)": "P",
    "MPS (Aer)": "^",
    "PP": "s",
}

METHOD_ALIASES = {
    "Fixed Angles*": "Fixed Angles*",
    "Fixed Angles": "Fixed Angles",
    "FA*": "Fixed Angles*",
    "FA": "Fixed Angles",
    "Fourier": "Fourier*",
    "Fourier*": "Fourier*",
    "INTERP": "Interp.*",
    "Interp": "Interp.*",
    "Interp.": "Interp.*",
    "Interp.*": "Interp.*",
    "Linear Ramp*": "Linear Ramp*",
    "Linear Ramp": "Linear Ramp",
    "LR*": "Linear Ramp*",
    "LR": "Linear Ramp",
    "Recursive TS": "Recursive TS*",
    "Transition States": "Recursive TS*",
    "TS": "Recursive TS*",
    "TQA*": "TQA*",
    "TQA": "TQA",
    "Parameter Transfer": "Parameter Transfer",
}

EVALUATION_ALIASES = {
    "SV": "SV",
    "MPS": "MPS (Quimb)",
    "MPS (Aer)": "MPS (Aer)",
    "MPS (Quimb)": "MPS (Quimb)",
    "PP": "PP",
}

EVALUATION_LINESTYLES = {
    "MPS (Quimb)": "-",
    "MPS (Aer)": "-",
    "PP": "-",
    "SV": "-",
}

ERRORBAR_KWARGS = {
    "ecolor": "k",
    "capsize": 3,
    "elinewidth": 1,
}

LEGEND_MARKER_SIZE = 8
LEGEND_MARKER_EDGE_WIDTH = 0.8

METHOD_LEGEND_KWARGS = {
    # "loc": "lower center",
    "bbox_to_anchor": (0.4, 0.18),
    "ncol": 2,
    "frameon": False,
    "columnspacing": 0.2,
    "handletextpad": 0.1,
    "borderpad": 0.2,
    "fontsize": "x-small",
}

EVALUATION_LEGEND_KWARGS = {
    # "loc": "lower right",
    "bbox_to_anchor": (0.8, 0.18),
    "ncol": 3,
    "frameon": True,
    # "columnspacing": 1,
    # "handletextpad": 0.5,
    # "borderpad": 0.6,
    "fontsize": "x-small",
}

PANEL_LABEL_KWARGS = {
    "x": 0.02,
    "y": 0.98,
    "fontsize": "small",
    "va": "top",
    "ha": "left",
}

SUBPLOT_ADJUST_KWARGS = {
    "bottom": 0.24,
    "wspace": 0.1,
    "hspace": 0.18,
}

SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": FIGURE_DPI,
}


def apply_rcparams(use_tex: bool | None = None) -> bool:
    """Apply the notebook rcParams and return whether LaTeX text rendering is on."""
    if use_tex is None:
        use_tex = shutil.which("latex") is not None
    plt.rcParams.update({"text.usetex": bool(use_tex), **RCPARAMS})
    return bool(use_tex)


def normalize_method(method: str) -> str:
    """Return the canonical notebook method label for common project aliases."""
    try:
        return METHOD_ALIASES[method]
    except KeyError as exc:
        if method in METHOD_TO_COLOUR:
            return method
        raise KeyError(f"Unknown plotting method: {method!r}") from exc


def normalize_evaluation(evaluation: str) -> str:
    """Return the canonical notebook evaluation label for common project aliases."""
    try:
        return EVALUATION_ALIASES[evaluation]
    except KeyError as exc:
        if evaluation in EVALUATION_MARKERS:
            return evaluation
        raise KeyError(f"Unknown evaluation method: {evaluation!r}") from exc


def format_method_label(method: str, format: str = "latex") -> str:
    """Format method labels like ``qaoa_parameter_setting.utils.labels``."""
    method = normalize_method(method)
    if format in ["latex", "siunitx"]:
        return (
            method.replace("*", r"$^\star$")
            .replace(DAGGER, r"$^\dagger$")
            .replace(MOJIBAKE_DAGGER, r"$^\dagger$")
        )
    return method


def style_scatter(method: str, evaluation: str) -> dict[str, Any]:
    """Return a style dictionary suitable for ``matplotlib.pyplot.scatter``."""
    method = normalize_method(method)
    evaluation = normalize_evaluation(evaluation)
    colour = METHOD_TO_COLOUR[method]
    is_unoptimized = DAGGER in method
    return {
        "edgecolor": "k" if "*" in method else (colour if is_unoptimized else "None"),
        "marker": EVALUATION_MARKERS[evaluation],
        "color": colour if not is_unoptimized else "None",
        "facecolor": "w" if is_unoptimized else colour,
    }


def style_plot(method: str, evaluation: str) -> dict[str, Any]:
    """Return a style dictionary suitable for ``matplotlib.pyplot.plot``."""
    method = normalize_method(method)
    kwargs = style_scatter(method=method, evaluation=evaluation)
    kwargs["markerfacecolor"] = kwargs.pop("facecolor")
    kwargs["markeredgecolor"] = kwargs.pop("edgecolor")
    kwargs["color"] = METHOD_TO_COLOUR[method]
    return kwargs


def style_errorbar(
    method: str,
    evaluation: str,
    *,
    linestyle: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return shared style kwargs for line/errorbar plots."""
    evaluation_key = normalize_evaluation(evaluation)
    kwargs = {
        **style_plot(method=method, evaluation=evaluation_key),
        **ERRORBAR_KWARGS,
        "linestyle": linestyle or EVALUATION_LINESTYLES[evaluation_key],
    }
    kwargs.update(overrides)
    return kwargs


def figsize(
    aspect_ratio: float = 1.0,
    two_columns: bool = False,
) -> tuple[float, float]:
    """Compute figure dimensions using the notebook column-width convention."""
    width = 7.0 if two_columns else 3.4
    return (width, width / aspect_ratio)


def legend_handle(label: str, method: str, evaluation: str) -> Line2D:
    """Create a combined method/evaluation marker handle."""
    return Line2D(
        [0],
        [0],
        **style_plot(method=method, evaluation=evaluation),
        linestyle="None",
        markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
        markersize=LEGEND_MARKER_SIZE,
        label=label,
    )


def method_legend_handle(label: str, method: str) -> Line2D:
    """Create a colored square method legend handle."""
    method = normalize_method(method)
    return Line2D(
        [0],
        [0],
        marker="s",
        linestyle="None",
        markerfacecolor=METHOD_TO_COLOUR[method],
        markeredgecolor="k",
        markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
        markersize=LEGEND_MARKER_SIZE,
        label=label,
    )


def evaluation_legend_handle(label: str, evaluation: str) -> Line2D:
    """Create a black marker evaluation legend handle."""
    evaluation = normalize_evaluation(evaluation)
    return Line2D(
        [0],
        [0],
        marker=EVALUATION_MARKERS[evaluation],
        linestyle="None",
        markerfacecolor="k",
        markeredgecolor="k",
        markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
        markersize=LEGEND_MARKER_SIZE,
        color="k",
        label=label,
    )


def method_legend_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default kwargs for the method legend."""
    kwargs = METHOD_LEGEND_KWARGS.copy()
    kwargs.update(overrides)
    return kwargs


def evaluation_legend_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default kwargs for the evaluation legend."""
    kwargs = EVALUATION_LEGEND_KWARGS.copy()
    kwargs.update(overrides)
    return kwargs


def add_panel_labels(
    axes: Iterable[Any],
    labels: Iterable[str] = ("a", "b", "c", "d", "e", "f"),
    **overrides: Any,
) -> None:
    """Add small panel labels with shared text styling."""
    kwargs = PANEL_LABEL_KWARGS.copy()
    kwargs.update(overrides)
    x = kwargs.pop("x")
    y = kwargs.pop("y")
    for ax, label in zip(np_flatten_axes(axes), labels):
        ax.text(
            x,
            y,
            f"{label}",
            transform=ax.transAxes,
            **kwargs,
        )


def np_flatten_axes(axes: Iterable[Any]) -> list[Any]:
    """Flatten matplotlib axes arrays without requiring numpy in this module."""
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return list(axes)


def adjust_subplots_for_legend(fig: Any, **overrides: Any) -> None:
    """Apply shared subplot spacing for a bottom figure legend."""
    kwargs = SUBPLOT_ADJUST_KWARGS.copy()
    kwargs.update(overrides)
    fig.subplots_adjust(**kwargs)


def savefig_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default savefig kwargs for styled figures."""
    kwargs = SAVEFIG_KWARGS.copy()
    kwargs.update(overrides)
    return kwargs

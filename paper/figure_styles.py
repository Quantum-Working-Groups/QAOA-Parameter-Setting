"""Shared plotting style helpers extracted from ``figure-styles.ipynb``."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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
    "Fourier refined*": "blue"
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

PANEL_LABEL_KWARGS = {
    "x": 0.02,
    "y": 0.98,
    "fontsize": "small",
    "va": "top",
    "ha": "left",
}

SUBPLOT_ADJUST_KWARGS = {
    "bottom": 0.24,
    "wspace": 0.2,
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


def add_panel_labels(
    axes: Iterable[Any],
    labels: Iterable[str] = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"),
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


def savefig_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default savefig kwargs for styled figures."""
    kwargs = SAVEFIG_KWARGS.copy()
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# PlotLayout — layout descriptor returned by construct_plot
# ---------------------------------------------------------------------------

@dataclass
class PlotLayout:
    """Describes a fully configured figure/axes layout.

    Returned by :func:`construct_plot` so callers can treat the result as a
    simple data object rather than having to unpack a tuple.

    Attributes:
        fig: The :class:`~matplotlib.figure.Figure` instance.
        axs: 2-D list of :class:`~matplotlib.axes.Axes`, indexed as
            ``axs[row][col]``.  For a 1×1 figure this is ``[[ax]]``.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        method_legend_kw: Kwargs ready to pass to ``legend()`` for the method
            legend.  ``bbox_to_anchor`` is already set to the correct figure
            coordinate for this layout.
        evaluation_legend_kw: Same as *method_legend_kw* but for the
            evaluation legend.  ``None`` when *n_legends* < 2.
        bottom_space: The fractional figure height reserved below the axes for
            legend(s).  Passed to ``fig.subplots_adjust(bottom=…)``.
    """

    fig: Figure
    axs: list[list[Axes]]
    nrows: int
    ncols: int
    method_legend_kw: dict[str, Any]
    evaluation_legend_kw: dict[str, Any] | None
    bottom_space: float

    # Convenience read-only helpers ----------------------------------------

    @property
    def ax(self) -> Axes:
        """The single axes object for 1×1 figures."""
        if self.nrows != 1 or self.ncols != 1:
            raise AttributeError(
                "PlotLayout.ax is only available for 1×1 figures; "
                f"this figure is {self.nrows}×{self.ncols}. Use .axs[r][c]."
            )
        return self.axs[0][0]

    def flat_axes(self) -> list[Axes]:
        """Return all axes in row-major order."""
        return [ax for row in self.axs for ax in row]

    def apply_subplots_adjust(self, **overrides: Any) -> None:
        """Call ``fig.subplots_adjust`` with layout-derived spacing.

        The ``bottom`` value is set to :attr:`bottom_space`.  ``wspace`` and
        ``hspace`` default to the values in :data:`SUBPLOT_ADJUST_KWARGS`.
        Any keyword passed as an override takes precedence.
        """
        kwargs = {
            "bottom": self.bottom_space,
            "wspace": SUBPLOT_ADJUST_KWARGS["wspace"],
            "hspace": SUBPLOT_ADJUST_KWARGS["hspace"],
        }
        kwargs.update(overrides)
        self.fig.subplots_adjust(**kwargs)


def construct_plot(
    nrows: int = 1,
    ncols: int = 1,
    *,
    aspect_ratio: float = 1.0,
    two_columns: bool = False,
    n_legends: int = 1,
    method_ncol: int = 2,
    evaluation_ncol: int = 3,
    dpi: int = FIGURE_DPI,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    **subplots_kwargs: Any,
) -> PlotLayout:
    """Create a fully styled figure with layout-aware legend positioning.

    This is the recommended entry point for creating paper figures.  It wraps
    :func:`matplotlib.pyplot.subplots`, computes appropriate figure dimensions,
    and pre-calculates ``bbox_to_anchor`` positions for up to two legends so
    that they sit neatly in the space below the axes — regardless of the number
    of subplot rows and columns.

    Parameters
    ----------
    nrows:
        Number of subplot rows (default 1).
    ncols:
        Number of subplot columns (default 1).
    aspect_ratio:
        Height-to-width ratio of the *entire figure* (default 1.0 → square).
        Passed through to :func:`figsize`.
    two_columns:
        If ``True``, use the two-column figure width (≈ 7 in) rather than the
        single-column width (≈ 3.4 in).
    n_legends:
        How many legends the figure will carry.

        * ``1`` — a single method legend anchored left-of-centre.
        * ``2`` — method legend on the left, evaluation legend on the right.
        * ``0`` — no automatic legend kwargs are computed.
    method_ncol:
        ``ncol`` passed to the method legend (default 2).
    evaluation_ncol:
        ``ncol`` passed to the evaluation legend (default 3).
    dpi:
        Figure DPI (default :data:`FIGURE_DPI`).
    subplot_kw:
        Forwarded verbatim to :func:`~matplotlib.pyplot.subplots` as
        ``subplot_kw``.
    gridspec_kw:
        Forwarded verbatim to :func:`~matplotlib.pyplot.subplots` as
        ``gridspec_kw``.
    **subplots_kwargs:
        Any additional keyword arguments forwarded to
        :func:`~matplotlib.pyplot.subplots`.

    Returns
    -------
    PlotLayout
        A :class:`PlotLayout` dataclass with ``fig``, ``axs``, layout-derived
        legend kwargs, and a helper ``apply_subplots_adjust()`` method.

    Examples
    --------
    Single-axes figure (like figure 12):

    >>> layout = construct_plot(aspect_ratio=1.0)
    >>> ax = layout.ax
    >>> ax.errorbar(...)
    >>> ax.legend(**layout.method_legend_kw)
    >>> layout.apply_subplots_adjust()
    >>> layout.fig.savefig("out.pdf", **savefig_kwargs())

    2×2 two-column figure with method + evaluation legends (like figure 17):

    >>> layout = construct_plot(2, 2, two_columns=True, aspect_ratio=1/0.8, n_legends=2)
    >>> for r, row in enumerate(layout.axs):
    ...     for c, ax in enumerate(row):
    ...         ax.plot(...)
    >>> fig.legend(handles=method_handles, **layout.method_legend_kw)
    >>> fig.legend(handles=eval_handles, **layout.evaluation_legend_kw)
    >>> layout.apply_subplots_adjust()
    >>> layout.fig.savefig("out.pdf", **savefig_kwargs())
    """
    # ── figure size ────────────────────────────────────────────────────────
    fig_width, fig_height = figsize(aspect_ratio=aspect_ratio, two_columns=two_columns)

    # ── create figure ──────────────────────────────────────────────────────
    fig, raw_axs = plt.subplots(
        nrows,
        ncols,
        dpi=dpi,
        figsize=(fig_width, fig_height),
        subplot_kw=subplot_kw or {},
        gridspec_kw=gridspec_kw or {},
        **subplots_kwargs,
    )

    # Normalise axes into a 2-D list regardless of nrows/ncols.
    if nrows == 1 and ncols == 1:
        axs_2d: list[list[Axes]] = [[raw_axs]]  # type: ignore[list-item]
    elif nrows == 1:
        axs_2d = [list(raw_axs)]  # type: ignore[arg-type]
    elif ncols == 1:
        axs_2d = [[ax] for ax in raw_axs]  # type: ignore[union-attr]
    else:
        axs_2d = [list(row) for row in raw_axs]  # type: ignore[union-attr]

    # ── legend geometry ────────────────────────────────────────────────────
    #
    # Strategy: legends sit in a horizontal strip below the axes.
    # bottom_space is passed to subplots_adjust(bottom=) to reserve room.
    # y_anchor pins the BOTTOM of each legend box (loc="lower ...")
    # so legends of different heights all align at the same baseline,
    # and bbox_inches="tight" cannot push them out of the canvas.
    #
    #   • no legends  → bottom_space=0.04
    #   • 1 legend    → bottom_space=0.14, y_anchor=0.01 (bottom of strip)
    #   • 2 legends   → bottom_space=0.18, y_anchor=0.01 (bottom of strip)

    LEGEND_STRIP_HEIGHTS = {0: 0.04, 1: 0.14, 2: 0.18}
    bottom_space = LEGEND_STRIP_HEIGHTS.get(n_legends, 0.18)
    y_anchor = 0.01   # pin legend bottom just above the figure edge

    if n_legends == 0:
        method_legend_kw: dict[str, Any] = {}
        evaluation_legend_kw: dict[str, Any] | None = None

    elif n_legends == 1:
        method_legend_kw = {
            **_base_method_legend_kw(method_ncol),
            "bbox_to_anchor": (0.5, y_anchor),
            "loc": "lower center",
        }
        evaluation_legend_kw = None

    else:  # n_legends == 2 (or more — treat as 2)
        # Two legends side-by-side; method legend gets left ~55 % of width.
        method_legend_kw = {
            **_base_method_legend_kw(method_ncol),
            "bbox_to_anchor": (0.3, y_anchor),
            "loc": "lower center",
        }
        evaluation_legend_kw = {
            **_base_evaluation_legend_kw(evaluation_ncol),
            "bbox_to_anchor": (0.75, y_anchor),
            "loc": "lower center",
        }

    return PlotLayout(
        fig=fig,
        axs=axs_2d,
        nrows=nrows,
        ncols=ncols,
        method_legend_kw=method_legend_kw,
        evaluation_legend_kw=evaluation_legend_kw,
        bottom_space=bottom_space,
    )


# ---------------------------------------------------------------------------
# Private helpers for constructing base legend kwargs
# ---------------------------------------------------------------------------

def _base_method_legend_kw(ncol: int) -> dict[str, Any]:
    """Base method-legend kwargs without position."""
    return {
        "ncol": ncol,
        "frameon": False,
        "columnspacing": 0.2,
        "handletextpad": 0.1,
        "borderpad": 0.2,
        "fontsize": "x-small",
    }


def _base_evaluation_legend_kw(ncol: int) -> dict[str, Any]:
    """Base evaluation-legend kwargs without position."""
    return {
        "ncol": ncol,
        "frameon": True,
        "fontsize": "x-small",
    }

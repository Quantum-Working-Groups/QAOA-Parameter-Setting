# We use Paul Tol's bright colour scheme
# Define color mapping for each method variant (with optimization markers * and †)
method_to_colour = {
    "Fixed Angles*": "#4477AA",
    "Fixed Angles": "#4477AA",
    "Fixed Angles†": "#4477AA",
    "Fourier*": "#EE6677",
    "Fourier": "#EE6677",
    "Interp.*": "#228833",
    "Interp.": "#228833",
    "Linear Ramp*": "#CCBB44",
    "Linear Ramp": "#CCBB44",
    "Linear Ramp†": "#CCBB44",
    "Recursive TS": "#66CCEE",
    "TQA*": "#AA3377",
    "TQA": "#AA3377",
    "TQA†": "#AA3377",
    "Parameter Transfer": "#BBBBBB",
}

# Define marker shapes for different evaluation methods
evaluation_markers = {
    "SV": "o",
    "MPS (Quimb)": "P",
    "MPS (Aer)": "^",
    "PP": "s",
}


def style_scatter(method: str, evaluation: str) -> dict:
    """Return a style dictionary suitable for :fun:`matplotlib.pyplot.scatter`.

    Args:
        method: The method label, with ``"*"`` or ``"†"`` if the method is
            optimised or unoptimised.
        evaluation: The energy evaluation method label, e.g., ``"SV"``, ``"MPS
            (Aer)"``, ``"MPS (Quimb)"``, or ``"PP"``.

    Returns:
        A dictionary of kwargs appropriate for matplotlib's scatter function.
    """
    # Configure marker styling based on optimization status:
    # - "*" (optimized angles): black edge, colored fill
    # - "†" (unoptimized): colored edge, white fill
    # - neither: no edge, colored fill
    return {
        "edgecolor": "k"
        if "*" in method
        else (method_to_colour[method] if "†" in method else "None"),
        "marker": evaluation_markers[evaluation],
        "color": method_to_colour[method] if "†" not in method else "None",
        "facecolor": "w" if "†" in method else method_to_colour[method],
    }


def style_plot(method: str, evaluation: str) -> dict:
    """Return a style dictionary suitable for :fun:`matplotlib.pyplot.plot`.

    Args:
        method: The method label, with ``"*"`` or ``"†"`` if the method is
            optimised or unoptimised.
        evaluation: The energy evaluation method label, e.g., ``"SV"``, ``"MPS
            (Aer)"``, ``"MPS (Quimb)"``, or ``"PP"``.

    Returns:
        A dictionary of kwargs appropriate for matplotlib's plot function.
    """
    # Start with scatter styling
    kwargs = style_scatter(method=method, evaluation=evaluation)
    # Rename entries from scatter() kwargs to plot() kwargs
    kwargs["markerfacecolor"] = kwargs.pop("facecolor")
    kwargs["markeredgecolor"] = kwargs.pop("edgecolor")
    # Set line color to method color
    kwargs["color"] = method_to_colour[method]
    return kwargs


def figsize(
    aspect_ratio: float = 1.0, two_columns: bool = False
) -> tuple[float, float]:
    # Calculate figure dimensions based on column layout
    if two_columns:
        width = 7.0
    else:
        width = 3.4
    height = width / aspect_ratio
    return (width, height)

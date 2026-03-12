"""Code to format a Pandas table generated from SummaryTable."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
import re
from typing import Any, Literal, TypeAlias, TypeVar

import pandas as pd
from pandas.io.formats.style import Styler, _background_gradient

from .constants import METHOD_OPTMARKER_PLACEHOLDER
from qaoa_parameter_setting.utils.summary.summary_table import SummaryTable
from qaoa_parameter_setting.utils.summary.utils import MethodJSON

ColorMapping: TypeAlias = (
    str
    | dict[str, str]
    | dict[tuple[str | None, str], str]
    | dict[tuple[str, str], str]
    | dict[tuple[None, str], str]
    | None
)
"""A single colormap specification.

Can be:
- A string: a single matplotlib colormap name applied to all evaluations and fields.
- A ``dict[str, str]``: maps evaluation method names (e.g. ``"MPS"``, ``"PP"``, ``"SV"``)
  to colormap names.
- A ``dict[tuple[str | None, str], str]`` or ``dict[tuple[str, str], str]``: maps
  ``(evaluation, field)`` tuples to colormap names, where ``evaluation`` is the
  evaluation method string (or ``None`` for all) and ``field`` is one of
  ``"approximation_ratio"``, ``"num_instances"``, or ``"energy"``.
- ``None``: no background colours are applied.

All entries within a single :data:`ColorMapping` share the same min/max range per metric.
"""

__slots__ = ["formatted_styler_for", "convert_evaluation_to_multicolumn_latex"]


def upgrade_label_to_latex(
    label: str,
    target_format: Literal["text", "latex", "siunitx"],
    optimized_marker: str,
) -> str:
    # Replace the placeholder with the actual optimized marker for all formats
    label = label.replace(METHOD_OPTMARKER_PLACEHOLDER, optimized_marker)

    # Apply LaTeX-specific formatting
    if target_format == "latex" or target_format == "siunitx":
        label = label.replace("†", "$^\\dagger$")

    return label


def __rename_index(obj, rename_func: Callable[[str], str]):
    """Rename indexes with rename_func."""
    if isinstance(obj, pd.MultiIndex):
        # Apply escaping to each level
        levels = [lvl.map(rename_func) for lvl in obj.levels]
        # Rebuild the MultiIndex using the same codes
        return pd.MultiIndex(
            levels=levels,
            codes=obj.codes,
            names=[rename_func(n) if n is not None else None for n in obj.names],
        )
    else:
        # Simple Index
        return obj.map(rename_func).set_names(
            [rename_func(n) if n is not None else None for n in obj.names]
        )


def __escape_underscores_index(obj):
    """Escape underscores in Index or MultiIndex levels.

    Converts non-string labels to strings first.
    """

    def escape(x):
        s = str(x)
        return s.replace("_", r"\_")

    return __rename_index(obj, escape)


def escape_underscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Escape underscores in both columns and index labels."""
    df = df.copy()
    df.columns = __escape_underscores_index(df.columns)
    df.index = __escape_underscores_index(df.index)
    return df


def remove_json_suffix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove suffix ``".json"`` from columns and indexes, returning a copy."""
    df = df.copy()

    def _remove_suffix_json(x: str | Any) -> str:
        if isinstance(x, str) and x.endswith(".json"):
            return x.removesuffix(".json")
        return x

    df.columns = __rename_index(df.columns, _remove_suffix_json)
    df.index = __rename_index(df.index, _remove_suffix_json)
    return df


def wrap_latex_maths(val: str) -> str:
    """Wrap ``val`` in LaTeX maths delimiters."""
    return "${}$".format(val)


def unwrap_latex_maths(val: str) -> str:
    """Remove LaTeX maths delimiters from the ends of val."""
    return val.removeprefix("$").removesuffix("$")


class AggregatorFormatter:
    """Abstract class for aggregating and formatting values in a dataframe."""

    def __init__(
        self, precision: int, format: Literal["text", "latex", "siunitx"] = "text"
    ):
        """Create the given aggregator-formatter.

        ``"text"`` is for Jupyter notebooks, ``"latex"`` is for simple LaTeX
        tables, and ``"siunitx"`` is for fancy LaTeX tables using the
        ``siunitx`` package for advanced number rendering.

        Args:
            precision (int): Precision for values.
            format: How to format values. ``"latex"`` and ``"siunitx"`` both
                target LaTeX. Defaults to "text".
        """
        self._format = format
        self._precision = precision

    @abstractmethod
    def __call__(self, vals: pd.Series) -> Any: ...

    """Aggregate and format a series of values."""

    def _range_sep(self) -> str:
        """Seperator for ranges."""
        if self._format == "text":
            return "-"
        elif self._format == "latex":
            return "--"
        return ""

    def _format_range(self, start: float | int, end: float | int) -> str:
        """Format a range from ``start`` to ``end``.

        Args:
            start: The start of the range.
            end: The end of the range.

        Returns:
            A string representing the range ``[start, end]``.
        """
        if self._format == "text":
            return "{start:0.{precision}f}-{end:0.{precision}f}".format(
                start=start, end=end, precision=self._precision
            )
        elif self._format == "latex":
            return "{}--{}".format(
                wrap_latex_maths(
                    "{start:0.{precision}f}".format(
                        start=start, precision=self._precision
                    )
                ),
                wrap_latex_maths(
                    "{end:0.{precision}f}".format(
                        end=end,
                        precision=self._precision,
                    )
                ),
            )
        else:
            return (
                r"\numrange[range-phrase=--,round-mode=places,"
                + "round-precision={precision}]{{{start}}}{{{end}}}".format(
                    start=start, end=end, precision=self._precision
                )
            )

    def _unformat_range(self, val: str) -> tuple[float, float]:
        """Reverse the effect of :meth:`_format_range`."""
        if self._format == "text":
            _start, _end = val.split("-")
            return (float(_start), float(_end))
        elif self._format == "latex":
            _start, _end = unwrap_latex_maths(val).split("--")
            return (float(_start), float(_end))
        else:
            pattern = re.compile(r".*{([\d\.]+)}{([\d\.]+)}")
            match = pattern.search(val)
            assert match is not None
            return (float(match.group(1)), float(match.group(2)))

    def _format_num(self, val: float | int) -> str:
        """Format a single number.

        Args:
            val: The number to format.

        Returns:
            ``val`` formatted as a single number.
        """
        if self._format == "text":
            return "{val:0.{precision}f}".format(val=val, precision=self._precision)
        elif self._format == "latex":
            return wrap_latex_maths(
                "{val:0.{precision}f}".format(val=val, precision=self._precision)
            )
        else:
            return (
                r"\num[round-mode=places,round-precision={precision}]{{{val}}}".format(
                    val=val, precision=self._precision
                )
            )

    def _unformat_num(self, val: str) -> float:
        """Reverse the effect of :meth:_format_num`."""
        if self._format == "text":
            if "." in val:
                return float(val)
            return int(val)
        elif self._format == "latex":
            if "." in val:
                return float(unwrap_latex_maths(val))
            return int(unwrap_latex_maths(val))
        else:
            pattern = re.compile(r".*{([\d\.]+)}")
            match = pattern.search(val)
            assert match is not None
            _arg = match.group(1)
            return float(_arg)

    def _format_uncertainty(self, val: float, uncertainty: float) -> str:
        """Format a value and some uncertainty.

        Args:
            val: The base value.
            uncertainty: The uncertainty in the value.

        Returns:
            The base value ``val`` formatted with its uncertainty.
        """
        if self._format == "text":
            return "{val:0.{precision}f}+-{uncertainty:0.{precision}f}".format(
                val=val, uncertainty=uncertainty, precision=self._precision
            )
        elif self._format == "latex":
            return wrap_latex_maths(
                r"{val:0.{precision}f}\pm {uncertainty:0.{precision}f}".format(
                    val=val, uncertainty=uncertainty, precision=self._precision
                )
            )
        else:
            return (
                r"\num[uncertainty-mode=separate,round-mode=uncertainty,"
                + "round-precision={precision}]{{{val}+-{uncertainty}}}".format(
                    val=val, uncertainty=uncertainty, precision=self._precision
                )
            )

    def _unformat_uncertainty(self, val: str) -> tuple[float, float]:
        """Reverse the effect of :meth:`_format_uncertainty`."""
        if self._format == "text":
            _val, _uncertainty = val.split("+-")
            return (float(_val), float(_uncertainty))
        elif self._format == "latex":
            _val, _uncertainty = unwrap_latex_maths(val).split(r"\pm ")
            return (float(_val), float(_uncertainty))
        else:
            pattern = re.compile(r".*{([\d\.]+)\+-([\d\.]+)}")
            match = pattern.search(val)
            assert match is not None
            _mean = match.group(1)
            _uncertainty = match.group(2)
            return (float(_mean), float(_uncertainty))

    def base_value(self, val: str | float | int) -> float:
        """Retrieve the original base value from ``val``.

        If ``val`` is not a string, then ``val`` is returned unmodified. If
        ``val`` is a string, then it is processed by :meth:`_base_value`.

        Args:
            val: The formatted value from which the base value should be extracted.

        Returns:
            The base value from ``val``.
        """
        if isinstance(val, (float, int)):
            return val
        return self._base_value(val)

    @abstractmethod
    def _base_value(self, val: str) -> float: ...


class MeanAggregator(AggregatorFormatter):
    """Compute the mean percentage and format as a single number."""

    def __call__(self, vals: pd.Series) -> str:
        # Get the mean and convert to a percentage.
        _mean = vals.mean() * 100
        return self._format_num(_mean)

    def _base_value(self, val: str) -> float:
        return self._unformat_num(val)


class CountAggregator(AggregatorFormatter):
    """Count the number of values and format as a single number."""

    def __call__(self, vals: pd.Series) -> Any:
        return self._format_num(vals.count())

    def _base_value(self, val: str) -> float:
        return self._unformat_num(val)


class FancyStdDevAggregator(AggregatorFormatter):
    """Calculate the mean and standard deviation percentage and format as the mean+-std dev.

    The exact format of ``+-`` is determined by :attr:`format`."""

    def __call__(self, vals: pd.Series) -> Any:
        _count = vals.count()
        # Get the mean and standard deviation and convert to percentage
        _mean = vals.mean() * 100
        _std = (vals * 100).std()
        if _count > 1:
            return self._format_uncertainty(_mean, _std)
        else:
            return self._format_num(_mean)

    def _base_value(self, val: str) -> float:
        if self._format == "text":
            if "+-" in val:
                return self._unformat_uncertainty(val)[0]
            else:
                return self._unformat_num(val)
        elif self._format == "latex":
            if r"\pm " in val:
                return self._unformat_uncertainty(val)[0]
            else:
                return self._unformat_num(val)
        else:
            if "+-" in val:
                return self._unformat_uncertainty(val)[0]
            else:
                return self._unformat_num(val)


class FancyCountRangeAggregator(AggregatorFormatter):
    """Count the number of values and their range, formatting as ``count [start-end]``."""

    def __call__(self, vals: pd.Series) -> Any:
        _min = vals.min()
        _max = vals.max()
        _count = vals.count()
        return "{count} [{range}]".format(
            count=self._format_num(_count),
            range=(
                self._format_num(_min)
                if _min == _max
                else self._format_range(_min, _max)
            ),
        )

    def _base_value(self, val: str) -> float:
        return self._unformat_num(val.split(" ")[0])


T = TypeVar("T", bound=pd.Series | pd.DataFrame)


def _unformat_data(formatted_data: T, aggregator: AggregatorFormatter) -> T:
    """Map ``formatted_data`` using :meth:`AggregatorFormatter.base_value`."""
    return formatted_data.map(aggregator.base_value)


def _unformatted_background_gradient(
    formatted_data: pd.Series, aggregator: AggregatorFormatter, **kwargs
):
    """Function that modifes Pandas style based on unformatted data from
    ``formatted_data`` and ``aggregator``."""
    _data = _unformat_data(formatted_data, aggregator=aggregator)
    return _background_gradient(_data, **kwargs)


def formatted_styler_for(
    table: SummaryTable,
    agg_values: Literal["approximation_ratio", "num_instances", "energy"]
    | list[Literal["approximation_ratio", "num_instances", "energy"]],
    depths: int | list[int] | None = None,
    num_nodes: int | list[int] | dict[str, int | list[int]] | None = None,
    with_fancy_values: bool = False,
    cmap: ColorMapping | list[ColorMapping] = "Greens",
    target_format: Literal["text", "latex", "siunitx"] = "text",
    precision: int
    | dict[Literal["approximation_ratio", "num_instances", "energy"], int]
    | None = None,
    missing_data_str: str = "-",
    show_empty_rows: bool = True,
    optimized_marker: str | None = None,
    exclude_methods: list[str] | None = None,
    restrict_mps_to_aer: bool = False,
    reference_methods: dict[tuple[str, str], MethodJSON] | None = None,
) -> tuple[pd.DataFrame, Styler, dict[str, tuple[float, float]]]:
    """Format a dataframe as a pivot table with appropriate styling.

    Args:
        table: The :class:`SummaryTable` instance whose data should be formatted.
        agg_values: Values to aggregate. Can be a single value or a list of values.
            Options are "approximation_ratio", "num_instances", or "energy".
        depths: Depths to show in the data. Results for different depths will
            not be aggregated. If None, all depths will be shown. Defaults to None.
        num_nodes: Graph sizes to show in the data. Results for different sizes
            will not be aggregated. If None, all graph sizes will be shown.
            If an ``int`` or ``list[int]``, only rows whose ``num_nodes`` value
            is in the given set are kept (across all evaluation methods).
            If a ``dict[str, int | list[int]]``, the keys are evaluation method
            names (e.g. ``"MPS"``, ``"PP"``, ``"SV"``) and the values specify
            which graph sizes to keep **for that evaluation method only**.
            Evaluation methods not present as keys are not filtered by
            ``num_nodes``.  Example::

                num_nodes = {"MPS": 100, "PP": 100, "SV": 20}

            Defaults to None.
        with_fancy_values: Whether aggregate values should show additional
            information, e.g., standard deviation for
            ``agg_values="approximation_ratio"`` or ``agg_values="energy"``. Defaults to False.
        cmap: The matplotlib colormap(s) to use for cell background colours.
            Defaults to ``"Greens"``.

            A single :data:`ColorMapping` or a list of :data:`ColorMapping` instances
            may be provided.  If a single :data:`ColorMapping` is given it is
            automatically wrapped in a list.

            Each :data:`ColorMapping` in the list is processed **independently**:
            the min/max values used to scale the colour range are computed
            separately for each :data:`ColorMapping` (per metric).  This means
            that two :data:`ColorMapping` instances never share their colour
            range.

            A :data:`ColorMapping` can be:

            - A **string** – a single matplotlib colormap name applied to all
              evaluation methods and all requested metrics.
            - A **dict[str, str]** – maps evaluation method names (e.g.
              ``"MPS"``, ``"PP"``, ``"SV"``) to colormap names.  All entries
              share the same min/max range (per metric).
            - A **dict[tuple[str | None, str], str]** – maps
              ``(evaluation, field)`` tuples to colormap names, where
              *evaluation* is the evaluation method string (or ``None`` for
              all methods) and *field* is one of ``"approximation_ratio"``,
              ``"num_instances"``, or ``"energy"``.  All entries share the
              same min/max range (per metric).
            - ``None`` – no background colours are applied for this entry.

            **Examples**::

                # All evaluations coloured green, shared min/max per metric.
                cmap = "Greens"

                # Different colormaps per evaluation, shared min/max per metric.
                cmap = {"MPS": "YlOrBr", "PP": "YlGn", "SV": "PuBu"}

                # Only the Energy metric is coloured; shared min/max per metric.
                cmap = {("MPS", "Energy"): "YlOrBr",
                        ("PP",  "Energy"): "YlGn",
                        ("SV",  "Energy"): "PuBu"}

                # MPS and PP share their min/max; SV has its own min/max.
                cmap = [
                    {"MPS": "YlOrBr", "PP": "YlGn"},
                    {"SV": "PuBu"},
                ]

        target_format: Dictates how values are formatted. See implementation of
            :class:`Aggregator` for details. Defaults to "text".
        precision: Optional precision of values, e.g., number of decimal places.
            If None, then the default value is ``4`` if any value in
            ``agg_values`` is "approximation_ratio" or "energy" and ``0`` otherwise. The
            precision for different ``agg_values`` can be different by providing
            a dictionary. See implementation of :class:`Aggregator` for details.
            Defaults to None.
        missing_data_str: String to put in place of missing data. Defaults to "-".
        show_empty_rows: Include all unique values in the rows, even if there
            are no results for a given configuration. For example, if there are
            no results for method ``TQA`` and the given depth, but there are for
            another depth, then the row for ``TQA`` will still be shown. If
            False, only row values present in the displayed data is shown.
            Defaults to False.
        optimized_marker: This parameter is no longer used.
            Method labels are now determined by the METHOD_CONFIG_TO_LABELS
            constant in the constants module. Defaults to None.
        exclude_methods: Optional list of method acronyms to exclude from the
            formatted table. Methods whose names start with any of these
            acronyms will be filtered out. For example, passing ``["FA", "I"]``
            would exclude all Fixed Angle and Interpolation methods. If None,
            no methods are excluded. Defaults to None.
        restrict_mps_to_aer: Restricts the table to methods that use the Qiskit
            Aer simulator, only for MPS energy evaluation methods. If False, all
            methods are shown for MPS. Defaults to False.
        reference_methods: Optional dictionary mapping (evaluation_method, graph_type)
            tuples to reference trainer config names. When provided, for each
            (evaluation_method, graph_type) pair, only graph instances present in
            the reference training method are included for all other training methods
            with the same evaluation method and graph type. This ensures consistent
            instance counts across training methods. For example::

                reference_methods = {
                    ("SV", "random_regular"): "F_SV.json",
                    ("MPS", "heavy_hex"): "LR_PP_opt.json"
                }

            If None, no instance filtering is applied. Defaults to None.

    Raises:
        ValueError: If ``agg_values`` is an unrecognised value.
        ValueError: If two :data:`ColorMapping` instances in ``cmap`` specify
            colours for the same ``(evaluation, field)`` combination.

    Returns:
        The tuple ``(df, styler, cmap_ranges)`` where ``styler`` contains appropriate
        colours, naming, etc. for inclusion in a paper, and ``cmap_ranges`` contains
        the min/max values used for coloring each field.
    """

    # Normalize agg_values to a list
    if isinstance(agg_values, str):
        agg_values_list = [agg_values]
    else:
        agg_values_list = list(agg_values)

    if precision is None:
        precision = {
            "approximation_ratio": 1,
            "num_instances": 0,
            "energy": 2,
        }
    elif isinstance(precision, int):
        precision = {
            "approximation_ratio": precision,
            "num_instances": precision,
            "energy": precision,
        }

    for _agg_value in agg_values_list:
        assert _agg_value in precision, (
            f"{_agg_value} not found in precision dictionary."
        )
    # Determine the optimized marker to use for replacing METHOD_OPTMARKER_PLACEHOLDER
    if optimized_marker is None:
        if target_format == "text":
            optimized_marker = "*"
        else:
            optimized_marker = "$^\\star$"

    # *** Apply instance filtering if reference configs are provided
    if reference_methods is not None:
        # Determine the depth to use for filtering
        filter_depth = None
        if depths is not None:
            if isinstance(depths, int):
                filter_depth = depths
            elif len(depths) == 1:
                filter_depth = depths[0]
            # If multiple depths, we don't filter by depth

        table = table.filter_to_common_instances(
            reference_methods=reference_methods,
            depth=filter_depth,
        )

    # *** Pivot dataframe and aggregate appropriately
    _data = table.to_dataframe()

    # Create aggregators and determine values for each field
    aggregators: dict[str, AggregatorFormatter] = {}
    __agg_values_dict: dict[str, str] = {}

    for agg_val in agg_values_list:
        if agg_val == "num_instances":
            __use_fancy_count = with_fancy_values
            __agg_values_dict[agg_val] = "num_nodes"

            if __use_fancy_count:
                aggregators[agg_val] = FancyCountRangeAggregator(
                    precision=precision[agg_val], format=target_format
                )
            else:
                aggregators[agg_val] = CountAggregator(
                    precision=precision[agg_val],
                    format=target_format,
                )
        elif agg_val == "approximation_ratio":
            if with_fancy_values:
                aggregators[agg_val] = FancyStdDevAggregator(
                    precision=precision[agg_val], format=target_format
                )
            else:
                aggregators[agg_val] = MeanAggregator(
                    precision=precision[agg_val],
                    format=target_format,
                )
            __agg_values_dict[agg_val] = agg_val
        elif agg_val == "energy":
            if with_fancy_values:
                aggregators[agg_val] = FancyStdDevAggregator(
                    precision=precision[agg_val], format=target_format
                )
            else:
                aggregators[agg_val] = MeanAggregator(
                    precision=precision[agg_val],
                    format=target_format,
                )
            __agg_values_dict[agg_val] = "energy"
        else:
            # We shouldn't get here, but it's good practice to handle this default
            # branch of the if statements.
            raise ValueError(
                "agg_values {!r} must be either 'num_instances', 'approximation_ratio', or 'energy'.".format(
                    agg_val
                )
            )

    # *** Filter based on depth, num_nodes, and excluded methods
    filtered_data = _data
    if depths is not None:
        if isinstance(depths, int):
            depths = [depths]
        filtered_data = filtered_data[filtered_data["depth"].isin([d for d in depths])]
    if num_nodes is not None:
        if isinstance(num_nodes, dict):
            # Per-evaluation filtering: build a boolean mask that keeps a row
            # when either (a) its evaluation method is not in the dict, or
            # (b) its num_nodes value is in the allowed set for that evaluation.
            def _num_nodes_mask(row) -> bool:
                eval_method = row["evaluation"]
                if eval_method not in num_nodes:
                    return True
                allowed = num_nodes[eval_method]
                if isinstance(allowed, int):
                    allowed = [allowed]
                return row["num_nodes"] in allowed

            filtered_data = filtered_data[filtered_data.apply(_num_nodes_mask, axis=1)]
        else:
            if isinstance(num_nodes, int):
                num_nodes = [num_nodes]
            filtered_data = filtered_data[
                filtered_data["num_nodes"].isin(list(num_nodes))
            ]

    # Filter out excluded methods by checking if method starts with any excluded acronym
    if exclude_methods is not None:
        # Create a mask that is True for methods we want to keep
        mask = ~filtered_data["method"].apply(
            lambda method: any(
                method.startswith(acronym) for acronym in exclude_methods
            )
        )
        filtered_data = filtered_data[mask]

    if restrict_mps_to_aer:
        # Remove rows where MPS is used without Aer
        filtered_data = filtered_data[
            ~((filtered_data["evaluation"] == "MPS") & (~filtered_data["uses_aer"]))
        ]

    # *** Pivot table
    # Pivot and use aggregators - create separate pivot for each value if multiple
    if len(agg_values_list) == 1:
        # Single value case - use original logic
        agg_val = agg_values_list[0]
        pivot = filtered_data.pivot_table(
            columns=["graph_type"],
            index=[
                "evaluation",
                "method_label",
            ],
            values=__agg_values_dict[agg_val],
            aggfunc=aggregators[agg_val],  # type:ignore
        )
    else:
        # Multiple values case - create dict of aggregators for pivot_table
        pivot = filtered_data.pivot_table(
            columns=["graph_type"],
            index=[
                "evaluation",
                "method_label",
            ],
            values=[__agg_values_dict[agg_val] for agg_val in agg_values_list],
            # values=agg_values_list,
            aggfunc={
                __agg_values_dict[agg_val]: aggregators[agg_val]
                for agg_val in agg_values_list
            },  # type:ignore
        )

    # *** Recreate indices to ensure we include all methods and relevant column values.
    # To recreate the columns, we need to (i) get a list of graph types, (ii)
    # figure out which attributes we have in the columns, and (iii) create a
    # list of tuples appropriate for reindexing.

    # Row index should include the evaluation method and method label, even
    # though one is derived from the other.
    if show_empty_rows:
        # Filter out excluded methods when creating the row index
        # Build list of (evaluation, method_label) tuples from all methods
        from .constants import METHOD_CONFIG_TO_LABELS

        _methods_to_include = table.all_methods()
        # Remove non-Aer methods restrict_mps_to_aer is True
        if restrict_mps_to_aer:
            # We only include methods that (1) are for PP or SV or (2) are for MPS and use Aer.
            _methods_to_include = [
                _method
                for _method in _methods_to_include
                if table.trainer_config_to_evaluation(_method) != "MPS"
                or "Aer" in table.trainer_config_to_method(_method)
            ]
        # If exclude_methods is provided, we must remove those methods
        if exclude_methods is not None:
            _methods_to_include = [
                _method_json
                for _method_json in table.all_methods()
                if not any(
                    table.trainer_config_to_method(_method_json).startswith(acronym)
                    for acronym in exclude_methods
                )
            ]

        # Create tuples and deduplicate since multiple MethodConfigJSON can map to same label
        _row_tuples = [
            (
                table.trainer_config_to_evaluation(_method_json),
                METHOD_CONFIG_TO_LABELS[table.trainer_config_to_method(_method_json)],
            )
            for _method_json in _methods_to_include
        ]
        # Remove duplicates while preserving order, then sort
        _row_index = pd.MultiIndex.from_tuples(sorted(list(dict.fromkeys(_row_tuples))))
    else:
        _row_index = pivot.index.sort_values()

    # Reindex:
    # 1. The row index now has two columns: the evaluation method
    #    and the method label. All methods are included.
    # 2. The columns contain the same number of rows. But now depth and
    #    num_nodes will contain all values if ``depths`` and ``num_nodes`` are
    #    None, respectively. Other columns will always contain all unique
    #    values, such as ``graph_type``.
    pivot = pivot.reindex(
        index=_row_index,
    )
    # *** End of reindexing

    # Rename columns to use graph types from table.
    pivot = pivot.rename(columns=table.formatter_graph_type())

    # Create a mapping to replace the placeholder in method labels (level 1)
    label_mapping = {
        label: upgrade_label_to_latex(
            label, target_format=target_format, optimized_marker=optimized_marker
        )
        for label in pivot.index.get_level_values(1).unique()
    }
    pivot = pivot.rename(index=label_mapping, level=1)

    # Rename columns
    column_renamings = {
        "approximation_ratio": "Approximation Ratio",
        "num_instances": "Num. Instances",
        # As we use num_nodes to compute the number of instances, we rename it
        # to the same as "num_instances"
        "num_nodes": "Num. Instances",
        "energy": "Energy",
    }
    pivot = pivot.rename(columns=column_renamings)

    # *** Handle styler colours, precision, etc.
    # Set format options
    styler = pivot.style.format(
        # Set string for missing values, represented by nan.
        na_rep=missing_data_str,
        # We're escaping characters with AggregatorFormatter, not Styler.
        escape=None,
    )

    # *** Set background colours
    cmap_ranges: dict[str, tuple[float, float]] = {}

    # Normalise cmap to list[ColorMapping]
    if not isinstance(cmap, list):
        cmap_list: list[ColorMapping] = [cmap]
    else:
        cmap_list = list(cmap)

    # Helper: expand a single ColorMapping to a canonical dict[tuple[str | None,
    # str], str] (evaluation, field) -> colormap_name. Returns an empty dict for
    # None entries.
    def _expand_color_mapping(
        cm: ColorMapping,
    ) -> dict[tuple[str | None, str], str]:
        if cm is None:
            return {}
        if isinstance(cm, str):
            # Single colormap for all evaluations and all fields
            return {(None, agg_val): cm for agg_val in agg_values_list}
        # It's a dict – inspect the first key to determine format.
        first_key = next(iter(cm.keys()))
        if isinstance(first_key, tuple):
            # New format: keys are already (evaluation, field) tuples.
            return dict(cm)  # type: ignore[arg-type]
        else:
            # Old format: keys are evaluation strings; apply to all fields.
            return {
                (eval_key, agg_val): colormap
                for eval_key, colormap in cm.items()  # type: ignore[union-attr]
                for agg_val in agg_values_list
            }

    # Validate: no two ColorMapping instances may cover the same
    # (evaluation, field) combination.
    _seen_keys: dict[tuple[str | None, str], int] = {}
    for _cm_idx, _cm in enumerate(cmap_list):
        _expanded = _expand_color_mapping(_cm)
        for _key in _expanded:
            if _key in _seen_keys:
                raise ValueError(
                    (
                        "Two ColorMapping instances (indices {first} and {second}) both "
                        + "specify a cmap for (evaluation={eval!r}, field={field!r}). "
                        + "Each (evaluation, field) combination must appear in at most one "
                        + "ColorMapping so that min/max ranges are unambiguous."
                    ).format(
                        first=_seen_keys[_key],
                        second=_cm_idx,
                        eval=_key[0],
                        field=_key[1],
                    )
                )
            _seen_keys[_key] = _cm_idx

    # Apply each ColorMapping independently with its own min/max range.
    if any(cm is not None for cm in cmap_list):
        for _cm in cmap_list:
            if _cm is None:
                continue

            _cmap_dict = _expand_color_mapping(_cm)
            if not _cmap_dict:
                continue

            # Determine which (evaluation, field) pairs are covered by this
            # ColorMapping so we can compute the min/max only over those rows.
            _covered_fields: set[str] = {field for (_, field) in _cmap_dict}
            _covered_evaluations: set[str | None] = {eval_ for (eval_, _) in _cmap_dict}

            # Compute min/max for each field covered by this ColorMapping.
            # The range is computed over all evaluations that appear in this
            # ColorMapping (or all evaluations when None is present).
            _vmin_dict: dict[str, float] = {}
            _vmax_dict: dict[str, float] = {}

            for agg_val in _covered_fields:
                aggregator = aggregators[agg_val]

                # Extract the full field data from the pivot table.
                if len(agg_values_list) == 1:
                    _field_data = pivot
                else:
                    _field_data = pivot[column_renamings[agg_val]]

                # If None is among the covered evaluations for this field, use
                # all rows; otherwise restrict to the covered evaluations.
                _evals_for_field: set[str | None] = {
                    eval_ for (eval_, f) in _cmap_dict if f == agg_val
                }
                if None in _evals_for_field:
                    # All evaluations contribute to the range.
                    _range_data = _field_data
                else:
                    # Only the listed evaluations contribute.
                    _eval_list = [e for e in _evals_for_field if e is not None]
                    # The first level of the row MultiIndex is the evaluation.
                    _range_data = _field_data.loc[
                        _field_data.index.get_level_values(0).isin(_eval_list)
                    ]

                _base_data = _unformat_data(_range_data, aggregator=aggregator)
                _vmin_dict[agg_val] = float(_base_data.min().min())
                _vmax_dict[agg_val] = float(_base_data.max().max())

            # Apply background gradient for each (evaluation, field) entry.
            for (_evaluation, _field), _cmap_name in _cmap_dict.items():
                aggregator = aggregators[_field]
                _vmin = _vmin_dict[_field]
                _vmax = _vmax_dict[_field]

                # Determine the pandas IndexSlice subset.
                if len(agg_values_list) == 1:
                    _subset = (
                        pd.IndexSlice[[_evaluation], :]
                        if _evaluation is not None
                        else slice(None)
                    )
                else:
                    if _evaluation is not None:
                        _subset = pd.IndexSlice[
                            [_evaluation],
                            column_renamings[__agg_values_dict[_field]],
                        ]
                    else:
                        _subset = pd.IndexSlice[
                            :,
                            column_renamings[__agg_values_dict[_field]],
                        ]

                # Check if the subset would be empty, i.e., if the evaluation method is even present
                # in the data. This can happen if show_empty_rows=False and there is insufficient
                # data.
                if _evaluation is not None:
                    if _evaluation not in pivot.index.get_level_values(0):
                        continue

                styler = styler.apply(
                    partial(_unformatted_background_gradient, aggregator=aggregator),
                    vmin=_vmin,
                    vmax=_vmax,
                    cmap=_cmap_name,
                    # Following parameters taken from implementation of
                    # Styler.background_gradient.
                    axis=0,
                    subset=_subset,
                    low=0,
                    high=0,
                    text_color_threshold=0.408,
                    gmap=None,
                )
                # Record the range used; later ColorMappings may overwrite this
                # for the same field, which is intentional (last writer wins for
                # the returned cmap_ranges dict).
                cmap_ranges[_field] = (_vmin, _vmax)
        # Missing values should not have a colour, which is the default with
        # Styler.background_gradient
        styler = styler.highlight_null(props="background-color:white;color:black")

    # If we are targeting LaTeX, escape underscores. We do this at the end so we
    # don't have to update column and row names with an underscore.
    if target_format in ["latex", "siunitx"]:
        pivot = escape_underscore_df(pivot)
        styler.data = escape_underscore_df(styler.data)

    column_rename = {
        "approximation_ratio": "Approximation Ratio",
        "num_instances": "Num. Instances",
        "energy": "Energy",
    }
    pivot = pivot.rename(columns=column_rename)
    styler.data = styler.data.rename(columns=column_rename)

    return pivot, styler, cmap_ranges


def convert_evaluation_to_multicolumn_latex(
    latex_str: str, num_metrics: int = 1
) -> str:
    """Convert evaluation method rows in LaTeX tables to multicolumn cells.

    This function post-processes LaTeX table output to:
    1. Remove the entire first column (evaluation column)
    2. Transform rows where the evaluation method appeared into multicolumn headers
    3. Add \\midrule after each evaluation header
    4. Update \\cline commands to reflect the new column count

    This creates cleaner section headers for each evaluation method (MPS, PP, SV).

    Args:
        latex_str: The LaTeX table string generated by Styler.to_latex()
        num_metrics: Number of metrics in the table (e.g., 1 for single metric like
            "num_instances", 2 for both "approximation_ratio" and "num_instances").
            Used to determine the number of graph columns. Defaults to 1.

    Returns:
        Modified LaTeX string with evaluation methods as multicolumn headers
        and the evaluation column removed

    Example:
        Input:
        \\begin{tabular}{llllll}
         & graph_type & ER & HH & L2F & RR \\\\
        evaluation & method &  &  &  &  \\\\
        \\multirow[c]{5}{*}{MPS} & Fourier & {...} \\\\
         & Fourier (Aer) & {...} \\\\
        \\cline{1-6}

        Output:
        \\begin{tabular}{lllll}
        graph_type & ER & HH & L2F & RR \\\\
        method &  &  &  &  \\\\
        \\multicolumn{5}{c}{MPS} \\\\
        \\midrule
        Fourier & {...} \\\\
        Fourier (Aer) & {...} \\\\
        \\cline{1-5}
    """
    lines = latex_str.split("\n")
    processed_lines = []

    # Track the number of columns (after removing the first column)
    num_columns = None
    in_header = True  # Track if we're still in the header section
    already_added_evaluation_method = False

    for line in lines:
        # Extract number of columns from \begin{tabular}{...} and remove first column
        tabular_match = re.match(r"\\begin\{tabular\}\{([lcrp|]+)\}", line)
        if tabular_match and num_columns is None:
            # Count the column specifiers (l, c, r, p)
            col_spec = tabular_match.group(1)
            # Remove the first column specifier (the evaluation/method column)
            new_col_spec = col_spec[1:]
            num_columns = len([c for c in new_col_spec if c in "lcrp"])

            # Build column specification with vertical borders separating metrics
            # The first column is the method name, then groups of graph columns
            col_chars = [c for c in new_col_spec if c in "lcrp"]
            formatted_cols = []

            # First column (method names) with border after it
            if col_chars:
                formatted_cols.append(col_chars[0])
                formatted_cols.append("|")

                # Calculate number of graph columns based on number of metrics
                # Total data columns = (number of columns - 1 for method column)
                # Number of graphs = total data columns / num_metrics
                data_cols = len(col_chars) - 1
                num_graph_types = data_cols // num_metrics

                for i in range(1, len(col_chars), num_graph_types):
                    # Replace left-align with centred as we are most likely
                    # dealing with numerical values or headings
                    formatted_cols.append(
                        "".join(col_chars[i : i + num_graph_types]).replace("l", "c")
                    )
                    formatted_cols.append("|")

            new_line = f"\\begin{{tabular}}{{{(''.join(formatted_cols))}}}"
            processed_lines.append(new_line)
            continue

        # Update \cline commands to reflect new column count
        cline_match = re.match(r"\\cline\{1-(\d+)\}", line)
        if cline_match and num_columns:
            # Replace with new column count
            new_line = f"\\cline{{1-{num_columns}}}"
            processed_lines.append(new_line)
            continue

        # Check if this line starts with \multirow (evaluation method row)
        multirow_match = re.match(
            r"^\s*\\multirow\[c\]\{\d+\}\{\*\}\{([^}]+)\}\s*&\s*(.+)$", line
        )

        if multirow_match:
            # Extract the evaluation method name and the rest of the row
            eval_method = multirow_match.group(1)
            rest_of_row = multirow_match.group(2)

            # Check if this looks like an evaluation method (MPS, PP, SV, etc.)
            if eval_method and len(eval_method) <= 10 and num_columns:
                # Figure out prefix to ensure the evaluation headings have the
                # correct vertical size. The first row needs to be slightly
                # shorter, probably because it doesn't have maths text above it.
                if not already_added_evaluation_method:
                    prefix = "\\rule{0pt}{8pt}"
                else:
                    already_added_evaluation_method = True
                    prefix = "\\rule{0pt}{10pt}"
                # Insert a multicolumn row for the evaluation method
                multicolumn_line = (
                    f"\\multicolumn{{{num_columns}}}{{c}}{{{prefix}{eval_method}}} \\\\"
                )
                processed_lines.append(multicolumn_line)
                # Add midrule after the evaluation header
                processed_lines.append("\\midrule")

                # Keep the data row without the first column
                processed_lines.append(rest_of_row)
                continue

        # Check if this is the header row with column index names (e.g., "graph_type & ER & HH...")
        # We want to remove this row, but keep rows with actual data headers like "Approximation Ratio"
        if in_header and "&" in line and "\\\\" in line:
            # Check if this line contains actual data headers (multicolumn with meaningful names)
            if (
                "Approximation Ratio" in line
                or "Num. Instances" in line
                or "graph\\_type" in line
            ):
                # This is a data header row with multicolumn - keep it but remove first column
                col_match = re.match(r"^\s*[^&]*&\s*(.+)$", line)
                if col_match:
                    new_line = col_match.group(1)
                    _graph_type = "graph\\_type"
                    new_line = new_line.replace(_graph_type, " " * len(_graph_type))
                    # We want headings to be centred, and to have an appropriate
                    # border. The borders aren't included by default as the
                    # headings use multicolumn.
                    if "Approximation Ratio" in line or "Num. Instances" in line:
                        new_line = new_line.replace("{r}", "{c|}")
                    processed_lines.append(new_line)
                    continue
            elif "method" in line:
                # Skip line as we don't want this 'empty' line.
                continue

        # For all other rows with content, remove the first column
        # This includes header rows and regular data rows
        if "&" in line and "\\\\" in line:
            # Match and remove first column: anything before first &
            col_match = re.match(r"^\s*[^&]*&\s*(.+)$", line)
            if col_match:
                new_line = col_match.group(1)
                processed_lines.append(new_line)
                continue

        # Mark end of header when we hit midrule after toprule
        if "\\midrule" in line and in_header:
            in_header = False

        # Keep lines without & (like \toprule, \midrule, \bottomrule, etc.) as-is
        processed_lines.append(line)

    return "\n".join(processed_lines)

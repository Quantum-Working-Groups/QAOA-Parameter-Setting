"""Code to format a Pandas table generated from SummaryTable."""

from abc import abstractmethod
from functools import partial
from typing import Any, Literal, TypeVar

import pandas as pd
from pandas.io.formats.style import Styler, _background_gradient

from qaoa_parameter_setting.utils.summary.summary_table import SummaryTable

__slots__ = ["formatted_styler_for"]


def __escape_underscores_index(obj):
    """Escape underscores in Index or MultiIndex levels.

    Converts non-string labels to strings first.
    """

    def escape(x):
        s = str(x)
        return s.replace("_", r"\_")

    if isinstance(obj, pd.MultiIndex):
        # Apply escaping to each level
        levels = [lvl.map(escape) for lvl in obj.levels]
        # Rebuild the MultiIndex using the same codes
        return pd.MultiIndex(
            levels=levels,
            codes=obj.codes,
            names=[escape(n) if n is not None else None for n in obj.names],
        )
    else:
        # Simple Index
        return obj.map(escape).set_names(
            [escape(n) if n is not None else None for n in obj.names]
        )


def escape_underscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Escape underscores in both columns and index labels."""
    df = df.copy()
    df.columns = __escape_underscores_index(df.columns)
    df.index = __escape_underscores_index(df.index)
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
            import re

            pattern = re.compile(r".*{([\d\.]+)}{([\d\.]+)}")
            match = pattern.search(val)
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
            import re

            pattern = re.compile(r".*{([\d\.]+)}")
            match = pattern.search(val)
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
            import re

            pattern = re.compile(r".*{([\d\.]+)\+-([\d\.]+)}")
            match = pattern.search(val)
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
    """Compute the mean value and format as a single number."""

    def __call__(self, vals: pd.Series) -> str:
        _mean = vals.mean()
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
    """Calculate the mean and standard deviation and format as the mean+-std dev.

    The exact format of ``+-`` is determined by :attr:`format`."""

    def __call__(self, vals: pd.Series) -> Any:
        _count = vals.count()
        _mean = vals.mean()
        _std = vals.std()
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
    depth: int | None,
    agg_values: Literal["approximation_ratio", "num_instances"],
    with_fancy_values: bool = False,
    cmap: str = "Greens",
    target_format: Literal["text", "latex", "siunitx"] = "text",
    precision: int = 4,
    missing_data_str: str = "-",
) -> tuple[pd.DataFrame, Styler]:

    # *** Pivot dataframe and aggregate appropriately
    _data = table.to_dataframe()
    if agg_values == "num_instances":
        # We use num_nodes as the dataframe column to aggregate so that we can
        # include node counts in the fancy value.
        __agg_values = "num_nodes"
        if with_fancy_values:
            aggregator = FancyCountRangeAggregator(
                precision=precision, format=target_format
            )
        else:
            aggregator = CountAggregator(
                precision=precision,
                format=target_format,
            )
    elif agg_values == "approximation_ratio":
        if with_fancy_values:
            aggregator = FancyStdDevAggregator(
                precision=precision, format=target_format
            )
        else:
            aggregator = MeanAggregator(
                precision=precision,
                format=target_format,
            )
        __agg_values = agg_values
    else:
        # We shouldn't get here, but it's good practice to handle this default
        # branch of the if statements.
        raise ValueError(
            "agg_values {!r} must be either 'num_instances' or 'approximation_ratio'.".format(
                agg_values
            )
        )

    # *** Filter based on depth
    if depth is not None:
        _data = _data[_data["depth"] == str(depth)]

    # *** Pivot table
    # Pivot and use aggregator
    pivot = _data.pivot_table(
        columns=[
            "graph_type",
            "depth",
        ],
        index=[
            "evaluation",
            "trainer_config",
        ],
        values=__agg_values,
        aggfunc=aggregator,  # type:ignore
    )
    # Recreate indices to ensure we include all methods.
    pivot = pivot.reindex(
        pd.MultiIndex.from_tuples(
            sorted(
                [
                    (table.trainer_config_to_evaluation(_method), _method)
                    for _method in table._methods
                ]
            )
        )
    )
    # Rename columns to use graph types from table.
    pivot = pivot.rename(columns=table.formatter_graph_type())
    # If we are targeting LaTeX, escape underscores.
    if target_format in ["latex", "siunitx"]:
        pivot = escape_underscore_df(pivot)

    # *** Handle styler colours, precision, etc.
    # Set format options
    styler = pivot.style.format(
        # Set string for missing values, represented by nan.
        na_rep=missing_data_str,
        # We're escaping characters with AggregatorFormatter, not Styler.
        escape=None,
    )

    # *** Set background colours
    if agg_values in ["approximation_ratio", "num_instances"] and with_fancy_values:
        # If we're running with fancy values, we need to apply colours based on
        # the "base" value. For this we map back to the base value with
        # _unformatted_background_gradient. Furthermore, we need to set the
        # limits on the colours from the entire table and not each column, which
        # is the default for background_gradient.

        # Map the dataframe back to the base data and compute the min and max.
        _base_data = _unformat_data(pivot, aggregator=aggregator)
        # Compute min and max, twice as the first returns the min/max for each column.
        _vmin = _base_data.min().min()
        _vmax = _base_data.max().max()

        # Apply a custom function that applies background_gradient using the
        # "base" value of the data. See _unformatted_background_gradient and
        # _unformat_data.
        styler = styler.apply(
            partial(_unformatted_background_gradient, aggregator=aggregator),
            vmin=_vmin,
            vmax=_vmax,
            cmap=cmap,
            # Following parameters taken from implementation of Styler.background_gradient.
            axis=0,
            subset=slice(None),
            low=0,
            high=0,
            text_color_threshold=0.408,
            gmap=None,
        )
    else:
        # Colours can be automatic, so use Styler.background_gradient. But we
        # need to set vmin and vmax to use the range of the entire table.
        # Call min/max twice as the first returns a series of values.
        _vmin = pivot.min().min()
        _vmax = pivot.max().max()
        styler = styler.background_gradient(
            cmap,
            vmin=_vmin,
            vmax=_vmax,
        )

    # Missing values should not have a colour, which is the default with
    # Styler.background_gradient
    styler = styler.highlight_null(props="background-color:white;color:black")
    return pivot, styler

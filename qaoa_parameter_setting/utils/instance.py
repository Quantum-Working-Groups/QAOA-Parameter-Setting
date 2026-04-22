"""Code to process graph instance filenames."""

from collections.abc import Callable
from os.path import basename
import re
from typing import Literal, NoReturn, TypeVar, overload


R = TypeVar("R")


@overload
def __regex_pattern_for(
    pattern: re.Pattern[str],
    error_noun: str,
    match_parse: Callable[[re.Match[str]], R],
    return_none_instead: Literal[False],
) -> Callable[[str], R | NoReturn]: ...
@overload
def __regex_pattern_for(
    pattern: re.Pattern[str],
    error_noun: str,
    match_parse: Callable[[re.Match[str]], R],
    return_none_instead: Literal[True],
) -> Callable[[str], R | None]: ...
def __regex_pattern_for(
    pattern: re.Pattern[str],
    error_noun: str,
    match_parse: Callable[[re.Match[str]], R],
    return_none_instead: bool = False,
) -> Callable[[str], R | None | NoReturn]:
    """Generate a function that parses a string with a regex pattern and raises an error if no matches occur.

    ``pattern`` must have at least one group. The :class:`re.Match` object
    returned by ``pattern.search`` is passed to ``match_parse``. The value
    returned by ``match_parse`` is then the returned value from the generated
    function.

    Args:
        pattern: The Regex pattern.
        error_noun: String for the raised error message, identifying what was not found.
        match_parse: Callable to convert a :class:`re.Match` object to a returned type.
        return_none_instead: If True, None will be returned instead of ``R`` if
            no match was found. If False and no match is found, an error is
            raised. Defaults to False.

    Returns:
        A callable that takes in a string, the path to a graph instance, and
        returns an extracted value parsed by ``match_parse``.
    """

    def __func(path: str) -> R | None | NoReturn:
        filename: str = basename(path)
        _match = pattern.search(filename)
        if _match is None:
            if return_none_instead:
                return None
            raise ValueError(
                "Could not determine {} from graph instance path {!r}.".format(
                    error_noun, path
                )
            )
        return match_parse(_match)

    return __func


re_num_nodes = re.compile(r"([\d]+)nodes")
num_nodes = __regex_pattern_for(
    re_num_nodes,
    "number of nodes",
    match_parse=lambda x: int(x.group(1)),
    return_none_instead=False,
)
"""Return the number of nodes for a graph instance at ``path``.

Number of nodes are assumed to be stored in the filename as an integer followed
by ``"nodes"``.

Args:
    path: Path to the instance filename.

Raises:
    ValueError: If pattern is not found in the path.

Returns:
    The number of nodes in the graph instance as determined by the filename.
"""

re_graph_idx = re.compile(r"^([\d]+)_")
graph_idx = __regex_pattern_for(
    re_graph_idx,
    "graph index",
    match_parse=lambda x: int(x.group(1)),
    return_none_instead=False,
)
"""Return the graph index for a graph instance at ``path``.

The graph index are assumed to be an integer at the beginning of the filename.

Args:
    path: Path to the instance filename.

Raises:
    ValueError: If pattern is not found in the path.

Returns:
    The graph index as determined by the filename.
"""


re_swap_layers = re.compile(r"([\d]+)swap_layers")
num_swap_layers = __regex_pattern_for(
    re_swap_layers,
    "number of nodes",
    match_parse=lambda x: int(x.group(1)),
    return_none_instead=True,
)
"""Return the number of swap layers for a graph instance at ``path``.

The number of swap layers is assumed to be an integer  followed by
``"swap_layers"``.

Args:
    path: Path to the instance filename.

Returns:
    The number of swap layers as determined by the filename. If no match is
    found, None is returned instead.
"""

re_heavyhex_dimensions = re.compile(r"([\d]+)_([\d]+)_heavyhex")
heavy_hex_dimensions = __regex_pattern_for(
    re_heavyhex_dimensions,
    "graph dimensions",
    match_parse=lambda x: (int(x.group(1)), int(x.group(2))),
    return_none_instead=True,
)
"""Return the heavy-hex dimensions as the tuple ``(rows, cols)``.

The dimensions are assumed to be integers in the following f-string
``f"{rows}_{cols}_heavyhex"``.

Args:
    path: Path to the instance filename.

Returns:
    The Heavy Hex dimensions as determined by the filename. If no match is
    found, None is returned instead.
"""

re_regular_degree = re.compile(r"random([\d]+)regular")
regular_degree = __regex_pattern_for(
    re_regular_degree,
    "degree",
    match_parse=lambda x: int(x.group(1)),
    return_none_instead=True,
)
"""Return the degree of a random regular graph filename.

The degree is assumed to be the integer ``degree`` in the f-string
``f"random{degree}regular"``.

Args:
    path: Path to the instance filename.

Returns:
    The degree of the random regular graph, as determined by the filename. If no
    match is found, None is returned instead.
"""

re_edge_probability = re.compile(r"([\d]+)percent")
edge_probability = __regex_pattern_for(
    re_edge_probability,
    "edge probability",
    match_parse=lambda x: float(x.group(1)) / 100,
    return_none_instead=True,
)
"""Return probability of an edge from a graph instance filename.

The probability is assumed to be an integer followed by the string "percent".

Args:
    path: Path to the instance filename.

Returns:
    The probability, as determined by the filename. If no match is found, None
    is returned instead.
"""

GraphType = Literal["erdos_renyi", "random_regular", "line_to_full", "heavy_hex"]


GRAPH_TYPE_REGEX_MAPPING: dict[GraphType, re.Pattern[str]] = {
    "erdos_renyi": re.compile(r"erdosrenyi"),
    "random_regular": re.compile(r"random([\d]+)regular"),
    "heavy_hex": re.compile("heavyhex"),
    "line_to_full": re.compile(r"swap_layers"),
}


def graph_type(path: str) -> GraphType:
    """Get the graph type from the instance path.

    Args:
        path: The path to the graph instance JSON file.

    Raises:
        ValueError: If the graph type cannot be determined because none of the
            graph type patterns match.
        ValueError: If the graph type cannot be determined because multiple
            patterns match.

    Returns:
        The graph type for the given instance JSON.
    """
    filename = basename(path)
    matches: dict[GraphType, bool] = {
        _graph_type: _pattern.search(filename) is not None
        for _graph_type, _pattern in GRAPH_TYPE_REGEX_MAPPING.items()
    }
    num_matches = list(matches.values()).count(True)
    if num_matches == 0:
        raise ValueError(
            "Cannot determine graph type from path {!r}; no matches to patterns.".format(
                path
            )
        )
    elif num_matches > 1:
        raise ValueError(
            "Path matches multiple graph types: path={!r} matches {}".format(
                path, ", ".join(matches.keys())
            )
        )
    else:
        return [k for k, _matches in matches.items() if _matches][0]

"""Allows us to parse the training data and summarise each method's performance."""

from collections.abc import Iterable
import glob
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, NoReturn, TypeAlias, TypedDict, cast, overload
import warnings

import numpy as np
import pandas as pd

from qaoa_parameter_setting.utils.graph_utils import maxcut_approximation_ratio
import qaoa_parameter_setting.utils.instance as Instance

from .utils import (
    GraphKey,
    Depth,
    ProblemClass,
    MethodConfigJSON,
    MethodJSON,
    MethodAcronym,
    guess_problem_class,
)


class ResultDict(TypedDict):
    """TypedDict for internal result dictionary structure."""

    energy: float
    qaoa_angles: list[float]
    result_filename: str
    trainer: str
    evaluation: str


SummaryData: TypeAlias = dict[GraphKey, dict[MethodConfigJSON, dict[Depth, ResultDict]]]
"""TypeAlias for :class:`SummaryTable` internal data type hinting."""


class MinMaxResult(TypedDict):
    """TypedDict for dict/JSON storing min-max cut information.

    Represents the dictionaries saved by ``compute_min_max_for_graph.py``.
    """

    max_cut: float
    min_cut: float
    sum_of_weights: float
    time_solve_max_cut_ns: float


MinMaxData: TypeAlias = dict[GraphKey, MinMaxResult]
"""TypeAlias for min-max cut data for graph instances."""


def sanitize_instance_key(key: str) -> str:
    """Convert instance keys, i.e., paths, into Posix paths for consistency.

    Args:
        key: The instance key, which can be a windows or posix path.

    Returns:
        ``key`` as an equivalent posix path.
    """
    if "/" in key:
        # Probably a PosixPath, but we convert to make sure.
        return str(PurePosixPath(key))
    if "\\" in key:
        # Probably a WindowsPath, but we convert to make sure.
        return str(PurePosixPath(PureWindowsPath(key)))
    # Cannot determine path type, so we try with the current system Path type.
    return str(PurePosixPath(Path(key)))


def sanitize_energy(energy_val: float | None | Literal["NA"]) -> float | None:
    """Sanitize an energy value from a results dictionary.

    Args:
        energy_val: Energy from a results dictionary, which can be a floating value, None, or ``"NA"``.

    Raises:
        ValueError: If the energy is an unknown string.
        ValueError: If the energy is an unknown type.

    Returns:
        A float value or None, representing the input energy. ``"NA`` is mapped to None.
    """
    if isinstance(energy_val, str):
        if energy_val == "NA":
            return None
        else:
            raise ValueError(f"Unknown energy value: {energy_val!r}")
    elif energy_val is None or isinstance(energy_val, (float, np.floating)):
        return energy_val
    else:
        raise ValueError(f"Unknown energy value type: {type(energy_val)!r}")


class SummaryTable:
    """Manages the construction of a summary table.

    The data is a nested dictionary with the following levels:
    - graph key
    - training method config
    - QAOA depth

    The value at the innermost level is a dictionary with the entries `energy`,
    `qaoa_angles`, and `result_filename`. Here, `result_filename` corresponds to
    the file in which we found the best parameters for a given QAOA depth, .
    """

    _problem_class: ProblemClass | None

    def __init__(
        self,
        filename: str | None = None,
        problem_class: ProblemClass | None = None,
    ):
        """Initialize the summary table instance.

        Args:
            filename: Filename for the saved summary table, to load preprocessed
                data.
            problem_class: List of acceptable problem class strings and/or
                None. All problem classes, as determined by
                :fun:`guess_problem_class`, that exist in this list are
                accepted when adding training data. Single strings are inserted
                into an empty list. If None, not ``[None]``, then all problem
                classes are accepted. Defaults to None.
        """
        self._data: SummaryData = {}
        self._minmax_data: MinMaxData = {}
        self._methods: list[str] = []
        self._additional_methods: set[str] = set()
        """Set of additional methods that may not have a method JSON.

        The set consists of methods from training data where a method JSON is not known and no-opt
        methods from the zeroth iteration of an opt method.
        """

        self._problem_class: ProblemClass | None = problem_class

        if filename is not None:
            with open(filename, "r") as fin:
                _data = json.load(fin)
                _loaded_data = _data["data"]
                # Previous versions had Depth:str instead of Depth:int, so we
                # need to convert them here.
                self._data = {
                    _graph_key: {
                        _trainer_config: {
                            # This is the only change we need to ensure.
                            int(_depth): _result
                            for _depth, _result in _trainer_dict.items()
                        }
                        for _trainer_config, _trainer_dict in _graph_dict.items()
                    }
                    for _graph_key, _graph_dict in _loaded_data.items()
                }

                self._minmax_data = _data["minmax_data"]
                self._methods = _data["methods"]

                # The default is None, where we accept all. This allows us to
                # handle old saved JSON tables, where all data was included.
                self._problem_class = _data.get("problem_class", None)

    @property
    def problem_class(self) -> ProblemClass | None:
        """Return the problem class of the summary table."""
        return self._problem_class

    @property
    def data(self) -> SummaryData:
        """Return the data."""
        return self._data

    def trainer_config_to_evaluation(self, trainer_config: MethodConfigJSON) -> str:
        """Convert a trainer config to an abbreviation of the evaluation method."""
        if "PP" in trainer_config:
            return "PP"
        elif "MPS" in trainer_config:
            return "MPS"
        elif "SV" in trainer_config:
            return "SV"
        else:
            raise ValueError(
                f"Unrecognised energy evaluation for method {trainer_config}"
            )

    def trainer_config_to_method(self, trainer_config: MethodConfigJSON) -> MethodJSON:
        """Convert a trainer config to an evaluation-independent string."""
        return MethodJSON(
            trainer_config.replace(
                "_{}".format(self.trainer_config_to_evaluation(trainer_config)), ""
            )
        )

    def method_uses_aer(self, method_config: str) -> bool:
        """Return if the given method_config uses AER."""
        return "Aer" in method_config

    def result_contains_noopt(self, result: dict[str, Any]) -> bool:
        """Returns if the given results dictionary contains a trainer whose zeroth iteration is a _noOpt run.

        This function first checks if the trainer JSON file is a TQA or FA
        config, then it checks if _opt is in the config JSON filename. If both
        are true, it returns True. Otherwise, it returns False.

        Args:
            result: The results dictionary.

        Returns:
            True if the zeroth iteration is a _noOpt run.
        """
        config: str = result["args"]["config"]
        parts = Path(config).parts[-1].split(".")[0].split("_")
        no_opt_matches = ["TQA", "FA", "FAAer", "TQAAer"]
        if all(x not in parts for x in no_opt_matches):
            return False
        lower_parts = [p.lower() for p in parts]
        if "opt" not in lower_parts:
            return False
        if "noopt" in lower_parts or "no_opt" in config.lower():
            return False
        return True

    def trainer_config_to_no_opt(self, result: dict[str, Any]) -> str:
        """Return the ``no_opt`` variant of the trainer config filename

        Args:
            result: The results dictionary for an opt run.

        Returns:
            Modified trainer config name.
        """
        OPT_TO_NO_OPT_MAPPING = {
            # Known mappings between Opt and No-Opt methods.
            "FA_MPS_opt.json": "FA_MPS_no_opt.json",
            "FA_PP_opt.json": "FA_PP_no_opt.json",
            "FA_SV_opt.json": "FA_SV_noOpt.json",
            "TQA_MPS_opt.json": "TQA_MPS_no_opt.json",
            "TQA_PP_opt.json": "TQA_PP_no_opt.json",
            "TQA_SV_opt.json": "TQA_SV_noOpt.json",
            # These no-opt methods don't exist, but we give them names anyway.
            "FA_MPSAer_opt.json": "FA_MPSAer_no_opt.json",
            "TQA_MPSAer_opt.json": "TQA_MPSAer_no_opt.json",
        }
        opt_trainer_config = str(result["args"]["config"])
        noopt_trainer_config: str | None = None
        # Check if any opt method exists. If yes, replace it and break.
        for k in OPT_TO_NO_OPT_MAPPING.keys():
            if k in opt_trainer_config:
                noopt_trainer_config = opt_trainer_config.replace(
                    k, OPT_TO_NO_OPT_MAPPING[k]
                )
                break
        # If we don't have an explicit mapping, just replace 'opt' with 'no_opt'.
        if noopt_trainer_config is None:
            noopt_trainer_config = opt_trainer_config.replace("opt", "no_opt")
        return noopt_trainer_config

    def config_path_to_config(self, config_path: str) -> MethodConfigJSON:
        """Convert a string path to a config/method to just the config name."""
        return MethodConfigJSON(config_path.split("/")[-1])

    def add_data(self, folder_name: str):
        """Load the training data from a folder and get the best result.

        Best results are chosen as the data with minimum energy for each graph
        key, training method, and QAOA depth. If the training data problem_class
        does not match :attr:`_problem_class`, the data is ignored."""

        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")
            with open(filename, "r") as fin:
                result = json.load(fin)

            # Get the problem class
            problem_class = guess_problem_class(filename, result)
            if self._problem_class is not None and problem_class != self._problem_class:
                # Result problem class not in acceptable problem classes, so we
                # ignore these results.
                if problem_class is None:
                    # We only raise this warning if we aren't filtering by problem class.
                    warnings.warn(
                        "Result with filename {!r} has no problem class!".format(
                            filename
                        )
                    )
                continue

            # Get the energy evaluation methodology
            config: MethodConfigJSON = self.config_path_to_config(
                result["args"]["config"]
            )

            # Check if we must translate an opt.json to noOpt.json entry.

            try:
                evaluation = self.trainer_config_to_evaluation(config)
            except ValueError as e:
                raise ValueError(
                    f"Unrecognised energy evaluation in {filename} for method {config}"
                ) from e

            # Get the graph instance
            graph_input = result["args"]["input"].replace("\\", "/")
            graph: GraphKey = graph_input.split("/")[-1]

            # Loop over the trainers in the result
            results: list[tuple[list[float], float | None, str, MethodConfigJSON]] = []
            self.populate_results(result, results, filename=filename, config=config)

            if graph not in self._data:
                self._data[graph] = dict()
            if config not in self._data[graph]:
                self._data[graph][config] = dict()

            # Track this config in additional_methods if it's not in methods
            # This ensures configs with data but no method JSON file are included
            if config not in self._methods and config not in self._additional_methods:
                self._additional_methods.add(config)

            # Check if we need to add the zeroth iteration, for opt to no-opt
            # trainer mappings.
            if self.result_contains_noopt(result):
                no_opt_config = self.config_path_to_config(
                    self.trainer_config_to_no_opt(result)
                )
                self._additional_methods.add(self.config_path_to_config(no_opt_config))
                self.populate_results(
                    result,
                    results,
                    filename=filename,
                    config=no_opt_config,
                    # Only use the zeroth entry.
                    iter_keys=["0"],
                )
                if no_opt_config not in self._data[graph]:
                    self._data[graph][no_opt_config] = dict()

            res_config: MethodConfigJSON
            for qaoa_angles, energy, trainer, res_config in results:
                # res_config is MethodConfigJSON from populate_results
                if qaoa_angles is None or energy is None:
                    continue

                depth = len(qaoa_angles) // 2

                # Note that depth is stored as a string in the results.
                if depth in self._data[graph][res_config]:
                    if self._data[graph][res_config][depth]["energy"] < energy:
                        self._data[graph][res_config][depth] = {
                            "energy": energy,
                            "qaoa_angles": qaoa_angles,
                            "result_filename": filename.split("/")[-1],
                            "trainer": trainer,
                            "evaluation": evaluation,
                        }
                else:
                    self._data[graph][res_config][depth] = {
                        "energy": energy,
                        "qaoa_angles": qaoa_angles,
                        "result_filename": filename.split("/")[-1],
                        "trainer": trainer,
                        "evaluation": evaluation,
                    }

    def add_methods(self, folder_name: str):
        """Add method/trainer config files from a folder.

        This is needed so we can identify methods that have not been tested on
        certain graphs and QAOA depths.
        """
        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")
            # Ignore the example json
            if self.config_path_to_config(filename) == "example_method.json":
                continue
            if filename in self._methods:
                continue
            try:
                with open(filename, "r") as fin:
                    _data = json.load(fin)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    "Failed to read method JSON file {!r}.".format(filename)
                ) from e
            if "trainer_chain" not in _data:
                raise RuntimeError(
                    "Method at {!r} does not appear to be a method json file.".format(
                        filename
                    )
                )
            self._methods.append(self.config_path_to_config(filename))

    def all_methods(self) -> list[MethodConfigJSON]:
        methods = [m for m in self._methods]
        methods.extend(self._additional_methods)
        return [MethodConfigJSON(m) for m in sorted(set(methods))]

    def add_minmax_cut_data(self, folder_name: str, replace: bool = False):
        """Load the min-max cut data from a folder.

        Args:
            replace: Whether to replace any existing minmax data stored in
                :class:`SummaryTable`. If False, an error is raised when minmax
                data for an already stored instance is found.
        """
        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")
            with open(filename, "r") as fin:
                _data = json.load(fin)
            graph_path = sanitize_instance_key(_data.pop("instance"))
            graph_key: GraphKey = graph_path.split("/")[-1]
            result = MinMaxResult(_data)
            if graph_key in self._minmax_data and not replace:
                raise KeyError(
                    "Multiple min-max cut data for instance {!r} and replace=False.".format(
                        graph_key
                    )
                )
            if set(MinMaxResult.__required_keys__) != set(result.keys()):
                raise ValueError(
                    "min-max cut JSON for {!r} does not appear to have the".format(
                        filename
                    )
                    + " expected keys."
                )
            self._minmax_data[graph_key] = result

    @staticmethod
    def get_iter_keys(keys: Iterable[str]) -> list[str]:
        """Return a list from an iterable, validating that they're all strings
        that can be converted to integers."""
        iterations: list[str] = []
        for key in keys:
            try:
                _ = int(key)
                iterations.append(key)
            except ValueError:
                pass

        return iterations

    @overload
    def maxcut_approximation_ratio(
        self,
        graph_key: GraphKey,
        energy: float,
        return_none: Literal[False],
    ) -> float | NoReturn: ...
    @overload
    def maxcut_approximation_ratio(
        self,
        graph_key: GraphKey,
        energy: float,
        return_none: Literal[True],
    ) -> float | None: ...
    def maxcut_approximation_ratio(
        self, graph_key: GraphKey, energy: float, return_none: bool = False
    ) -> float | None | NoReturn:
        if graph_key not in self._minmax_data:
            if return_none:
                return None
            else:
                raise KeyError(
                    "graph instance {!r} does not have min-max cut data loaded.".format(
                        graph_key
                    )
                )

        # Compute approximation ratio
        min_cut = self._minmax_data[graph_key]["min_cut"]
        max_cut = self._minmax_data[graph_key]["max_cut"]
        sum_weights = self._minmax_data[graph_key]["sum_of_weights"]
        return maxcut_approximation_ratio(
            min_cut=min_cut, max_cut=max_cut, sum_weights=sum_weights, energy=energy
        )

    def trainer_depth_combinations(
        self, graph_keys: list[str] | None = None
    ) -> list[tuple[str, int]]:
        """List of all combinations of trainer config filenames and QAOA depths
           present in the data.

        Args:
            graphs: Optional list of graphs to select over. Returned
                combinations is only over trainer-depth pairs for the selected
                graphs. If None, all graphs are selected. Defaults to None.

        Returns:
            List of all combinations of trainer configs and depths in the data,
            for the selected graphs.
        """
        _combinations: set[tuple[str, int]] = set()
        if graph_keys is None:
            graph_keys = list(self._data.keys())
        for graph_key in graph_keys:
            sub_data = self._data[graph_key]
            for config, sub_config_data in sub_data.items():
                for depth in sub_config_data.keys():
                    _combinations.add((config, depth))
        return list(sorted(_combinations))

    def missing_trainer_depth_combinations(
        self, graph_keys: list[str] | None = None
    ) -> dict[str, list[tuple[str, int]]]:
        combinations = self.trainer_depth_combinations(graph_keys=None)
        missing: dict[str, list[tuple[str, int]]] = {}
        if graph_keys is None:
            graph_keys = list(self._data.keys())
        for graph_key in graph_keys:
            missing[graph_key] = list(
                set(combinations) - set(self.trainer_depth_combinations([graph_key]))
            )
        return missing

    def missing_minmax_cuts(self) -> list[GraphKey]:
        """List of graph keys/JSON filenames that do not have min-max cuts data loaded."""
        return [graph for graph in self._data.keys() if graph not in self._minmax_data]

    def populate_results(
        self,
        result: dict[str, Any],
        result_data: list[tuple[list[float], float | None, str, MethodConfigJSON]],
        filename: str | None = None,
        iter_keys: Iterable[str] | None = None,
        config: MethodConfigJSON | None = None,
    ):
        """Get the parameters from the `result` dict.

        Args:
            result: Input results dictionary to process.
            result_data: Output list of results.
            filename: Optional name of original JSON file for results, for
            warnings. If None, warnings will not refer to a specific file.
            iter_keys: Iterable of keys for sub-results. If None, generated with
            :meth:`get_iter_keys`.
        """
        if iter_keys:
            __iter_keys = self.get_iter_keys(iter_keys)
        else:
            __iter_keys = self.get_iter_keys(result.keys())

        if config is None:
            __config = result.get("args", {}).get("config", None)
            if __config is None:
                raise ValueError("Cannot determine config.")
            config = self.config_path_to_config(__config)

        for key in __iter_keys:
            sub_result = result[key]

            version = sub_result["system_info"]["qaoa_training_pipeline_version"]
            trainer = sub_result["trainer"]["trainer_name"]

            # Skip known bugs
            if trainer == "TQATrainer" and version < 16:
                # Double check if the TQATrainer bug in #35 is fixed for this subtrainer.
                if not sub_result["system_info"].get("tqa_trainer_fix_applied", False):
                    warnings.warn(
                        "Result JSON not corrected for TQATrainer bug, for qaoa-training-pipeline<16. Use `fix_saved_tqatrainer.py`"
                        + (
                            " on {!r}.".format(filename)
                            if filename is not None
                            else "."
                        )
                    )

            if trainer == "RecursionTrainer":
                self.populate_results(
                    sub_result, result_data, filename=filename, config=config
                )

            qaoa_angles = sub_result["optimized_qaoa_angles"]
            energy = sanitize_energy(energy_val=sub_result["energy"])

            # Ensure length of 2p
            if len(qaoa_angles) % 2 != 0:
                continue

            result_data.append((qaoa_angles, energy, trainer, config))

    def save_data(self, filename: str, overwrite: bool = False):
        """Dump the data to a file."""

        if os.path.isfile(filename) and not overwrite:
            raise ValueError(f"File {filename} already exists.")

        with open(filename, "w") as fout:
            json.dump(
                {
                    "data": self._data,
                    "minmax_data": self._minmax_data,
                    "methods": self._methods,
                    "problem_class": self._problem_class,
                },
                fout,
            )

    def get_monotonic_data(
        self,
    ) -> SummaryData:
        """Return data that only contains depths for which `E_p > E_q` for `p > q`."""
        monotonic_data = dict()

        for graph_key, mdict in self._data.items():
            if graph_key not in monotonic_data:
                monotonic_data[graph_key] = dict()

            for trainer_config, vdict in mdict.items():
                if trainer_config not in monotonic_data[graph_key]:
                    monotonic_data[graph_key][trainer_config] = dict()

                depths = [v for v in sorted([int(d) for d in vdict.keys()])]

                prev_energy: float | None = None
                for depth in depths:
                    if prev_energy is None:
                        monotonic_data[graph_key][trainer_config][depth] = vdict[depth]
                        prev_energy = float(vdict[depth]["energy"])

                    elif prev_energy < float(vdict[depth]["energy"]):
                        monotonic_data[graph_key][trainer_config][depth] = vdict[depth]
                        prev_energy = float(vdict[depth]["energy"])
                    else:
                        continue

        return monotonic_data

    def __result_to_record(
        self,
        graph_key: GraphKey,
        config: MethodConfigJSON,
        depth: Depth,
        result: ResultDict,
    ) -> dict[str, Any]:
        _heavy_hex_dimensions = Instance.heavy_hex_dimensions(graph_key)
        _approx_ratio = self.maxcut_approximation_ratio(
            graph_key=graph_key, energy=result["energy"], return_none=True
        )

        # Compute normalized energy for MIS problems
        _num_nodes = Instance.num_nodes(graph_key)
        _energy = result["energy"]
        _normalized_energy = (
            _energy / _num_nodes
            if _num_nodes is not None and self._problem_class == "MIS"
            else None
        )

        _method = self.trainer_config_to_method(config)

        return {
            # Pandas will put the first keys as the first columns, so we put the
            # important columns first.
            "graph_type": Instance.graph_type(graph_key),
            "graph_idx": Instance.graph_idx(graph_key),
            "num_nodes": Instance.num_nodes(graph_key),
            "trainer_config": config,
            "method": _method,
            "depth": depth,
            "uses_aer": self.method_uses_aer(_method),
            # We now put other _key_ information that is dependent on the graph type.
            "edge_probability": Instance.edge_probability(graph_key),
            "regular_degree": Instance.regular_degree(graph_key),
            "heavy_hex_rows": (
                _heavy_hex_dimensions[0] if _heavy_hex_dimensions is not None else None
            ),
            "heavy_hex_cols": (
                _heavy_hex_dimensions[1] if _heavy_hex_dimensions is not None else None
            ),
            "num_swap_layers": Instance.num_swap_layers(graph_key),
            # We also want the graph_key JSON file.
            "graph_key": graph_key,
            # We now include the results information.
            **result,
            "approximation_ratio": (
                _approx_ratio if _approx_ratio is not None else None
            ),
            "energy": _energy,
            "normalized_energy": _normalized_energy,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                self.__result_to_record(
                    graph_key=graph_key, config=config, depth=depth, result=result
                )
                for graph_key, graph_result in self._data.items()
                for config, config_data in graph_result.items()
                for depth, result in config_data.items()
            ]
        )

    def formatter_graph_type(self) -> dict[GraphKey, str]:
        return {
            "erdos_renyi": "ER",
            "random_regular": "RR",
            "heavy_hex": "HH",
            "line_to_full": "L2F",
        }

    def get_graph_instances_for_config(
        self,
        trainer_method: MethodJSON,
        graph_type: str | None = None,
        depth: Depth | None = None,
        evaluation_method: str | None = None,
    ) -> set[GraphKey]:
        """Get all graph instances for a given trainer config.

        Args:
            trainer_method: The trainer method to query (e.g., "LR_opt.json").
            graph_type: Optional graph type filter (e.g., "random_regular", "erdos_renyi").
                If None, all graph types are included.
            depth: Optional depth filter. If None, all depths are included.
            evaluation_method: Optional evaluation method filter (e.g., "SV", "MPS", "PP").
                If None, all evaluation methods are included.

        Returns:
            Set of graph keys (instance filenames) that have results for the given
            trainer method, optionally filtered by graph type, depth, and evaluation method.
        """
        instances: set[GraphKey] = set()

        for graph_key, graph_data in self._data.items():
            # Filter by graph type if specified
            if graph_type is not None:
                if Instance.graph_type(graph_key) != graph_type:
                    continue

            # Check if this trainer config exists for this graph. The trainer configs in graph_data
            # must be mapped to methods, but we need the trainer config to index the graph data
            # later. Also filter by evaluation method if specified.
            trainer_config: MethodConfigJSON | None = None
            for __trainer_config in graph_data.keys():
                __current_method = self.trainer_config_to_method(__trainer_config)
                if trainer_method == __current_method:
                    # If evaluation_method is specified, check if it matches
                    if evaluation_method is not None:
                        __current_eval = self.trainer_config_to_evaluation(
                            __trainer_config
                        )
                        if __current_eval != evaluation_method:
                            continue
                    trainer_config = __trainer_config
                    break
            if trainer_config is None:
                continue

            # Filter by depth if specified
            if depth is not None:
                if depth not in graph_data[trainer_config]:
                    continue

            instances.add(graph_key)

        return instances

    def filter_to_common_instances(
        self,
        reference_methods: dict[tuple[str, str], MethodJSON],
        depth: Depth | None = None,
    ) -> "SummaryTable":
        """Filter data to only include graph instances common to reference configs.

        For each (evaluation_method, graph_type) pair, this method identifies the
        graph instances present in the reference training method and filters all
        other training methods for that (evaluation_method, graph_type) to only
        include those same instances.

        Args:
            reference_methods: Dictionary mapping (evaluation_method, graph_type)
                tuples to reference trainer config names. For example:
                {("SV", "random_regular"): "F_SV.json", ("MPS", "heavy_hex"): "I_MPS.json"}
            depth: Optional depth to filter on. If None, filtering is done across
                all depths.

        Returns:
            A new SummaryTable instance with filtered data.
        """
        # Create a new SummaryTable with the same metadata
        filtered_table = SummaryTable(problem_class=self._problem_class)
        filtered_table._methods = self._methods.copy()
        filtered_table._additional_methods = self._additional_methods.copy()
        filtered_table._minmax_data = self._minmax_data.copy()

        # Build a mapping of (evaluation, graph_type) -> set of allowed instances
        allowed_instances: dict[tuple[str, str], set[GraphKey]] = {}

        for (eval_method, graph_type), ref_method in reference_methods.items():
            instances = self.get_graph_instances_for_config(
                trainer_method=ref_method,
                graph_type=graph_type,
                depth=depth,
                evaluation_method=eval_method,
            )
            allowed_instances[(eval_method, graph_type)] = instances

        # Filter the data
        for graph_key, graph_data in self._data.items():
            current_graph_type = Instance.graph_type(graph_key)

            for trainer_config, config_data in graph_data.items():
                eval_method = self.trainer_config_to_evaluation(trainer_config)

                # Check if we have a reference config for this (eval, graph_type)
                key = (eval_method, current_graph_type)
                if key not in allowed_instances or graph_key in allowed_instances[key]:
                    # No filtering for this combination if key is not in allowed instances, or if
                    # this graph instance is in the allowed set.
                    if graph_key not in filtered_table._data:
                        filtered_table._data[graph_key] = {}
                    filtered_table._data[graph_key][trainer_config] = config_data.copy()

        return filtered_table

    def get_missing_instances_after_filter(
        self,
        reference_methods: dict[tuple[str, str], str],
        depth: Depth | None = None,
        num_nodes: int | list[int] | dict[str, int | list[int]] | None = None,
        methods_to_exclude: list[str] | None = None,
    ) -> dict[tuple[str, str, str], set[GraphKey]]:
        """Identify missing graph instances for each (eval_method, graph_type, trainer_method).

        After filtering to common instances based on reference methods, this identifies
        which graph instances are missing for each training method. This is useful for
        determining which experiments need to be run to complete the dataset.

        Args:
            reference_methods: Dictionary mapping (evaluation_method, graph_type)
                tuples to reference trainer method names (without evaluation suffix).
                For example: {("SV", "line_to_full"): "LR_opt.json"}
                The evaluation method from the key will be used to find the right config.
            depth: Optional depth to filter on. If None, checks across all depths.
            num_nodes: Optional filter for number of nodes. Can be:
                - int: Only include instances with this number of nodes
                - list[int]: Include instances with any of these numbers of nodes
                - dict[str, int | list[int]]: Per-evaluation method filtering
            methods_to_exclude: Optional list of methods to exclude when compiling list of missing
                instances. If None, all methods are included. Defaults to None.

        Returns:
            Dictionary mapping (eval_method, graph_type, trainer_method) tuples to sets
            of missing graph keys. Only includes entries where instances are missing.

        Example:
            >>> missing = table.get_missing_instances_after_filter(
            ...     {("SV", "line_to_full"): "LR_opt.json"},
            ...     depth=10
            ... )
            >>> # Returns: {("SV", "line_to_full", "FA"): {"000N40L2S8.json", ...}}
        """
        if methods_to_exclude is None:
            methods_to_exclude = []

        # Build mapping of (eval, graph_type) -> set of reference instances
        reference_instances: dict[tuple[str, str], set[GraphKey]] = {}

        for (eval_method, graph_type), ref_method in reference_methods.items():
            instances = self.get_graph_instances_for_config(
                trainer_method=MethodJSON(ref_method),
                graph_type=graph_type,
                depth=depth,
                evaluation_method=eval_method,
            )

            # Apply num_nodes filtering
            if num_nodes is not None:
                filtered_instances = set()
                for graph_key in instances:
                    graph_num_nodes = Instance.num_nodes(graph_key)
                    if graph_num_nodes is None:
                        continue

                    # Check if this instance should be included based on num_nodes filter
                    if isinstance(num_nodes, dict):
                        # Per-evaluation filtering
                        if eval_method not in num_nodes:
                            # No filter for this evaluation method, include all
                            filtered_instances.add(graph_key)
                        else:
                            allowed = num_nodes[eval_method]
                            if isinstance(allowed, int):
                                allowed = [allowed]
                            if graph_num_nodes in allowed:
                                filtered_instances.add(graph_key)
                    else:
                        # Global filtering
                        allowed_nodes = (
                            [num_nodes] if isinstance(num_nodes, int) else num_nodes
                        )
                        if graph_num_nodes in allowed_nodes:
                            filtered_instances.add(graph_key)

                instances = filtered_instances

            reference_instances[(eval_method, graph_type)] = instances

        # Find missing instances for each method
        missing: dict[tuple[str, str, str], set[GraphKey]] = {}

        # For each (eval, graph_type) pair with a reference
        for (
            eval_method,
            graph_type,
        ), expected_instances in reference_instances.items():
            # Get trainer methods that actually have data for this specific (eval_method, graph_type)
            # This ensures we only check methods that are compatible with the evaluation method
            relevant_trainer_methods = set()

            for graph_key, graph_data in self._data.items():
                # Extract trainer methods from configs that match this evaluation method
                for trainer_config in graph_data.keys():
                    try:
                        config_eval = self.trainer_config_to_evaluation(trainer_config)
                        # Only include if evaluation method matches
                        if config_eval == eval_method:
                            trainer_method = self.trainer_config_to_method(
                                trainer_config
                            )
                            # If this trainer method should be excluded, do not process it.
                            if trainer_method in methods_to_exclude:
                                continue
                            relevant_trainer_methods.add(trainer_method)
                    except ValueError:
                        # Skip configs with unrecognized evaluation methods
                        continue

            # Check each relevant trainer method
            for trainer_method in relevant_trainer_methods:
                # Get instances this method actually has
                actual_instances = self.get_graph_instances_for_config(
                    trainer_method=trainer_method,
                    graph_type=graph_type,
                    depth=depth,
                    evaluation_method=eval_method,
                )

                # Apply num_nodes filtering to actual instances
                if num_nodes is not None:
                    filtered_actual = set()
                    for graph_key in actual_instances:
                        graph_num_nodes = Instance.num_nodes(graph_key)
                        if graph_num_nodes is None:
                            continue

                        if isinstance(num_nodes, dict):
                            if eval_method not in num_nodes:
                                filtered_actual.add(graph_key)
                            else:
                                allowed = num_nodes[eval_method]
                                if isinstance(allowed, int):
                                    allowed = [allowed]
                                if graph_num_nodes in allowed:
                                    filtered_actual.add(graph_key)
                        else:
                            allowed_nodes = (
                                [num_nodes] if isinstance(num_nodes, int) else num_nodes
                            )
                            if graph_num_nodes in allowed_nodes:
                                filtered_actual.add(graph_key)

                    actual_instances = filtered_actual

                # Find missing instances
                missing_instances = expected_instances - actual_instances

                # Only include if there are missing instances
                if missing_instances:
                    key = (eval_method, graph_type, trainer_method)
                    missing[key] = missing_instances

        return missing

    def get_missing_instances_summary(
        self,
        reference_methods: dict[tuple[str, str], str],
        depth: Depth | None = None,
        num_nodes: int | list[int] | dict[str, int | list[int]] | None = None,
        methods_to_exclude: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get a summary DataFrame of missing instances after filtering.

        This provides a human-readable summary of which experiments need to be run,
        organized by evaluation method, graph type, and trainer method.

        Args:
            reference_methods: Dictionary mapping (evaluation_method, graph_type)
                tuples to reference trainer method names (without evaluation suffix).
                For example: {("SV", "line_to_full"): "LR_opt.json"}
            depth: Optional depth to filter on. If None, checks across all depths.
            num_nodes: Optional filter for number of nodes. Can be:
                - int: Only include instances with this number of nodes
                - list[int]: Include instances with any of these numbers of nodes
                - dict[str, int | list[int]]: Per-evaluation method filtering
            methods_to_exclude: Optional list of methods to exclude when getting the missing
                instances. These methods will be ignored when compiling the list of instances to
                run. If None, no methods are excluded.

        Returns:
            DataFrame with columns: eval_method, graph_type, trainer_method,
            num_missing, missing_instances (list of graph keys).

        Example:
            >>> summary = table.get_missing_instances_summary(
            ...     {("SV", "random_regular"): "TQA_SV_opt"},
            ...     num_nodes=40
            ... )
            >>> print(summary)
        """
        if methods_to_exclude is None:
            methods_to_exclude = []
        missing = self.get_missing_instances_after_filter(
            reference_methods=reference_methods,
            depth=depth,
            num_nodes=num_nodes,
            methods_to_exclude=methods_to_exclude,
        )

        records = []
        for (eval_method, graph_type, trainer_method), instances in missing.items():
            records.append(
                {
                    "eval_method": eval_method,
                    "graph_type": graph_type,
                    "trainer_method": trainer_method,
                    "num_missing": len(instances),
                    "missing_instances": sorted(list(instances)),
                }
            )

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.sort_values(
                ["eval_method", "graph_type", "num_missing", "trainer_method"],
                ascending=[True, True, False, True],
            )

        return df

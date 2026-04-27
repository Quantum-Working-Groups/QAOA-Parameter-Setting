"""Database class for storing and querying all QAOA training results."""

import glob
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NoReturn,
    TypeAlias,
    TypedDict,
    cast,
    overload,
)

import numpy as np
import pandas as pd

import qaoa_parameter_setting.utils as utils
from qaoa_parameter_setting.utils.types import (
    Depth,
    EvaluationType,
    GraphKey,
    GraphType,
    MethodConfigJSON,
    MethodJSON,
    ProblemClass,
)

NumNodesFilter: TypeAlias = (
    Iterable[int] | dict[str, Iterable[int] | dict[bool, Iterable[int]]]
)
"""Type for num_nodes filtering.

Can be:
- Iterable[int]: Simple list of node counts to filter by
- dict[str, Iterable[int] | dict[bool, Iterable[int]]]: Evaluation-specific filtering
  where the key is evaluation method ("SV", "MPS", "PP") and value is either:
  - Iterable[int]: Node counts for that evaluation
  - dict[bool, Iterable[int]]: For MPS, maps with_aer to node counts
"""


class MinMaxResult(TypedDict):
    """Dictionary representing min/max cut results for an instance.

    Attributes:
        max_cut: Maximum cut value.
        min_cut: Minimum cut value.
        sum_of_weights: Sum of all edge weights.
        time_solve_max_cut_ns: Time to solve max cut in nanoseconds.
    """

    max_cut: float
    min_cut: float
    sum_of_weights: float
    time_solve_max_cut_ns: float


class ResultsDatabase:
    """Database for storing and querying all QAOA training results.

    Unlike BestParameterManager (which stores best parameters per evaluation/instance/depth)
    and SummaryTable (which stores results per instance/method/depth), this class stores
    ALL results from all files, tracking the source file for each result.

    The data is a nested dictionary with the following levels:
    - instance name (graph key)
    - method config
    - depth
    - result index (for multiple results at same depth)

    Each result entry contains: energy, qaoa_angles, trainer, trainer_config, method,
    evaluation, evaluation_label, method_label, with_aer, source_file, train_duration,
    and metadata.
    """

    NONTRAINER_KEYS = {"args", "pre_processing", "cost_operator"}

    def __init__(
        self,
        file_name: str | None = None,
        problem_class: ProblemClass | None = None,
        num_nodes: NumNodesFilter | None = None,
    ):
        """Initialize the results database.

        Args:
            file_name: Optional filename to load saved data. If None, an empty database is created.
            problem_class: Problem class to filter by ("MC" or "MIS"). None means "MC" (MaxCut).
            num_nodes: Optional num_nodes filter. Can be:
                - Iterable[int]: Simple list of node counts
                - dict[EvaluationType, Iterable[int] | dict[bool, Iterable[int]]]: Evaluation-specific filtering
                - None: No filtering
        """
        self._data: defaultdict[
            GraphKey,
            defaultdict[MethodConfigJSON, defaultdict[Depth, list[dict[str, Any]]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._source_files: set[str] = set()
        """Set of all source files that have been added to the database."""
        self._additional_methods: set[MethodConfigJSON] = set()
        """Set of additional methods that may not have a method JSON.

        The set consists of methods from training data where a method JSON is not known and no-opt
        methods from the zeroth iteration of an opt method.
        """
        self._problem_class: ProblemClass = (
            problem_class if problem_class is not None else "MC"
        )
        """Problem class filter: 'MC' for MaxCut or 'MIS' for Maximum Independent Set."""
        self._num_nodes_filter: NumNodesFilter | None = num_nodes
        """Num_nodes filter: can be simple set or evaluation-specific dict."""
        self._minmax_data: dict[GraphKey, MinMaxResult] = {}
        """Min/max cut data for MaxCut instances."""

        if file_name is not None:
            with open(file_name, "r") as fin:
                _json_data = json.load(fin)
                loaded_data = self.__convert_str_depth_to_int_and_parse_datetime(
                    _json_data["data"]
                )
                # Convert loaded dict to defaultdict structure
                for instance, instance_data in loaded_data.items():
                    for method, method_data in instance_data.items():
                        for depth, results in method_data.items():
                            self._data[instance][method][depth] = results
                self._source_files = set(_json_data["source_files"])
                self._additional_methods = set(_json_data.get("additional_methods", []))

                # Load problem_class and num_nodes if present
                if "problem_class" in _json_data:
                    self._problem_class = _json_data["problem_class"]
                if "num_nodes_filter" in _json_data:
                    loaded_filter = _json_data["num_nodes_filter"]
                    # Convert string boolean keys back to actual booleans
                    if isinstance(loaded_filter, dict):
                        self._num_nodes_filter = {}
                        for eval_key, eval_value in loaded_filter.items():
                            if isinstance(eval_value, dict):
                                # Convert string keys "True"/"False" to boolean
                                self._num_nodes_filter[eval_key] = {
                                    (k == "True" if isinstance(k, str) else k): v
                                    for k, v in eval_value.items()
                                }
                            else:
                                self._num_nodes_filter[eval_key] = eval_value
                    else:
                        self._num_nodes_filter = loaded_filter

                # Load minmax_data if present
                if "minmax_data" in _json_data:
                    self._minmax_data = _json_data["minmax_data"]

    @property
    def problem_class(self) -> ProblemClass:
        """Problem class of the instances."""
        return self._problem_class

    def _should_include_by_num_nodes(
        self,
        instance_num_nodes: int,
        evaluation: str | None = None,
        with_aer: bool | None = None,
    ) -> bool:
        """Check if an instance should be included based on num_nodes filter.

        Args:
            instance_num_nodes: Number of nodes in the instance.
            evaluation: Evaluation method (e.g., "SV", "MPS", "PP"). Required if filter is dict.
            with_aer: Whether Aer is used. Required for MPS if filter is dict.

        Returns:
            True if instance should be included, False otherwise.
        """
        if self._num_nodes_filter is None:
            return True

        # Check if it's a simple iterable
        if not isinstance(self._num_nodes_filter, dict):
            return instance_num_nodes in set(self._num_nodes_filter)

        # It's a dict - need evaluation
        if evaluation is None:
            return False

        # Cast to dict to satisfy type checker
        filter_dict: dict[str, Iterable[int] | dict[bool, Iterable[int]]] = (
            self._num_nodes_filter
        )  # type: ignore[assignment]

        if evaluation not in filter_dict:
            return False

        filter_value = filter_dict[evaluation]

        # Check if filter_value is a simple iterable or dict
        if not isinstance(filter_value, dict):
            return instance_num_nodes in set(filter_value)

        # It's a dict - need with_aer
        if with_aer is None:
            return False

        if with_aer not in filter_value:
            return False

        return instance_num_nodes in set(filter_value[with_aer])

    def __convert_str_depth_to_int(
        self,
        str_data: dict[
            GraphKey, dict[MethodConfigJSON, dict[str, list[dict[str, Any]]]]
        ],
    ) -> dict[GraphKey, dict[MethodConfigJSON, dict[Depth, list[dict[str, Any]]]]]:
        """Convert string depths to integers, necessary for deserializing JSON data."""
        return {
            instance: {
                method: {
                    int(depth): depth_data for depth, depth_data in method_data.items()
                }
                for method, method_data in instance_data.items()
            }
            for instance, instance_data in str_data.items()
        }

    def __convert_str_depth_to_int_and_parse_datetime(
        self,
        str_data: dict[
            GraphKey, dict[MethodConfigJSON, dict[str, list[dict[str, Any]]]]
        ],
    ) -> dict[GraphKey, dict[MethodConfigJSON, dict[Depth, list[dict[str, Any]]]]]:
        """Convert string depths to integers and parse datetime strings, necessary for deserializing JSON data."""
        result = {}
        for instance, instance_data in str_data.items():
            result[instance] = {}
            for method, method_data in instance_data.items():
                result[instance][method] = {}
                for depth_str, results_list in method_data.items():
                    depth = int(depth_str)
                    result[instance][method][depth] = []
                    for result_dict in results_list:
                        result_copy = result_dict.copy()
                        # Parse run_datetime from ISO format string to datetime object
                        if "run_datetime" in result_copy and isinstance(
                            result_copy["run_datetime"], str
                        ):
                            result_copy["run_datetime"] = datetime.fromisoformat(
                                result_copy["run_datetime"]
                            )
                        result[instance][method][depth].append(result_copy)
        return result

    @staticmethod
    def load_failed_configs_from_json(
        filepath: str | Path,
    ) -> dict[GraphKey, dict[MethodConfigJSON, dict[Depth, str]]]:
        """Load failed configs from JSON file and convert string depths to integers.

        When loading failed configs from JSON, depth keys are strings because JSON
        doesn't support integer keys. This method loads the file and converts them
        to integers.

        Args:
            filepath: Path to the JSON file containing failed configs.

        Returns:
            Failed configs dict with integer depth keys.

        Example:
            >>> failed_configs = ResultsDatabase.load_failed_configs_from_json("failed_runs.json")
            >>> # Now pass failed_configs to get_missing_configs()
            >>> missing = db.get_missing_configs(
            ...     target_methods=methods,
            ...     target_instances=instances,
            ...     target_depths=[1, 2, 3],
            ...     failed_configs=failed_configs
            ... )
        """
        filepath = Path(filepath)
        with filepath.open("r") as f:
            json_data = json.load(f)

        # Convert string depths to integers
        return {
            instance: {
                method: {int(depth): reason for depth, reason in depths.items()}
                for method, depths in methods.items()
            }
            for instance, methods in json_data.items()
        }

    @staticmethod
    def save_failed_configs_to_json(
        failed_configs: dict[GraphKey, dict[MethodConfigJSON, dict[Depth, str]]],
        filepath: str | Path,
    ) -> None:
        """Save failed configs to JSON file, converting integer depths to strings.

        JSON doesn't support integer keys, so depths are converted to strings
        before saving.

        Args:
            failed_configs: Failed configs dict with integer depth keys.
            filepath: Path where the JSON file should be saved.

        Example:
            >>> failed_configs = {
            ...     "instance1.json": {
            ...         "method1.json": {
            ...             1: "reason1",
            ...             2: "reason2"
            ...         }
            ...     }
            ... }
            >>> ResultsDatabase.save_failed_configs_to_json(failed_configs, "failed_runs.json")
        """
        filepath = Path(filepath)

        # We don't convert integer depths to strings as json.dump does that for us.
        with filepath.open("w") as f:
            json.dump(failed_configs, f, indent=2)

    @property
    def data(
        self,
    ) -> dict[GraphKey, dict[MethodConfigJSON, dict[Depth, list[dict[str, Any]]]]]:
        """Return the data.

        Note: Returns the defaultdict as a regular dict type for compatibility.
        The defaultdict behavior is preserved at runtime.
        """
        return self._data  # type: ignore[return-value]

    @property
    def source_files(self) -> set[str]:
        """Return the set of source files."""
        return self._source_files

    @property
    def additional_methods(self) -> set[MethodConfigJSON]:
        """Return the set of additional methods."""
        return self._additional_methods

    def _create_empty_copy(
        self, num_nodes_filter: NumNodesFilter | None = None
    ) -> "ResultsDatabase":
        """Create an empty ResultsDatabase with copied metadata attributes.

        This helper method creates a new empty ResultsDatabase instance and copies
        over the metadata attributes (problem_class, num_nodes_filter, minmax_data,
        source_files, additional_methods) from the current instance.

        Args:
            num_nodes_filter: Optional num_nodes filter to use for the new instance.
                If None, uses the current instance's num_nodes_filter.

        Returns:
            New empty ResultsDatabase with copied metadata.
        """
        new_db = ResultsDatabase()
        new_db._problem_class = self._problem_class
        new_db._num_nodes_filter = (
            num_nodes_filter if num_nodes_filter is not None else self._num_nodes_filter
        )
        new_db._minmax_data = self._minmax_data.copy()
        new_db._source_files = self._source_files.copy()
        new_db._additional_methods = self._additional_methods.copy()
        return new_db

    def config_path_to_config(self, config_path: str) -> MethodConfigJSON:
        """Convert a string path to a config/method to just the config name."""
        basename = utils.instance.sanitize_path(config_path).split("/")[-1]
        return utils.labels.sanitize_trainer_config(basename)

    @staticmethod
    def extract_datetime_from_filename(filename: str) -> datetime:
        """Extract datetime from JSON filename.

        Filenames are expected to have format: YYYYmmdd_HHMMSS*.json
        For example: 20260407_235310_*.json is 7 April 2026 at 23:53:10 UTC.

        Args:
            filename: Path to the JSON file.

        Returns:
            datetime object in UTC.

        Raises:
            ValueError: If datetime pattern not found or cannot be parsed.
        """
        # Extract just the filename from the path
        basename = Path(filename).name

        # Pattern: YYYYMMDD_HHmmss at start of filename
        pattern = r"^(\d{8})_(\d{6})"
        match = re.match(pattern, basename)

        if not match:
            raise ValueError(
                f"Could not extract datetime from filename: {basename}. "
                f"Expected format: YYYYmmdd_HHMMSS*.json"
            )

        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHmmss

        try:
            # Parse as UTC datetime
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return dt
        except ValueError as e:
            raise ValueError(
                f"Could not parse datetime from filename: {basename}. "
                + f"Date string: {date_str}, Time string: {time_str}"
            ) from e

    def _get_config_info(self, config: MethodConfigJSON) -> dict[str, Any]:
        """Get cached information derived from a method config.

        This method computes and caches expensive lookups that are used repeatedly
        in the inner loop of add_data.

        Args:
            config: The method configuration filename.

        Returns:
            Dictionary containing:
                - evaluation: Evaluation method type (e.g., "SV", "MPS", "PP")
                - method: Evaluation-independent method name
                - evaluation_label: Human-readable evaluation label
                - with_aer: Whether the method uses Qiskit Aer
                - method_label: Human-readable method label from constants
        """
        evaluation: EvaluationType = utils.labels.trainer_config_to_evaluation(config)
        method: MethodJSON = utils.labels.trainer_config_to_method(config)
        evaluation_label: str = utils.labels.trainer_config_to_evaluation_label(config)
        with_aer: bool = utils.labels.method_uses_aer(config)

        # Get method label from constants, with fallback and warning
        method_label: str = utils.labels.method_to_method_label(method)
        # method_label 'should' be different than method. If method_label is the
        # same as method, then there probably isn't a defined human-readable
        # label. In this case, we warn the user.
        if method_label == method:
            warnings.warn(
                f"Method {method!r} most likely does have a human-readable label. "
                + "Using the method as the label."
            )

        return {
            "evaluation": evaluation,
            "method": method,
            "evaluation_label": evaluation_label,
            "with_aer": with_aer,
            "method_label": method_label,
        }

    def _process_noopt_variant(
        self,
        result: dict[str, Any],
        results: list[
            tuple[
                list[float],
                float | None,
                str,
                float,
                MethodConfigJSON,
                dict[str, Any],
                float,
            ]
        ],
        filename: str,
        run_datetime: datetime,
    ) -> None:
        """Process and add no-opt variant from opt method's zeroth iteration.

        This method extracts the zeroth iteration from an optimized training run
        and adds it as a separate no-opt configuration to the database.

        Args:
            result: Loaded JSON data from the file.
            results: List to append results to.
            filename: Path to the result file.
            run_datetime: Datetime when the run was performed.
        """
        opt_config: str = result["args"]["config"].replace("\\\\", "/")
        no_opt_config = self.config_path_to_config(
            utils.labels.trainer_config_to_no_opt(
                self.config_path_to_config(opt_config)
            )
        )
        # Track the no-opt config in additional_methods
        self._additional_methods.add(no_opt_config)

        self.populate_results(
            result,
            results,
            filename=filename,
            config=no_opt_config,
            # Only use the zeroth entry.
            iter_keys=["0"],
            run_datetime=run_datetime,
        )

    def _validate_result_file(
        self,
        filename: str,
        result: dict[str, Any],
        iter_keys: list[str] | None = None,
    ) -> None:
        config: MethodConfigJSON = self.config_path_to_config(result["args"]["config"])

        warning_suffix = f" for file {filename!r}"

        if iter_keys is None:
            iter_keys = self.get_iter_keys(result.keys())
        if (
            config.startswith("LR_")
            and config.endswith("_opt.json")
            and not config.endswith("_no_opt.json")
            and not config.endswith("_angle_opt.json")
        ):
            # We have non-angle-optimised Linear Ramp data which should only contain a "0" step.
            if len(iter_keys) != 1:
                warnings.warn(
                    f"Expected 1 iteration key for linear ramp ({config!r}), got {len(iter_keys)}"
                    + warning_suffix
                )
            if iter_keys != ["0"]:
                warnings.warn(
                    f'Expected `["0"]` iteration keys for linear ramp ({config!r}), got {iter_keys} instead'
                    + warning_suffix
                )

    def _process_result_file(
        self,
        filename: str,
        result: dict[str, Any],
    ) -> None:
        """Process a single result file and add its data to the database.

        Args:
            filename: Path to the result file.
            result: Loaded JSON data from the file.
        """
        # Track this source file
        self._source_files.add(filename)

        # Get the energy evaluation methodology
        config: MethodConfigJSON = self.config_path_to_config(result["args"]["config"])

        try:
            evaluation = utils.labels.trainer_config_to_evaluation(config)
        except ValueError as e:
            raise ValueError(
                f"Unrecognised energy evaluation in {filename} for method {config}"
            ) from e

        # Get the graph instance
        graph = utils.instance.sanitize_instance_key(result["args"]["input"]).split(
            "/"
        )[-1]

        # Extract datetime from filename
        run_datetime = self.extract_datetime_from_filename(filename)

        # Cache config-derived properties to avoid repeated lookups
        config_info = self._get_config_info(config)

        # Validate results
        self._validate_result_file(
            filename,
            result,
        )

        # Loop over the trainers in the result
        results: list[
            tuple[
                list[float],
                float | None,
                str,
                float,
                MethodConfigJSON,
                dict[str, Any],
                float,
            ]
        ] = []
        self.populate_results(
            result,
            results,
            filename=filename,
            run_datetime=run_datetime,
        )

        # Track this config in additional_methods if it's not already tracked
        # This ensures configs with data but no method JSON file are included
        self._additional_methods.add(config)

        # Check if we need to add the zeroth iteration, for opt to no-opt
        # trainer mappings.
        if utils.results.result_contains_noopt(result):
            self._process_noopt_variant(result, results, filename, run_datetime)

        # Process all results
        for (
            qaoa_angles,
            energy,
            trainer,
            duration,
            sub_config,
            metadata,
            result_key_index,
        ) in results:
            if qaoa_angles is None or energy is None:
                continue

            depth = len(qaoa_angles) // 2

            # Get cached config info for sub_config (may differ from main config)
            sub_config_info = self._get_config_info(sub_config)

            # Add this result to the list (defaultdict auto-creates nested structure)
            self._data[graph][sub_config][depth].append(
                {
                    "energy": energy,
                    "qaoa_angles": qaoa_angles,
                    "train_duration": duration,
                    "source_file": filename,
                    "trainer": trainer,
                    "trainer_config": sub_config,
                    "method": sub_config_info["method"],
                    "evaluation": sub_config_info["evaluation"],
                    "evaluation_label": sub_config_info["evaluation_label"],
                    "method_label": sub_config_info["method_label"],
                    "with_aer": sub_config_info["with_aer"],
                    "metadata": metadata,
                    "run_datetime": run_datetime,
                    "result_key_index": result_key_index,
                }
            )

    def add_data(
        self,
        folder_name: str,
        ignore_file_function: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        """Load training data from a folder and add all results to the database.

        Args:
            folder_name: Path to folder containing results as JSON files. All files ending in
                ``.json`` will be processed.
            ignore_file_function: Optional function used to ignore files. The function takes the file
                name and the JSON data as arguments and returns a boolean indicating whether the
                file should be ignored. If provided, only files whose returned value is False will
                be processed. If None, no files are ignored. Defaults to None.
        """
        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")

            with open(filename, "r") as fin:
                result = json.load(fin)

            # Determine file's problem class
            file_problem_class = utils.metadata.guess_problem_class(filename, result)

            # Skip if problem class doesn't match filter
            if self._problem_class != file_problem_class:
                continue

            # Extract num_nodes from graph input and filter if needed
            if self._num_nodes_filter is not None:
                graph_input = result.get("args", {}).get("input", "")
                if graph_input:
                    try:
                        instance_num_nodes = utils.instance.num_nodes(graph_input)
                        # For add_data, we don't have evaluation info yet, so use simple check
                        if not self._should_include_by_num_nodes(instance_num_nodes):
                            continue
                    except ValueError:
                        # If we can't extract num_nodes, skip this file
                        warnings.warn(
                            f"Couldn't determine graph-size for instance {graph_input!r}."
                        )
                        continue

            if ignore_file_function is not None and ignore_file_function(
                filename, result
            ):
                continue

            self._process_result_file(filename, result)

    def add_minmax_cut_data(self, folder_name: str, replace: bool = False) -> None:
        """Load min-max cut data from JSON files in a folder.

        Args:
            folder_name: Path to folder containing min-max cut JSON files.
                Files should match pattern `*_maxmin_cut.json`.
            replace: If True, replace existing min-max data. If False, only add new entries.

        Warns:
            If problem_class is not "MC", warns that min-max cut data is only for MaxCut.
        """
        if self._problem_class != "MC":
            warnings.warn(
                f"Loading min-max cut data but problem_class is '{self._problem_class}'. "
                "Min-max cut data is only applicable to MaxCut problems.",
                UserWarning,
                stacklevel=2,
            )

        for filename in glob.glob(f"{folder_name}/*_maxmin_cut.json"):
            filename = filename.replace("\\", "/")

            with open(filename, "r") as fin:
                data = json.load(fin)

            # Extract instance filename and normalize path separators
            instance_name = utils.instance.sanitize_instance_key(
                data["instance"]
            ).split("/")[-1]

            # Skip if not replacing and already exists
            if not replace and instance_name in self._minmax_data:
                continue

            # Store the min-max result
            self._minmax_data[instance_name] = MinMaxResult(
                max_cut=data["max_cut"],
                min_cut=data["min_cut"],
                sum_of_weights=data["sum_of_weights"],
                time_solve_max_cut_ns=data["time_solve_max_cut_ns"],
            )

    @overload
    def maxcut_approximation_ratio(
        self, graph_key: str, energy: float, return_none: Literal[False]
    ) -> float | NoReturn: ...

    @overload
    def maxcut_approximation_ratio(
        self, graph_key: str, energy: float, return_none: Literal[True]
    ) -> float | None: ...

    def maxcut_approximation_ratio(
        self, graph_key: str, energy: float, return_none: bool = False
    ) -> float | None | NoReturn:
        """Calculate the MaxCut approximation ratio for a given energy.

        Formula: (energy - min_cut) / (max_cut - min_cut)

        Args:
            graph_key: Instance name (filename).
            energy: Energy value to compute approximation ratio for.
            return_none: If True, return None when min-max data is missing.
                If False, raise KeyError when min-max data is missing.

        Returns:
            Approximation ratio, or None if return_none=True and data is missing.

        Raises:
            KeyError: If return_none=False and min-max data is missing for the instance.
        """
        if graph_key not in self._minmax_data:
            if return_none:
                return None
            raise KeyError(
                f"No min-max cut data available for instance '{graph_key}'. "
                f"Use add_minmax_cut_data() to load the data."
            )

        minmax = self._minmax_data[graph_key]
        max_cut = minmax["max_cut"]
        min_cut = minmax["min_cut"]
        sum_of_weights = minmax["sum_of_weights"]

        return utils.problem.maxcut_approximation_ratio(
            min_cut=min_cut, max_cut=max_cut, sum_weights=sum_of_weights, energy=energy
        )

    def missing_minmax_cuts(self) -> list[str]:
        """Return list of instances in database that lack min-max cut data.

        Returns:
            List of instance names (filenames) that are in the database but not in _minmax_data.
        """
        instances_in_data = set(self._data.keys())
        instances_with_minmax = set(self._minmax_data.keys())
        return sorted(instances_in_data - instances_with_minmax)

    @staticmethod
    def get_iter_keys(keys):
        """Extract iteration keys from result dictionary keys."""
        iterations = []
        for key in keys:
            try:
                _ = int(key)
                iterations.append(key)
            except ValueError:
                pass
        return iterations

    def populate_results(
        self,
        result: dict[str, Any],
        result_data: list[
            tuple[
                list[float],
                float | None,
                str,
                float,
                MethodConfigJSON,
                dict[str, Any],
                float,
            ]
        ],
        filename: str | None = None,
        iter_keys: Iterable[str] | None = None,
        config: MethodConfigJSON | None = None,
        run_datetime: datetime | None = None,
        parent_key_index: float = 0.0,
    ):
        """Get the parameters from the `result` dict.

        Args:
            result: Input results dictionary to process.
            result_data: Output list of results.
            filename: Optional name of original JSON file for results, for warnings.
                If None, warnings will not refer to a specific file.
            iter_keys: Iterable of keys for sub-results. If None, generated with
                :meth:`get_iter_keys`.
            config: Optional method config. If None, extracted from result.
            run_datetime: Datetime when the run was performed. Must be provided (extracted from filename).
            parent_key_index: Key index from parent recursion level. Used for recursive trainers.
        """
        if run_datetime is None:
            raise ValueError("run_datetime must be provided")
        if iter_keys:
            __iter_keys = self.get_iter_keys(iter_keys)
        else:
            __iter_keys = self.get_iter_keys(result.keys())
        if config is None:
            __config: str | None = result.get("args", {}).get("config", None)
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

            # Calculate result_key_index for this iteration
            # For recursive trainers, scale by 1/10 and add to parent index
            current_key_index = parent_key_index + float(key)

            if trainer == "RecursionTrainer":
                # Recursively process sub-results with scaled key indices
                self.populate_results(
                    sub_result,
                    result_data,
                    filename=filename,
                    config=config,
                    run_datetime=run_datetime,
                    # We divide by ten so that indexes are the decimal digits in the parent_key_index.
                    parent_key_index=current_key_index / 10.0,
                )

            qaoa_angles = sub_result["optimized_qaoa_angles"]
            energy = utils.results.sanitize_energy(energy_val=sub_result["energy"])
            duration = float(sub_result["train_duration"])

            # Ensure length of 2p
            if len(qaoa_angles) % 2 != 0:
                continue

            # Store metadata
            # NOTE: Additional metadata fields can be added here in the future
            # as needed (e.g., convergence info, optimizer details, etc.)
            metadata = {
                "iteration": key,
                "version": version,
            }

            result_data.append(
                (
                    qaoa_angles,
                    energy,
                    trainer,
                    duration,
                    config,
                    metadata,
                    current_key_index,
                )
            )

    def list_methods(self) -> list[MethodConfigJSON]:
        """List all unique methods in the database.

        This includes both methods found in the data and additional methods
        (e.g., no-opt variants derived from opt methods).

        Returns:
            Sorted list of all method config names present in the database.
        """
        methods = set()
        # Get methods from actual data
        for instance_data in self._data.values():
            methods.update(instance_data.keys())
        # Add additional methods (includes no-opt variants)
        methods.update(self._additional_methods)
        return sorted(methods)

    def print_methods_by_evaluation(self) -> None:
        """Print a text table showing method config names organized by evaluation label.

        Each column represents an evaluation label and lists the method config names
        present in the data for that evaluation. Columns may have different numbers of rows.
        """
        configs_by_eval: defaultdict[str, list[MethodConfigJSON]] = defaultdict(list)

        for method in self.list_methods():
            try:
                eval_label = utils.labels.trainer_config_to_evaluation_label(method)
            except ValueError:
                # Skip methods with unrecognized evaluation
                continue
            configs_by_eval[eval_label].append(method)

        if not configs_by_eval:
            print("No methods found in database.")
            return

        sorted_configs_by_eval = {
            eval_label: sorted(configs)
            for eval_label, configs in sorted(configs_by_eval.items())
        }
        eval_labels = list(sorted_configs_by_eval.keys())

        col_widths: dict[str, int] = {}
        max_rows = 0
        for eval_label, configs in sorted_configs_by_eval.items():
            col_widths[eval_label] = max(
                len(eval_label),
                max(len(config) for config in configs),
            )
            max_rows = max(max_rows, len(configs))

        header = " | ".join(f"{label:^{col_widths[label]}}" for label in eval_labels)
        print(header)
        print("-" * len(header))

        for row_idx in range(max_rows):
            print(
                " | ".join(
                    f"{configs[row_idx]:<{col_widths[eval_label]}}"
                    if row_idx < len(configs)
                    else " " * col_widths[eval_label]
                    for eval_label, configs in sorted_configs_by_eval.items()
                )
            )

    def filter_by(
        self,
        instance_filter: dict[EvaluationType, set[str] | dict[bool, set[str]]]
        | None = None,
        num_nodes: NumNodesFilter | None = None,
    ) -> "ResultsDatabase":
        """Filter database by instance-evaluation combinations and/or num_nodes.

        Args:
            instance_filter: Dictionary with evaluation strings as keys (e.g., "SV", "MPS", "PP").
                Values can be sets or dicts mapping with_aer (bool) to sets of instance names.
                Only results matching these combinations will be included. If None, no instance filtering.
            num_nodes: Num_nodes filter. Can be:
                - Iterable[int]: Simple list of node counts
                - dict[EvaluationType, Iterable[int] | dict[bool, Iterable[int]]]: Evaluation-specific filtering
                - None: No filtering

        Returns:
            New ResultsDatabase containing only the filtered results.
        """
        filtered_db = self._create_empty_copy(num_nodes_filter=num_nodes)

        for instance, instance_data in self._data.items():
            # Get instance num_nodes for filtering
            try:
                instance_num_nodes = utils.instance.num_nodes(instance)
            except ValueError:
                # Skip instances where we can't determine num_nodes
                if num_nodes is not None:
                    continue
                instance_num_nodes = -1  # Dummy value when no filter

            for method, method_data in instance_data.items():
                # Get evaluation and with_aer for filtering
                try:
                    evaluation = utils.labels.trainer_config_to_evaluation(method)
                    with_aer = utils.labels.method_uses_aer(method)
                except ValueError:
                    # Skip methods we can't extract evaluation from
                    continue

                # Filter by num_nodes if specified
                if num_nodes is not None and instance_num_nodes != -1:
                    if not filtered_db._should_include_by_num_nodes(
                        instance_num_nodes, evaluation, with_aer
                    ):
                        continue

                # Filter by instance_filter if specified
                if instance_filter is not None:
                    # Check if this evaluation is in the filter
                    if evaluation not in instance_filter:
                        continue

                    filter_value = instance_filter[evaluation]

                    # Determine if this instance should be included
                    should_include = False
                    if isinstance(filter_value, set):
                        should_include = instance in filter_value
                    else:
                        if with_aer in filter_value:
                            should_include = instance in filter_value[with_aer]

                    if not should_include:
                        continue

                # Copy all depth data for this method (defaultdict auto-creates structure)
                for depth, results in method_data.items():
                    filtered_db._data[instance][method][depth] = results.copy()

                # Track this method in additional_methods if it was in the original
                if method in self._additional_methods:
                    filtered_db._additional_methods.add(method)

        return filtered_db

    def only_common_instances(
        self,
        reference_methods: Mapping[
            EvaluationType | tuple[Literal["MPS"], bool],
            Mapping[GraphType, MethodConfigJSON],
        ],
        per_depth: bool = True,
    ) -> "ResultsDatabase":
        """Filter database to only include instances common to reference methods.

        For each (evaluation, graph_type) pair in reference_methods, this method identifies
        the instances present for the reference method and filters all other methods for
        that (evaluation, graph_type) to only include those same instances.

        Args:
            reference_methods: Dictionary mapping evaluation keys to graph_type->method mappings.
                Evaluation keys can be:
                - str: Simple evaluation type (e.g., "SV", "PP")
                - tuple[str, bool]: For MPS, (evaluation, with_aer) where with_aer indicates
                  whether Qiskit Aer is used. E.g., ("MPS", False) for MPS without Aer,
                  ("MPS", True) for MPS with Aer.
                Example: {"SV": {"random_regular": "F_SV.json"}}
            per_depth: If True, instances are filtered independently for each depth value.
                If False, instances are filtered across all depths (an instance must be
                present in the reference method at any depth to be included at all depths).
                Defaults to True.

        Returns:
            A new ResultsDatabase instance with filtered data containing only common instances.

        Example:
            >>> reference_methods = {
            ...     "SV": {"random_regular": "F_SV.json"},
            ...     ("MPS", False): {"heavy_hex": "I_MPS.json"}
            ... }
            >>> # Filter per depth (default)
            >>> filtered_db = db.only_common_instances(reference_methods, per_depth=True)
            >>> # Filter across all depths
            >>> filtered_db = db.only_common_instances(reference_methods, per_depth=False)
        """
        # Create a new filtered database
        filtered_db = self._create_empty_copy()

        # Build mapping of (eval_key, graph_type) -> set of allowed instances
        allowed_instances: dict[
            tuple[EvaluationType | tuple[Literal["MPS"], bool], GraphType],
            dict[None | int, set[GraphKey]],
        ] = {}

        for eval_key, graph_type_methods in reference_methods.items():
            for graph_type_name, ref_method in graph_type_methods.items():
                # Find all instances for this reference method and graph type
                # We track instances per depth and combine later if per_depth is
                # False.
                instances_per_depth: dict[int, set[GraphKey]] = {}

                for instance, instance_data in self._data.items():
                    # Check if instance matches the graph type
                    try:
                        instance_graph_type = utils.instance.graph_type(instance)
                    except ValueError:
                        # Skip instances where graph type cannot be determined
                        continue

                    if instance_graph_type != graph_type_name:
                        continue

                    # Check if this instance has data for the reference method
                    if ref_method in instance_data:
                        # Track instances separately for each depth
                        for depth in instance_data[ref_method].keys():
                            if depth not in instances_per_depth:
                                instances_per_depth[depth] = set()
                            instances_per_depth[depth].add(instance)

                # Initialize the nested dict for this (eval_key, graph_type) combination
                if (eval_key, graph_type_name) not in allowed_instances:
                    allowed_instances[(eval_key, graph_type_name)] = {}

                # Store the allowed instances
                if per_depth:
                    # Store per-depth instances with depth as key
                    for depth, instances in instances_per_depth.items():
                        allowed_instances[(eval_key, graph_type_name)][depth] = (
                            instances
                        )
                else:
                    # Store all-depth instances with None as key
                    allowed_instances[(eval_key, graph_type_name)][None] = set()
                    for depth, instances in instances_per_depth.items():
                        allowed_instances[(eval_key, graph_type_name)][None].update(
                            instances
                        )

        # Filter the data
        for instance, instance_data in self._data.items():
            # Get graph type for this instance
            try:
                instance_graph_type = utils.instance.graph_type(instance)
            except ValueError:
                # Skip instances where graph type cannot be determined
                continue

            for method, method_data in instance_data.items():
                # Get evaluation info for this method
                evaluation = utils.labels.trainer_config_to_evaluation(method)
                with_aer = utils.labels.method_uses_aer(method)

                # Determine the eval_key to use for lookup
                # MPS always uses tuple form, others use string
                if evaluation == "MPS":
                    eval_key: EvaluationType | tuple[Literal["MPS"], bool] = (
                        evaluation,
                        with_aer,
                    )
                else:
                    eval_key = cast(EvaluationType, evaluation)

                # Check if we have a filter for this (eval_key, graph_type) combination
                lookup_key: tuple[
                    EvaluationType | tuple[Literal["MPS"], bool], GraphType
                ] = (eval_key, instance_graph_type)
                if lookup_key in allowed_instances:
                    # We have a filter for this combination
                    depth_filters = allowed_instances[lookup_key]

                    for depth, results in method_data.items():
                        # Check if instance is allowed for this depth
                        if per_depth:
                            if (
                                depth not in depth_filters
                                or instance not in depth_filters[depth]
                            ):
                                continue
                        else:
                            if (
                                None not in depth_filters
                                or instance not in depth_filters[None]
                            ):
                                continue
                        filtered_db._data[instance][method][depth] = results.copy()
                else:
                    # No filter for this combination, include all data
                    for depth, results in method_data.items():
                        filtered_db._data[instance][method][depth] = results.copy()

                # Track this method in additional_methods if it was in the original
                if method in self._additional_methods:
                    filtered_db._additional_methods.add(method)

        return filtered_db

    def only_best_parameters(
        self, by: Literal["instance", "config"]
    ) -> "ResultsDatabase":
        """Filter database to keep only results with maximum energy within groups.

        Args:
            by: Grouping strategy for selecting best parameters:
                - "instance": Group by (instance, evaluation_label, depth). Best parameters chosen
                  irrespective of method, but separately for each evaluation type.
                - "config": Group by (instance, evaluation_label, method_label, depth). Best
                  parameters per method and evaluation type.

        Returns:
            New ResultsDatabase containing only the best results for each group.
        """
        filtered_db = self._create_empty_copy()

        if by == "instance":
            # Group by (instance, evaluation_label, depth) - best irrespective of method
            # First, collect all results with their grouping keys
            instance_groups: dict[
                tuple[GraphKey, str, Depth],
                list[tuple[MethodConfigJSON, int, dict[str, Any]]],
            ] = {}

            for instance, instance_data in self._data.items():
                for trainer_config, method_data in instance_data.items():
                    for depth, results_list in method_data.items():
                        for result_idx, result in enumerate(results_list):
                            # Get evaluation_label from result
                            evaluation_label = result.get("evaluation_label")
                            if evaluation_label is None:
                                # Fallback: compute it
                                evaluation_label = (
                                    utils.labels.trainer_config_to_evaluation_label(
                                        trainer_config
                                    )
                                )

                            group_key = (instance, evaluation_label, depth)
                            if group_key not in instance_groups:
                                instance_groups[group_key] = []

                            instance_groups[group_key].append(
                                (trainer_config, result_idx, result)
                            )

            # For each group, find the result(s) with maximum energy
            for (instance, evaluation_label, depth), results in instance_groups.items():
                max_energy = max(r[2]["energy"] for r in results)
                best_results = [r for r in results if r[2]["energy"] == max_energy]

                # If multiple results have the same max energy, apply tie-breaking rules:
                # 1. Most recent run_datetime
                # 2. Smallest result_key_index
                # 3. First in list
                if len(best_results) > 1:
                    best_results.sort(
                        key=lambda r: (
                            -r[2][
                                "run_datetime"
                            ].timestamp(),  # Most recent first (negative for descending)
                            r[2]["result_key_index"],  # Smallest first (ascending)
                        )
                    )
                    # Keep only the first (best) result
                    best_results = [best_results[0]]

                # Add best result to filtered database (defaultdict auto-creates structure)
                for trainer_config, result_idx, result in best_results:
                    filtered_db._data[instance][trainer_config][depth].append(
                        result.copy()
                    )

        elif by == "config":
            # Group by (instance, evaluation_label, method_label, depth) - best per method and evaluation
            config_groups: dict[
                tuple[GraphKey, str, str, Depth],
                list[tuple[MethodConfigJSON, int, dict[str, Any]]],
            ] = {}

            for instance, instance_data in self._data.items():
                for trainer_config, method_data in instance_data.items():
                    for depth, results_list in method_data.items():
                        for result_idx, result in enumerate(results_list):
                            # Get evaluation_label from result
                            evaluation_label = result.get("evaluation_label")
                            if evaluation_label is None:
                                # Fallback: compute it
                                evaluation_label = (
                                    utils.labels.trainer_config_to_evaluation_label(
                                        trainer_config
                                    )
                                )

                            # Get method_label from result (should already be computed)
                            method_label = result.get("method_label")
                            if method_label is None:
                                # Fallback: compute it
                                method_label = (
                                    utils.labels.trainer_config_to_method_label(
                                        trainer_config
                                    )
                                )

                            group_key = (
                                instance,
                                evaluation_label,
                                method_label,
                                depth,
                            )
                            if group_key not in config_groups:
                                config_groups[group_key] = []

                            config_groups[group_key].append(
                                (trainer_config, result_idx, result)
                            )

            # For each group, find the result(s) with maximum energy
            for (
                instance,
                evaluation_label,
                method_label,
                depth,
            ), results in config_groups.items():
                max_energy = max(r[2]["energy"] for r in results)
                best_results = [r for r in results if r[2]["energy"] == max_energy]

                # If multiple results have the same max energy, apply tie-breaking rules:
                # 1. Most recent run_datetime (prefer newer training runs)
                # 2. Smallest result_key_index (prefer earlier iterations)
                # 3. First in list
                if len(best_results) > 1:
                    best_results.sort(
                        key=lambda r: (
                            -r[2][
                                "run_datetime"
                            ].timestamp(),  # Most recent first (negative for descending)
                            r[2]["result_key_index"],  # Smallest first (ascending)
                        )
                    )
                    # Keep only the first (best) result
                    best_results = [best_results[0]]

                # Add best result to filtered database (defaultdict auto-creates structure)
                for trainer_config, result_idx, result in best_results:
                    filtered_db._data[instance][trainer_config][depth].append(
                        result.copy()
                    )

        else:
            raise ValueError(
                f"Invalid value for 'by': {by!r}. Must be 'instance' or 'config'."
            )

        return filtered_db

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the database to a pandas DataFrame.

        Returns:
            DataFrame with columns: instance, trainer_config, method, depth, energy, trainer,
            evaluation, evaluation_label, method_label, with_aer, source_file, train_duration,
            metadata (as dict), result_index (for multiple results at same depth), run_datetime
            (as datetime object), result_key_index, approximation_ratio.
        """
        records = []
        missing_minmax_warned = set()  # Track instances we've warned about

        for instance, instance_data in self._data.items():
            for trainer_config, method_data in instance_data.items():
                for depth, results_list in method_data.items():
                    for idx, result in enumerate(results_list):
                        energy = result["energy"]

                        # Compute approximation ratio
                        approx_ratio = np.nan
                        if self._problem_class == "MC":
                            if instance in self._minmax_data:
                                try:
                                    approx_ratio = self.maxcut_approximation_ratio(
                                        instance, energy, return_none=False
                                    )
                                except (KeyError, ZeroDivisionError):
                                    # Should not happen since we checked, but handle gracefully
                                    approx_ratio = np.nan
                            else:
                                # Warn once per instance
                                if instance not in missing_minmax_warned:
                                    warnings.warn(
                                        f"Missing min-max cut data for instance '{instance}'. "
                                        "Approximation ratio will be NaN.",
                                        UserWarning,
                                        stacklevel=2,
                                    )
                                    missing_minmax_warned.add(instance)
                        # For non-MC problems, approximation_ratio is np.nan (no warning)

                        records.append(
                            {
                                "instance": instance,
                                "num_nodes": utils.instance.num_nodes(instance),
                                "graph_type": utils.instance.graph_type(instance),
                                "trainer_config": result.get(
                                    "trainer_config", trainer_config
                                ),
                                "method": result.get("method", None),
                                "depth": depth,
                                "energy": result["energy"],
                                "trainer": result["trainer"],
                                "evaluation": result["evaluation"],
                                "evaluation_label": result.get(
                                    "evaluation_label", None
                                ),
                                "method_label": result.get("method_label", None),
                                "with_aer": result["with_aer"],
                                "source_file": result["source_file"],
                                "train_duration": result["train_duration"],
                                "metadata": result["metadata"],
                                "result_index": idx,
                                "run_datetime": result["run_datetime"],
                                "result_key_index": result["result_key_index"],
                                "approximation_ratio": approx_ratio,
                            }
                        )

        return pd.DataFrame(records)

    def save(self, file_name: str, overwrite: bool = False) -> None:
        """Save the database to a JSON file.

        Args:
            file_name: Path to save the database.
            overwrite: If True, overwrite existing file. If False, raise error if file exists.

        Raises:
            ValueError: If file exists and overwrite is False.
        """
        import os

        if os.path.isfile(file_name) and not overwrite:
            raise ValueError(f"File {file_name} already exists.")

        # Convert datetime objects to ISO format strings for JSON serialization
        serializable_data = {}
        for instance, instance_data in self._data.items():
            serializable_data[instance] = {}
            for method, method_data in instance_data.items():
                serializable_data[instance][method] = {}
                for depth, results_list in method_data.items():
                    serializable_data[instance][method][depth] = []
                    for result in results_list:
                        result_copy = result.copy()
                        if (
                            "run_datetime" in result_copy
                            and result_copy["run_datetime"] is not None
                        ):
                            result_copy["run_datetime"] = result_copy[
                                "run_datetime"
                            ].isoformat()
                        serializable_data[instance][method][depth].append(result_copy)

        # Serialize num_nodes_filter
        num_nodes_serialized = None
        if self._num_nodes_filter is not None:
            if isinstance(self._num_nodes_filter, dict):
                # Convert dict structure, ensuring iterables become lists
                num_nodes_serialized = {}
                for eval_key, eval_value in self._num_nodes_filter.items():
                    if isinstance(eval_value, dict):
                        num_nodes_serialized[eval_key] = {
                            str(k): list(v) for k, v in eval_value.items()
                        }
                    else:
                        num_nodes_serialized[eval_key] = list(eval_value)
            else:
                # Simple iterable
                num_nodes_serialized = list(self._num_nodes_filter)

        with open(file_name, "w") as fout:
            json.dump(
                {
                    "data": serializable_data,
                    "source_files": list(self._source_files),
                    "additional_methods": list(self._additional_methods),
                    "problem_class": self._problem_class,
                    "num_nodes_filter": num_nodes_serialized,
                    "minmax_data": self._minmax_data,
                },
                fout,
            )

    def _build_target_configs(
        self,
        target_methods: dict[
            EvaluationType, list[MethodConfigJSON] | dict[bool, list[MethodConfigJSON]]
        ],
        target_instances: dict[
            EvaluationType, set[GraphKey] | dict[bool, set[GraphKey]]
        ],
        target_depths_set: set[Depth],
    ) -> set[
        tuple[
            EvaluationType | tuple[Literal["MPS"], bool],
            MethodConfigJSON,
            Depth,
            GraphKey,
        ]
    ]:
        """Build the complete set of target configurations.

        Args:
            target_methods: Dictionary with evaluation strings as keys.
            target_instances: Dictionary with evaluation strings as keys.
            target_depths_set: Set of depths to check.

        Returns:
            Set of tuples (eval_key, method, depth, instance).
        """
        target_configs: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ] = set()

        for evaluation, methods_value in target_methods.items():
            instances_value = target_instances.get(evaluation, None)
            if instances_value is None:
                continue

            # Handle both simple and nested structures
            if isinstance(methods_value, list) and isinstance(instances_value, set):
                # Simple case: SV or PP
                eval_key = evaluation
                for method in methods_value:
                    for depth in target_depths_set:
                        for instance in instances_value:
                            target_configs.add((eval_key, method, depth, instance))

            elif isinstance(methods_value, dict) and isinstance(instances_value, dict):
                # Type-narrowing for evaluation. We should only have dict[bool,
                # ...] values for evaluation="MPS".
                if evaluation != "MPS":
                    raise KeyError(
                        "Only MPS evaluation can have a dict[bool, list[MethodConfigJSON]] "
                        + f"as a value type in target_methods. Got {evaluation!r} instead."
                    )
                # Complex case: MPS with with_aer distinction
                for with_aer in [False, True]:
                    if with_aer in methods_value and with_aer in instances_value:
                        eval_key = (evaluation, with_aer)
                        for method in methods_value[with_aer]:
                            for depth in target_depths_set:
                                for instance in instances_value[with_aer]:
                                    target_configs.add(
                                        (eval_key, method, depth, instance)
                                    )

            else:
                raise ValueError(
                    f"Mismatched types for evaluation {evaluation!r}: "
                    f"target_methods[{evaluation!r}] value is {type(methods_value).__name__}, "
                    f"target_instances[{evaluation!r}] value is {type(instances_value).__name__}. "
                    f"Expected both to be list/set or both to be dict."
                )

        return target_configs

    def _build_existing_configs(
        self,
        seen_methods: dict[
            tuple[EvaluationType | tuple[Literal["MPS"], bool], MethodConfigJSON], bool
        ],
        seen_instances: dict[
            tuple[EvaluationType | tuple[Literal["MPS"], bool], GraphKey], bool
        ],
    ) -> set[
        tuple[
            EvaluationType | tuple[Literal["MPS"], bool],
            MethodConfigJSON,
            Depth,
            GraphKey,
        ]
    ]:
        """Build the set of existing configurations from database.

        Args:
            seen_methods: Dictionary to track which target methods are in the database.
            seen_instances: Dictionary to track which target instances are in the database.

        Returns:
            Set of tuples (eval_key, method, depth, instance).
        """
        existing_configs: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ] = set()

        for instance, instance_data in self._data.items():
            for method, method_data in instance_data.items():
                try:
                    evaluation = utils.labels.trainer_config_to_evaluation(method)
                except ValueError:
                    continue

                with_aer = utils.labels.method_uses_aer(method)

                # Determine eval_key based on evaluation type
                eval_key: EvaluationType | tuple[Literal["MPS"], bool]
                if evaluation == "MPS":
                    eval_key = (evaluation, with_aer)
                elif evaluation in ["SV", "PP"]:
                    eval_key = cast(EvaluationType, evaluation)
                else:
                    raise KeyError(f"Unknown evaluation {evaluation!r}.")

                # Mark this method and instance as seen if they're in our targets
                method_key = (eval_key, method)
                if method_key in seen_methods:
                    seen_methods[method_key] = True

                instance_key = (eval_key, instance)
                if instance_key in seen_instances:
                    seen_instances[instance_key] = True

                for depth in method_data.keys():
                    existing_configs.add((eval_key, method, depth, instance))

        return existing_configs

    def _filter_derived_configs(
        self,
        missing: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ],
        failed_set: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ],
        failed_configs: dict[GraphKey, dict[MethodConfigJSON, dict[Depth, str]]] | None,
    ) -> set[
        tuple[
            EvaluationType | tuple[Literal["MPS"], bool],
            MethodConfigJSON,
            Depth,
            GraphKey,
        ]
    ]:
        """Filter out configurations that can be derived from other configurations.

        Args:
            missing: Set of missing configurations.
            failed_set: Set of failed configurations.
            failed_configs: Optional dictionary mapping GraphKey -> MethodConfigJSON -> Depth -> reason.

        Returns:
            Set of configurations to remove (those that can be derived).
        """
        configs_to_remove: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ] = set()

        # First, build a mapping of (eval_key, method, instance) -> set of depths
        # This helps us find the maximum depth for interpolation methods
        # Include both missing and failed configs in the depth map
        depth_map: dict[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                GraphKey,
            ],
            set[Depth],
        ] = {}
        for eval_key, method, depth, instance in missing:
            key = (eval_key, method, instance)
            if key not in depth_map:
                depth_map[key] = set()
            depth_map[key].add(depth)

        # Also add failed configs to depth map for interpolation methods
        if failed_configs is not None:
            for eval_key, method, depth, instance in failed_set:
                method_base = method.replace(".json", "")
                if method_base.startswith("I_") or method_base.startswith("F_"):
                    key = (eval_key, method, instance)
                    if key not in depth_map:
                        depth_map[key] = set()
                    depth_map[key].add(depth)

        for eval_key, method, depth, instance in missing:
            # Extract the base method name (without evaluation suffix)
            method_base = method.replace(".json", "")

            # Rule 1: Interpolation methods (I_*, F_*) at depth p can be derived from any higher depth
            # If the source config (higher depth) is in missing OR failed, remove this config
            if method_base.startswith("I_") or method_base.startswith("F_"):
                key = (eval_key, method, instance)
                depths_for_config = depth_map.get(key, set())
                if depths_for_config:
                    # Find all depths greater than current depth
                    higher_depths = [d for d in depths_for_config if d > depth]
                    if higher_depths:
                        # If any higher depth exists (in missing or failed), this config can be derived
                        configs_to_remove.add((eval_key, method, depth, instance))

            # Rule 2: no_opt variants derived from Opt variants (zeroth iteration)
            # Only applicable for TQA and FA methods
            if ("no_opt" in method) and (
                method_base.startswith("TQA_") or method_base.startswith("FA_")
            ):
                # Determine the corresponding opt method
                opt_method = method.replace("no_opt", "opt")

                # Check if the opt method is in missing OR failed
                opt_config = (eval_key, opt_method, depth, instance)
                # if opt_config in missing or opt_config in failed_set:
                if opt_config in missing or opt_config in failed_set:
                    # Remove the no_opt config (it can be derived from opt, even if opt failed)
                    configs_to_remove.add((eval_key, method, depth, instance))

            # Rule 3: LR_*_opt.json derived from LR_*_angle_opt.json (zeroth iteration)
            # Check if this is an LR_opt method (but not angle_opt)
            if (
                method_base.startswith("LR_")
                and method_base.endswith("_opt")
                and not method_base.endswith("_no_opt")
                and "angle" not in method_base
            ):
                # Determine the corresponding angle_opt method
                # Insert "angle_" before "opt"
                angle_opt_method = method.replace("_opt.json", "_angle_opt.json")

                # Check if the angle_opt method is in missing OR failed
                angle_opt_config = (eval_key, angle_opt_method, depth, instance)
                if angle_opt_config in missing or angle_opt_config in failed_set:
                    # Remove the LR_opt config (it can be derived from angle_opt, even if angle_opt failed)
                    configs_to_remove.add((eval_key, method, depth, instance))

        return configs_to_remove

    def _apply_failed_configs_filter(
        self,
        missing: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ],
        failed_set: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ],
    ) -> set[
        tuple[
            EvaluationType | tuple[Literal["MPS"], bool],
            MethodConfigJSON,
            Depth,
            GraphKey,
        ]
    ]:
        """Remove failed configurations from missing set.

        Args:
            missing: Set of missing configurations.
            failed_set: Set of failed configurations.

        Returns:
            Set of missing configurations with failed configs removed.
        """
        return missing - failed_set

    def get_missing_configs(
        self,
        target_methods: dict[
            EvaluationType, list[MethodConfigJSON] | dict[bool, list[MethodConfigJSON]]
        ],
        target_instances: dict[
            EvaluationType, set[GraphKey] | dict[bool, set[GraphKey]]
        ],
        target_depths: Iterable[Depth],
        failed_configs: dict[GraphKey, dict[MethodConfigJSON, dict[Depth, str]]]
        | None = None,
        with_derived_configs: bool = False,
    ) -> dict[
        EvaluationType | tuple[Literal["MPS"], bool],
        dict[MethodConfigJSON, dict[Depth, set[GraphKey]]],
    ]:
        """Find missing configurations in the database.

        Compares target configurations (methods, instances, depths) against existing data
        and returns configurations that are missing from the database, excluding any
        configurations that are marked as failed.

        Args:
            target_methods: Dictionary with evaluation strings as keys. Values can be:
                - A list of MethodConfigJSON (for SV and PP evaluations)
                - A dict mapping with_aer (bool) to lists of MethodConfigJSON (for MPS)
            target_instances: Dictionary with evaluation strings as keys. Values can be:
                - A set of instance names (for SV and PP evaluations)
                - A dict mapping with_aer (bool) to sets of instance names (for MPS)
            target_depths: Iterable of depths to check.
            failed_configs: Optional dictionary mapping GraphKey -> MethodConfigJSON -> Depth -> reason.
                Configurations matching these failed runs will be excluded from the
                returned missing configurations. If None, no failed configs are filtered.
                Note: Depths must be integers. When loading from JSON, use
                :meth:`load_failed_configs_from_json` to load and convert string depths to integers,
                as JSON doesn't support integer keys in dictionaries.
            with_derived_configs: If False (default), filters out missing configurations
                that can be derived from other configurations. For example, if method A
                can be derived from method B, and both are missing, only B will be returned
                (since running B will also provide results for A). If True, no filtering
                is applied and all missing configurations are returned. Defaults to False.

        Returns:
            Dictionary mapping evaluation keys to missing configurations. Keys are either:
                - EvaluationType string (for SV and PP)
                - Tuple of (EvaluationType, with_aer) (for MPS)
            Values are dicts mapping MethodConfigJSON to dicts of Depth to sets of missing instances.

        Warnings:
            - Warns if a target method has no data in the database at all
            - Warns if a target instance is not present in the database at all

        Notes:
            Examples of derivable configurations (when with_derived_configs=False):
            - I_SV.json at depth p can be derived from depth p+1 (interpolation from depth p-1 to depth p+1)
            - F_SV.json and similar methods have the same behaviour
            - TQA_SV_no_opt.json can be derived from TQA_SV_opt.json (zeroth iteration)
            - LR_SV_opt.json can be derived from LR_SV_angle_opt.json (zeroth iteration)
        """
        # Convert target_depths to a set for efficient lookup
        target_depths_set = set(target_depths)

        # Convert failed_configs to a set of tuples for efficient lookup
        failed_set: set[
            tuple[
                EvaluationType | tuple[Literal["MPS"], bool],
                MethodConfigJSON,
                Depth,
                GraphKey,
            ]
        ] = set()
        if failed_configs is not None:
            for instance, methods_dict in failed_configs.items():
                for method, depths_dict in methods_dict.items():
                    for depth, reason in depths_dict.items():
                        # Extract evaluation and with_aer from the method name
                        evaluation = utils.labels.trainer_config_to_evaluation(method)
                        with_aer = utils.labels.method_uses_aer(method)

                        # Determine eval_key based on evaluation type
                        # For MPS, we need to distinguish between Aer and non-Aer
                        if evaluation == "MPS":
                            eval_key = (evaluation, with_aer)
                        else:
                            eval_key = cast(EvaluationType, evaluation)

                        failed_set.add((eval_key, method, depth, instance))

        # Track which target methods and instances we've seen in the database
        # Keys are (eval_key, method) or (eval_key, instance), values are booleans
        seen_methods: dict[
            tuple[EvaluationType | tuple[Literal["MPS"], bool], MethodConfigJSON], bool
        ] = {}
        seen_instances: dict[
            tuple[EvaluationType | tuple[Literal["MPS"], bool], GraphKey], bool
        ] = {}

        # Initialize tracking dictionaries with False for all targets
        for evaluation, methods_value in target_methods.items():
            if isinstance(methods_value, list):
                eval_key = evaluation
                for method in methods_value:
                    seen_methods[(eval_key, method)] = False
            elif isinstance(methods_value, dict):
                # Type-narrowing for evaluation. We should only have dict[bool,
                # ...] values for evaluation="MPS".
                if evaluation != "MPS":
                    raise KeyError(
                        "Only MPS evaluation can have a dict[bool, list[MethodConfigJSON]] "
                        + f"as a value type in target_methods. Got {evaluation!r} instead."
                    )
                for with_aer, methods_list in methods_value.items():
                    eval_key = (evaluation, with_aer)
                    for method in methods_list:
                        seen_methods[(eval_key, method)] = False
            else:
                raise ValueError(
                    f"Unexpected type for target_methods values: {type(methods_value)}."
                )

        for evaluation, instances_value in target_instances.items():
            if isinstance(instances_value, set):
                eval_key = evaluation
                for instance in instances_value:
                    seen_instances[(eval_key, instance)] = False
            elif isinstance(instances_value, dict):
                # Type-narrowing for evaluation. We should only have dict[bool,
                # ...] values for evaluation="MPS".
                if evaluation != "MPS":
                    raise KeyError(
                        "Only MPS evaluation can have a dict[bool, list[MethodConfigJSON]] "
                        + f"as a value type in target_methods. Got {evaluation!r} instead."
                    )
                for with_aer, instances_set in instances_value.items():
                    eval_key = (evaluation, with_aer)
                    for instance in instances_set:
                        seen_instances[(eval_key, instance)] = False
            else:
                raise ValueError(
                    f"Unexpected type for target_instances values: {type(instances_value)}."
                )

        # Build the complete set of target configurations
        target_configs = self._build_target_configs(
            target_methods, target_instances, target_depths_set
        )

        # Build the set of existing configurations from database
        existing_configs = self._build_existing_configs(seen_methods, seen_instances)

        # Issue warnings for methods that were never seen
        for (eval_key, method), was_seen in seen_methods.items():
            if not was_seen:
                if isinstance(eval_key, tuple):
                    evaluation, with_aer = eval_key
                    aer_str = "with Aer" if with_aer else "without Aer"
                    warnings.warn(
                        f"Method {method!r} for evaluation {evaluation!r} ({aer_str}) has no data in the database"
                    )
                else:
                    warnings.warn(
                        f"Method {method!r} for evaluation {eval_key!r} has no data in the database"
                    )

        # Issue warnings for instances that were never seen
        for (eval_key, instance), was_seen in seen_instances.items():
            if not was_seen:
                if isinstance(eval_key, tuple):
                    evaluation, with_aer = eval_key
                    aer_str = "with Aer" if with_aer else "without Aer"
                    warnings.warn(
                        f"Instance {instance!r} for evaluation {evaluation!r} ({aer_str}) is not present in the database"
                    )
                else:
                    warnings.warn(
                        f"Instance {instance!r} for evaluation {eval_key!r} is not present in the database"
                    )

        # Find missing configurations
        missing = target_configs - existing_configs

        # Remove failed configurations from missing set
        if failed_configs is not None:
            missing = self._apply_failed_configs_filter(missing, failed_set)

        # Filter out derived configurations if requested
        if not with_derived_configs:
            configs_to_remove = self._filter_derived_configs(
                missing, failed_set, failed_configs
            )
            missing = missing - configs_to_remove

        # Organize missing configurations into the return structure
        result: dict[
            EvaluationType | tuple[Literal["MPS"], bool],
            dict[MethodConfigJSON, dict[Depth, set[GraphKey]]],
        ] = {}

        for eval_key, method, depth, instance in missing:
            if eval_key not in result:
                result[eval_key] = {}
            if method not in result[eval_key]:
                result[eval_key][method] = {}
            if depth not in result[eval_key][method]:
                result[eval_key][method][depth] = set()
            result[eval_key][method][depth].add(instance)

        return result

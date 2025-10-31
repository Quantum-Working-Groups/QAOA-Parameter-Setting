"""Allows us to parse the training data and summarise each method's performance."""

from collections.abc import Iterable
import glob
import json
import os
from typing import Any, Literal, NoReturn, TypeAlias, TypedDict, overload

import pandas as pd

import qaoa_parameter_setting.utils.instance as Instance
from qaoa_training_pipeline.utils.graph_utils import approximation_ratio


# *** type aliases to make _data type hints easier.
GraphKey: TypeAlias = str
"""GraphKey instance filename."""
TrainerConfig: TypeAlias = str
"""Config filename for trainer."""
Depth: TypeAlias = str
"""Depth of QAOA. For consistency with BestParameters we treat it as a string."""


class ResultDict(TypedDict):
    """TypedDict for internal result dictionary structure."""

    energy: float
    qaoa_angles: list[float]
    result_filename: str
    trainer: str
    evaluation: str


SummaryData: TypeAlias = dict[GraphKey, dict[TrainerConfig, dict[Depth, ResultDict]]]
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

    def __init__(self, filename: str | None = None):
        """Initialize the summary table instance."""
        self._data: SummaryData = {}
        self._minmax_data: MinMaxData = {}
        self._methods: list[str] = []

        if filename is not None:
            with open(filename, "r") as fin:
                _data = json.load(fin)
                self._data = _data["data"]
                self._minmax_data = _data["minmax_data"]
                self._methods = _data["methods"]

    @property
    def data(self) -> SummaryData:
        """Return the data."""
        return self._data

    def trainer_config_to_evaluation(self, trainer_config: TrainerConfig) -> str:
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

    def add_data(self, folder_name: str):
        """Load the training data from a folder and get the best result.

        Best results are chosen as the data with minimum energy for each graph
        key, training method, and QAOA depth."""

        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")
            with open(filename, "r") as fin:
                result = json.load(fin)

            # Get the energy evaluation methodology
            config: TrainerConfig = result["args"]["config"].split("/")[-1]

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
            results = []
            self.populate_results(result, results)

            if graph not in self._data:
                self._data[graph] = dict()
            if config not in self._data[graph]:
                self._data[graph][config] = dict()

            for qaoa_angles, energy, trainer in results:
                if qaoa_angles is None or energy is None:
                    continue

                depth = str(len(qaoa_angles) // 2)

                if depth in self._data[graph][config]:
                    if self._data[graph][config][depth]["energy"] < energy:
                        self._data[graph][config][depth] = {
                            "energy": energy,
                            "qaoa_angles": qaoa_angles,
                            "result_filename": filename.split("/")[-1],
                            "trainer": trainer,
                            "evaluation": evaluation,
                        }
                else:
                    self._data[graph][config][depth] = {
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
            if filename.split("/")[-1] == "example_method.json":
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
            self._methods.append(filename.split("/")[-1])

    def add_minmax_cut_data(self, folder_name: str):
        """Load the min-max cut data from a folder."""
        for filename in glob.glob(f"{folder_name}/*.json"):
            filename = filename.replace("\\", "/")
            with open(filename, "r") as fin:
                _data = json.load(fin)
            graph_path = _data.pop("instance")
            graph_key: GraphKey = graph_path.split("/")[-1]
            result = MinMaxResult(_data)
            if graph_key in self._minmax_data:
                raise KeyError(
                    "Multiple min-max cut data for instance {!r}.".format(graph_key)
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
    def approximation_ratio_for(
        self,
        graph_key: GraphKey,
        energy: float,
        return_none: Literal[False],
    ) -> float | NoReturn: ...
    @overload
    def approximation_ratio_for(
        self,
        graph_key: GraphKey,
        energy: float,
        return_none: Literal[True],
    ) -> float | None: ...
    def approximation_ratio_for(
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
        return approximation_ratio(
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

    def populate_results(self, result: dict, result_data: list) -> list:
        """Get the parameters from the `result` dict."""
        iter_keys = self.get_iter_keys(result.keys())

        for key in iter_keys:
            sub_result = result[key]

            version = sub_result["system_info"]["qaoa_training_pipeline_version"]
            trainer = sub_result["trainer"]["trainer_name"]

            # # Skip known bugs
            # if trainer == "TQATrainer" and version <= 13:
            #     continue

            if trainer == "RecursionTrainer":
                self.populate_results(sub_result, result_data)

            qaoa_angles = sub_result["optimized_qaoa_angles"]
            energy = sub_result["energy"]

            # Ensure length of 2p
            if len(qaoa_angles) % 2 != 0:
                continue

            result_data.append((qaoa_angles, energy, trainer))

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

                depths = [str(v) for v in sorted([int(d) for d in vdict.keys()])]

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
        config: TrainerConfig,
        depth: Depth,
        result: ResultDict,
    ) -> dict[str, Any]:
        _heavy_hex_dimensions = Instance.heavy_hex_dimensions(graph_key)
        _approx_ratio = self.approximation_ratio_for(
            graph_key=graph_key, energy=result["energy"], return_none=True
        )
        return {
            # Pandas will put the first keys as the first columns, so we put the
            # important columns first.
            "graph_type": Instance.graph_type(graph_key),
            "graph_idx": Instance.graph_idx(graph_key),
            "num_nodes": Instance.num_nodes(graph_key),
            "trainer_config": config,
            "depth": depth,
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

"""Allows us to parse the trainig data and extract the best QAOA angles."""

import glob
import json
import os
import warnings


class BestParameterManager:
    """Manages the summary of best parameters.

    The data is a nested dictionary with the following levels:
    - graph_name
    - energy evaluation method
    - QAOA depth

    The value to the innermost key is a dictionary with the entries `energy`,
    `qaoa_angles`, and `result_file_name`. Here, `result_file_name` corresponds to
    the file in which we found the best parameters for a given QAOA depth.
    """

    NONTRAINER_KEYS = {"args", "pre_processing", "cost_operator"}

    def __init__(self, file_name: str = None):
        """Initialize the manager."""
        self._data = dict()

        if file_name is not None:
            with open(file_name, "r") as fin:
                self._data = json.load(fin)

    @property
    def data(self) -> dict:
        """Return the data."""
        return self._data

    def add_data(self, folder_name: str):
        """Load the training data from a folder and get the best QAOA angle values."""

        for file_name in glob.glob(f"{folder_name}/*.json"):
            file_name = file_name.replace("\\", "/")
            with open(file_name, "r") as fin:
                result = json.load(fin)

            # Get the energy evaluation methodology
            config = result["args"]["config"].split("/")[-1]

            if "PP" in config:
                evaluation = "PP"
            elif "MPS" in config:
                evaluation = "MPS"
            elif "SV" in config:
                evaluation = "SV"
            else:
                raise ValueError(
                    f"Unrecognised energy evaluation in {file_name} for method {config}"
                )

            if evaluation not in self._data:
                self._data[evaluation] = dict()

            # Get the graph instance
            graph_input = result["args"]["input"].replace("\\", "/")
            graph = graph_input.split("/")[-1]

            if graph not in self._data[evaluation]:
                self._data[evaluation][graph] = dict()

            # Loop over the trainers in the result
            results = []
            self.populate_results(result, results, filename=file_name)

            for qaoa_angles, energy, trainer, duration in results:
                if qaoa_angles is None or energy is None:
                    continue

                depth = str(len(qaoa_angles) // 2)

                if depth in self._data[evaluation][graph]:
                    if self._data[evaluation][graph][depth]["energy"] < energy:
                        self._data[evaluation][graph][depth] = {
                            "energy": energy,
                            "qaoa_angles": qaoa_angles,
                            "train_duration": duration,
                            "result_file_name": file_name.split("/")[-1],
                            "trainer": trainer,
                        }
                else:
                    self._data[evaluation][graph][depth] = {
                        "energy": energy,
                        "qaoa_angles": qaoa_angles,
                        "train_duration": duration,
                        "result_file_name": file_name.split("/")[-1],
                        "trainer": trainer,
                    }

    @staticmethod
    def get_iter_keys(keys):
        iterations = []
        for key in keys:
            try:
                val = int(key)
                iterations.append(key)
            except ValueError:
                pass

        return iterations

    def populate_results(
        self, result: dict, result_data: list, filename: str | None = None
    ) -> list:
        """Get the parameters from the `result` dict.

        Args:
            result: Input results dictionary to process.
            result_data: Output list of results.
            filename: Optional name of original JSON file for results, for
            warnings. If None, warnings will not refer to a specific file.
        """
        iter_keys = self.get_iter_keys(result.keys())

        for key in iter_keys:
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
                self.populate_results(sub_result, result_data, filename=filename)

            qaoa_angles = sub_result["optimized_qaoa_angles"]
            energy = sub_result["energy"]
            duration = sub_result["train_duration"]

            # Ensure length of 2p
            if len(qaoa_angles) % 2 != 0:
                continue

            result_data.append((qaoa_angles, energy, trainer, duration))

    def save_summary(self, file_name, overwrite: bool = False):
        """Dump the data to a file."""

        if os.path.isfile(file_name) and not overwrite:
            raise ValueError(f"File {file_name} already exists.")

        with open(file_name, "w") as fout:
            json.dump(self._data, fout, indent=4)

    def get_monotonic_data(self) -> dict:
        """Return data that only contains depths for which `E_p > E_q` for `p > q`."""
        monotonic_data = dict()

        for method, mdict in self._data.items():
            if method not in monotonic_data:
                monotonic_data[method] = dict()

            for graph_key, vdict in mdict.items():
                if graph_key not in monotonic_data[method]:
                    monotonic_data[method][graph_key] = dict()

                depths = [str(v) for v in sorted([int(d) for d in vdict.keys()])]

                prev_energy = None
                for depth in depths:
                    if prev_energy is None:
                        monotonic_data[method][graph_key][depth] = vdict[depth]
                        prev_energy = vdict[depth]["energy"]

                    elif prev_energy < vdict[depth]["energy"]:
                        monotonic_data[method][graph_key][depth] = vdict[depth]
                        prev_energy = vdict[depth]["energy"]
                    else:
                        continue

        return monotonic_data

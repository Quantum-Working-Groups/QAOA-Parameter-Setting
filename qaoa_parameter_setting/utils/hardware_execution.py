"""Create and run circuits."""

import glob
import json
import os

import matplotlib.pyplot as plt

from qaoa_parameter_setting.utils.best_parameters import BestParameterManager
from qaoa_parameter_setting.transpilation.make_qaoa_circuits import make_qaoa_circuit


class HarwareExecutor:
    """Class to prepare circuits from results with standardized metadata."""

    def __init__(self, instance_name: str, short_name: str, folder: str, base_path: str="."):
        """
        Args:
            instance_name: This is the name of a graph in the repository. For example,
                '000_5_3_heavyhex_106nodes_weighted.json'.
            short_name: The short name of the instance used to identify the result files.
                This short name must be consistent with the `instance_name` provided.
            folder: The folder under `data/training/` where to find the results.
        """
        self._base_path = base_path
        self._instance_name = instance_name
        self._folder = folder
        self._short_name = short_name

        # Location to hold the template circuits.
        self._circuit = None
        self._swap_circuit = None

        # Load all the training results that match the short name.
        # This will allow us to compare the different methods.
        self._results = []

        for file_name in glob.glob(f"{self._base_path}/data/training/{folder}/*.json"):
            if self._short_name in file_name:
                with open(file_name, "r") as fin:
                    self._results.append(json.load(fin))

        # Validate that all the results point to the same problem instance
        for res in self._results:
            graph_file = res["args"]["input"].split("/")[-1]
            save_file = res["args"]["save_file"]
            if graph_file != self._instance_name:
                raise ValueError(
                    f"Graph instance {graph_file} from the file {save_file} "
                    f"does not matche the anticipated file {self._instance_name}"
                )
        
        # Extract the parameters into a more manageable format
        # For each result file and depth we want to find the best parameters
        # that the method reported.
        self._all_methods_summary = dict()
        manager = BestParameterManager()

        for res in self._results:
            method = res["args"]["config"].split("/")[-1].replace(".json", "")
            summary = []
            manager.populate_results(res, summary)
            summary_dict = dict()
            
            for _res in summary:
                # Keep track of the location where the data came from
                data_entry = (_res[0], _res[1], _res[2], res["args"]["save_file"])

                depth = len(_res[0])//2
                if depth not in summary_dict:
                    summary_dict[depth] = data_entry
                else:
                    energy = _res[1]

                    if energy is None:
                        print(f"No energy for {method}")
                    else:
                        if energy > summary_dict[depth][1]:
                            summary_dict[depth] = data_entry

            self._all_methods_summary[method] = summary_dict
    
    def manual_add_parameters(self, qaoa_angles, energy, method: str, file_name: str ="no file"):
        """Allows us to inject parameters into the _all_methods_summary to construct PUBs."""
        if len(qaoa_angles) % 2 !=0:
            raise ValueError("QAOA angles should have length 2.")
        
        depth = len(qaoa_angles) // 2
        self._all_methods_summary[method] = {depth: (qaoa_angles, energy, method, file_name)}

    def make_sampler_pub(self, backend, qaoa_depth: int, problem_class: str="maxcut", meas_threshold: float=0.0):
        """Prepares the payload for the Sampler."""
        self._circuit = None
        self._swap_circuit = None
        file_name = f"{self._base_path}/instances/{self._folder}/{self._instance_name}"

        # Create a template circuit into which we will assign the parameters.
        self._circuit, self._swap_circuit = make_qaoa_circuit(
            file_name, 
            problem_class=problem_class, 
            reps=qaoa_depth, 
            backend=backend, 
            meas_threshold=meas_threshold,
        )

        sampler_pub = []
        for method, res in self._all_methods_summary.items():
            if qaoa_depth in res:

                mcirc = self._circuit.assign_parameters(res[qaoa_depth][0], inplace=False)
                mcirc.metadata["method"] = method
                mcirc.metadata["params"] = res[qaoa_depth][0]
                mcirc.metadata["eval_energy"] = res[qaoa_depth][1]
                mcirc.metadata["trainer"] = res[qaoa_depth][2]
                mcirc.metadata["result_file"] = res[qaoa_depth][3]
                mcirc.metadata["short_name"] = self._short_name

                sampler_pub.append(mcirc)

        return sampler_pub

    def save_result_summary(self, job, overwrite: bool=False):
        """Make a serializable object of the results that we can json dump."""
        result = job.result()
        job_id = job.job_id()
        backend_name = job.backend().name

        result_summary = []
        for res in result:
            entry = {
                "job_id": job_id,
                "metadata": res.metadata,
                "counts": res.data.c.get_counts(),
                "backend_name": backend_name,
            }

            result_summary.append(entry)

        # Save to JSon in data/ folder
        file_short_name = result[0].metadata["circuit_metadata"]["short_name"]
        name = f"{file_short_name}_{job_id}.json" 

        file_name = f"{self._base_path}/data/hardware/{self._folder}/{name}"

        if os.path.isfile(file_name) and not overwrite:
            raise ValueError(f"File {file_name} already exists.")

        with open(file_name, "w") as fout:
            json.dump(result_summary, fout)

        return result_summary

    def plot_qaoa_angles(self, qaoa_depth, fig=None, axis1=None, axis2=None, plot_args: dict=None):
        """Plot the QAOA betas in axis1 and the gammas in axis2"""

        if fig is None or axis1 is None or axis2 is None:
            fig, axis = plt.subplots(1, 2)
            axis1, axis2 = axis[0], axis[1]

        if plot_args is None:
            plot_args = dict()

        for method, res in self._all_methods_summary.items():
            if qaoa_depth not in res:
                continue
        
            angles = res[qaoa_depth][0]
        
            p = len(angles) // 2
            axis1.plot(angles[:p], label=method, **plot_args.get(method, dict()))
            axis2.plot(angles[p:], label=method, **plot_args.get(method, dict()))


        axis1.set_xlabel("QAOA depth")
        axis2.set_xlabel("QAOA depth")

        axis1.set_ylabel(r"$\beta$")
        axis2.set_ylabel("$\\gamma$")
        
        return fig, axis1, axis2

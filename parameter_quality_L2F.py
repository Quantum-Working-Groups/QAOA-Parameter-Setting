import json
import glob
import matplotlib.pyplot as plt

from qaoa_training_pipeline.utils.graph_utils import load_graph, graph_to_operator

from qaoa_parameter_setting.utils.hardware_execution import HarwareExecutor

instance_keys = ["100nodes", "2swap_layers"]
selected_instances = []
folder = "line_to_full"
short_name_base = "N100L2S2"

for file_name in glob.glob(f"instances/{folder}/*"):
    file_name = file_name.replace("\\", "/").split("/")[-1]
    selected = all(key in file_name for key in instance_keys)
    if selected:
        selected_instances.append((file_name, file_name[0:3] + short_name_base))

graphs = {inst: load_graph(f"instances/{folder}/{inst}") for inst, _ in selected_instances}
cost_ops = {inst: graph_to_operator(graph, pre_factor=-0.5) for inst, graph in graphs.items()}

print(selected_instances)

executors = []
folder = "line_to_full"
for instance, short_name in selected_instances:
    executors.append(HarwareExecutor(instance, short_name, folder))

# from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
# service = QiskitRuntimeService(instance="PremAccess")
# backend = service.backend("ibm_boston")
# pub = []

# for executor in executors:
#     pub += executor.make_sampler_pub(backend, qaoa_depth=2, problem_class="maxcut")
# print("Instruction count:", pub[0].count_ops())
# print("2Q gate depth:", pub[0].depth(lambda x: len(x.qubits) > 1))

# sampler = SamplerV2(mode=backend)
# job = sampler.run(pub, shots=2**13)
# print(job.status())
# print(job.job_id())
# job_ = service.job(job.job_id())
# res_summary = executor.save_result_summary(job_)

short_name = "003N100L2S2"
with open(f"data/hardware/{folder}/{short_name}_d5kss5v853es738ea5h0.json", "r") as fin:
    res_summary = json.load(fin)
import numpy as np

from qaoa_training_pipeline.visualization.plotting import plot_cdf
from qaoa_training_pipeline.utils.graph_utils import solve_max_cut

from qopt_best_practices.cost_function.cost_utils import counts_to_maxcut_cost

from qaoa_parameter_setting.utils.analysis import mean_obj, standard_error_mean
from pathlib import Path

m_cuts = {}
path = Path("data/minmax_cuts/line_to_full")
for inst, cost_op in cost_ops.items():
    print(inst)
    inst_path = Path(inst)
    instmaxmin_name = inst_path.with_name(inst_path.stem + "_maxmin_cut" + inst_path.suffix)
    file = path / instmaxmin_name
    if file not in path.iterdir():
        print(f"Not found {instmaxmin_name}")
        break
        max_cut, min_cut, _ = solve_max_cut(cost_op)
    else: 
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        max_cut = data["max_cut"]
        min_cut = data["min_cut"]
    m_cuts[inst] = (max_cut, min_cut)
mc_costs, means, sems, mc_insts = [], [], [], []

for res in res_summary:
    inst = res["metadata"]["circuit_metadata"]["file_name"].split("/")[-1]
    eval_energy = res["metadata"]["circuit_metadata"]["eval_energy"]
    
    
    mc_costs.append(counts_to_maxcut_cost(graphs[inst], res["counts"]))
    
    means.append(100*(mean_obj(mc_costs[-1]) - m_cuts[inst][1])/(m_cuts[inst][0] - m_cuts[inst][1]))

    sems.append(100*(standard_error_mean(mc_costs[-1], sum(res["counts"].values())))/(m_cuts[inst][0] - m_cuts[inst][1]))
    mc_insts.append(inst)
means = [100*(mean_obj(mc_cost) - min_cut)/(max_cut - min_cut) for mc_cost in mc_costs]
sems = np.array([100*(standard_error_mean(mc_cost, sum(res_summary[idx]["counts"].values())))/(max_cut - min_cut) for idx, mc_cost in enumerate(mc_costs)])
def approx_ratio(max_cut, min_cut, graph, energy):
    sum_weights = sum(val[2].get("weight", 1.0) for val in graph.edges(data=True))

    cut_val = energy + 0.5 * sum_weights

    return (cut_val - min_cut) / (max_cut - min_cut)
from collections import defaultdict

per_lbl_m, per_lbl_s = defaultdict(list), defaultdict(list)

per_lbl_ar = defaultdict(list)

# Goupe the data
for idx, mc_cost in enumerate(mc_costs):
    label = res_summary[idx]["metadata"]["circuit_metadata"]["method"]
    per_lbl_m[label].append(means[idx])
    per_lbl_s[label].append(sems[idx])

    inst = mc_insts[idx]
    max_cut, min_cut = m_cuts[inst]
    ar = approx_ratio(max_cut, min_cut, graphs[inst], res_summary[idx]["metadata"]["circuit_metadata"]["eval_energy"])
    per_lbl_ar[label].append(ar * 100)

fig, ax = plt.subplots(figsize=(6, 6))


markers = {"I_MPS": "s", "I_PP": "X", "F_PP": "D", "F_MPS": ">", "PT_PP_AAAM": "o", "FA_PP_opt": "v", "FA_MPS_opt": "^", "TQA_PP_opt": "<",  "TQA_MPS_opt": "1"}
colors = plt.cm.Set1.colors
colors = {"I_MPS": colors[0], "I_PP": colors[1], "F_PP": colors[2], "F_MPS": colors[3], "PT_PP_AAAM": colors[4], "FA_PP_opt": colors[5], "FA_MPS_opt": colors[6], "TQA_PP_opt": colors[7],  "TQA_MPS_opt": colors[8]}
nice_labels = {
    "I_MPS": "INTERP with MPS", 
    "I_PP": "INTERP with PP", 
    "F_PP": "Fourier with PP", 
    "F_MPS": "Fourier with MPS",
    "FA_PP_opt": "Fixed angle with PP and opt",
    "FA_MPS_opt": "Fixed angle with MPS and opt", 
    "TQA_PP_opt": "TQA with PP and opt",  
    "TQA_MPS_opt": "TQA with MPS and opt",
    "PT_PP_AAAM": "PT with PP, AAA, and MDM"
}

counter = 0
for lbl, val in per_lbl_m.items():
    opts = {"marker": markers[lbl], "mec": "k", "mew": 0.75, "color": colors[lbl], "ls": "none", "ecolor": "k", "capsize": 3, "elinewidth": 1}
    avg_ar, std_ar = np.average(per_lbl_ar[lbl]), np.std(per_lbl_ar[lbl])
    avg_hw, std_hw = np.average(per_lbl_m[lbl]), np.std(per_lbl_m[lbl])
    ax.errorbar(per_lbl_ar[lbl], per_lbl_m[lbl], yerr=per_lbl_s[lbl], ms=4, alpha=0.5, **opts)
    ax.errorbar([avg_ar], [avg_hw], yerr=[std_hw], xerr=[std_ar], ms=7, label=nice_labels[lbl], **opts)
    counter += 1

plot_args = {
    key: {"color": colors[key]} for key in ["I_MPS", "F_MPS", "I_PP", "F_PP", "PT_PP_AAAM", "FA_PP_opt", "FA_MPS_opt", "TQA_PP_opt",  "TQA_MPS_opt"]
}
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

ax.set_xlabel("Estimated approximation ratio (%)")
ax.set_ylabel("Hardware approximation ratio (%)")
ax.set_title("N100L2S2 depth 2")

fig.savefig("figures/20260112_d5kss5v853es738ea5h0_N100L2S2.pdf", bbox_inches="tight")
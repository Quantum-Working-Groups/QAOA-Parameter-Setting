"""A method to make hardware native QAOA circuits."""

from typing import List
from warnings import warn
import networkx as nx
import rustworkx as rx
from networkx.algorithms import isomorphism
import numpy as np


from qiskit.circuit.library import CXGate
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.providers import Backend
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    Commuting2qGateRouter,
)
from qiskit.transpiler.passes import (
    BasisTranslator,
    UnrollCustomDefinitions,
    HighLevelSynthesis,
    InverseCancellation,
)

from qaoa_training_pipeline.utils.problem_classes import MaxCut, MaxIndependentSet
from qaoa_training_pipeline.utils.data_utils import load_input
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph

from qopt_best_practices.qubit_selection import BackendEvaluator
from qopt_best_practices.transpilation.cost_layer import get_cost_layer
from qopt_best_practices.transpilation.prepare_cost_layer import PrepareCostLayer
from qopt_best_practices.transpilation.qaoa_construction_pass import (
    QAOAConstructionPass,
)
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping
from qopt_best_practices.sat_mapping import SATMapper

from qaoa_parameter_setting.transpilation.swap_strategies import make_2d_grid_swap_strategy


# Exclusion list for Heron 156 qubits to get a line.
HERON_EXCLUDE = [
    16,
    17,
    18,
    37,
    38,
    39,
    56,
    57,
    58,
    77,
    78,
    79,
    96,
    97,
    98,
    117,
    118,
    119,
    136,
    137,
    138,
]


def get_a_path(backend: Backend, length: int, nodes_to_exclude: List[int] = None):
    """Finding good paths at utility-scale can be very time inefficient.

    This function returns a path without any consideration for fidelity.
    Furthermore, it requires that the user remove nodes from the backend to
    simplify its coupling map to limit the space of simple paths.

    args:
        backend: The backend for which we want to get a path.
        nodes_to_exclude:
    """

    coupling_map = CouplingMap(backend.coupling_map)

    # Remove nodes to simplify the coupling map
    nodes_to_exclude = nodes_to_exclude or HERON_EXCLUDE

    for node in nodes_to_exclude:
        coupling_map.graph.remove_node(node)

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=length,
        cutoff=length,
    ).values()

    paths = np.asarray(
        [
            (list(c), list(sorted(list(c))))
            for a in iter(all_paths)
            for b in iter(a)
            for c in iter(a[b])
        ]
    )

    # filter out duplicated paths
    _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)
    paths = paths[:, 0][unique_indices].tolist()

    return paths[0]


def make_qaoa_circuit(
    file_name: str,
    problem_class: str,
    reps: int,
    backend,
    sat_timeout: int = 60,
    find_layout: bool = False,
    meas_threshold: float = 0.99,
):
    """Specify a problem instance by name and make a hardware native QAOA circuit.

    The graph instances are transpiled in the following way in this repository
    1. random-k-regular graphs are SAT mapped and stranspiled to a line with a swap strategy.
    2. Erdos-Renyi graphs are SAT mapped and stranspiled to a line with a swap strategy.
    3. Hardware native heavy-hex graphs are mapped directly to the heavy-hex hardware.
    4. Graphs from a line to fully connected are directly implemented with a swap strategy.

    Args:
        file_name: The name of the file in which there is a graph to load.
        problem_class: Either `maxcut` or `mis:X` where `X` is the value of the penalty.
        reps: The number of QAOA layers.
        find_layout: Find a layout of qubits to use. This can take a lot of time depending on
            the problem size. If this is set to false then we chose a simple line on the
            heavy-hex map without any considerations for fidelity.
        meas_threshold: When looking for heavy-hex rings we exclude qubits with a measurement
            fidelity below this threshold. For large graves this likely needs to be reduced
            from the default value.
    """

    # 1. Load and pre-process data into a cost-operator.
    input_data = load_input(file_name)

    graph_type = None
    if "random_regular" in file_name:
        graph_type = "random_regular"
    elif "erdos_renyi" in file_name:
        graph_type = "erdos_renyi"
    elif "line_to_full" in file_name:
        graph_type = "line_to_full"
    elif "heavy_hex" in file_name:
        graph_type = "heavy_hex"
    else:
        raise ValueError("Invalid graph type.")

    if problem_class == "maxcut":
        cost_op = MaxCut().cost_operator(input_data)
    elif problem_class[0:3] == "mis":
        mis = MaxIndependentSet.from_str(problem_class[4:])
        cost_op = mis.cost_operator(input_data)
    else:
        raise ValueError("Invalid problem class.")

    num_qubits = cost_op.num_qubits

    # 2. Make the hardware native circuit.

    ## 2.1 get an edge coloring and swap network for the router, also do the SAT mapping
    layout_info = {}
    sat_mapper, edge_map, min_k = None, "", ""

    if graph_type == "heavy_hex":
        graph = dict_to_graph(input_data)

        # Make the edge colors
        edge_coloring = nx.greedy_color(
            nx.line_graph(graph), strategy="saturation_largest_first"
        )
        edge_coloring.update({(k[1], k[0]): v for k, v in edge_coloring.items()})

        num_colors = len(set(edge_coloring.values()))

        if num_colors > 3:
            raise ValueError(
                "something in the heavy-hex edge coloring went wrong."
                f"Got {num_colors}, was expecting 3."
            )

        # Make an empty swap strategy
        cmap = CouplingMap(graph.edges())
        cmap.make_symmetric()

        swap_strat = SwapStrategy(cmap, ())  # no SWAPs needed
    else:
        if backend.name in ["ibm_miami"]:
            edge_coloring = None

            # Find the smallest rectangular grid that fits the graph.
            rows, cols = int(np.ceil(np.sqrt(num_qubits))), int(np.floor(np.sqrt(num_qubits)))
    
            if rows*cols < num_qubits:
                if rows < cols:
                    rows += 1
                else:
                    cols += 1

            # Seems like the SATMApper preferes more rows?
            if rows < cols:
                rows, cols = cols, rows

            print("rows and cols:", rows, cols)

            swap_strat = make_2d_grid_swap_strategy(rows, cols)
        else:
            edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits - 1)}

            swap_strat = SwapStrategy.from_line(range(num_qubits))

        # SAT map problems that need SAT mapping.    
        if graph_type in ["random_regular", "erdos_renyi"]:
            if sat_timeout > 0:
                sat_mapper = SATMapper(timeout=sat_timeout)
                cost_op, edge_map, min_k = sat_mapper.remap_graph_with_sat(
                    graph=cost_op, 
                    swap_strategy=swap_strat,
                )

    ## 2.2 Get the QAOA cost layer.
    cost_layer = get_cost_layer(cost_op)

    ## 2.3 Find a qubit layout that is good.
    if graph_type == "heavy_hex":
        best_layout, quality, num_subsets = get_best_hex_ring(
            file_name,
            backend,
            meas_threshold=meas_threshold,
        )
        initial_layout = Layout(
            {cost_layer.qregs[0][k]: v for k, v in best_layout.items()}
        )
        layout_info["fidelity"] = {"2Q gates": quality[0], "meas": quality[1]}
        layout_info["path"] = best_layout
        layout_info["num_subsets"] = num_subsets
    else:

        if backend.name in ["ibm_miami"]:
            best_layout, quality, num_subsets = get_best_subgrid(
                rows,
                cols,
                backend,
                meas_threshold=meas_threshold,
            )

            initial_layout = Layout(
                {cost_layer.qregs[0][k]: v for k, v in best_layout.items()}
            )
            layout_info["fidelity"] = {"2Q gates": quality[0], "meas": quality[1]}
            layout_info["path"] = best_layout
            layout_info["num_subsets"] = num_subsets

        else:
            if find_layout:
                path_finder = BackendEvaluator(backend)
                path, fidelity, num_subsets = path_finder.evaluate(num_qubits)
            else:
                path, fidelity, num_subsets = get_a_path(backend, num_qubits), "NA", "NM"

            initial_layout = Layout.from_intlist(path, cost_layer.qregs[0])

            layout_info["path"] = path
            layout_info["fidelity"] = fidelity
            layout_info["num_subsets"] = num_subsets

    ## 2.4 Construct the pass manager

    # Apply the SWAP strategy to the cost layer.
    pre_init = PassManager(
        [
            PrepareCostLayer(),
            Commuting2qGateRouter(swap_strat, edge_coloring),
            SwapToFinalMapping(),  # Removes unnecessary SWAP gates that the end of the block
            HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
            InverseCancellation(gates_to_cancel=[CXGate()]),
        ]
    )

    # Unroll gates
    post_init = PassManager(
        [
            UnrollCustomDefinitions(
                _sel, basis_gates=backend.operation_names, min_qubits=3
            ),
            BasisTranslator(_sel, target_basis=backend.operation_names, min_qubits=3),
        ]
    )

    staged_pm = generate_preset_pass_manager(3, backend, initial_layout=initial_layout)
    staged_pm.pre_init = pre_init
    staged_pm.init = PassManager([QAOAConstructionPass(num_layers=reps)])
    staged_pm.post_init = post_init

    ## 2.5 Run the pass manager
    isa_circuit = staged_pm.run(cost_layer)

    isa_circuit.metadata["file_name"] = file_name
    isa_circuit.metadata["problem_class"] = problem_class
    isa_circuit.metadata["layout_info"] = layout_info

    if sat_mapper is not None:
        isa_circuit.metadata["sat mapper"] = {"sat_edge_map": edge_map, "sat_min_k": min_k}

    # 3. Validation: return the swap routing so that we can check it.
    validation_pm = PassManager(
        [
            PrepareCostLayer(),
            Commuting2qGateRouter(swap_strat, edge_coloring),
            SwapToFinalMapping(),
        ]
    )

    swap_circuit = validation_pm.run(cost_layer)

    if graph_type == "heavy_hex":
        if swap_circuit.depth(lambda x: len(x.qubits) == 2) > 3:
            raise ValueError(
                "Something went wrong with the heavy-hex routing. "
                f"The circuit depth {swap_circuit.depth()} is instead of 3."
            )

    return isa_circuit, swap_circuit


def get_best_subgrid(rows, cols, backend, meas_threshold: float = 0.99):

    # backend coupling map and properies.
    props = backend.properties()
    cmap = nx.from_edgelist(backend.coupling_map.get_edges())

    graph = nx.grid_2d_graph(rows, cols)
    graph.name = f"{rows}x{cols} 2D grid graph."

    match_dict, quality, num_qualities = get_and_evaluate_matches(cmap, graph, props, meas_threshold)

    # For 2D grids the nodes are indexed by tuples but we need a single number
    match_dict = {k[0] * cols + k[1]: v for k, v in match_dict.items()}

    return match_dict, quality, num_qualities


def get_best_hex_ring(file_name: str, backend, meas_threshold: float = 0.99):
    """This function finds good qubits for heavy-hex graphs."""

    graph = dict_to_graph(load_input(file_name))
    graph = nx.from_edgelist(graph.edges())
    graph.name = f"graph loaded from {file_name}"

    # backend coupling map and properies.
    props = backend.properties()
    cmap = nx.from_edgelist(backend.coupling_map.get_edges())

    return get_and_evaluate_matches(cmap, graph, props, meas_threshold) 


def get_and_evaluate_matches(cmap, graph, props, meas_threshold):


    unique_matches = set()
    match_dicts = []

    matcher = isomorphism.GraphMatcher(cmap, graph)
    for match_dict in matcher.subgraph_isomorphisms_iter():
        cmap_nodes = tuple(sorted(match_dict.keys()))

        if cmap_nodes not in unique_matches:
            unique_matches.add(cmap_nodes)
            match_dicts.append(
                {v: k for k, v in match_dict.items()}
            )  # store the inverse mapping

    if len(match_dicts) == 0:
        raise ValueError(f"No matches found for {graph.name}.")

    qualities = []
    max_fid, best_idx = 0, 0
    for idx, match_dict in enumerate(match_dicts):
        gate_fidelity = 1.0
        meas_fidelity = float(
            np.prod([1 - props.readout_error(v) for v in match_dict.values()])
        )
        for u, v in graph.edges():
            gate_fidelity *= 1 - props.gate_error("cz", (match_dict[u], match_dict[v]))

        qualities.append((gate_fidelity, meas_fidelity))

        if gate_fidelity > max_fid and meas_fidelity >= meas_threshold ** graph.order():
            best_idx = idx
            max_fid = gate_fidelity

    if qualities[best_idx][0] == 0:
        warn(f"No good rings found to execute {graph.name} on based on gate fidelity.")

    if qualities[best_idx][1] == 0:
        warn(f"No good rings found to execute on {graph.name} based on measurement fidelity.")

    return match_dicts[best_idx], qualities[best_idx], len(qualities)

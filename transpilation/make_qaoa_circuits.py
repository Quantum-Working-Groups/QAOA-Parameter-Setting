
"""A method to make hardware native QAOA circuits."""

import networkx as nx

from qiskit.circuit.library import CXGate
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
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
    InverseCancellation
)

from qaoa_training_pipeline.pre_processing import SATMapper
from qaoa_training_pipeline.utils.problem_classes import MaxCut, MaxIndependentSet
from qaoa_training_pipeline.utils.data_utils import load_input
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph

from qopt_best_practices.qubit_selection import BackendEvaluator
from qopt_best_practices.transpilation.cost_layer import get_cost_layer
from qopt_best_practices.transpilation.prepare_cost_layer import PrepareCostLayer
from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


from qiskit.transpiler import TransformationPass


def make_qaoa_circuit(
    file_name: str, 
    problem_class: str, 
    reps: int,
    backend,
    sat_timeout: int = 60, 
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

    # 1.1 SAT map problems that need SAT mapping
    sat_mapper = None
    if graph_type in ["random_regular", "erdos_renyi"]:
        if sat_timeout > 0:
            sat_mapper = SATMapper(timeout=sat_timeout)
            input_data = sat_mapper(input_data)

    if problem_class == "maxcut":
        cost_op = MaxCut().cost_operator(input_data)
    elif problem_class[0:3] == "mis":
        mis = MaxIndependentSet.from_str(problem_class[4:])
        cost_op = mis.cost_operator(input_data)
    else:
        raise ValueError("Invalid problem class.")

    # 2. Get the QAOA cost layer.
    num_qubits = cost_op.num_qubits

    cost_layer = get_cost_layer(cost_op)

    # 3. Make the hardware native circuit.

    ## 3.1 get an edge coloring and swap network for the router
    if graph_type == "heavy_hex":
        graph = dict_to_graph(input_data)

        # Make the edge colors
        edge_coloring = nx.greedy_color(nx.line_graph(graph), strategy="saturation_largest_first")
        edge_coloring.update({(k[1], k[0]): v for k, v in edge_coloring.items()})

        num_colors = len(set(edge_coloring.values()))

        if num_colors != 3:
            raise ValueError(
                "something in the heavy-hex edge coloring went wrong."
                f"Got {num_colors}, was expecting 3."
            )
        
        # Make an empty swap strategy
        cmap = CouplingMap(graph.edges())
        cmap.make_symmetric()

        swap_strat = SwapStrategy(cmap, ())  # no SWAPs needed
    else:
        edge_coloring = {(idx, idx + 1): idx % 2 for idx in range(num_qubits - 1)}

        swap_strat = SwapStrategy.from_line(range(num_qubits))

    ## 3.2 Find a qubit layout that is good. (TODO)
    layout_info = {}
    if graph_type == "heavy_hex":
        initial_layout = None
        raise NotImplementedError("Hex finding is not yet supported.")
    else:
        path_finder = BackendEvaluator(backend)
        path, fidelity, num_subsets = path_finder.evaluate(num_qubits)
        initial_layout = Layout.from_intlist(path, cost_layer.qregs[0])
        layout_info = {
            "path": path,
            "fidelity": fidelity,
            "num_subsets": num_subsets,
        }

    ## 3.3 Construct the pass manager

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
            UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names, min_qubits=3),
            BasisTranslator(_sel, target_basis=backend.operation_names, min_qubits=3),
        ]
    )

    staged_pm = generate_preset_pass_manager(3, backend, initial_layout=initial_layout)
    staged_pm.pre_init = pre_init
    staged_pm.init = PassManager([QAOAConstructionPass(num_layers=reps)])
    staged_pm.post_init = post_init

    ## 3.4 Run the pass manager
    isa_circuit = staged_pm.run(cost_layer)

    isa_circuit.metadata["file_name"] = file_name
    isa_circuit.metadata["problem_class"] = problem_class
    isa_circuit.metadata["layout_info"] = layout_info

    if sat_mapper is not None:
        isa_circuit.metadata["sat mapper"] = sat_mapper.to_config()

    # 4. Validation TODO
    validation_pm = PassManager(
        [
            PrepareCostLayer(),
            Commuting2qGateRouter(swap_strat, edge_coloring),
            SwapToFinalMapping(),
        ]
    )

    return isa_circuit, validation_pm.run(cost_layer)

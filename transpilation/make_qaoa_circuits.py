
"""A method to make hardware native QAOA circuits."""

import networkx as nx

from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz, CXGate
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy, 
    Commuting2qGateRouter,
)
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)
from qiskit.transpiler.passes import (
    BasisTranslator,
    UnrollCustomDefinitions,
    HighLevelSynthesis,
    InverseCancellation
)

from qaoa_training_pipeline.utils.problem_classes import MaxCut, MaxIndependentSet
from qaoa_training_pipeline.utils.data_utils import load_input
from qaoa_training_pipeline.utils.graph_utils import dict_to_graph

from qopt_best_practices.qubit_selection import BackendEvaluator
from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


def make_qaoa_circuit(
    file_name: str, 
    problem_class: str, 
    turn_off_sat: bool = False, 
    backend = None,
):
    """Specify a problem instance by name and make a hardware native QAOA circuit.
    
    The graph instances are transpiled in the following way in this repository
    1. random-k-regular graphs are SAT mapped and stranspiled to a line with a swap strategy.
    2. Erdos-Renyi graphs are SAT mapped and stranspiled to a line with a swap strategy.
    3. Hardware native heavy-hex graphs are mapped directly to the heavy-hex hardware.
    4. Graphs from a line to fully connected are directly implemented with a swap strategy.
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
    if graph_type in ["random_regular", "erdos_renyi"]:
        raise NotImplementedError("SAT mapping not implemented yet.")

    if problem_class == "maxcut":
        cost_op = MaxCut().cost_operator(input_data)
    elif problem_class[0:3] == "mis":
        mis = MaxIndependentSet.from_str(problem_class)
        cost_op = mis.cost_operator(input_data)
    else:
        raise ValueError("Invalid problem class.")

    # 2. Get the QAOA cost layer.
    num_qubits = cost_op.num_qubits

    dummy_mixer_operator = SparsePauliOp.from_sparse_list(
        [("I", [i], 1) for i in range(num_qubits)], 
        num_qubits,
    )

    cost_layer = qaoa_ansatz(
        cost_op,
        reps=1,
        initial_state=QuantumCircuit(num_qubits),
        mixer_operator=dummy_mixer_operator,
        name="QAOA cost block",
    )

    # qaoa_ansatz will have a left-over beta parameter which we set to zero.
    cost_layer.assign_parameters({cost_layer.parameters[0]: 0}, inplace=True)

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
        edge_coloring.update({(idx, idx + 1): idx % 2 for idx in range(num_qubits - 1)})

        swap_strat = SwapStrategy.from_line(range(num_qubits))

    ## 3.2 Find a qubit layout that is good. (TODO)
    if graph_type == "heavy_hex":
        initial_layout = None
        raise NotImplementedError("Hex finding is not yet supported.")
    else:
        path_finder = BackendEvaluator(backend)
        path, fidelity, num_subsets = path_finder.evaluate(num_qubits)
        initial_layout = Layout.from_intlist(path, cost_layer.qregs[0])

    ## 3.3 Construct the pass manager

    # Apply the SWAP strategy to the cost layer.
    pre_init = PassManager(
        [
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
    staged_pm.init = PassManager([QAOAConstructionPass(reps=3)])
    staged_pm.post_init = post_init

    dag_circ = circuit_to_dag(cost_layer)
    instruction = Commuting2qBlock(list(dag_circ.topological_op_nodes()))

    block_qaoa_ansatz = QuantumCircuit(num_qubits)
    block_qaoa_ansatz.compose(instruction, inplace=True)

    isa_circuit = staged_pm.run(block_qaoa_ansatz)

    # TODO: do some validation

    return isa_circuit

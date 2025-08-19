# Results and data

This folder contains any data and results that we obtaine.
It contains the following three sub-folders

* **examples**: a folder with example data to show data structures in the result files.
* **hardware**: a folder to contain data from quantum hardware, such as samples from QAOA circuits.
* **simulations**: a folder to contain the results from QAOA parameter training runs done on classical hardware.

## Naming convention

The file names are designed to be traceable back to the instance name and method.
To keep the names short we use abbreviations.
The format of the result files is `time_graph_class_trainer_evaluator_tag_depth.json`.
Here, the **`graph`** has the format `<Index><NumNodes><Type>`. 
The `<Index>` is the three digits that we have to identify different instances.
The `<NumNodex>` is the number of nodex given, e.g., as `N50` for a 50 node graph.
The `<Type>` is the type of graph like `HH` for heavy-hex or `ER` for Erdos-Renyi.
For example, the four classes of graphs we have are abbriviated as follows

* `000N12ER30` for `000_12nodes_erdosrenyi30percent.json`. Here, the last `30` represents the 30% edge probability.
* `001N49HH32` for `001_3_2_heavyhex_49nodes_weighted.json`. Here, the last `32` represents the `3_2` which means 3 times 2 heavy-hex rings.
* `003N18L2S12` for `003_18nodes_12swap_layers.json`. Here, the last `12` represents the 12 swap layers.
* `005N35R8R` for `005_35nodes_random8regular.json`.

The **`class`** in the file format is either `MC` for maximum-cut problems or `MISX` for maximum independent set where `X` is the weight of the penalty.
Typically, we will set a weight of 2 for this penalty, i.e., `MIS2`.

The **`trainer`** corresponds to the trainer method with the same abbreviation. The abbreviations are the following

* `F` for Fourier.
* `TQA` for the Trotterized Quantum Annealing.
* `RTS` for recursive transition states.
* `FA` for the fixed-angle conjecture.
* `I` for the INTERP method.

The **`evaluator`** corresponds to the methods used to evaluate the energy. 
We have the following conventions

* `SV` for state-vector.
* `MPS` for the matrix-product state. In the future we might add other MPS methods and they will receive their own different tag.
* `PP` for Pauli propagation.

The **`tag`** can represent somthing like `opt` or `noOpt` for optimization or no optimization, respectively.

Finally, the **`depth`** represents the QAOA depth and is an int.

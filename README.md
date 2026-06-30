# Setting angles in quantum approximate optimization at utility-scale
We investigate the performance of various Quantum Approximate Optimization Algorithm (QAOA) angle-setting strategies across different problem instances including utility-scale graphs, both in simulation and on IBM's superconducting hardware, 

[![arXiv](https://img.shields.io/badge/arXiv-2606.05311-B31B1B.svg)](https://arxiv.org/abs/2606.05311)
[![](https://img.shields.io/badge/GitHub-qaoa--training--pipeline-181717?logo=github)](https://github.com/qiskit-community/qaoa_training_pipeline)

## Description
This repository contains the data, experimental results, and code to produce figures, tables, and data management for the paper **Setting angles in quantum approximate optimization at utility-scale**.

##  Table of Contents

- [Overview](#overview)
- [Angle-setting Strategies](#angle-setting-strategies)
- [Energy Evaluation Methods](#energy-evaluation-methods)
- [Problems studied](#problems-studied)
- [Graph instances](#graph-instances)
- [High-level Repository Structure](#high-level-repository-structure)
- [Folder descriptions](#folder-descriptions)
- [Exploring the data](#exploring-the-data)
- [Contributors and acknowledgments](#contributors-and-acknowledgments)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

##  Overview

A critical challenge in QAOA is determining the optimal angles to achieve good performance, and there is multiple angle-setting methods available. However, it is unclear which methods are best and under which conditions or for which problems. This work provides practical guidance for practitioners on how to set QAOA angles by:
- Benchmark of 44 different combinations of angle-setting strategies and energy evaluation methods
- 4 different graph families: Random regular, Erdos-Renyi, line-based, and heavy hex. 
- Hardware evaluation and statistical analysis of the achieved performance in hardware
- Parameter transfer from small-scale to utility scale instances
- Resource and cost analysis of the different methods and running the overall pipeline. 

### Angle-setting Strategies

The following strategies have been benchmark within this work:

| Method | Description | Reference |
|----------|----------|----------|
| Fixed-Angle Conjecture (FA) | Provides universally good angles with a guaranteed lower bound for MaxCut on random regular graphs | [Wurtz & Love](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042612), [Wurtz & Lykov](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.052419), [fixed-angle-QAOA](https://github.com/danlkv/fixed-angle-QAOA) |
| Trotterized Quantum Annealing (TQA)   | Quantum annealing with one-slope linear schedule performed with QAOA via Trotterization and time discretization  | [Sack et al.](https://quantum-journal.org/papers/q-2021-07-01-491/)     |
| INTERP (I)     | Recursive interpolation and optimization of the QAOA angles| [Zhou et al.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067)     |
| FOURIER (F) | Recursive optimization of the QAOA angles in a sine/cosine basis      | [Zhou et al.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067)     |
| Transition States (TS) | Recursive greedy initialization via Transition States | [Sack et al.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062404)     |
| Linear ramps method (LR) | TQA with a two-slope linear schedule     | [Montanez-Barrera & Michielsen](https://www.nature.com/articles/s41534-025-01082-1)     |
| Parameter Transfer (PT)| Transfer of QAOA angles from small-scale problems to utility-scale problems, based on numerical observations of angles clustering for similar graphs  | -     |

### Energy evaluation methods

Each method is tested with different energy evaluators:

| Evaluator| Description |
|----------|----------|
| Statevector (SV)    | Exact simulation and extraction of the energy, only feasible for small-scale instances|
| Matrix Product States (MPS) | Approximate simulation and extraction of the energy, feasible for utility-scale instances. Represents the state via tensor network with chain topology|
| Pauli Propagation (PP)    | Approximate simulation and extraction of the energy, feasible for utility-scale instances. Computes expectation values via observable backpropagation|


### Problems studied
| Problem| Description |
|----------|----------|
| Maximum Cut (MaxCut)| Given a graph $G = (V, E)$, find the partitioning $V_0 \cap V_1 = \emptyset$, $V = V_0 \cup V_1$ such that the sum of edges between partitions is maximized.| 
| Maximum Independent Set (MIS)|Given a graph $G = (V, E)$, find the largest $S \subseteq V$ such that no nodes in $S$ share an edge.|
| Low Autocorrelation Binary Sequence (LABS)| Find a sequence $S = (s_1, s_2, ..., s_N)$, $s_i\in\{-1,+1\}$, such that the sum of squared off-peak autocorrelations is minimized. |
### Graph instances
| Instance Type| Description |
|----------|----------|
| Unweighted random regular | Unweighted random regular graph $G(n,d)$, where $n$ is the number of nodes and $d$ is the degree of each node. |
| Erdos-Renyi | Erdos-Renyi random graph $G(n,p)$, where $n$ is the number of nodes and $p$ is the probability of an edge between any two nodes. |
| Weighted Line-based | Built from a line graph with $n$ nodes by applying $k$ SWAP layers, and weights drawn uniformly from $\{-1,1\}$. |
| Weighted heavy-hex | Hardware native graphs, and weights drawn from $\mathcal{N}(0,1)$.

##  High-level Repository Structure

```
QAOA-Parameter-Setting/
├── data/                # Experimental results and training data
├── data_test_results/   # Hardware testing results
├── figures/             # Generated figures and analysis notebooks
├── instances/           # Problem instance definitions (graphs)
│   ├── random_regular/  # Random regular graphs
│   ├── erdos_renyi/     # Erdős-Rényi random graphs
│   ├── heavy_hex/       # Heavy-hexagon topology (IBM hardware)
│   └── line_to_full/    # Line-to-full graph instances
├── methods/              # JSON configuration files for parameter-setting strategies
├── qaoa_parameter_setting/  # Diverse Python utilities
├── reference/           # Reference implementations (e.g., SDP solver)
├── scripts/             # Execution scripts for running diverse tasks related to the experiments
├── summary/             # Summary tables and statistics
└── test/                # Various test files
```

## Exploring the data
1. Explore the instances used in this work by navigating to the instances directory ([`instances/`](./instances/)) and check its README to get to know the 4 instance types better.  
2. Go into each of the instance types (e.g. [`instances/random_regular`](./instances/random_regular/)) to check the files defining the structure of the graphs.
3. Explore the methods used in this work by navigating to the methods directory ([`methods/`](./methods/)) and check its README to get to know the different combinations of algorithms and energy evaluation methods.
4. Explore the data produced in our work by navigating to the data directory ([`data/`](./data/)). Start by checking its README to get to know the format of the data files naming.
5. Go into each of the data subfolders that group them by different types (e.g. [`data/training/random_regular`](./data/training/random_regular/)) to check the files produced by the QAOA training pipeline for each of the methods.
6. Explore the [`paper/`](./paper/) folder to see how the plots in the paper were generated.
## Contributors and Acknowledgments
This repository was created as part of the Quantum Optimization Working Group effort initiated in July 2023 by IBM Quantum and its partners.

### Authors
Maosheng Guo, Joel Jurado Diaz, Anurag Ramesh, Conrad J. Haupt, Alberto Baiardi, Dimitrios Athanasakos, M. Emre Sahin, Oscar Wallis, George Pennington, Christian Arenz, Sebastian Brandhofer, Georgios Korpas,Ieva Čepaite, J. A. Montañez-Barrera, Jakub Marecek, Davide Venturelli, Stephan Eidenbenz, David E. Bernal Neira, and Daniel J. Egger

## Citation
If you use this data in your research, please cite:
```
@misc{
  guo2026settingqaoaangles,
  title={{Setting angles in quantum approximate optimization at utility-scale}, 
  author={Guo, Maosheng and Diaz, Joel Jurado and Ramesh, Anurag and Haupt, Conrad J and Baiardi, Alberto and Athanasakos, Dimitrios and Sahin, M Emre and Wallis, Oscar and Pennington, George and Arenz, Christian and others},
  year={2026},
  eprint={2606.05311},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2606.05311}, 
}
```

## License

[Apache License 2.0](./LICENSE.txt)

## Contact
The corresponding author is Daniel J. Egger, deg@zurich.ibm.com

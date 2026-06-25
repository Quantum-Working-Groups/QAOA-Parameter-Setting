# Setting angles in quantum approximate optimization at utility-scale
We investigate the performance of various Quantum Approximate Optimization Algorithm (QAOA) angle-setting strategies across different problem instances, including utility-scale graphs.

[![arXiv](https://img.shields.io/badge/arXiv-2606.05311-B31B1B.svg)](https://arxiv.org/abs/2606.05311)
[![](https://img.shields.io/badge/GitHub-qaoa--training--pipeline-181717?logo=github)](https://github.com/qiskit-community/qaoa_training_pipeline)

## Description
This repository contains the data, experimental results, and code to produce figures, tables, and data management for the paper **Setting angles in quantum approximate optimization at utility-scale**.

##  Table of Contents

- [Overview](#overview)
- [High-level Repository Structure](#high-level-repository-structure)
- [Related](#related)

##  Overview

QAOA is a type of variational quantum algorithm for solving combinatorial optimization problems. A critical challenge is determining the optimal angles for the quantum circuit. The following strategies have been benchmark within this work:

- **Fixed-Angle Conjecture (FA)**: fixed angles strategy extracted from [Wurtz & Love](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042612), [Wurtz & Lykov](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.052419), [fixed-angle-QAOA](https://github.com/danlkv/fixed-angle-QAOA)
- **Trotterized Quantum Annealing (TQA)**: quantum annealing method with a linear ramp based on [Sack et al.](https://quantum-journal.org/papers/q-2021-07-01-491/)
- **Interpolation (INTERP) and Fourier (F)**: Recursive interpolation and optimization in the Fourier basis, based on [Zhou et al.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067)
- **Transition States (TS)**: Recursive greedy initialization via transition states, based on [Sack et al.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062404)
- **Linear ramps method (LR)**: quantum annealing based method with two linear ramps, based on [Montanez-Barrera and Michielsen](https://www.nature.com/articles/s41534-025-01082-1)
- **Parameter Transfer (PT)**: transfer of QAOA angles from small-scale problems to utility-scale problems, based on numerical observations of angles clustering for similar graphs. 

Each method is tested with different quantum state evaluators:
- **SV**: Statevector simulation (exact, valid only for small-scale)
- **MPS**: Matrix Product State simulation (approximate, valid for utility-scale)
- **PP**: Pauli Propagation (approximate, vaid for utility-scale large-scale)

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

##  Related

- [qaoa_training_pipeline](../qaoa_training_pipeline): Core training infrastructure
- Paper: [arXiv:2606.05311](https://arxiv.org/pdf/2606.05311)


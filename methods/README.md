# Angle Setting Methods

This directory contains QAOA angle setting method configurations in JSON format, following the configuration convention given by the [QAOA Training pipeline](https://github.com/qiskit-community/qaoa_training_pipeline). 

## Overview

Each JSON file specifies a complete training pipeline through a `trainer_chain`. The trainers specified are executed sequentally, improving the QAOA angles at each step.

## Naming Convention

The method file naming convention follows the pattern: `{METHOD}_{EVALUATOR}_{FLAG}.json`

- **METHOD**: The QAOA angle-setting algorithm.
- **EVALUATOR**: The method for energy evaluation (SV, MPS, MPSAer, PP).
- **FLAG**: Indicates if the method used further processing (such an additional optimization step). 

There is an example configuration file, [`example_method.json`](example_method.json), which gives a random QAOA angle initialization followed by optimization.

### Evaluator Types

| Evaluator | Full Name | Description |
|-----------|-----------|-------------|
| **SV** | StatevectorEvaluator | Exact statevector simulation |
| **MPS** | MPSEvaluator | Matrix Product State simulation |
| **MPSAer** | MPSAer (Aer-based MPS) | Qiskit Aer MPS backend |
| **PP** | PPEvaluator | Pauli Propagation |

### Flag types

- **opt**: Indicates the usage of an additional QAOA angle optimization step
- **no_opt** / **noOpt**: Uses only the base method without additional optimization
- **angle_opt**: Indicates the usage of an additional QAOA angl optimization step within the context of Linear Ramp (where no_opt indicates the base method with optimization over the ramps)
- **AAA**: Indicates the usage of an Average Angle Aggregator, specific to the parameter transfer method. The angles retrieved during the transfer processed are averaged.

## List of included methods

### 1. Fixed Angle Conjecture (FA) Methods

Based on the Fixed-Angle Conjecture, which gives universally good angles  with a guaranteed lower-bound performance, introduced and studied in  [Wurtz & Love](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042612), [Wurtz & Lykov](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.052419), [fixed-angle-QAOA](https://github.com/danlkv/fixed-angle-QAOA).

**Files:**
- [`FA_SV_opt.json`](FA_SV_opt.json) - Exact evaluation with extra angle optimization
- [`FA_SV_noOpt.json`](FA_SV_noOpt.json) - Exact evaluation without extra angle optimization
- [`FA_MPS_opt.json`](FA_MPS_opt.json) - MPS evaluation with extra angle optimization
- [`FA_MPS_no_opt.json`](FA_MPS_no_opt.json) - MPS evaluation without extra angle optimization
- [`FA_MPSAer_opt.json`](FA_MPSAer_opt.json) - Aer MPS evaluation with extra angle optimization
- [`FA_PP_opt.json`](FA_PP_opt.json) - Pauli Propagation with extra angle optimization
- [`FA_PP_no_opt.json`](FA_PP_no_opt.json) - Pauli Propagation without extra angle optimization

### 2. Trotterized Quantum Annealing (TQA) Methods

The first-order Trotter discretization of quantum annealing introduced in [Sack et al.](https://quantum-journal.org/papers/q-2021-07-01-491/).

**Files:**
- [`TQA_SV_opt.json`](TQA_SV_opt.json) - Exact evaluation with extra angle optimization
- [`TQA_SV_noOpt.json`](TQA_SV_noOpt.json) - Exact evaluation without extra angle optimization
- [`TQA_MPS_opt.json`](TQA_MPS_opt.json) - MPS evaluation with extra angle  optimization
- [`TQA_MPS_no_opt.json`](TQA_MPS_no_opt.json) - MPS evaluation without extra angle optimization
- [`TQA_MPSAer_opt.json`](TQA_MPSAer_opt.json) - Aer MPS evaluation with extra angle optimization
- [`TQA_PP_opt.json`](TQA_PP_opt.json) - Pauli Propagation with extra angle  optimization
- [`TQA_PP_no_opt.json`](TQA_PP_no_opt.json) - Pauli Propagation without extra angle optimization

### 3. Interpolation (I) Methods

The INTERP method introduced in [Zhou et al.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067).

**Files:**
- [`I_SV.json`](I_SV.json) - Exact statevector evaluation
- [`I_MPS.json`](I_MPS.json) - MPS evaluation
- [`I_MPSAer.json`](I_MPSAer.json) - Aer MPS evaluation
- [`I_PP.json`](I_PP.json) - Pauli Propagation evaluation

### 4. Fourier (F) Methods

The FOURIER method, which optimizes the Fourier coefficients of the QAOA angles in a sine/cosine basis, introduced in [Zhou et al.](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067)

**Files:**
- [`F_SV.json`](F_SV.json) - Exact statevector evaluation
- [`F_MPS.json`](F_MPS.json) - MPS evaluation
- [`F_MPSAer.json`](F_MPSAer.json) - Aer MPS evaluation
- [`F_PP.json`](F_PP.json) - Pauli Propagation evaluation


### 5. Linear Ramp (LR) Methods

The Linear Ramp protocol, which is similar to TQA, but it decouples the slopes of $\gamma$ and $\beta$, introduced in [Montanez-Barrera and Michielsen](https://www.nature.com/articles/s41534-025-01082-1)

**Files:**
- [`LR_SV_opt.json`](LR_SV_opt.json) - Exact evaluation with ramps optimization
- [`LR_SV_angle_opt.json`](LR_SV_angle_opt.json) - Exact evaluation with angle optimization
- [`LR_MPS_opt.json`](LR_MPS_opt.json) - MPS with ramps optimization
- [`LR_MPS_angle_opt.json`](LR_MPS_angle_opt.json) - MPS with angle optimization
- [`LR_MPSAer_opt.json`](LR_MPSAer_opt.json) - Aer MPS with ramps optimization
- [`LR_MPSAer_angle_opt.json`](LR_MPSAer_angle_opt.json) - Aer MPS with angle optimization
- [`LR_PP_opt.json`](LR_PP_opt.json) - Pauli Propagation with ramps optimization
- [`LR_PP_angle_opt.json`](LR_PP_angle_opt.json) - Pauli Propagation with angle optimization

### 6. Recursive Transition States (TS/RTS) Methods

The recursive greedy initialization using transition states, introduced in [Sack et al.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062404).

**Files:**
- [`TS_SV.json`](TS_SV.json) - Exact statevector evaluation
- [`RTS_MPS.json`](RTS_MPS.json) - MPS evaluation
- [`RTS_PP.json`](RTS_PP.json) - Pauli Propagation evaluation

### 8. Parameter Transfer (PT) Methods

QAOA angles transfer from small-scale instances to utility-scale instances within a graph family. 

**Files:**
- [`PT_PP_AAA.json`](PT_PP_AAA.json) - Parameter transfer with Average Angle Aggregator and Pauli Propagation

# Parameter setting methods

This folder contains QAOA parameter setting methods in JSON format.
These files are intended as input methods for the QAOA training pipeline.
They define the algorithms with which we find QAOA anagles.

## List of included methods

The following list of methods will be filled out soon.

* example_method.json generates a random set of QAOA angles and use them as initial point to optimize the energy with COBYLA. This method is intended as an example.

### Exact energy evaluation

The following methods are design to optimize small-scale instances where the energy can be compute exactly.
* `fixedangleconjecture_with_statevectorevaluator.json` employes the fixed-angle conjecture of Wurtz and Lykov, Phys. Rev. A **104**, 052419 (2021) to produce QAOA angles.
* `optimized_fixedangleconjecture_with_statevectorevaluator.json` is the same as fixedangleconjecture_with_statevectorevaluator.json but further refines the QAOA angles with SciPy.
* `tqatrainer_with_statevectorevaluator.json` employes the trotterized quantum annealing ansatz of Sack and Serbyn, Quantum **5**, 491 (2021) to produce QAOA angles.
* `optimized_tqatrainer_with_statevectorevaluator.json` is the same as tqatrainer_with_statevectorevaluator.json but further refines the QAOA angles with SciPy.
* `interp_with_statevectorevaluator.json` employes the recursive parameter interpolation of Zhou *et al.*, PRX **10**, 021067 (2020) to find QAOA angles.
  The angles at depth-one are found by an efficient depth-one scan over a 2D grid followed by a finer SciPy optimization.
  These angles then seed the interpolation recursion.

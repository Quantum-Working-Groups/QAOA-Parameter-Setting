# Instances
## 🎯 Overview

The instances in this directory serve as benchmarks for evaluating QAOA parameter-setting methods across different graph topologies and scales. They range from small-scale graphs (10 nodes) suitable for exact statevector simulation to utility-scale graphs (100+ nodes) requiring approximate methods like MPS or Pauli Propagation. For each graph type and specific graph properties, there is 10 different, randomly generated instances.

## 📊 Instance Types

### Random Regular Graphs

Unweighted random k-regular graphs where each node has degree k.

**Naming pattern:** `{idx}_{n}nodes_random{k}regular.json`
- Example: `000_20nodes_random4regular.json` (20-node 4-regular graph)

---

### Erdős-Rényi Graphs

Unweighted graphs where each edge exists independently with probability p. 

**Naming pattern:** `{idx}_{n}nodes_erdosrenyi{p}percent.json`
- Example: `000_20nodes_erdosrenyi20percent.json` (20-node graph, 20% edge probability)

---

### Heavy-Hex Graphs

Graphs matching IBM quantum hardware topology, enabling better execution on quantum processors at utility scale. Weights are normally distributed with N(0,1).

**Naming patterns:**
- Small-scale: `{idx}_{rows}_{cols}_heavyhex_{n}nodes_weighted.json`
  - Example: `000_2_2_heavyhex_35nodes_weighted.json`

---

### Line-to-Full Graphs

Graphs with controlled connectivity ranging from linear chains to fully connected graphs. Random ±1 weights

**Naming pattern:** `{idx}_{n}nodes_{k}swap_layers.json`
- Example: `000_100nodes_2swap_layers.json` (100-node graph, 2 swap layers, seed 0)

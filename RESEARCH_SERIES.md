# Learning-Augmented MIP Research Series

This repository belongs to a broader set of independent projects on machine learning for exact optimization. The projects are intentionally kept as separate repositories because they answer different research questions or operate at different integration levels.

## Solver-internal learning

| Repository | Primary question | Role in the series |
|---|---|---|
| `learning-to-branch-milp` | Can a learned policy imitate expensive branching decisions in a transparent MILP laboratory? | Foundational branching sandbox with classical controls, MLPs, and GNNs |
| `learning-to-branch-mip-gnn-scip-pytorch` | Can a learned branching policy be inserted into a real SCIP branch-and-bound loop? | Solver-integration experiment |
| `learning-to-cut-milp` | Which valid cuts should be selected from a candidate pool? | Learned cut selection |
| `learning-to-presolve-mip` | Which presolve configuration should be used for an instance? | Learned presolve control |
| `learning-to-prune-bnb-node-selection` | Which branch-and-bound nodes should be explored or deprioritized? | Learned tree-search control |
| `neural-diving-mip-solution-prediction` | Can learned variable predictions guide primal diving? | Learned primal heuristic |
| `gnn-guided-generalized-assignment-variable-fixing-pytorch` | Which variables can be fixed safely or profitably before/inside search? | Learned variable fixing |
| `ml-warm-start-constraint-generation` | Can learning predict useful constraints or warm starts for iterative exact optimization? | Learned acceleration of constraint generation |
| `learning-to-price-column-generation-cvrptw` | Can learning guide the pricing stage of column generation? | Learned decomposition subproblem control |
| `reinforcement-learning-benders-decomposition` | Can reinforcement learning guide decisions inside Benders decomposition? | Learned decomposition control |
| `learning-augmented-mip-solver` | What happens when several learned solver components are integrated into one exact pipeline? | Integration sandbox |

## Why these repositories remain separate

The repositories should not be treated as duplicates. A branching policy, a cut selector, a presolve selector, a primal heuristic, a pricing policy, and a Benders controller intervene at different points of the optimization algorithm and require different state representations, labels, safety boundaries, and evaluation metrics.

The two learning-to-branch repositories are also intentionally separate: one is a transparent research laboratory, while the SCIP/PyTorch project studies actual solver callbacks and branch-and-bound consequences.

## Suggested reading order

1. `learning-to-branch-milp`
2. `learning-to-cut-milp`
3. `learning-to-presolve-mip`
4. `neural-diving-mip-solution-prediction`
5. `learning-to-prune-bnb-node-selection`
6. `learning-to-branch-mip-gnn-scip-pytorch`
7. `learning-to-price-column-generation-cvrptw`
8. `reinforcement-learning-benders-decomposition`
9. `learning-augmented-mip-solver`

The ordering is pedagogical rather than a ranking of methods.
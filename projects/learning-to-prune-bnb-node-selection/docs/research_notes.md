# Research notes and design decisions

## Why node selection and pruning are separate components

Branch-and-bound maintains a frontier of open subproblems. A node selector chooses which open node
is processed next; pruning/fathoming decides that a node does not need to be processed further.
Those operations have different correctness consequences.

SCIP's documentation formalizes node selection as a plugin whose core job is to select the next
leaf from the current B&B tree and define an ordering among leaves. This is the conceptual model
used by the exact learned-selection mode in this repository.

## Primary literature

### He, Daumé III, Eisner — NeurIPS 2014

**Learning to Search in Branch and Bound Algorithms**

- NeurIPS page:
  https://papers.neurips.cc/paper_files/paper/2014/hash/533d190f5aa2926b2a8a30c8bea0e05d-Abstract.html
- Paper:
  https://proceedings.neurips.cc/paper_files/paper/2014/file/533d190f5aa2926b2a8a30c8bea0e05d-Paper.pdf
- Historical reference implementation:
  https://github.com/hhexiy/scip-dagger

The paper learns a node-selection ranking policy and a node-pruning classifier by imitation
learning. Its oracle uses known optimal solutions on training instances to identify the optimal path.
The paper explicitly studies a setting aimed at finding good solutions quickly without necessarily
paying the full cost of proving optimality. That distinction motivates the exact/approximate split in
this repository.

### Zhang et al. — ICLR 2025

**Learning to Select Nodes in Branch and Bound with Sufficient Tree Representation**

- ICLR proceedings:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/82f625a28d822d2748b9f5c4f9a89bb9-Abstract-Conference.html

This later work shows that learned node selection remains an active topic. It introduces a tripartite
tree representation and reinforcement-learning/GNN policy. The current repository does not attempt
to reproduce that architecture; it provides a simpler baseline with a transparent exactness contract.

### Huang et al. — B&B survey

**Branch and Bound in Mixed Integer Linear Programming Problems: A Survey of Techniques and
Trends**

https://arxiv.org/abs/2111.06257

The survey treats branching-variable selection, node selection, node pruning, and cutting-plane
selection as distinct core components and reviews learning-based approaches for them.

### Bengio, Lodi, Prouvost — EJOR 2021

**Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon**

https://doi.org/10.1016/j.ejor.2020.07.063

This survey motivates the broader pattern of learning expensive or hand-designed decisions inside
combinatorial-optimization algorithms while retaining the mathematical algorithmic backbone.

## Solver documentation

### SCIP node selectors

https://scipopt.org/doc/html/NODESEL.php

SCIP documents node selectors as the mechanism deciding which leaf of the current branching tree is
processed next. It exposes selection and comparison callbacks for custom implementations.

### PySCIPOpt node-selector tutorial

https://pyscipopt.readthedocs.io/en/latest/tutorials/nodeselector.html

PySCIPOpt provides Python-level support for custom node selectors. A future production-oriented
extension of this repository can use that API after policy design and evaluation are mature enough
to justify solver integration.

## What is deliberately not claimed

- This is not SCIP, Gurobi, CPLEX, or a replacement for their node selectors.
- The linear classifier is not a reproduction of TRGNN or another GNN architecture.
- The offline oracle imitation loop is not a full DAgger reproduction.
- A learned-pruning run does not prove optimality.
- Small synthetic knapsack timing results do not imply a speedup on industrial MILPs.

## Reproducibility policy

For every exact benchmark method, the benchmark harness independently enumerates all binary
solutions for the small held-out instance and checks objective equality. An exactness mismatch is a
hard failure. Approximate pruning is evaluated by gap and is never upgraded to an exact claim based
on empirical success.

# Learning to Search Branch-and-Bound Nodes

A compact research module for **learned node expansion order** in branch-and-bound.

This is deliberately separate from `learning-to-branch-milp`: branching chooses a variable at the current node, whereas node search chooses **which open node to process next**. It is also separate from learned pruning, because the policy only reorders open nodes and does not discard them.

## Research lineage

The project is motivated by He, Daumé III, and Eisner, *Learning to Search in Branch and Bound Algorithms* (NeurIPS 2014), which formulates adaptive B&B node-search policy learning with imitation learning.

## Safety boundary

Changing node order does not alter the feasible region. Exactness is preserved when all required nodes are eventually processed and pruning remains based on valid mathematical bounds.

## Contents

- `NodeState` and six-dimensional node-state features;
- a small PyTorch priority network;
- supervised fitting against an expert priority signal;
- deterministic node ranking;
- root-level CI smoke coverage.

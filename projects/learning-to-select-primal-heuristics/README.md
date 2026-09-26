# Learning to Select Primal Heuristics

A small **primal-heuristic portfolio selection** project for binary packing MILPs.

Instead of predicting a full solution or fixing variables, the learner chooses among complete constructive heuristics. This fills a different solver-control layer from Neural Diving and GNN variable fixing.

## Portfolio

The current transparent baseline portfolio contains:

- profit-first greedy;
- profit-per-average-resource-use greedy;
- profit-per-active-constraint-count greedy.

Every heuristic is independently feasibility checked while it constructs the incumbent.

## Learning task

Cheap instance features are mapped to the heuristic that produced the best feasible incumbent on training instances. The implementation uses a small PyTorch classifier so it remains consistent with the rest of the monorepo.

## Safety boundary

The selector can choose a weak heuristic, but it cannot invalidate the MILP: the chosen heuristic only supplies a feasible incumbent to the exact search.

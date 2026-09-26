# Literature map

This monorepo is literature-informed; it does not claim to reproduce a paper unless a project explicitly says so.

## Branching

- Gasse et al., *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*, NeurIPS 2019. MILPs are represented as variable-constraint bipartite graphs and branching policies imitate strong branching.
- `learning-to-branch-milp` is the transparent sandbox; `learning-to-branch-mip-gnn-scip-pytorch` is the solver-integration specialization.

## Cut selection

- Paulus et al., *Learning to Cut by Looking Ahead: Cutting Plane Selection via Imitation Learning*, ICML 2022. NeuralCut imitates an expensive lookahead expert that measures bound improvement.

## Presolve

- Liu et al., *L2P-MIP: Learning to Presolve for Mixed Integer Programming*, ICLR 2024. The central idea is instance-specific presolve configuration rather than one static configuration for all MIPs.

## Neural primal guidance

- Nair et al., *Solving Mixed Integer Programs Using Neural Networks* (2020), develops Neural Diving and Neural Branching around a base MIP solver.

## Node search

- He, Daumé III, and Eisner, *Learning to Search in Branch and Bound Algorithms*, NeurIPS 2014, studies imitation learning for adaptive node-search order in B&B.

## CP-SAT hints

- Google OR-Tools documents solution hints as partial assignments that guide CP-SAT search rather than hard constraints. This motivates keeping learned hints advisory while CP-SAT remains authoritative.

## Repository rule

For every learned component ask:

1. What solver decision is learned?
2. What classical or expensive expert supplies the reference?
3. What guarantee remains after inserting learning?
4. What exact validation/fallback prevents silent infeasibility?

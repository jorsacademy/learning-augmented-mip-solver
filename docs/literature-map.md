# Literature map

This monorepo is **literature-informed**. It does not claim to reproduce a paper unless a project explicitly says so.

## Branching

- Gasse, Chételat, Ferroni, Charlin, Lodi — *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*, NeurIPS 2019. Official paper page: https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
- The core idea is the variable-constraint bipartite representation of MILPs and imitation of strong branching.
- `learning-to-branch-milp` is the transparent sandbox; `learning-to-branch-mip-gnn-scip-pytorch` is the solver-integration specialization.

## Cut selection

- Paulus et al. — *Learning to Cut by Looking Ahead: Cutting Plane Selection via Imitation Learning*, ICML 2022: https://proceedings.mlr.press/v162/paulus22a
- The learned selector imitates an expensive lookahead expert based on bound improvement.

## Presolve

- Liu et al. — *L2P-MIP: Learning to Presolve for Mixed Integer Programming*, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/1e6e0c2edb159b2ad2f9419b898f56d3-Abstract-Conference.html
- The key research question is instance-specific presolve control rather than one fixed configuration for all MIPs.

## Neural primal guidance

- Nair et al. — *Solving Mixed Integer Programs Using Neural Networks* (2020): https://arxiv.org/abs/2012.13349
- The paper develops Neural Diving and Neural Branching around a base MIP solver.

## Node search

- He, Daumé III, Eisner — *Learning to Search in Branch and Bound Algorithms*, NeurIPS 2014: https://proceedings.neurips.cc/paper_files/paper/2014/hash/533d190f5aa2926b2a8a30c8bea0e05d-Abstract.html
- This motivates `learning-to-search-bnb-nodes`, which learns node expansion order rather than branching-variable selection.

## Solver configuration and portfolio selection

- Xu, Hutter, Hoos, Leyton-Brown — *Hydra-MIP: Automated Algorithm Configuration and Selection for Mixed Integer Programming*: https://www.cs.ubc.ca/tr/2011/tr-2011-01
- This motivates per-instance solver/configuration selection and the new primal-heuristic portfolio perspective.

## Solver-in-the-loop environments

- Ecole provides extensible learning environments for combinatorial optimization and SCIP-based control problems, including branching with NodeBipartite observations: https://github.com/ds4dm/ecole
- It is a useful integration reference for experiments that should move beyond the transparent in-repo solver.

## CP-SAT hints

- Google OR-Tools documents solution hints as partial assignments that guide CP-SAT search rather than hard constraints: https://github.com/google/or-tools/blob/stable/ortools/sat/docs/model.md
- This motivates keeping learned hints advisory while CP-SAT remains authoritative.

## Repository rule

For every learned component ask:

1. **What solver decision is learned?**
2. **What classical or expensive expert supplies the reference?**
3. **What guarantee remains after inserting learning?**
4. **What exact validation or fallback prevents silent infeasibility?**

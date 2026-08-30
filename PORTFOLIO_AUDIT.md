# AI × OR Portfolio Audit

This audit is intentionally scoped to the modern **machine-learning + operations-research / solver-learning** part of the Jors Academy portfolio. It is not an inventory of every classical OR, simulation, reinforcement-learning, or metaheuristic repository.

## Executive finding

The portfolio already covers the major modern AI × OR lines discussed during the research build-out. The main issue is no longer missing topics; it is **overlap and canonicalization**.

The most useful next-level project is therefore an integration repository rather than another isolated method demo. `learning-augmented-mip-solver` fills that role.

## Solver-internal learning: canonical map

| Solver decision | Canonical repository | Status | Audit note |
|---|---|---|---|
| Branching | `learning-to-branch-milp` | canonical | Full research sandbox with classical branching baselines, learned MLP/GNN policies, repeated-seed evaluation and OOD scenarios. |
| Cut selection | `learning-to-cut-milp` | canonical | Learned ranking of valid cover cuts using a one-step LP-bound-improvement expert. |
| Presolve | `learning-to-presolve-mip` | canonical | Instance-specific selection among safe presolve configurations. |
| Primal guidance / variable fixing | `gnn-guided-generalized-assignment-variable-fixing-pytorch` | canonical specialization | GNN confidence fixing + exact residual MILP repair. This already occupies much of the learned-primal-guidance / neural-diving neighborhood. |
| Integrated solver | `learning-augmented-mip-solver` | canonical integration | Combines safe presolve, learned cut ranking, learned branching and feasible primal guidance in one exact-search pipeline. |

### Branching overlap

There are two substantial branching repositories:

- `learning-to-branch-milp`
- `learning-to-branch-mip-gnn-scip-pytorch`

Recommendation: treat `learning-to-branch-milp` as the canonical transparent research sandbox and the SCIP/PyTorch repository as the solver-integration specialization. Do not create a third standalone branching repository.

## Predict → optimize / differentiable decision learning

| Area | Repositories | Audit decision |
|---|---|---|
| Decision-focused learning / SPO | `decision-focused-learning-spo` | canonical; no additional SPO repo needed |
| Differentiable optimization | `differentiable-optimization-pytorch`, `differentiable-optimization-portfolio`, `differentiable-black-box-supplier-selection-pytorch` | already heavily covered; keep as general tutorial + domain specializations |
| Contextual optimization | `contextual-optimization-newsvendor` | canonical focused example |
| Inverse optimization | `inverse-optimization-shortest-path` | canonical focused example |

### Differentiable-optimization overlap

This is the clearest duplication cluster. The repositories are not identical:

- `differentiable-optimization-pytorch` is the general/tutorial-oriented differentiable-QP project;
- `differentiable-optimization-portfolio` is a `cvxpylayers` decision application;
- `differentiable-black-box-supplier-selection-pytorch` is a discrete/black-box specialization.

Recommendation: keep all three, but present the first as the canonical concept repo and the others as application specializations. Do not add another generic differentiable-optimization repo.

## Neural combinatorial optimization / learned heuristics

| Area | Repositories | Audit decision |
|---|---|---|
| NCO | `neural-combinatorial-optimization-tsp`, `neural-combinatorial-optimization-tsp-attention-model-pytorch` | overlap; use the newer research-sandbox repo as canonical and the attention-model repo as architecture specialization |
| Diffusion for CO | `diffusion-neural-combinatorial-optimization-tsp-pytorch` | distinct; keep |
| Neural LNS | `neural-large-neighborhood-search-cvrp`, `neural-large-neighborhood-search-job-shop-scheduling-pytorch` | not harmful duplication; same methodology on two OR domains |
| Generic graph-neural CO | `graph-neural-solver-combinatorial-optimization` | adjacent umbrella project; avoid using it as evidence that a specific solver component is covered |

## Adjacent AI × OR repositories

The portfolio also contains relevant but conceptually different projects, including:

- `constraint-learning-for-industrial-engineering` / `constraint_learning_feasible_regions`;
- `learning-augmented-online-machine-scheduling-python`;
- `reinforcement-learning-job-shop-scheduling-pytorch`;
- several RL-based industrial control and scheduling repositories.

These should remain outside the core MIP-solver-learning taxonomy because they learn constraints, policies, or online decisions rather than solver-internal MIP decisions.

## Canonical portfolio taxonomy

### A. Learning before optimization

- contextual optimization
- inverse optimization
- constraint learning
- prediction / decision-focused learning

### B. Differentiating through optimization

- differentiable optimization
- differentiable black-box optimization

### C. Learning constructive / generative solvers

- neural combinatorial optimization
- graph-neural combinatorial solvers
- diffusion for combinatorial optimization

### D. Learning metaheuristics

- neural large neighborhood search

### E. Learning inside exact MIP solvers

- learning to presolve
- learning to cut
- learning to branch
- learned primal guidance / variable fixing
- integrated learning-augmented MIP solver

## Portfolio maintenance recommendations

1. **Stop adding near-duplicate topic repos.** The major modern AI × OR families are already represented.
2. **Use canonical vs specialization labels in README files.** This is more accurate than deleting useful variants.
3. **Keep exactness claims local.** A learned heuristic may accelerate or guide an exact solver without itself being exact.
4. **Prefer matched-instance baselines.** Learned policies should be compared with cheap classical rules, expensive experts/oracles, and ablations on the same instances.
5. **Separate smoke reproducibility from scientific performance claims.** Small CI experiments establish correctness, not solver superiority.
6. **Use `learning-augmented-mip-solver` as the integration endpoint.** Future MIP-learning work should usually extend or benchmark this repo rather than opening another isolated repository.

## Audit conclusion

For the modern AI × OR scope, the portfolio is now **coverage-complete enough that integration, evaluation quality, and canonicalization matter more than adding topics**. The largest historical gap was solver-internal integration; this repository is intended to close that gap while preserving clear methodological boundaries.

# Learning-Augmented MIP Solver

A transparent research sandbox for **integrating multiple learned decisions inside one exact mixed-integer programming pipeline**.

The project does **not** claim to reproduce one monolithic published solver. Instead, it combines four component-level research directions that are individually established in the literature:

1. **learning to presolve** — instance-specific presolve configuration;
2. **learning to cut** — ranking valid cutting planes;
3. **learning to branch** — imitating strong-branching variable selection;
4. **learned primal guidance** — generating high-quality feasible incumbents without changing the feasible region.

The mathematical backbone is an exact depth-first branch-and-cut solver for binary packing MILPs. Learned components choose *how* the solver operates; they are not allowed to invalidate the model. Safe presolve transformations preserve equivalence, cover cuts are globally valid, learned primal solutions are only incumbents, and learned branching only changes search order. Therefore, when the search terminates without a node limit, optimality does not depend on prediction correctness.

## Why this repository exists

Modern exact solvers contain many expensive or hand-designed decisions. Machine learning is a natural candidate for amortizing those decisions across related instances. This general viewpoint is surveyed by Bengio, Lodi, and Prouvost (EJOR 2021).

This repository is grounded specifically in:

- Gasse et al., **Exact Combinatorial Optimization with Graph Convolutional Neural Networks**, NeurIPS 2019 — imitation of strong branching;
- Paulus et al., **Learning to Cut by Looking Ahead**, ICML 2022 — imitation of a lookahead LP-bound-improvement cut expert;
- Tang, Agrawal, and Faenza, **Reinforcement Learning for Integer Programming: Learning to Cut**, ICML 2020 — data-driven cut selection inside cutting-plane / branch-and-cut methods;
- Liu et al., **L2P-MIP: Learning to Presolve for Mixed Integer Programming**, ICLR 2024 — instance-specific presolve selection;
- Nair et al., **Solving Mixed Integer Programs Using Neural Networks**, 2020 — Neural Diving / Neural Branching as learned components augmenting a base MIP solver.

The implementation here is intentionally smaller and more transparent than those systems. It is a research integration sandbox, not SCIP/Gurobi/CPLEX replacement code.

## Integrated pipeline

```text
binary packing MILP
      |
      v
instance features
      |
      +--> learned presolve selector ----> safe presolve transformations
      |
      v
root LP relaxation
      |
      +--> valid cover-cut pool
      |       |
      |       +--> learned cut scorer ----> selected global cuts
      |
      +--> learned primal scorer --------> feasible incumbent
      |
      v
branch-and-cut search
      |
      +--> learned branching scorer -----> fractional branching variable
      |
      v
optimal solution / certified search result
```

## Problem class

The sandbox solves binary packing MILPs

\[
\max \; c^T x
\]

subject to

\[
Ax \le b, \qquad x \in \{0,1\}^n,
\]

with nonnegative constraint coefficients. This class is deliberately chosen because it allows transparent safe presolve rules and valid cover inequalities while still exposing the main solver decisions.

## Policies

Each learned decision has a classical control:

| Stage | Classical/control | Learned/oracle |
|---|---|---|
| Presolve | `none`, `full` | learned config, oracle config |
| Cuts | efficacy | learned score, lookahead oracle |
| Primal | profit-density greedy | learned-score greedy |
| Branching | most fractional | learned score, strong branching |

This makes ablations explicit: the augmented solver can be compared against a classical pipeline while changing one component at a time.

## Exactness boundary

The solver is **exact only when `optimal=True`**. A node budget can stop the search early.

Learned predictions cannot silently redefine feasibility:

- presolve uses only mathematically safe transformations;
- generated cover inequalities are valid for the original packing rows;
- primal guidance only proposes a feasible incumbent;
- branching changes the search order, not the search space.

The learned cut scorer ranks already-valid cuts. It never invents unconstrained neural inequalities.

## Repository layout

```text
src/lamip/
  problem.py       problem generation and exact reference solve
  presolve.py      safe presolve configurations
  cuts.py          valid cover cuts and cut features
  models.py        PyTorch learned policies
  training.py      oracle-data collection and supervised training
  solver.py        integrated exact branch-and-cut pipeline
scripts/
  train.py         train all four learned components
  benchmark.py     matched baseline / augmented benchmark
PORTFOLIO_AUDIT.md
```

## Run

```bash
pip install -e ".[dev]"
pytest -q
python scripts/train.py --checkpoint checkpoints/lamip.pt
python scripts/benchmark.py --checkpoint checkpoints/lamip.pt
```

## Research interpretation

A strong result would not be "the neural network solves MILPs." The relevant questions are narrower:

- can an instance-specific presolve policy reduce residual work?
- can a learned cut scorer approximate expensive lookahead selection?
- can learned branching reduce processed nodes relative to cheap branching rules?
- can learned primal guidance find stronger incumbents early?
- do these components remain useful when combined rather than evaluated in isolation?

Those are the same kinds of solver-internal decisions studied by the cited learning-for-MIP literature.

## Limitations

- synthetic binary packing instances, not MIPLIB-scale production models;
- small MLPs rather than the exact GNN architectures in Gasse et al. or NeuralCut;
- root-level learned cut selection rather than a production separator stack;
- simple safe presolve rules rather than SCIP's full presolver ecosystem;
- learned primal guidance is a feasible greedy incumbent generator, not a reproduction of Neural Diving;
- benchmark timings on small synthetic models are not evidence of production speedup.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.

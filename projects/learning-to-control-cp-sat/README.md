# Learning to Control CP / CP-SAT Search

A compact, reproducible research benchmark for **learning search-control decisions in constraint programming**. The project studies fixed-`k` graph coloring as a finite-domain CSP and deliberately separates two settings:

1. a transparent exact CP search where variable ordering is controllable at every search node; and
2. OR-Tools CP-SAT experiments restricted to search controls that the public Python API actually exposes.

This is an independent research/education implementation. It is **not** a reproduction of a published codebase and it does not claim paper-level benchmark performance.

## Motivation

Constraint-programming search can change dramatically under different variable/value orderings even when the feasible set is unchanged. Classical CP therefore relies on first-fail / minimum-domain rules, degree and weighted-degree heuristics, impact/activity signals, restarts, and heuristic portfolios.

The learning question is narrower: can a model use the current propagated state to rank branching variables so that subsequent exact search becomes cheaper?

This is related to learning-to-branch in MIP, but the solver state is different:

- **MIP:** LP relaxation + fractional branching candidates + pseudocost/strong-branching information.
- **CP:** finite domains + propagation + failures + variable/value ordering + backtracking.

The repository does not invent an LP-style callback inside CP-SAT where the Python API does not provide one.

## Research Question

> Can supervised imitation of an expensive propagation-based lookahead heuristic reduce finite-domain CP search cost on held-out graph-coloring CSPs, relative to interpretable classical variable-ordering baselines, without changing exactness?

The benchmark separately asks:

- Does expert-imitation accuracy translate into fewer nodes/backtracks?
- How does the learned dynamic policy compare with MRV/first-fail, domain/degree, degree, fixed, and random ordering?
- Does behavior persist under larger/denser OOD graphs?
- Is any search reduction worth the learned-policy inference cost?
- What happens when learned scores are reduced to a root-level static ordering for CP-SAT?

Prediction accuracy and decision quality are therefore different reported quantities.

## Mathematical Problem

For an undirected graph `G=(V,E)` and fixed color count `k`:

\[
x_v \in \{0,1,\ldots,k-1\} \qquad \forall v\in V,
\]

\[
x_u \neq x_v \qquad \forall (u,v)\in E.
\]

The first version is a **feasibility CSP**, not an optimization model. The principal research outcome is search effort rather than coloring quality.

Synthetic instances are explicit:

- **planted feasible:** edges are sampled only between different planted color classes;
- **certified infeasible:** a planted graph is augmented with a `(k+1)`-clique, which certifies non-`k`-colorability.

Small instances are additionally checked with independent exhaustive enumeration.

## Methodology

### Transparent finite-domain CP search

`src/learning_cp_sat/search.py` implements exact depth-first search with singleton propagation for binary `!=` constraints. Assigning `x_v=c` removes `c` from neighboring domains; newly singleton domains propagate until fixpoint or contradiction.

The learned component changes **only variable ordering**. Value ordering is fixed to ascending color so that the experiment isolates the branching-variable decision.

With no node/time limit, the learned rule changes search order but does not prune admissible values. Exhaustive failure therefore proves infeasibility. If a configured budget is hit first, status is `UNKNOWN`, never `INFEASIBLE`.

### Propagation-lookahead expert

At each sampled search state, every value of every candidate variable is tentatively assigned and propagated. The expert score rewards:

- immediate contradictions; and
- domain reduction on surviving probes.

This is a CP-specific strong-lookahead analogue: expensive supervision, not a globally optimal search-tree oracle.

### Learned policy

Candidate features include normalized domain size, static degree, remaining dynamic degree, singleton-neighbor fraction, saturation, mean/minimum neighboring domain size, and normalized depth.

Two supervised predictors are fitted on the training split:

- standardized ridge regression;
- random-forest regression.

Validation top-1 imitation accuracy selects the fitted model. Test and OOD search results are not used for model selection.

### CP-SAT external-control layer

OR-Tools exposes declarative `add_decision_strategy(...)` rules and `FIXED_SEARCH`. Its public Python surface does **not** expose an arbitrary per-node user variable-selection callback for internal CP-SAT branching.

Accordingly the CP-SAT benchmark uses only real public controls:

- `default`: CP-SAT's configured search;
- `fixed`: fixed vertex order + `CHOOSE_FIRST`;
- `degree`: static graph-degree order + `CHOOSE_FIRST`;
- `min_domain`: public `CHOOSE_MIN_DOMAIN_SIZE`, a first-fail-style dynamic selector;
- `learned_root`: learned scores at the root converted into a static permutation + `CHOOSE_FIRST`.

The learned CP-SAT experiment is therefore **external/root-static control**, not a claim of dynamic learned branching inside CP-SAT internals. The `min_domain` baseline is important because the public API can still provide a genuine dynamic classical selector even though it cannot call an arbitrary learned policy at every node.

## Baselines

Transparent search baselines:

- `fixed`
- `degree`
- `mrv` (minimum remaining values / first-fail, degree tie-break)
- `dom_degree`
- seeded `random`
- expensive propagation-lookahead `expert`
- `learned`

CP-SAT baselines:

- `default`
- `fixed`
- `degree`
- `min_domain`
- `learned_root`

The repository does not assume the learned method wins. Negative results are valid outputs.

## Exactness / Verification

For transparent search:

- unlimited DFS is complete;
- learned control only reorders branching variables;
- `FEASIBLE` assignments are independently audited;
- `INFEASIBLE` means exhaustive search completed;
- budget exhaustion is `UNKNOWN`.

`src/learning_cp_sat/oracle.py` provides an independent exhaustive Cartesian-enumeration oracle for very small instances. It deliberately does not reuse the project's propagation/search implementation.

For CP-SAT, statuses are reported as returned by OR-Tools. Returned assignments are independently audited by this repository; CP-SAT's internal proof machinery is not reimplemented here.

## Feasibility Audit

Every returned coloring is checked after search for:

- exactly one decision slot per vertex;
- no missing decision;
- color-domain membership;
- every edge satisfying `x_u != x_v`.

An assignment that fails the audit is not counted as a successful solve.

## Evaluation Protocol

`configs/benchmark.json` uses disjoint deterministic seed namespaces for train, validation, in-distribution test, OOD test, policy randomness, CP-SAT, statistics, and exhaustive-oracle checks.

Main transparent-search metrics:

- solved rate under limits;
- nodes;
- backtracks;
- propagation events;
- wall-clock time;
- learned candidate evaluations;
- expert lookahead probes.

CP-SAT metrics:

- solved rate;
- branches;
- conflicts;
- wall-clock time.

Aggregates include mean, sample standard deviation, median, and p90. Paired learned-minus-MRV node differences are reported with a bootstrap 95% confidence interval on identical held-out instances.

`configs/smoke.json` exists only for CI integration. **CI smoke output is not a scientific benchmark result.**

## Computational Cost

Search nodes alone can hide expensive control logic. The benchmark therefore records learned candidate evaluations and expert lookahead probes separately from nodes, backtracks, propagations, and time. A policy that reduces nodes but greatly increases total compute is not described as unconditionally better.

## Reproducibility

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Run the CI-scale smoke experiment:

```bash
python scripts/run_benchmark.py \
  --config configs/smoke.json \
  --output artifacts/smoke.json
```

Run the larger research configuration:

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark.json \
  --output artifacts/benchmark.json
```

The JSON result retains raw per-instance records as well as aggregate statistics.

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── benchmark.json
│   └── smoke.json
├── docs/literature-and-protocol.md
├── scripts/run_benchmark.py
├── src/learning_cp_sat/
│   ├── audit.py
│   ├── cpsat_control.py
│   ├── experiment.py
│   ├── expert.py
│   ├── features.py
│   ├── instances.py
│   ├── learning.py
│   ├── oracle.py
│   ├── propagation.py
│   ├── search.py
│   └── statistics.py
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

## CI

GitHub Actions runs Python 3.11 and 3.12 and checks:

- editable install;
- `pip check`;
- Ruff lint;
- Ruff format check;
- mathematical/methodological pytest suite;
- end-to-end smoke training/search/CP-SAT/oracle run;
- smoke output schema.

Tests cover deterministic generation, planted infeasibility certificates, exhaustive-oracle agreement, feasibility auditing, exact status semantics, seed determinism, learned-policy exactness, imitation metric bounds, CP-SAT public-strategy integration (including `CHOOSE_MIN_DOMAIN_SIZE`), statistics, and end-to-end execution.

## Experimental Interpretation

Possible outcomes include:

- better imitation accuracy but worse search;
- fewer nodes but worse wall-clock time;
- gains in the transparent dynamic policy but no gain in CP-SAT's root-static learned ordering;
- degradation under OOD graphs;
- classical MRV/domain-based rules remaining stronger than the learned policy.

None of these outcomes is hidden or redefined as success.

## Limitations

- Synthetic graph coloring is not an industrial scheduling dataset.
- The transparent propagation engine handles binary inequality constraints with singleton propagation; it is not a production CP kernel with global constraints, nogoods, explanations, or sophisticated restart machinery.
- The lookahead expert is one-step, not globally optimal.
- Features are compact and hand-designed rather than a generic GNN state encoder.
- The first version learns variable ordering only; value ordering and restart policy learning are out of scope.
- Dynamic learned CP-SAT branching cannot be implemented through a nonexistent public Python callback; `learned_root` is intentionally static after root scoring.
- Wall-clock measurements depend on hardware and software versions.

## Claims Boundary

This repository does **not** claim:

- state-of-the-art performance;
- superiority of the learned policy over classical heuristics;
- that imitation accuracy equals decision/search quality;
- that the lookahead expert is globally optimal;
- that CP-SAT exposes a custom per-node Python branching callback;
- that CI smoke output is a scientific benchmark;
- that synthetic graph-coloring results transfer directly to industrial scheduling/planning;
- production readiness or industrial savings.

The word **exact** applies to unlimited transparent exhaustive search and certified solver statuses, not to heuristic ordering quality.

## Research Context / Related Repositories

This repository is standalone but complements several projects in the same portfolio:

- [`learning-to-branch-mip-gnn-scip-pytorch`](https://github.com/jorsacademy/learning-to-branch-mip-gnn-scip-pytorch): MIP learning-to-branch through SCIP's real branching interface and LP-derived state; this repository studies the CP counterpart using domains and propagation.
- [`optimal-conference-meeting-scheduling-cp-sat`](https://github.com/jorsacademy/optimal-conference-meeting-scheduling-cp-sat): CP-SAT used to solve a scheduling model; this repository instead studies solver search control.
- [`constraint-learning-for-industrial-engineering`](https://github.com/jorsacademy/constraint-learning-for-industrial-engineering): broader constraint-learning context.
- [`sequential-decision-analytics`](https://github.com/jorsacademy/sequential-decision-analytics): related sequential-decision methodology.
- [`neural-combinatorial-optimization-tsp-attention-model-pytorch`](https://github.com/jorsacademy/neural-combinatorial-optimization-tsp-attention-model-pytorch): direct neural construction, contrasted with learning inside exact search.

No code dependency is introduced between these repositories.

## References

Classical CP search:

- Haralick, R. M., Elliott, G. L. (1980). *Increasing Tree Search Efficiency for Constraint Satisfaction Problems*. Artificial Intelligence 14(3), 263–313. https://doi.org/10.1016/0004-3702(80)90051-X
- Boussemart, F., Hemery, F., Lecoutre, C., Sais, L. (2004). *Boosting Systematic Search by Weighting Constraints*. ECAI 2004.
- Refalo, P. (2004). *Impact-Based Search Strategies for Constraint Programming*. CP 2004. https://doi.org/10.1007/978-3-540-30201-8_41

Learning CP search:

- Song, W., Cao, Z., Zhang, J., Xu, C., Lim, A. (2022). *Learning Variable Ordering Heuristics for Solving Constraint Satisfaction Problems*. Engineering Applications of Artificial Intelligence 109, 104603. https://doi.org/10.1016/j.engappai.2021.104603
- Doolaard, F., Yorke-Smith, N. (2022). *Online Learning of Variable Ordering Heuristics for Constraint Optimisation Problems*. Annals of Mathematics and Artificial Intelligence. https://doi.org/10.1007/s10472-022-09816-z
- Marty, T. et al. (2024). *Learning and Fine-Tuning a Generic Value-Selection Heuristic inside a Constraint Programming Solver*. Constraints 29, 234–260. https://doi.org/10.1007/s10601-024-09377-4
- Xu, J., Wu, Y., Li, H., Yin, M. (2025). *Prediction-Based Adaptive Variable Ordering Heuristics for Constraint Satisfaction Problems*. AAAI 2025. https://doi.org/10.1609/aaai.v39i11.33239
- Cappart, Q., Guns, T., Lombardi, M., Pesant, G., Tsouros, D. (2025). *Combining Constraint Programming and Machine Learning: From Current Progress to Future Opportunities*. JAIR 84. https://doi.org/10.1613/jair.1.19533

Relation to MIP learning-to-branch:

- Gasse, M., Chételat, D., Ferroni, N., Charlin, L., Lodi, A. (2019). *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*. NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html

Official OR-Tools sources used to define the CP-SAT capability boundary:

- CP-SAT Python API: https://github.com/google/or-tools/blob/stable/ortools/sat/python/cp_model.py
- Decision-strategy proto: https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.proto
- Search/branching parameters: https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto
- CP-SAT documentation/examples: https://github.com/google/or-tools/tree/stable/ortools/sat/docs

As of the 2026 OR-Tools 9.15 line, the benchmark stays within these public strategy/configuration interfaces.

## License

MIT. See `LICENSE`.

# Learning to Configure Optimization Solvers

A small research benchmark for **instance-specific solver configuration** in mixed-integer optimization.

The pipeline is deliberately instance-level:

```text
MILP instance
    ↓
cheap structural instance features
    ↓
learned selector
    ↓
HiGHS solver configuration
    ↓
solve the original MILP
    ↓
independent feasibility + objective audit
```

The project does not learn a branching decision inside branch-and-bound. It learns which **solver strategy/configuration** to use before solving a new instance.

## Motivation

Modern MIP solvers expose many parameters that affect presolve, primal heuristics, symmetry processing, branching/search behavior, tolerances, and resource limits. A single default configuration is intentionally general-purpose, while a single globally tuned configuration can overfit a distribution. The research question here is whether inexpensive instance features can support a better **per-instance configuration choice** from a small, documented configuration portfolio.

This is a benchmark about solver control, not a claim that machine learning should replace mathematical optimization. HiGHS remains the optimizer. The learned component only chooses a configuration.

## Research Question

> Given a heterogeneous distribution of set-covering MILPs and a fixed portfolio of valid HiGHS parameter configurations, can a model trained only on training/validation instances select configurations for unseen instances that reduce penalized runtime regret relative to the per-instance portfolio oracle, while preserving feasibility and solver exactness whenever HiGHS returns an optimal certificate?

The benchmark separates five questions:

1. **OR decision problem:** minimum-cost set covering.
2. **What ML learns:** a mapping from static instance features to the expected performance of each candidate solver configuration.
3. **What part of the mathematical problem changes:** none. The MILP formulation, feasible set, objective coefficients, time budget, threads, and solver seed are fixed; only documented HiGHS strategy parameters vary.
4. **Classical baselines:** HiGHS default configuration, globally best single configuration on train+validation, and random configuration selection.
5. **Decision metric:** penalized solve performance and selector regret, not configuration-label classification accuracy.

Prediction quality and decision quality are therefore not treated as equivalent. The model is trained as a **cost predictor** for each configuration and is evaluated by downstream solver cost.

## Mathematical Problem

For a set of rows `i = 1,...,m` and covering columns `j = 1,...,n`, let `A_ij ∈ {0,1}` indicate whether column `j` covers row `i`, `c_j > 0` be its cost, and `x_j ∈ {0,1}` be the selection decision.

\[
\begin{aligned}
\min_x \quad & \sum_{j=1}^{n} c_j x_j \\
\text{s.t.} \quad & \sum_{j=1}^{n} A_{ij}x_j \ge 1 && \forall i, \\
& x_j \in \{0,1\} && \forall j.
\end{aligned}
\]

Synthetic instances are generated independently from four structural regimes:

- `uniform`: uniform positive costs;
- `skewed_cost`: log-normal-like cost heterogeneity;
- `degree_correlated`: cost correlated with column coverage degree;
- `near_duplicate`: controlled column redundancy with small perturbations.

Every row is explicitly guaranteed to be coverable. Train, validation, test, and OOD splits use disjoint seed ranges. No instance is created as a mutation of a shared base instance across splits.

## Methodology

### 1. Candidate HiGHS configurations

The initial portfolio intentionally stays small. It uses only options exposed by the official HiGHS API:

| Configuration | Changed options |
|---|---|
| `default` | none |
| `presolve_off` | `presolve="off"` |
| `heuristics_off` | `mip_heuristic_effort=0.0` |
| `heuristics_high` | `mip_heuristic_effort=0.20` |
| `symmetry_off` | `mip_detect_symmetry=false` |
| `lean_search` | `presolve="on"`, heuristics off, symmetry off |

The common experiment budget is **not** part of the learned configuration. Every candidate receives the same `time_limit`, `threads=1`, `parallel="off"`, and solver random seed. This avoids rewarding a configuration merely for terminating early under a smaller budget.

HiGHS documents `presolve`, `mip_heuristic_effort`, and `mip_detect_symmetry` as solver options; the benchmark fails if HiGHS rejects an option rather than silently inventing or ignoring parameters.

### 2. Static instance features

The selector uses cheap, solver-independent features only:

- number of rows and columns;
- constraint-matrix density;
- objective mean, standard deviation, and coefficient of variation;
- row-degree and column-degree summary statistics;
- singleton-row fraction;
- duplicate-column fraction.

No LP relaxation, probing run, root solve, or hidden solver statistic is used as an input feature in v0.1. Feature extraction time is measured separately.

### 3. Learned selector

For every candidate configuration, a random-forest regressor predicts its penalized runtime score from the instance feature vector. At inference time, the selector chooses

\[
\hat{k}(x)=\arg\min_{k \in K} \widehat{C}_k(x).
\]

This is deliberately a **decision-cost regression** design rather than ordinary multiclass prediction of the oracle label. A selector can misclassify the identity of the best configuration but still incur negligible regret if the selected configuration is nearly tied with the oracle; conversely, a rare but expensive misclassification can dominate decision quality.

Hyperparameters are selected on validation regret only. The final model is refit on train+validation and evaluated once on test and OOD. Multiple selector random seeds are reported.

## Algorithm Selection vs. Configuration

Classical **algorithm selection** chooses among distinct algorithms/solvers from a portfolio. SATzilla is a canonical example: instance features drive selection among solver components.

**Algorithm configuration** chooses parameter settings of a parameterized algorithm. ParamILS and SMAC are canonical offline configuration methods that search for configurations performing well over a training distribution.

**Instance-specific algorithm configuration (ISAC)** combines these ideas: features of the incoming instance determine the parameter configuration used for that instance.

This repository is closest to **instance-specific configuration**. The candidate actions are different parameterizations of the same HiGHS MIP solver. In the broader algorithm-selection view, each fixed configuration can also be treated as a portfolio member.

## Difference from Learning to Branch

`learning-to-branch-mip-gnn-scip-pytorch` learns a **within-solve branching action** repeatedly at branch-and-bound states: the model sees a current SCIP LP state and chooses a branching variable.

This repository operates one level above that control loop:

```text
Learning to Branch:
instance → solver state → branching candidate → branch → next solver state → ...

This repository:
instance → static features → one solver configuration → complete solve
```

The learned selector does not replace pseudocosts, strong branching, or an internal branch rule. It chooses an instance-level strategy before the run begins.

## Baselines

The required comparison set is built into the benchmark:

- **HiGHS default:** no portfolio parameter overrides;
- **globally best single configuration:** minimum mean train+validation PAR10, then frozen before test;
- **random configuration:** repeated random portfolio selection with fixed seeds;
- **per-instance portfolio oracle:** best observed candidate configuration for each test instance;
- **learned selector:** random-forest cost model chosen by validation regret.

The per-instance portfolio oracle is an **upper-bound reference for the selector over this finite portfolio**. It is not a mathematical optimization oracle and is not available at deployment time because computing it requires running every candidate configuration.

## Evaluation Protocol

The experiment uses four disjoint splits:

- `train`: fit configuration-cost models;
- `validation`: choose selector hyperparameters;
- `test`: final in-distribution evaluation;
- `ood`: larger/sparser instances outside the train/test size-density range.

The benchmark evaluates the full configuration grid on all splits so that baselines and oracle regret can be computed consistently. This is an offline benchmarking expense, not the deployment cost of the learned selector.

Primary metrics are:

- optimal-solved rate;
- feasible-solution rate;
- wall-clock solver runtime;
- final relative MIP gap;
- MIP node count;
- PAR10, where unsolved runs cost `10 × time_limit`;
- normalized PAR10 relative to the per-instance portfolio oracle;
- selector regret: `PAR10(selected) - PAR10(oracle)`.

Reports include mean, median, p90, standard deviation where applicable, and a paired bootstrap 95% CI for mean regret. Random configuration selection is repeated. Learned selectors are trained with repeated model seeds.

### Statistical interpretation

Instances are the paired experimental units. Every configuration is run on the same generated instance set under the same solver budget. The benchmark does not use test performance for hyperparameter selection. A tiny CI smoke run is treated only as an integration test and must not be interpreted as a scientific result.

## Exactness / Verification

There are three different notions of “oracle” or “exactness” and they are intentionally separated.

1. **HiGHS optimality:** a run is called optimal only when HiGHS returns its `kOptimal` model status.
2. **Independent exact oracle for tiny instances:** tests exhaustively enumerate all subsets for small set-cover instances and verify that the HiGHS optimal objective matches the independently computed optimum.
3. **Portfolio oracle:** the best candidate configuration for an instance. This is exact only with respect to the finite configuration portfolio and observed benchmark cost; it says nothing about configurations outside that portfolio.

A run that hits the time limit is never relabeled “optimal” merely because its gap is small.

## Feasibility Audit

Every returned incumbent is checked outside HiGHS:

- decision-vector length;
- binary/integrality deviation;
- every covering constraint `A x >= 1`;
- variable bounds;
- objective recomputation from `c^T x`.

A solver result with no independently verified feasible decision is not counted as feasible. The raw JSONL contains maximum integrality and cover violation fields for every run.

## Computational Cost

The benchmark reports separately:

- total solver calls required to build/evaluate the configuration matrix;
- solver runtime for every instance/configuration pair;
- MIP node counts when provided by HiGHS;
- feature-extraction wall time;
- learned-selector prediction wall time.

A learned selector therefore cannot hide the offline cost of collecting configuration-performance data. At deployment time, its online cost is feature extraction + model inference + one solver run; the portfolio oracle needs all candidate runs and is not a deployable baseline.

## Reproducibility

Randomness is explicit and centralized in JSON config files:

- independent seed ranges per data split;
- fixed HiGHS random seed;
- fixed validation seed;
- repeated selector training seeds;
- repeated random-baseline seeds.

The solver is forced to one thread and HiGHS parallel MIP search is disabled for cleaner timing comparability. Timing is still machine-dependent; compare methods within the same environment rather than treating raw seconds as portable constants.

## Run Commands

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

Run lint/format checks:

```bash
ruff check .
ruff format --check .
```

Run the CI-sized integration benchmark:

```bash
solverconfig benchmark --config configs/smoke.json --output results/smoke
```

Run the larger research configuration:

```bash
solverconfig benchmark --config configs/benchmark.json --output results/benchmark
```

Outputs:

```text
summary.json       experiment metadata, budgets, selected hyperparameters, solver-call count
metrics.csv        test/OOD method summaries
raw_results.jsonl  one independently audited record per solver run
```

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── benchmark.json
│   └── smoke.json
├── scripts/
│   └── run_benchmark.py
├── src/solverconfig/
│   ├── audit.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── configurations.py
│   ├── features.py
│   ├── instances.py
│   ├── metrics.py
│   ├── oracle.py
│   ├── selector.py
│   └── solver.py
├── tests/
│   ├── test_audit.py
│   ├── test_benchmark_schema.py
│   ├── test_features.py
│   ├── test_instances.py
│   ├── test_oracle.py
│   ├── test_selector.py
│   └── test_solver_integration.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Tests

The suite is methodological rather than import-only. It checks:

- deterministic and feasible instance generation;
- fixed finite feature schema;
- independent objective/constraint feasibility auditing;
- exhaustive exact-oracle correctness on a hand-checkable instance;
- learned selector behavior on a controlled cost surface;
- HiGHS optimum versus independent exhaustive enumeration;
- end-to-end smoke benchmark output schema.

GitHub Actions runs Python 3.11 and 3.12, performs `pip check`, Ruff lint, Ruff format check, tests, a complete smoke experiment, and validates that benchmark artifacts are nonempty.

## Experimental Interpretation

Three outcomes are all scientifically acceptable:

1. the learned selector reduces regret versus the global/default baselines;
2. it is statistically indistinguishable at this scale;
3. it performs worse.

The repository does not suppress negative results. A configuration-label model can have good prediction accuracy but poor downstream cost; conversely, it can have modest label accuracy and low regret if errors occur among near-tied configurations. For this reason, selector regret and PAR-style solve cost are the main criteria.

## Limitations

- The current data is synthetic and uses one MILP family, set covering.
- The portfolio contains only six HiGHS configurations and explores a tiny fraction of the solver's full parameter space.
- Random forests are a pragmatic low-data selector, not a claim that this model class is best for solver configuration.
- Wall-clock timing is noisy and hardware-dependent even with one solver thread.
- The OOD split changes size/density but is not a substitute for real industrial distribution shift.
- The benchmark does not perform dynamic configuration during a solve.
- It does not tune feasibility/optimality tolerances per instance because changing stopping semantics would make comparisons harder to interpret.
- It does not claim transfer from synthetic set covering to arbitrary MILPs.

## Claims Boundary

This repository supports claims only about **the implemented benchmark protocol and observed runs**.

It does **not** claim:

- state-of-the-art solver configuration;
- universal superiority over HiGHS defaults;
- paper-level benchmark performance from CI smoke runs;
- production-ready solver tuning;
- industrial savings;
- that the portfolio oracle is the globally best HiGHS configuration;
- that a time-limited feasible incumbent is optimal;
- that configuration selection and branching-policy learning are the same problem;
- that synthetic results transfer unchanged to industrial MILPs.

## Research Context and Related Repositories

This repository extends a broader portfolio line while remaining standalone:

- [`learning-to-branch-mip-gnn-scip-pytorch`](https://github.com/jorsacademy/learning-to-branch-mip-gnn-scip-pytorch): learned **within-tree branching**; this repo instead chooses one instance-level solver configuration before solving.
- [`gnn-guided-generalized-assignment-variable-fixing-pytorch`](https://github.com/jorsacademy/gnn-guided-generalized-assignment-variable-fixing-pytorch): learned **primal variable fixing + exact repair**; this repo leaves decisions to HiGHS and changes solver strategy only.
- [`predict-then-optimize-production-planning-spo-plus-pytorch`](https://github.com/jorsacademy/predict-then-optimize-production-planning-spo-plus-pytorch): **decision-focused learning** over uncertain objective coefficients; this repo learns solver performance, not problem coefficients.
- [`differentiable-optimization-pytorch`](https://github.com/jorsacademy/differentiable-optimization-pytorch): differentiable optimization research context; there is no differentiable solver layer in the present benchmark.
- [`sequential-decision-analytics`](https://github.com/jorsacademy/sequential-decision-analytics): broader sequential decision context; the present selector is a one-shot decision, not an online policy.

No code dependency is introduced between these repositories.

## Literature Map

### Foundational algorithm selection and configuration

- Rice, J. R. (1976). *The Algorithm Selection Problem*. Advances in Computers, 15, 65–118.
- Xu, L., Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2008). *SATzilla: Portfolio-based Algorithm Selection for SAT*. Journal of Artificial Intelligence Research, 32, 565–606. https://doi.org/10.1613/jair.2490
- Hutter, F., Hoos, H. H., Leyton-Brown, K., & Stützle, T. (2009). *ParamILS: An Automatic Algorithm Configuration Framework*. Journal of Artificial Intelligence Research, 36, 267–306. https://doi.org/10.1613/jair.2861
- Kadioglu, S., Malitsky, Y., Sellmann, M., & Tierney, K. (2010). *ISAC – Instance-Specific Algorithm Configuration*. ECAI 2010, 751–756. https://doi.org/10.3233/978-1-60750-606-5-751
- Xu, L., Hoos, H. H., & Leyton-Brown, K. (2010). *Hydra: Automatically Configuring Algorithms for Portfolio-Based Selection*. AAAI 2010. https://doi.org/10.1609/aaai.v24i1.7565
- Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011). *Sequential Model-Based Optimization for General Algorithm Configuration*. LION 5, 507–523. https://doi.org/10.1007/978-3-642-25566-3_40
- Schede, E., Brandt, J., Tornede, A., Wever, M., Bengs, V., Hüllermeier, E., & Tierney, K. (2022). *A Survey of Methods for Automated Algorithm Configuration*. Journal of Artificial Intelligence Research, 75, 425–487.

### MIP and learned solver control

- Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*. NeurIPS 2019. This is a branching-policy paper, included to clarify the different control level addressed here.
- König et al. (2023). *Speeding up neural network robustness verification via algorithm configuration and an optimised mixed integer linear programming solver portfolio*. Machine Learning. The work combines MIP solver configuration and portfolio ideas in a verification setting.

### 2024–2026 developments

- Iommazzo, G., D'Ambrosio, C., Frangioni, A., & Liberti, L. (2024). *Learning to Configure Mathematical Programming Solvers by Mathematical Programming*. arXiv:2401.05041. Uses learned relationships between instances, configurations and performance, followed by an optimization model enforcing configuration consistency.
- Kemminer, R., Lange, J., Kempkes, J. P., Tierney, K., et al. (2024). *Configuring Mixed-Integer Programming Solvers for Large-Scale Instances*. Operations Research Forum, 5, 48. https://doi.org/10.1007/s43069-024-00327-7
- Liu, C., Dong, Z., Ma, H., et al. (2024). *L2P-MIP: Learning to Presolve for Mixed Integer Programming*. ICLR 2024. This is dynamic/learned presolve control rather than the fixed instance-level portfolio used here.
- Long, F. X., Frenzel, M., Krause, P., et al. (2024). *Landscape-Aware Automated Algorithm Configuration Using Multi-output Mixed Regression and Classification*. PPSN 2024.
- Weiß, D., Schede, E., & Tierney, K. (2025). *Selector: Ensemble-Based Automated Algorithm Configuration*. Journal of Heuristics, 31, 28. https://doi.org/10.1007/s10732-025-09561-6

The 2024–2026 papers are related directions, not code sources for this repository. The implementation here is independent and intentionally narrower.

## Solver Documentation

The benchmark targets the open-source HiGHS solver through `highspy`:

- Python interface: https://ergo-code.github.io/HiGHS/stable/interfaces/python/
- Python examples: https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/
- Option API: https://ergo-code.github.io/HiGHS/stable/options/intro/
- Option definitions: https://ergo-code.github.io/HiGHS/stable/options/definitions/
- Model status enum: https://ergo-code.github.io/HiGHS/stable/structures/enums/

## License

MIT.

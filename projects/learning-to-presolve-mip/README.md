# Learning to Presolve for MIP

A reproducible research sandbox for **instance-specific presolve configuration selection** in mixed-integer programming.

The central design principle is conservative: machine learning chooses **which safe presolve configuration to run**. It never decides whether an arbitrary variable or constraint may be deleted. A poor prediction can therefore waste computational effort, but it cannot invalidate the model by itself.

The repository is inspired by **L2P-MIP: Learning to Presolve for Mixed Integer Programming** (ICLR 2024), which studies instance-specific presolving rather than fixed instance-agnostic settings. This project is intentionally smaller and transparent; it is not a reproduction of L2P-MIP and it does not reimplement the presolve stack of SCIP, Gurobi, CPLEX, Xpress, or HiGHS.

## Problem class

The synthetic instances are binary packing MIPs:

```text
maximize      c^T x
subject to    A x <= b
              x in {0,1}^n
```

with nonnegative resource coefficients `A >= 0` and `b >= 0`.

The generator creates heterogeneous structural regimes: ordinary instances, instances with safe variable-fixing opportunities, instances with componentwise redundant packing rows, and mixed instances containing both. This prevents one configuration from trivially dominating every generated case.

## Safe transformations

Two rule families are implemented.

### Variable fixing and elimination

For this restricted packing model, the following transformations are optimality preserving:

- if `c_j <= 0`, set `x_j = 0`;
- if `A_ij > b_i` for any row `i`, set `x_j = 0` because selecting the item alone violates that packing row;
- if column `j` is identically zero and `c_j > 0`, set `x_j = 1` because it consumes no constrained resource and improves the maximization objective.

Fixed variables are eliminated from the residual model. Objective offsets and the original-variable mapping are retained so that an optimal residual solution can be reconstructed and checked in original variable space.

The nonpositive-profit rule can remove feasible solutions with a zero-profit variable set to one, so it is described correctly as **optimality preserving**, not as preserving the entire feasible set.

### Redundant packing-row elimination

For two packing rows

```text
A_s x <= b_s
A_w x <= b_w
```

with `x >= 0`, if

```text
A_s >= A_w  componentwise
b_s <= b_w
```

then the strong row implies the weak row, and the weak row can be removed safely. All-zero rows with nonnegative right-hand sides are also redundant.

These reductions are deliberately simple. Production presolvers contain additional mechanisms such as propagation, probing, aggregation, coefficient strengthening, implication processing, sparsification, dual reductions, and specialized constraint-handler presolve.

## Configurations

The learned selector chooses one of four configurations:

| Configuration | Safe variable fixing | Redundant-row scan |
| --- | ---: | ---: |
| `none` | no | no |
| `fixing` | yes | no |
| `rows` | no | yes |
| `full` | yes | yes |

The transformations themselves are deterministic and mathematically certified. Learning affects strategy, not correctness.

## Exact downstream solve

Every residual model is solved with `scipy.optimize.milp`, which uses HiGHS. HiGHS' own presolve is disabled in this experiment using `options={"presolve": False}` so that the effect of the custom transformations remains visible.

For every evaluated configuration the code solves the residual binary MIP, reconstructs the solution in original variable space, checks original feasibility, recomputes the original objective, and verifies exact objective equivalence across all safe configurations.

Disabling built-in presolve is an experimental control, not a recommendation for production optimization.

## Deterministic work proxy

Tiny synthetic MIPs solve too quickly for wall-clock measurements to provide stable supervised labels. The repository therefore uses a deterministic work proxy rather than treating noisy microbenchmarks as reliable runtime measurements.

For residual complexity

```text
R = residual_nonzeros
    + 2 * residual_variables
    + 2 * residual_constraints
```

and HiGHS branch-and-bound node count `N`, the score is

```text
work proxy = (N + 1) * R + 0.025 * presolve_checks
```

The first term rewards a smaller residual model and fewer B&B nodes. The second charges explicit work performed by the custom presolver, so an expensive redundancy scan is not free when it finds nothing.

This score is transparent and reproducible, but it is **not a calibrated CPU-time model**.

## Oracle labels

For each training instance all four safe configurations are evaluated:

```text
MIP instance
    |
    +--> none --------+
    +--> fixing ------+
    +--> rows --------+--> exact solve + work proxy --> oracle label
    +--> full --------+
```

The oracle first verifies exact objective equivalence, then selects the minimum-work configuration. It is exhaustive over this declared four-action menu, not over every possible presolve schedule or solver parameterization.

## Instance features

The classifier receives 12 transparent pre-presolve features: log numbers of variables and constraints, matrix density, mean and standard deviation of row tightness, fractions of nonpositive-profit variables, individually infeasible variables, and zero columns, normalized average column nonzero count, maximum row cosine similarity, coefficient variation, and positive-profit variation.

The feature vector describes structural signals without directly executing the full custom presolver.

## Learned selector

`PresolveSelector` is a small PyTorch multilayer perceptron:

```text
12 structural features
       |
  standardization
       |
 Linear + ReLU
       |
 Linear + ReLU
       |
 4 configuration logits
```

Feature normalization statistics are fitted only on the training split and stored in the checkpoint. Training uses weighted cross-entropy so minority oracle configurations are not ignored when labels are imbalanced.

## Baselines

The unseen benchmark compares:

- always `none`;
- always `fixing`;
- always `rows`;
- always `full`;
- a transparent handcrafted heuristic;
- the learned selector;
- the exhaustive finite-action oracle.

Reported metrics include mean work proxy, B&B nodes, residual model size, residual nonzeros, presolve checks, oracle-match rate, and normalized regret relative to the oracle configuration.

## Development experiment

The following run used disjoint training, validation, and benchmark seed ranges:

```bash
python scripts/train.py \
  --instances 120 \
  --validation-instances 40 \
  --vars 28 \
  --constraints 9 \
  --epochs 100 \
  --hidden-dim 40 \
  --learning-rate 0.002 \
  --seed 2026 \
  --checkpoint checkpoints/presolve_selector.pt

python scripts/benchmark.py \
  --checkpoint checkpoints/presolve_selector.pt \
  --instances 60 \
  --vars 28 \
  --constraints 9 \
  --seed-start 302600
```

Training oracle labels were:

```text
none       46
fixing     54
rows       12
full        8
```

On the disjoint 40-instance validation set:

```text
oracle-label accuracy          0.7000
mean normalized work regret    0.0402
```

The separate 60-instance benchmark produced:

```text
policy       mean work    mean nodes   mean residual size   mean regret   oracle match
none           977.733        3.85             37.00          0.0605        0.417
fixing         963.661        3.90             35.53          0.0172        0.467
rows          1001.951        3.88             36.50          0.1291        0.050
full           992.543        3.97             35.03          0.0974        0.067
heuristic      965.363        3.97             35.03          0.0393        0.517
learned        952.766        3.90             35.48          0.0110        0.767
oracle         931.134        3.75             35.63          0.0000        1.000
```

The learned selector has the lowest mean work proxy among the deployable policies in this development experiment and lower mean regret than the handcrafted heuristic. This result is limited to the declared synthetic generator, feature set, work proxy, seed ranges, and configuration menu. It is **not evidence of production-solver speedup**.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Inspect one instance

```bash
python -m learning_to_presolve \
  --vars 32 \
  --constraints 10 \
  --seed 7
```

With a trained checkpoint:

```bash
python -m learning_to_presolve \
  --vars 32 \
  --constraints 10 \
  --seed 7 \
  --checkpoint checkpoints/presolve_selector.pt
```

The CLI prints JSON containing every configuration's exact objective, B&B nodes, residual dimensions, custom-presolve checks, work proxy, oracle configuration, and optional learned prediction.

## Train and benchmark

```bash
python scripts/train.py --checkpoint checkpoints/presolve_selector.pt
python scripts/benchmark.py \
  --checkpoint checkpoints/presolve_selector.pt \
  --instances 80
```

Machine-readable benchmark output is available with `--json`.

## Tests

```bash
pytest -q
```

The regression suite checks reproducible instance generation, hand-checkable safe fixings, original-space solution reconstruction, redundant-row elimination, objective preservation across every configuration and multiple seeds, reconstructed feasibility, structural features, oracle consistency, dataset shapes, finite model training, regret metrics, checkpoint round trips, benchmark baselines, and CLI JSON output.

GitHub Actions runs package installation, compilation, Ruff, the full test suite, and a small end-to-end train/benchmark smoke experiment on Python 3.11 and 3.12.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── scripts/
│   ├── benchmark.py
│   └── train.py
├── src/learning_to_presolve/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── core.py
│   ├── io.py
│   └── model.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_cli.py
│   ├── test_core.py
│   └── test_model.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Methodological boundaries

This repository does **not** claim to provide a production presolver, direct learned deletion of arbitrary rows or columns, uncertified learned variable fixing, a reproduction of L2P-MIP, end-to-end configuration of a commercial/open-source solver's full presolve parameter space, MIPLIB speedups, calibrated solve-time prediction, or generalization beyond the synthetic binary-packing distribution.

A stronger research extension would integrate with SCIP, expose genuine presolver parameters, separate training families from out-of-distribution test families, measure primal-dual integral and solve time under repeated controlled runs, and account explicitly for feature-extraction and inference overhead.

## Research grounding

- C. Liu, Z. Dong, H. Ma, W. Luo, X. Li, B. Pang, J. Zeng, and J. Yan, *L2P-MIP: Learning to Presolve for Mixed Integer Programming*, ICLR 2024: https://openreview.net/forum?id=McfYbKnpT8
- SCIP documentation, *How to add presolvers*: https://scipopt.org/doc/html/PRESOL.php
- SCIP documentation, *Default Presolvers*: https://scipopt.org/scip/doc/html/group__DEFPLUGINS__PRESOL.php
- SciPy documentation, `scipy.optimize.milp`.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.

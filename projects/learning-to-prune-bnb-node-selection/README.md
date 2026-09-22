# Learning to Prune / Branch-and-Bound Node Selection

A small, auditable research implementation of **learning-guided branch-and-bound search**.
The repository separates two ideas that are often conflated:

1. **learned node selection** changes which open B&B node is processed next but never deletes a
   node, so the solver remains exact;
2. **learned node pruning** may discard a node based on a learned classifier, so it is explicitly
   approximate and does **not** claim an optimality certificate.

The implementation uses 0-1 knapsack as a controlled problem family. The point is not to beat
SCIP/Gurobi on knapsack. The point is to expose the complete research loop in code that is small
enough to inspect:

```text
training instances
      |
exact reference solutions
      |
B&B states + oracle path labels
      |
class-balanced logistic imitation models
      |
      +--> learned node score --------> exact B&B search order
      |
      +--> learned expand probability -> optional approximate pruning
      |
held-out brute-force verification + benchmark
```

## Research lineage

The design is motivated primarily by He, Daumé III, and Eisner,
**Learning to Search in Branch and Bound Algorithms** (NeurIPS 2014). That work learns both a
node-selection policy and a node-pruning policy by imitation learning. It also makes an important
methodological distinction: aggressive learned pruning can trade away a proof of optimality in
exchange for faster search.

Node selection remains an active research problem. Zhang et al.,
**Learning to Select Nodes in Branch and Bound with Sufficient Tree Representation** (ICLR 2025),
uses a richer tree representation, GNNs, and reinforcement learning. This repository is deliberately
much smaller: it is an educational/research baseline, not a reproduction of TRGNN.

SCIP itself exposes node selection as a solver plugin because the choice of the next open leaf can
materially affect the search trajectory. A production-grade learned selector would naturally move
from this sandbox to SCIP/PySCIPOpt callbacks after the policy and evaluation methodology are
validated.

References are listed in [`docs/research_notes.md`](docs/research_notes.md).

## Exactness boundary

This repository treats exactness as a first-class contract.

| Mode | Learned component | Deletes nodes? | Optimality guarantee |
|---|---|---:|---:|
| `best_bound_exact` | none | no | yes |
| `dfs_exact` | none | no | yes |
| `learned_selection_exact` | node ranking | no | yes |
| `learned_selection_plus_pruning_approx` | ranking + prune classifier | **yes** | **no** |

The exact modes use only mathematically safe fathoming:

- infeasible capacity states are never created;
- a node is safely pruned when its fractional-knapsack upper bound cannot beat the incumbent;
- learned node selection only reorders the open list.

The approximate mode adds a separate learned pruning gate. The result object therefore reports
`optimality_guaranteed=False` whenever learned pruning is enabled.

## Problem and bound

For sorted items, the 0-1 knapsack model is

```text
maximize    sum_i value[i] * x[i]
subject to  sum_i weight[i] * x[i] <= capacity
            x[i] in {0, 1}
```

At a partial assignment, the upper bound is the standard fractional-knapsack relaxation: remaining
capacity is filled greedily by value/weight ratio, allowing the last item fractionally. Because it is
an upper bound for the integer problem, it is safe for exact fathoming.

## Node features

The learned policies receive scale-normalized state features only; the true optimum and oracle label
are never input features:

- depth fraction;
- capacity used and capacity slack;
- current node value;
- fractional upper bound;
- remaining bound above the node value;
- incumbent value;
- bound-to-incumbent gap;
- remaining item fraction;
- last branch decision.

Oracle labels are used only during training: a node is labeled `expand=1` when its fixed prefix is
consistent with one exact optimal solution for that training instance.

## Learning method

To keep the dependency surface auditable, both policies are implemented as NumPy logistic models
rather than a deep-learning framework.

The training procedure:

1. generates deterministic random knapsack instances;
2. obtains exact reference solutions by exhaustive enumeration for the deliberately small training
   scale;
3. runs exact best-bound B&B and records encountered states;
4. labels whether each encountered state lies on the known optimal prefix;
5. standardizes features;
6. fits class-balanced logistic models for selection and pruning;
7. gives the pruning model extra positive-class weight so false pruning of an oracle-path node is
   penalized more heavily.

This is **offline oracle imitation**, inspired by the earlier learning-to-search literature. It is not
claimed to be a complete reproduction of DAgger, TRGNN, SCIP's internal node selectors, or any
commercial solver policy.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Tests

```bash
ruff check .
pytest
```

The tests verify, among other things:

- deterministic instance generation;
- correct ordering for the fractional-knapsack upper bound;
- root LP bound dominates the exact integer optimum;
- best-bound, DFS, and learned-selection exact modes match brute force on multiple instances;
- learned pruning is explicitly marked approximate;
- oracle datasets contain both positive and negative states;
- learned-policy JSON serialization is stable.

## Train a policy

```bash
python scripts/train.py \
  --seed 42 \
  --train-instances 80 \
  --validation-instances 20 \
  --items 12 \
  --out artifacts/policy.json \
  --metrics-out artifacts/training_metrics.json
```

The training script writes both the policy and validation metadata as JSON.

## Benchmark

```bash
python scripts/benchmark.py \
  --policy artifacts/policy.json \
  --seed 7000 \
  --instances 30 \
  --items 14 \
  --prune-threshold 0.50 \
  --output artifacts/benchmark.json
```

Every benchmark instance is also solved by exhaustive enumeration. The script raises an error if an
**exact** method disagrees with the reference optimum. The approximate learned-pruning method is
measured by relative objective gap instead of being silently treated as exact.

## Pre-publication deterministic smoke validation

Before publishing the repository, the following small deterministic configuration was executed
locally against the code in this repository:

```text
training instances       20
validation instances      8
items / training instance 10
training states          451
validation states        125
validation accuracy      70.4%
validation balanced acc. 59.375%
validation recall        98.75%
```

A held-out 8-instance / 11-item benchmark with pruning threshold `0.50` produced:

```text
method                                      mean gap    max gap
best_bound_exact                              0.000%     0.000%
dfs_exact                                     0.000%     0.000%
learned_selection_exact                       0.000%     0.000%
learned_selection_plus_pruning_approx         0.016%     0.101%
```

These are **smoke-test observations**, not solver-performance claims. The learned exact selector is
slower than the compact best-bound implementation on this tiny Python workload; that negative
result is expected to be reported rather than hidden. The purpose of the experiment is to validate
the correctness boundary and end-to-end learned-search mechanics.

## GitHub Actions

`.github/workflows/ci.yml` runs:

- linting and unit tests on Python 3.10, 3.11, and 3.12;
- a deterministic training smoke run;
- a held-out benchmark that automatically fails if any exact method disagrees with brute force;
- upload of the generated policy, training metrics, and benchmark JSON as a workflow artifact.

The research-smoke job intentionally exercises approximate pruning with threshold `0.50`, while
still enforcing exactness on every method that claims an optimality guarantee.

## Repository map

```text
src/ltp_bnb/
  problem.py      deterministic knapsack family + fractional upper bound + brute-force oracle
  features.py     leakage-free normalized B&B state features
  policy.py       NumPy logistic imitation models + JSON persistence
  bnb.py          exact B&B core, learned node ranking, opt-in approximate learned pruning
  training.py     oracle-state collection, training, validation metrics
  evaluation.py   held-out exactness checks and benchmark aggregation
scripts/
  train.py
  benchmark.py
tests/
.github/workflows/ci.yml
docs/research_notes.md
```

## What this project demonstrates

The central engineering lesson is that **learned search guidance and learned pruning have different
correctness semantics**.

A learned node selector can be poor and merely waste time while an otherwise exact B&B algorithm
still eventually proves the optimum. A learned prune decision can remove the only subtree
containing the optimum. Treating these as interchangeable "ML for B&B" components hides the most
important deployment boundary.

For a production extension, the natural next step is a PySCIPOpt/SCIP node-selector plugin with
solver-native state features and a benchmark on standard MILP families. That should be a separate
research phase rather than retroactively treating this compact knapsack sandbox as production MIP
acceleration evidence.

## Limitations

- controlled 0-1 knapsack family rather than heterogeneous MIPLIB instances;
- exhaustive oracle generation intentionally limits training/validation problem size;
- linear policies instead of GNN/tree encoders;
- offline imitation rather than full interactive DAgger or reinforcement learning;
- no claim that learned selection beats tuned solver-native node selectors;
- learned pruning is intentionally approximate and cannot certify optimality;
- Python microsecond/millisecond timings are not portable performance guarantees.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.

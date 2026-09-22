# GNN-Guided Generalized Assignment Variable Fixing

A from-scratch PyTorch graph-neural-network project for using learned predictions as **primal guidance** inside an exact Generalized Assignment Problem (GAP) solver.

The neural network does not replace the optimizer.

```text
GAP instance
   ↓
bipartite machine-job graph
   ↓
message-passing GNN
   ↓
job → machine probabilities
   ↓
validation-calibrated high-confidence fixing
   ↓
exact HiGHS MILP repair
```

The central research question is whether a GNN can identify sufficiently reliable assignment variables to reduce the residual MIP without surrendering exact feasibility repair.

## GAP formulation

For machines `i` and jobs `j`, decision variables are `x[i,j] ∈ {0,1}`. The model minimizes total assignment cost subject to exactly one machine per job and machine capacity limits. The downstream problem is solved with `scipy.optimize.milp` / HiGHS.

## Graph representation

The model uses machine nodes, job nodes and assignment edges. Machine features summarize capacity pressure and assignment costs; job features summarize relative cost/resource difficulty; edge features contain normalized assignment cost and resource/capacity information. No optimal-solution label is included in the graph features.

The GNN is implemented directly in PyTorch without PyTorch Geometric. Each message-passing layer forms edge messages from machine, job and edge embeddings, aggregates them back to both partitions, applies residual node updates and LayerNorm, then scores every machine-job edge. A softmax across machines yields `P(machine i | job j, instance)`.

Training labels come from independently solved optimal GAP instances and per-job cross entropy is used for supervised learning.

## Validation-calibrated fixing

A raw softmax threshold is not assumed calibrated. The repository chooses a confidence threshold on a disjoint validation set. Given a target fixing precision and minimum coverage, it chooses the largest-coverage threshold that satisfies the precision requirement. If the target cannot be met, the conservative fallback returns a threshold above one and fixes no jobs.

## Exact repair

High-confidence assignments can collectively make the residual GAP infeasible. The repair procedure therefore:

```text
solve residual MILP
      ↓
if infeasible: relax least-confident fixing
      ↓
re-solve until feasible
```

The resulting repair MILP is exact **conditional on the surviving fixings**. A wrong but feasible fixing may exclude the globally optimal solution of the original GAP, so learned fixing is not described as exact optimization.

## Controls and baselines

The project includes:

- vanilla exact HiGHS MILP;
- multi-start capacity-aware greedy heuristic;
- GNN confidence fixing + exact repair;
- oracle fixing of the same job set using the known optimal assignment.

The oracle control separates residual-problem-size reduction from damage caused by incorrect learned fixings.

## Independent exact oracle

On tiny regression fixtures, every possible machine assignment is exhaustively enumerated and capacity-infeasible assignments are discarded. The resulting optimum must match HiGHS to numerical tolerance.

## Development benchmark

Seed-42 local development configuration:

```text
machines                  6
jobs                     18
training instances       64
validation instances     16
test instances           16
GNN epochs               20
hidden dimension         48
message-passing layers    3

target fixing precision 95%
minimum fixing coverage   5%
margin threshold          0.05
```

Observed local result:

```text
best validation assignment accuracy   78.8%
calibrated confidence threshold        0.807
validation fixing precision            95.2%
validation fixing coverage             43.4%

test assignment accuracy               78.5%
mean job confidence                     0.743
initially fixed jobs                    8.25 / 18
finally fixed jobs                      8.06 / 18
wrong initial fixes                     0.250 / instance
relaxed infeasible fixings              0.188 / instance

GNN repair objective gap                0.045%
greedy heuristic objective gap          0.669%
```

Solver behavior in that run:

```text
mean HiGHS MIP nodes
vanilla MILP       0.81
GNN repair         0.88
oracle fixing      0.81

mean solve time
vanilla MILP       0.0064 s
GNN repair         0.0071 s
oracle fixing      0.0049 s
```

This does **not** demonstrate a learned solver speedup. The small synthetic GAPs are usually solved at or near the root node, so learned-fixing overhead can exceed any reduction benefit. This negative result is retained deliberately: classification accuracy does not imply branch-and-bound acceleration.

## Validated GitHub Actions run

GitHub Actions run `33103724522` completed successfully on Ubuntu 24.04 / CPython 3.12.14 with CPU PyTorch 2.13.0, NumPy 2.5.2 and SciPy 1.18.1. The self-test and all **8 regression tests** passed before the end-to-end GNN fixing smoke experiment.

Smoke configuration:

```text
machines                  5
jobs                     12
training instances       24
validation instances      8
test instances            8
epochs                     6
hidden dimension          32
message-passing layers     2

target fixing precision  85%
minimum fixing coverage    5%
margin threshold          0.02
```

Observed GitHub-runner result:

```text
best validation assignment accuracy   70.8%
calibrated confidence threshold        1.010
validation fixing precision           100.0%
validation fixing coverage              0.0%

test assignment accuracy               72.9%
mean job confidence                     0.311
initially fixed jobs                    0.00 / 12
finally fixed jobs                      0.00 / 12
wrong initial fixes                     0.000
relaxed fixings                         0.000

GNN repair objective gap                0.000%
greedy heuristic objective gap          0.465%
```

Solver metrics in the smoke run:

```text
mean HiGHS nodes
vanilla MILP       1.00
GNN repair         1.00
oracle fixing      1.00

mean solve time
vanilla MILP       0.0108 s
GNN repair         0.0135 s
oracle fixing      0.0107 s
```

The calibrated threshold `1.010` means the short CI-trained network did **not** achieve the requested 85% fixing precision at 5% minimum validation coverage, so the conservative calibration intentionally fixed no jobs. The zero repair gap is therefore not evidence of learned guidance quality; it is the expected result of falling back to the vanilla residual problem. This is retained as a correctness/safety behavior, not hidden as a failed benchmark.

Run: https://github.com/jorsacademy/gnn-guided-generalized-assignment-variable-fixing-pytorch/actions/runs/33103724522

## Tests

The regression suite checks:

- HiGHS MILP vs full enumeration on a tiny GAP;
- post-solve feasibility;
- deterministic instance generation;
- bipartite GNN tensor shapes and gradient flow;
- exact MILP training labels;
- validation confidence calibration;
- infeasible learned-fixing relaxation;
- end-to-end GNN training + exact repair.

## Run

```bash
pip install -r requirements.txt
python gnn_gap_variable_fixing.py --self-test
python -m unittest discover -s tests -v
```

Development experiment:

```bash
python gnn_gap_variable_fixing.py \
  --seed 42 \
  --machines 6 \
  --jobs 18 \
  --train-instances 64 \
  --validation-instances 16 \
  --test-instances 16 \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 48 \
  --layers 3 \
  --target-fix-precision 0.95 \
  --minimum-fix-coverage 0.05
```

## Exactness and claims

Exact statements:

- training labels are produced by HiGHS solutions reported optimal;
- vanilla test GAPs are solved to HiGHS optimality;
- tiny regression instances are independently checked by exhaustive enumeration;
- repair is solved exactly for the residual MILP conditional on surviving fixings.

Not claimed:

- GNN predictions are exact;
- learned fixing preserves the original global optimum;
- the current benchmark accelerates HiGHS;
- classification accuracy implies branch-and-bound improvement;
- synthetic timings transfer to production MIPs.

The point of the project is to expose the full learned-primal-guidance pipeline and measure both its benefits and its failure modes.

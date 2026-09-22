# Research Smoke Results

These results come from GitHub Actions run `33274576957` using the deliberately small `Research Smoke Benchmark` preset. They validate the end-to-end training and evaluation pipeline; they are **not** intended as final scientific claims because the training budgets, number of test instances, and number of seeds are intentionally small.

## Training snapshot

| Model | Training data | Final loss | Training top-1 expert agreement |
|---|---:|---:|---:|
| MLP-MSE | 275 tree nodes / 541 candidate rows | 0.156709 | 0.698182 |
| MLP-listwise | 275 tree nodes / 541 candidate rows | 0.540756 | 0.789091 |
| Bipartite GNN | 105 graph states | 0.592493 | 0.742857 |

## Paired repeated-seed benchmark

The benchmark used 3 seeds and 4 matched MILP instances per seed. Values are seed-level means with Student-t 95% confidence-interval half-widths.

| Policy | B&B nodes | LP solves | Wall sec/instance | Node reduction vs most-frac | Node reduction vs reliability |
|---|---:|---:|---:|---:|---:|
| most_fractional | 89.333 ± 132.012 | 89.333 ± 132.012 | 0.119 ± 0.176 | 0.000 ± 0.000% | — |
| pseudocost | 75.167 ± 74.961 | 75.167 ± 74.961 | 0.102 ± 0.100 | 9.822 ± 39.544% | 1.719 ± 66.780% |
| reliability | 75.000 ± 24.434 | 108.667 ± 24.883 | 0.146 ± 0.032 | 1.188 ± 97.787% | 0.000 ± 0.000% |
| strong | 70.000 ± 24.810 | 211.667 ± 74.961 | 0.281 ± 0.095 | 7.594 ± 90.617% | 5.766 ± 43.508% |
| learned_mse | 70.000 ± 39.843 | 70.000 ± 39.843 | 0.101 ± 0.057 | 11.184 ± 66.086% | 6.904 ± 36.422% |
| learned_listwise | 78.333 ± 19.415 | 78.333 ± 19.415 | 0.114 ± 0.028 | -4.598 ± 111.504% | -4.741 ± 10.529% |
| learned_gnn | 76.667 ± 26.822 | 76.667 ± 26.822 | 0.119 ± 0.042 | -0.853 ± 99.954% | -2.134 ± 2.972% |

## Held-out strong-branching top-1 agreement

| Learned policy | Agreement |
|---|---:|
| learned_mse | 0.417 ± 0.781 |
| learned_listwise | 0.458 ± 0.949 |
| learned_gnn | 0.648 ± 0.239 |

## Smoke-run interpretation

- All branching policies returned the same optimum on the paired benchmark instances.
- `learned_mse` matched strong branching's mean node count in this smoke run while using far fewer LP solves, but the uncertainty is large and the sample is too small for a performance claim.
- The GNN achieved the highest held-out expert agreement of the learned models in this smoke run, but it did not beat reliability branching on mean node count.
- Strong branching reduced mean node count but paid a large LP-solve overhead, which is exactly the trade-off the learned policies are designed to target.
- OOD smoke runs also completed for nominal `12v/5c`, `16v/5c`, `20v/8c`, tighter packing, and looser packing distributions. Because those runs used only 2 seeds and 2 instances per seed, the resulting Student-t intervals are extremely wide and should be treated only as execution/regression checks.

The complete workflow artifact also contains the three trained checkpoints, raw ID benchmark output, raw OOD benchmark output, and the generated `RESULTS.md` table.

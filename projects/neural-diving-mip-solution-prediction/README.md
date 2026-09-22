# Neural Diving for MIP Solution Prediction

A reproducible research sandbox for **learning partial integer assignments and completing them
with a trusted MILP solver**.

This repository is inspired by Neural Diving from Nair et al., *Solving Mixed Integer Programs
Using Neural Networks* (2020/2021), and its use as the initialization stage in Sonnerat et al.,
*Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs* (2021).

It is **not** a reproduction of those systems. The goal is to make the core mechanism small,
auditable, testable, and CI-friendly.

## Core idea

```text
related training MILPs
        |
        v
exact HiGHS solutions -> variable labels x_i in {0,1}
        |
        v
coefficient-derived variable features
        |
        v
PyTorch MLP -> P(x_i = 1)
        |
        +--> choose high-confidence variables
        +--> deterministic and stochastic partial assignments
        |
        v
fix selected variables
        |
        v
smaller residual sub-MIP
        |
        v
SciPy / HiGHS completes + validates solution
        |
        +--> infeasible? reduce fixation fraction and retry
        +--> still infeasible? fall back to full MIP
```

The learned model proposes. The mathematical solver completes and verifies.

## What is implemented

- deterministic synthetic family of related 0-1 packing MILPs;
- exact training labels from `scipy.optimize.milp` / HiGHS;
- ten variable features computed only from objective/constraint coefficients;
- a two-hidden-layer PyTorch solution predictor;
- class-balanced `BCEWithLogitsLoss` training;
- confidence-ranked deterministic partial fixing;
- stochastic partial assignments for multiple dives;
- adaptive backoff from aggressive to conservative fixing;
- full-MIP fallback;
- random partial-fixing baseline using the same sub-MIP solver;
- held-out objective-gap, fixation, fallback, node-count and runtime reporting;
- brute-force verifier for tiny test instances;
- Ruff + pytest + Python 3.10/3.11/3.12 GitHub Actions;
- deterministic research-smoke training and benchmark artifacts.

## Why this matches the Neural Diving research pattern

Nair et al. describe Neural Diving as a learned primal heuristic that produces multiple partial
assignments of integer variables, leaving the unassigned variables to define smaller sub-MIPs
which are completed by an off-the-shelf MIP solver. This repository implements that architectural
pattern in a compact open stack.

The important boundary is that **partial fixing is heuristic**. A wrong fixed value can exclude
the optimum or create infeasibility. This implementation therefore never treats the neural output
as a certificate and includes an explicit backoff/fallback path.

## Solver stack

The residual MIPs are solved with `scipy.optimize.milp`, which wraps the HiGHS mixed-integer
solver. The benchmark uses `mip_rel_gap=0` for the full reference solve and reports HiGHS node
counts/gaps when available.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Run tests

```bash
ruff check .
pytest
```

## Train

```bash
python scripts/train.py \
  --seed 42 \
  --train-instances 28 \
  --validation-instances 10 \
  --items 24 \
  --constraints 5 \
  --epochs 120 \
  --out artifacts/predictor.pt \
  --metrics-out artifacts/training_metrics.json
```

Training and validation are split by **instance seed**, not by individual variables, to avoid
leaking variables from the same MIP into both partitions.

## Benchmark

```bash
python scripts/benchmark.py \
  --model artifacts/predictor.pt \
  --seed 9000 \
  --instances 10 \
  --items 28 \
  --constraints 6 \
  --stochastic-samples 3 \
  --output artifacts/benchmark.json
```

The benchmark compares:

| Method | Learned? | Fixes variables? | Solver completion | Optimality claim |
|---|---:|---:|---:|---:|
| Full HiGHS reference | No | No | HiGHS | Yes when solver returns optimal |
| Neural Dive | Yes | Yes | HiGHS sub-MIP | No for a fixed candidate |
| Random Dive | No | Yes | HiGHS sub-MIP | No for a fixed candidate |
| Neural full fallback | Prediction ignored | No | HiGHS | Same as full solver |

## Tests enforce

- HiGHS equals brute-force optimum on a tiny instance;
- fixed sub-MIP bounds are honored;
- features contain no NaNs/Infs and have stable dimensionality;
- model serialization preserves predictions;
- predictor learns a known synthetic decision boundary;
- confidence fixing selects the intended variables;
- aggressive neural fixing can back off and still return a feasible solution;
- random baseline also has a trusted full-MIP fallback.

## Research interpretation

A strong result here is **not** "the neural network solves MILPs." Useful questions are:

- how much of an assignment can be fixed before infeasibility rises sharply?
- does confidence ranking outperform random fixing at the same fixed fraction?
- how does out-of-distribution instance size affect solution-prediction accuracy?
- do multiple stochastic partial assignments improve best-found primal quality?
- when does sub-MIP reduction compensate for neural inference and retry overhead?

The CI smoke benchmark is intentionally small and must not be interpreted as evidence of
production speedup.

## References

1. Vinod Nair et al. **Solving Mixed Integer Programs Using Neural Networks.** arXiv:2012.13349,
   2020; revised 2021.
2. Nicolas Sonnerat et al. **Learning a Large Neighborhood Search Algorithm for Mixed Integer
   Programs.** arXiv:2107.10201, 2021.
3. SciPy documentation: `scipy.optimize.milp`, a wrapper over HiGHS.

See [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) for scope and simplifications.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.

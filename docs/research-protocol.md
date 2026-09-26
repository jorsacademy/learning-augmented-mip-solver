# Research protocol

## Matched-instance evaluation

Learned and classical policies must be evaluated on the same generated instances and seeds.

## Minimum metrics

Report, when applicable:

- feasibility rate;
- objective value / optimality gap;
- B&B nodes, LP solves, cut rounds, pricing calls, or separation calls;
- wall-clock time;
- learned-component inference time;
- fallback rate;
- out-of-distribution behavior.

## Claim discipline

Small CI experiments establish correctness and reproducibility, not production speedups. Production claims require integration into a real solver, controlled hardware, and sufficiently large benchmark families.

## Preferred safety pattern

```text
learned proposal
      ↓
validation / confidence gate
      ↓
exact residual solve or classical fallback
      ↓
independent feasibility / bound check
```

When learning changes only ordering—branching, node search, cut ranking—the exact solver should remain responsible for the mathematical validity of every accepted action.

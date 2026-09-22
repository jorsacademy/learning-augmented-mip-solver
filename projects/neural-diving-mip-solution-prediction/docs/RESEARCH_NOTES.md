# Research notes

## Scope

This repository is an independent, compact implementation of the Neural Diving *pattern* for
binary packing MILPs. It is not a reproduction of the proprietary-scale architecture, datasets,
or SCIP integration used by Nair et al.

The design keeps the research boundary explicit:

1. solve related training instances with an exact/off-the-shelf MILP solver;
2. learn per-variable assignment probabilities from coefficient-derived features;
3. form partial assignments by fixing only high-confidence variables;
4. ask a classical MILP solver to complete the remaining sub-MIP;
5. back off when an aggressive partial assignment makes the sub-MIP infeasible;
6. compare against full MILP solving and random partial fixing.

## Literature grounding

- Nair et al., *Solving Mixed Integer Programs Using Neural Networks* (2020/2021): Neural
  Diving generates multiple partial assignments and solves the smaller residual MIPs with a
  base solver. The paper also emphasizes learning across related MIP instances.
- Sonnerat et al., *Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs*
  (2021): uses Neural Diving to generate an initial assignment before learned neighborhood
  selection.

## Deliberate simplifications

- binary packing family rather than MIPLIB-scale heterogeneous MILPs;
- independent per-variable MLP rather than a large graph/generative architecture;
- optimal solution labels rather than the paper's richer collection of feasible assignments;
- SciPy/HiGHS rather than SCIP callbacks;
- CPU-only deterministic training in CI;
- small held-out experiments intended to validate mechanics, not claim solver speedups.

## Safety / correctness boundary

A learned fixing can remove the true optimum or make a sub-MIP infeasible. Therefore the neural
prediction is never treated as a certificate. HiGHS validates feasibility of every completed
candidate. The backoff schedule relaxes fixings, and the final fallback solves the original MIP.
This means the *pipeline* can always return to the trusted solver, while a nonzero-fix candidate
is a heuristic primal solution rather than an optimality proof.

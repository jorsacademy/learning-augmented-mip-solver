# Literature and benchmark protocol notes

This document records design choices that are easy to lose when only reading code.

## What is being learned?

The learned object is a **variable-ordering score** for the current finite-domain CP state. It does not predict a coloring directly and it does not remove values or constraints. At a branching node, every unassigned candidate is featurized; the fitted model scores candidates; the highest score is selected; exact DFS then continues with the same propagation and value ordering as the classical baselines.

The expensive supervision target is propagation lookahead. For every candidate variable, every value in the current domain is tentatively assigned and propagated. The score rewards immediate contradictions and domain reduction. This is intentionally analogous in spirit—not implementation—to strong branching in MIP.

## Why graph coloring?

Graph coloring makes domain propagation, first-fail behavior, degree structure, and infeasibility certificates transparent. It also permits a genuinely independent exhaustive oracle on small instances. The first version therefore favors methodological visibility over industrial realism.

## Why no custom CP-SAT callback?

The public OR-Tools Python CP-SAT model surface exposes `add_decision_strategy` with finite enumerations of variable-selection and domain-reduction strategies. Solver parameters expose `FIXED_SEARCH` and other global search modes. The public solution callback is for observing solutions, not arbitrary per-node branching decisions.

Consequently, the learned CP-SAT experiment is root/static control only. Classical public fixed-search baselines may still use built-in dynamic selectors such as `CHOOSE_MIN_DOMAIN_SIZE`. A future version could use a solver that exposes a true search callback, or a lower-level solver integration, but it should not pretend the current Python API provides one.

## Exactness boundary

For the transparent solver, learned control is exact if no limit is imposed because it reorders exhaustive branching. A time/node limit changes the claim: an unfinished infeasibility proof is `UNKNOWN`.

For CP-SAT, solver status comes from OR-Tools. Feasible assignments are independently audited by this project; the repository does not independently verify CP-SAT's internal proof system.

## Data splitting

Training, validation, in-distribution test, OOD, and exhaustive-oracle instances use disjoint deterministic seed namespaces. The selected supervised model is chosen by validation imitation accuracy only. Test/OOD search results do not feed model choice.

## Statistics

The benchmark retains raw per-instance records. Aggregates include mean, sample standard deviation, median, p90, solved rate, and paired learned-minus-MRV node differences with a bootstrap confidence interval. Wall time is secondary to structural search metrics because Python and runner noise can dominate small instances.

## Cost accounting

Search nodes alone can make an expensive learned/expert rule look artificially attractive. Therefore the custom solver records learned candidate evaluations and expert lookahead probes separately from nodes/backtracks/propagations and wall time.

## Literature chronology

Classical foundations establish first-fail, weighted-degree, impact-based search, and general backtracking methodology. The modern ML line learns variable/value orderings, search-heuristic selection, or solver parameters. Recent work includes propagation-aware RL/value selection (Marty et al., 2024), search-tree-topology-based adaptive VOH selection (Xu et al., AAAI 2025), and the broad CP+ML survey by Cappart et al. (JAIR 2025).

The benchmark does not reproduce any of these systems. Its aim is to provide a compact independent implementation with explicit exactness, feasibility, split discipline, cost accounting, and CP-SAT API boundaries.

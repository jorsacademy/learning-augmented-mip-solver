from __future__ import annotations

import argparse

import numpy as np

from lamip import SolverConfig, generate_binary_packing, load_components, solve, solve_reference


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--vars", type=int, default=18)
    parser.add_argument("--constraints", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    learned = load_components(args.checkpoint)
    policies = {
        "classical": SolverConfig(
            presolve="full",
            cuts="efficacy",
            branching="most_fractional",
            primal="greedy",
        ),
        "augmented": SolverConfig(
            presolve="learned",
            cuts="learned",
            branching="learned",
            primal="learned",
        ),
        "expert_controls": SolverConfig(
            presolve="full",
            cuts="oracle",
            branching="strong",
            primal="greedy",
        ),
    }
    metrics: dict[str, dict[str, list[float]]] = {
        name: {"nodes": [], "lp_solves": [], "initial_incumbent": []} for name in policies
    }

    for offset in range(args.instances):
        problem = generate_binary_packing(args.vars, args.constraints, args.seed + offset)
        exact = solve_reference(problem, presolve=False)
        for name, config in policies.items():
            result = solve(problem, config=config, learned=learned)
            if not result.optimal:
                raise RuntimeError(f"{name} hit node limit")
            if not np.isclose(result.objective, exact.objective, atol=1e-7):
                raise RuntimeError(f"{name} objective mismatch")
            metrics[name]["nodes"].append(float(result.nodes_processed))
            metrics[name]["lp_solves"].append(float(result.lp_solves))
            metrics[name]["initial_incumbent"].append(float(result.initial_incumbent))

    for name, values in metrics.items():
        print(f"[{name}]")
        for metric, samples in values.items():
            mean, std = summarize(samples)
            print(f"{metric}: mean={mean:.3f} std={std:.3f}")


if __name__ == "__main__":
    main()

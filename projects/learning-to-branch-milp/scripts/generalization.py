from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ltb_milp.gnn import BipartiteBranchingGNN, gnn_branch_policy
from ltb_milp.models import BranchingMLP
from ltb_milp.policies import learned_branch_policy
from ltb_milp.problem import generate_binary_packing
from ltb_milp.solver import solve_branch_and_bound
from ltb_milp.statistics import summarize_samples


@dataclass(frozen=True)
class Scenario:
    name: str
    n_vars: int
    n_constraints: int
    tightness: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned branching under distribution shift.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--instances-per-seed", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002, 2003, 2004])
    parser.add_argument("--reliability-threshold", type=int, default=2)
    return parser.parse_args()


def _summary(values: list[float]) -> str:
    stats = summarize_samples(values)
    return (
        f"mean={stats.mean:.6f} std={stats.std:.6f} "
        f"ci95=±{stats.ci95:.6f} seeds={stats.n}"
    )


def main() -> None:
    args = parse_args()
    if args.instances_per_seed <= 0:
        raise ValueError("instances-per-seed must be positive")
    if args.reliability_threshold <= 0:
        raise ValueError("reliability-threshold must be positive")
    if args.checkpoint is None and args.gnn_checkpoint is None:
        raise ValueError("at least one learned checkpoint is required")

    policies: dict[str, object] = {
        "most_fractional": "most_fractional",
        "pseudocost": "pseudocost",
        "reliability": "reliability",
        "strong": "strong",
    }

    if args.checkpoint is not None:
        model = BranchingMLP()
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"])
        policies["learned_mlp"] = learned_branch_policy(model)

    if args.gnn_checkpoint is not None:
        payload = torch.load(args.gnn_checkpoint, map_location="cpu", weights_only=True)
        hidden_dim = int(payload.get("hidden_dim", 64))
        gnn_model = BipartiteBranchingGNN(hidden_dim=hidden_dim)
        gnn_model.load_state_dict(payload["model_state_dict"])
        policies["learned_gnn"] = gnn_branch_policy(gnn_model)

    scenarios = (
        Scenario("id_12v_5c", 12, 5, None),
        Scenario("larger_16v_5c", 16, 5, None),
        Scenario("larger_20v_8c", 20, 8, None),
        Scenario("tighter_12v_5c", 12, 5, 0.30),
        Scenario("looser_12v_5c", 12, 5, 0.70),
    )

    for scenario_index, scenario in enumerate(scenarios):
        seed_nodes = {name: [] for name in policies}
        seed_lp = {name: [] for name in policies}
        seed_seconds = {name: [] for name in policies}

        for seed in args.seeds:
            per_seed_nodes = {name: [] for name in policies}
            per_seed_lp = {name: [] for name in policies}
            per_seed_seconds = {name: [] for name in policies}

            for offset in range(args.instances_per_seed):
                problem = generate_binary_packing(
                    scenario.n_vars,
                    scenario.n_constraints,
                    seed=scenario_index * 10_000_000 + seed * 10_000 + offset,
                    tightness=scenario.tightness,
                )
                reference: float | None = None
                for name, policy in policies.items():
                    started = time.perf_counter()
                    result = solve_branch_and_bound(
                        problem,
                        policy=policy,
                        reliability_threshold=args.reliability_threshold,
                    )
                    elapsed = time.perf_counter() - started
                    if not result.optimal:
                        raise RuntimeError(f"{scenario.name}/{name} did not prove optimality")
                    if reference is None:
                        reference = result.objective
                    elif abs(result.objective - reference) > 1e-6:
                        raise RuntimeError("branching policies returned inconsistent optima")
                    per_seed_nodes[name].append(float(result.nodes_processed))
                    per_seed_lp[name].append(float(result.lp_solves))
                    per_seed_seconds[name].append(elapsed)

            for name in policies:
                seed_nodes[name].append(sum(per_seed_nodes[name]) / len(per_seed_nodes[name]))
                seed_lp[name].append(sum(per_seed_lp[name]) / len(per_seed_lp[name]))
                seed_seconds[name].append(
                    sum(per_seed_seconds[name]) / len(per_seed_seconds[name])
                )

        print(
            f"\n[{scenario.name}] vars={scenario.n_vars} "
            f"constraints={scenario.n_constraints} tightness={scenario.tightness}"
        )
        for name in policies:
            print(f"{name}.nodes: {_summary(seed_nodes[name])}")
            print(f"{name}.lp_solves: {_summary(seed_lp[name])}")
            print(f"{name}.wall_seconds: {_summary(seed_seconds[name])}")


if __name__ == "__main__":
    main()

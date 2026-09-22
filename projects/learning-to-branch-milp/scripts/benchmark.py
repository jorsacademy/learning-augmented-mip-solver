from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ltb_milp.dataset import collect_root_strong_branching_dataset
from ltb_milp.gnn import BipartiteBranchingGNN, gnn_branch_policy, gnn_top1_expert_agreement
from ltb_milp.graph_dataset import collect_tree_graph_dataset
from ltb_milp.models import BranchingMLP
from ltb_milp.policies import learned_branch_policy
from ltb_milp.problem import generate_binary_packing
from ltb_milp.solver import solve_branch_and_bound
from ltb_milp.statistics import paired_deltas, paired_percent_reductions, summarize_samples
from ltb_milp.training import top1_expert_agreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-seed benchmark of MILP branching policies.")
    parser.add_argument("--instances-per-seed", type=int, default=20)
    parser.add_argument("--vars", type=int, default=12)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--mse-checkpoint", type=Path)
    parser.add_argument("--listwise-checkpoint", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--agreement-instances", type=int, default=30)
    parser.add_argument("--reliability-threshold", type=int, default=2)
    return parser.parse_args()


def load_mlp_policy(path: Path):
    model = BranchingMLP()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return model, learned_branch_policy(model)


def load_gnn_policy(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hidden_dim = int(payload.get("hidden_dim", 64))
    model = BipartiteBranchingGNN(hidden_dim=hidden_dim)
    model.load_state_dict(payload["model_state_dict"])
    return model, gnn_branch_policy(model)


def print_summary(name: str, values: list[float]) -> None:
    summary = summarize_samples(values)
    print(
        f"{name}: mean={summary.mean:.6f} std={summary.std:.6f} "
        f"ci95=±{summary.ci95:.6f} seeds={summary.n}"
    )


def print_paired_node_comparison(
    reference_name: str,
    comparison_name: str,
    seed_nodes: dict[str, list[float]],
) -> None:
    deltas = paired_deltas(seed_nodes[reference_name], seed_nodes[comparison_name])
    reductions = paired_percent_reductions(
        seed_nodes[reference_name], seed_nodes[comparison_name]
    )
    print(f"\n[{comparison_name}_vs_{reference_name}]")
    print_summary("paired_node_delta", deltas)
    print_summary("paired_node_reduction_percent", reductions)


def main() -> None:
    args = parse_args()
    if args.instances_per_seed <= 0 or args.agreement_instances <= 0:
        raise ValueError("instance counts must be positive")
    if args.reliability_threshold <= 0:
        raise ValueError("reliability-threshold must be positive")
    if not args.seeds:
        raise ValueError("at least one seed is required")

    policies: dict[str, object] = {
        "most_fractional": "most_fractional",
        "pseudocost": "pseudocost",
        "reliability": "reliability",
        "strong": "strong",
    }
    mlp_models: dict[str, BranchingMLP] = {}
    gnn_model: BipartiteBranchingGNN | None = None

    if args.mse_checkpoint:
        model, policy = load_mlp_policy(args.mse_checkpoint)
        mlp_models["learned_mse"] = model
        policies["learned_mse"] = policy
    if args.listwise_checkpoint:
        model, policy = load_mlp_policy(args.listwise_checkpoint)
        mlp_models["learned_listwise"] = model
        policies["learned_listwise"] = policy
    if args.gnn_checkpoint:
        gnn_model, policy = load_gnn_policy(args.gnn_checkpoint)
        policies["learned_gnn"] = policy

    seed_nodes = {name: [] for name in policies}
    seed_lp_solves = {name: [] for name in policies}
    seed_seconds = {name: [] for name in policies}
    seed_objectives = {name: [] for name in policies}

    for seed in args.seeds:
        per_seed_nodes = {name: [] for name in policies}
        per_seed_lp = {name: [] for name in policies}
        per_seed_seconds = {name: [] for name in policies}
        per_seed_objectives = {name: [] for name in policies}

        for offset in range(args.instances_per_seed):
            problem = generate_binary_packing(
                args.vars,
                args.constraints,
                seed=seed * 100_000 + offset,
            )
            reference_objective: float | None = None
            for name, policy in policies.items():
                started = time.perf_counter()
                result = solve_branch_and_bound(
                    problem,
                    policy=policy,
                    reliability_threshold=args.reliability_threshold,
                )
                elapsed = time.perf_counter() - started
                if not result.optimal:
                    raise RuntimeError(f"{name} hit the node limit before proving optimality")

                per_seed_nodes[name].append(float(result.nodes_processed))
                per_seed_lp[name].append(float(result.lp_solves))
                per_seed_seconds[name].append(elapsed)
                per_seed_objectives[name].append(result.objective)

                if reference_objective is None:
                    reference_objective = result.objective
                elif abs(result.objective - reference_objective) > 1e-6:
                    raise RuntimeError("branching policies returned inconsistent optima")

        for name in policies:
            seed_nodes[name].append(sum(per_seed_nodes[name]) / len(per_seed_nodes[name]))
            seed_lp_solves[name].append(sum(per_seed_lp[name]) / len(per_seed_lp[name]))
            seed_seconds[name].append(sum(per_seed_seconds[name]) / len(per_seed_seconds[name]))
            seed_objectives[name].append(
                sum(per_seed_objectives[name]) / len(per_seed_objectives[name])
            )

    print("Repeated-seed benchmark")
    print(f"seeds={args.seeds} instances_per_seed={args.instances_per_seed}")
    print(f"reliability_threshold={args.reliability_threshold}")
    print("confidence_intervals=Student-t 95% over seed-level replicates")
    for name in policies:
        print(f"\n[{name}]")
        print_summary("nodes_processed", seed_nodes[name])
        print_summary("lp_solves", seed_lp_solves[name])
        print_summary("wall_seconds_per_instance", seed_seconds[name])
        print_summary("objective", seed_objectives[name])

    for name in policies:
        if name != "most_fractional":
            print_paired_node_comparison("most_fractional", name, seed_nodes)

    if "reliability" in policies:
        for name in policies:
            if name not in {"most_fractional", "reliability"}:
                print_paired_node_comparison("reliability", name, seed_nodes)

    if mlp_models or gnn_model is not None:
        print("\n[heldout_strong_branching_agreement]")
        for name, model in mlp_models.items():
            agreements: list[float] = []
            for seed in args.seeds:
                dataset = collect_root_strong_branching_dataset(
                    args.agreement_instances,
                    args.vars,
                    args.constraints,
                    seed=seed * 100_000 + 50_000,
                )
                agreements.append(top1_expert_agreement(model, dataset))
            print_summary(name, agreements)

        if gnn_model is not None:
            agreements = []
            for seed in args.seeds:
                states = collect_tree_graph_dataset(
                    args.agreement_instances,
                    args.vars,
                    args.constraints,
                    seed=seed * 100_000 + 50_000,
                    max_nodes_per_instance=8,
                )
                agreements.append(gnn_top1_expert_agreement(gnn_model, states))
            print_summary("learned_gnn", agreements)


if __name__ == "__main__":
    main()

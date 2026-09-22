from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ltb_milp.dataset import (
    collect_root_strong_branching_dataset,
    collect_tree_strong_branching_dataset,
)
from ltb_milp.models import BranchingMLP
from ltb_milp.training import top1_expert_agreement, train_branching_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP to imitate strong branching.")
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--vars", type=int, default=12)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=["tree", "root"], default="tree")
    parser.add_argument("--loss", choices=["listwise", "mse"], default="listwise")
    parser.add_argument("--max-nodes-per-instance", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/branching_mlp.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.dataset == "root":
        dataset = collect_root_strong_branching_dataset(
            args.instances,
            args.vars,
            args.constraints,
            seed=args.seed,
        )
    else:
        dataset = collect_tree_strong_branching_dataset(
            args.instances,
            args.vars,
            args.constraints,
            seed=args.seed,
            max_nodes_per_instance=args.max_nodes_per_instance,
        )

    model = BranchingMLP()
    history = train_branching_model(
        model,
        dataset,
        loss=args.loss,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    agreement = top1_expert_agreement(model, dataset)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.checkpoint)
    print(
        f"dataset={args.dataset} loss={args.loss} nodes={len(dataset.group_sizes)} "
        f"samples={len(dataset.targets)} final_loss={history[-1]:.6f} "
        f"top1_agreement={agreement:.6f}"
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

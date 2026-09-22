from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ltb_milp.gnn import BipartiteBranchingGNN, gnn_top1_expert_agreement, train_gnn_branching_model
from ltb_milp.graph_dataset import collect_tree_graph_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bipartite GNN to imitate strong branching.")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--vars", type=int, default=12)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--max-nodes-per-instance", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/branching_gnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    states = collect_tree_graph_dataset(
        args.instances,
        args.vars,
        args.constraints,
        seed=args.seed,
        max_nodes_per_instance=args.max_nodes_per_instance,
    )
    model = BipartiteBranchingGNN(hidden_dim=args.hidden_dim)
    history = train_gnn_branching_model(
        model,
        states,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    agreement = gnn_top1_expert_agreement(model, states)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
        },
        args.checkpoint,
    )
    print(
        f"states={len(states)} final_loss={history[-1]:.6f} "
        f"top1_agreement={agreement:.6f}"
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

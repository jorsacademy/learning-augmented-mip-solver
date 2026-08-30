from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lamip.training import checkpoint_payload, collect_training_data, train_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--vars", type=int, default=18)
    parser.add_argument("--constraints", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/lamip.pt")
    args = parser.parse_args()

    data = collect_training_data(
        n_instances=args.instances,
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
    )
    components, losses = train_components(
        data,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(components, args.hidden_dim), path)
    for name, value in losses.items():
        print(f"{name}_loss={value:.6f}")
    print(f"checkpoint={path}")


if __name__ == "__main__":
    main()

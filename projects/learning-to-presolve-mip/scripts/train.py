from __future__ import annotations

import argparse
import json

import torch

from learning_to_presolve import (
    CONFIGS,
    PresolveSelector,
    collect_dataset,
    evaluate_selector,
    save_checkpoint,
    train_selector,
)


def label_counts(labels: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(labels, minlength=len(CONFIGS))
    return {config: int(counts[index]) for index, config in enumerate(CONFIGS)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a safe presolve-configuration selector")
    parser.add_argument("--instances", type=int, default=180)
    parser.add_argument("--validation-instances", type=int, default=60)
    parser.add_argument("--vars", type=int, default=32)
    parser.add_argument("--constraints", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="checkpoints/presolve_selector.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    training = collect_dataset(
        n_instances=args.instances,
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed,
    )
    validation = collect_dataset(
        n_instances=args.validation_instances,
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed=args.seed + 100_000,
    )

    model = PresolveSelector(hidden_dim=args.hidden_dim)
    initial_loss, final_loss = train_selector(
        model,
        training,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_metrics = evaluate_selector(model, training)
    validation_metrics = evaluate_selector(model, validation)
    save_checkpoint(model, args.checkpoint)

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "training_instances": training.size,
                "validation_instances": validation.size,
                "training_label_counts": label_counts(training.labels),
                "validation_label_counts": label_counts(validation.labels),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "training_accuracy": train_metrics.accuracy,
                "training_mean_normalized_regret": train_metrics.mean_normalized_regret,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_mean_normalized_regret": validation_metrics.mean_normalized_regret,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

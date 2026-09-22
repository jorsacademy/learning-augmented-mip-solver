from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from neural_diving.data import solved_dataset
from neural_diving.model import save_predictor, train_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-instances", type=int, default=28)
    parser.add_argument("--validation-instances", type=int, default=10)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--out", default="artifacts/predictor.pt")
    parser.add_argument("--metrics-out", default="artifacts/training_metrics.json")
    args = parser.parse_args()

    train_seeds = list(range(args.seed, args.seed + args.train_instances))
    val_start = args.seed + 10_000
    val_seeds = list(range(val_start, val_start + args.validation_instances))
    x_train, y_train, _ = solved_dataset(
        train_seeds, n_items=args.items, n_constraints=args.constraints
    )
    x_val, y_val, _ = solved_dataset(val_seeds, n_items=args.items, n_constraints=args.constraints)

    model, metrics = train_predictor(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=args.seed,
        epochs=args.epochs,
    )
    save_predictor(model, args.out)
    payload = {
        "metrics": asdict(metrics),
        "counts": {
            "train_instances": args.train_instances,
            "validation_instances": args.validation_instances,
            "train_variables": int(len(y_train)),
            "validation_variables": int(len(y_val)),
        },
        "config": vars(args),
    }
    path = Path(args.metrics_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

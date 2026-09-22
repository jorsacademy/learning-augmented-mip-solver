from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ltp_bnb.training import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train node-selection/pruning imitation policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-instances", type=int, default=80)
    parser.add_argument("--validation-instances", type=int, default=20)
    parser.add_argument("--items", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("artifacts/policy.json"))
    parser.add_argument(
        "--metrics-out", type=Path, default=Path("artifacts/training_metrics.json")
    )
    args = parser.parse_args()

    policy, metrics, counts = train_policy(
        seed=args.seed,
        train_instances=args.train_instances,
        validation_instances=args.validation_instances,
        n_items=args.items,
    )
    policy.save(args.out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": asdict(metrics), "counts": counts, "config": vars(args).copy()}
    payload["config"]["out"] = str(args.out)
    payload["config"]["metrics_out"] = str(args.metrics_out)
    args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

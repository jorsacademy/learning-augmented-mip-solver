from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltp_bnb.evaluation import benchmark_policy
from ltp_bnb.policy import ImitationPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact selection and approximate pruning."
    )
    parser.add_argument("--policy", type=Path, default=Path("artifacts/policy.json"))
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--items", type=int, default=14)
    parser.add_argument("--prune-threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()

    policy = ImitationPolicy.load(args.policy)
    payload = benchmark_policy(
        policy,
        seed=args.seed,
        instances=args.instances,
        n_items=args.items,
        prune_threshold=args.prune_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

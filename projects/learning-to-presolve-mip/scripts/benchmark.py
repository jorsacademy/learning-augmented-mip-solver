from __future__ import annotations

import argparse
import json

from learning_to_presolve import benchmark_selector, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark presolve configuration policies")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=80)
    parser.add_argument("--vars", type=int, default=32)
    parser.add_argument("--constraints", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=300_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    result = benchmark_selector(
        model,
        n_instances=args.instances,
        n_vars=args.vars,
        n_constraints=args.constraints,
        seed_start=args.seed_start,
    )
    payload = {
        "instances": args.instances,
        "label_counts": result.label_counts,
        "policies": {
            summary.policy: {
                "mean_work_proxy": summary.mean_work_proxy,
                "mean_nodes": summary.mean_nodes,
                "mean_residual_size": summary.mean_residual_size,
                "mean_residual_nonzeros": summary.mean_residual_nnz,
                "mean_presolve_checks": summary.mean_checks,
                "mean_normalized_regret": summary.mean_normalized_regret,
                "oracle_match_rate": summary.oracle_match_rate,
            }
            for summary in result.summaries
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"instances={args.instances}")
    print(f"oracle_label_counts={result.label_counts}")
    for summary in result.summaries:
        print(
            f"{summary.policy:10s} "
            f"work={summary.mean_work_proxy:9.3f} "
            f"nodes={summary.mean_nodes:6.2f} "
            f"size={summary.mean_residual_size:7.2f} "
            f"nnz={summary.mean_residual_nnz:8.2f} "
            f"checks={summary.mean_checks:9.2f} "
            f"regret={summary.mean_normalized_regret:8.4f} "
            f"match={summary.oracle_match_rate:7.3f}"
        )


if __name__ == "__main__":
    main()

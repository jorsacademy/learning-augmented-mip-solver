from __future__ import annotations

import argparse
import json

from .core import CONFIGS, generate_packing, instance_features, oracle_configuration
from .io import load_checkpoint
from .model import predict_configuration


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect safe presolve configurations on one MIP")
    parser.add_argument("--vars", type=int, default=32)
    parser.add_argument("--constraints", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    problem = generate_packing(args.vars, args.constraints, args.seed)
    oracle = oracle_configuration(problem)
    payload: dict[str, object] = {
        "instance": {
            "variables": problem.n_vars,
            "constraints": problem.n_constraints,
            "nonzeros": problem.nnz,
            "seed": args.seed,
        },
        "oracle_configuration": oracle.selected,
        "configurations": {},
    }
    configs_payload: dict[str, object] = {}
    for config, evaluation in zip(CONFIGS, oracle.evaluations, strict=True):
        configs_payload[config] = {
            "objective": evaluation.objective,
            "nodes": evaluation.nodes,
            "residual_variables": evaluation.residual_vars,
            "residual_constraints": evaluation.residual_rows,
            "residual_nonzeros": evaluation.residual_nnz,
            "presolve_checks": evaluation.checks,
            "work_proxy": evaluation.work_proxy,
        }
    payload["configurations"] = configs_payload
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
        payload["learned_configuration"] = predict_configuration(model, instance_features(problem))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

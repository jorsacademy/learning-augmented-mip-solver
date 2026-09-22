from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_diving.diving import neural_dive, random_dive
from neural_diving.features import extract_variable_features
from neural_diving.model import load_predictor, predict_probabilities
from neural_diving.problem import generate_instance
from neural_diving.solver import solve_mip


def relative_gap(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate) / max(abs(reference), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/predictor.pt")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--items", type=int, default=28)
    parser.add_argument("--constraints", type=int, default=6)
    parser.add_argument("--stochastic-samples", type=int, default=3)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()

    model = load_predictor(args.model)
    rows = []
    for offset in range(args.instances):
        instance = generate_instance(
            args.seed + offset,
            n_items=args.items,
            n_constraints=args.constraints,
        )
        reference = solve_mip(instance, mip_rel_gap=0.0, time_limit=20.0)
        if not reference.success or reference.objective is None:
            raise RuntimeError(f"reference solve failed on {instance.name}")
        probs = predict_probabilities(model, extract_variable_features(instance))
        neural = neural_dive(
            instance,
            probs,
            stochastic_samples=args.stochastic_samples,
            seed=args.seed + offset,
            time_limit=10.0,
        )
        random = random_dive(
            instance,
            fraction=0.70,
            seed=args.seed + offset,
            samples=1 + args.stochastic_samples,
            time_limit=10.0,
        )
        if not neural.solve.success or neural.solve.objective is None:
            raise RuntimeError("neural dive failed even after full-MIP fallback")
        if not random.solve.success or random.solve.objective is None:
            raise RuntimeError("random dive failed even after full-MIP fallback")
        rows.append(
            {
                "instance": instance.name,
                "reference_objective": reference.objective,
                "reference_seconds": reference.elapsed_seconds,
                "reference_nodes": reference.mip_node_count,
                "neural_objective": neural.solve.objective,
                "neural_gap": relative_gap(reference.objective, neural.solve.objective),
                "neural_total_seconds": neural.elapsed_seconds,
                "neural_best_submip_seconds": neural.solve.elapsed_seconds,
                "neural_fraction_used": neural.fraction_used,
                "neural_fixed_count": neural.fixed_count,
                "neural_fallback": neural.fallback_used,
                "random_objective": random.solve.objective,
                "random_total_seconds": random.elapsed_seconds,
                "random_gap": relative_gap(reference.objective, random.solve.objective),
                "random_fraction_used": random.fraction_used,
                "random_fixed_count": random.fixed_count,
                "random_fallback": random.fallback_used,
            }
        )

    payload = {
        "summary": {
            "mean_neural_gap": float(np.mean([r["neural_gap"] for r in rows])),
            "max_neural_gap": float(np.max([r["neural_gap"] for r in rows])),
            "mean_random_gap": float(np.mean([r["random_gap"] for r in rows])),
            "max_random_gap": float(np.max([r["random_gap"] for r in rows])),
            "mean_reference_seconds": float(np.mean([r["reference_seconds"] for r in rows])),
            "mean_neural_total_seconds": float(
                np.mean([r["neural_total_seconds"] for r in rows])
            ),
            "mean_random_total_seconds": float(
                np.mean([r["random_total_seconds"] for r in rows])
            ),
            "mean_neural_fixed_fraction": float(
                np.mean([r["neural_fixed_count"] / args.items for r in rows])
            ),
            "neural_fallback_rate": float(np.mean([r["neural_fallback"] for r in rows])),
            "random_fallback_rate": float(np.mean([r["random_fallback"] for r in rows])),
        },
        "config": vars(args),
        "instances": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

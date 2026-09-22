from __future__ import annotations

import argparse

from gap_gnn import (
    GAPMILP,
    generate_gap_instance,
    generate_labeled_dataset,
    train_gnn,
    evaluate_model_on_instances,
    calibrate_confidence_threshold,
)
from gap_gnn.graph import instance_to_features
from gap_gnn.model import BipartiteGNN
from gap_gnn.inference import propose_confident_fixings, exact_repair_with_relaxation


def self_test():
    instance = generate_gap_instance(seed=5, n_machines=4, n_jobs=8)
    exact = GAPMILP(instance).solve()
    assert exact is not None and exact.status == "OPTIMAL"
    assert exact.max_violation <= 1e-7

    mf, jf, ef = instance_to_features(instance)
    model = BipartiteGNN(hidden_dim=24, layers=2)
    logits = model(mf[None], jf[None], ef[None])
    assert logits.shape == (1, 4, 8)

    proposal = propose_confident_fixings(
        model,
        instance,
        confidence_threshold=0.0,
        margin_threshold=-1.0,
    )
    repaired = exact_repair_with_relaxation(
        instance,
        proposal,
        optimal_assignment=exact.assignment,
    )
    assert repaired.solution.max_violation <= 1e-7
    print("GNN-guided GAP variable-fixing self-test: OK")


def run_experiment(args):
    train = generate_labeled_dataset(
        args.train_instances,
        seed=args.seed,
        n_machines=args.machines,
        n_jobs=args.jobs,
    )
    validation = generate_labeled_dataset(
        args.validation_instances,
        seed=args.seed + 100_000,
        n_machines=args.machines,
        n_jobs=args.jobs,
    )
    test = generate_labeled_dataset(
        args.test_instances,
        seed=args.seed + 200_000,
        n_machines=args.machines,
        n_jobs=args.jobs,
    )

    trained = train_gnn(
        train,
        validation,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )

    calibration = calibrate_confidence_threshold(
        trained.model,
        validation.instances,
        validation.labels,
        target_precision=args.target_fix_precision,
        margin_threshold=args.margin_threshold,
        minimum_coverage=args.minimum_fix_coverage,
    )
    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else calibration.confidence_threshold
    )

    result = evaluate_model_on_instances(
        trained.model,
        test.instances,
        confidence_threshold=threshold,
        margin_threshold=args.margin_threshold,
    )

    print("=" * 105)
    print("GNN-GUIDED GENERALIZED ASSIGNMENT — CONFIDENCE FIXING + EXACT MILP REPAIR")
    print("=" * 105)
    print(f"best validation assignment accuracy : {trained.best_validation_accuracy:.3f}")
    print(
        f"calibrated confidence threshold     : {threshold:.3f} "
        f"(validation precision={calibration.validation_precision:.3f}, "
        f"coverage={calibration.validation_coverage:.3f})"
    )
    print(f"test assignment accuracy            : {result.mean('accuracy'):.3f}")
    print(f"mean edge confidence                : {result.mean('mean_confidence'):.3f}")
    print(f"mean initially fixed jobs           : {result.mean('initially_fixed'):.2f}/{args.jobs}")
    print(f"mean finally fixed jobs             : {result.mean('finally_fixed'):.2f}/{args.jobs}")
    print(f"mean wrong initial fixes            : {result.mean('wrong_fixes'):.3f}")
    print(f"mean relaxed fixings                : {result.mean('relaxed_fixes'):.3f}")
    print(f"mean repair objective gap           : {result.mean('repair_gap_pct'):.3f}%")
    print(f"mean greedy objective gap           : {result.mean('greedy_gap_pct'):.3f}%")
    print()
    print(f"vanilla MILP nodes                  : {result.mean('vanilla_nodes'):.2f}")
    print(f"GNN repair MILP nodes               : {result.mean('repair_nodes'):.2f}")
    print(f"oracle-fixing MILP nodes            : {result.mean('oracle_fix_nodes'):.2f}")
    print(f"vanilla solve time                  : {result.mean('vanilla_seconds'):.4f}s")
    print(f"GNN repair solve time               : {result.mean('repair_seconds'):.4f}s")
    print(f"oracle-fixing solve time            : {result.mean('oracle_fix_seconds'):.4f}s")
    print()
    print(
        "The GNN is a primal guidance mechanism, not an exact solver. "
        "The repair MILP is exact conditional on the surviving fixings."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--machines", type=int, default=6)
    p.add_argument("--jobs", type=int, default=18)
    p.add_argument("--train-instances", type=int, default=96)
    p.add_argument("--validation-instances", type=int, default=24)
    p.add_argument("--test-instances", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--margin-threshold", type=float, default=0.05)
    p.add_argument("--target-fix-precision", type=float, default=0.95)
    p.add_argument("--minimum-fix-coverage", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(args)

import itertools
import unittest

import numpy as np
import torch

from gap_gnn.graph import batch_features
from gap_gnn.inference import (
    FixingProposal,
    calibrate_confidence_threshold,
    exact_repair_with_relaxation,
    propose_confident_fixings,
)
from gap_gnn.model import BipartiteGNN
from gap_gnn.planning import GAPMILP, generate_gap_instance
from gap_gnn.training import generate_labeled_dataset, train_gnn


def brute_force_gap(instance):
    M, J = instance.n_machines, instance.n_jobs
    best_obj = float("inf")
    best = None
    for assignment in itertools.product(range(M), repeat=J):
        load = np.zeros(M)
        obj = 0.0
        feasible = True
        for j, m in enumerate(assignment):
            load[m] += instance.resources[m, j]
            if load[m] > instance.capacities[m] + 1e-9:
                feasible = False
                break
            obj += instance.costs[m, j]
        if feasible and obj < best_obj:
            best_obj = obj
            best = np.asarray(assignment, dtype=int)
    return best, best_obj


class GNNGuidedGAPTests(unittest.TestCase):
    def test_exact_milp_matches_tiny_bruteforce_oracle(self):
        instance = generate_gap_instance(
            seed=11,
            n_machines=3,
            n_jobs=6,
            capacity_slack=0.12,
        )
        solution = GAPMILP(instance).solve()
        assignment, objective = brute_force_gap(instance)
        self.assertIsNotNone(assignment)
        self.assertAlmostEqual(solution.objective, objective, places=6)

    def test_postsolve_feasibility_audit(self):
        instance = generate_gap_instance(seed=12, n_machines=4, n_jobs=10)
        solution = GAPMILP(instance).solve()
        self.assertEqual(solution.status, "OPTIMAL")
        self.assertLessEqual(solution.max_violation, 1e-7)

    def test_instance_generation_reproducible(self):
        a = generate_gap_instance(seed=13, n_machines=5, n_jobs=12)
        b = generate_gap_instance(seed=13, n_machines=5, n_jobs=12)
        np.testing.assert_array_equal(a.costs, b.costs)
        np.testing.assert_array_equal(a.resources, b.resources)
        np.testing.assert_array_equal(a.capacities, b.capacities)

    def test_bipartite_gnn_shape_and_gradient_flow(self):
        instances = [
            generate_gap_instance(seed=20+i, n_machines=4, n_jobs=9)
            for i in range(3)
        ]
        mf, jf, ef = batch_features(instances)
        model = BipartiteGNN(hidden_dim=24, layers=2)
        logits = model(mf, jf, ef)
        self.assertEqual(tuple(logits.shape), (3, 4, 9))
        labels = torch.zeros((3, 9), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        ))

    def test_labeled_dataset_uses_feasible_exact_assignments(self):
        data = generate_labeled_dataset(
            4,
            seed=30,
            n_machines=4,
            n_jobs=9,
        )
        for instance, label, optimum in zip(
            data.instances,
            data.labels,
            data.objectives,
        ):
            objective = float(sum(
                instance.costs[label[j], j]
                for j in range(instance.n_jobs)
            ))
            self.assertAlmostEqual(objective, optimum, places=6)

    def test_confidence_calibration_meets_declared_precision_when_available(self):
        train = generate_labeled_dataset(12, seed=40, n_machines=4, n_jobs=8)
        val = generate_labeled_dataset(6, seed=50, n_machines=4, n_jobs=8)
        trained = train_gnn(
            train,
            val,
            seed=4,
            epochs=3,
            batch_size=6,
            hidden_dim=24,
            layers=2,
        )
        calibration = calibrate_confidence_threshold(
            trained.model,
            val.instances,
            val.labels,
            target_precision=0.70,
            margin_threshold=0.0,
            minimum_coverage=0.02,
        )
        if calibration.fixed_predictions > 0:
            self.assertGreaterEqual(calibration.validation_precision, 0.70 - 1e-12)
            self.assertGreater(calibration.validation_coverage, 0.0)

    def test_infeasible_fixings_are_relaxed_until_exact_repair_is_feasible(self):
        instance = generate_gap_instance(seed=60, n_machines=4, n_jobs=10)
        exact = GAPMILP(instance).solve()
        proposal = FixingProposal(
            fixed_jobs={j: 0 for j in range(instance.n_jobs)},
            confidences={j: 0.50 + 0.01*j for j in range(instance.n_jobs)},
            predicted_assignment=np.zeros(instance.n_jobs, dtype=int),
            probabilities=np.full((instance.n_machines, instance.n_jobs), 0.25),
        )
        repaired = exact_repair_with_relaxation(
            instance,
            proposal,
            optimal_assignment=exact.assignment,
        )
        self.assertGreater(repaired.relaxed_jobs, 0)
        self.assertLessEqual(repaired.solution.max_violation, 1e-7)

    def test_short_training_and_guided_repair_smoke(self):
        train = generate_labeled_dataset(16, seed=70, n_machines=4, n_jobs=9)
        val = generate_labeled_dataset(6, seed=80, n_machines=4, n_jobs=9)
        trained = train_gnn(
            train,
            val,
            seed=7,
            epochs=3,
            batch_size=8,
            hidden_dim=24,
            layers=2,
        )
        instance = generate_gap_instance(seed=999, n_machines=4, n_jobs=9)
        exact = GAPMILP(instance).solve()
        proposal = propose_confident_fixings(
            trained.model,
            instance,
            confidence_threshold=0.0,
            margin_threshold=-1.0,
        )
        repaired = exact_repair_with_relaxation(
            instance,
            proposal,
            optimal_assignment=exact.assignment,
        )
        self.assertLessEqual(repaired.solution.max_violation, 1e-7)
        self.assertGreaterEqual(
            repaired.solution.objective + 1e-7,
            exact.objective,
        )


if __name__ == "__main__":
    unittest.main()

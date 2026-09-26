import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from lamip.problem import generate_binary_packing

ROOT = Path(__file__).resolve().parents[1]


def load(rel_path: str, name: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_node_search_feature_and_ranking_pipeline() -> None:
    mod = load(
        "projects/learning-to-search-bnb-nodes/node_selection.py",
        "node_selection_project",
    )
    states = [
        mod.NodeState(0, 10.0, 4.0, 5, 0),
        mod.NodeState(2, 8.0, 4.0, 2, 3),
        mod.NodeState(1, 9.0, 4.0, 4, 1),
    ]
    X = np.stack([mod.node_features(s) for s in states])
    y = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    model = mod.fit_priority_model(X, y, epochs=10, seed=0)
    ranking = mod.rank_nodes(model, states)
    assert sorted(ranking.tolist()) == [0, 1, 2]


def test_primal_heuristic_portfolio_is_feasible() -> None:
    mod = load(
        "projects/learning-to-select-primal-heuristics/selector.py",
        "heuristic_selector_project",
    )
    problem = generate_binary_packing(14, 5, seed=17)
    for name in mod.HEURISTICS:
        x, objective = mod.run_heuristic(problem, name)
        assert np.all(problem.A @ x <= problem.b + 1e-9)
        assert np.isclose(objective, problem.profits @ x)


def test_primal_selector_training_smoke() -> None:
    mod = load(
        "projects/learning-to-select-primal-heuristics/selector.py",
        "heuristic_selector_training_project",
    )
    X, y = mod.collect_training_data(n_instances=12, n_vars=10, n_constraints=4, seed=30)
    model = mod.train_selector(X, y, epochs=5, seed=0)
    problem = generate_binary_packing(10, 4, seed=99)
    assert mod.choose_heuristic(model, problem) in mod.HEURISTICS

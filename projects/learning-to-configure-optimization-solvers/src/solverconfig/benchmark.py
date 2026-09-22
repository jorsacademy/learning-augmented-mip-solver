from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .configurations import CANDIDATE_CONFIGURATIONS
from .features import FEATURE_NAMES, extract_features
from .instances import SetCoverInstance, generate_split
from .metrics import MethodMetrics, par10, summarize_method
from .selector import PortfolioCostSelector
from .solver import SolveResult, solve_with_highs


@dataclass(frozen=True)
class BenchmarkArtifacts:
    summary: dict[str, Any]
    metrics: list[MethodMetrics]
    raw_results: list[SolveResult]


def _instances_from_spec(split: str, spec: dict[str, Any]) -> list[SetCoverInstance]:
    return generate_split(
        split=split,
        count=int(spec["count"]),
        seed_start=int(spec["seed_start"]),
        row_range=tuple(spec["row_range"]),
        col_range=tuple(spec["col_range"]),
        density_range=tuple(spec["density_range"]),
        regimes=tuple(spec["regimes"]),
    )


def _evaluate_grid(
    instances: list[SetCoverInstance], *, time_limit: float, solver_seed: int
) -> tuple[np.ndarray, np.ndarray, list[float], list[SolveResult]]:
    features: list[np.ndarray] = []
    feature_times: list[float] = []
    all_results: list[SolveResult] = []
    costs = np.empty((len(instances), len(CANDIDATE_CONFIGURATIONS)), dtype=float)
    for i, instance in enumerate(instances):
        vector, feature_time = extract_features(instance)
        features.append(vector)
        feature_times.append(feature_time)
        for j, config in enumerate(CANDIDATE_CONFIGURATIONS):
            result = solve_with_highs(
                instance,
                config,
                time_limit=time_limit,
                random_seed=solver_seed,
            )
            all_results.append(result)
            costs[i, j] = par10(result, time_limit)
    return np.vstack(features), costs, feature_times, all_results


def _result_matrix(results: list[SolveResult], instance_ids: list[str]) -> list[list[SolveResult]]:
    by_key = {(r.instance_id, r.configuration): r for r in results}
    return [
        [by_key[(instance_id, config.name)] for config in CANDIDATE_CONFIGURATIONS]
        for instance_id in instance_ids
    ]


def _select_rows(matrix: list[list[SolveResult]], indices: np.ndarray) -> list[SolveResult]:
    return [row[int(idx)] for row, idx in zip(matrix, indices, strict=True)]


def run_benchmark(config_path: str | Path, output_dir: str | Path) -> BenchmarkArtifacts:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    time_limit = float(config["solver"]["time_limit_seconds"])
    solver_seed = int(config["solver"]["random_seed"])

    splits = {name: _instances_from_spec(name, spec) for name, spec in config["splits"].items()}
    evaluated: dict[str, tuple[np.ndarray, np.ndarray, list[float], list[SolveResult]]] = {}
    for split_name in ("train", "validation", "test", "ood"):
        evaluated[split_name] = _evaluate_grid(
            splits[split_name], time_limit=time_limit, solver_seed=solver_seed
        )

    x_train, c_train, train_ft, train_results = evaluated["train"]
    x_val, c_val, val_ft, val_results = evaluated["validation"]
    x_test, c_test, test_ft, test_results = evaluated["test"]
    x_ood, c_ood, ood_ft, ood_results = evaluated["ood"]

    candidates = config["selector"]["hyperparameters"]
    best_params: dict[str, int] | None = None
    best_val_regret = float("inf")
    for candidate in candidates:
        selector = PortfolioCostSelector(
            tuple(c.name for c in CANDIDATE_CONFIGURATIONS),
            n_estimators=int(candidate["n_estimators"]),
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            random_state=int(config["selector"]["validation_seed"]),
        ).fit(x_train, c_train)
        val_indices, _ = selector.predict_indices(x_val)
        val_selected = c_val[np.arange(c_val.shape[0]), val_indices]
        val_oracle = c_val.min(axis=1)
        mean_regret = float(np.mean(val_selected - val_oracle))
        if mean_regret < best_val_regret:
            best_val_regret = mean_regret
            best_params = {
                "n_estimators": int(candidate["n_estimators"]),
                "min_samples_leaf": int(candidate["min_samples_leaf"]),
            }
    if best_params is None:
        raise RuntimeError("selector hyperparameter grid is empty")

    x_fit = np.vstack([x_train, x_val])
    c_fit = np.vstack([c_train, c_val])
    config_names = tuple(c.name for c in CANDIDATE_CONFIGURATIONS)
    test_matrix = _result_matrix(test_results, [x.instance_id for x in splits["test"]])
    ood_matrix = _result_matrix(ood_results, [x.instance_id for x in splits["ood"]])

    global_best_idx = int(np.argmin(c_fit.mean(axis=0)))
    default_idx = config_names.index("default")
    selector_seeds = [int(seed) for seed in config["selector"]["training_seeds"]]
    metrics: list[MethodMetrics] = []
    learned_prediction_seconds = 0.0

    def evaluate_split(
        split_name: str,
        features: np.ndarray,
        costs: np.ndarray,
        matrix: list[list[SolveResult]],
    ) -> None:
        nonlocal learned_prediction_seconds
        oracle_idx = np.argmin(costs, axis=1)
        oracle_cost = costs.min(axis=1)
        fixed_methods = {
            "default": np.full(costs.shape[0], default_idx),
            "global_best_single": np.full(costs.shape[0], global_best_idx),
            "portfolio_oracle": oracle_idx,
        }
        for method, indices in fixed_methods.items():
            metrics.append(
                summarize_method(
                    method=f"{split_name}:{method}",
                    selected=_select_rows(matrix, indices),
                    oracle_par10=oracle_cost,
                    time_limit=time_limit,
                    ci_seed=solver_seed,
                )
            )

        random_repeats = int(config["selector"]["random_baseline_repeats"])
        random_selected: list[SolveResult] = []
        random_oracle: list[float] = []
        for repeat in range(random_repeats):
            rng = np.random.default_rng(int(config["selector"]["random_baseline_seed"]) + repeat)
            indices = rng.integers(0, len(config_names), size=costs.shape[0])
            random_selected.extend(_select_rows(matrix, indices))
            random_oracle.extend(oracle_cost.tolist())
        metrics.append(
            summarize_method(
                method=f"{split_name}:random_config",
                selected=random_selected,
                oracle_par10=np.asarray(random_oracle),
                time_limit=time_limit,
                ci_seed=solver_seed + 1,
            )
        )

        for training_seed in selector_seeds:
            selector = PortfolioCostSelector(
                config_names,
                n_estimators=best_params["n_estimators"],
                min_samples_leaf=best_params["min_samples_leaf"],
                random_state=training_seed,
            ).fit(x_fit, c_fit)
            indices, prediction_seconds = selector.predict_indices(features)
            learned_prediction_seconds += prediction_seconds
            metrics.append(
                summarize_method(
                    method=f"{split_name}:learned_selector_seed_{training_seed}",
                    selected=_select_rows(matrix, indices),
                    oracle_par10=oracle_cost,
                    time_limit=time_limit,
                    ci_seed=training_seed,
                )
            )

    evaluate_split("test", x_test, c_test, test_matrix)
    evaluate_split("ood", x_ood, c_ood, ood_matrix)

    all_results = train_results + val_results + test_results + ood_results
    feature_seconds = float(sum(train_ft + val_ft + test_ft + ood_ft))
    total_instances = sum(len(v) for v in splits.values())
    summary = {
        "schema_version": 1,
        "problem_family": "synthetic_set_covering_milp",
        "candidate_configurations": [
            {"name": config.name, "options": config.options} for config in CANDIDATE_CONFIGURATIONS
        ],
        "feature_names": list(FEATURE_NAMES),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "time_limit_seconds": time_limit,
        "solver_calls": total_instances * len(CANDIDATE_CONFIGURATIONS),
        "feature_extraction_seconds": feature_seconds,
        "selector_prediction_seconds": learned_prediction_seconds,
        "selected_hyperparameters": best_params,
        "validation_mean_regret": best_val_regret,
        "global_best_single_configuration": config_names[global_best_idx],
        "claims_note": (
            "Smoke/benchmark outputs are empirical runs on synthetic instances, "
            "not paper-level claims."
        ),
    }

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in metrics)
    with (output / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in all_results:
            handle.write(json.dumps(result.to_dict()) + "\n")

    return BenchmarkArtifacts(summary, metrics, all_results)

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from .audit import audit_assignment
from .cpsat_control import solve_with_cpsat
from .expert import collect_expert_samples
from .instances import GraphColoringInstance, generate_split
from .learning import fit_random_forest, fit_ridge, top1_imitation_accuracy
from .oracle import brute_force_oracle
from .search import solve_coloring
from .statistics import paired_difference, summarize


def _make_split(spec: dict[str, Any], base_seed: int) -> list[GraphColoringInstance]:
    return generate_split(
        count=int(spec["count"]),
        n_vertices=int(spec["n_vertices"]),
        n_colors=int(spec["n_colors"]),
        edge_probability=float(spec["edge_probability"]),
        base_seed=base_seed,
        infeasible_fraction=float(spec.get("infeasible_fraction", 0.5)),
    )


def _collect(
    instances: list[GraphColoringInstance],
    *,
    max_states: int,
    node_limit: int,
    prefix: str,
):
    samples = []
    for i, instance in enumerate(instances):
        samples.extend(
            collect_expert_samples(
                instance,
                max_states=max_states,
                node_limit=node_limit,
                state_prefix=f"{prefix}-{i}",
            )
        )
    return samples


def _custom_record(instance_id: str, policy: str, result: Any) -> dict[str, Any]:
    row = asdict(result)
    row["instance_id"] = instance_id
    row["policy"] = policy
    row["assignment"] = list(result.assignment) if result.assignment is not None else None
    return row


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seeds = config["seeds"]
    train = _make_split(config["train"], int(seeds["train"]))
    validation = _make_split(config["validation"], int(seeds["validation"]))
    test = _make_split(config["test"], int(seeds["test"]))
    ood = _make_split(config["ood"], int(seeds["ood"]))

    collection = config["expert_collection"]
    train_samples = _collect(
        train,
        max_states=int(collection["max_states_per_instance"]),
        node_limit=int(collection["node_limit"]),
        prefix="train",
    )
    validation_samples = _collect(
        validation,
        max_states=int(collection["max_states_per_instance"]),
        node_limit=int(collection["node_limit"]),
        prefix="validation",
    )

    ridge = fit_ridge(train_samples)
    forest_seed = int(config["model"]["seed"])
    forest = fit_random_forest(train_samples, seed=forest_seed)
    val_accuracy = {
        "ridge": top1_imitation_accuracy(ridge, validation_samples),
        "random_forest": top1_imitation_accuracy(forest, validation_samples),
    }
    selected_name = max(val_accuracy, key=val_accuracy.get)
    selected_model = ridge if selected_name == "ridge" else forest

    custom_records: list[dict[str, Any]] = []
    policies = ["fixed", "degree", "mrv", "dom_degree", "random", "expert", "learned"]
    search_cfg = config["custom_search"]
    for split_name, instances in (("test", test), ("ood", ood)):
        for i, instance in enumerate(instances):
            instance_id = f"{split_name}-{i}"
            for policy in policies:
                result = solve_coloring(
                    instance,
                    policy=policy,
                    seed=int(seeds["policy"]) + i,
                    model=selected_model if policy == "learned" else None,
                    node_limit=int(search_cfg["node_limit"]),
                    time_limit_seconds=float(search_cfg["time_limit_seconds"]),
                )
                if result.assignment is not None:
                    if not audit_assignment(instance, result.assignment).feasible:
                        raise AssertionError("post-solve audit failed")
                row = _custom_record(instance_id, policy, result)
                row["split"] = split_name
                row["expected_feasible"] = instance.expected_feasible
                custom_records.append(row)

    custom_summary: dict[str, Any] = {}
    for split_name in ("test", "ood"):
        custom_summary[split_name] = {}
        for policy in policies:
            rows = [
                row
                for row in custom_records
                if row["split"] == split_name and row["policy"] == policy
            ]
            custom_summary[split_name][policy] = {
                "solved_rate": sum(row["status"] != "UNKNOWN" for row in rows) / len(rows),
                "nodes": summarize(row["nodes"] for row in rows),
                "backtracks": summarize(row["backtracks"] for row in rows),
                "wall_time_seconds": summarize(row["wall_time_seconds"] for row in rows),
                "policy_evaluations": summarize(row["policy_evaluations"] for row in rows),
                "expert_probes": summarize(row["expert_probes"] for row in rows),
            }

        learned = [
            row["nodes"]
            for row in custom_records
            if row["split"] == split_name and row["policy"] == "learned"
        ]
        mrv = [
            row["nodes"]
            for row in custom_records
            if row["split"] == split_name and row["policy"] == "mrv"
        ]
        custom_summary[split_name]["paired_learned_minus_mrv_nodes"] = paired_difference(
            learned, mrv, bootstrap_seed=int(seeds["statistics"])
        )

    cpsat_records: list[dict[str, Any]] = []
    cp_cfg = config["cp_sat"]
    for i, instance in enumerate(test):
        instance_id = f"test-{i}"
        for strategy in ("default", "fixed", "degree", "min_domain", "learned_root"):
            result = solve_with_cpsat(
                instance,
                strategy=strategy,
                seed=int(seeds["cp_sat"]) + i,
                time_limit_seconds=float(cp_cfg["time_limit_seconds"]),
                model=selected_model if strategy == "learned_root" else None,
            )
            row = asdict(result)
            row["instance_id"] = instance_id
            row["assignment"] = list(result.assignment) if result.assignment is not None else None
            cpsat_records.append(row)

    cpsat_summary: dict[str, Any] = {}
    for strategy in ("default", "fixed", "degree", "min_domain", "learned_root"):
        rows = [row for row in cpsat_records if row["strategy"] == strategy]
        cpsat_summary[strategy] = {
            "solved_rate": sum(
                row["status"] in {"OPTIMAL", "FEASIBLE", "INFEASIBLE"} for row in rows
            )
            / len(rows),
            "branches": summarize(row["branches"] for row in rows),
            "conflicts": summarize(row["conflicts"] for row in rows),
            "wall_time_seconds": summarize(row["wall_time_seconds"] for row in rows),
        }

    oracle_cfg = config["oracle"]
    oracle_instances = _make_split(oracle_cfg, int(seeds["oracle"]))
    oracle_checks = []
    for i, instance in enumerate(oracle_instances):
        oracle = brute_force_oracle(instance)
        solver = solve_coloring(instance, policy="mrv")
        solver_feasible = solver.status == "FEASIBLE"
        if solver.status == "UNKNOWN":
            raise AssertionError("unlimited custom solver cannot be UNKNOWN")
        if oracle.feasible != solver_feasible:
            raise AssertionError("custom solver disagrees with exhaustive oracle")
        oracle_checks.append(
            {
                "instance_id": f"oracle-{i}",
                "oracle_feasible": oracle.feasible,
                "solver_status": solver.status,
                "complete_assignments_checked": oracle.complete_assignments_checked,
            }
        )

    return {
        "schema_version": 1,
        "run_kind": config.get("run_kind", "benchmark"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "seeds": seeds,
        "learning": {
            "train_candidate_samples": len(train_samples),
            "validation_candidate_samples": len(validation_samples),
            "validation_top1_imitation_accuracy": val_accuracy,
            "selected_model": selected_name,
        },
        "custom_search": {
            "records": custom_records,
            "summary": custom_summary,
        },
        "cp_sat": {
            "records": cpsat_records,
            "summary": cpsat_summary,
            "control_surface": "public add_decision_strategy + FIXED_SEARCH only",
        },
        "oracle_checks": oracle_checks,
        "claims_boundary": {
            "smoke_is_scientific_benchmark": False,
            "learned_policy_changes_feasibility_region": False,
            "prediction_accuracy_equals_search_quality": False,
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_result(result: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

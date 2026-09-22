import pytest

pytest.importorskip("ortools")

from learning_cp_sat.experiment import load_config, run_experiment


def test_tiny_end_to_end_output_schema(tmp_path) -> None:
    config = load_config("configs/smoke.json")
    config["train"]["count"] = 2
    config["validation"]["count"] = 2
    config["test"]["count"] = 2
    config["ood"]["count"] = 2
    config["expert_collection"]["max_states_per_instance"] = 8
    result = run_experiment(config)

    assert result["schema_version"] == 1
    assert result["learning"]["selected_model"] in {"ridge", "random_forest"}
    assert result["oracle_checks"]
    assert set(result["custom_search"]["summary"]) == {"test", "ood"}
    assert "default" in result["cp_sat"]["summary"]

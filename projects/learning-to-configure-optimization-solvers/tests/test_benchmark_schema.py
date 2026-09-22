import json
from pathlib import Path

from solverconfig.benchmark import run_benchmark


def test_smoke_benchmark_writes_expected_schema(tmp_path: Path) -> None:
    artifacts = run_benchmark("configs/smoke.json", tmp_path)
    assert artifacts.summary["schema_version"] == 1
    assert artifacts.summary["solver_calls"] == 18 * 6
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "raw_results.jsonl").exists()
    reloaded = json.loads((tmp_path / "summary.json").read_text())
    assert reloaded["problem_family"] == "synthetic_set_covering_milp"

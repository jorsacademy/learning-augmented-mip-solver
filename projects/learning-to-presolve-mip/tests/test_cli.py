import json
import subprocess
import sys


def test_cli_emits_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "learning_to_presolve",
            "--vars",
            "16",
            "--constraints",
            "5",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["oracle_configuration"] in {"none", "fixing", "rows", "full"}
    assert set(payload["configurations"]) == {"none", "fixing", "rows", "full"}

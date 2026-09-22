import numpy as np

from solverconfig.instances import SetCoverInstance
from solverconfig.oracle import exhaustive_set_cover


def test_exhaustive_oracle_known_solution() -> None:
    inst = SetCoverInstance(
        "tiny",
        "uniform",
        np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8),
        np.array([2.0, 3.0, 4.0]),
        1,
    )
    result = exhaustive_set_cover(inst)
    assert result.objective == 4.0
    assert result.decision.tolist() == [0, 0, 1]

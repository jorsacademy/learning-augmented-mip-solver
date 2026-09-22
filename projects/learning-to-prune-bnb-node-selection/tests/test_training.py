import json

import numpy as np

from ltp_bnb.policy import ImitationPolicy, fit_imitation_policy
from ltp_bnb.training import collect_oracle_dataset, make_instances


def test_oracle_dataset_has_both_classes() -> None:
    x, y = collect_oracle_dataset(make_instances(10, count=6, n_items=9))
    assert x.shape[1] == 10
    assert set(np.unique(y)) == {0, 1}


def test_policy_roundtrip(tmp_path) -> None:
    x, y = collect_oracle_dataset(make_instances(100, count=8, n_items=9))
    policy = fit_imitation_policy(x, y)
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = ImitationPolicy.load(path)
    p1 = policy.selection_score(x[0])
    p2 = loaded.selection_score(x[0])
    assert np.isclose(p1, p2)
    payload = json.loads(path.read_text())
    assert payload["format_version"] == 1

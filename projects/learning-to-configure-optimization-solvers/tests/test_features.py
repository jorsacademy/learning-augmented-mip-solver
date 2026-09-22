import numpy as np

from solverconfig.features import FEATURE_NAMES, extract_features
from solverconfig.instances import generate_set_cover


def test_features_are_finite_and_fixed_schema() -> None:
    inst = generate_set_cover(
        instance_id="f",
        regime="degree_correlated",
        n_rows=12,
        n_cols=18,
        density=0.2,
        seed=9,
    )
    features, elapsed = extract_features(inst)
    assert len(features) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(features))
    assert elapsed >= 0

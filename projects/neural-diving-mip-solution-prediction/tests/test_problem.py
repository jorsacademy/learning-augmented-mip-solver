import numpy as np

from neural_diving.features import FEATURE_NAMES, extract_variable_features
from neural_diving.problem import generate_instance


def test_generation_is_deterministic_and_valid():
    a = generate_instance(7, n_items=12, n_constraints=3)
    b = generate_instance(7, n_items=12, n_constraints=3)
    assert np.allclose(a.c, b.c)
    assert np.allclose(a.A, b.A)
    assert np.allclose(a.b, b.b)


def test_features_have_expected_shape_and_no_solution_leakage():
    instance = generate_instance(8, n_items=13, n_constraints=4)
    features = extract_variable_features(instance)
    assert features.shape == (13, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))

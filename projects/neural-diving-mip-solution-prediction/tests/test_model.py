import numpy as np

from neural_diving.model import (
    load_predictor,
    predict_probabilities,
    save_predictor,
    train_predictor,
)


def test_predictor_trains_and_roundtrips(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 10)).astype(np.float32)
    y = (x[:, 0] + 0.8 * x[:, 1] > 0).astype(np.float32)
    model, metrics = train_predictor(x[:220], y[:220], x[220:], y[220:], seed=5, epochs=80)
    assert metrics.accuracy > 0.80
    before = predict_probabilities(model, x[220:230])
    path = tmp_path / "model.pt"
    save_predictor(model, path)
    loaded = load_predictor(path)
    after = predict_probabilities(loaded, x[220:230])
    assert np.allclose(before, after)

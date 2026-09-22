from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .expert import ExpertCandidateSample


def samples_to_xy(samples: Iterable[ExpertCandidateSample]) -> tuple[np.ndarray, np.ndarray]:
    rows = list(samples)
    if not rows:
        raise ValueError("no expert samples")
    x = np.vstack([sample.features for sample in rows])
    y = np.asarray([sample.target_score for sample in rows], dtype=np.float64)
    return x, y


def fit_ridge(samples: Iterable[ExpertCandidateSample]) -> Any:
    x, y = samples_to_xy(samples)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    return model


def fit_random_forest(samples: Iterable[ExpertCandidateSample], *, seed: int) -> Any:
    x, y = samples_to_xy(samples)
    model = RandomForestRegressor(
        n_estimators=96,
        max_depth=10,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, y)
    return model


def top1_imitation_accuracy(model: Any, samples: Iterable[ExpertCandidateSample]) -> float:
    grouped: dict[str, list[ExpertCandidateSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.state_id].append(sample)
    if not grouped:
        raise ValueError("no validation states")

    correct = 0
    for candidates in grouped.values():
        x = np.vstack([sample.features for sample in candidates])
        predictions = model.predict(x)
        predicted = int(np.argmax(predictions))
        if candidates[predicted].is_best:
            correct += 1
    return correct / len(grouped)

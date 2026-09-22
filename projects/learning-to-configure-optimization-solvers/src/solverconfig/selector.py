from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class PortfolioCostSelector:
    """Predict per-configuration PAR cost and choose the minimum predicted cost."""

    config_names: tuple[str, ...]
    n_estimators: int = 200
    min_samples_leaf: int = 2
    random_state: int = 0

    def fit(self, features: np.ndarray, costs: np.ndarray) -> PortfolioCostSelector:
        if costs.shape != (features.shape[0], len(self.config_names)):
            raise ValueError("cost matrix shape does not match features/configurations")
        self.models_: list[RandomForestRegressor] = []
        for config_idx in range(len(self.config_names)):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + config_idx,
                n_jobs=1,
            )
            model.fit(features, costs[:, config_idx])
            self.models_.append(model)
        return self

    def predict_indices(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        if not hasattr(self, "models_"):
            raise RuntimeError("selector must be fitted before prediction")
        start = perf_counter()
        predicted = np.column_stack([model.predict(features) for model in self.models_])
        return np.argmin(predicted, axis=1), perf_counter() - start

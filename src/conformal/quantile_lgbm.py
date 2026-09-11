"""Slim LightGBM quantile regressor (Week 3 port)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95)

WEEK3_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 164,
    "learning_rate": 0.01028783936636433,
    "num_leaves": 65,
    "min_child_samples": 32,
    "subsample": 0.6622097205938371,
    "colsample_bytree": 0.6613879171683648,
    "verbosity": -1,
}


def enforce_quantile_order(
    predictions: dict[float, np.ndarray],
    quantiles: Sequence[float],
) -> dict[float, np.ndarray]:
    """Row-wise sort so quantile predictions are monotonic."""
    q_list = sorted(quantiles)
    stacked = np.column_stack([np.asarray(predictions[q], dtype=float) for q in q_list])
    ordered = np.sort(stacked, axis=1)
    return {q: ordered[:, j] for j, q in enumerate(q_list)}


class QuantileLGBM:
    """Train one LightGBM quantile model per level (pinball / quantile loss)."""

    def __init__(
        self,
        quantiles: Sequence[float] | None = None,
        *,
        random_seed: int = 42,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.quantiles = list(quantiles or DEFAULT_QUANTILES)
        self.random_seed = random_seed
        self.model_params = dict(model_params or {})
        self._models: dict[float, Any] = {}
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileLGBM:
        """Fit one LGBMRegressor per quantile level."""
        import lightgbm as lgb

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = None

        y_arr = np.asarray(y, dtype=float)
        self._models = {}
        base_params = {
            "verbosity": -1,
            **self.model_params,
        }

        for q in self.quantiles:
            estimator = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                random_state=self.random_seed,
                **base_params,
            )
            estimator.fit(X, y_arr)
            self._models[q] = estimator
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        enforce_monotonic: bool = True,
    ) -> dict[float, np.ndarray]:
        """Predict all quantile levels."""
        if not self._models:
            msg = "QuantileLGBM must be fitted before predict"
            raise ValueError(msg)

        if self._feature_names is not None:
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                msg = f"predict input missing feature columns: {sorted(missing)}"
                raise ValueError(msg)
            X = X[self._feature_names]

        preds = {
            q: np.asarray(model.predict(X), dtype=float) for q, model in self._models.items()
        }
        if enforce_monotonic:
            return enforce_quantile_order(preds, self.quantiles)
        return preds

    def estimator_at(self, quantile: float) -> Any:
        """Return the fitted estimator for one quantile (for MAPIE prefit)."""
        if quantile not in self._models:
            msg = f"quantile {quantile} not in trained quantiles {self.quantiles}"
            raise KeyError(msg)
        return self._models[quantile]

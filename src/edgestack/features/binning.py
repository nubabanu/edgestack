"""Train-fitted transformers.

Anything that learns parameters from data (quantile edges, scaler moments)
must be fit on the training slice of a fold and only *applied* elsewhere.
These classes make that contract explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edgestack.exceptions import LeakageError


@dataclass
class QuantileBinner:
    """Per-feature quantile thresholds learned from training data only."""

    quantiles: tuple[float, ...]
    _edges: dict[str, np.ndarray] | None = None

    def fit(self, train: pd.DataFrame, columns: tuple[str, ...]) -> QuantileBinner:
        edges: dict[str, np.ndarray] = {}
        for col in columns:
            values = train[col].dropna().to_numpy()
            if len(values) == 0:
                raise LeakageError(f"cannot fit quantiles for empty column {col}")
            edges[col] = np.quantile(values, self.quantiles)
        self._edges = edges
        return self

    def threshold(self, column: str, quantile: float) -> float:
        """The train-fitted value of ``quantile`` for ``column``."""
        if self._edges is None:
            raise LeakageError("QuantileBinner used before fit()")
        if column not in self._edges:
            raise LeakageError(f"QuantileBinner was not fitted on column {column}")
        try:
            idx = self.quantiles.index(quantile)
        except ValueError:
            raise LeakageError(
                f"quantile {quantile} not in fitted grid {self.quantiles}"
            ) from None
        return float(self._edges[column][idx])

    @property
    def is_fitted(self) -> bool:
        return self._edges is not None

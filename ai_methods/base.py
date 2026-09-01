"""Common interface only; method algorithms remain owned by their modules."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ForecastMethod(ABC):
    """Minimal contract implemented independently by each forecasting method."""

    method_id: str

    @abstractmethod
    def train(self, province_history: pd.DataFrame, **kwargs: Any) -> Any:
        """Train one method for one province and return its opaque artifact."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        artifact: Any,
        province_history: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return predictions in the integration layer's tabular form."""
        raise NotImplementedError

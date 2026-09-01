"""Integration contracts for independently specified AI forecasting methods."""

from .base import ForecastMethod
from .registry import METHOD_IDS, METHOD_REGISTRY, get_method_class

__all__ = ["ForecastMethod", "METHOD_IDS", "METHOD_REGISTRY", "get_method_class"]

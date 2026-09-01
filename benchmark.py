"""Shared benchmark schemas and metrics; contains no forecasting algorithm."""

from collections.abc import Iterable

import numpy as np


PREDICTION_SCHEMA = (
    "Method_ID",
    "Run_ID",
    "Province",
    "Train_Cutoff",
    "Forecast_Month",
    "Horizon",
    "Predicted_Load_kWh",
)

ACTUAL_SCHEMA = (
    "Province",
    "Forecast_Month",
    "Actual_Load_kWh",
)

BENCHMARK_SCHEMA = PREDICTION_SCHEMA + (
    "Actual_Load_kWh",
    "APE_pct",
    "Squared_Error",
)

PASS_MAPE_THRESHOLD_PCT = 10.0


def _paired_arrays(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and prediction values must be non-empty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and prediction values must be finite")
    return actual, predicted


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(actual - predicted) / denominator) * 100.0)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred)
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total == 0.0:
        return float("nan")
    residual = float(np.sum((actual - predicted) ** 2))
    return float(1.0 - residual / total)


def passes_mape(mape_pct: float) -> bool:
    return bool(np.isfinite(mape_pct) and mape_pct <= PASS_MAPE_THRESHOLD_PCT)


def metric_summary(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> dict[str, float | bool]:
    actual, predicted = _paired_arrays(y_true, y_pred)
    mape_pct = mape(actual, predicted)
    return {
        "MAPE_pct": mape_pct,
        "RMSE": rmse(actual, predicted),
        "R2": r2(actual, predicted),
        "PASS": passes_mape(mape_pct),
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from ai_methods.base import ForecastMethod

METHOD_ID = "chatgpt_carr1m"
METHOD_NAME = "CARR-1M — Coverage-Aware Residual Reconciliation"
VALIDATION_MONTHS = 12


# ============================================================
# CARR-1M FREEZE CONFIGURATION
# ============================================================

RIDGE_ALPHA = 10.0

RIDGE_WEIGHT = 0.60
HGB_WEIGHT = 0.40

RESIDUAL_SHRINK = 0.55

GROWTH_LOWER = 0.85
GROWTH_UPPER = 1.15

HGB_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.04,
    max_iter=120,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=6,
    l2_regularization=2.0,
    early_stopping=False,
    random_state=2026,
)

FEATURE_COLUMNS = [
    "log_y_lag1",
    "log_y_lag2",
    "log_y_lag3",
    "log_y_lag6",
    "log_y_lag12",
    "log_y_lag24",
    "median_ratio_12",
    "rolling_median_3",
    "rolling_median_6",
    "rolling_median_12",
    "recent_growth_1",
    "recent_growth_3",
    "trend_12",
    "month_sin",
    "month_cos",
    "next_temp",
    "temp_lag1",
    "temp_delta",
    "temp_month_anomaly",
    "log_anchor",
]


# ============================================================
# HELPERS
# ============================================================

def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Province", "Date", "Load_(kWh)", "Temperature_C"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df[required].copy()
    x["Date"] = pd.to_datetime(x["Date"])

    x = (
        x.sort_values("Date")
         .drop_duplicates(["Province", "Date"], keep="last")
         .reset_index(drop=True)
    )

    if (x["Load_(kWh)"] < 0).any():
        raise ValueError("Negative Load_(kWh) detected.")

    return x


def _safe_log1p(x):
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def _seasonal_anchor(y: np.ndarray, origin_idx: int) -> float:
    """
    Forecast target is origin_idx + 1.

    Anchor:
        y[t-11] = same calendar month one year ago.

    Growth:
        y[t-k] / y[t-12-k], k=0..3
    """
    t = origin_idx

    if t < 15:
        raise ValueError("At least 16 historical observations are required.")

    anchor = y[t - 11]

    ratios = []
    for k in range(4):
        a = y[t - k]
        b = y[t - 12 - k]

        if b > 0:
            ratios.append(a / b)

    if not ratios:
        growth = 1.0
    else:
        growth = float(np.median(ratios))

    growth = float(np.clip(
        growth,
        GROWTH_LOWER,
        GROWTH_UPPER
    ))

    return max(anchor * growth, 0.0)


def _trend_slope(log_y: np.ndarray) -> float:
    """
    Linear slope of last 12 log observations.
    """
    if len(log_y) < 12:
        return 0.0

    z = log_y[-12:]
    x = np.arange(12, dtype=float)

    return float(np.polyfit(x, z, 1)[0])


def _build_origin_row(
    history: pd.DataFrame,
    origin_idx: int,
    next_temperature: float,
    monthly_temp_climatology: Dict[int, float],
) -> tuple[pd.DataFrame, float]:
    """
    Build a NEW feature row for origin t -> target t+1.

    Critical rule:
        No target-month Load is accessed here.
    """
    y = history["Load_(kWh)"].to_numpy(dtype=float)
    temp = history["Temperature_C"].to_numpy(dtype=float)

    if origin_idx < 24:
        raise ValueError(
            "Need at least 24 months of history for the frozen feature set."
        )

    target_month = int(
        history.iloc[origin_idx]["Date"].month
    )
    next_month = 1 if target_month == 12 else target_month + 1

    anchor = _seasonal_anchor(y, origin_idx)

    ratios = []
    for k in range(4):
        a = y[origin_idx - k]
        b = y[origin_idx - 12 - k]
        if b > 0:
            ratios.append(a / b)

    median_ratio = (
        float(np.median(ratios))
        if ratios else 1.0
    )

    log_y = _safe_log1p(y)

    row = {
        "log_y_lag1": log_y[origin_idx],
        "log_y_lag2": log_y[origin_idx - 1],
        "log_y_lag3": log_y[origin_idx - 2],
        "log_y_lag6": log_y[origin_idx - 5],
        "log_y_lag12": log_y[origin_idx - 11],
        "log_y_lag24": log_y[origin_idx - 23],

        "median_ratio_12": median_ratio,

        "rolling_median_3": np.log1p(
            np.median(y[origin_idx - 2:origin_idx + 1])
        ),

        "rolling_median_6": np.log1p(
            np.median(y[origin_idx - 5:origin_idx + 1])
        ),

        "rolling_median_12": np.log1p(
            np.median(y[origin_idx - 11:origin_idx + 1])
        ),

        "recent_growth_1": (
            y[origin_idx] / y[origin_idx - 1] - 1.0
            if y[origin_idx - 1] > 0 else 0.0
        ),

        "recent_growth_3": (
            y[origin_idx] / y[origin_idx - 3] - 1.0
            if y[origin_idx - 3] > 0 else 0.0
        ),

        "trend_12": _trend_slope(log_y[:origin_idx + 1]),

        "month_sin": np.sin(
            2.0 * np.pi * next_month / 12.0
        ),

        "month_cos": np.cos(
            2.0 * np.pi * next_month / 12.0
        ),

        "next_temp": float(next_temperature),

        "temp_lag1": float(temp[origin_idx]),

        "temp_delta": (
            float(next_temperature - temp[origin_idx])
        ),

        "temp_month_anomaly": (
            float(next_temperature)
            - float(monthly_temp_climatology.get(next_month,
                                                 np.mean(temp)))
        ),

        "log_anchor": np.log1p(anchor),
    }

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    return X, anchor


def _build_training_matrix(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create one-step-ahead training examples.

    Each training example uses only information available
    at its own historical forecast origin.

    For origin t:
        features <= t
        target = Load[t+1]
    """
    history = history.sort_values("Date").reset_index(drop=True)

    y = history["Load_(kWh)"].to_numpy(dtype=float)
    temp = history["Temperature_C"].to_numpy(dtype=float)

    n = len(history)

    if n < 48:
        raise ValueError(
            "CARR-1M requires at least 48 months of history."
        )

    # Monthly temperature climatology is computed using all
    # temperatures in the TRAINING DATA ONLY.
    climatology = (
        history.assign(month=history["Date"].dt.month)
        .groupby("month")["Temperature_C"]
        .mean()
        .to_dict()
    )

    rows = []
    targets = []

    # Need 24 months of history and one target month.
    for t in range(24, n - 1):
        X_t, anchor = _build_origin_row(
            history=history,
            origin_idx=t,
            next_temperature=temp[t + 1],
            monthly_temp_climatology=climatology,
        )

        target = y[t + 1]

        # Residual target in log space.
        residual = (
            np.log1p(max(target, 0.0))
            - np.log1p(max(anchor, 0.0))
        )

        rows.append(X_t.iloc[0])
        targets.append(residual)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y_resid = pd.Series(
        targets,
        name="residual"
    )

    return X, y_resid


# ============================================================
# ARTIFACT
# ============================================================

@dataclass
class CARR1MArtifact:
    province: str
    train_cutoff: pd.Timestamp
    ridge: Any
    hgb: Any
    temperature_climatology: Dict[int, float]
    feature_columns: list
    specification_version: str = "CARR-1M-v1.0"
    residual_shrink: float = RESIDUAL_SHRINK
    ridge_weight: float = RIDGE_WEIGHT
    hgb_weight: float = HGB_WEIGHT


# ============================================================
# PUBLIC INTERFACE
# ============================================================

def _canonical_train(
    province_history: pd.DataFrame,
    train_cutoff,
    optional_exogenous: Optional[pd.DataFrame] = None,
) -> CARR1MArtifact:
    """
    Common interface:

        train(province_history, train_cutoff, optional_exogenous)

    province_history must contain only observations available
    by train_cutoff.
    """

    df = _prepare_history(province_history)

    train_cutoff = pd.Timestamp(train_cutoff)

    df = df[df["Date"] <= train_cutoff].copy()

    if df.empty:
        raise ValueError("No training data before train_cutoff.")

    provinces = df["Province"].dropna().unique()

    if len(provinces) != 1:
        raise ValueError(
            "train() expects one province at a time."
        )

    province = str(provinces[0])

    df = df.sort_values("Date").reset_index(drop=True)

    # Strict check: monthly frequency and uniqueness.
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    X, y_resid = _build_training_matrix(df)

    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA))
    ])

    hgb = HistGradientBoostingRegressor(
        **HGB_PARAMS
    )

    ridge.fit(X, y_resid)
    hgb.fit(X, y_resid)

    climatology = (
        df.assign(month=df["Date"].dt.month)
          .groupby("month")["Temperature_C"]
          .mean()
          .to_dict()
    )

    return CARR1MArtifact(
        province=province,
        train_cutoff=train_cutoff,
        ridge=ridge,
        hgb=hgb,
        temperature_climatology=climatology,
        feature_columns=FEATURE_COLUMNS.copy(),
    )


def _canonical_predict(
    artifact: CARR1MArtifact,
    province_history: pd.DataFrame,
    next_temperature: float,
    optional_exogenous: Optional[pd.DataFrame] = None,
) -> float:
    """
    Common interface:

        predict(
            artifact,
            province_history,
            next_temperature,
            optional_exogenous
        )

    province_history may contain historical data up to the
    forecast origin, but MUST NOT contain target-month Actual Load.
    """

    df = _prepare_history(province_history)

    df = (
        df[df["Province"] == artifact.province]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No history for artifact province.")

    latest_date = df["Date"].max()

    if latest_date != artifact.train_cutoff:
        raise ValueError(
            f"Expected history through {artifact.train_cutoff.date()}, "
            f"got {latest_date.date()}."
        )

    # This row is generated from history only.
    X_future, anchor = _build_origin_row(
        history=df,
        origin_idx=len(df) - 1,
        next_temperature=float(next_temperature),
        monthly_temp_climatology=artifact.temperature_climatology,
    )

    ridge_resid = float(
        artifact.ridge.predict(X_future)[0]
    )

    hgb_resid = float(
        artifact.hgb.predict(X_future)[0]
    )

    ensemble_resid = (
        artifact.ridge_weight * ridge_resid
        +
        artifact.hgb_weight * hgb_resid
    )

    final_log_resid = (
        artifact.residual_shrink * ensemble_resid
    )

    forecast = (
        np.exp(
            np.log1p(max(anchor, 0.0))
            + final_log_resid
        )
        - 1.0
    )

    return float(max(forecast, 0.0))


# ============================================================
# OPTIONAL: LEAKAGE TEST
# ============================================================

def assert_no_target_month_actual(
    province_history: pd.DataFrame,
    forecast_month,
):
    """
    Safety check for production pipeline.

    The history passed into prediction must end before
    forecast_month.
    """

    df = _prepare_history(province_history)

    forecast_month = pd.Timestamp(forecast_month)

    if df["Date"].max() >= forecast_month:
        raise AssertionError(
            "Leakage detected: history contains target-month "
            "or later observations."
        )


# ============================================================
# OPTIONAL: METRICS
# ============================================================

def evaluate_predictions(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("actual and forecast length mismatch.")

    error_pct = np.where(
        np.abs(actual) > 0,
        100.0 * (forecast - actual) / np.abs(actual),
        np.nan,
    )

    ape = np.abs(error_pct)

    mape = np.nanmean(ape)

    rmse = np.sqrt(
        np.mean((actual - forecast) ** 2)
    )

    ss_res = np.sum((actual - forecast) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    r2 = (
        np.nan
        if ss_tot == 0
        else 1.0 - ss_res / ss_tot
    )

    return {
        "MAPE_percent": float(mape),
        "RMSE_kWh": float(rmse),
        "R2": float(r2),
        "Mean_Error_percent": float(np.nanmean(error_pct)),
        "Median_APE_percent": float(np.nanmedian(ape)),
        "P90_APE_percent": float(np.nanpercentile(ape, 90)),
    }


def _validation_metrics(history: pd.DataFrame) -> dict:
    """Canonical 12-origin expanding validation, isolated from final fitting."""
    origins = list(range(max(48, len(history) - VALIDATION_MONTHS), len(history)))
    actual, forecast = [], []
    for target_idx in origins:
        fold = history.iloc[:target_idx].copy()
        if len(fold) < 48:
            continue
        target = history.iloc[target_idx]
        artifact = _canonical_train(fold, fold["Date"].iloc[-1])
        value = _canonical_predict(
            artifact,
            fold,
            next_temperature=float(target["Temperature_C"]),
        )
        actual.append(float(target["Load_(kWh)"]))
        forecast.append(value)
    if len(actual) < 2:
        raise ValueError("CARR-1M requires at least two validation origins")
    raw = evaluate_predictions(np.asarray(actual), np.asarray(forecast))
    return {
        "MAPE": raw["MAPE_percent"],
        "RMSE": raw["RMSE_kWh"],
        "R2": raw["R2"],
    }


class ChatGPTCARR1M(ForecastMethod):
    """Application adapter; the frozen CARR-1M equations remain above."""

    method_id = METHOD_ID
    method_name = METHOD_NAME

    def train(self, province_history: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        cutoff = pd.Timestamp(kwargs.get("train_cutoff", province_history["Date"].max()))
        history = _prepare_history(province_history)
        history = history[history["Date"] <= cutoff].reset_index(drop=True)
        artifact = _canonical_train(history, cutoff, kwargs.get("optional_exogenous"))
        return {
            "method_id": METHOD_ID,
            "method_name": METHOD_NAME,
            "province": artifact.province,
            "train_cutoff": artifact.train_cutoff,
            "model": artifact,
            "metrics": _validation_metrics(history),
            "validation_method": "expanding_window_latest_12",
        }

    def predict(
        self,
        artifact: dict[str, Any],
        province_history: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if artifact.get("method_id") != METHOD_ID:
            raise ValueError("Artifact does not belong to CARR-1M")
        next_temperature = float(kwargs["next_temperature"])
        history = _prepare_history(province_history)
        history = history[history["Date"] <= artifact["train_cutoff"]].reset_index(drop=True)
        value = _canonical_predict(
            artifact["model"], history, next_temperature, kwargs.get("optional_exogenous")
        )
        forecast_month = pd.Timestamp(artifact["train_cutoff"]) + pd.DateOffset(months=1)
        return pd.DataFrame([{
            "Method_ID": METHOD_ID,
            "Method_Name": METHOD_NAME,
            "Province": artifact["province"],
            "Train_Cutoff": pd.Timestamp(artifact["train_cutoff"]),
            "Forecast_Month": forecast_month,
            "Predicted_Load_kWh": max(float(value), 0.0),
            "Temperature_C": next_temperature,
        }])

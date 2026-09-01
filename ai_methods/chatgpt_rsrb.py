"""RSRB — Robust Seasonal Residual Boosting.

This module owns the complete ChatGPT RSRB method.  All load-derived historical
features are shifted so a row dated ``t`` only uses load observations before
``t``.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .base import ForecastMethod


METHOD_ID = "chatgpt_rsrb"
METHOD_NAME = "RSRB — Robust Seasonal Residual Boosting"
CORRECTION_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
VALIDATION_MONTHS = 18
FEATURE_COLUMNS = (
    "lag1", "lag2", "lag3", "lag6", "lag12",
    "MA3", "MA6", "Median3", "Volatility6", "CV6",
    "sin_month", "cos_month", "Temperature_C", "temp_lag1",
    "temp_diff", "temperature_anomaly", "growth_factor",
)


def _new_residual_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=120,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42,
    )


def _prepare_history(
    province_history: pd.DataFrame, train_cutoff: Any | None = None
) -> pd.DataFrame:
    required = {"Province", "Date", "Load_(kWh)", "Temperature_C"}
    missing = sorted(required.difference(province_history.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    history = province_history.loc[:, list(required)].copy()
    history["Province"] = history["Province"].astype(str).str.strip()
    if history["Province"].nunique() != 1:
        raise ValueError("RSRB train/predict accepts exactly one province")
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Load_(kWh)"] = pd.to_numeric(history["Load_(kWh)"], errors="coerce")
    history["Temperature_C"] = pd.to_numeric(history["Temperature_C"], errors="coerce")
    history = history.dropna(subset=["Date"]).copy()
    history["Date"] = history["Date"].dt.to_period("M").dt.to_timestamp()

    if train_cutoff is not None:
        cutoff = pd.Timestamp(train_cutoff).to_period("M").to_timestamp()
        history = history[history["Date"] <= cutoff].copy()

    history = history.sort_values("Date").reset_index(drop=True)
    if history.empty:
        raise ValueError("No history is available through the train cutoff")
    if history["Date"].duplicated().any():
        raise ValueError("Duplicate province-month rows are not allowed")
    expected = pd.date_range(history["Date"].min(), history["Date"].max(), freq="MS")
    if len(expected) != len(history) or not history["Date"].equals(pd.Series(expected)):
        raise ValueError("Monthly history must be continuous")
    numeric = history[["Load_(kWh)", "Temperature_C"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Load and temperature history must be finite")
    return history


def _same_month_temperature_means(history: pd.DataFrame) -> pd.Series:
    means = []
    for index, row in history.iterrows():
        prior = history.iloc[:index]
        same_month = prior.loc[
            prior["Date"].dt.month == row["Date"].month, "Temperature_C"
        ]
        means.append(float(same_month.mean()) if not same_month.empty else np.nan)
    return pd.Series(means, index=history.index, dtype=float)


def _historical_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    load = frame["Load_(kWh)"].astype(float)
    temperature = frame["Temperature_C"].astype(float)

    for lag in (1, 2, 3, 6, 12):
        frame[f"lag{lag}"] = load.shift(lag)
    frame["MA3"] = load.shift(1).rolling(3).mean()
    frame["MA6"] = load.shift(1).rolling(6).mean()
    frame["Median3"] = load.shift(1).rolling(3).median()
    frame["Volatility6"] = load.shift(1).rolling(6).std()
    frame["CV6"] = frame["Volatility6"] / frame["MA6"].abs().clip(lower=1e-6)

    month = frame["Date"].dt.month
    frame["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
    frame["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)
    frame["temp_lag1"] = temperature.shift(1)
    frame["temp_diff"] = temperature - frame["temp_lag1"]
    frame["temperature_anomaly"] = temperature - _same_month_temperature_means(frame)

    prior_year_recent3 = load.shift(13).rolling(3).mean()
    raw_growth = frame["MA3"] / prior_year_recent3.abs().clip(lower=1e-6)
    frame["growth_factor"] = raw_growth.clip(lower=0.75, upper=1.25)
    frame["scaled_lag12"] = frame["lag12"] * frame["growth_factor"]
    frame["Seasonal_Baseline"] = (
        0.60 * frame["scaled_lag12"]
        + 0.25 * frame["MA3"]
        + 0.15 * frame["Median3"]
    )
    frame["Residual"] = frame["Load_(kWh)"] - frame["Seasonal_Baseline"]
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURE_COLUMNS, "Seasonal_Baseline", "Residual"]
    ).reset_index(drop=True)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(actual), 1e-6)
    mape = float(np.mean(np.abs(actual - predicted) / denominator) * 100.0)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum((actual - predicted) ** 2) / total) if total else float("nan")
    return {"MAPE": mape, "RMSE": rmse, "R2": r2}


def _next_month_features(history: pd.DataFrame, next_temperature: float) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    if not np.isfinite(float(next_temperature)):
        raise ValueError("next_temperature must be finite")
    load = history["Load_(kWh)"].astype(float).reset_index(drop=True)
    temperature = history["Temperature_C"].astype(float).reset_index(drop=True)
    if len(load) < 15:
        raise ValueError("At least 15 continuous months are required for next-month features")

    next_date = history["Date"].iloc[-1] + pd.DateOffset(months=1)
    ma3 = float(load.tail(3).mean())
    ma6 = float(load.tail(6).mean())
    median3 = float(load.tail(3).median())
    volatility6 = float(load.tail(6).std())
    prior_year_recent3 = float(load.iloc[-15:-12].mean())
    growth_factor = float(np.clip(ma3 / max(abs(prior_year_recent3), 1e-6), 0.75, 1.25))
    lag12 = float(load.iloc[-12])
    same_month_temps = history.loc[
        history["Date"].dt.month == next_date.month, "Temperature_C"
    ]
    if same_month_temps.empty:
        raise ValueError("Historical same-month temperature is unavailable")

    row = {
        "lag1": float(load.iloc[-1]),
        "lag2": float(load.iloc[-2]),
        "lag3": float(load.iloc[-3]),
        "lag6": float(load.iloc[-6]),
        "lag12": lag12,
        "MA3": ma3,
        "MA6": ma6,
        "Median3": median3,
        "Volatility6": volatility6,
        "CV6": volatility6 / max(abs(ma6), 1e-6),
        "sin_month": float(np.sin(2.0 * np.pi * next_date.month / 12.0)),
        "cos_month": float(np.cos(2.0 * np.pi * next_date.month / 12.0)),
        "Temperature_C": float(next_temperature),
        "temp_lag1": float(temperature.iloc[-1]),
        "temp_diff": float(next_temperature - temperature.iloc[-1]),
        "temperature_anomaly": float(next_temperature - same_month_temps.mean()),
        "growth_factor": growth_factor,
    }
    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Next-month feature row contains NaN or infinity")
    baseline = 0.60 * lag12 * growth_factor + 0.25 * ma3 + 0.15 * median3
    return features, float(baseline), pd.Timestamp(next_date)


class ChatGPTRSRB(ForecastMethod):
    method_id = METHOD_ID
    method_name = METHOD_NAME

    def train(self, province_history: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        train_cutoff = kwargs.get("train_cutoff")
        history = _prepare_history(province_history, train_cutoff=train_cutoff)
        featured = _historical_feature_frame(history)
        if len(featured) < 20:
            raise ValueError("Insufficient history for RSRB walk-forward validation")

        validation_count = min(VALIDATION_MONTHS, len(featured) - 8)
        if validation_count < 1:
            raise ValueError("No supported expanding validation month is available")
        validation_start = len(featured) - validation_count
        validation_rows = []
        for position in range(validation_start, len(featured)):
            fold_train = featured.iloc[:position]
            fold_target = featured.iloc[position]
            model = _new_residual_model()
            model.fit(fold_train[list(FEATURE_COLUMNS)], fold_train["Residual"])
            residual_prediction = float(
                model.predict(fold_target[list(FEATURE_COLUMNS)].to_frame().T)[0]
            )
            validation_rows.append(
                {
                    "Date": fold_target["Date"],
                    "Actual_Load_kWh": float(fold_target["Load_(kWh)"]),
                    "Seasonal_Baseline": float(fold_target["Seasonal_Baseline"]),
                    "Residual_Prediction": residual_prediction,
                }
            )

        validation = pd.DataFrame(validation_rows)
        actual = validation["Actual_Load_kWh"].to_numpy(dtype=float)
        scores: dict[float, dict[str, float]] = {}
        for weight in CORRECTION_WEIGHTS:
            prediction = (
                validation["Seasonal_Baseline"].to_numpy(dtype=float)
                + weight * validation["Residual_Prediction"].to_numpy(dtype=float)
            )
            scores[weight] = _metrics(actual, prediction)
        best_weight = min(CORRECTION_WEIGHTS, key=lambda value: scores[value]["MAPE"])
        validation["Predicted_Load_kWh"] = (
            validation["Seasonal_Baseline"]
            + best_weight * validation["Residual_Prediction"]
        )

        final_model = _new_residual_model()
        final_model.fit(featured[list(FEATURE_COLUMNS)], featured["Residual"])
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "province": str(history["Province"].iloc[0]),
            "train_cutoff": pd.Timestamp(history["Date"].iloc[-1]),
            "model": final_model,
            "correction_weight": float(best_weight),
            "feature_columns": FEATURE_COLUMNS,
            "validation_method": "expanding_walk_forward_latest_18",
            "validation_predictions": validation,
            "metrics": scores[best_weight],
            "weight_metrics": scores,
            "training_rows": int(len(featured)),
        }

    def predict(
        self,
        artifact: dict[str, Any],
        province_history: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if artifact.get("method_id") != self.method_id:
            raise ValueError("Artifact does not belong to ChatGPT RSRB")
        if "next_temperature" not in kwargs:
            raise ValueError("next_temperature is required")
        history = _prepare_history(
            province_history, train_cutoff=artifact["train_cutoff"]
        )
        if pd.Timestamp(history["Date"].iloc[-1]) != pd.Timestamp(artifact["train_cutoff"]):
            raise ValueError("History does not reach the artifact train cutoff")
        if str(history["Province"].iloc[0]) != str(artifact["province"]):
            raise ValueError("Artifact province does not match prediction history")

        features, baseline, forecast_month = _next_month_features(
            history, float(kwargs["next_temperature"])
        )
        residual = float(artifact["model"].predict(features)[0])
        prediction = baseline + float(artifact["correction_weight"]) * residual
        if not np.isfinite(prediction):
            raise ValueError("RSRB prediction is not finite")
        return pd.DataFrame(
            [{
                "Method_ID": self.method_id,
                "Method_Name": self.method_name,
                "Province": artifact["province"],
                "Train_Cutoff": pd.Timestamp(artifact["train_cutoff"]),
                "Forecast_Month": forecast_month,
                "Horizon": 1,
                "Predicted_Load_kWh": float(prediction),
                "Temperature_C": float(kwargs["next_temperature"]),
                "Seasonal_Baseline": baseline,
                "Correction_Weight": float(artifact["correction_weight"]),
            }]
        )

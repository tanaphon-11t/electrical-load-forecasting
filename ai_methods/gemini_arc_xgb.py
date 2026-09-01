"""ARC-XGB — Adaptive Residual-Corrected XGBoost Ensemble."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .base import ForecastMethod


METHOD_ID = "gemini_arc_xgb"
METHOD_NAME = "ARC-XGB — Adaptive Residual-Corrected XGBoost Ensemble"
SEEDS = (42, 100, 2024, 777, 999)
FEATURE_COLUMNS = (
    "Quarter", "Month_Sin", "Month_Cos",
    "Load_Lag1", "Load_Lag2", "Load_Lag3", "Load_Lag12",
    "Load_Roll_Mean_3", "Load_Roll_Std_3", "Load_Roll_Mean_12",
    "Load_EWMA_3", "Load_EWMA_6", "Temperature_C", "Temp_Lag1",
    "Temp_Diff_1", "Cooling_Degree_Proxy", "Temp_Load_Interaction",
)


def _xgb_regressor(seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("gemini_arc_xgb requires the xgboost package") from exc
    return XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
    )


def _prepare_history(
    province_history: pd.DataFrame, train_cutoff: Any | None = None
) -> pd.DataFrame:
    required = ["Province", "Date", "Load_(kWh)", "Temperature_C"]
    missing = [column for column in required if column not in province_history.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    history = province_history[required].copy()
    history["Province"] = history["Province"].astype(str).str.strip()
    if history["Province"].nunique() != 1:
        raise ValueError("ARC-XGB train/predict accepts exactly one province")
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
    values = history[["Load_(kWh)", "Temperature_C"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Load and temperature history must be finite")
    return history


def _rolling_iqr_winsorize(load: pd.Series) -> pd.Series:
    """Clip each point using only its preceding 12 raw observations."""
    raw = load.astype(float).reset_index(drop=True)
    winsorized = raw.copy()
    for position in range(12, len(raw)):
        historical_window = raw.iloc[position - 12:position]
        q1 = float(historical_window.quantile(0.25))
        q3 = float(historical_window.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        winsorized.iloc[position] = float(np.clip(raw.iloc[position], lower, upper))
    return winsorized


def _feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    raw_load = frame["Load_(kWh)"].astype(float)
    load = _rolling_iqr_winsorize(raw_load)
    temperature = frame["Temperature_C"].astype(float)
    frame["Winsorized_Load"] = load

    month = frame["Date"].dt.month
    frame["Quarter"] = ((month - 1) // 3 + 1).astype(int)
    frame["Month_Sin"] = np.sin(2.0 * np.pi * month / 12.0)
    frame["Month_Cos"] = np.cos(2.0 * np.pi * month / 12.0)
    for lag in (1, 2, 3, 12):
        frame[f"Load_Lag{lag}"] = load.shift(lag)
    frame["Load_Roll_Mean_3"] = load.shift(1).rolling(3).mean()
    frame["Load_Roll_Std_3"] = load.shift(1).rolling(3).std()
    frame["Load_Roll_Mean_12"] = load.shift(1).rolling(12).mean()
    frame["Load_EWMA_3"] = load.shift(1).ewm(span=3, adjust=False).mean()
    frame["Load_EWMA_6"] = load.shift(1).ewm(span=6, adjust=False).mean()
    frame["Temp_Lag1"] = temperature.shift(1)
    frame["Temp_Diff_1"] = temperature - frame["Temp_Lag1"]
    frame["Cooling_Degree_Proxy"] = (temperature - 25.0).clip(lower=0.0)
    frame["Temp_Load_Interaction"] = temperature * frame["Load_Lag1"]
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURE_COLUMNS, "Winsorized_Load", "Load_(kWh)"]
    ).reset_index(drop=True)


def _fit_ensemble(features: pd.DataFrame, target: pd.Series) -> list[Any]:
    models = []
    for seed in SEEDS:
        model = _xgb_regressor(seed)
        model.fit(features, target)
        models.append(model)
    return models


def _ensemble_predict(models: list[Any], features: pd.DataFrame) -> np.ndarray:
    predictions = np.vstack([model.predict(features) for model in models])
    return predictions.mean(axis=0)


def _fit_base_and_residual(
    features: pd.DataFrame, winsorized_load: pd.Series, raw_load: pd.Series
) -> tuple[list[Any], list[Any]]:
    base_models = _fit_ensemble(features, winsorized_load)
    base_training_prediction = _ensemble_predict(base_models, features)
    residual_target = raw_load.to_numpy(dtype=float) - base_training_prediction
    residual_models = _fit_ensemble(features, pd.Series(residual_target, index=features.index))
    return base_models, residual_models


def _combined_predict(
    base_models: list[Any], residual_models: list[Any], features: pd.DataFrame
) -> np.ndarray:
    return _ensemble_predict(base_models, features) + _ensemble_predict(
        residual_models, features
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(actual), 1e-6)
    mape = float(np.mean(np.abs(actual - predicted) / denominator) * 100.0)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum((actual - predicted) ** 2) / total) if total else float("nan")
    return {"MAPE": mape, "RMSE": rmse, "R2": r2}


def _next_month_features(
    history: pd.DataFrame, next_temperature: float
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not np.isfinite(float(next_temperature)):
        raise ValueError("next_temperature must be finite")
    if len(history) < 12:
        raise ValueError("At least 12 continuous months are required")
    load = _rolling_iqr_winsorize(history["Load_(kWh)"]).reset_index(drop=True)
    temperature = history["Temperature_C"].astype(float).reset_index(drop=True)
    next_date = pd.Timestamp(history["Date"].iloc[-1]) + pd.DateOffset(months=1)
    month = next_date.month
    row = {
        "Quarter": int((month - 1) // 3 + 1),
        "Month_Sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "Month_Cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "Load_Lag1": float(load.iloc[-1]),
        "Load_Lag2": float(load.iloc[-2]),
        "Load_Lag3": float(load.iloc[-3]),
        "Load_Lag12": float(load.iloc[-12]),
        "Load_Roll_Mean_3": float(load.tail(3).mean()),
        "Load_Roll_Std_3": float(load.tail(3).std()),
        "Load_Roll_Mean_12": float(load.tail(12).mean()),
        "Load_EWMA_3": float(load.ewm(span=3, adjust=False).mean().iloc[-1]),
        "Load_EWMA_6": float(load.ewm(span=6, adjust=False).mean().iloc[-1]),
        "Temperature_C": float(next_temperature),
        "Temp_Lag1": float(temperature.iloc[-1]),
        "Temp_Diff_1": float(next_temperature - temperature.iloc[-1]),
        "Cooling_Degree_Proxy": float(max(0.0, next_temperature - 25.0)),
        "Temp_Load_Interaction": float(next_temperature * load.iloc[-1]),
    }
    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Next-month feature row contains NaN or infinity")
    return features, pd.Timestamp(next_date)


class GeminiARCXGB(ForecastMethod):
    method_id = METHOD_ID
    method_name = METHOD_NAME

    def train(self, province_history: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        history = _prepare_history(
            province_history, train_cutoff=kwargs.get("train_cutoff")
        )
        featured = _feature_frame(history)
        if len(featured) < 16:
            raise ValueError("Insufficient history for expanding-window 3-fold validation")

        features = featured[list(FEATURE_COLUMNS)]
        raw_load = featured["Load_(kWh)"].astype(float)
        winsorized_load = featured["Winsorized_Load"].astype(float)
        splitter = TimeSeriesSplit(n_splits=3)
        validation_rows = []
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(features), 1):
            train_x = features.iloc[train_indices]
            validation_x = features.iloc[validation_indices]
            base_models, residual_models = _fit_base_and_residual(
                train_x,
                winsorized_load.iloc[train_indices].reset_index(drop=True),
                raw_load.iloc[train_indices].reset_index(drop=True),
            )
            prediction = _combined_predict(base_models, residual_models, validation_x)
            for position, predicted in zip(validation_indices, prediction):
                validation_rows.append({
                    "Fold": fold,
                    "Date": featured.iloc[position]["Date"],
                    "Actual_Load_kWh": float(raw_load.iloc[position]),
                    "Predicted_Load_kWh": float(predicted),
                })

        validation = pd.DataFrame(validation_rows)
        metrics = _metrics(
            validation["Actual_Load_kWh"].to_numpy(dtype=float),
            validation["Predicted_Load_kWh"].to_numpy(dtype=float),
        )
        base_models, residual_models = _fit_base_and_residual(
            features, winsorized_load.reset_index(drop=True), raw_load.reset_index(drop=True)
        )
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "province": str(history["Province"].iloc[0]),
            "train_cutoff": pd.Timestamp(history["Date"].iloc[-1]),
            "base_models": base_models,
            "residual_models": residual_models,
            "feature_columns": FEATURE_COLUMNS,
            "validation_method": "expanding_window_3fold",
            "validation_predictions": validation,
            "metrics": metrics,
            "training_rows": int(len(featured)),
        }

    def predict(
        self,
        artifact: dict[str, Any],
        province_history: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if artifact.get("method_id") != self.method_id:
            raise ValueError("Artifact does not belong to Gemini ARC-XGB")
        if "next_temperature" not in kwargs:
            raise ValueError("next_temperature is required")
        history = _prepare_history(
            province_history, train_cutoff=artifact["train_cutoff"]
        )
        if pd.Timestamp(history["Date"].iloc[-1]) != pd.Timestamp(artifact["train_cutoff"]):
            raise ValueError("History does not reach the artifact train cutoff")
        if str(history["Province"].iloc[0]) != str(artifact["province"]):
            raise ValueError("Artifact province does not match prediction history")
        features, forecast_month = _next_month_features(
            history, float(kwargs["next_temperature"])
        )
        base_prediction = float(_ensemble_predict(artifact["base_models"], features)[0])
        residual_prediction = float(
            _ensemble_predict(artifact["residual_models"], features)[0]
        )
        prediction = base_prediction + residual_prediction
        if not np.isfinite(prediction):
            raise ValueError("ARC-XGB prediction is not finite")
        return pd.DataFrame([{
            "Method_ID": self.method_id,
            "Method_Name": self.method_name,
            "Province": artifact["province"],
            "Train_Cutoff": pd.Timestamp(artifact["train_cutoff"]),
            "Forecast_Month": forecast_month,
            "Horizon": 1,
            "Predicted_Load_kWh": float(prediction),
            "Temperature_C": float(kwargs["next_temperature"]),
            "Base_Prediction": base_prediction,
            "Residual_Correction": residual_prediction,
        }])

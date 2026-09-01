import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import STL
from sklearn.base import BaseEstimator, TransformerMixin
import hashlib
import json

from ai_methods.base import ForecastMethod

METHOD_ID = "gemini_cg_hae_v1"
METHOD_NAME = "CG-HAE-v1 — Central-GovLoad Hierarchical Adjusted Ensemble"
VALIDATION_MONTHS = 6

class CGHAEForecaster:
    """
    Central-GovLoad-Hierarchical-Adjusted-Ensemble (CG-HAE-v1)
    Reference Implementation for One-Month-Ahead Forecast
    """
    def __init__(self):
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 150,
            'learning_rate': 0.03,
            'num_leaves': 15,
            'max_depth': 4,
            'min_child_samples': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        self.models = {}
        self.weights = {}

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """สร้าง Feature vectors ป้องกัน Leakage อย่างเคร่งครัด"""
        df = df.sort_values('Date').copy()
        
        # Lag Features
        df['lag_1'] = df['Load_(kWh)'].shift(1)
        df['lag_2'] = df['Load_(kWh)'].shift(2)
        df['lag_12'] = df['Load_(kWh)'].shift(12)
        
        # Rolling Metrics
        df['roll_mean_3'] = df['lag_1'].rolling(window=3).mean()
        df['roll_std_6'] = df['lag_1'].rolling(window=6).std()
        
        # Calendar Features
        df['sin_month'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
        
        # Temperature Proxy
        df['cdd_proxy'] = np.maximum(0, df['Temperature_C'] - 26.0)
        
        return df

    def train(self, province_history: pd.DataFrame, train_cutoff: str, optional_exogenous: pd.DataFrame = None):
        """
        train_cutoff: str รูปแบบ 'YYYY-MM-DD'
        province_history: DataFrame ข้อมูลในอดีต [Province, Date, Load_(kWh), Month_Num, Temperature_C]
        """
        df = province_history.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        train_df = df[df['Date'] <= pd.to_datetime(train_cutoff)].copy()
        
        provinces = train_df['Province'].unique()
        
        for p in provinces:
            p_df = train_df[train_df['Province'] == p].sort_values('Date')
            
            # STL Baseline Component
            stl = STL(p_df['Load_(kWh)'], period=12, robust=True)
            res = stl.fit()
            last_seasonal = res.seasonal.iloc[-12:].values
            
            # LightGBM Component
            feat_df = self._build_features(p_df)
            feat_df = feat_df.dropna()
            
            X = feat_df[['lag_1', 'lag_2', 'lag_12', 'roll_mean_3', 'roll_std_6', 'sin_month', 'cos_month', 'cdd_proxy']]
            y = feat_df['Load_(kWh)']
            
            model = LGBMRegressor(**self.lgbm_params)
            model.fit(X, y)
            
            # Save Artifacts per Province
            self.models[p] = {
                'lgbm': model,
                'last_stl_seasonal': last_seasonal,
                'last_history': p_df.tail(15) # เก็บย้อนหลังเพื่อสร้าง Lag Row
            }
            # Freeze Weight Ratio (Validation Baseline)
            self.weights[p] = {'stl': 0.3, 'lgbm': 0.7}

    def predict(self, province_history: pd.DataFrame, next_temperature: float, target_province: str, optional_exogenous: dict = None) -> float:
        """
        พยากรณ์ค่าเดือน T+1 โดยปราศจาก Leakage
        """
        if target_province not in self.models:
            raise ValueError(f"No trained artifact found for province: {target_province}")
            
        artifacts = self.models[target_province]
        hist = artifacts['last_history'].copy()
        
        # 1. สร้าง Next-Month Feature Row (Isolated Row)
        last_date = hist['Date'].max()
        next_date = last_date + pd.DateOffset(months=1)
        next_month_num = next_date.month
        
        # 2. Extract Lags
        lag_1 = hist['Load_(kWh)'].iloc[-1]
        lag_2 = hist['Load_(kWh)'].iloc[-2]
        lag_12 = hist['Load_(kWh)'].iloc[-12]
        roll_mean_3 = hist['Load_(kWh)'].iloc[-3:].mean()
        roll_std_6 = hist['Load_(kWh)'].iloc[-6:].std()
        
        # 3. Features Vector Assembly
        X_next = pd.DataFrame([{
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_12': lag_12,
            'roll_mean_3': roll_mean_3,
            'roll_std_6': roll_std_6,
            'sin_month': np.sin(2 * np.pi * next_month_num / 12),
            'cos_month': np.cos(2 * np.pi * next_month_num / 12),
            'cdd_proxy': max(0.0, next_temperature - 26.0)
        }])
        
        # 4. Predict Components
        pred_lgbm = artifacts['lgbm'].predict(X_next)[0]
        
        # STL Forecast (Naive Extension of Seasonal + Trend Shift)
        stl_seasonal_val = artifacts['last_stl_seasonal'][(next_month_num - 1) % 12]
        trend_proxy = roll_mean_3
        pred_stl = trend_proxy + stl_seasonal_val
        
        # 5. Ensemble Aggregation
        w_stl = self.weights[target_province]['stl']
        w_lgbm = self.weights[target_province]['lgbm']
        
        final_forecast = (w_stl * pred_stl) + (w_lgbm * pred_lgbm)
        
        # Optional Exogenous Scaling (Reporting Nodes Adjustment)
        if optional_exogenous and 'expected_active_nodes_ratio' in optional_exogenous:
            final_forecast *= optional_exogenous['expected_active_nodes_ratio']
            
        return float(final_forecast)


def _prepare_history(df: pd.DataFrame, cutoff=None) -> pd.DataFrame:
    required = {"Province", "Date", "Load_(kWh)", "Temperature_C"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["Load_(kWh)"] = pd.to_numeric(out["Load_(kWh)"], errors="coerce")
    out["Temperature_C"] = pd.to_numeric(out["Temperature_C"], errors="coerce")
    out = out.dropna(subset=["Date", "Load_(kWh)", "Temperature_C"])
    if cutoff is not None:
        out = out[out["Date"] <= pd.Timestamp(cutoff).to_period("M").to_timestamp()]
    out = out.sort_values("Date").reset_index(drop=True)
    if out["Province"].nunique() != 1:
        raise ValueError("CG-HAE-v1 accepts exactly one province")
    if "Month_Num" not in out:
        out["Month_Num"] = out["Date"].dt.month
    if len(out) < 36:
        raise ValueError("CG-HAE-v1 requires at least 36 months")
    if not np.isfinite(out[["Load_(kWh)", "Temperature_C"]].to_numpy()).all():
        raise ValueError("Load and temperature must be finite")
    return out


def _metric_values(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mape = float(np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), 1e-6)) * 100)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum((actual - predicted) ** 2) / total) if total else float("nan")
    return {"MAPE": mape, "RMSE": rmse, "R2": r2}


def _validation_metrics(history: pd.DataFrame) -> dict:
    actual, predicted = [], []
    start = max(36, len(history) - VALIDATION_MONTHS)
    for target_idx in range(start, len(history)):
        fold = history.iloc[:target_idx].copy()
        target = history.iloc[target_idx]
        model = CGHAEForecaster()
        model.train(fold, fold["Date"].iloc[-1])
        province = str(fold["Province"].iloc[0])
        value = model.predict(fold, float(target["Temperature_C"]), province)
        actual.append(float(target["Load_(kWh)"]))
        predicted.append(max(float(value), 0.0))
    if len(actual) < 2:
        raise ValueError("CG-HAE-v1 requires at least two validation origins")
    return _metric_values(actual, predicted)


class GeminiCGHAEV1(ForecastMethod):
    """Application adapter for the frozen Gemini CG-HAE-v1 reference."""

    method_id = METHOD_ID
    method_name = METHOD_NAME

    def train(self, province_history: pd.DataFrame, **kwargs):
        cutoff = pd.Timestamp(kwargs.get("train_cutoff", province_history["Date"].max()))
        history = _prepare_history(province_history, cutoff)
        forecaster = CGHAEForecaster()
        forecaster.train(history, cutoff, kwargs.get("optional_exogenous"))
        return {
            "method_id": METHOD_ID,
            "method_name": METHOD_NAME,
            "province": str(history["Province"].iloc[0]),
            "train_cutoff": pd.Timestamp(history["Date"].iloc[-1]),
            "model": forecaster,
            "metrics": _validation_metrics(history),
            "validation_method": "expanding_window_latest_6",
        }

    def predict(self, artifact, province_history: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if artifact.get("method_id") != METHOD_ID:
            raise ValueError("Artifact does not belong to CG-HAE-v1")
        history = _prepare_history(province_history, artifact["train_cutoff"])
        next_temperature = float(kwargs["next_temperature"])
        value = artifact["model"].predict(
            history,
            next_temperature,
            artifact["province"],
            kwargs.get("optional_exogenous"),
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

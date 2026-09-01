"""RAMEn: Robust Adaptive Multi-Horizon Gradient Ensemble."""

import warnings
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import mstats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .base import ForecastMethod

warnings.filterwarnings("ignore")


class RAMEnForecaster:
    METHOD_NAME = "RAMEn (Robust Adaptive Multi-horizon Gradient Ensemble)"

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.weights = None
        self.feature_cols = None
        self.log_transform = True
        self.winsor_limits = None
        self.recent_bias = 0.0

    def _create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        d = df.copy().sort_values("Date").reset_index(drop=True)
        d["Date"] = pd.to_datetime(d["Date"])
        d["year"] = d["Date"].dt.year
        d["month"] = d["Date"].dt.month
        d["quarter"] = d["Date"].dt.quarter
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        d["time_idx"] = (d["Date"] - d["Date"].min()).dt.days // 30
        d["is_post_covid"] = (d["Date"] >= "2020-03-01").astype(int)
        d["is_peak_summer"] = d["month"].isin([3, 4, 5]).astype(int)
        d["is_cool_season"] = d["month"].isin([11, 12, 1, 2]).astype(int)
        d["is_rainy_season"] = d["month"].isin([6, 7, 8, 9, 10]).astype(int)

        load = d["Load_(kWh)"]
        for lag in [1, 2, 3, 6, 12]:
            d[f"load_lag_{lag}"] = load.shift(lag)
        for window in [3, 6, 12]:
            d[f"roll_mean_{window}"] = load.shift(1).rolling(
                window=window, min_periods=1
            ).mean()
            if window in [3, 6]:
                d[f"roll_std_{window}"] = load.shift(1).rolling(
                    window=window, min_periods=1
                ).std()
        for span in [3, 6, 12]:
            d[f"ema_{span}"] = load.shift(1).ewm(span=span, min_periods=1).mean()
        d["diff_1"] = load.shift(1) - load.shift(2)
        d["diff_12"] = load.shift(1) - load.shift(13)
        d["mom_growth"] = (
            (load.shift(1) - load.shift(2)) / load.shift(2).replace(0, np.nan)
        ).clip(-2, 2)
        d["yoy_growth"] = (
            (load.shift(1) - load.shift(13)) / load.shift(13).replace(0, np.nan)
        ).clip(-2, 2)
        d["volatility_6"] = load.shift(1).rolling(window=6, min_periods=1).std()

        temp = d["Temperature_C"]
        d["temp_lag_1"] = temp.shift(1)
        d["temp_lag_2"] = temp.shift(2)
        d["temp_roll_mean_3"] = temp.shift(1).rolling(window=3, min_periods=1).mean()
        d["temp_diff_1"] = temp.shift(1) - temp.shift(2)
        d["temp_x_month_sin"] = temp * d["month_sin"]
        d["temp_x_is_peak_summer"] = temp * d["is_peak_summer"]
        d["seasonal_naive"] = load.shift(12)
        return d.dropna().reset_index(drop=True)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        lower, upper = d["Load_(kWh)"].quantile([0.01, 0.99])
        self.winsor_limits = (lower, upper)
        d["Load_(kWh)"] = d["Load_(kWh)"].clip(lower, upper)
        rolling_median = d["Load_(kWh)"].shift(1).rolling(6, min_periods=1).median()
        mad = (
            d["Load_(kWh)"].shift(1) - rolling_median
        ).abs().rolling(6, min_periods=1).median()
        upper_mad = rolling_median + 3 * 1.4826 * mad
        lower_mad = rolling_median - 3 * 1.4826 * mad
        d["Load_(kWh)"] = d["Load_(kWh)"].clip(lower=lower_mad, upper=upper_mad)
        if self.log_transform:
            d["Load_transformed"] = np.log1p(d["Load_(kWh)"])
        else:
            d["Load_transformed"] = d["Load_(kWh)"]
        return d

    @staticmethod
    def _mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def _rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def train_method(self, df_province: pd.DataFrame) -> Dict[str, Any]:
        df_proc = self._preprocess(df_province)
        df_feat = self._create_features(df_proc, is_training=True)
        exclude = [
            "Province", "Date", "Load_(kWh)", "Load_transformed",
            "year", "seasonal_naive",
        ]
        self.feature_cols = [column for column in df_feat.columns if column not in exclude]
        n_test = min(12, max(1, len(df_feat) // 5))
        df_trainval = df_feat.iloc[:-n_test].copy()
        df_test = df_feat.iloc[-n_test:].copy()
        if len(df_trainval) < 24:
            raise ValueError(
                f"Insufficient training data: {len(df_trainval)} rows. Need >= 24."
            )

        X_tv = df_trainval[self.feature_cols]
        y_tv = df_trainval["Load_transformed"]
        X_test = df_test[self.feature_cols]
        y_test_actual = df_test["Load_(kWh)"]
        X_tv_scaled = self.scaler.fit_transform(X_tv)
        X_test_scaled = self.scaler.transform(X_test)

        n_tv = len(df_trainval)
        min_train = 18
        if n_tv >= min_train + 12:
            splits = []
            for i in range(2, 0, -1):
                split_idx = n_tv - (i * 6)
                if split_idx >= min_train:
                    splits.append(split_idx)
        else:
            splits = [n_tv - 6] if n_tv - 6 >= min_train else []
        if not splits:
            splits = [int(n_tv * 0.8)]

        val_preds = {"lgb": [], "ridge": [], "naive": []}
        val_actuals = []
        for split_idx in splits:
            X_tr = X_tv.iloc[:split_idx]
            y_tr = y_tv.iloc[:split_idx]
            X_val = X_tv.iloc[split_idx:]
            y_val = y_tv.iloc[split_idx:]
            scaler_fold = StandardScaler()
            X_tr_s = scaler_fold.fit_transform(X_tr)
            X_val_s = scaler_fold.transform(X_val)
            lgb_model = lgb.LGBMRegressor(
                objective="regression", metric="rmse", boosting_type="gbdt",
                num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=5, n_estimators=200,
                verbose=-1, random_state=42,
            )
            lgb_model.fit(X_tr, y_tr)
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_tr_s, y_tr)
            p_lgb = lgb_model.predict(X_val)
            p_ridge = ridge_model.predict(X_val_s)
            p_naive = df_trainval["Load_transformed"].iloc[
                split_idx - 12:split_idx - 12 + len(X_val)
            ].values
            val_preds["lgb"].extend(p_lgb.tolist())
            val_preds["ridge"].extend(p_ridge.tolist())
            val_preds["naive"].extend(p_naive.tolist())
            val_actuals.extend(y_val.tolist())

        def objective(weights):
            w1, w2, w3 = weights
            ensemble = (
                w1 * np.array(val_preds["lgb"])
                + w2 * np.array(val_preds["ridge"])
                + w3 * np.array(val_preds["naive"])
            )
            if self.log_transform:
                ensemble = np.expm1(ensemble)
                actuals = np.expm1(np.array(val_actuals))
            else:
                actuals = np.array(val_actuals)
            return self._mape(actuals, ensemble)

        w0 = [0.5, 0.3, 0.2]
        bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
        constraints = {
            "type": "eq", "fun": lambda weights: weights[0] + weights[1] + weights[2] - 1
        }
        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        self.weights = result.x

        self.models["lgb"] = lgb.LGBMRegressor(
            objective="regression", metric="rmse", boosting_type="gbdt",
            num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5, n_estimators=200,
            verbose=-1, random_state=42,
        )
        self.models["lgb"].fit(X_tv, y_tv)
        self.models["ridge"] = Ridge(alpha=1.0, random_state=42)
        self.models["ridge"].fit(X_tv_scaled, y_tv)
        self.models["naive_values"] = y_tv.iloc[-12:].values

        p_lgb_test = self.models["lgb"].predict(X_test)
        p_ridge_test = self.models["ridge"].predict(X_test_scaled)
        p_naive_test = X_test["load_lag_12"].values
        if self.log_transform:
            p_naive_test = np.log1p(p_naive_test)
        ensemble_test = (
            self.weights[0] * p_lgb_test
            + self.weights[1] * p_ridge_test
            + self.weights[2] * p_naive_test
        )
        ensemble_test_inv = np.expm1(ensemble_test) if self.log_transform else ensemble_test
        mape = self._mape(y_test_actual, ensemble_test_inv)
        rmse = self._rmse(y_test_actual, ensemble_test_inv)
        r2 = r2_score(y_test_actual, ensemble_test_inv)
        if len(y_test_actual) >= 3:
            if self.log_transform:
                recent_actual = df_test["Load_transformed"].iloc[-3:].values
                recent_pred = ensemble_test[-3:]
            else:
                recent_actual = y_test_actual.iloc[-3:].values
                recent_pred = ensemble_test_inv[-3:]
            self.recent_bias = np.mean(recent_actual - recent_pred)
        else:
            self.recent_bias = 0.0
        return {
            "models": self.models,
            "scaler": self.scaler,
            "weights": self.weights,
            "feature_cols": self.feature_cols,
            "winsor_limits": self.winsor_limits,
            "log_transform": self.log_transform,
            "recent_bias": self.recent_bias,
            "mape": mape,
            "rmse": rmse,
            "r2": r2,
            "method_name": self.METHOD_NAME,
            "n_train": len(df_trainval),
            "n_test": len(df_test),
        }

    def predict_next_month(
        self,
        df_province: pd.DataFrame,
        trained_result: Dict[str, Any],
        next_temperature: float,
    ) -> Dict[str, Any]:
        self.models = trained_result["models"]
        self.scaler = trained_result["scaler"]
        self.weights = trained_result["weights"]
        self.feature_cols = trained_result["feature_cols"]
        self.winsor_limits = trained_result["winsor_limits"]
        self.log_transform = trained_result["log_transform"]
        self.recent_bias = trained_result["recent_bias"]
        df_proc = df_province.copy()
        lower, upper = self.winsor_limits
        df_proc["Load_(kWh)"] = df_proc["Load_(kWh)"].clip(lower, upper)
        if self.log_transform:
            df_proc["Load_transformed"] = np.log1p(df_proc["Load_(kWh)"])
        else:
            df_proc["Load_transformed"] = df_proc["Load_(kWh)"]
        df_feat = self._create_features(df_proc, is_training=False)
        last_row = df_feat.iloc[-1:].copy()
        last_date = pd.to_datetime(df_proc["Date"].iloc[-1])
        if last_date.month == 12:
            next_date = last_date.replace(year=last_date.year + 1, month=1)
        else:
            next_date = last_date.replace(month=last_date.month + 1)
        next_row = last_row.copy()
        next_row["Date"] = next_date
        next_row["year"] = next_date.year
        next_row["month"] = next_date.month
        next_row["quarter"] = (next_date.month - 1) // 3 + 1
        next_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
        next_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)
        next_row["time_idx"] = next_row["time_idx"] + 1
        next_row["is_post_covid"] = 1 if next_date >= pd.Timestamp("2020-03-01") else 0
        next_row["is_peak_summer"] = 1 if next_date.month in [3, 4, 5] else 0
        next_row["is_cool_season"] = 1 if next_date.month in [11, 12, 1, 2] else 0
        next_row["is_rainy_season"] = 1 if next_date.month in [6, 7, 8, 9, 10] else 0
        hist_load = pd.Series(df_proc["Load_(kWh)"].tolist())
        next_row["load_lag_12"] = hist_load.iloc[-12]
        next_row["load_lag_6"] = hist_load.iloc[-6]
        next_row["load_lag_3"] = hist_load.iloc[-3]
        next_row["load_lag_2"] = hist_load.iloc[-2]
        next_row["load_lag_1"] = hist_load.iloc[-1]
        next_row["roll_mean_3"] = hist_load.iloc[-3:].mean()
        next_row["roll_mean_6"] = hist_load.iloc[-6:].mean()
        next_row["roll_mean_12"] = hist_load.iloc[-12:].mean()
        next_row["roll_std_3"] = hist_load.iloc[-3:].std()
        next_row["roll_std_6"] = hist_load.iloc[-6:].std()
        next_row["ema_3"] = hist_load.ewm(span=3, min_periods=1).mean().iloc[-1]
        next_row["ema_6"] = hist_load.ewm(span=6, min_periods=1).mean().iloc[-1]
        next_row["ema_12"] = hist_load.ewm(span=12, min_periods=1).mean().iloc[-1]
        next_row["diff_1"] = hist_load.iloc[-1] - hist_load.iloc[-2]
        next_row["diff_12"] = hist_load.iloc[-1] - hist_load.iloc[-13] if len(hist_load) >= 13 else 0
        next_row["mom_growth"] = (
            ((hist_load.iloc[-1] - hist_load.iloc[-2]) / hist_load.iloc[-2]).clip(-2, 2)
            if len(hist_load) >= 2 else 0
        )
        next_row["yoy_growth"] = (
            ((hist_load.iloc[-1] - hist_load.iloc[-13]) / hist_load.iloc[-13]).clip(-2, 2)
            if len(hist_load) >= 13 else 0
        )
        next_row["volatility_6"] = hist_load.iloc[-6:].std()
        full_temp = df_proc["Temperature_C"].tolist()
        next_row["Temperature_C"] = next_temperature
        next_row["temp_lag_1"] = full_temp[-1]
        next_row["temp_lag_2"] = full_temp[-2] if len(full_temp) >= 2 else full_temp[-1]
        next_row["temp_roll_mean_3"] = pd.Series(full_temp).iloc[-3:].mean()
        next_row["temp_diff_1"] = next_temperature - full_temp[-1]
        next_row["temp_x_month_sin"] = next_temperature * next_row["month_sin"].values[0]
        next_row["temp_x_is_peak_summer"] = (
            next_temperature * next_row["is_peak_summer"].values[0]
        )
        X_next = next_row[self.feature_cols]
        X_next_scaled = self.scaler.transform(X_next)
        p_lgb = self.models["lgb"].predict(X_next)[0]
        p_ridge = self.models["ridge"].predict(X_next_scaled)[0]
        if self.log_transform:
            naive_val = hist_load.iloc[-12] if len(hist_load) >= 12 else hist_load.iloc[0]
            p_naive = np.log1p(naive_val)
        else:
            p_naive = hist_load.iloc[-12] if len(hist_load) >= 12 else hist_load.iloc[0]
        ensemble = (
            self.weights[0] * p_lgb
            + self.weights[1] * p_ridge
            + self.weights[2] * p_naive
        )
        ensemble += self.recent_bias * (0.5 if self.log_transform else 1.0)
        prediction = np.expm1(ensemble) if self.log_transform else ensemble
        prediction = max(prediction, 0)
        pred_lgb_orig = np.expm1(p_lgb) if self.log_transform else p_lgb
        pred_ridge_orig = np.expm1(p_ridge) if self.log_transform else p_ridge
        pred_naive_orig = np.expm1(p_naive) if self.log_transform else p_naive
        return {
            "prediction": float(prediction),
            "prediction_ci_lower": float(prediction * 0.85),
            "prediction_ci_upper": float(prediction * 1.15),
            "mape": float(trained_result["mape"]),
            "rmse": float(trained_result["rmse"]),
            "r2": float(trained_result["r2"]),
            "method_name": trained_result["method_name"],
            "model_contributions": {
                "lightgbm": float(pred_lgb_orig),
                "ridge": float(pred_ridge_orig),
                "seasonal_naive": float(pred_naive_orig),
                "weights": {
                    "lightgbm": float(self.weights[0]),
                    "ridge": float(self.weights[1]),
                    "seasonal_naive": float(self.weights[2]),
                },
            },
            "next_month": next_date.strftime("%Y-%m"),
            "recent_bias_applied": float(self.recent_bias),
        }


def _prepare_single_province(
    province_history: pd.DataFrame, train_cutoff: Any | None = None
) -> pd.DataFrame:
    required = ["Province", "Date", "Load_(kWh)", "Temperature_C"]
    missing = [column for column in required if column not in province_history.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    history = province_history[required].copy()
    history["Province"] = history["Province"].astype(str).str.strip()
    if history["Province"].nunique() != 1:
        raise ValueError("RAMEn train/predict accepts exactly one province")
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Load_(kWh)"] = pd.to_numeric(history["Load_(kWh)"], errors="coerce")
    history["Temperature_C"] = pd.to_numeric(history["Temperature_C"], errors="coerce")
    history = history.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if train_cutoff is not None:
        cutoff = pd.Timestamp(train_cutoff).to_period("M").to_timestamp()
        history = history[history["Date"] <= cutoff].copy().reset_index(drop=True)
    if history.empty:
        raise ValueError("No history is available through the train cutoff")
    if not np.isfinite(history[["Load_(kWh)", "Temperature_C"]].to_numpy(dtype=float)).all():
        raise ValueError("Load and temperature history must be finite")
    return history


class KimiRAMEn(ForecastMethod):
    method_id = "kimi_ramen"
    method_name = "RAMEn — Robust Adaptive Multi-Horizon Gradient Ensemble"

    def train(self, province_history: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        history = _prepare_single_province(
            province_history, train_cutoff=kwargs.get("train_cutoff")
        )
        forecaster = RAMEnForecaster()
        artifact = forecaster.train_method(history)
        artifact.update({
            "method_id": self.method_id,
            "province": str(history["Province"].iloc[0]),
            "train_cutoff": pd.Timestamp(history["Date"].iloc[-1]),
        })
        return artifact

    def predict(
        self,
        artifact: Dict[str, Any],
        province_history: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if artifact.get("method_id") != self.method_id:
            raise ValueError("Artifact does not belong to Kimi RAMEn")
        if "next_temperature" not in kwargs:
            raise ValueError("next_temperature is required")
        history = _prepare_single_province(
            province_history, train_cutoff=artifact["train_cutoff"]
        )
        if str(history["Province"].iloc[0]) != str(artifact["province"]):
            raise ValueError("Artifact province does not match prediction history")
        forecaster = RAMEnForecaster()
        result = forecaster.predict_next_month(
            history, artifact, float(kwargs["next_temperature"])
        )
        return pd.DataFrame([{
            "Method_ID": self.method_id,
            "Method_Name": self.method_name,
            "Province": artifact["province"],
            "Train_Cutoff": pd.Timestamp(artifact["train_cutoff"]),
            "Forecast_Month": pd.Period(result["next_month"]).to_timestamp(),
            "Horizon": 1,
            "Predicted_Load_kWh": result["prediction"],
            "Temperature_C": float(kwargs["next_temperature"]),
            "CI_Lower": result["prediction_ci_lower"],
            "CI_Upper": result["prediction_ci_upper"],
        }])

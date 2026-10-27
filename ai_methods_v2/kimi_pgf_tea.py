"""
PGF-TEA: Provincial Government Forecast — Trend-Exponential-Adjustment
Reference Implementation
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy import stats
import statsmodels.api as sm

from ai_methods.base import ForecastMethod

METHOD_ID = "kimi_pgf_tea"
METHOD_NAME = "PGF-TEA — Provincial Government Forecast Trend-Exponential-Adjustment"
VALIDATION_MONTHS = 12


@dataclass
class PGFTEAConfig:
    """Freeze-ready hyperparameters."""
    t_base_cdd: float = 26.0
    t_base_hdd: float = 18.0
    ewma_spans: tuple = (3, 6, 12)
    alpha_candidates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)
    outlier_threshold: float = 3.5
    hampel_window: int = 12
    min_train_months: int = 36
    coverage_adjustment: bool = False  # เปิดได้ถ้ามีข้อมูล coverage


@dataclass
class PGFTEAArtifact:
    """Model artifact ที่ส่งต่อจาก train ไป predict."""
    config: PGFTEAConfig
    province: str
    seasonal_baseline: Dict[int, float]  # month -> baseline
    beta0: float
    beta1: float
    best_alpha: float
    recent_level: float
    last_load: float
    last_month: int
    last_year: int
    coverage_stats: Optional[Dict[str, Any]] = None


def _hampel_filter(series: pd.Series, window: int, threshold: float) -> pd.Series:
    """Replace outliers with median ± threshold * MAD."""
    # Keep the working series as float so replacing integer loads with a
    # Hampel boundary does not emit thousands of dtype warnings during CV.
    result = series.astype(float).copy()
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    mask = np.abs(series - rolling_median) > threshold * rolling_mad
    result[mask] = rolling_median[mask] + np.sign(series[mask] - rolling_median[mask]) * threshold * rolling_mad[mask]
    return result


def _compute_cdd(temperature: float, base: float) -> float:
    return max(temperature - base, 0.0)


def _compute_hdd(temperature: float, base: float) -> float:
    return max(base - temperature, 0.0)


def _create_features(df: pd.DataFrame, config: PGFTEAConfig) -> pd.DataFrame:
    """Create leakage-free features from historical data."""
    df = df.copy().sort_values('Date')
    df['CDD'] = df['Temperature_C'].apply(lambda x: _compute_cdd(x, config.t_base_cdd))
    df['HDD'] = df['Temperature_C'].apply(lambda x: _compute_hdd(x, config.t_base_hdd))
    
    # Lag features
    df['load_lag1'] = df['Load_(kWh)'].shift(1)
    df['load_lag12'] = df['Load_(kWh)'].shift(12)
    
    # EWMA features
    for span in config.ewma_spans:
        df[f'load_ewma{span}'] = df['Load_(kWh)'].ewm(span=span, adjust=False).mean()
    
    # YoY growth
    df['load_yoy'] = (df['Load_(kWh)'] - df['Load_(kWh)'].shift(12)) / df['Load_(kWh)'].shift(12)
    
    # Seasonal encoding
    df['month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
    
    # Recent trend (slope from last 6 months)
    def _recent_trend(x):
        if len(x) < 2:
            return 0.0
        y = np.array(x)
        t = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(t, y)
        return slope
    
    df['recent_trend'] = df['Load_(kWh)'].rolling(window=6, min_periods=2).apply(
        _recent_trend, raw=True
    )
    
    return df


def train(
    province_history: pd.DataFrame,
    train_cutoff: str,
    optional_exogenous: Optional[Dict[str, Any]] = None
) -> PGFTEAArtifact:
    """
    Train PGF-TEA model for a single province.
    
    Parameters
    ----------
    province_history : pd.DataFrame
        Columns: Province, Date, Year, Month, Load_(kWh), Month_Num, Temperature_C
        Optional: num_reporting_agencies, total_agencies, etc.
    train_cutoff : str
        Last date in training set (YYYY-MM-DD)
    optional_exogenous : dict, optional
        Additional data if available (coverage, etc.)
    
    Returns
    -------
    PGFTEAArtifact
        Model artifact for prediction
    """
    config = PGFTEAConfig()
    if optional_exogenous and 'config_override' in optional_exogenous:
        # อนุญาต override เฉพาะบาง field ที่ไม่ใช่ hyperparameter หลัก
        pass
    
    # --- P1. Sort and filter ---
    df = province_history.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] <= pd.to_datetime(train_cutoff)].sort_values('Date')
    
    if len(df) < config.min_train_months:
        raise ValueError(f"Insufficient data: {len(df)} months < {config.min_train_months}")
    
    province = df['Province'].iloc[0]
    
    # --- P2. Outlier handling ---
    df['Load_clean'] = _hampel_filter(df['Load_(kWh)'], config.hampel_window, config.outlier_threshold)
    
    # --- P3. Create features ---
    df = _create_features(df, config)
    
    # --- Stage 1: Structural Component (OLS on detrended) ---
    # Detrend with EWMA-12
    df['trend_ewma12'] = df['Load_clean'].ewm(span=12, adjust=False).mean()
    df['detrended'] = df['Load_clean'] - df['trend_ewma12']
    
    # Drop NaN for OLS
    df_ols = df.dropna(subset=['detrended', 'CDD'])
    
    # Create month dummies (reference = December)
    month_dummies = pd.get_dummies(df_ols['Month_Num'], prefix='month', drop_first=False)
    month_dummies = month_dummies.drop(columns=['month_12'], errors='ignore')  # reference
    
    X = pd.concat([month_dummies, df_ols[['CDD']]], axis=1)
    X = sm.add_constant(X) if 'const' not in X.columns else X
    y = df_ols['detrended']
    
    # Fit OLS
    model_ols = sm.OLS(y, X).fit()
    
    # Extract coefficients
    beta0 = model_ols.params.get('const', 0.0)
    beta1 = model_ols.params.get('CDD', 0.0)
    
    # Seasonal baseline (month effects)
    seasonal_baseline = {}
    for m in range(1, 13):
        col = f'month_{m}'
        if col in model_ols.params.index:
            seasonal_baseline[m] = model_ols.params[col]
        else:
            seasonal_baseline[m] = 0.0  # reference month
    
    # --- Stage 2: Recent Adjustment (tune alpha) ---
    df['structural_pred'] = beta0
    for m in range(1, 13):
        mask = df['Month_Num'] == m
        df.loc[mask, 'structural_pred'] += seasonal_baseline.get(m, 0.0)
    df['structural_pred'] += beta1 * df['CDD']
    
    df['residual'] = df['Load_clean'] - df['structural_pred']
    
    # Rolling CV to select alpha
    best_alpha = 0.3  # default
    best_mape = float('inf')
    
    for alpha in config.alpha_candidates:
        mape_scores = []
        # Expanding window CV starting from month 37
        for i in range(config.min_train_months, len(df)):
            train_df = df.iloc[:i]
            actual = df.iloc[i]['Load_clean']
            
            # Simple exponential smoothing on residuals
            residuals = train_df['residual'].values
            if len(residuals) == 0:
                pred_adj = 0.0
            else:
                # EWMA of residuals
                weights = np.exp(alpha * np.arange(len(residuals)))
                weights /= weights.sum()
                pred_adj = np.dot(weights, residuals)
            
            pred = train_df.iloc[-1]['structural_pred'] + pred_adj
            mape_scores.append(np.abs(pred - actual) / actual * 100)
        
        avg_mape = np.mean(mape_scores)
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_alpha = alpha
    
    # Final recent level (EWMA of all residuals with best_alpha)
    residuals_all = df['residual'].values
    weights = np.exp(best_alpha * np.arange(len(residuals_all)))
    weights /= weights.sum()
    recent_level = np.dot(weights, residuals_all)
    
    # Coverage stats (if available)
    coverage_stats = None
    if config.coverage_adjustment and optional_exogenous:
        if 'num_reporting_agencies' in df.columns:
            coverage_stats = {
                'recent_avg_agencies': df['num_reporting_agencies'].tail(3).mean(),
                'total_agencies': optional_exogenous.get('total_agencies', None)
            }
    
    artifact = PGFTEAArtifact(
        config=config,
        province=province,
        seasonal_baseline=seasonal_baseline,
        beta0=beta0,
        beta1=beta1,
        best_alpha=best_alpha,
        recent_level=recent_level,
        last_load=df['Load_(kWh)'].iloc[-1],
        last_month=df['Month_Num'].iloc[-1],
        last_year=df['Year'].iloc[-1],
        coverage_stats=coverage_stats
    )
    
    return artifact


def predict(
    artifact: PGFTEAArtifact,
    province_history: pd.DataFrame,
    next_temperature: float,
    optional_exogenous: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict one-month-ahead load.
    
    Parameters
    ----------
    artifact : PGFTEAArtifact
        From train()
    province_history : pd.DataFrame
        Historical data up to current month
    next_temperature : float
        Forecast temperature for next month (from NASA/fallback)
    optional_exogenous : dict, optional
        Additional data for next month (coverage, working days, etc.)
    
    Returns
    -------
    dict
        {
            'forecast_kwh': float,
            'prediction_interval_95': (lower, upper),
            'components': {...},
            'warnings': [...]
        }
    """
    config = artifact.config
    
    # Determine next month
    next_month = artifact.last_month + 1
    next_year = artifact.last_year
    if next_month > 12:
        next_month = 1
        next_year += 1
    
    # --- Component 1: Seasonal Baseline ---
    seasonal = artifact.seasonal_baseline.get(next_month, 0.0)
    
    # --- Component 2: Temperature-Driven ---
    cdd_next = _compute_cdd(next_temperature, config.t_base_cdd)
    hdd_next = _compute_hdd(next_temperature, config.t_base_hdd)
    temp_component = artifact.beta0 + artifact.beta1 * cdd_next
    
    # --- Component 3: Recent Level Adjustment ---
    recent_adj = artifact.recent_level
    
    # --- Component 4: Coverage Adjustment (if available) ---
    coverage_adj = 0.0
    warnings = []
    
    if config.coverage_adjustment and artifact.coverage_stats:
        expected_agencies = artifact.coverage_stats['recent_avg_agencies']
        if optional_exogenous and 'expected_num_agencies_next' in optional_exogenous:
            expected_next = optional_exogenous['expected_num_agencies_next']
            if expected_agencies > 0:
                coverage_adj = (artifact.last_load / expected_agencies) * (expected_next - expected_agencies)
        else:
            warnings.append("Coverage adjustment enabled but expected_num_agencies_next not provided")
    
    # --- Final Forecast ---
    forecast = seasonal + temp_component + recent_adj + coverage_adj
    
    # Ensure non-negative
    forecast = max(forecast, 0.0)
    
    # --- Prediction Interval (from rolling residuals) ---
    # ใช้ historical residuals จาก training
    df_hist = province_history.copy()
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    df_hist = df_hist[df_hist['Date'] <= pd.to_datetime(f"{artifact.last_year}-{artifact.last_month:02d}-01")]
    
    # Recompute residuals for PI
    df_hist['CDD'] = df_hist['Temperature_C'].apply(lambda x: _compute_cdd(x, config.t_base_cdd))
    df_hist['structural'] = artifact.beta0
    for _, row in df_hist.iterrows():
        m = int(row['Month_Num'])
        df_hist.loc[_, 'structural'] += artifact.seasonal_baseline.get(m, 0.0)
    df_hist['structural'] += artifact.beta1 * df_hist['CDD']
    df_hist['residual'] = df_hist['Load_(kWh)'] - df_hist['structural']
    
    # 95% PI from last 24 months residuals
    recent_residuals = df_hist['residual'].tail(24).abs()
    q975 = recent_residuals.quantile(0.975) if len(recent_residuals) > 0 else forecast * 0.1
    
    pi_lower = max(forecast - q975, 0.0)
    pi_upper = forecast + q975
    
    return {
        'forecast_kwh': float(forecast),
        'prediction_interval_95': (float(pi_lower), float(pi_upper)),
        'components': {
            'seasonal_baseline': float(seasonal),
            'temperature_component': float(temp_component),
            'cdd_used': float(cdd_next),
            'recent_adjustment': float(recent_adj),
            'coverage_adjustment': float(coverage_adj)
        },
        'warnings': warnings,
        'metadata': {
            'province': artifact.province,
            'forecast_month': f"{next_year}-{next_month:02d}",
            'alpha_used': artifact.best_alpha,
            'beta0': artifact.beta0,
            'beta1': artifact.beta1
        }
    }


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
        raise ValueError("PGF-TEA accepts exactly one province")
    if "Month_Num" not in out:
        out["Month_Num"] = out["Date"].dt.month
    if "Year" not in out:
        out["Year"] = out["Date"].dt.year
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
    start = max(PGFTEAConfig().min_train_months, len(history) - VALIDATION_MONTHS)
    for target_idx in range(start, len(history)):
        fold = history.iloc[:target_idx].copy()
        target = history.iloc[target_idx]
        artifact = train(fold, fold["Date"].iloc[-1])
        result = predict(artifact, fold, float(target["Temperature_C"]))
        actual.append(float(target["Load_(kWh)"]))
        predicted.append(float(result["forecast_kwh"]))
    if len(actual) < 2:
        raise ValueError("PGF-TEA requires at least two validation origins")
    return _metric_values(actual, predicted)


class KimiPGFTEA(ForecastMethod):
    """Application adapter for the canonical Kimi PGF-TEA source."""

    method_id = METHOD_ID
    method_name = METHOD_NAME

    def train(self, province_history: pd.DataFrame, **kwargs):
        cutoff = pd.Timestamp(kwargs.get("train_cutoff", province_history["Date"].max()))
        history = _prepare_history(province_history, cutoff)
        model = train(history, cutoff, kwargs.get("optional_exogenous"))
        return {
            "method_id": METHOD_ID,
            "method_name": METHOD_NAME,
            "province": model.province,
            "train_cutoff": pd.Timestamp(history["Date"].iloc[-1]),
            "model": model,
            "metrics": _validation_metrics(history),
            "validation_method": "expanding_window_latest_12",
        }

    def predict(self, artifact, province_history: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if artifact.get("method_id") != METHOD_ID:
            raise ValueError("Artifact does not belong to PGF-TEA")
        history = _prepare_history(province_history, artifact["train_cutoff"])
        next_temperature = float(kwargs["next_temperature"])
        result = predict(
            artifact["model"], history, next_temperature, kwargs.get("optional_exogenous")
        )
        forecast_month = pd.Timestamp(artifact["train_cutoff"]) + pd.DateOffset(months=1)
        return pd.DataFrame([{
            "Method_ID": METHOD_ID,
            "Method_Name": METHOD_NAME,
            "Province": artifact["province"],
            "Train_Cutoff": pd.Timestamp(artifact["train_cutoff"]),
            "Forecast_Month": forecast_month,
            "Predicted_Load_kWh": max(float(result["forecast_kwh"]), 0.0),
            "Temperature_C": next_temperature,
            "PI_Lower": result["prediction_interval_95"][0],
            "PI_Upper": result["prediction_interval_95"][1],
        }])

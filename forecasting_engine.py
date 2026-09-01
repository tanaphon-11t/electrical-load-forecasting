#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production forecasting engine for monthly electricity demand.

Key rules:
- Train province-by-province.
- Forecast the true next month after the latest month in the input data.
- Build a real next-month feature row instead of reusing the last historical row.
- Select adaptive fallback strategies only from historical walk-forward validation.
- Never use future Actual_Load files inside this engine.
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

MODEL_VERSION = "production_next_month_v8_calendar_workday"
RANDOM_STATE = 42
ADAPTIVE_PROVINCES = {
    "Chanthaburi",
    "Kanchanaburi",
    "Nakhon Nayok",
    "Nonthaburi",
    "Prachin Buri",
    "Trat",
}

# Fixed-date Thai public holidays that are known before the forecast month.
# Lunar holidays are intentionally excluded because their dates vary by year
# and should only be added from a verified calendar source.
THAI_FIXED_HOLIDAYS = {
    (1, 1), (4, 6), (4, 13), (4, 14), (4, 15), (5, 1), (5, 4),
    (6, 3), (7, 28), (8, 12), (10, 23), (12, 5), (12, 10), (12, 31),
}


def _calendar_features(date):
    date = pd.Timestamp(date).to_period("M").to_timestamp()
    dates = pd.date_range(date, date + pd.offsets.MonthEnd(0), freq="D")
    weekdays = int((dates.dayofweek < 5).sum())
    weekend_days = int((dates.dayofweek >= 5).sum())
    fixed_holidays = sum(
        1 for day in dates
        if (day.month, day.day) in THAI_FIXED_HOLIDAYS and day.dayofweek < 5
    )
    effective_workdays = max(weekdays - fixed_holidays, 1)
    return {
        "days_in_month": int(len(dates)),
        "weekdays_in_month": weekdays,
        "weekend_days": weekend_days,
        "fixed_holidays_weekday": int(fixed_holidays),
        "effective_workdays": effective_workdays,
        "working_day_ratio": effective_workdays / len(dates),
    }


def _clean_load_kwh(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _prepare_history(df_province):
    df = df_province.copy()
    if df.empty:
        raise ValueError("ไม่มีข้อมูลย้อนหลังสำหรับจังหวัดนี้")
    if "Date" not in df.columns or "Load_(kWh)" not in df.columns:
        raise ValueError("ข้อมูลต้องมีคอลัมน์ Date และ Load_(kWh)")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Load_Model"] = _clean_load_kwh(df["Load_(kWh)"])
    df["Temperature_C"] = pd.to_numeric(df.get("Temperature_C", np.nan), errors="coerce")
    if "Province" in df.columns:
        df["Province"] = df["Province"].astype(str).str.strip()

    df = df.dropna(subset=["Date", "Load_Model"]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError("ข้อมูล Date หรือ Load_(kWh) ไม่ถูกต้อง")

    # The engine is monthly. Normalize dates and collapse accidental duplicate
    # province/month rows before creating lags.
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    if df.duplicated(["Date"]).any():
        aggregations = {
            "Load_Model": "sum",
            "Temperature_C": "mean",
            "Load_(kWh)": "first",
        }
        for col in df.columns:
            if col not in aggregations and col != "Date":
                aggregations[col] = "first"
        df = df.groupby("Date", as_index=False).agg(aggregations).sort_values("Date")

    expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq="MS")
    if len(expected) != len(df) or not df["Date"].reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("ข้อมูลรายเดือนมีช่วงเวลาขาดหาย กรุณาเติมเดือนที่หายก่อน train")

    if df["Temperature_C"].notna().any():
        df["Temperature_C"] = df["Temperature_C"].interpolate(limit_direction="both")
    else:
        df["Temperature_C"] = 25.0
    df = df.reset_index(drop=True)
    return df


def clip_outliers_iqr(series, whisker=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return series
    return series.clip(q1 - whisker * iqr, q3 + whisker * iqr)


def create_features(df_province, use_clipped=False):
    df = _prepare_history(df_province)
    if use_clipped:
        df["Load_Model"] = clip_outliers_iqr(df["Load_Model"])

    df["month_num"] = df["Date"].dt.month
    df["time_index"] = np.arange(len(df))
    calendar = pd.DataFrame([_calendar_features(d) for d in df["Date"]], index=df.index)
    for col in calendar.columns:
        df[col] = calendar[col]

    for lag in [1, 2, 3, 4, 5, 6, 9, 12]:
        df[f"lag{lag}"] = df["Load_Model"].shift(lag)

    for window in [2, 3, 6, 12]:
        df[f"ma{window}"] = df["Load_Model"].shift(1).rolling(window).mean()

    for span in [3, 6]:
        df[f"ema{span}"] = df["Load_Model"].shift(1).ewm(span=span, adjust=False).mean()

    df["sin_month"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["quarter"] = ((df["month_num"] - 1) // 3 + 1).astype(int)
    df["sin_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)
    denom = max(len(df) - 1, 1)
    df["trend"] = df["time_index"] / denom
    df["trend_squared"] = df["trend"] ** 2

    df["is_summer"] = df["month_num"].isin([4, 5, 6]).astype(int)
    df["is_winter"] = df["month_num"].isin([12, 1, 2]).astype(int)
    df["is_monsoon"] = df["month_num"].isin([7, 8, 9, 10]).astype(int)

    temp = df["Temperature_C"].fillna(df["Temperature_C"].median())
    df["Temperature_C"] = temp
    df["temp_sq"] = temp ** 2
    df["temp_lag1"] = temp.shift(1)
    df["temp_x_summer"] = temp * df["is_summer"]
    df["temp_x_winter"] = temp * df["is_winter"]
    df["temp_x_monsoon"] = temp * df["is_monsoon"]

    for diff in [1, 2, 3, 6]:
        df[f"diff{diff}"] = df["Load_Model"].shift(1).diff(diff)

    for window in [3, 6]:
        df[f"volatility{window}"] = df["Load_Model"].shift(1).rolling(window).std()

    for lag in [3, 6]:
        df[f"momentum{lag}"] = df["Load_Model"].shift(1) - df["Load_Model"].shift(lag + 1)

    return df


def get_candidate_models(alpha=1.0):
    return {
        f"Ridge_a{float(alpha):g}": Ridge(alpha=float(alpha)),
        "Ridge_a10": Ridge(alpha=10.0),
        "Huber_e135_a0001": HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000),
        "Huber_e15_a001": HuberRegressor(epsilon=1.5, alpha=0.001, max_iter=2000),
        "Lasso_a01": Lasso(alpha=0.01, max_iter=10000),
        "GB_n80_d2": GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.06, max_depth=2,
            random_state=RANDOM_STATE
        ),
        "ExtraTrees_n250_leaf2": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RF_n250_leaf2": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def _feature_columns(df_featured):
    # Only engine-generated features are allowed. This prevents unrelated input
    # columns from silently becoming training features and then being filled with
    # zero for the forecast month.
    prefixes = ("lag", "ma", "ema", "diff", "volatility", "momentum")
    exact = {
        "Temperature_C", "sin_month", "cos_month", "quarter",
        "sin_quarter", "cos_quarter", "trend", "trend_squared",
        "is_summer", "is_winter", "is_monsoon", "temp_sq",
        "temp_lag1", "temp_x_summer", "temp_x_winter", "temp_x_monsoon",
        "days_in_month", "weekdays_in_month", "weekend_days",
        "fixed_holidays_weekday", "effective_workdays", "working_day_ratio",
    }
    return [c for c in df_featured.columns if c in exact or c.startswith(prefixes)]


def forecast_province(df_province, test_ratio=0.2, alpha=1.0, _select_adaptive=True, **kwargs):
    if df_province is None or df_province.empty:
        raise ValueError("ไม่มีข้อมูลสำหรับจังหวัดนี้")

    best = None
    # Do not select a model against a target clipped using the full dataset:
    # that leaks validation distribution information and understates real error.
    for use_clipped in [False]:
        df_feat = create_features(df_province, use_clipped=use_clipped)
        train_df = df_feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(train_df) < 24:
            continue

        feature_cols = _feature_columns(train_df)
        test_size = max(3, min(6, int(round(len(train_df) * test_ratio / 2))))
        if len(train_df) - 3 * test_size < 24:
            test_size = max(2, (len(train_df) - 24) // 3)
        if test_size < 2:
            continue

        X = train_df[feature_cols]
        y = train_df["Load_Model"]
        splitter = TimeSeriesSplit(n_splits=3, test_size=test_size)

        for name, model in get_candidate_models(alpha=alpha).items():
            try:
                actual_parts = []
                pred_parts = []
                for train_idx, valid_idx in splitter.split(X):
                    fold_scaler = RobustScaler()
                    X_train_s = fold_scaler.fit_transform(X.iloc[train_idx])
                    X_valid_s = fold_scaler.transform(X.iloc[valid_idx])
                    fold_model = clone(model)
                    fold_model.fit(X_train_s, y.iloc[train_idx])
                    pred_parts.append(fold_model.predict(X_valid_s))
                    actual_parts.append(y.iloc[valid_idx].to_numpy())

                actual = np.concatenate(actual_parts)
                pred = np.concatenate(pred_parts)
                mape = safe_mape(actual, pred)
                if not np.isfinite(mape):
                    continue
                if best is None or mape < best["metrics_test"]["MAPE"]:
                    best = {
                        "model": clone(model),
                        "model_name": name,
                        "feature_cols": feature_cols,
                        "use_clipped": use_clipped,
                        "metrics_test": {
                            "MAPE": mape,
                            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                            "R2": float(r2_score(actual, pred)),
                        },
                        "validation_method": "expanding_window_3fold",
                    }
            except Exception:
                continue

    if best is None:
        raise ValueError("ไม่สามารถ train โมเดลได้: ข้อมูลไม่เพียงพอหรือ feature ไม่สมบูรณ์")

    df_feat = create_features(df_province, use_clipped=best["use_clipped"])
    train_df = df_feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    X_all = train_df[best["feature_cols"]]
    y_all = train_df["Load_Model"]

    final_scaler = RobustScaler()
    X_all_s = final_scaler.fit_transform(X_all)
    final_model = clone(best["model"])
    final_model.fit(X_all_s, y_all)

    best["model"] = final_model
    best["scaler"] = final_scaler
    best["training_rows_final"] = len(train_df)
    best["model_version"] = MODEL_VERSION
    best["Best_Model"] = best["model_name"]

    if _select_adaptive:
        best["adaptive_strategy"] = _select_adaptive_strategy(df_province, best, test_ratio=test_ratio, alpha=alpha)
    else:
        best["adaptive_strategy"] = {"strategy": "ml_model", "enabled": False}
    return best


def _fallback_components(df_history, target_date, ml_prediction=None):
    df = _prepare_history(df_history)
    load = df["Load_Model"].astype(float).reset_index(drop=True)
    if len(load) < 12:
        raise ValueError("ข้อมูลย้อนหลังไม่พอสำหรับ fallback")

    target_month = pd.Timestamp(target_date).month
    same_month = df[df["Date"].dt.month == target_month]["Load_Model"].astype(float)

    last_month = float(load.iloc[-1])
    recent3 = float(load.tail(3).mean())
    recent6 = float(load.tail(6).mean())
    recent3_median = float(load.tail(3).median())
    recent6_median = float(load.tail(6).median())
    target_calendar = _calendar_features(target_date)
    last_calendar = _calendar_features(df["Date"].iloc[-1])
    workday_scale = (
        target_calendar["effective_workdays"]
        / max(last_calendar["effective_workdays"], 1)
    )
    same_last = float(same_month.iloc[-1]) if len(same_month) else last_month
    same_avg = float(same_month.mean()) if len(same_month) else last_month
    same_median = float(same_month.median()) if len(same_month) else last_month

    if len(load) >= 15 and abs(float(load.iloc[-15:-12].mean())) > 1e-9:
        scale = float(load.tail(3).mean() / load.iloc[-15:-12].mean())
    else:
        scale = 1.0
    same_scaled = same_last * scale

    candidates = {
        "same_month_last_year": same_last,
        "same_month_average": same_avg,
        "same_month_median": same_median,
        "same_month_scaled": same_scaled,
        "last_month": last_month,
        "recent3_level": recent3,
        "recent6_level": recent6,
        "recent3_median": recent3_median,
        "recent6_median": recent6_median,
        "last_month_workday_scaled": last_month * workday_scale,
        "recent3_workday_scaled": recent3 * workday_scale,
        "blend_season_recent": 0.50 * same_last + 0.50 * recent3,
        "blend_scaled_recent": 0.50 * same_scaled + 0.50 * recent3,
        "robust_median_blend": float(np.median([same_last, same_avg, same_scaled, recent3, recent6])),
    }

    if ml_prediction is not None:
        ml = float(ml_prediction)
        candidates.update({
            "ml_model": ml,
            "ml70_season30": 0.70 * ml + 0.30 * same_last,
            "ml70_scaled30": 0.70 * ml + 0.30 * same_scaled,
            "ml50_scaled50": 0.50 * ml + 0.50 * same_scaled,
            "ml50_recent50": 0.50 * ml + 0.50 * recent3,
            "ml50_scaled25_recent25": 0.50 * ml + 0.25 * same_scaled + 0.25 * recent3,
            "robust_blend_ml_season_recent": float(np.median([ml, same_last, recent3])),
        })

    pct = load.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol_threshold = float(pct.abs().quantile(0.80)) if len(pct) >= 12 else 0.15
    recent_gap = abs(recent3 / same_last - 1) if abs(same_last) > 1e-9 else 0
    mom = abs(load.iloc[-1] / load.iloc[-2] - 1) if len(load) > 2 and abs(load.iloc[-2]) > 1e-9 else 0
    if mom > vol_threshold or recent_gap > max(0.20, 1.5 * vol_threshold):
        anchor = np.median([same_last, same_median, recent6_median])
        candidates["shock_aware_fallback"] = float(0.50 * anchor + 0.50 * recent3_median)
    else:
        candidates["shock_aware_fallback"] = candidates.get("robust_blend_ml_season_recent", candidates["robust_median_blend"])

    return candidates


def _historical_month_temperature(df_history, target_date):
    df = _prepare_history(df_history)
    same = df[df["Date"].dt.month == pd.Timestamp(target_date).month]["Temperature_C"].dropna()
    if len(same):
        return float(same.mean())
    recent = df["Temperature_C"].dropna()
    if len(recent):
        return float(recent.tail(3).mean())
    return 25.0


def _select_adaptive_strategy(df_province, base_result, test_ratio=0.2, alpha=1.0, validation_months=18):
    df = _prepare_history(df_province)
    if len(df) < 36:
        return {"strategy": "ml_model", "enabled": False, "validation_mape": np.nan}

    validation_dates = list(df["Date"].sort_values().unique()[-validation_months:])
    errors = {}

    for target_date in validation_dates:
        target_date = pd.Timestamp(target_date)
        hist = df[df["Date"] < target_date].copy()
        actual_row = df[df["Date"] == target_date]
        if len(hist) < 24 or actual_row.empty:
            continue

        actual = float(actual_row["Load_Model"].iloc[0])
        if abs(actual) < 1e-6:
            continue

        # Validate deterministic fallback levels over the same 18-month horizon
        # as rolling CV. Avoid recursively retraining a model for every fold.
        candidates = _fallback_components(hist, target_date, ml_prediction=None)
        for name, pred in candidates.items():
            errors.setdefault(name, []).append(abs(actual - float(pred)) / abs(actual) * 100)

    avg_errors = {k: float(np.mean(v)) for k, v in errors.items() if v}
    if not avg_errors:
        return {"strategy": "ml_model", "enabled": False, "validation_mape": np.nan}

    best_strategy = min(avg_errors, key=avg_errors.get)
    best_mape = avg_errors[best_strategy]
    ml_mape = float(base_result["metrics_test"]["MAPE"])
    compared_errors = {**avg_errors, "ml_model": ml_mape}

    # Use a fallback only when it wins the matching historical horizon clearly.
    if (
        best_strategy != "ml_model"
        and best_mape <= ml_mape - 0.50
    ):
        return {
            "strategy": best_strategy,
            "enabled": True,
            "validation_mape": best_mape,
            "ml_validation_mape": ml_mape,
            "all_strategy_mape": compared_errors,
        }

    return {
        "strategy": "ml_model",
        "enabled": False,
        "validation_mape": ml_mape,
        "candidate_mape": best_mape,
        "all_strategy_mape": compared_errors,
    }


def build_next_month_feature_row(df_province, next_temp, feature_cols=None, use_clipped=False):
    if next_temp is None or not np.isfinite(float(next_temp)):
        raise ValueError("กรุณาระบุ Temperature_C สำหรับเดือนที่จะพยากรณ์")

    df = _prepare_history(df_province)
    if len(df) < 12:
        raise ValueError("ข้อมูลย้อนหลังไม่พอ ต้องมีอย่างน้อย 12 เดือน")
    if use_clipped:
        df["Load_Model"] = clip_outliers_iqr(df["Load_Model"])

    last_date = df["Date"].max().to_period("M").to_timestamp()
    next_date = last_date + pd.DateOffset(months=1)
    next_month = int(next_date.month)
    next_time_index = len(df)
    # Match the normalization used for historical rows: index / (n - 1).
    # Therefore the genuine next row is n / (n - 1), not 1.0.
    denom = max(len(df) - 1, 1)
    load = df["Load_Model"].astype(float).reset_index(drop=True)
    temp = df["Temperature_C"].reset_index(drop=True)

    row = {
        "Temperature_C": float(next_temp),
        "sin_month": np.sin(2 * np.pi * next_month / 12),
        "cos_month": np.cos(2 * np.pi * next_month / 12),
        "quarter": int((next_month - 1) // 3 + 1),
        "sin_quarter": np.sin(2 * np.pi * int((next_month - 1) // 3 + 1) / 4),
        "cos_quarter": np.cos(2 * np.pi * int((next_month - 1) // 3 + 1) / 4),
        "trend": next_time_index / denom,
        "trend_squared": (next_time_index / denom) ** 2,
        "is_summer": int(next_month in [4, 5, 6]),
        "is_winter": int(next_month in [12, 1, 2]),
        "is_monsoon": int(next_month in [7, 8, 9, 10]),
        "temp_sq": float(next_temp) ** 2,
        "temp_lag1": float(temp.dropna().iloc[-1]) if len(temp.dropna()) else float(next_temp),
    }
    row.update(_calendar_features(next_date))
    row["temp_x_summer"] = row["Temperature_C"] * row["is_summer"]
    row["temp_x_winter"] = row["Temperature_C"] * row["is_winter"]
    row["temp_x_monsoon"] = row["Temperature_C"] * row["is_monsoon"]

    for lag in [1, 2, 3, 4, 5, 6, 9, 12]:
        row[f"lag{lag}"] = float(load.iloc[-lag]) if len(load) >= lag else np.nan
    for window in [2, 3, 6, 12]:
        row[f"ma{window}"] = float(load.tail(window).mean()) if len(load) >= window else float(load.mean())
    for span in [3, 6]:
        row[f"ema{span}"] = float(load.ewm(span=span, adjust=False).mean().iloc[-1])
    for diff in [1, 2, 3, 6]:
        row[f"diff{diff}"] = float(load.iloc[-1] - load.iloc[-1 - diff]) if len(load) > diff else 0.0
    for window in [3, 6]:
        row[f"volatility{window}"] = float(load.tail(window).std()) if len(load) >= window else 0.0
    for lag in [3, 6]:
        row[f"momentum{lag}"] = float(load.iloc[-1] - load.iloc[-1 - lag]) if len(load) > lag else 0.0

    X_next = pd.DataFrame([row])
    if feature_cols is not None:
        for col in feature_cols:
            if col not in X_next.columns:
                X_next[col] = 0.0
        X_next = X_next[feature_cols]

    X_next = X_next.replace([np.inf, -np.inf], np.nan)
    if X_next.isna().any().any():
        missing = X_next.columns[X_next.isna().any()].tolist()
        raise ValueError(f"สร้าง feature เดือนถัดไปไม่ครบ: {missing}")
    return X_next, next_date


def predict_next_month(df, province=None, next_temp=None, alpha=1.0, trained_result=None, return_feature_row=False):
    if province is not None and "Province" in df.columns:
        df_province = df[df["Province"].astype(str).str.strip() == str(province).strip()].copy()
    else:
        df_province = df.copy()
    if df_province.empty:
        raise ValueError(f"ไม่พบข้อมูลจังหวัด {province}")

    result = trained_result or forecast_province(df_province, alpha=alpha)
    X_next, next_date = build_next_month_feature_row(
        df_province,
        next_temp=next_temp,
        feature_cols=result["feature_cols"],
        use_clipped=result["use_clipped"],
    )
    pred = float(result["model"].predict(result["scaler"].transform(X_next))[0])

    adaptive = result.get("adaptive_strategy", {}) or {}
    if adaptive.get("enabled") and adaptive.get("strategy") != "ml_model":
        candidates = _fallback_components(df_province, next_date, ml_prediction=pred)
        pred = float(candidates.get(adaptive["strategy"], pred))

    if pred < 0:
        raise ValueError(f"โมเดลพยากรณ์ค่าติดลบ ({pred:.2f}) กรุณาตรวจสอบข้อมูลและอุณหภูมิ")
    if return_feature_row:
        return pred, next_date, X_next
    return pred, next_date


def forecast_all_provinces(df_all, test_ratio=0.2, alpha=1.0):
    if "Province" not in df_all.columns:
        raise ValueError("ข้อมูลต้องมีคอลัมน์ Province")
    results = {}
    for province in sorted(df_all["Province"].dropna().astype(str).str.strip().unique()):
        results[province] = forecast_province(
            df_all[df_all["Province"].astype(str).str.strip() == province],
            test_ratio=test_ratio,
            alpha=alpha,
        )
    return results


def create_summary_dataframe(results, df=None):
    rows = []
    if isinstance(results, dict):
        iterator = results.items()
    elif isinstance(results, list):
        iterator = [(r.get("Province"), r) for r in results]
    else:
        return pd.DataFrame(results)

    for province, result in iterator:
        if not result:
            continue
        metrics = result.get("metrics_test", {})
        adaptive = result.get("adaptive_strategy", {}) or {}
        rows.append({
            "Province": province,
            "Best_Model": result.get("model_name", result.get("Best_Model", "Unknown")),
            "MAPE_Test": metrics.get("MAPE", result.get("MAPE")),
            "RMSE_Test": metrics.get("RMSE"),
            "R2_Test": metrics.get("R2"),
            "Clipped": result.get("use_clipped", False),
            "Adaptive_Strategy": adaptive.get("strategy", "ml_model"),
            "Adaptive_Enabled": adaptive.get("enabled", False),
            "Model_Version": result.get("model_version", MODEL_VERSION),
        })
    return pd.DataFrame(rows)


def load_data(file_path="Dataset.xlsx"):
    return pd.read_excel(file_path)


def preprocess_data(df):
    return _prepare_history(df)


def train_models(df):
    return forecast_all_provinces(df)

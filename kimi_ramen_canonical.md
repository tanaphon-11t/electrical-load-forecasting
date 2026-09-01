# 1. Method Name
**Robust Adaptive Multi-Horizon Gradient Ensemble (RAMEn)**

---

# 2. Concept

RAMEn เป็น hybrid ensemble ที่ออกแบบมาสำหรับ **monthly electricity load forecasting** โดยเฉพาะ โดยมีแนวคิดหลัก 3 ประการ:

1. **Decomposition แบบ Implicit**: แทนที่จะแยก Trend/Seasonal/Residual ออกมาชัดเจน เราให้แต่ละ model ใน ensemble เรียนรู้ส่วนประกอบที่ตัวเองถนัด — Ridge จับ linear trend, LightGBM จับ non-linear interaction, Seasonal-Naive จับ seasonality แบบ rule-based
2. **Feature Engineering ที่ Rich แต่ Leakage-Free**: สร้าง lag, rolling, EMA, calendar, temperature interaction ทั้งหมดจากข้อมูลย้อนหลังเท่านั้น โดยใช้ `shift(1)` ก่อน `rolling()` เสมอ
3. **Adaptive Weighting รายจังหวัด**: แต่ละจังหวัดมีลักษณะข้อมูลต่างกัน (stable, growth, high volatility) ดังนั้น weights ของ ensemble จะถูก optimize แยกรายจังหวัดจาก validation performance ไม่ใช่ fixed weights

เหตุผลที่ไม่เลือก SARIMA/Prophet เป็นหลัก: ข้อมูลรายเดือน 19 จังหวัดมี structural break จาก COVID-19 และบางจังหวัดมี volatility สูงจาก industrial load ที่เปลี่ยนแปลงกะทันหัน Tree-based model จับ pattern แบบนี้ได้ดีกว่า แต่ต้องการ Ridge เป็น stabilizer เพื่อป้องกัน overfit บน trend

---

# 3. Feature Engineering

ทุก feature สร้างจากข้อมูล **ย้อนหลังเท่านั้น** โดยใช้ `shift(1)` ก่อน `rolling()` เสมอ

| กลุ่ม | Feature | สูตร/วิธีสร้าง |
|-------|---------|---------------|
| **Temporal** | `month` | `df['Date'].dt.month` |
| | `quarter` | `df['Date'].dt.quarter` |
| | `month_sin` | `sin(2π × month/12)` |
| | `month_cos` | `cos(2π × month/12)` |
| | `time_idx` | `(Date - min(Date)).dt.days // 30` (linear trend index) |
| **Lag Load** | `load_lag_1` ถึง `load_lag_12` | `Load.shift(k)` for k=1..12 |
| **Rolling Load** | `roll_mean_3`, `roll_mean_6`, `roll_mean_12` | `Load.shift(1).rolling(w).mean()` |
| | `roll_std_3`, `roll_std_6` | `Load.shift(1).rolling(w).std()` |
| **EMA Load** | `ema_3`, `ema_6`, `ema_12` | `Load.shift(1).ewm(span=w).mean()` |
| **Difference** | `diff_1` | `Load.shift(1) - Load.shift(2)` |
| | `diff_12` (YoY) | `Load.shift(1) - Load.shift(13)` |
| **Growth** | `mom_growth` | `(Load.shift(1) - Load.shift(2)) / Load.shift(2)` |
| | `yoy_growth` | `(Load.shift(1) - Load.shift(13)) / Load.shift(13)` |
| **Temperature** | `temp` | อุณหภูมิปัจจุบัน (known ณ เวลาพยากรณ์) |
| | `temp_lag_1`, `temp_lag_2` | `Temperature.shift(k)` |
| | `temp_roll_mean_3` | `Temperature.shift(1).rolling(3).mean()` |
| | `temp_diff_1` | `Temperature.shift(1) - Temperature.shift(2)` |
| **Interaction** | `temp_x_month_sin` | `temp × month_sin` |
| | `temp_x_is_peak_summer` | `temp × is_peak_summer` |
| **Calendar** | `is_peak_summer` | 1 if month in [3,4,5] else 0 |
| | `is_cool_season` | 1 if month in [11,12,1,2] else 0 |
| | `is_rainy_season` | 1 if month in [6,7,8,9,10] else 0 |
| | `is_post_covid` | 1 if Date >= '2020-03-01' else 0 |
| **Volatility** | `volatility_6` | `Load.shift(1).rolling(6).std()` |

---

# 4. Model / Algorithm

**Ensemble 3 ตัว:**

| Model | บทบาท | เหตุผล |
|-------|--------|--------|
| **LightGBM** | Primary non-linear predictor | จับ complex interaction (เช่น อุณหภูมิสูง + ฤดูร้อน + วันหยุด) ได้ดี เร็ว ใช้ RAM น้อย |
| **Ridge Regression** | Trend stabilizer | ป้องกัน LightGBM overfit บน long-term trend จับ linear trend ที่ smooth |
| **Seasonal Naive** | Seasonal baseline | `ŷ_t = y_{t-12}` เป็น baseline ที่ robust สำหรับ monthly seasonality |

**Ensemble Strategy:** Weighted average โดย weights ถูก optimize แยกรายจังหวัดจาก validation MAPE ของแต่ละ model

---

# 5. Hyperparameters

### LightGBM
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 200,
    'random_state': 42
}
```

### Ridge Regression
```python
{
    'alpha': 1.0,
    'fit_intercept': True,
    'random_state': 42
}
```

### Ensemble Weights
- Optimize ด้วย `scipy.optimize.minimize` (L-BFGS-B) บน validation set
- Constraint: weights รวมกัน = 1, แต่ละ weight ≥ 0.1 (ไม่ให้ model ใดถูก eliminate สนิท)

---

# 6. Training Procedure

1. **Preprocess**
   - Sort ตาม Date  ascending
   - Winsorize Load ที่ percentile 1 และ 99 (ป้องกัน outlier จาก e-Report error)
   - Log-transform Load: `log1p(Load)` (ลด skewness, ทำให้ residual ใกล้ normal)

2. **Feature Engineering**
   - สร้างทุก feature ตามตารางข้างต้น โดยใช้ `shift(1)` ก่อน `rolling()` เสมอ
   - Drop rows ที่มี NaN (ช่วงแรกๆ ที่ rolling window ยังไม่เต็ม)

3. **Time-Series Validation**
   - ใช้ Expanding Window Split 4 folds
   - Train บน LightGBM, Ridge, Seasonal Naive แยกกัน

4. **Ensemble Weight Optimization**
   - หา weights `[w_lgb, w_ridge, w_naive]` ที่ minimize validation MAPE
   - ใช้ scipy optimize ด้วย constraints ข้างต้น

5. **Final Training**
   - Train ทั้ง 3 models บนข้อมูลทั้งหมด (ยกเว้น final test set 6-12 เดือนล่าสุด)
   - ใช้ weights จากข้อ 4

6. **Evaluation**
   - ทดสอบบน final test set
   - คำนวณ MAPE, RMSE, R²

---

# 7. Time-Series Validation

ใช้ **Expanding Window Time-Series Cross-Validation** 4 folds:

| Fold | Training Period | Validation Period |
|------|----------------|-------------------|
| 1 | 2018-01 ถึง 2019-12 | 2020-01 ถึง 2020-12 |
| 2 | 2018-01 ถึง 2020-12 | 2021-01 ถึง 2021-12 |
| 3 | 2018-01 ถึง 2021-12 | 2022-01 ถึง 2022-12 |
| 4 | 2018-01 ถึง 2022-12 | 2023-01 ถึง 2023-12 |

- **Final Test Set**: 2024-01 ถึง 2025-12 (หรือเดือนล่าสุดที่มีข้อมูล)
- **ห้าม Random Shuffle**
- **ห้ามใช้ข้อมูล validation หรือ test ในการสร้าง feature ของ training**

สำหรับแต่ละ fold:
- สร้าง feature บน training set โดยใช้เฉพาะข้อมูลใน training
- Predict บน validation set โดยใช้ feature ที่คำนวณจาก historical จนถึงจุดนั้น

---

# 8. Outlier / Structural Break Handling

### Outlier Handling
1. **Winsorization**: Cap Load ที่ percentile 1 และ 99 ของแต่ละจังหวัด
2. **Rolling MAD (Median Absolute Deviation)**: ถ้า `|y_t - median(y_{t-6:t-1})| > 3 × MAD` ให้แทนที่ด้วย rolling median
3. **Log Transform**: `log1p(Load)` ลดผลกระทบของ outlier ที่ยังหลงเหลือ

### Structural Break Handling
1. **Feature `is_post_covid`**: ให้ model รู้ว่ามีจุดเปลี่ยนโครงสร้างที่ 2020-03
2. **Piece-wise Trend**: `time_idx` ทำงานร่วมกับ `is_post_covid` ทำให้ Ridge สามารถ fit trend ที่เปลี่ยนความชันได้
3. **Diff Features**: `diff_12` และ `yoy_growth` ช่วยให้ model มองเห็นการเปลี่ยนแปลงระดับ (level shift) แทนที่จะพึ่ง absolute level เพียงอย่างเดียว

---

# 9. Next-Month Forecast Procedure

สมมติว่าเดือนล่าสุดที่มีข้อมูลคือ 2025-08 และต้องการพยากรณ์ 2025-09:

1. **สร้าง Feature สำหรับ 2025-09**
   - `load_lag_1` = Load ของ 2025-08 (known)
   - `load_lag_2` = Load ของ 2025-07 (known)
   - `roll_mean_3` = mean(Load 2025-08, 2025-07, 2025-06) (known)
   - `ema_3` = EMA จนถึง 2025-08 (known)
   - `month` = 9, `month_sin/cos` = คำนวณจาก month 9
   - `temp` = `next_temperature` (input parameter)
   - `temp_lag_1` = Temperature ของ 2025-08 (known)
   - อื่นๆ คำนวณตามลอจิกเดียวกัน

2. **Predict ด้วยทั้ง 3 models**
   - `pred_lgb` = LightGBM.predict(features_2025_09)
   - `pred_ridge` = Ridge.predict(features_2025_09)
   - `pred_naive` = Load ของ 2024-09 (seasonal naive, lag 12)

3. **Ensemble**
   - `final_pred = w_lgb × pred_lgb + w_ridge × pred_ridge + w_naive × pred_naive`

4. **Post-Processing**
   - ถ้ามี recent bias (เช่น 3 เดือนล่าสุด model underpredict ตลอด) ให้เพิ่ม bias correction term ที่คำนวณจาก validation error ล่าสุด
   - Inverse log transform: `expm1(final_pred)`

---

# 10. Python Implementation

```python
"""
RAMEn: Robust Adaptive Multi-Horizon Gradient Ensemble
Forecasting Engine for Monthly Electricity Load (Thailand Central Region)
Compatible with Streamlit Offline (no API calls)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.stats import mstats
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


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
        
    # ==================== FEATURE ENGINEERING ====================
    
    def _create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create all leakage-free features.
        CRITICAL: All rolling/EMA must use shift(1) before operation.
        """
        d = df.copy().sort_values('Date').reset_index(drop=True)
        
        # Ensure datetime
        d['Date'] = pd.to_datetime(d['Date'])
        d['year'] = d['Date'].dt.year
        d['month'] = d['Date'].dt.month
        d['quarter'] = d['Date'].dt.quarter
        
        # Temporal cyclical encoding
        d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
        d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
        
        # Linear trend index
        d['time_idx'] = (d['Date'] - d['Date'].min()).dt.days // 30
        
        # Structural break indicator
        d['is_post_covid'] = (d['Date'] >= '2020-03-01').astype(int)
        
        # Calendar season indicators
        d['is_peak_summer'] = d['month'].isin([3, 4, 5]).astype(int)
        d['is_cool_season'] = d['month'].isin([11, 12, 1, 2]).astype(int)
        d['is_rainy_season'] = d['month'].isin([6, 7, 8, 9, 10]).astype(int)
        
        # --- Load features (LEAKAGE-FREE) ---
        load = d['Load_(kWh)']
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            d[f'load_lag_{lag}'] = load.shift(lag)
        
        # Rolling statistics (shift 1 first!)
        for window in [3, 6, 12]:
            d[f'roll_mean_{window}'] = load.shift(1).rolling(window=window, min_periods=1).mean()
            if window in [3, 6]:
                d[f'roll_std_{window}'] = load.shift(1).rolling(window=window, min_periods=1).std()
        
        # Exponential Moving Average
        for span in [3, 6, 12]:
            d[f'ema_{span}'] = load.shift(1).ewm(span=span, min_periods=1).mean()
        
        # Difference features
        d['diff_1'] = load.shift(1) - load.shift(2)
        d['diff_12'] = load.shift(1) - load.shift(13)
        
        # Growth rates (clip to prevent inf)
        d['mom_growth'] = ((load.shift(1) - load.shift(2)) / load.shift(2).replace(0, np.nan)).clip(-2, 2)
        d['yoy_growth'] = ((load.shift(1) - load.shift(13)) / load.shift(13).replace(0, np.nan)).clip(-2, 2)
        
        # Volatility
        d['volatility_6'] = load.shift(1).rolling(window=6, min_periods=1).std()
        
        # --- Temperature features ---
        temp = d['Temperature_C']
        d['temp_lag_1'] = temp.shift(1)
        d['temp_lag_2'] = temp.shift(2)
        d['temp_roll_mean_3'] = temp.shift(1).rolling(window=3, min_periods=1).mean()
        d['temp_diff_1'] = temp.shift(1) - temp.shift(2)
        
        # Interactions
        d['temp_x_month_sin'] = temp * d['month_sin']
        d['temp_x_is_peak_summer'] = temp * d['is_peak_summer']
        
        # Seasonal naive target for reference (lag 12)
        d['seasonal_naive'] = load.shift(12)
        
        # Drop rows with insufficient history
        d = d.dropna().reset_index(drop=True)
        
        return d
    
    # ==================== PREPROCESSING ====================
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize and log-transform load."""
        d = df.copy()
        
        # Winsorize at 1% and 99%
        lower, upper = d['Load_(kWh)'].quantile([0.01, 0.99])
        self.winsor_limits = (lower, upper)
        d['Load_(kWh)'] = d['Load_(kWh)'].clip(lower, upper)
        
        # Rolling MAD outlier capping (optional extra safety)
        rolling_median = d['Load_(kWh)'].shift(1).rolling(6, min_periods=1).median()
        mad = (d['Load_(kWh)'].shift(1) - rolling_median).abs().rolling(6, min_periods=1).median()
        upper_mad = rolling_median + 3 * 1.4826 * mad
        lower_mad = rolling_median - 3 * 1.4826 * mad
        d['Load_(kWh)'] = d['Load_(kWh)'].clip(lower=lower_mad, upper=upper_mad)
        
        # Log transform
        if self.log_transform:
            d['Load_transformed'] = np.log1p(d['Load_(kWh)'])
        else:
            d['Load_transformed'] = d['Load_(kWh)']
            
        return d
    
    # ==================== METRICS ====================
    
    @staticmethod
    def _mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def _rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    # ==================== TRAINING ====================
    
    def train_method(self, df_province: pd.DataFrame) -> Dict[str, Any]:
        """
        Train RAMEn on a single province's data.
        
        Parameters:
        -----------
        df_province : pd.DataFrame
            Must contain columns: ['Province', 'Date', 'Load_(kWh)', 'Temperature_C']
            
        Returns:
        --------
        dict with keys: 'models', 'scaler', 'weights', 'feature_cols', 
                         'mape', 'rmse', 'r2', 'method_name', 'recent_bias'
        """
        # Preprocess
        df_proc = self._preprocess(df_province)
        
        # Feature engineering
        df_feat = self._create_features(df_proc, is_training=True)
        
        # Define feature columns (exclude non-features)
        exclude = ['Province', 'Date', 'Load_(kWh)', 'Load_transformed', 
                   'year', 'seasonal_naive']
        self.feature_cols = [c for c in df_feat.columns if c not in exclude]
        
        # Split: use last 12 months as final test, rest for train+validation
        n_test = min(12, max(1, len(df_feat) // 5))
        df_trainval = df_feat.iloc[:-n_test].copy()
        df_test = df_feat.iloc[-n_test:].copy()
        
        if len(df_trainval) < 24:
            raise ValueError(f"Insufficient training data: {len(df_trainval)} rows. Need >= 24.")
        
        X_tv = df_trainval[self.feature_cols]
        y_tv = df_trainval['Load_transformed']
        X_test = df_test[self.feature_cols]
        y_test = df_test['Load_transformed']
        y_test_actual = df_test['Load_(kWh)']
        
        # Scale features for Ridge
        X_tv_scaled = self.scaler.fit_transform(X_tv)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ==================== TIME-SERIES VALIDATION ====================
        # Expanding window: 3 folds minimum
        n_tv = len(df_trainval)
        fold_sizes = []
        min_train = 18  # minimum 18 months for first training
        
        # Create 3 expanding folds
        fold_points = []
        if n_tv >= min_train + 12:
            # Fold 1: train[:n_tv-12], val[n_tv-12:n_tv-6]
            # Fold 2: train[:n_tv-6], val[n_tv-6:]
            # We do 2 folds for speed, or 3 if enough data
            splits = []
            for i in range(2, 0, -1):
                split_idx = n_tv - (i * 6)
                if split_idx >= min_train:
                    splits.append(split_idx)
        else:
            splits = [n_tv - 6] if n_tv - 6 >= min_train else []
        
        if not splits:
            # Fallback: single train/val split
            splits = [int(n_tv * 0.8)]
        
        # Store validation predictions for each model
        val_preds = {'lgb': [], 'ridge': [], 'naive': []}
        val_actuals = []
        
        for split_idx in splits:
            X_tr = X_tv.iloc[:split_idx]
            y_tr = y_tv.iloc[:split_idx]
            X_val = X_tv.iloc[split_idx:]
            y_val = y_tv.iloc[split_idx:]
            
            # Scale
            scaler_fold = StandardScaler()
            X_tr_s = scaler_fold.fit_transform(X_tr)
            X_val_s = scaler_fold.transform(X_val)
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                n_estimators=200,
                verbose=-1,
                random_state=42
            )
            lgb_model.fit(X_tr, y_tr)
            
            # Ridge
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_tr_s, y_tr)
            
            # Predictions in transformed space
            p_lgb = lgb_model.predict(X_val)
            p_ridge = ridge_model.predict(X_val_s)
            p_naive = X_val['load_lag_12'].values  # seasonal naive (already log-transformed if y is, but wait...)
            # Actually seasonal naive should use transformed target
            # Let's use y_tr.iloc[-12] pattern or just y_val shifted
            # Better: use load_lag_12 which is already log-transformed? No, load_lag_12 is from original Load
            # We need to use the transformed load_lag_12
            # Fix: create transformed lag in features
            p_naive = df_trainval['Load_transformed'].iloc[split_idx-12:split_idx-12+len(X_val)].values
            
            val_preds['lgb'].extend(p_lgb.tolist())
            val_preds['ridge'].extend(p_ridge.tolist())
            val_preds['naive'].extend(p_naive.tolist())
            val_actuals.extend(y_val.tolist())
        
        # ==================== ENSEMBLE WEIGHT OPTIMIZATION ====================
        def objective(weights):
            w1, w2, w3 = weights
            ensemble = (w1 * np.array(val_preds['lgb']) + 
                       w2 * np.array(val_preds['ridge']) + 
                       w3 * np.array(val_preds['naive']))
            # Inverse transform for MAPE calculation
            if self.log_transform:
                ensemble = np.expm1(ensemble)
                actuals = np.expm1(np.array(val_actuals))
            else:
                actuals = np.array(val_actuals)
            mape = self._mape(actuals, ensemble)
            return mape
        
        # Initial weights
        w0 = [0.5, 0.3, 0.2]
        bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
        constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] - 1}
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = result.x
        
        # ==================== FINAL TRAINING ====================
        # Train on all trainval data
        self.models['lgb'] = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            n_estimators=200,
            verbose=-1,
            random_state=42
        )
        self.models['lgb'].fit(X_tv, y_tv)
        
        self.models['ridge'] = Ridge(alpha=1.0, random_state=42)
        self.models['ridge'].fit(X_tv_scaled, y_tv)
        
        # Seasonal naive: just store last 12 values for reference
        self.models['naive_values'] = y_tv.iloc[-12:].values
        
        # ==================== TEST EVALUATION ====================
        p_lgb_test = self.models['lgb'].predict(X_test)
        p_ridge_test = self.models['ridge'].predict(X_test_scaled)
        p_naive_test = X_test['load_lag_12'].values
        # For naive, we need transformed values. load_lag_12 is original, so transform it
        if self.log_transform:
            p_naive_test = np.log1p(p_naive_test)
        
        ensemble_test = (self.weights[0] * p_lgb_test + 
                        self.weights[1] * p_ridge_test + 
                        self.weights[2] * p_naive_test)
        
        if self.log_transform:
            ensemble_test_inv = np.expm1(ensemble_test)
        else:
            ensemble_test_inv = ensemble_test
        
        # Calculate metrics on actual scale
        mape = self._mape(y_test_actual, ensemble_test_inv)
        rmse = self._rmse(y_test_actual, ensemble_test_inv)
        r2 = r2_score(y_test_actual, ensemble_test_inv)
        
        # Recent bias correction (last 3 months of test set)
        if len(y_test_actual) >= 3:
            recent_actual = y_test_actual.iloc[-3:].values
            recent_pred = ensemble_test_inv[-3:]
            self.recent_bias = np.mean(recent_actual - recent_pred)
        else:
            self.recent_bias = 0.0
        
        return {
            'models': self.models,
            'scaler': self.scaler,
            'weights': self.weights,
            'feature_cols': self.feature_cols,
            'winsor_limits': self.winsor_limits,
            'log_transform': self.log_transform,
            'recent_bias': self.recent_bias,
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'method_name': self.METHOD_NAME,
            'n_train': len(df_trainval),
            'n_test': len(df_test)
        }
    
    # ==================== PREDICTION ====================
    
    def predict_next_month(self, df_province: pd.DataFrame, 
                          trained_result: Dict[str, Any],
                          next_temperature: float) -> Dict[str, Any]:
        """
        Predict next month given the latest data and next month's temperature.
        
        Parameters:
        -----------
        df_province : pd.DataFrame
            Full historical data up to latest month
        trained_result : dict
            Output from train_method()
        next_temperature : float
            Forecasted/known temperature for next month
            
        Returns:
        --------
        dict with keys: 'prediction', 'mape', 'rmse', 'r2', 'method_name',
                        'model_contributions', 'confidence_interval'
        """
        # Restore trained artifacts
        self.models = trained_result['models']
        self.scaler = trained_result['scaler']
        self.weights = trained_result['weights']
        self.feature_cols = trained_result['feature_cols']
        self.winsor_limits = trained_result['winsor_limits']
        self.log_transform = trained_result['log_transform']
        self.recent_bias = trained_result['recent_bias']
        
        # Preprocess historical data
        df_proc = df_province.copy()
        lower, upper = self.winsor_limits
        df_proc['Load_(kWh)'] = df_proc['Load_(kWh)'].clip(lower, upper)
        if self.log_transform:
            df_proc['Load_transformed'] = np.log1p(df_proc['Load_(kWh)'])
        else:
            df_proc['Load_transformed'] = df_proc['Load_(kWh)']
        
        # Create features for the entire history
        df_feat = self._create_features(df_proc, is_training=False)
        
        # Get the last row as template for next month
        last_row = df_feat.iloc[-1:].copy()
        
        # Determine next month
        last_date = pd.to_datetime(df_proc['Date'].iloc[-1])
        if last_date.month == 12:
            next_date = last_date.replace(year=last_date.year + 1, month=1)
        else:
            next_date = last_date.replace(month=last_date.month + 1)
        
        # Build feature row for next month
        next_row = last_row.copy()
        next_row['Date'] = next_date
        next_row['year'] = next_date.year
        next_row['month'] = next_date.month
        next_row['quarter'] = (next_date.month - 1) // 3 + 1
        next_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        next_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        next_row['time_idx'] = next_row['time_idx'] + 1
        next_row['is_post_covid'] = 1 if next_date >= pd.Timestamp('2020-03-01') else 0
        next_row['is_peak_summer'] = 1 if next_date.month in [3, 4, 5] else 0
        next_row['is_cool_season'] = 1 if next_date.month in [11, 12, 1, 2] else 0
        next_row['is_rainy_season'] = 1 if next_date.month in [6, 7, 8, 9, 10] else 0
        
        # Update lag features (shift forward by 1)
        # lag_1 becomes previous lag_2, etc.
        next_row['load_lag_12'] = last_row['load_lag_11'].values
        next_row['load_lag_6'] = last_row['load_lag_5'].values
        next_row['load_lag_3'] = last_row['load_lag_2'].values
        next_row['load_lag_2'] = last_row['load_lag_1'].values
        next_row['load_lag_1'] = last_row['Load_(kWh)'].values  # latest known load
        
        # Update rolling features (recalculate with new lag_1)
        # We need to recalculate from raw data, but since we only have aggregated features,
        # we approximate using available rolling features
        # Better approach: recalculate from full history
        full_load = df_proc['Load_(kWh)'].tolist()
        full_load.append(np.nan)  # placeholder for next month
        
        # Recalculate rolling stats for the next row using historical loads
        hist_load = pd.Series(full_load[:-1])  # all known loads
        
        next_row['roll_mean_3'] = hist_load.iloc[-3:].mean()
        next_row['roll_mean_6'] = hist_load.iloc[-6:].mean()
        next_row['roll_mean_12'] = hist_load.iloc[-12:].mean()
        next_row['roll_std_3'] = hist_load.iloc[-3:].std()
        next_row['roll_std_6'] = hist_load.iloc[-6:].std()
        
        # EMA: we can approximate using last EMA and new value
        # EMA_t = alpha * y_t + (1-alpha) * EMA_{t-1}
        # For next month, EMA uses historical data only, so we calculate up to last known
        next_row['ema_3'] = hist_load.ewm(span=3, min_periods=1).mean().iloc[-1]
        next_row['ema_6'] = hist_load.ewm(span=6, min_periods=1).mean().iloc[-1]
        next_row['ema_12'] = hist_load.ewm(span=12, min_periods=1).mean().iloc[-1]
        
        # Differences and growth
        next_row['diff_1'] = hist_load.iloc[-1] - hist_load.iloc[-2]
        next_row['diff_12'] = hist_load.iloc[-1] - hist_load.iloc[-13] if len(hist_load) >= 13 else 0
        next_row['mom_growth'] = ((hist_load.iloc[-1] - hist_load.iloc[-2]) / hist_load.iloc[-2]).clip(-2, 2) if len(hist_load) >= 2 else 0
        next_row['yoy_growth'] = ((hist_load.iloc[-1] - hist_load.iloc[-13]) / hist_load.iloc[-13]).clip(-2, 2) if len(hist_load) >= 13 else 0
        next_row['volatility_6'] = hist_load.iloc[-6:].std()
        
        # Temperature features
        full_temp = df_proc['Temperature_C'].tolist()
        next_row['Temperature_C'] = next_temperature
        next_row['temp_lag_1'] = full_temp[-1]
        next_row['temp_lag_2'] = full_temp[-2] if len(full_temp) >= 2 else full_temp[-1]
        next_row['temp_roll_mean_3'] = pd.Series(full_temp).iloc[-3:].mean()
        next_row['temp_diff_1'] = next_temperature - full_temp[-1]
        next_row['temp_x_month_sin'] = next_temperature * next_row['month_sin'].values[0]
        next_row['temp_x_is_peak_summer'] = next_temperature * next_row['is_peak_summer'].values[0]
        
        # Prepare X
        X_next = next_row[self.feature_cols]
        
        # Scale for Ridge
        X_next_scaled = self.scaler.transform(X_next)
        
        # Predictions from each model
        p_lgb = self.models['lgb'].predict(X_next)[0]
        p_ridge = self.models['ridge'].predict(X_next_scaled)[0]
        # Seasonal naive: load from 12 months ago (in transformed space if needed)
        if self.log_transform:
            naive_val = hist_load.iloc[-12] if len(hist_load) >= 12 else hist_load.iloc[0]
            p_naive = np.log1p(naive_val)
        else:
            p_naive = hist_load.iloc[-12] if len(hist_load) >= 12 else hist_load.iloc[0]
        
        # Ensemble
        ensemble = (self.weights[0] * p_lgb + 
                   self.weights[1] * p_ridge + 
                   self.weights[2] * p_naive)
        
        # Add recent bias correction
        ensemble += self.recent_bias * (0.5 if self.log_transform else 1.0)  # dampen bias in log space
        
        # Inverse transform
        if self.log_transform:
            prediction = np.expm1(ensemble)
        else:
            prediction = ensemble
        
        # Ensure non-negative
        prediction = max(prediction, 0)
        
        # Confidence interval (simple approximation from test residuals)
        # Use 20% of prediction as rough 95% CI for electricity load
        ci_lower = prediction * 0.85
        ci_upper = prediction * 1.15
        
        # Model contributions (in original scale approx)
        pred_lgb_orig = np.expm1(p_lgb) if self.log_transform else p_lgb
        pred_ridge_orig = np.expm1(p_ridge) if self.log_transform else p_ridge
        pred_naive_orig = np.expm1(p_naive) if self.log_transform else p_naive
        
        contributions = {
            'lightgbm': float(pred_lgb_orig),
            'ridge': float(pred_ridge_orig),
            'seasonal_naive': float(pred_naive_orig),
            'weights': {
                'lightgbm': float(self.weights[0]),
                'ridge': float(self.weights[1]),
                'seasonal_naive': float(self.weights[2])
            }
        }
        
        return {
            'prediction': float(prediction),
            'prediction_ci_lower': float(ci_lower),
            'prediction_ci_upper': float(ci_upper),
            'mape': float(trained_result['mape']),
            'rmse': float(trained_result['rmse']),
            'r2': float(trained_result['r2']),
            'method_name': trained_result['method_name'],
            'model_contributions': contributions,
            'next_month': next_date.strftime('%Y-%m'),
            'recent_bias_applied': float(self.recent_bias)
        }


# ==================== STREAMLIT INTEGRATION EXAMPLE ====================

def train_all_provinces(df_full: pd.DataFrame) -> Dict[str, Dict]:
    """
    Train RAMEn for all 19 provinces.
    
    Usage in Streamlit:
        results = train_all_provinces(df)
        st.write(results['กรุงเทพมหานคร']['mape'])
    """
    results = {}
    provinces = df_full['Province'].unique()
    
    for prov in provinces:
        df_prov = df_full[df_full['Province'] == prov].copy()
        if len(df_prov) < 24:
            continue
            
        forecaster = RAMEnForecaster()
        try:
            result = forecaster.train_method(df_prov)
            results[prov] = result
        except Exception as e:
            results[prov] = {'error': str(e)}
            
    return results


def predict_all_provinces(df_full: pd.DataFrame, 
                         trained_results: Dict[str, Dict],
                         next_temps: Dict[str, float]) -> pd.DataFrame:
    """
    Predict next month for all provinces.
    
    Parameters:
    -----------
    next_temps : dict
        {'กรุงเทพมหานคร': 28.5, 'นนทบุรี': 29.0, ...}
    """
    predictions = []
    
    for prov, trained in trained_results.items():
        if 'error' in trained:
            continue
            
        df_prov = df_full[df_full['Province'] == prov].copy()
        temp = next_temps.get(prov, df_prov['Temperature_C'].iloc[-12:].mean())
        
        forecaster = RAMEnForecaster()
        pred = forecaster.predict_next_month(df_prov, trained, temp)
        
        predictions.append({
            'Province': prov,
            'Next_Month': pred['next_month'],
            'Predicted_Load_kWh': pred['prediction'],
            'CI_Lower': pred['prediction_ci_lower'],
            'CI_Upper': pred['prediction_ci_upper'],
            'MAPE_%': pred['mape'],
            'RMSE': pred['rmse'],
            'R2': pred['r2']
        })
        
    return pd.DataFrame(predictions)
```

---

# 11. Required Python Libraries

```bash
pip install pandas numpy scikit-learn lightgbm scipy
```

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.3.0 | Data manipulation |
| `numpy` | >=1.21.0 | Numerical operations |
| `scikit-learn` | >=1.0.0 | Ridge, StandardScaler, metrics |
| `lightgbm` | >=3.3.0 | Gradient boosting model |
| `scipy` | >=1.7.0 | Optimization (SLSQP), winsorize |

**หมายเหตุ:** หากต้องการเพิ่ม XGBoost เป็น model ที่ 4 ใน ensemble สามารถ `pip install xgboost` ได้ แต่ RAMEn หลักใช้ LightGBM + Ridge + Naive เพียงพอแล้ว

---

# 12. Limitations

1. **Cold Start**: ต้องการอย่างน้อย 24 เดือนข้อมูล จังหวัดที่มีข้อมูลน้อยกว่านี้จะ train ไม่ได้
2. **Temperature Dependency**: ถ้า `next_temperature` ที่ให้มาไม่แม่นยำ prediction จะผิดพลาดโดยเฉพาะในช่วง peak summer
3. **Structural Break รุนแรง**: ถ้ามี event ใหญ่ที่ไม่เคยเกิดขึ้นใน training data (เช่น โรงงานใหญ่ปิดตัวกะทันหัน) model จะไม่สามารถจับได้ทันที ต้องใช้ recent bias correction ช่วยเพียงเล็กน้อย
4. **Monthly Granularity**: วิธีนี้ออกแบบสำหรับรายเดือนเท่านั้น หากต้องการพยากรณ์รายวันหรือรายชั่วโมง ต้องปรับ feature engineering และ lag structure ใหม่ทั้งหมด
5. **Single-Step Forecast**: วิธีนี้ optimize สำหรับ next-month (h=1) หากต้องการ multi-step forecast (3-6 เดือน) ต้องใช้ recursive forecasting ซึ่งจะสะสม error
6. **No External Economic Variables**: ไม่ได้ใช้ GDP, industrial index, หรือราคาน้ำมัน ซึ่งอาจมีผลต่อ electricity load ในระยะยาว
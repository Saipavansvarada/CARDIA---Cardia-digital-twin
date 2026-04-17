"""
CARDIA — Ensemble CES Scoring Model
XGBoost + Random Forest + LightGBM voting together.
3 models agreeing = more reliable CES score.
Output: models/ces_ensemble.pkl
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config import PATHS, ML

os.makedirs("models", exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "hr", "hrv_rmssd", "activity_load",
    "lf_hf_ratio", "sleep_quality", "resting_hr",
    "hr_zone", "fatigue", "is_workout",
    "spo2", "temperature", "rr_interval",
]
TARGET = "ces_score"

print("=" * 50)
print("CARDIA — Ensemble CES Model Training")
print("=" * 50)
print()

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = PATHS["combined_data"]
if not os.path.exists(data_path):
    data_path = PATHS["synthetic_data"]
    print("Using synthetic data (run physionet_parser.py for better results)")

print(f"Loading: {data_path}")
df = pd.read_csv(data_path)

# Keep only columns that exist
available = [f for f in FEATURES if f in df.columns]
print(f"Loaded {len(df)} samples | Features: {len(available)}")
print()

X = df[available].values
y = df[TARGET].values

# ── Split + Scale ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled   = scaler.transform(X_test)

print("Training 3 models...")
print()

# ── Model 1: XGBoost ──────────────────────────────────────────────────────────
print("  1/3 XGBoost...")
xgb = XGBRegressor(
    n_estimators=ML["ces_n_estimators"],
    max_depth=ML["ces_max_depth"],
    learning_rate=ML["ces_learning_rate"],
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
print(f"     MAE: {mean_absolute_error(y_test, xgb_pred):.2f} | "
      f"R2: {r2_score(y_test, xgb_pred):.4f}")

# ── Model 2: Random Forest ────────────────────────────────────────────────────
print("  2/3 Random Forest...")
rf = RandomForestRegressor(
    n_estimators=ML["ces_n_estimators"],
    max_depth=10,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
print(f"     MAE: {mean_absolute_error(y_test, rf_pred):.2f} | "
      f"R2: {r2_score(y_test, rf_pred):.4f}")

# ── Model 3: LightGBM ─────────────────────────────────────────────────────────
print("  3/3 LightGBM...")
lgbm = LGBMRegressor(
    n_estimators=ML["ces_n_estimators"],
    max_depth=ML["ces_max_depth"],
    learning_rate=ML["ces_learning_rate"],
    subsample=0.8,
    random_state=42,
    verbose=-1,
)
lgbm.fit(X_train_scaled, y_train)
lgbm_pred = lgbm.predict(X_test_scaled)
print(f"     MAE: {mean_absolute_error(y_test, lgbm_pred):.2f} | "
      f"R2: {r2_score(y_test, lgbm_pred):.4f}")

# ── Ensemble ──────────────────────────────────────────────────────────────────
print()
ensemble_pred = (xgb_pred + rf_pred + lgbm_pred) / 3
ens_mae = mean_absolute_error(y_test, ensemble_pred)
ens_r2  = r2_score(y_test, ensemble_pred)
print(f"  Ensemble MAE: {ens_mae:.2f} | R2: {ens_r2:.4f}  <- best!")

# ── Feature Importance ────────────────────────────────────────────────────────
importance = dict(sorted(
    zip(available, xgb.feature_importances_),
    key=lambda x: x[1], reverse=True
))
print()
print("Feature Importance:")
for feat, imp in importance.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<20} {bar} {imp:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    "xgb_model":          xgb,
    "rf_model":           rf,
    "lgbm_model":         lgbm,
    "scaler":             scaler,
    "features":           available,
    "feature_importance": importance,
    "metrics": {
        "xgb_mae":      round(mean_absolute_error(y_test, xgb_pred), 2),
        "rf_mae":       round(mean_absolute_error(y_test, rf_pred), 2),
        "lgbm_mae":     round(mean_absolute_error(y_test, lgbm_pred), 2),
        "ensemble_mae": round(ens_mae, 2),
        "ensemble_r2":  round(ens_r2, 4),
    },
}

with open(PATHS["ces_model"], "wb") as f:
    pickle.dump(bundle, f)

print()
print(f"Saved to: {PATHS['ces_model']}")
print()

# ── Quick test ────────────────────────────────────────────────────────────────
print("Quick test prediction:")
test_input = np.array([[145, 32, 7.2, 2.8, 6.5, 68, 4, 3.0, 1, 97.5, 37.1, 414]])
# Pad or trim to match available features
test_input = test_input[:, :len(available)]
test_scaled = scaler.transform(test_input)

xgb_out  = xgb.predict(test_scaled)[0]
rf_out   = rf.predict(test_scaled)[0]
lgbm_out = lgbm.predict(test_scaled)[0]
final    = (xgb_out + rf_out + lgbm_out) / 3

print(f"  Input:      HR=145, HRV=32, Load=7.2, SpO2=97.5")
print(f"  XGBoost:    CES = {xgb_out:.1f}")
print(f"  RandForest: CES = {rf_out:.1f}")
print(f"  LightGBM:   CES = {lgbm_out:.1f}")
print(f"  ENSEMBLE:   CES = {final:.1f}  <- used in app")
print()
print("CES Ensemble Model ready!")


# ── Prediction function (used by main.py) ─────────────────────────────────────
def predict_ces(reading: dict) -> dict:
    """
    Predict CES score from live sensor reading.
    Args:
        reading: dict from esp32_wifi_reader.get_reading()
    Returns:
        dict with ces_score + individual model scores
    """
    with open(PATHS["ces_model"], "rb") as f:
        bundle = pickle.load(f)

    features = bundle["features"]
    values   = np.array([[reading.get(f, 0) for f in features]])
    scaled   = bundle["scaler"].transform(values)

    xgb_s  = bundle["xgb_model"].predict(scaled)[0]
    rf_s   = bundle["rf_model"].predict(scaled)[0]
    lgbm_s = bundle["lgbm_model"].predict(scaled)[0]
    ens    = (xgb_s + rf_s + lgbm_s) / 3

    return {
        "ces_score":  round(float(np.clip(ens,    0, 100)), 1),
        "xgb_score":  round(float(np.clip(xgb_s,  0, 100)), 1),
        "rf_score":   round(float(np.clip(rf_s,   0, 100)), 1),
        "lgbm_score": round(float(np.clip(lgbm_s, 0, 100)), 1),
        "confidence": round(float(
            1 - np.std([xgb_s, rf_s, lgbm_s]) / 10), 3),
    }
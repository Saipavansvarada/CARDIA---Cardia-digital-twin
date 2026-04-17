"""
CARDIA — Isolation Forest Safety Detector
Detects dangerous cardiac anomalies.
Fires: SAFE / CAUTION / DANGER alerts.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import PATHS, ML, SAFETY, USER

os.makedirs("models", exist_ok=True)

MAX_HR     = USER["max_hr"]
RESTING_HR = USER["resting_hr"]

FEATURES = ["hr", "hrv_rmssd", "activity_load", "spo2", "rr_interval"]

print("=" * 50)
print("CARDIA — Isolation Forest Training")
print("=" * 50)
print()

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = PATHS["combined_data"]
if not os.path.exists(data_path):
    data_path = PATHS["synthetic_data"]
    print("Using synthetic data...")

print(f"Loading: {data_path}")
df = pd.read_csv(data_path)

# Use only safe samples to define "normal"
safe_df = df[df["safety_label"] == 0].copy()
available = [f for f in FEATURES if f in safe_df.columns]
print(f"Normal samples: {len(safe_df)} | Features: {len(available)}")
print()

X_safe = safe_df[available].values

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_safe)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Isolation Forest on normal cardiac data...")
iso = IsolationForest(
    contamination=ML["iso_contamination"],
    n_estimators=ML["iso_n_estimators"],
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_scaled)
print("Training complete!")

# ── Evaluate on full dataset ──────────────────────────────────────────────────
all_available = [f for f in available if f in df.columns]
X_all    = df[all_available].values
X_all_sc = scaler.transform(X_all)
preds    = iso.predict(X_all_sc)   # 1=normal, -1=anomaly

detected = (preds == -1).sum()
total    = len(preds)
print(f"\nAnomaly detection rate: {detected}/{total} "
      f"({detected/total*100:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    "model":    iso,
    "scaler":   scaler,
    "features": available,
}
with open(PATHS["iso_model"], "wb") as f:
    pickle.dump(bundle, f)
print(f"Saved to: {PATHS['iso_model']}")

# ── Test predictions ──────────────────────────────────────────────────────────
print("\nTest predictions:")
tests = [
    {"label": "Normal resting",   "values": [70,  45, 1.5, 98.5, 857]},
    {"label": "Hard exercise",    "values": [155, 28, 8.2, 97.1, 387]},
    {"label": "Danger zone",      "values": [178, 16, 9.8, 91.0, 337]},
]
for t in tests:
    vals   = np.array([t["values"][:len(available)]])
    scaled = scaler.transform(vals)
    pred   = iso.predict(scaled)[0]
    score  = iso.decision_function(scaled)[0]
    status = "ANOMALY ⚠️" if pred == -1 else "Normal ✅"
    print(f"  {t['label']:<22} → {status} (score: {score:.3f})")

print()
print("Isolation Forest ready!")


# ── Prediction class (used by main.py) ────────────────────────────────────────
class AnomalyDetector:
    """
    Used by main.py for real-time anomaly detection.

    Usage:
        detector = AnomalyDetector()
        result   = detector.predict(hr=145, hrv=28, load=8.2, spo2=97, rr=414)
    """

    def __init__(self):
        if not os.path.exists(PATHS["iso_model"]):
            raise FileNotFoundError(
                "Run isolation_forest.py first to train the model!")
        with open(PATHS["iso_model"], "rb") as f:
            bundle = pickle.load(f)
        self.model    = bundle["model"]
        self.scaler   = bundle["scaler"]
        self.features = bundle["features"]

    def predict(self, hr, hrv, load, spo2=98, rr=800):
        """
        Returns safety status dict.
        anomaly: -1 = anomaly detected, 1 = normal
        safety:  SAFE / CAUTION / DANGER
        """
        vals   = {"hr":hr,"hrv_rmssd":hrv,"activity_load":load,
                  "spo2":spo2,"rr_interval":rr}
        X      = np.array([[vals.get(f,0) for f in self.features]])
        Xsc    = self.scaler.transform(X)
        pred   = int(self.model.predict(Xsc)[0])
        score  = float(self.model.decision_function(Xsc)[0])

        # Rule-based safety on top of Isolation Forest
        hr_pct = hr / MAX_HR * 100
        if (hr_pct >= SAFETY["danger_hr_pct"] or
            hrv     <  SAFETY["danger_hrv"]    or
            load    >  SAFETY["danger_load"]   or
            spo2    <  SAFETY["danger_spo2"]   or
            pred    == -1):
            safety = "DANGER"
        elif (hr_pct >= SAFETY["caution_hr_pct"] or
              hrv     <  SAFETY["caution_hrv"]   or
              load    >  SAFETY["caution_load"]  or
              spo2    <  SAFETY["caution_spo2"]):
            safety = "CAUTION"
        else:
            safety = "SAFE"

        return {
            "anomaly": pred,
            "safety":  safety,
            "score":   round(score, 3),
        }


if __name__ == "__main__":
    pass
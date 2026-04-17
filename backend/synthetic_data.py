"""
CARDIA — Synthetic Data Generator
Generates realistic cardiac training data for all ML models.
"""

import numpy as np
import pandas as pd
import os
from config import USER, SAFETY, PATHS, HR_ZONES, STRATEGIES

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)

# ── Settings ──────────────────────────────────────────────────────────────────
N_DAYS           = 90
SAMPLES_PER_DAY  = 48
N_SAMPLES        = N_DAYS * SAMPLES_PER_DAY
RESTING_HR       = USER["resting_hr"]
MAX_HR           = USER["max_hr"]
BASE_HRV         = 45.0

print("=" * 50)
print("CARDIA — Synthetic Data Generator")
print("=" * 50)
print(f"Generating {N_SAMPLES} samples over {N_DAYS} days...")
print()

records = []

for i in range(N_SAMPLES):
    day         = i // SAMPLES_PER_DAY
    time_of_day = (i % SAMPLES_PER_DAY) / SAMPLES_PER_DAY

    circadian = np.sin(time_of_day * 2 * np.pi - np.pi / 2) * 8

    is_workout_day  = (day % 7) in [1, 3, 5]
    workout_window  = (0.35 < time_of_day < 0.55)
    is_workout      = is_workout_day and workout_window

    if is_workout:
        intensity = np.random.uniform(0.6, 0.92)
        hr        = RESTING_HR + (MAX_HR - RESTING_HR) * intensity
        hr        += np.random.normal(0, 5)
        load      = intensity * 10
        hrv       = BASE_HRV * (1 - intensity * 0.55)
    else:
        hr        = RESTING_HR + circadian + np.random.normal(0, 3)
        load      = np.random.uniform(0.5, 3.5)
        hrv       = BASE_HRV + np.random.normal(0, 8)

    fatigue = 0.0
    if day % 7 in [4, 5, 6]:
        fatigue = np.random.uniform(2, 7)
    hrv  -= fatigue
    hr   += fatigue * 0.5

    fitness_gain = day * 0.06
    hrv         += fitness_gain
    hr          -= fitness_gain * 0.08

    hr   = float(np.clip(hr,   40, 200))
    hrv  = float(np.clip(hrv,   8, 120))
    load = float(np.clip(load,  0,  10))

    hr_pct = hr / MAX_HR * 100
    zone   = (0 if hr_pct < 50 else
              1 if hr_pct < 60 else
              2 if hr_pct < 70 else
              3 if hr_pct < 80 else
              4 if hr_pct < 90 else 5)

    spo2       = float(np.clip(99 - max(0, (hr-160)*0.05) + np.random.normal(0, 0.3), 90, 100))
    lf_hf      = float(np.clip(2.0 + load*0.18 + np.random.normal(0, 0.2), 0.5, 8.0))
    sleep      = float(np.clip(np.random.uniform(5, 9) - load*0.3, 2, 10))
    resting_hr = float(np.clip(RESTING_HR - fitness_gain*0.05 + np.random.normal(0, 2), 45, 90))
    temperature= float(np.clip(36.5 + hr*0.003 + np.random.normal(0, 0.1), 35.5, 39.5))
    rr_interval= round(60000 / hr + np.random.normal(0, 15), 1)

    hr_score  = max(0, min(40, (MAX_HR - hr) / (MAX_HR - RESTING_HR) * 40))
    hrv_score = max(0, min(35, hrv / 80 * 35))
    load_pen  = max(0, (load - 7) * 3.5)
    ces       = round(max(0, min(100, hr_score + hrv_score + 25 - load_pen)))

    if hr_pct >= SAFETY["danger_hr_pct"] or hrv < SAFETY["danger_hrv"] or load > SAFETY["danger_load"] or spo2 < SAFETY["danger_spo2"]:
        safety = 2
    elif hr_pct >= SAFETY["caution_hr_pct"] or hrv < SAFETY["caution_hrv"] or load > SAFETY["caution_load"] or spo2 < SAFETY["caution_spo2"]:
        safety = 1
    else:
        safety = 0

    strategy = 2 if (ces < 40 or load > 8) else 1 if hrv < 30 else 0

    records.append({
        "day": day, "time_of_day": round(time_of_day, 4),
        "hr": round(hr, 1), "hrv_rmssd": round(hrv, 1),
        "activity_load": round(load, 2), "hr_zone": zone,
        "spo2": round(spo2, 1), "lf_hf_ratio": round(lf_hf, 3),
        "sleep_quality": round(sleep, 1), "resting_hr": round(resting_hr, 1),
        "temperature": round(temperature, 1), "rr_interval": round(rr_interval, 1),
        "is_workout": int(is_workout), "fatigue": round(fatigue, 2),
        "fitness_gain": round(fitness_gain, 2), "ces_score": ces,
        "safety_label": safety, "strategy_label": strategy, "source": "synthetic",
    })

df = pd.DataFrame(records)
df.to_csv(PATHS["synthetic_data"], index=False)

print(f"Generated {len(df)} samples across {N_DAYS} days")
print(f"Saved to: {PATHS['synthetic_data']}")
print()
print("Dataset Summary:")
print(f"  HR range:        {df['hr'].min():.0f} - {df['hr'].max():.0f} BPM")
print(f"  HRV range:       {df['hrv_rmssd'].min():.0f} - {df['hrv_rmssd'].max():.0f} ms")
print(f"  SpO2 range:      {df['spo2'].min():.0f} - {df['spo2'].max():.0f} %")
print(f"  CES range:       {df['ces_score'].min()} - {df['ces_score'].max()}")
print(f"  Safe samples:    {(df['safety_label']==0).sum()}")
print(f"  Caution samples: {(df['safety_label']==1).sum()}")
print(f"  Danger samples:  {(df['safety_label']==2).sum()}")
print()
print("Ready to train ML models!")
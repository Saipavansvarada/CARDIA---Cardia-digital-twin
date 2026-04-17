"""
CARDIA — PhysioNet Data Parser
Reads MIT-BIH real cardiac data + combines with synthetic data.
Output: data/combined_data.csv — used to train ALL ML models.

Before running:
1. pip install wfdb
2. Download MIT-BIH from physionet.org/content/mitdb/1.0.0/
3. Extract into: data/physionet/
"""

import numpy as np
import pandas as pd
import os
from config import PATHS, USER, SAFETY

os.makedirs("data", exist_ok=True)

RESTING_HR = USER["resting_hr"]
MAX_HR     = USER["max_hr"]

MITBIH_RECORDS = [
    "100","101","102","103","104","105","106","107",
    "108","109","111","112","113","114","115","116",
    "117","118","119","121","122","123","124","200",
    "201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

def compute_rmssd(rr_ms):
    if len(rr_ms) < 2: return 45.0
    return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

def compute_pnn50(rr_ms):
    if len(rr_ms) < 2: return 20.0
    return float(np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms) * 100)

def get_zone(hr):
    pct = hr / MAX_HR * 100
    return (0 if pct<50 else 1 if pct<60 else 2 if pct<70 else
            3 if pct<80 else 4 if pct<90 else 5)

def get_safety(hr, hrv, load, spo2=98):
    pct = hr / MAX_HR * 100
    if (pct>=SAFETY["danger_hr_pct"] or hrv<SAFETY["danger_hrv"] or
        load>SAFETY["danger_load"]   or spo2<SAFETY["danger_spo2"]): return 2
    if (pct>=SAFETY["caution_hr_pct"] or hrv<SAFETY["caution_hrv"] or
        load>SAFETY["caution_load"]   or spo2<SAFETY["caution_spo2"]): return 1
    return 0

def compute_ces(hr, hrv, load):
    hr_score  = max(0, min(40, (MAX_HR-hr)/(MAX_HR-RESTING_HR)*40))
    hrv_score = max(0, min(35, hrv/80*35))
    load_pen  = max(0, (load-7)*3.5)
    return round(max(0, min(100, hr_score+hrv_score+25-load_pen)))

def extract_from_record(record_name):
    try:
        import wfdb
    except ImportError:
        print("  wfdb not installed: pip install wfdb")
        return []

    record_path = os.path.join(PATHS["physionet_data"], record_name)
    results     = []
    try:
        ann     = wfdb.rdann(record_path, "atr")
        record  = wfdb.rdrecord(record_path)
        fs      = record.fs
        normal  = ["N","L","R","B","A","a","J","S","e","j"]
        r_peaks = [ann.sample[i] for i,s in enumerate(ann.symbol) if s in normal]
        if len(r_peaks) < 10: return []

        rr = np.diff(r_peaks) / fs * 1000
        window, step = 30, 10

        for i in range(0, len(rr)-window, step):
            w_rr = rr[i:i+window]
            w_rr = w_rr[(w_rr>300)&(w_rr<2000)]
            if len(w_rr) < 10: continue

            hr        = round(60000/np.mean(w_rr), 1)
            hrv_rmssd = round(compute_rmssd(w_rr), 1)
            if not (30<=hr<=200): continue
            if not (5<=hrv_rmssd<=150): continue

            load        = round(max(0, min(10,(hr-50)/15)), 2)
            lf_hf       = float(np.clip(2.0+load*0.2+np.random.normal(0,0.15), 0.5, 8.0))
            sleep       = float(np.clip(np.random.uniform(5,9)-load*0.2, 2, 10))
            resting_hr  = float(np.clip(hr*0.65+np.random.normal(0,3), 45, 90))
            fatigue     = round(max(0,load-5)*0.8, 2)
            spo2        = float(np.clip(99-max(0,(hr-160)*0.05)+np.random.normal(0,0.3), 90, 100))
            temperature = float(np.clip(36.5+hr*0.003+np.random.normal(0,0.1), 35.5, 39.5))
            rr_interval = round(60000/hr+np.random.normal(0,15), 1)
            ces         = compute_ces(hr, hrv_rmssd, load)
            safety      = get_safety(hr, hrv_rmssd, load, spo2)
            strategy    = 2 if (ces<40 or load>8) else 1 if hrv_rmssd<30 else 0

            results.append({
                "day":0, "time_of_day":0.5,
                "hr":hr, "hrv_rmssd":hrv_rmssd,
                "activity_load":load, "hr_zone":get_zone(hr),
                "spo2":round(spo2,1), "lf_hf_ratio":round(lf_hf,3),
                "sleep_quality":round(sleep,1), "resting_hr":round(resting_hr,1),
                "temperature":round(temperature,1), "rr_interval":round(rr_interval,1),
                "is_workout":int(load>5), "fatigue":fatigue,
                "fitness_gain":0.0, "ces_score":ces,
                "safety_label":safety, "strategy_label":strategy,
                "source":"physionet",
            })
    except Exception as e:
        print(f"    Skipping {record_name}: {e}")
    return results

def load_physionet():
    print("Processing PhysioNet MIT-BIH records...")
    all_records = []
    success     = 0
    for i, rec in enumerate(MITBIH_RECORDS):
        print(f"  [{i+1:02d}/{len(MITBIH_RECORDS)}] Record {rec}...", end=" ")
        records = extract_from_record(rec)
        if records:
            all_records.extend(records)
            print(f"{len(records)} samples")
            success += 1
        else:
            print("skipped")
    print(f"\nProcessed {success}/{len(MITBIH_RECORDS)} records")
    print(f"Total PhysioNet samples: {len(all_records)}")
    return pd.DataFrame(all_records)

def combine_datasets():
    print("=" * 50)
    print("CARDIA — PhysioNet Data Parser")
    print("=" * 50)
    print()

    # Load synthetic
    print("Loading synthetic data...")
    if not os.path.exists(PATHS["synthetic_data"]):
        print("ERROR: Run synthetic_data.py first!")
        return None
    synthetic_df = pd.read_csv(PATHS["synthetic_data"])
    print(f"Synthetic samples: {len(synthetic_df)}")

    # Check PhysioNet
    physionet_exists = os.path.exists(PATHS["physionet_data"])
    physionet_files  = []
    if physionet_exists:
        physionet_files = [f for f in os.listdir(PATHS["physionet_data"])
                           if f.endswith(".hea")]

    if not physionet_files:
        print()
        print("PhysioNet data not found — using synthetic only")
        print("To add later:")
        print("  1. physionet.org/content/mitdb/1.0.0/")
        print("  2. Download ZIP")
        print(f"  3. Extract to: {PATHS['physionet_data']}")
        print("  4. Run again")
        synthetic_df.to_csv(PATHS["combined_data"], index=False)
        combined = synthetic_df
    else:
        print()
        physionet_df = load_physionet()
        cols = [
            "hr","hrv_rmssd","activity_load","hr_zone",
            "spo2","lf_hf_ratio","sleep_quality","resting_hr",
            "temperature","rr_interval","is_workout","fatigue",
            "fitness_gain","ces_score","safety_label",
            "strategy_label","source"
        ]
        if len(physionet_df) == 0:
            combined = synthetic_df
        else:
            combined = pd.concat([
                synthetic_df[cols],
                physionet_df[cols]
            ], ignore_index=True)
            combined = combined.dropna()
            combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        combined.to_csv(PATHS["combined_data"], index=False)

    print()
    print("Combined Dataset Summary:")
    print(f"  Total samples:  {len(combined)}")
    if "source" in combined.columns:
        for src, cnt in combined["source"].value_counts().items():
            print(f"  {src+':':<20} {cnt} samples")
    print(f"  HR range:       {combined['hr'].min():.0f} - {combined['hr'].max():.0f} BPM")
    print(f"  HRV range:      {combined['hrv_rmssd'].min():.0f} - {combined['hrv_rmssd'].max():.0f} ms")
    print(f"  CES range:      {combined['ces_score'].min()} - {combined['ces_score'].max()}")
    print(f"  Safe:           {(combined['safety_label']==0).sum()}")
    print(f"  Caution:        {(combined['safety_label']==1).sum()}")
    print(f"  Danger:         {(combined['safety_label']==2).sum()}")
    print()
    print(f"Saved to: {PATHS['combined_data']}")
    print()
    print("Ready to train ML models!")
    return combined

if __name__ == "__main__":
    combine_datasets()
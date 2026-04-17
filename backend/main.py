"""
CARDIA — FastAPI Main Server
Use this to Run main server: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import warnings, os, pickle
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config           import DATA_SOURCE, PATHS, STRATEGIES, MAX_HR, RESTING_HR, SAFETY
from database         import init_db, insert_reading, get_recent, get_stats
from kalman_filter    import CardiacDataSmoother
from llama_coach      import get_coaching
from isolation_forest import AnomalyDetector
from rl_agent         import load_agent

app = FastAPI(title="CARDIA API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
init_db()
smoother = CardiacDataSmoother()
agent    = load_agent()

try:
    detector = AnomalyDetector()
    print("✅ Isolation Forest loaded!")
except:
    detector = None
    print("⚠️  Run isolation_forest.py to enable anomaly detection")

ces_bundle = None
if os.path.exists(PATHS["ces_model"]):
    with open(PATHS["ces_model"], "rb") as f:
        ces_bundle = pickle.load(f)
    print("✅ CES model loaded!")
else:
    print("⚠️  Run xgboost_ces.py to enable CES scoring")

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
# This stores the LATEST reading from ESP32
_latest_esp32    = None      # raw data from ESP32
_latest_processed= None      # processed result sent to frontend
_last_esp32_time = None      # when ESP32 last sent data
_history         = []        # for LSTM forecast


# ── Core processing pipeline ──────────────────────────────────────────────────
def process_reading(raw: dict) -> dict:
    hr    = float(raw.get("hr",   72))
    hrv   = float(raw.get("hrv",  45))
    load  = float(raw.get("load",  3))
    sleep = float(raw.get("sleep", 7))
    spo2  = float(raw.get("spo2", 98))
    rr    = float(raw.get("rr",  833))

    # Clamp physiological limits
    hr   = float(np.clip(hr,   40, 220))
    hrv  = float(np.clip(hrv,   5, 150))
    spo2 = float(np.clip(spo2, 70, 100))
    load = float(np.clip(load,  0,  10))

    # 1. Kalman filter
    cleaned = smoother.process(hr, hrv)
    hr_c    = cleaned["clean_hr"]
    hrv_c   = cleaned["clean_hrv"]

    # 2. HR Zone
    hr_pct = hr_c / MAX_HR * 100
    zone   = (0 if hr_pct < 50 else 1 if hr_pct < 60 else
              2 if hr_pct < 70 else 3 if hr_pct < 80 else
              4 if hr_pct < 90 else 5)

    # 3. CES Score
    if ces_bundle:
        features     = ces_bundle["features"]
        reading_dict = {
            "hr": hr_c, "hrv_rmssd": hrv_c,
            "activity_load": load, "lf_hf_ratio": 2.0,
            "sleep_quality": sleep, "resting_hr": RESTING_HR,
            "hr_zone": zone, "fatigue": 0,
            "is_workout": int(load > 5),
            "spo2": spo2, "temperature": 36.6, "rr_interval": rr,
        }
        vals   = np.array([[reading_dict.get(f, 0) for f in features]])
        scaled = ces_bundle["scaler"].transform(vals)
        xgb_s  = ces_bundle["xgb_model"].predict(scaled)[0]
        rf_s   = ces_bundle["rf_model"].predict(scaled)[0]
        lgbm_s = ces_bundle["lgbm_model"].predict(scaled)[0]
        ces    = round(float(np.clip((xgb_s+rf_s+lgbm_s)/3, 0, 100)), 1)
    else:
        hr_score  = max(0, min(40, (MAX_HR-hr_c)/(MAX_HR-RESTING_HR)*40))
        hrv_score = max(0, min(35, hrv_c/80*35))
        load_pen  = max(0, (load-7)*3.5)
        ces       = round(max(0, min(100, hr_score+hrv_score+25-load_pen)), 1)

    # 4. Anomaly + Safety
    if detector:
        det     = detector.predict(hr_c, hrv_c, load, spo2, rr)
        anomaly = det["anomaly"]
        safety  = det["safety"]
    else:
        anomaly = 1
        hr_pct2 = hr_c / MAX_HR * 100
        if (hr_pct2 >= SAFETY["danger_hr_pct"]  or hrv_c < SAFETY["danger_hrv"] or
            load    >  SAFETY["danger_load"]     or spo2  < SAFETY["danger_spo2"]):
            safety = "DANGER"
        elif (hr_pct2 >= SAFETY["caution_hr_pct"] or hrv_c < SAFETY["caution_hrv"] or
              load    >  SAFETY["caution_load"]    or spo2  < SAFETY["caution_spo2"]):
            safety = "CAUTION"
        else:
            safety = "SAFE"

    # 5. RL Strategy
    safety_code   = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}.get(safety, 0)
    action, _     = agent.predict([ces, hrv_c, load, sleep, spo2, safety_code])
    strategy_name = STRATEGIES[int(action)]["name"]

    # 6. AI Coach
    coach = get_coaching(
        hr=round(hr_c, 1), hrv=round(hrv_c, 1),
        ces=ces, strategy=strategy_name,
        spo2=spo2, safety=safety, anomaly=anomaly
    )

    # 7. Save to DB
    insert_reading(
        hr=hr_c, hrv=hrv_c, load=load, sleep=sleep,
        spo2=spo2, ces=ces, strategy=strategy_name,
        anomaly=anomaly, zone=zone, safety=safety, coach=coach
    )

    # 8. Update history
    _history.append({
        "hr": hr_c, "hrv_rmssd": hrv_c,
        "ces_score": ces, "activity_load": load, "spo2": spo2
    })
    if len(_history) > 100:
        _history.pop(0)

    return {
        "heart_rate": round(hr_c, 1),
        "hrv":        round(hrv_c, 1),
        "spo2":       round(spo2, 1),
        "load":       round(load, 2),
        "zone":       zone,
        "ces":        ces,
        "strategy":   strategy_name,
        "anomaly":    anomaly,
        "safety":     safety,
        "coach":      coach,
        "source":     "esp32",
        "timestamp":  datetime.now().isoformat(),
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    esp32_connected = _last_esp32_time is not None
    return {
        "message":       "CARDIA API Running",
        "esp32_connected": esp32_connected,
        "last_data":     _last_esp32_time,
    }


@app.post("/sensor-data")
def receive_sensor_data(data: dict):
    """
    ESP32 sends data here via WiFi POST request.
    Processes it and stores as latest reading.
    """
    global _latest_esp32, _latest_processed, _last_esp32_time

    print(f"📡 ESP32 data received: HR={data.get('hr')} "
          f"HRV={data.get('hrv')} SpO2={data.get('spo2')}")

    _latest_esp32    = data
    _last_esp32_time = datetime.now().isoformat()
    _latest_processed= process_reading(data)

    return _latest_processed


@app.get("/live")
def get_live_data():
    """
    Frontend polls this every 2 seconds.
    Returns REAL ESP32 data if available.
    Returns waiting message if ESP32 not connected.
    """
    global _latest_processed, _last_esp32_time

    # Check if ESP32 has sent data recently (within 10 seconds)
    if _latest_processed and _last_esp32_time:
        from datetime import datetime as dt
        last = dt.fromisoformat(_last_esp32_time)
        seconds_ago = (dt.now() - last).total_seconds()

        if seconds_ago < 10:
            # Fresh ESP32 data — return it!
            print(f"✅ Returning ESP32 data ({seconds_ago:.1f}s old)")
            return _latest_processed
        else:
            # ESP32 data is stale
            print(f"⚠️  ESP32 data is {seconds_ago:.1f}s old")
            result = dict(_latest_processed)
            result["warning"] = f"ESP32 data is {int(seconds_ago)}s old — check connection"
            return result

    # No ESP32 data yet — return waiting state
    print("⏳ No ESP32 data yet — waiting...")
    return {
        "heart_rate": 0,
        "hrv":        0,
        "spo2":       0,
        "load":       0,
        "zone":       0,
        "ces":        0,
        "strategy":   "Waiting...",
        "anomaly":    1,
        "safety":     "SAFE",
        "coach":      "Waiting for ESP32 sensor data... Place finger on MAX30102",
        "source":     "waiting",
        "timestamp":  datetime.now().isoformat(),
    }


@app.get("/status")
def get_status():
    """Check if ESP32 is connected and sending data."""
    if _last_esp32_time:
        from datetime import datetime as dt
        last       = dt.fromisoformat(_last_esp32_time)
        secs_ago   = (dt.now() - last).total_seconds()
        connected  = secs_ago < 10
        return {
            "esp32_connected": connected,
            "last_data_ago":   f"{int(secs_ago)}s ago",
            "last_hr":         _latest_processed.get("heart_rate") if _latest_processed else None,
        }
    return {"esp32_connected": False, "last_data_ago": "never"}


@app.get("/history")
def get_history(limit: int = 50):
    return get_recent(limit)


@app.get("/stats")
def get_session_stats():
    return get_stats()


@app.get("/forecast")
def get_forecast():
    try:
        from lstm_model import forecast_next_days
        return {"forecast": forecast_next_days(_history, days=7)}
    except Exception as e:
        return {"forecast": [], "error": str(e)}


@app.get("/strategies")
def get_strategies():
    return STRATEGIES


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
CARDIA — Central Configuration
All project settings in one place.
Change anything here — updates everywhere automatically.
"""

# ── Data Source ───────────────────────────────────────────────────────────────
# "simulate" → Use this now (no hardware needed)
# "esp32"    → Change to this when ESP32 arrives
DATA_SOURCE = "esp32"

# ── ESP32 WiFi Settings ───────────────────────────────────────────────────────
ESP32_IP       = "10.135.101.216"   # Your ESP32's IP from serial monitor
ESP32_PORT     = 80
ESP32_ENDPOINT = "/data"
ESP32_WS_URL   = f"ws://{ESP32_IP}:{ESP32_PORT}/ws"

# ── User Profile ──────────────────────────────────────────────────────────────
USER = {
    "name":          "Athlete",
    "age":           22,
    "resting_hr":    68,
    "max_hr":        190,
    "weight_kg":     70,
    "height_cm":     175,
    "fitness_level": "intermediate",
}

# ── Safety Thresholds ─────────────────────────────────────────────────────────
SAFETY = {
    "danger_hr_pct":  93,
    "caution_hr_pct": 82,
    "danger_hrv":     18,
    "caution_hrv":    28,
    "danger_load":    9.5,
    "caution_load":   8.0,
    "danger_spo2":    90,
    "caution_spo2":   94,
}

# ── ML Model Settings ─────────────────────────────────────────────────────────
ML = {
    "ces_n_estimators":  200,
    "ces_max_depth":     6,
    "ces_learning_rate": 0.05,
    "lstm_sequence_len": 14,
    "lstm_forecast_days":7,
    "lstm_epochs":       50,
    "lstm_batch_size":   32,
    "iso_contamination": 0.05,
    "iso_n_estimators":  200,
    "rl_learning_rate":  0.0003,
    "rl_timesteps":      50000,
    "rl_gamma":          0.99,
}

# ── Kalman Filter Settings ────────────────────────────────────────────────────
KALMAN = {
    "hr_process_noise":      2.0,
    "hr_measurement_noise":  8.0,
    "hrv_process_noise":     3.0,
    "hrv_measurement_noise": 12.0,
}

# ── Data Paths ────────────────────────────────────────────────────────────────
PATHS = {
    "synthetic_data": "data/cardiac_data.csv",
    "physionet_data": "data/physionet/",
    "combined_data":  "data/combined_data.csv",
    "ces_model":      "models/ces_ensemble.pkl",
    "lstm_model":     "models/lstm_model.h5",
    "iso_model":      "models/isolation_forest.pkl",
    "rl_model":       "models/rl_agent.pkl",
    "database":       "data/cardia.db",
}

# ── Ollama Settings ───────────────────────────────────────────────────────────
OLLAMA = {
    "url":         "http://localhost:11434/api/generate",
    "model":       "llama3",
    "max_tokens":  300,
    "temperature": 0.7,
}

# ── Server Settings ───────────────────────────────────────────────────────────
SERVER = {
    "host":    "0.0.0.0",
    "port":    8000,
    "reload":  True,
}

# ── Frontend ──────────────────────────────────────────────────────────────────
FRONTEND = {
    "url":             "http://localhost:3000",
    "update_interval": 2,
}

# ── Computed Values ───────────────────────────────────────────────────────────
MAX_HR     = USER["max_hr"]
RESTING_HR = USER["resting_hr"]
DANGER_HR  = int(MAX_HR * SAFETY["danger_hr_pct"]  / 100)
CAUTION_HR = int(MAX_HR * SAFETY["caution_hr_pct"] / 100)

# ── HR Zones ──────────────────────────────────────────────────────────────────
HR_ZONES = {
    0: {"name": "Rest",   "min": 0,              "max": int(MAX_HR*0.50), "color": "#90CAF9"},
    1: {"name": "Zone 1", "min": int(MAX_HR*0.50),"max": int(MAX_HR*0.60),"color": "#66BB6A"},
    2: {"name": "Zone 2", "min": int(MAX_HR*0.60),"max": int(MAX_HR*0.70),"color": "#FFA726"},
    3: {"name": "Zone 3", "min": int(MAX_HR*0.70),"max": int(MAX_HR*0.80),"color": "#FF7043"},
    4: {"name": "Zone 4", "min": int(MAX_HR*0.80),"max": int(MAX_HR*0.90),"color": "#EF5350"},
    5: {"name": "Zone 5", "min": int(MAX_HR*0.90),"max": MAX_HR,          "color": "#B71C1C"},
}

# ── Strategies ────────────────────────────────────────────────────────────────
STRATEGIES = {
    0: {"name": "HIIT",               "weeks": 6, "ces_gain": 12, "color": "#E53935"},
    1: {"name": "HRV Breathing",      "weeks": 4, "ces_gain": 8,  "color": "#1565C0"},
    2: {"name": "Active Recovery",    "weeks": 2, "ces_gain": 6,  "color": "#2E7D32"},
    3: {"name": "LISS Cardio",        "weeks": 4, "ces_gain": 7,  "color": "#6A1B9A"},
    4: {"name": "Full Rest",          "weeks": 1, "ces_gain": 4,  "color": "#37474F"},
}

if __name__ == "__main__":
    print("CARDIA Config loaded!")
    print(f"  Data source: {DATA_SOURCE}")
    print(f"  Max HR:      {MAX_HR} BPM")
    print(f"  Danger HR:   {DANGER_HR} BPM")
    print(f"  Caution HR:  {CAUTION_HR} BPM")
    print("All settings ready!")
"""
CARDIA — Llama AI Coach
Reads ALL live ML outputs before every response.
Falls back to smart pre-written advice if Ollama offline.
"""

import requests
from config import OLLAMA

FALLBACK_TIPS = {
    "DANGER": [
        "🔴 STOP immediately! Your heart rate is in the danger zone. Rest now.",
        "🔴 Critical alert: Your HRV is dangerously low. Stop all activity.",
        "🔴 Warning: Anomaly detected. Sit down, breathe slowly, rest.",
    ],
    "CAUTION": [
        "🟡 Reduce intensity now. Your body is showing stress signals.",
        "🟡 Slow down — your HRV suggests your recovery isn't complete.",
        "🟡 Take it easy. Consider switching to light movement or rest.",
    ],
    "SAFE": [
        "🟢 You're in great shape! Keep up the consistent training.",
        "🟢 HRV looks strong today — good day for quality work.",
        "🟢 Your cardiac efficiency is improving. Stay the course!",
    ],
}

_tip_index = {"DANGER": 0, "CAUTION": 0, "SAFE": 0}


def get_coaching(hr, hrv, ces, strategy,
                 spo2=98, safety="SAFE", anomaly=1):
    """
    Get AI coaching advice from Llama 3.

    Args:
        hr:       Current heart rate
        hrv:      Current HRV RMSSD
        ces:      Cardiac Enhancement Score
        strategy: Recommended strategy name
        spo2:     Blood oxygen level
        safety:   SAFE / CAUTION / DANGER
        anomaly:  1=normal, -1=anomaly

    Returns:
        str: coaching advice
    """
    prompt = f"""You are CARDIA, an expert AI cardiac coach.
Current athlete stats:
- Heart Rate: {hr} BPM
- HRV (RMSSD): {hrv} ms
- SpO2: {spo2}%
- Cardiac Enhancement Score (CES): {ces}/100
- Recommended Strategy: {strategy}
- Safety Status: {safety}
- Anomaly Detected: {"YES - ALERT!" if anomaly == -1 else "No"}

Give short (2-3 sentences), specific coaching advice based on these exact numbers.
Be direct. No generic advice. Reference the actual values."""

    try:
        response = requests.post(
            OLLAMA["url"],
            json={
                "model":  OLLAMA["model"],
                "prompt": prompt,
                "stream": False,
            },
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
    except Exception:
        pass

    # Fallback when Ollama offline
    tips  = FALLBACK_TIPS.get(safety, FALLBACK_TIPS["SAFE"])
    idx   = _tip_index[safety] % len(tips)
    _tip_index[safety] += 1
    return tips[idx]


if __name__ == "__main__":
    print("Testing Llama Coach...")
    advice = get_coaching(
        hr=145, hrv=28, ces=58,
        strategy="LISS Cardio",
        spo2=97.5, safety="CAUTION"
    )
    print(f"Coach says: {advice}")